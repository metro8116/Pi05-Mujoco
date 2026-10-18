"""Fine-tune Pi05 on local FR3 pick-and-place demonstrations.

Validate dataset features and statistics, adapt the pretrained policy to
the project's cameras and joint interface, and launch the LeRobot trainer.
Support CUDA training, periodic checkpoints, optional W&B logging, and
resuming from saved training state. Dataset checks and configuration
previews are available without starting a training run.

Inputs: A local dataset, a pretrained policy, and training options.
Outputs: Training logs and checkpoints containing exported policy files
    and the state needed to resume training.
Usage: python 3.train_pi05.py --offline --device cuda --steps 20000
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

from tools.pi05_runtime import (
    HERE, check_dataset_files, configure_network, configure_training_storage, install_strict_pi05_loader,
    local_policy_root, prepare_task_policy, read_dataset, require_device,
)
from mujoco_env.robot_config import DATA_DIR, TRAIN_DIR, REPO_ID, STATE_DIM, ACTION_DIM

DEFAULT_OUT = HERE / TRAIN_DIR


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--policy-path", default="lerobot/pi05_base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--job-name", default="pi05_fr3_pick_place")
    # None preserves saved hyperparameters unless explicitly overridden on resume.
    for name in ("batch-size", "steps", "save-freq", "log-freq", "num-workers", "seed"):
        ap.add_argument("--" + name, type=int, default=None)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    ap.add_argument("--device", default="cuda")
    for name in ("train-expert-only", "gradient-checkpointing", "freeze-vision-encoder"):
        ap.add_argument("--" + name, choices=["true", "false"], default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wandb", choices=["true", "false"], default="false")
    ap.add_argument("--wandb-upload-model", choices=["true", "false"], default="false",
                    help="默认只记录指标；true 时同时上传 checkpoint 模型到 W&B")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from", help="checkpoint、pretrained_model 目录或 train_config.json")
    ap.add_argument("--dry-run", action="store_true", help="只打印配置，不下载、不训练、不创建输出目录")
    ap.add_argument("--check-data", action="store_true", help="检查本地数据和视频，不导入 torch")
    ap.add_argument("--offline", action="store_true", help="只使用本地模型/缓存")
    ap.add_argument("--use-proxy", action="store_true", help="保留代理变量；默认下载使用 AutoDL 直连")
    return ap


def resume_config(args):
    candidate = Path(args.resume_from).expanduser() if args.resume_from else Path(args.out or DEFAULT_OUT) / "checkpoints/last"
    if candidate.is_dir():
        if (candidate / "pretrained_model").is_dir():
            candidate = candidate / "pretrained_model"
        candidate = candidate / "train_config.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"找不到恢复配置: {candidate}")
    checkpoint = candidate.parent.parent
    if not (candidate.parent / "model.safetensors").is_file() or not (checkpoint / "training_state/training_step.json").is_file():
        raise FileNotFoundError(f"checkpoint 缺少模型或训练状态: {checkpoint}")
    return candidate.resolve()


def training_command(args, data_root, out, policy_root=None, config_path=None):
    flags = [f"--policy.device={args.device}", "--policy.push_to_hub=false",
             "--save_checkpoint_to_hub=false", f"--wandb.enable={args.wandb}",
             f"--wandb.disable_artifact={str(args.wandb_upload_model != 'true').lower()}"]
    if config_path:
        # --policy.path would bypass native optimizer/checkpoint restoration.
        flags += [f"--config_path={config_path}", "--resume=true"]
        if args.out:
            flags.append(f"--output_dir={out}")
        if args.data_root:
            flags.append(f"--dataset.root={data_root}")
    else:
        flags += [f"--dataset.repo_id={args.repo_id}", f"--dataset.root={data_root}",
                  "--dataset.video_backend=pyav", f"--policy.path={policy_root}",
                  f"--output_dir={out}", f"--job_name={args.job_name}"]
    defaults = {"batch_size": 16, "steps": 20000, "save_freq": 2000,
                "log_freq": 50, "num_workers": 4, "seed": 1000}
    for name, default in defaults.items():
        value = getattr(args, name)
        if value is not None or not config_path:
            flags.append(f"--{name}={value if value is not None else default}")
    policy_defaults = {"dtype": "float32" if args.device == "cpu" else "bfloat16",
                       "train_expert_only": "false", "gradient_checkpointing": "true",
                       "freeze_vision_encoder": "true"}
    for name, default in policy_defaults.items():
        value = getattr(args, name)
        if value is not None or not config_path:
            flags.append(f"--policy.{name}={value if value is not None else default}")
    if args.lr is not None:
        if config_path:
            raise ValueError("续训保留 checkpoint 优化器/调度器；--lr 仅用于新训练")
        flags.append(f"--policy.optimizer_lr={args.lr}")
    return flags


def main():
    ap = parser()
    args = ap.parse_args()
    for key in ("batch_size", "steps", "log_freq"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            ap.error(f"--{key.replace('_', '-')} 必须大于零")
    if args.num_workers is not None and args.num_workers < 0:
        ap.error("--num-workers 不能为负数")
    configure_network(args.use_proxy, args.offline)
    config_path = resume_config(args) if args.resume or args.resume_from else None
    saved = json.loads(config_path.read_text()) if config_path else {}
    if config_path:
        policy = saved.get('policy', {})
        if (policy.get('input_features', {}).get('observation.state', {}).get('shape') != [STATE_DIM]
                or policy.get('output_features', {}).get('action', {}).get('shape') != [ACTION_DIM]):
            raise ValueError('FR3 续训需要 7维状态/8维动作的 FR3 checkpoint，不能从旧 OMY checkpoint 续训')
    data_root = Path(args.data_root or saved.get("dataset", {}).get("root") or HERE / DATA_DIR).resolve()
    out = Path(args.out or saved.get("output_dir") or DEFAULT_OUT).resolve()
    info, _ = read_dataset(data_root)
    print(f"数据集: {data_root}，{info['total_episodes']} 条 / {info['total_frames']} 帧 / {info['total_tasks']} 条任务")
    print(f"FR3 输入: agentview + wrist 图像、{STATE_DIM} 维实际关节角；输出: {ACTION_DIM} 维（7关节目标 + 夹爪）")
    if args.check_data:
        check_dataset_files(data_root, info)
        return
    if not config_path and out.exists():
        raise FileExistsError(f"训练输出目录已存在: {out}。请选择新 --out，或使用 --resume。")
    flags = training_command(args, data_root, out, "<运行时生成的本任务模型目录>", config_path)
    print("LeRobot 参数:\n  " + shlex.join([sys.executable, "-m", "lerobot.scripts.lerobot_train", *flags]))
    if args.dry_run:
        print("干跑通过：未下载、未训练、未创建训练输出目录。")
        return
    storage = configure_training_storage()
    print(f"训练临时目录: {storage['TMPDIR']}；W&B 暂存: {storage['WANDB_DATA_DIR']}")
    require_device(args.device)
    if not config_path:
        source = local_policy_root(args.policy_path, args.offline)
        dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
        prepared = prepare_task_policy(source, data_root, args.device, dtype, HERE / "outputs/prepared_policy")
        flags = training_command(args, data_root, out, prepared)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        local_policy_root(config_path.parent, offline=True)
    install_strict_pi05_loader()
    from lerobot.scripts.lerobot_train import train
    previous_argv = sys.argv
    sys.argv = ["lerobot_train", *flags]
    try:
        train()
    except KeyboardInterrupt:
        print("训练已中断。只有已写入的 checkpoint 可以恢复。")
        return 130
    finally:
        sys.argv = previous_argv
    print(f"训练完成。评估权重: {out}/checkpoints/last/pretrained_model")
    print("python 4.deploy_pi05.py --policy-path " + shlex.quote(str(out / "checkpoints/last/pretrained_model")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"错误: {error}")
