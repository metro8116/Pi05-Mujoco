"""Evaluate Pi05 on the three FR3 pick-and-place tasks in MuJoCo.

Run apple-only, green-mug-only, and combined-object trials using a base
policy or a fine-tuned checkpoint. Configure trial counts and step limits
globally or per task, and evaluate completion with the environment's
placement, gripper-release, and retreat checks. Support headless EGL
rendering, offline policy loading, and environment-only diagnostics.

Inputs: Policy files, scene XML, evaluation settings, and dataset metadata
    when needed to adapt a base policy.
Outputs: records/, videos/, summary.json, and evaluation_config.json
    directly inside --out; the summary contains overall and per-task scores.
Usage: python 4.deploy_pi05.py --policy-path PATH --offline --out OUTPUT
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tools.pi05_runtime import (
    HERE, IMAGE_KEYS, configure_network,
    install_strict_pi05_loader, local_policy_root, prepare_task_policy,
    read_dataset, require_device, validate_weights,
)
from mujoco_env.robot_config import STATE_DIM, ACTION_DIM, DATA_DIR


TASK_TARGETS = {"apple": ("apple",), "mug": ("mug_green",), "both": ("apple", "mug_green")}


def resolve_task_settings(args):
    """Task-specific settings override the shared CLI defaults."""
    if args.n_per_task <= 0 or args.max_steps <= 0:
        raise ValueError("--n-per-task 和 --max-steps 必须大于零")
    settings = {}
    for name in TASK_TARGETS:
        runs = getattr(args, f"{name}_runs")
        steps = getattr(args, f"{name}_max_steps")
        if runs is not None and runs <= 0:
            raise ValueError(f"--{name}-runs 必须大于零")
        if steps is not None and steps <= 0:
            raise ValueError(f"--{name}-max-steps 必须大于零")
        settings[name] = {"runs": args.n_per_task if runs is None else runs,
                          "max_steps": args.max_steps if steps is None else steps}
    return settings


def slug(text):
    return text.replace("Put the ", "").replace(" in the basket.", "").replace(" and the ", "+").replace(" ", "_").replace(".", "")


def validate_task_config(config):
    expected = set(IMAGE_KEYS) | {"observation.state"}
    if set(config.get("input_features", {})) != expected:
        raise ValueError("checkpoint 相机键不匹配。请用修复后的 3.train_pi05.py 训练，或使用 lerobot/pi05_base 做链路检查。")
    if config["input_features"]["observation.state"]["shape"] != [STATE_DIM] or config["output_features"]["action"]["shape"] != [ACTION_DIM]:
        raise ValueError(f"FR3 checkpoint 必须配置 {STATE_DIM} 维状态和 {ACTION_DIM} 维动作；不能直接使用旧 OMY 权重")


def policy_directory(args):
    root = local_policy_root(args.policy_path, args.offline)
    config = json.loads((root / "config.json").read_text())
    if config.get("type") != "pi05":
        raise ValueError("本脚本仅支持 pi05")
    # Only adapt the untrained base. Fine-tuned checkpoints keep their saved stats.
    if not set(IMAGE_KEYS).issubset(config.get("input_features", {})):
        base_cameras = {"observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb",
                        "observation.images.right_wrist_0_rgb"}
        if not base_cameras.issubset(config.get("input_features", {})):
            validate_task_config(config)
        print("底座尚未针对本任务微调：使用本地数据统计适配输入，仅用于检查部署链路。")
        dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
        root = prepare_task_policy(root, args.data_root, args.device, dtype, HERE / "outputs/prepared_policy")
        config = json.loads((root / "config.json").read_text())
    validate_task_config(config)
    return root


def load_policy(root, args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    install_strict_pi05_loader()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    cfg = PreTrainedConfig.from_pretrained(str(root), cli_overrides=[])
    cfg.device = args.device
    if args.dtype:
        cfg.dtype = args.dtype
    cfg.pretrained_path = root
    # Keep saved rename/normalization steps intact; only change the runtime device.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(root),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    # Fail on missing tokenizer/processor files before allocating the full model.
    policy = PI05Policy.from_pretrained(str(root), config=cfg).eval()
    return policy, preprocessor, postprocessor, torch


def resize_images(agent, wrist, info):
    images = []
    for key, img in zip(IMAGE_KEYS, (agent, wrist)):
        height, width, _ = info["features"][key]["shape"]
        # Same Pillow resize as the collection script.
        images.append(np.asarray(Image.fromarray(img).resize((width, height))))
    return images


class OffscreenCameras:
    """Render the same fixed cameras and 800x600 viewport as the collector."""

    def __init__(self, env):
        import mujoco
        self.env = env
        model = env.env.model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 800)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 600)
        self.renderer = mujoco.Renderer(model, height=600, width=800)

    def grab_image(self):
        images = []
        for camera in ("agentview", "egocentric"):
            self.renderer.update_scene(self.env.env.data, camera=camera)
            images.append(self.renderer.render().copy())
        return images

    def close(self):
        self.renderer.close()


def preprocess_observation(observation, torch):
    out = {"task": observation["task"]}
    for key in IMAGE_KEYS:
        out[key] = torch.from_numpy(np.ascontiguousarray(observation[key].transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    out["observation.state"] = torch.from_numpy(np.ascontiguousarray(observation["observation.state"])).float().unsqueeze(0)
    return out


def ensure_display():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        executable = shutil.which("xvfb-run")
        if executable is None:
            raise RuntimeError("无 DISPLAY；请安装 xvfb 和 xauth，或用 --env-only --no-render 检查动力学")
        os.execvpe(executable, [executable, "-a", "-s", "-screen 0 1280x1024x24",
                              sys.executable, "-u", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]], os.environ.copy())


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-path", default="lerobot/pi05_base")
    ap.add_argument("--data-root", default=str(HERE / DATA_DIR))
    ap.add_argument("--n-per-task", type=int, default=10,
                    help="所有任务默认评测次数，默认 10；可由各任务 --*-runs 覆盖")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="所有任务默认每次最大步数，默认 1000；可由各任务 --*-max-steps 覆盖")
    for name, label in (("apple", "苹果"), ("mug", "绿色杯子"), ("both", "苹果和杯子双目标")):
        ap.add_argument(f"--{name}-runs", type=int, help=f"{label}任务的评测次数")
        ap.add_argument(f"--{name}-max-steps", type=int, help=f"{label}任务每次评测的最大步数")
    ap.add_argument("--out", default=str(HERE / "outputs/deploy/deploy_base"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"])
    ap.add_argument("--xml", default=str(HERE / "asset/example_scene_pick_place.xml"))
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--save-video", type=int, choices=[0, 1], default=1)
    ap.add_argument("--render-backend", choices=["glfw", "egl"], default="egl")
    ap.add_argument("--env-only", action="store_true", help="只检查仿真；不加载 Pi05，结果不代表策略成功率")
    ap.add_argument("--no-render", action="store_true", help="仅用于 --env-only，无窗口、无图像/视频")
    ap.add_argument("--preflight", action="store_true", help="只检查本地模型完整性和数据元信息，不下载、不推理")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--use-proxy", action="store_true")
    return ap


def main(args=None, *, env_factory=None, evaluation_metadata=None, task_targets=None):
    """Shared evaluation loop; optional factory supports alternate test layouts."""
    ap = parser()
    if args is None:
        args = ap.parse_args()
    try:
        task_settings = resolve_task_settings(args)
    except ValueError as error:
        ap.error(str(error))
    if args.fps is not None and args.fps <= 0:
        ap.error("fps 必须大于零")
    if args.no_render and not args.env_only:
        ap.error("--no-render 仅能用于 --env-only；Pi05 推理需要相机图像")
    configure_network(args.use_proxy, args.offline)
    if args.env_only:
        # A fresh FR3 checkout can be checked before collecting demonstrations.
        info = {"fps": 20, "features": {key: {"shape": [256, 256, 3]} for key in IMAGE_KEYS}}
    else:
        info, _ = read_dataset(args.data_root)
    args.fps = args.fps or info["fps"]
    if args.preflight:
        root = local_policy_root(args.policy_path, offline=True)
        config = json.loads((root / "config.json").read_text())
        if set(IMAGE_KEYS).issubset(config.get("input_features", {})):
            validate_task_config(config)
        print(f"本地模型完整性检查通过: {root}；GPU 推理尚未执行。")
        return
    backend = 'glfw' if sys.platform == 'win32' and args.render_backend == 'egl' else args.render_backend
    os.environ["MUJOCO_GL"] = "disable" if args.no_render else backend
    offscreen = not args.no_render and args.render_backend == "egl"
    if not args.no_render and not offscreen:
        ensure_display()
    os.chdir(HERE)  # XML includes and custom environment assets use project-relative paths.
    policy = preprocessor = postprocessor = torch = None
    if not args.env_only:
        require_device(args.device)
        root = policy_directory(args)
        policy, preprocessor, postprocessor, torch = load_policy(root, args)
    else:
        print("环境检查模式：不下载、不加载 Pi05；结果不代表模型成功率。")
    if env_factory is None:
        from mujoco_env.pick_place_env import PickPlaceEnv
        env_factory = PickPlaceEnv
    env = env_factory(args.xml, action_type="joint_angle", seed=args.seed,
                       enable_viewer=not args.no_render and not offscreen)
    cameras = None
    target_mapping = TASK_TARGETS if task_targets is None else task_targets
    task_keys = {frozenset(targets): name for name, targets in target_mapping.items()}
    tasks = [(env.make_instruction(list(targets)), task_settings[task_keys[frozenset(targets)]])
             for targets in env.all_task_targets()]
    output = Path(args.out).resolve()
    videos_dir, records_dir = output / "videos", output / "records"
    videos_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    evaluation_config = {"mode": "env-only" if args.env_only else "pi05",
                         "policy_path": None if args.env_only else args.policy_path,
                         "n_per_task": args.n_per_task, "max_steps": args.max_steps,
                         "fps": args.fps, "seed": args.seed,
                         "tasks": {task: settings for task, settings in tasks}}
    if evaluation_metadata:
        evaluation_config["test_layout"] = evaluation_metadata
    task_scores, all_records = {}, []
    try:
        if offscreen:
            cameras = OffscreenCameras(env)
        if args.fps > 1 / env.env.dt:
            raise ValueError("fps 不能超过仿真物理频率")
        for task, settings in tasks:
            print(f"评测配置: {task}，{settings['runs']} 次，每次最多 {settings['max_steps']} 步")
            records = []
            for run in range(settings["runs"]):
                env.reset(seed=args.seed + run)
                env.set_instruction(task)
                if policy is not None:
                    policy.reset()
                    for processor in (preprocessor, postprocessor):
                        if hasattr(processor, "reset"):
                            processor.reset()
                initial = np.asarray(env.obj_init_pose, dtype=float).tolist()
                name = f"run_{slug(task)}_{run}"
                writer, video, steps, success = None, None, 0, False
                start = time.time()
                try:
                    if args.save_video and not args.no_render:
                        import imageio.v2 as imageio
                        video = (Path("videos") / (name + ".mp4")).as_posix()
                        writer = imageio.get_writer(str(output / video), fps=args.fps)
                    while steps < settings["max_steps"]:
                        if not args.no_render and not offscreen and not env.env.is_viewer_alive():
                            break
                        env.step_env()
                        if not env.env.loop_every(HZ=args.fps):
                            continue
                        state = np.asarray(env.get_joint_state()[:STATE_DIM], dtype=np.float32)
                        images = None
                        if not args.no_render:
                            raw_images = cameras.grab_image() if cameras is not None else env.grab_image()
                            images = resize_images(*raw_images, info)
                        if policy is None:
                            action = np.concatenate([state, [env.gripper_cmd]])
                        else:
                            observation = {IMAGE_KEYS[0]: images[0], IMAGE_KEYS[1]: images[1],
                                           "observation.state": state, "task": [task]}
                            batch = preprocessor(preprocess_observation(observation, torch))
                            with torch.inference_mode():
                                action = postprocessor(policy.select_action(batch))
                            action = action.detach().float().cpu().numpy()
                        action = np.asarray(action, dtype=np.float32).reshape(-1)
                        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                            raise ValueError(f"无效动作：必须是有限的 {ACTION_DIM} 维向量，当前形状 {action.shape}")
                        env.step(action)
                        steps += 1
                        if writer is not None:
                            writer.append_data(np.concatenate(images, axis=1))
                        if env.env.use_mujoco_viewer:
                            env.env.render()  # Poll GLFW events and keep the viewer responsive.
                        if env.check_success():
                            success = True
                            break
                finally:
                    if writer is not None:
                        writer.close()
                record = {"task": task, "run": run, "success": bool(success), "steps": steps,
                          "seconds": round(time.time() - start, 1), "obj_init": initial, "video": video}
                (records_dir / (name + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                records.append(record)
                print(f"[{task} #{run}] success={success} steps={steps}")
            ok = sum(r["success"] for r in records)
            task_scores[task] = {"runs": len(records), "success": ok,
                                 "success_rate": ok / len(records),
                                 "avg_steps": float(np.mean([r["steps"] for r in records]))}
            all_records.extend(records)
        total = sum(v["runs"] for v in task_scores.values())
        success = sum(v["success"] for v in task_scores.values())
        summary = {"total": {"runs": total, "success": success, "success_rate": success / total},
                   "tasks": task_scores, "records": all_records}
        (output / "evaluation_config.json").write_text(
            json.dumps(evaluation_config, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果: {output / 'summary.json'}")
    finally:
        if cameras is not None:
            cameras.close()
        if env.env.use_mujoco_viewer:
            env.env.close_viewer()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit(f"错误: {error}")
