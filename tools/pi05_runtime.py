"""Provide shared Pi05 configuration, validation, and model loading.

Validate LeRobot dataset features and statistics, locate local or cached
policies, check weight completeness, and prepare task-specific policy
configuration and processors. Centralize offline/proxy settings, device
checks, strict weight loading, and training cache/temporary-file paths.
The training and evaluation entry points call these helpers to keep
dataset adaptation and policy loading consistent across workflows.
"""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from mujoco_env.robot_config import STATE_DIM, ACTION_DIM, DATA_DIR, TRAIN_DIR

# Keep defaults rooted at the project after placing this shared module in tools/.
HERE = Path(__file__).resolve().parents[1]
IMAGE_KEYS = ("observation.images.agentview", "observation.images.wrist")
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
              "http_proxy", "https_proxy", "all_proxy")


def configure_network(use_proxy=False, offline=False):
    # Only affect this script, never the already-running Codex process.
    if not use_proxy:
        for key in PROXY_KEYS:
            os.environ.pop(key, None)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def configure_training_storage():
    """Keep cloud training caches and temporary model copies beside the project."""
    cache = HERE.parent / "pi05-cache"
    defaults = {
        "WANDB_DATA_DIR": cache / "wandb-data",
        "WANDB_CACHE_DIR": cache / "wandb-cache",
        "WANDB_ARTIFACT_DIR": cache / "wandb-artifacts",
        "TMPDIR": cache / "tmp",
        "HF_HOME": HERE.parent / "hf-cache",
    }
    for key, default in defaults.items():
        os.environ.setdefault(key, str(default))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    # A caller may have imported tempfile before configuring these paths.
    import tempfile
    tempfile.tempdir = None
    return {key: os.environ[key] for key in defaults}


def read_dataset(root):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    features = info["features"]
    for key, shape in (("observation.state", [STATE_DIM]), ("action", [ACTION_DIM])):
        if features.get(key, {}).get("shape") != shape:
            raise ValueError(f"FR3 的 {key} 必须是 {shape}，当前为 {features.get(key)}；旧 OMY 数据不能直接用于 FR3")
        for quantile in ("q01", "q99"):
            if len(stats.get(key, {}).get(quantile, [])) != shape[0]:
                raise ValueError(f"{key} 缺少长度为 {shape[0]} 的 {quantile} 统计")
    for key in IMAGE_KEYS:
        shape = features.get(key, {}).get("shape", [])
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"缺少 HWC RGB 图像特征: {key}")
    if info.get("total_frames", 0) <= 0 or info.get("fps", 0) <= 0:
        raise ValueError("数据集为空或 fps 无效")
    return info, stats


def task_features(info):
    inputs = {key: {"type": "VISUAL", "shape": [3, *info["features"][key]["shape"][:2]]}
              for key in IMAGE_KEYS}
    inputs["observation.state"] = {"type": "STATE", "shape": [STATE_DIM]}
    return inputs, {"action": {"type": "ACTION", "shape": [ACTION_DIM]}}


def check_dataset_files(root, info):
    import av
    import numpy as np
    import pyarrow.parquet as pq

    root = Path(root)
    files = sorted((root / "data").glob("*/*.parquet"))
    rows = 0
    for file in files:
        table = pq.read_table(file, columns=["observation.state", "action"])
        rows += table.num_rows
        for key, size in (("observation.state", STATE_DIM), ("action", ACTION_DIM)):
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (table.num_rows, size) or not np.isfinite(values).all():
                raise ValueError(f"{file}: {key} 形状或数值错误")
    if rows != info["total_frames"]:
        raise ValueError(f"数据帧数 {rows} 与 meta/info.json 的 {info['total_frames']} 不一致")
    episodes = sum(pq.read_metadata(f).num_rows for f in (root / "meta/episodes").glob("*/*.parquet"))
    if episodes != info["total_episodes"]:
        raise ValueError(f"episode 数量不一致: {episodes}")
    for key in IMAGE_KEYS:
        videos = sorted((root / "videos" / key).glob("*/*.mp4"))
        if not videos:
            raise FileNotFoundError(f"缺少视频: {key}")
        for video in videos:
            with av.open(str(video)) as container:
                frame = next(container.decode(video=0))
                if [frame.height, frame.width, 3] != info["features"][key]["shape"]:
                    raise ValueError(f"视频尺寸与元数据不一致: {video}")
    print(f"数据检查通过: {episodes} 条 / {rows} 帧，两路视频可解码")


def cached_policy_root(source):
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or str(source).startswith(("./", "../")):
        raise FileNotFoundError(f"模型目录不存在: {source}")
    from huggingface_hub import hf_hub_download
    try:
        return Path(hf_hub_download(str(source), "config.json", local_files_only=True)).parent
    except OSError:
        return None


def validate_weights(root):
    """Check the safetensors container without reading a multi-GB tensor into RAM."""
    path = Path(root) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"权重尚未完整下载: {path}（.incomplete 文件不能使用）")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        length = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < length <= 100_000_000:
            raise ValueError(f"无效 safetensors 文件: {path}")
        header = json.loads(stream.read(length))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    if not tensors or path.stat().st_size != 8 + length + max(v["data_offsets"][1] for v in tensors):
        raise ValueError(f"权重文件不完整: {path}")
    return path


def local_policy_root(source, offline=False):
    def validate_policy(root):
        validate_weights(root)
        for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"模型目录缺少配置: {root / name}")
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            pipeline = json.loads((root / name).read_text(encoding="utf-8"))
            for step in pipeline["steps"]:
                if step.get("state_file") and not (root / step["state_file"]).is_file():
                    raise FileNotFoundError(f"缺少处理器状态文件: {root / step['state_file']}")

    root = cached_policy_root(source)
    if root is not None:
        try:
            validate_policy(root)
            return root
        except FileNotFoundError:
            if Path(source).is_dir() or offline:
                raise
    if offline:
        raise FileNotFoundError(f"离线缓存不完整: {source}")
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(
        str(source), max_workers=2,
        allow_patterns=["config.json", "model.safetensors", "policy_*processor.json", "*.safetensors"],
    ))
    validate_policy(root)
    return root


def prepare_task_policy(source_root, data_root, device, dtype, destination):
    """Link weights and write task metadata without changing the original checkpoint."""
    source_root, destination = Path(source_root), Path(destination)
    info, stats = read_dataset(data_root)
    config = json.loads((source_root / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "pi05":
        raise ValueError("本脚本仅支持 pi05 checkpoint")
    if set(IMAGE_KEYS).issubset(config.get('input_features', {})):
        if (config['input_features'].get('observation.state', {}).get('shape') != [STATE_DIM]
                or config.get('output_features', {}).get('action', {}).get('shape') != [ACTION_DIM]):
            raise ValueError('这是旧 OMY 任务权重，不能直接改维度用于 FR3；请从 pi05_base 重新微调')
    if config.get('max_state_dim', 32) < STATE_DIM or config.get('max_action_dim', 32) < ACTION_DIM:
        raise ValueError('基础模型的状态/动作容量不足以容纳 FR3 的 7/8 维接口')
    if config.get("use_relative_actions", False):
        raise ValueError("当前数据是绝对关节角；新训练请使用 use_relative_actions=false 的底座")
    config["input_features"], config["output_features"] = task_features(info)
    config.update(device=device, dtype=dtype, push_to_hub=False, repo_id=None)
    config["normalization_mapping"] = {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"}
    identity = hashlib.sha256(json.dumps([str(source_root), config, stats], sort_keys=True).encode()).hexdigest()[:12]
    target = destination / identity
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for file in source_root.glob("*.safetensors"):
        link = target / file.name
        if not link.exists():
            try:
                link.symlink_to(file.resolve())
            except OSError:
                # Windows without Developer Mode cannot create symlinks. A
                # same-volume hard link keeps weights shared; cross-volume
                # installations need a local copy. Only read weights here.
                try:
                    link.hardlink_to(file.resolve())
                except OSError:
                    shutil.copy2(file, link)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        pipeline = json.loads((source_root / name).read_text(encoding="utf-8"))
        for step in pipeline["steps"]:
            key, cfg = step["registry_name"], step.setdefault("config", {})
            if key in ("normalizer_processor", "unnormalizer_processor"):
                step.pop("state_file", None)
                cfg.update(features={**config["input_features"], **config["output_features"]}
                           if key == "normalizer_processor" else config["output_features"],
                           stats=copy.deepcopy(stats), norm_map=config["normalization_mapping"])
            elif key == "rename_observations_processor":
                cfg["rename_map"] = {}
            elif key == "device_processor":
                cfg["device"] = device if name == "policy_preprocessor.json" else "cpu"
        (target / name).write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    return target


def require_device(device):
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前实例没有可用 CUDA。请切换到 GPU 实例；无卡检查用 --check-data / --dry-run / --env-only。")


def install_strict_pi05_loader():
    """Replace the upstream silent-random-weight fallback in this process only."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    def strict_load(cls, pretrained_name_or_path, *, config=None, **kwargs):
        from lerobot.configs.policies import PreTrainedConfig
        from safetensors.torch import load_file

        root = local_policy_root(pretrained_name_or_path, offline=True)
        if config is None:
            config = PreTrainedConfig.from_pretrained(str(root))
        policy = cls(config)
        state = load_file(str(validate_weights(root)), device="cpu")
        state = policy._fix_pytorch_state_dict_keys(state, config)
        state = {(key if key.startswith("model.") else f"model.{key}"): value for key, value in state.items()}
        policy.load_state_dict(state, strict=True)
        del state
        print(f"预训练权重已严格加载: {root}")
        return policy

    PI05Policy.from_pretrained = classmethod(strict_load)
