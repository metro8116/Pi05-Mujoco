"""Collect FR3 pick-and-place demonstrations with keyboard teleoperation.

Control the robot in MuJoCo while collecting the apple, green-mug, and
combined tasks. Record synchronized agent-view and wrist RGB images,
seven arm-joint positions, eight-dimensional actions, and object poses.
Successful demonstrations are saved automatically; compatible datasets
can be extended without replacing their existing episodes.

Inputs: Scene XML, keyboard commands, output dataset path, and random seed.
Outputs: A LeRobot v3 dataset with data/, meta/, and videos/ directories.
Usage: python 1.collect_data_pick_place.py --num-demo 20
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path

# 当前 conda 环境的 Library\bin（Windows 上 MuJoCo/GLFW 需要）
conda_prefix = sys.prefix
dll_path = os.path.join(conda_prefix, "Library", "bin")
if os.name == 'nt' and os.path.isdir(dll_path):
    _dll_directory = os.add_dll_directory(dll_path)

import numpy as np
import pandas as pd
from PIL import Image
from mujoco_env.robot_config import STATE_DIM, ACTION_DIM, ROBOT_TYPE, DATA_DIR, REPO_ID, ARM_JOINT_NAMES

HERE = Path(__file__).resolve().parent


def parser():
    ap = argparse.ArgumentParser(description='键盘遥操作采集 LeRobot v3 数据集（本地窗口）')
    ap.add_argument('--root', type=Path, default=HERE / DATA_DIR)
    ap.add_argument('--xml', type=Path, default=HERE / 'asset/example_scene_pick_place.xml')
    ap.add_argument('--num-demo', type=int, default=20, help='本次新增成功 episode 数')
    ap.add_argument('--seed', type=int, default=0)
    return ap


def read_disk_counts(root):
    """从磁盘读"每条任务已采多少条"，用于最少优先调度。

    为什么读磁盘而不是 dataset.meta.episodes：`LeRobotDataset.resume()` 之后
    `meta.episodes` 是 None（元数据懒加载），直接访问会 TypeError。
    """
    counts = {}
    files = sorted(glob.glob(os.path.join(root, 'meta', 'episodes', 'chunk-*', 'file-*.parquet')))
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:                          # noqa: BLE001
            continue
        if 'tasks' not in df.columns:
            continue
        for tasks in df['tasks']:
            items = tasks if isinstance(tasks, (list, tuple, np.ndarray)) else [tasks]
            for t in items:
                counts[str(t)] = counts.get(str(t), 0) + 1
    return counts


def read_disk_info(root):
    """读磁盘上的 meta/info.json（不存在就返回 None）。"""
    p = os.path.join(root, 'meta', 'info.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def check_features_match(root, features):
    """续采前比对磁盘数据集的图像键/维度。

    ⚠️ 实测：特征不一致时 `LeRobotDataset.resume()` **不会报错**，
    而是静默把数据集重置成"只有新特征"的空数据集，旧 episode 一起没了。
    所以这里自己在 resume 之前拦一道。
    """
    info = read_disk_info(root)
    if info is None:
        raise SystemExit(
            f'\n{root} 里没有 meta/info.json —— 数据集不完整，不能续采。\n'
            f'删掉这个目录再重新执行本脚本即可从零开始采。')
    have = info.get('features', {})
    bad = []
    for k, v in features.items():
        hv = have.get(k)
        if (hv is None or tuple(hv.get('shape', [])) != tuple(v['shape'])
                or hv.get('dtype') != v['dtype']):
            bad.append(k)
    if bad:
        raise SystemExit(
            f'\n现有数据集的特征和新脚本不一致：{bad}\n'
            f'  现有图像键: {sorted(k for k in have if "image" in k)}\n'
            f'  需要图像键: {sorted(k for k in features if "image" in k)}\n'
            f'继续 resume 会被 lerobot 静默重置（旧 episode 全部丢失），已中止。\n'
            f'请把 {root} 改名（留档）或删掉，再重新执行本脚本从零开始采。')
    return info


def task_stream(pool, counts, rng):
    """无限产出任务：每次挑"已采条数最少"的那条（平局随机），保证分布自动均衡。"""
    while True:
        fewest = min(counts.get(t, 0) for t in pool)
        cands = [t for t in pool if counts.get(t, 0) == fewest]
        t = rng.choice(cands)
        counts[t] = counts.get(t, 0) + 1
        yield t


def main():
    ap = parser()
    args = ap.parse_args()
    if args.num_demo <= 0:
        ap.error('--num-demo 必须大于零')
    from mujoco_env.pick_place_env import PickPlaceEnv
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    SEED = args.seed                  # 物体摆放/任务调度的随机种子
    REPO_NAME = REPO_ID
    NUM_DEMO = args.num_demo           # 这一次再采多少条（成功一条算一条）
    ROOT = args.root.expanduser().resolve()
    XML_PATH = str(args.xml.expanduser().resolve())
    IMG_SIZE = 256
    FPS = 20

    OBJ_DIM = len(PickPlaceEnv.OBJECTS) * 7       # 每物体 (x,y,z,qw,qx,qy,qz)

    features = {
        'observation.images.agentview': {
            'dtype': 'video', 'shape': (IMG_SIZE, IMG_SIZE, 3),
            'names': ['height', 'width', 'channels']},
        'observation.images.wrist': {
            'dtype': 'video', 'shape': (IMG_SIZE, IMG_SIZE, 3),
            'names': ['height', 'width', 'channels']},
        'observation.state': {'dtype': 'float32', 'shape': (STATE_DIM,), 'names': list(ARM_JOINT_NAMES)},
        'action': {'dtype': 'float32', 'shape': (ACTION_DIM,), 'names': [*ARM_JOINT_NAMES, 'gripper_closure']},
        'obj_init': {'dtype': 'float32', 'shape': (OBJ_DIM,), 'names': ['obj_init']},
    }

    create_new = True
    if os.path.exists(ROOT):
        existing = read_disk_info(ROOT)
        if existing and (existing.get('features', {}).get('observation.state', {}).get('shape') != [STATE_DIM]
                         or existing.get('features', {}).get('action', {}).get('shape') != [ACTION_DIM]):
            raise SystemExit('现有目录包含非 FR3 数据，已保留。请使用新目录 --root ./pick_place_data_fr3。')
        print(f"Directory {ROOT} already exists.")
        ans = input("Do you want to delete it? (y/n) ")
        if ans == 'y':
            shutil.rmtree(ROOT)
        else:
            create_new = False

    if create_new:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            fps=FPS,
            features=features,
            root=ROOT,
            robot_type=ROBOT_TYPE,
            use_videos=True,              # 图像存成 mp4
            video_backend='pyav',         # 绕开 torchcodec 探测（本机它加载不了）
            image_writer_threads=4,
            encoder_threads=2,           # CPU 编码，两端均不依赖 CUDA/TorchCodec
        )
        counts = {}
        print(f'新建 v3.0 数据集：{ROOT}（stats/quantile 会在 save_episode 时自动生成）')
    else:
        print("Resume previous dataset")
        info = check_features_match(ROOT, features)    # ★ 先校验，避免被 lerobot 静默重置
        if info.get('fps') != FPS:
            raise SystemExit(f'现有数据集 fps={info.get("fps")}，本脚本需要 {FPS}，已中止续采。')
        counts = read_disk_counts(ROOT)
        n_have = info.get('total_episodes', sum(counts.values()))
        dataset = LeRobotDataset.resume(repo_id=REPO_NAME, root=ROOT,
                                        video_backend='pyav',
                                        image_writer_threads=4, encoder_threads=2)
        print(f'已有 {n_have} 条，这一次再采 {NUM_DEMO} 条')

    PnPEnv = None
    episode_id = 0
    try:
        PnPEnv = PickPlaceEnv(XML_PATH, seed=SEED)
        print(f'环境就绪：{len(PnPEnv.OBJECTS)} 个物体 {PnPEnv.OBJECTS}，'
              f'obj_init 维度 {OBJ_DIM}，任务 {len(PnPEnv.all_task_targets())} 条')
        # ---- 任务调度：最少优先 ----
        pool = [PnPEnv.make_instruction(list(c)) for c in PnPEnv.all_task_targets()]
        print('\n任务分布（已有条数 -> 本次优先补最少的）:')
        for t in pool:
            print(f'  {t:<48} {counts.get(t, 0)}')
        print()
        tasks = task_stream(pool, counts, random.Random(SEED))
        PnPEnv.set_instruction(next(tasks))
        action = np.zeros(7)
        record_flag = False
        frame_count = 0

        while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
            PnPEnv.step_env()
            if PnPEnv.env.loop_every(HZ=FPS):
                # ---- 成功：存盘 + 重置 + 下一条 ----
                if frame_count > 0 and PnPEnv.check_success():
                    n_frames = frame_count
                    dataset.save_episode()          # 这里会自动算 stats/quantile
                    print(f"  [episode {episode_id}] 完成并存盘（{n_frames} 帧，"
                          f"{n_frames / FPS:.1f} 秒）")
                    PnPEnv.reset()
                    PnPEnv.set_instruction(next(tasks))
                    episode_id += 1
                    record_flag = False
                    frame_count = 0
                    continue

                # ---- 键盘遥操作 ----
                action, reset = PnPEnv.teleop_robot()
                if not record_flag and np.any(action != 0):
                    record_flag = True
                    print(f"  [episode {episode_id}] 开始记录："
                          f"{PnPEnv.instruction}  目标={PnPEnv.target_names}")

                if reset:                           # 按 z：丢弃当前 episode
                    PnPEnv.reset()
                    PnPEnv.set_instruction(next(tasks))
                    dataset.clear_episode_buffer()
                    record_flag = False
                    frame_count = 0
                    continue

                # ---- 图像 ----
                agent_image, wrist_image = PnPEnv.grab_image()
                agent_image = np.array(Image.fromarray(agent_image).resize((IMG_SIZE, IMG_SIZE)))
                wrist_image = np.array(Image.fromarray(wrist_image).resize((IMG_SIZE, IMG_SIZE)))

                # ---- 真正驱动仿真 ----
                joint_q = PnPEnv.step(action)

                if record_flag:
                    dataset.add_frame({
                        'observation.images.agentview': agent_image,
                        'observation.images.wrist': wrist_image,
                        'observation.state': np.asarray(joint_q[:STATE_DIM], dtype=np.float32),
                        'action': PnPEnv.q[:ACTION_DIM].astype(np.float32),
                        'obj_init': PnPEnv.obj_init_pose,
                        'task': PnPEnv.instruction,     # ★ v3.0：指令写在每帧里
                    })
                    frame_count += 1

                PnPEnv.render(teleop=True, idx=episode_id)
    finally:
        try:
            dataset.finalize()      # ★ 必须调用：写 parquet footer / 收尾元数据
        finally:
            if PnPEnv is not None:
                PnPEnv.env.close_viewer()

    images_dir = dataset.root / 'images'
    if images_dir.exists():
        shutil.rmtree(images_dir)

    print(f"\n本轮采到 {episode_id} 条，数据在 {ROOT}")

if __name__ == '__main__':
    main()
