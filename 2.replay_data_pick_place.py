"""Replay recorded FR3 pick-and-place demonstrations in MuJoCo.

Restore each episode's initial robot and object poses, then execute its
recorded actions in simulation. Display the recorded camera images next
to the simulation using sequential video decoding. Episodes are replayed
in a loop; optional profiling reports data, physics, and rendering costs.

Inputs: A local LeRobot v3 dataset and its matching scene XML.
Outputs: An interactive replay window with optional camera overlays.
Usage: python 2.replay_data_pick_place.py --root ./pick_place_data
"""
import argparse
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

# 当前 conda 环境的 Library\bin（Windows 上 MuJoCo/GLFW 需要）
conda_prefix = sys.prefix
dll_path = os.path.join(conda_prefix, "Library", "bin")
if os.name == 'nt' and os.path.isdir(dll_path):
    _dll_directory = os.add_dll_directory(dll_path)

import numpy as np
from mujoco_env.robot_config import DATA_DIR, REPO_ID, STATE_DIM, ACTION_DIM, FINGER_JOINT_NAMES, FINGER_OPEN

from tools.replay_video import SequentialVideoReader

HERE = Path(__file__).resolve().parent


def parser():
    ap = argparse.ArgumentParser(description='窗口回放本地 LeRobot v3 数据集')
    ap.add_argument('--root', type=Path, default=HERE / DATA_DIR)
    ap.add_argument('--xml', type=Path, default=HERE / 'asset/example_scene_pick_place.xml')
    ap.add_argument('--no-video', action='store_true', help='只回放三维仿真，不显示录制的相机画面')
    ap.add_argument('--profile', action='store_true', help='每 100 帧打印读取、物理和渲染耗时')
    return ap


def main():
    args = parser().parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from mujoco_env.pick_place_env import PickPlaceEnv

    REPO_NAME = REPO_ID
    ROOT = args.root.expanduser().resolve()
    XML_PATH = str(args.xml.expanduser().resolve())
    if not (ROOT / 'meta/info.json').is_file():
        raise SystemExit(f'找不到本地数据集 {ROOT}/meta/info.json，请先采集或检查 --root。')
    import json
    info = json.loads((ROOT / 'meta/info.json').read_text(encoding='utf-8'))
    for key, dimension in (('observation.state', STATE_DIM), ('action', ACTION_DIM)):
        if info.get('features', {}).get(key, {}).get('shape') != [dimension]:
            raise SystemExit(f'FR3 回放需要 {key}={dimension} 维；旧 OMY 数据不能直接回放。')

    dataset = LeRobotDataset(REPO_NAME, root=ROOT, video_backend='pyav')
    num_episodes = dataset.num_episodes
    print(f'数据集：{num_episodes} 条 episode，共 {dataset.num_frames} 帧')
    if num_episodes == 0:
        raise SystemExit('数据集没有已保存的 episode，请先完成至少一条采集。')

    # v3.0 每条 episode 的帧区间在 meta.episodes 里（dataset_from_index/to_index）。
    # 注意：新版 meta.episodes 是 HF datasets 的 Dataset（不是 DataFrame），
    # 用列访问而不是 .iterrows()。
    eps = dataset.meta.episodes
    spans = [(int(f), int(t)) for f, t in zip(eps['dataset_from_index'],
                                              eps['dataset_to_index'])]

    # 回放用关节目标动作（数据集里存的就是关节角 + 夹爪）
    PnPEnv = PickPlaceEnv(XML_PATH, action_type='joint_angle')
    try:
        PnPEnv.reset()
        replay(dataset, PnPEnv, spans, show_video=not args.no_video, profile=args.profile)
    finally:
        PnPEnv.env.close_viewer()


def replay(dataset, PnPEnv, spans, show_video=True, profile=False):
    num_episodes = len(spans)
    episode_index = 0
    from_idx, to_idx = spans[episode_index]
    frame_ptr = from_idx
    step = 0
    need_setup = True
    print(f'=== Start episode {episode_index} ({to_idx - from_idx} frames) ===')
    # Read numeric rows without invoking LeRobot's per-frame video decoder or
    # converting images uint8 -> Torch float32 -> NumPy uint8 on every frame.
    rows = dataset.hf_dataset.with_format('numpy')
    cameras = ('observation.images.agentview', 'observation.images.wrist')
    timings = {'read': 0.0, 'physics': 0.0, 'render': 0.0}
    report_start = time.perf_counter()
    report_frames = 0
    with ExitStack() as video_stack:
        readers = {}
        offsets = {}
        while PnPEnv.env.is_viewer_alive():
            started = time.perf_counter()
            PnPEnv.step_env()
            timings['physics'] += time.perf_counter() - started
            if PnPEnv.env.loop_every(HZ=dataset.fps):
                # ---- 当前 episode 播完，切下一个 ----
                if frame_ptr >= to_idx:
                    episode_index = (episode_index + 1) % num_episodes
                    from_idx, to_idx = spans[episode_index]
                    frame_ptr = from_idx
                    step = 0
                    need_setup = True
                    video_stack.close()
                    readers.clear()
                    PnPEnv.reset()
                    print(f'=== Switching to episode {episode_index} '
                          f'({to_idx - from_idx} frames) ===')
                    continue

                started = time.perf_counter()
                data = rows[frame_ptr]

                # ---- 每条第一帧：设置指令 + 复原物体初始位姿 ----
                if step == 0 and need_setup:
                    PnPEnv.set_instruction(dataset.meta.tasks.iloc[int(data['task_index'])].name)
                    PnPEnv.set_obj_pose(np.asarray(data['obj_init'], dtype=np.float64))
                    # 机械臂也要恢复到这一条开始时的关节角：录制是从"上一次结束的姿态"
                    # 起步的，回放若从 home 位逐步追赶，抓取位姿就对不上（杯子会掉）。
                    PnPEnv.env.set_qpos_joints(
                        PnPEnv.joint_names,
                        np.asarray(data['observation.state'], dtype=np.float64))
                    closure = float(np.clip(data['action'][-1], 0, 1))
                    opening = FINGER_OPEN * (1 - closure)
                    PnPEnv.env.set_qpos_joints(list(FINGER_JOINT_NAMES), np.full(2, opening))
                    PnPEnv.gripper_cmd = closure
                    need_setup = False
                    if show_video:
                        episode = dataset.meta.episodes[episode_index]
                        for key in cameras:
                            offsets[key] = float(episode[f'videos/{key}/from_timestamp'])
                            path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
                            readers[key] = video_stack.enter_context(
                                SequentialVideoReader(path, offsets[key], dataset.fps))

                # ---- 执行 action ----
                action = data['action']
                PnPEnv.step(np.asarray(action, dtype=np.float64))

                # ---- 连续解码两路视频，直接使用 HWC uint8 ----
                if show_video:
                    timestamp = float(data['timestamp'])
                    PnPEnv.rgb_agent = readers[cameras[0]].read(offsets[cameras[0]] + timestamp)
                    PnPEnv.rgb_ego = readers[cameras[1]].read(offsets[cameras[1]] + timestamp)
                else:
                    PnPEnv.rgb_agent = PnPEnv.rgb_ego = None
                timings['read'] += time.perf_counter() - started
                # idx 必须显式传，否则屏幕上的标签永远是 Episode 0
                started = time.perf_counter()
                PnPEnv.render(idx=episode_index)
                timings['render'] += time.perf_counter() - started

                step += 1
                frame_ptr += 1
                report_frames += 1
                if profile and report_frames == 100:
                    elapsed = time.perf_counter() - report_start
                    costs = '，'.join(f'{name}={seconds * 1000 / report_frames:.1f}ms/帧'
                                     for name, seconds in timings.items())
                    print(f'回放实测 {report_frames / elapsed:.1f} FPS（数据 {dataset.fps}Hz）：{costs}')
                    timings = dict.fromkeys(timings, 0.0)
                    report_start = time.perf_counter()
                    report_frames = 0

if __name__ == '__main__':
    main()
