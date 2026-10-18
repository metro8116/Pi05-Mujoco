<h1 align="center">Pi05 × MuJoCo</h1>

<p align="center"><strong>Language-Guided Pick-and-Place with π0.5</strong></p>

<p align="center">
  <a href="readme.md">English</a> · <a href="readme_zh.md">简体中文</a>
</p>

<p align="center">
  🤗<a href="https://huggingface.co/Metroooooooo/pi05_fr3_finetuned_12k"> π0.5_Finetuned</a> |
  🤗<a href="https://huggingface.co/datasets/Metroooooooo/fr3-pick-place-dataset"> Dataset</a>
</p>

<a id="overview"></a>

## 📌 Overview

A complete simulation workflow for a **Franka FR3 with a parallel gripper**, built with MuJoCo and LeRobot. Collect keyboard demonstrations, fine-tune [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base), and evaluate three language-conditioned tasks: put the apple, the green mug, or both into a basket.

<p align="center">
  <img src="media/pi05_inference_12k.gif" width="760" alt="A fine-tuned Pi05 policy puts the apple and green mug into the basket in MuJoCo">
</p>
<p align="center"><sub>12k-step checkpoint · autonomous inference · 2× playback</sub></p>

The current dataset contains **120 successful demonstrations / 39,747 frames**, recorded at **20 Hz** with agent-view and wrist cameras. The best evaluated checkpoint reaches **97.5% success (39/40)** at **12,000 training steps**.

<a id="contents"></a>

## 📚 Contents

- [🛠️ Installation](#installation)
- [🎮 Collect demonstrations](#collect-demonstrations)
- [🎬 Replay data](#replay-data)
- [🧠 Train π0.5](#train-pi05)
- [🤗 Models and dataset](#models-and-dataset)
- [🚀 Deploy π0.5](#deploy-pi05)
- [📈 Results](#results)
- [📍 Position generalization](#position-generalization)
- [🙏 Acknowledgements](#acknowledgements)

<details>
<summary>📂 Project structure</summary>

```text
mujoco_pick_place_pi05/
├── 1.collect_data_pick_place.py   # Keyboard teleoperation and data collection
├── 2.replay_data_pick_place.py    # Replay recorded actions in MuJoCo
├── 3.train_pi05.py               # Fine-tune Pi05 with LeRobot
├── 4.deploy_pi05.py              # Evaluate the three pick-and-place tasks
├── 5.position_generation_test.py # Evaluate unseen object positions
├── asset/                       # FR3, objects, basket, and scene assets
├── mujoco_env/                  # Simulation and robot control
├── tools/                       # Runtime helpers and utilities
├── docs/                        # Parameters, output formats, and result data
├── media/                       # GIFs, training curves, and evaluation plots
├── environment.yml              # Local Conda bootstrap
├── requirements.txt             # Shared collection/simulation dependencies
├── requirements_train.txt       # Additional cloud training/deployment dependencies
├── install_local.bat            # Windows / Conda installer
└── install_remote.sh            # Cloud GPU / uv installer
```

`pick_place_data/` and `outputs/` are generated locally and are not needed in the GitHub source distribution. Keep the scene assets and the runtime helpers in `tools/`, including `pi05_runtime.py`, `replay_video.py`, and `__init__.py`.

</details>

<a id="installation"></a>

## 🛠️ Installation

Use **Python 3.12** for both environments. Collect and replay on a local Windows machine; train and evaluate on a Linux cloud GPU server. The pinned stack uses **MuJoCo 3.1.6**, **LeRobot 0.6.1**, and **PyTorch 2.11.0**; the cloud environment selects the CUDA 12.8 wheels.

<a id="local-windows--conda"></a>

### 💻 Local: Windows + Conda

Open **Anaconda Prompt** in the project directory:

```bat
cd /d D:\VLA\mujoco_pick_place_pi05
install_local.bat
conda activate lerobot_pi05
```

The installer creates or reuses `lerobot_pi05` from `environment.yml`, installs `requirements.txt`, and runs `pip check`.

<a id="cloud-gpu"></a>

### ☁️ Cloud GPU: Linux + uv

Run these commands from the project root on your cloud GPU server:

```bash
python -m pip install --user uv -i https://mirrors.aliyun.com/pypi/simple
bash install_remote.sh
apt-get update
apt-get install -y libegl1 libgl1 libglfw3
source .venv/bin/activate
```

`requirements_train.txt` includes the shared requirements. Install it through the script in one step; a second installation of `requirements.txt` is unnecessary. The cloud installer uses domestic mirrors and checks the installed dependencies.

<a id="prepare-the-model-cache"></a>

### 📦 Prepare the model cache

Before using the offline commands below, download the base weights and tokenizer once in the cloud environment:

```bash
export HF_HOME="$(dirname "$PWD")/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ENDPOINT=https://hf-mirror.com
hf download lerobot/pi05_base --max-workers 3
```

The PaliGemma tokenizer may require an approved Hugging Face account. After obtaining access, authenticate and cache its small tokenizer/configuration files from the official endpoint:

```bash
HF_ENDPOINT=https://huggingface.co hf auth login
HF_ENDPOINT=https://huggingface.co hf download google/paligemma-3b-pt-224 \
  --include 'tokenizer*' 'special_tokens_map.json' 'config.json'
```

Keep these cache variables available in new terminals, or pass a local model directory with `--policy-path`. Remove `--offline` when intentionally fetching uncached files.

<a id="collect-demonstrations"></a>

## 🎮 Collect demonstrations

Run locally with the simulation window focused:

```bash
python 1.collect_data_pick_place.py --num-demo 30 --seed 1 --root ./pick_place_data
```

`--num-demo` is the number of **additional successful episodes** to collect in this session. A compatible dataset is resumed automatically. Recording starts with the first nonzero control action; successful episodes are saved automatically. `Z` discards the current attempt.

| Keys | Action |
| --- | --- |
| `W A S D` | Move in the horizontal plane |
| `R / F` | Move up / down |
| `Q / E`, arrow keys | Rotate the end effector |
| `Space` | Toggle the gripper |
| `Z` | Discard the current episode and reset |

![Keyboard demonstration collection](media/teleop_collect_data.gif)

The collector writes **LeRobot v3** data:

```text
pick_place_data/
├── data/chunk-000/file-*.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-*.parquet
└── videos/
    ├── observation.images.agentview/chunk-000/file-*.mp4
    └── observation.images.wrist/chunk-000/file-*.mp4
```

| Feature | Shape | Meaning |
| --- | --- | --- |
| `observation.images.agentview` | `256 × 256 × 3` | Fixed-camera RGB video |
| `observation.images.wrist` | `256 × 256 × 3` | Wrist-camera RGB video |
| `observation.state` | `7` | Actual FR3 arm joint angles, in radians |
| `action` | `8` | Seven absolute arm joint targets + gripper closure (`0` open, `1` closed) |
| `obj_init` | `14` | Initial apple and mug poses, each `[x, y, z, qw, qx, qy, qz]` |
| `task_index` | `1` | Language instruction stored in `meta/tasks.parquet` |

State/action values are stored in physical coordinates; training processors apply dataset-statistics normalization. `obj_init` restores scenes during replay and is not a policy input.

<a id="replay-data"></a>

## 🎬 Replay data

```bash
python 2.replay_data_pick_place.py --root ./pick_place_data
```

The simulator restores the recorded initial poses and executes the saved actions. Camera overlays show the recorded RGB streams. Episodes advance automatically and loop.

![Recorded demonstration replay](media/replay_data.gif)

<a id="train-pi05"></a>

## 🧠 Train π0.5

Copy the complete `pick_place_data/` directory to your cloud GPU server, activate `.venv`, and log in to W&B:

```bash
source .venv/bin/activate
wandb login
python 3.train_pi05.py \
  --policy-path lerobot/pi05_base \
  --data-root ./pick_place_data \
  --offline \
  --gradient-checkpointing false \
  --batch-size 16 \
  --num-workers 8 \
  --steps 20000 \
  --save-freq 2000 \
  --log-freq 10 \
  --wandb true \
  --out ./outputs/train/pi05_fr3_pick_place
```

This run uses **batch size 16**, **20k total steps**, **8 workers**, **bfloat16 on CUDA**, and **checkpoints every 2k steps**. The vision encoder is frozen, `train_expert_only=false`, and gradient checkpointing is disabled. W&B records metrics; model uploads are disabled by default.

Resume from the last completed checkpoint:

```bash
python 3.train_pi05.py --resume --offline --wandb true \
  --out ./outputs/train/pi05_fr3_pick_place
```

[Training parameters and checkpoint format →](docs/training.md)

<table>
  <tr>
    <td align="center"><img src="media/train_loss.png" alt="Training loss across 20,000 optimizer steps"><br><sub>Training loss</sub></td>
    <td align="center"><img src="media/train_lr.png" alt="Learning rate across 20,000 optimizer steps"><br><sub>Learning rate</sub></td>
  </tr>
</table>

<a id="models-and-dataset"></a>

## 🤗 Models and dataset

Hugging Face repositories for the **12k-step fine-tuned model** and the dataset containing **120 demonstrations** (available after upload):

| Artifact | Hugging Face link |
| --- | --- |
| π0.5_Finetuned | [Model](https://huggingface.co/Metroooooooo/pi05_fr3_finetuned_12k) |
| Dataset | [Dataset](https://huggingface.co/datasets/Metroooooooo/fr3-pick-place-dataset) |

<a id="deploy-pi05"></a>

## 🚀 Deploy π0.5

Evaluate the **12k checkpoint** used in the results below:

```bash
python 4.deploy_pi05.py \
  --policy-path ./outputs/train/pi05_fr3_pick_place/checkpoints/012000/pretrained_model \
  --data-root ./pick_place_data \
  --offline \
  --both-runs 20 \
  --both-max-steps 1500 \
  --seed 100 \
  --out ./outputs/deploy/deploy_finetuned_12k
```

This evaluates apple and mug tasks **10 times each** with **1,000 control steps** per trial, and the two-object task **20 times** with **1,500 steps**. At 20 Hz those limits correspond to 50 s and 75 s of simulation time; inference affects wall-clock duration. Success requires all instructed objects in the basket, the gripper open, and the end effector retreated above 0.9 m.

[Deployment parameters and output format →](docs/deployment.md)

![Pi05 autonomous inference at 12k steps, played at 2x speed](media/pi05_inference_12k.gif)

<a id="results"></a>

## 📈 Results

![Success rate versus fine-tuning steps](media/success_rate_vs_steps.png)

All checkpoints use the same task counts, step limits, and scene-seed range beginning at 100. The overall rate is weighted over **40 trials**: 10 apple, 10 mug, and 20 two-object trials.

**The 12k-step checkpoint performs best among the evaluated checkpoints: 97.5% overall success (39/40).** Training continues to 20k steps, but the lowest training loss does not identify the highest rollout success in this evaluation.

| Checkpoint | Apple | Green mug | Both objects | Overall |
| --- | ---: | ---: | ---: | ---: |
| Base | 0% | 0% | 0% | 0% |
| 2k | 70% | 40% | 55% | 55% |
| 4k | 70% | 100% | 75% | 80% |
| 6k | 100% | 100% | 60% | 80% |
| 8k | 90% | 90% | 85% | 87.5% |
| 10k | 100% | 100% | 80% | 90% |
| **12k** | **100%** | **100%** | **95%** | **97.5%** |
| 14k | 100% | 80% | 80% | 85% |
| 16k | 100% | 100% | 75% | 87.5% |
| 18k | 100% | 100% | 85% | 92.5% |
| 20k | 100% | 90% | 85% | 90% |

[Underlying scores (CSV)](docs/success_rates.csv) · [Evaluation protocols and statistics (JSON)](docs/evaluation_results.json)

These are finite-sample results from one evaluation seed schedule; the checkpoint comparison does not claim a statistically established ranking across repeated training/evaluation runs.

<a id="position-generalization"></a>

## 📍 Position generalization

Keep each object's original spawn center and test only **outside its original 10 cm square but inside a new 15 cm square**. Objects are not swapped. Original spacing constraints remain active, and settled poses are checked again before evaluation.

```bash
python 5.position_generation_test.py \
  --policy-path ./outputs/train/pi05_fr3_pick_place/checkpoints/012000/pretrained_model \
  --data-root ./pick_place_data \
  --offline \
  --both-runs 20 \
  --max-steps 2000 \
  --outer-side 0.15 \
  --seed 100 \
  --out ./outputs/position_generation_test
```

Use a fresh output directory for a new evaluation. `--outer-side` controls the outer square's side length in metres.

| Task | Successes | Success rate |
| --- | ---: | ---: |
| Apple | 8 / 10 | 80% |
| Green mug | 10 / 10 | 100% |
| Both objects | 14 / 20 | 70% |
| **Overall** | **32 / 40** | **80%** |

The recorded test uses the **12k checkpoint**, seed 100, and **2,000 steps per trial** for every task. Its longer horizon differs from the standard evaluation above, so the two percentages are not a controlled comparison of position alone.

![Pi05 at unseen spawn positions, played at 2x speed](media/position_generalization.gif)

<a id="acknowledgements"></a>

## 🙏 Acknowledgements

- [LeRobot MuJoCo Tutorial](https://github.com/jeongeun980906/lerobot-mujoco-tutorial) — the simulation and demonstration-learning workflow that this project builds on.
- [Hugging Face LeRobot](https://github.com/huggingface/lerobot) — dataset tools, Pi05 integration, and training infrastructure.
