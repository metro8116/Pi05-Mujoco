# 部署参数与输出格式

[返回 README](../readme_zh.md#部署-pi05) · [English](deployment.md)

## 可用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--policy-path` | `lerobot/pi05_base` | Hugging Face 模型 ID 或本地 pretrained_model 目录。 |
| `--data-root` | `./pick_place_data` | 读取本地数据的相机尺寸和 FPS；适配底座时还会读取数据统计量。 |
| `--out` | `./outputs/deploy/deploy_base` | 结果直接写入此目录，不创建 pick_place 子文件夹。 |
| `--n-per-task` | `10` | 所有任务共用的默认评测次数。 |
| `--max-steps` | `1000` | 所有任务共用的单次最大控制步数。 |
| `--apple-runs / --apple-max-steps` | `shared values` | 单独覆盖苹果任务的次数和最大步数。 |
| `--mug-runs / --mug-max-steps` | `shared values` | 单独覆盖绿色杯子任务的次数和最大步数。 |
| `--both-runs / --both-max-steps` | `shared values` | 单独覆盖双目标任务的次数和最大步数。 |
| `--seed` | `0` | 场景种子，每条任务的第 i 次使用 seed + i。 |
| `--device` | `cuda` | 模型推理设备。 |
| `--dtype` | `saved policy configuration` | 可覆盖为 bfloat16 / float32；未适配底座默认在 CUDA 使用 bfloat16、CPU 使用 float32。 |
| `--xml` | `./asset/example_scene_pick_place.xml` | MuJoCo 场景 XML。 |
| `--fps` | `dataset FPS (20 here)` | 控制与视频帧率，不能超过物理仿真频率。 |
| `--save-video` | `1` | 0 / 1：关闭 / 开启 MP4 输出；关闭视频仍需相机参与推理。 |
| `--render-backend` | `egl` | egl 用于无窗口渲染；glfw 使用交互窗口。 |
| `--offline` | `off` | 只使用已缓存或本地模型。 |
| `--use-proxy` | `off` | 保留代理环境变量；否则清除代理以便直连下载。 |
| `--preflight` | `off` | 仅检查本地模型完整性与数据元信息，不推理、不下载。 |
| `--env-only` | `off` | 只运行物理环境，不加载 Pi05；此模式结果不代表模型成功率。 |
| `--no-render` | `off` | 仅与 env-only 配合，跳过图像、视频和窗口。 |

评测次数和最大步数必须为正整数，各任务独立参数优先于共用参数。20 Hz 时 `--max-steps 1000` 对应最多 50 秒仿真时间，不是实际墙钟超时。

底座模型未针对本 FR3 任务微调，适配本地数据统计主要用于检查部署链路；任务能力需与微调模型分别评估。微调 checkpoint 会使用自己保存的归一化处理器。不过，即使部署微调权重，脚本仍会从 `--data-root` 读取相机图像尺寸与 FPS，因此需要保留对应的本地数据元信息。

## 成功条件

三条指令为：

```text
Put the apple in the basket.
Put the green mug in the basket.
Put the apple and the green mug in the basket.
```

所有指令目标的物体参考点需位于篮子内（水平距离小于 0.085 m 且满足篮子高度范围），实际夹爪闭合量不超过 0.15，TCP 高度至少 0.9 m。双目标任务不要求放置顺序。成功、达到控制步数上限或关闭窗口时结束本次评测。

相同场景 seed 用于复现出生位置采样设置，但脚本没有固定 PyTorch 策略噪声，也没有强制 CUDA 运算确定性，所以不能保证机械臂轨迹逐步完全一致。

## 输出目录

```text
outputs/deploy/deploy_finetuned_12k/
├── summary.json
├── evaluation_config.json
├── records/
│   ├── run_apple_0.json
│   ├── run_green_mug_0.json
│   ├── run_apple+green_mug_0.json
│   └── ...
└── videos/
    ├── run_apple_0.mp4
    ├── run_green_mug_0.mp4
    ├── run_apple+green_mug_0.mp4
    └── ...
```

所有文件直接写入 `--out`。视频将固定相机与腕部相机横向拼接。JSON 中的 `video` 路径相对于输出根目录，例如 `videos/run_apple_0.mp4`，即使这条 JSON 位于 `records/` 内也是如此。不保存视频时该字段为 `null`。

### summary.json

顶层字段依次为 **total → tasks → records**。以下为真实 12k 结果的节选；完整的 `records` 有 40 条，为便于阅读，示例省略了单条记录中的 `obj_init`。

```json
{
  "total": {
    "runs": 40,
    "success": 39,
    "success_rate": 0.975
  },
  "tasks": {
    "Put the apple in the basket.": {
      "runs": 10,
      "success": 10,
      "success_rate": 1.0,
      "avg_steps": 223.7
    },
    "Put the green mug in the basket.": {
      "runs": 10,
      "success": 10,
      "success_rate": 1.0,
      "avg_steps": 274.0
    },
    "Put the apple and the green mug in the basket.": {
      "runs": 20,
      "success": 19,
      "success_rate": 0.95,
      "avg_steps": 660.05
    }
  },
  "records": [
    {
      "task": "Put the apple in the basket.",
      "run": 0,
      "success": true,
      "steps": 422,
      "seconds": 9.7,
      "video": "videos/run_apple_0.mp4"
    }
  ]
}
```

- `total.success_rate`：所有成功次数 / 所有评测次数，按任务次数加权，不是三种任务成功率的简单平均。
- `tasks`：每种任务的次数、成功次数、成功率，以及包含失败试验在内的平均控制步数。
- `records`：全部单次记录，按评测任务和运行顺序排列；每条包含 `task`、从 0 开始的 `run`、`success`、`steps`、实际耗时 `seconds`、14 维 `obj_init` 和 `video`。

`records/run_*.json` 保存对应的单次记录。`obj_init` 依次为苹果、杯子的 `[x, y, z, qw, qx, qy, qz]`。`evaluation_config.json` 记录模型、运行模式、FPS、seed 和每条任务实际生效的评测次数及步数上限。脚本4在完整评测结束后保存汇总和配置，单次记录会更早落盘。

不同 checkpoint 或重复试验请指定独立 `--out`；脚本4可能覆盖已有目录中的同名文件。

## 位置泛化参数

`5.position_generation_test.py` 复用上述参数，另有：

| 设置 | 行为 |
| --- | --- |
| 默认输出目录 | `./outputs/position_generation_test` |
| `--outer-side` | 新正方形边长，单位米，默认 `0.15`，必须大于原区域的 `0.10` |

每个物体保持原生成中心，只在大正方形内、整个旧正方形外（包括排除旧边界）采样。出生高度、随机朝向、物体间距和物理落稳流程不变。落稳后复查位置，不合格就重新采样，不会退回旧区域。

输出格式相同。`evaluation_config.json` 的 `test_layout` 额外记录 `name: square_ring`、原区域、新区域边界与中心。脚本5会拒绝非空输出目录，保留旧评测结果。

## 预检查与渲染

```bash
python 4.deploy_pi05.py --offline --preflight \
  --policy-path ./outputs/train/pi05_fr3_pick_place/checkpoints/012000/pretrained_model

python 4.deploy_pi05.py --env-only --no-render \
  --n-per-task 1 --max-steps 10 --out ./outputs/deploy/physics_check
```

默认 EGL 渲染无需 `xvfb-run`。如果服务器无显示器且使用 `--render-backend glfw`，安装 `xvfb` 和 `xauth` 后脚本会自动启动虚拟显示。预检查和 env-only 模式都不能代表已验证策略的实际任务成功率。
