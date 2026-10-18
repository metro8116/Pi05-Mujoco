# 训练参数与输出格式

[返回 README](../readme_zh.md#训练-pi05) · [English](training.md)

## 可用参数

下表是**新训练实际生效的默认值**。整数与模型开关在内部先保持未指定状态，因此续训时会保留 checkpoint 中的配置，除非显式覆盖。布尔值使用 `true` / `false`；`--resume` 等独立开关无需填写值。

| 参数 | 新训练默认值 | 说明 |
| --- | --- | --- |
| `--data-root` | `./pick_place_data` | 本地 LeRobot v3 数据集目录。 |
| `--repo-id` | `fr3_pick_place_basket` | LeRobot 数据集标识；数据从 data-root 读取。 |
| `--policy-path` | `lerobot/pi05_base` | 新训练使用的底座仓库 ID 或本地模型目录。 |
| `--out` | `./outputs/train/pi05_fr3_pick_place` | 训练输出目录；新训练要求该目录尚不存在。 |
| `--job-name` | `pi05_fr3_pick_place` | LeRobot 训练任务名称。 |
| `--batch-size` | `16` | 训练 batch size。 |
| `--steps` | `20000` | 训练目标总步数；续训时也表示最终总步数。 |
| `--save-freq` | `2000` | 每隔多少个训练步保存 checkpoint。 |
| `--log-freq` | `50` | 日志间隔；README 中的实验设置为 10。 |
| `--num-workers` | `4` | 数据加载进程数；本次实验为 8。 |
| `--seed` | `1000` | 训练随机种子，与部署场景种子分别设置。 |
| `--device` | `cuda` | 模型运行设备；cpu 可用于小规模检查。 |
| `--dtype` | `bfloat16 (CUDA)` | 可选 bfloat16 / float32；CPU 默认 float32。 |
| `--train-expert-only` | `false` | 是否只训练动作专家模块。 |
| `--freeze-vision-encoder` | `true` | 是否冻结视觉编码器。 |
| `--gradient-checkpointing` | `true` | 以计算换显存；本次实验设置为 false。 |
| `--lr` | `source-policy setting` | 只对新训练覆盖学习率；本次实验使用 2.5e-5。 |
| `--wandb` | `false` | true / false：是否使用 W&B 记录训练指标。 |
| `--wandb-upload-model` | `false` | true / false：是否额外上传模型文件到 W&B。 |
| `--resume` | `off` | 从 out/checkpoints/last 指向的 checkpoint 恢复。 |
| `--resume-from` | `unset` | 指定 checkpoint、pretrained_model 目录或 train_config.json，自动启用续训。 |
| `--dry-run` | `off` | 检查元数据并打印训练配置，不加载模型、不启动训练。 |
| `--check-data` | `off` | 检查数据集元信息、文件和视频完整性，不训练。 |
| `--offline` | `off` | 仅使用本地模型文件与缓存。 |
| `--use-proxy` | `off` | 保留代理环境变量；默认清除代理变量以直连下载。 |

包装脚本只接收表中的参数，不会自动透传任意 LeRobot 原生参数。某次训练的完整配置请查看其 `pretrained_model/train_config.json`。

## 本次实验配置

以下设置已与 12k checkpoint 保存的配置核对：

| 设置 | 数值 |
| --- | --- |
| 底座 | `lerobot/pi05_base` |
| 总步数 / 保存间隔 | `20000` / `2000` |
| Batch size / 数据加载进程数 | `16` / `8` |
| 设备 / 精度 | `cuda` / `bfloat16` |
| 仅训练动作专家 | `false` |
| 冻结视觉编码器 | `true` |
| 梯度检查点 | `false` |
| 训练种子 / 日志间隔 | `1000` / `10` |
| 优化器 | AdamW，学习率 `2.5e-5`，weight decay `0.01` |
| AdamW betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| 梯度裁剪范数 | `1.0` |
| 保存的调度器配置 | 余弦衰减，warmup 1,000 步，decay 30,000 步，末端学习率 `2.5e-6` |
| 动作块长度 / 执行动作步数 | `50` / `50` |
| W&B 指标 / 模型上传 | 开启 / 关闭 |

视觉编码器被冻结，因此并非更新整个模型的每一个参数。显存不足时可以减小 batch size 或设置 `--gradient-checkpointing true`。当前流程通过独立部署脚本评估 rollout 成功率，训练 loss 不能直接代表任务成功率。

## Checkpoint 输出格式

```text
outputs/train/pi05_fr3_pick_place/
├── checkpoints/
│   ├── 002000/
│   ├── ...
│   ├── 012000/
│   │   ├── pretrained_model/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── train_config.json
│   │   │   ├── policy_preprocessor.json
│   │   │   ├── policy_preprocessor_step_3_normalizer_processor.safetensors
│   │   │   ├── policy_postprocessor.json
│   │   │   └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
│   │   └── training_state/
│   │       ├── optimizer_param_groups.json
│   │       ├── optimizer_state.safetensors
│   │       ├── scheduler_state.json
│   │       ├── rng_state.safetensors
│   │       └── training_step.json
│   ├── 020000/
│   └── last -> 020000/
└── ...
```

数字目录表示训练步数，例如 `012000` 是 12k checkpoint。`last` 指向最近保存的一份，不会自动指向任务成功率最高的模型。示例中的处理器状态文件名来自本次训练，处理器顺序变化时文件名也可能变化。

- **部署推理：** 使用完整的 `pretrained_model/`，包括处理器配置和归一化统计文件。
- **恢复训练：** 同时保留 `pretrained_model/` 与同级 `training_state/`；仅有模型权重无法恢复优化器、调度器与随机状态。
- **准备模型目录：** `outputs/prepared_policy/<hash>/` 用于将底座的配置和数据统计适配到 FR3 接口，与训练 checkpoint 分开。部署微调 checkpoint 时直接使用其中保存的处理器。

本次完整 checkpoint（模型 + 训练状态）约 **19.7 GiB**，保存十份约需 **197 GiB**。W&B 默认不上传模型副本。训练临时文件与 W&B 暂存默认放在项目上一级的 `pi05-cache/`，按文档的 AutoDL 目录布局位于数据盘。

## 恢复训练与监控

```bash
python 3.train_pi05.py --resume --offline --wandb true \
  --out ./outputs/train/pi05_fr3_pick_place
```

脚本自动查找 `checkpoints/last/pretrained_model/train_config.json`，交给原生训练器恢复状态和超参数。例如从 4k 恢复时，`--steps 20000` 表示训练到总计 20k 步，而不是再增加 20k 步。续训时不允许使用 `--lr`，因为优化器和调度器来自保存状态。

启用 W&B 前先执行 `wandb login`。需要保持远程会话并保存终端日志时：

```bash
screen -S pi05_train -L -Logfile /root/autodl-tmp/train.log
# 在screen中运行训练命令。
# 按Ctrl+A，再按D，退出会话但保持训练运行。
```

快速检查；干跑的输出目录需尚未存在：

```bash
python 3.train_pi05.py --check-data --data-root ./pick_place_data
python 3.train_pi05.py --dry-run --out ./outputs/train/config_check
```
