# Training reference

[Back to README](../readme.md#train-pi05) · [简体中文](training_zh.md)

## Command-line options

These are effective defaults for a **new training run**. Integer and policy switches are internally left unset until configuration is assembled, so resuming a checkpoint keeps its saved values unless you explicitly override them. Boolean values are written as `true` / `false`; standalone switches such as `--resume` take no value.

| Option | New-run default | Meaning |
| --- | --- | --- |
| `--data-root` | `./pick_place_data` | Local LeRobot v3 dataset directory. |
| `--repo-id` | `fr3_pick_place_basket` | Dataset identifier used by LeRobot; local files are read from data-root. |
| `--policy-path` | `lerobot/pi05_base` | Base repository ID or local policy directory for a new run. |
| `--out` | `./outputs/train/pi05_fr3_pick_place` | Training output directory; a new run requires a directory that does not yet exist. |
| `--job-name` | `pi05_fr3_pick_place` | LeRobot training job name. |
| `--batch-size` | `16` | Training batch size. |
| `--steps` | `20000` | Target total optimizer steps, including on resume. |
| `--save-freq` | `2000` | Checkpoint interval in optimizer steps. |
| `--log-freq` | `50` | Logging interval; the documented experiment uses 10. |
| `--num-workers` | `4` | Data-loader workers; the experiment uses 8. |
| `--seed` | `1000` | Training seed; separate from deployment scene seeds. |
| `--device` | `cuda` | Policy device; cpu is available for small checks. |
| `--dtype` | `bfloat16 (CUDA)` | bfloat16 or float32; CPU defaults to float32. |
| `--train-expert-only` | `false` | Whether to train only the action expert. |
| `--freeze-vision-encoder` | `true` | Freeze the vision encoder. |
| `--gradient-checkpointing` | `true` | Reduce activation memory at a throughput cost; the experiment sets false. |
| `--lr` | `source-policy setting` | Override policy optimizer learning rate for a new run only; this run uses 2.5e-5. |
| `--wandb` | `false` | true / false: record training metrics in W&B. |
| `--wandb-upload-model` | `false` | true / false: also upload model artifacts to W&B. |
| `--resume` | `off` | Restore the checkpoint referenced by out/checkpoints/last. |
| `--resume-from` | `unset` | Explicit checkpoint directory, pretrained_model directory, or train_config.json; also enables resume. |
| `--dry-run` | `off` | Validate metadata and print the training configuration without loading a model or starting training. |
| `--check-data` | `off` | Check dataset metadata, files, and video integrity without training. |
| `--offline` | `off` | Use local model files/cache only. |
| `--use-proxy` | `off` | Retain proxy environment variables; by default the script clears them for direct downloads. |

Only the wrapper options in this table are accepted directly. For the complete native configuration used by a run, inspect its `pretrained_model/train_config.json`; native LeRobot flags are not automatically forwarded by this wrapper.

## Configuration used for the reported run

Verified against the saved 12k checkpoint configuration:

| Setting | Value |
| --- | --- |
| Base policy | `lerobot/pi05_base` |
| Total steps / saved interval | `20000` / `2000` |
| Batch size / workers | `16` / `8` |
| Device / precision | `cuda` / `bfloat16` |
| Expert-only training | `false` |
| Frozen vision encoder | `true` |
| Gradient checkpointing | `false` |
| Training seed / log interval | `1000` / `10` |
| Optimizer | AdamW, LR `2.5e-5`, weight decay `0.01` |
| AdamW betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Gradient clipping norm | `1.0` |
| Saved scheduler configuration | cosine decay with 1,000 warmup steps; 30,000 decay steps; terminal LR `2.5e-6` |
| Action chunk / executed action steps | `50` / `50` |
| W&B metrics / model artifacts | enabled / disabled |

The frozen vision encoder means this is not an update of every parameter in the full model. Use a smaller batch or `--gradient-checkpointing true` when adapting the command to a GPU with less memory. The original workflow uses a separate deployment script for rollout evaluation; training loss is not a task-success metric.

## Checkpoints and output files

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

Numbered directories use zero-padded training steps: `012000` is the 12k checkpoint. `last` points to the most recently saved checkpoint, not to the checkpoint with the best rollout success. The shown processor state-file names are from this run and can differ if processor ordering changes.

- **Deployment:** use a complete `pretrained_model/` directory, including saved processors and normalization statistics.
- **Resume training:** keep both `pretrained_model/` and its sibling `training_state/`. The model weights alone do not restore the optimizer, scheduler, or random state.
- **Prepared policy:** `outputs/prepared_policy/<hash>/` adapts the base model's configuration and dataset statistics to the local FR3 interface. It is separate from training checkpoints. Fine-tuned deployment uses its saved processors directly.

The observed full checkpoint size is about **19.7 GiB** for model plus training state; ten saved checkpoints require about **197 GiB**. W&B model-artifact upload is off by default. Training temporary files and W&B staging default to the project's parent `pi05-cache/` directory, on the AutoDL data disk for the documented layout.

## Resume and monitoring

```bash
python 3.train_pi05.py --resume --offline --wandb true \
  --out ./outputs/train/pi05_fr3_pick_place
```

The wrapper resolves `checkpoints/last/pretrained_model/train_config.json`, restores native training state, and preserves saved hyperparameters. For example, resuming a 4k checkpoint with `--steps 20000` targets 20k total steps, not 20k additional steps. `--lr` is rejected when resuming because the saved optimizer and scheduler are restored.

Use `wandb login` before enabling W&B. To keep a remote training session and its terminal log:

```bash
screen -S pi05_train -L -Logfile /root/autodl-tmp/train.log
# Run the training command inside this screen.
# Detach with Ctrl+A, then D.
```

Quick checks, using a fresh directory for dry-run output:

```bash
python 3.train_pi05.py --check-data --data-root ./pick_place_data
python 3.train_pi05.py --dry-run --out ./outputs/train/config_check
```
