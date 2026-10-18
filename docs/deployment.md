# Deployment reference

[Back to README](../readme.md#deploy-pi05) · [简体中文](deployment_zh.md)

## Command-line options

| Option | Default | Meaning |
| --- | --- | --- |
| `--policy-path` | `lerobot/pi05_base` | Hugging Face policy ID or local pretrained_model directory. |
| `--data-root` | `./pick_place_data` | Local dataset metadata for camera dimensions and FPS; also supplies statistics when adapting the base policy. |
| `--out` | `./outputs/deploy/deploy_base` | Write results directly into this directory; no pick_place subfolder. |
| `--n-per-task` | `10` | Fallback number of trials for every task. |
| `--max-steps` | `1000` | Fallback maximum control steps per trial. |
| `--apple-runs / --apple-max-steps` | `shared values` | Override apple trial count and horizon. |
| `--mug-runs / --mug-max-steps` | `shared values` | Override green-mug trial count and horizon. |
| `--both-runs / --both-max-steps` | `shared values` | Override two-object trial count and horizon. |
| `--seed` | `0` | Scene seed; trial i uses seed + i for each task. |
| `--device` | `cuda` | Policy inference device. |
| `--dtype` | `saved policy configuration` | Optional bfloat16 / float32 override. An unadapted base uses bfloat16 on CUDA or float32 on CPU when unspecified. |
| `--xml` | `./asset/example_scene_pick_place.xml` | MuJoCo scene XML. |
| `--fps` | `dataset FPS (20 here)` | Control and video frame rate, no higher than physics frequency. |
| `--save-video` | `1` | 0 / 1: disable / enable MP4 output; cameras are still needed for policy inference. |
| `--render-backend` | `egl` | egl for offscreen inference; glfw for an interactive viewer. |
| `--offline` | `off` | Use cached/local policy files only. |
| `--use-proxy` | `off` | Keep proxy environment variables; otherwise clear them for direct downloads. |
| `--preflight` | `off` | Check local model integrity and dataset metadata, without GPU inference or downloads. |
| `--env-only` | `off` | Run physics without loading Pi05; the results are not model success rates. |
| `--no-render` | `off` | Only valid with env-only; skip images, video and the viewer. |

Trial counts and step limits must be positive integers. Task-specific overrides take precedence over the shared values. `--max-steps 1000` means up to 50 simulation seconds at 20 Hz; it does not set a wall-clock timeout.

The supplied base policy has not been fine-tuned on this FR3 task. Its local dataset-statistics adaptation checks the deployment pipeline; task results should be evaluated separately from a fine-tuned checkpoint. Fine-tuned checkpoints retain their own saved normalization processors. Even when deploying a checkpoint, `--data-root` is read for the image feature sizes and FPS, so keep the expected local dataset metadata available.

## Success criteria

The three instructions are:

```text
Put the apple in the basket.
Put the green mug in the basket.
Put the apple and the green mug in the basket.
```

A trial succeeds when every instructed object's body reference point is in the basket (horizontal distance < 0.085 m and basket-height bounds), actual gripper closure is at most 0.15, and TCP height is at least 0.9 m. The two-object task does not impose a placement order. Trials stop on success, the control-step limit, or viewer closure.

A repeated scene seed reproduces the spawn sampling setup, but this script does not seed PyTorch's policy noise or enforce deterministic CUDA kernels. Matching seed values therefore do not guarantee identical policy trajectories.

## Output layout

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

Every output is directly under `--out`. Videos concatenate agent-view and wrist images side by side. `video` values such as `videos/run_apple_0.mp4` are relative to that output root, including inside a JSON stored in `records/`. If video saving is off, the field is `null`.

### summary.json

Top-level keys are ordered **total → tasks → records**. An abbreviated excerpt of the actual 12k results follows; the real `records` array contains 40 entries, and the record's `obj_init` field is omitted here for readability.

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

- `total.success_rate`: total successes divided by total trials, weighted by task counts rather than an unweighted average of task rates.
- `tasks`: task-level runs, successes, success rate, and average control steps across all trials (including failures).
- `records`: every trial, grouped in the same task/run order as evaluation. Each entry contains `task`, zero-based `run`, `success`, `steps`, wall-clock `seconds`, 14-value `obj_init`, and `video`.

Each `records/run_*.json` contains the matching single record. `obj_init` stores apple then mug, with `[x, y, z, qw, qx, qy, qz]` for each object. `evaluation_config.json` stores the policy, mode, FPS, seed and effective per-task counts/step limits. Script 4 saves the aggregate summary/configuration after the full evaluation completes; completed individual records are saved earlier.

Use a separate `--out` for each checkpoint or repeated experiment: script 4 can overwrite matching files in an existing directory.

## Position-generalization options

`5.position_generation_test.py` reuses these options, with two differences:

| Setting | Behavior |
| --- | --- |
| Default output | `./outputs/position_generation_test` |
| `--outer-side` | New square side length in metres; default `0.15`, greater than the original `0.10` |

Each object retains its original region center. Positions must lie inside the new square and outside the complete original square, including its boundary. Normal spawn heights, random yaw, separation constraints and physical settling are preserved. Settled centers are checked again; invalid placements are resampled rather than accepted inside the training region.

The output format remains identical. `evaluation_config.json` additionally includes `test_layout` with `name: square_ring`, old/new boundaries and centers. Unlike script 4, script 5 rejects a nonempty output directory to preserve old evaluation results.

## Checks and rendering

```bash
python 4.deploy_pi05.py --offline --preflight \
  --policy-path ./outputs/train/pi05_fr3_pick_place/checkpoints/012000/pretrained_model

python 4.deploy_pi05.py --env-only --no-render \
  --n-per-task 1 --max-steps 10 --out ./outputs/deploy/physics_check
```

The default EGL backend does not require `xvfb-run`. If using `--render-backend glfw` on a server with no display, install `xvfb` and `xauth`; the script starts `xvfb-run` automatically. Neither preflight nor env-only checks demonstrate a working learned policy.
