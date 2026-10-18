# Franka FR3 + Franka Hand

The arm is the real seven-axis **FR3**, not a renamed Panda arm. It uses the
MuJoCo Menagerie FR3 visual/collision meshes, joint limits and inertial values.
The Franka Hand is composed from the hand-only portion of Menagerie's
Panda package. Both upstream licenses and unmodified source XMLs are included.

Source repository: https://github.com/google-deepmind/mujoco_menagerie

Pinned commit: `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`.
`SOURCE.json` records source paths and SHA256 hashes. All runtime assets are local.

Local adaptations in `fr3.xml`:

- MuJoCo 3.1.6-compatible general position servos replace newer `kv` /
  `inheritrange` shorthand. Source gains, damping and arm force limits are kept.
- Asset paths are include-relative; there is no global meshdir that changes
  unrelated apple, basket or mug asset resolution.
- Base placement is `(-0.20, 0, 0.80)` metres in the tabletop scene.
- Gravity compensation is enabled on the arm and hand bodies, as expected for
  the position-controlled robot; this does not change gravity on task objects.
- Finger slides use 0..0.045 m travel (90 mm nominal total aperture), coupled
  by equality and an averaged tendon. A single bounded actuator drives both.
  Finger joints explicitly use zero friction loss instead of inheriting arm
  joint friction, allowing the servo to reach the commanded opening.
- Existing finger meshes are stretched 1.4 times along their length. Contact
  pad positions/sizes and finger mass/inertia follow the same stretch; the main
  contact pad grows from 17 to 23.8 mm tall. No new mesh files are required.
- Calibrated `tcp_link` / `grasp_site` lies between the fingertip pads. The
  `egocentric` camera sits outside the hand housing and looks at the grasp area.
- OBJ normals are smoothed for rendering; visual meshes never generate contacts.

Regenerate the local composition after editing the builder:

```sh
python tools/build_fr3_model.py
python tools/check_fr3.py --preview -v
```

`tools/fetch_fr3_assets.py` is an optional network setup command. It reuses the
recorded commit when SOURCE.json exists. It is never called by simulation.

The current mug rim is about 90 mm wide; continue grasping the handle instead
of relying on a close fit around the rim. Objects were not resized.

This is a customized simulation hand; the increased travel is not a stock
Franka Hand hardware specification. `mujoco_env/robot_config.py` holds the
travel and length scale; rebuild with `tools/build_fr3_model.py` after edits.
State/action dimensions remain 7/8, with closure normalized to 0..1. Earlier
FR3 demonstrations use a shorter tool and different closure-to-width mapping;
collect new demonstrations for consistent replay/training with this hand.
