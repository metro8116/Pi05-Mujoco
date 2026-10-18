"""Define the shared Franka FR3 interface and project defaults.

Centralize joint names, state/action dimensions, gripper geometry, home
configuration, and default data and training paths. The state contains
seven arm-joint positions in radians. Each eight-dimensional action
contains seven absolute joint targets followed by gripper closure,
where 0 means open and 1 means closed. Collection, training, and inference
import these constants to use the same robot interface.
"""
ROBOT_TYPE = 'franka_fr3'
ARM_JOINT_NAMES = tuple(f'fr3_joint{i}' for i in range(1, 8))
ARM_DOF = len(ARM_JOINT_NAMES)
STATE_DIM = ARM_DOF
ACTION_DIM = ARM_DOF + 1
CARTESIAN_ACTION_DIM = 7
FINGER_JOINT_NAMES = ('finger_joint1', 'finger_joint2')
FINGER_BODY_NAMES = ('left_finger', 'right_finger')
GRIPPER_ACTUATOR = 'fr3_gripper'
# Simulation customization of the existing Franka Hand, not stock hardware limits.
FINGER_OPEN = 0.045  # Per finger; 90 mm nominal total aperture.
FINGER_LENGTH_SCALE = 1.4
FINGER_BASE_Z = 0.0584
FINGER_PAD_CENTER_Z = 0.0445
TCP_BODY = 'tcp_link'
HOME_Q = (0.0, -0.5, 0.0, -2.2, 0.0, 1.7, 0.7853981634)
DATA_DIR = 'pick_place_data'
REPO_ID = 'fr3_pick_place_basket'
TRAIN_DIR = 'outputs/train/pi05_fr3_pick_place'
