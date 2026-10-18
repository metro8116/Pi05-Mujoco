"""Simulate language-conditioned FR3 pick-and-place tasks in MuJoCo.

Manage the apple, green mug, basket, and robot across task resets.
Sample object positions near their fixed spawn centers, settle the scene,
and expose camera observations and joint-action control to collection and
evaluation scripts. Determine success from target placement, gripper
release, and robot retreat so training data and evaluation share task rules.
Inputs include the scene and task settings; outputs include observations,
object poses, and task-completion status.
"""
import copy
import random

import glfw
import mujoco
import numpy as np

from mujoco_env.ik import solve_ik
from mujoco_env.transforms import r2rpy, rpy2r
from mujoco_env.utils import add_title_to_img
from mujoco_env.y_env2 import SimpleEnv2
from mujoco_env.robot_config import (
    ARM_DOF, ACTION_DIM, STATE_DIM, HOME_Q, ROBOT_TYPE,
    FINGER_JOINT_NAMES, FINGER_BODY_NAMES, FINGER_OPEN, GRIPPER_ACTUATOR,
)


class PickPlaceEnv(SimpleEnv2):
    """抓放环境：按语言指令把苹果、绿色杯子或两者放进篮子。"""

    # ------------------------------------------------------------------ 场景常量
    XML_PATH = './asset/example_scene_pick_place.xml'
    ROBOT_TYPE = ROBOT_TYPE
    ARM_DOF = ARM_DOF
    STATE_DIM = STATE_DIM
    ACTION_DIM = ACTION_DIM

    OBJECTS = ('apple', 'mug_green')
    BODY = {
        'apple': 'body_obj_apple',
        'mug_green': 'body_obj_mug_green',
    }
    # 指令里显示的名字（回放/评估时按这些词解析指令）
    DISPLAY = {
        'apple': 'apple',
        'mug_green': 'green mug',
    }

    # 篮子（静止刚体，位置写在 asset/fruit/fruit_basket.xml 里，这里读出来做判定）
    # 数值与 XML 生成脚本 generate_fruit_meshes.py 的 BASKET 参数一致
    BASKET_BODY = 'body_basket'
    BASKET_INNER_R = 0.105      # 竖条围成的内切半径（只装 1 杯 + 1 苹果，篮子收小了一点）
    BASKET_FLOOR_Z = 0.810      # 底板顶面的世界 z
    BASKET_RIM_Z = 0.896        # 篮沿顶面的世界 z
    IN_BASKET_R = 0.085         # 判定"在篮子里"的水平半径（篮子内径 10.5cm，留些余量：
                                # 杯子歪着靠在壁上时中心会偏到 8cm 左右，卡太严会导致明明放进去却不判成功）

    # Table coordinates are retained; the FR3 base is at (-0.20, 0, 0.80).
    TABLE_TOP_Z = 0.80
    WORK_X = (0.145, 0.340)
    WORK_Y = (-0.035, 0.315)

    # 与 OBJECTS 一一对应：苹果始终在近处，绿杯始终在远处，不交换区域。
    # 每个区域 10cm x 10cm；保留原中心，各轴向两侧扩展 5cm。
    SPAWN_SPOTS = (
        ((0.145, 0.245), (-0.035, 0.065)),  # apple：近处，中心 (0.195, 0.015)
        ((0.240, 0.340), (0.215, 0.315)),   # mug_green：远处，中心 (0.290, 0.265)
    )
    MIN_OBJECT_DIST = 0.15       # 两两之间的最小间距（含与篮子的距离），即 >=15cm
    MIN_BASE_DIST = 0.18         # Minimum horizontal distance to the FR3 base
    MAX_SPAWN_TRIES = 1500      # 随机采样上限，超了就检查各自区域中心
    MAX_RESPAWN = 5             # 落稳后如果物体互相碰到，最多重摆几次
    DROP_H = 0.0005             # 出生时离桌面只留 0.5mm：落下越轻，落稳后越不容易带残余转动

    # 机械臂 home 位姿（与 y_env2 保持一致）
    HOME_P = np.array([0.3, 0.0, 1.05])
    HOME_RPY_DEG = np.array([180.0, 0.0, 0.0])

    # 初始视角（比 y_env2 更近，能同时看到篮子和三样物体）
    VIEW_AZIMUTH = 150.0
    VIEW_DISTANCE = 1.80
    VIEW_ELEVATION = -25.0
    VIEW_LOOKAT = [0.10, -0.02, 1.12]

    # Single Franka Hand tendon, length in metres. Policy closure stays 0..1.
    GRIPPER_KP = 1200.0
    GRIPPER_FORCE = 30.0
    GRIPPER_RATE = 0.10  # Per 20 Hz control update, rather than per physics tick.
    GRASP_DETECT = True
    GRASP_ERR = 0.12     # Normalized closure error when an object blocks the fingers.
    GRASP_VEL = 0.4      # Normalized finger speed per second.
    GRASP_EXTRA = 0.06

    # ------------------------------------------------------------------ 初始化
    def __init__(self,
                 xml_path=XML_PATH,
                 action_type='eef_pose',
                 state_type='joint_angle',
                 seed=None,
                 instruction_weights=None,
                 enable_viewer=True):
        """
        args:
            xml_path: 场景 XML，默认 ./asset/example_scene_pick_place.xml
            action_type: 与 SimpleEnv2 相同，'eef_pose' / 'delta_joint_angle' / 'joint_angle'
            state_type: 与 SimpleEnv2 相同
            seed: 随机种子。给了就同时固定物体摆放(np.random)和任务抽取(self.task_rng)，
                  不给则每次都不一样（注意：y_env2 用了未播种的 random，
                  这个环境特意改成可复现的）
            instruction_weights: {'single': 1.0, 'pair': 1.0, 'all': 1.0}
                  想多录单物体任务就调大 single
        """
        self.enable_viewer = enable_viewer
        # 这些字段要在 super().__init__ 之前准备好，因为父类构造里就会调用 self.reset()
        self.instruction_weights = dict(instruction_weights or {'single': 1.0, 'pair': 1.0, 'all': 1.0})
        self.task_rng = random.Random(seed)      # 播种 -> 任务顺序/打乱可复现
        self.task_queue = []                     # 本轮还没用掉的任务
        self.task_round = 0                      # 已经循环到第几轮
        # 完成判定是否要求"松爪 + 机械臂抬到 0.9 以上"（和 y_env2 的成功条件一致）
        self.require_retreat = True
        self.target_names = list(self.OBJECTS)
        self.target_objects = [self.BODY[f] for f in self.OBJECTS]
        self.obj_target = self.target_objects[0]
        self.gripper_cmd = 0.0          # 限速后实际发出的夹爪命令
        self.grasp_hold = None          # 夹住后冻结的命令值
        self.basket_p = np.array([0.30, -0.32, 0.80])   # 与 fruit_basket.xml 一致（reset 里会重新读）
        self.basket_xy = self.basket_p[:2].copy()
        self.obj_init_pose = np.zeros(len(self.OBJECTS) * 7, dtype=np.float32)

        super().__init__(xml_path,
                         action_type=action_type,
                         state_type=state_type,
                         seed=seed)

        self.soften_gripper()

    def soften_gripper(self):
        """Calibrate the single Franka Hand tendon actuator (SI units)."""
        model = self.env.model
        index = int(model.actuator(GRIPPER_ACTUATOR).id)
        model.actuator_gainprm[index, 0] = self.GRIPPER_KP
        model.actuator_biasprm[index, 1:3] = [-self.GRIPPER_KP, -12.0]
        model.actuator_forcelimited[index] = 1
        model.actuator_forcerange[index] = [-self.GRIPPER_FORCE, self.GRIPPER_FORCE]
        self.gripper_actuator_ids = [index]
        self._finger_body_ids = {int(model.body(name).id) for name in FINGER_BODY_NAMES}
        self._grip_qpos_adr = [int(model.jnt_qposadr[model.joint(name).id]) for name in FINGER_JOINT_NAMES]
        self._grip_dof_adr = [int(model.jnt_dofadr[model.joint(name).id]) for name in FINGER_JOINT_NAMES]
        print(f'  FR3 gripper: kp={self.GRIPPER_KP:g} N/m, force_limit={self.GRIPPER_FORCE:g} N')

    def init_viewer(self):
        """初始视角：对准桌面上的物体和篮子（不传 azimuth/lookat 会落到
        MuJoCoParserClass 的通用默认值，开局视角会很偏）。"""
        if not self.enable_viewer:
            self.env.reset()
            return
        self.env.reset()
        self.env.init_viewer(
            azimuth           = self.VIEW_AZIMUTH,
            distance          = self.VIEW_DISTANCE,
            elevation         = self.VIEW_ELEVATION,
            lookat            = list(self.VIEW_LOOKAT),
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay   = False,
            loc_rgb_overlay   = 'top right',
        )

    def step(self, action):
        """Rate-limit normalized closure; stop after a loaded fingertip stalls."""
        state = super().step(action)
        target = float(np.clip(action[-1], 0.0, 1.0))
        if target < self.gripper_cmd - 1e-9:
            self.grasp_hold = None
            goal = target
        else:
            if self.GRASP_DETECT and self.gripper_cmd > 0.2 and self.grasp_hold is None:
                actual = self.get_gripper_state()
                velocity = float(np.mean(np.abs(self.env.data.qvel[self._grip_dof_adr]))) / FINGER_OPEN
                if self.gripper_cmd - actual > self.GRASP_ERR and velocity < self.GRASP_VEL:
                    self.grasp_hold = min(self.gripper_cmd + self.GRASP_EXTRA, 1.0)
            goal = target if self.grasp_hold is None else min(target, self.grasp_hold)
        rate = self.GRIPPER_RATE
        self.gripper_cmd = goal if rate is None else float(
            self.gripper_cmd + np.clip(goal - self.gripper_cmd, -rate, rate))
        self.q[-1] = self.gripper_cmd
        return state

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None):
        """回到初始状态：机械臂回 home、物体重新随机摆放、重新抽一条指令。"""
        if seed is not None:
            np.random.seed(seed)
            self.task_rng.seed(seed)
            self.task_queue = []      # 同一个 seed 重新开一轮，出题顺序可复现

        # 1) 机械臂 home
        q_zero, _, _ = solve_ik(
            env                = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'tcp_link',
            q_init             = np.asarray(HOME_Q, dtype=float),
            p_trgt             = self.HOME_P,
            R_trgt             = rpy2r(np.deg2rad(self.HOME_RPY_DEG)),
            ik_err_th=1e-5, ik_eps=1e-5, ik_th=np.radians(5.0),
        )
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)
        # 清掉上一回合残留的速度/外力：否则 reset(同一个 seed) 摆出来的位置
        # 会不一样（机械臂扫过物体、物体带着余速滚动）
        self.env.data.qvel[:] = 0.0
        self.env.data.qacc[:] = 0.0
        self.env.data.xfrc_applied[:] = 0.0
        self.env.data.ctrl[:] = 0.0
        self.env.data.ctrl[:ARM_DOF] = q_zero
        self.env.data.ctrl[-1] = FINGER_OPEN
        for name in FINGER_JOINT_NAMES:
            self.env.data.qpos[self.env.model.jnt_qposadr[self.env.model.joint(name).id]] = FINGER_OPEN
        mujoco.mj_forward(self.env.model, self.env.data)

        # 2) 读一下篮子实际位置（静止刚体，位置在 XML 里），用于成功判定
        self.basket_p, _ = self.env.get_pR_body(body_name=self.BASKET_BODY)
        self.basket_xy = self.basket_p[:2].copy()

        # 3) 随机摆放物体
        self.randomize_objects()

        # 4) 机器人内部状态（父类 step()/teleop 依赖这些字段）
        self.last_q = copy.deepcopy(q_zero)
        self.q = np.concatenate([q_zero, [0.0]])
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')

        # 5) 让物体自己落稳。正常间距下它们不应该互相碰到；
        #    万一落下来挤到一起/叠起来（比如绿杯子架在蓝杯子上），就重新摆一次。
        self._object_body_owner = {}
        for name in self.OBJECTS:
            root = int(self.env.model.body(self.BODY[name]).id)
            for bid in range(root, self.env.model.nbody):
                current = bid
                while current and current != root:
                    current = int(self.env.model.body_parentid[current])
                if current == root:
                    self._object_body_owner[bid] = name
        self._fruit_body_ids = set(self._object_body_owner)
        for attempt in range(self.MAX_RESPAWN):
            for _ in range(100):
                self.step_env()
            if not self.fruits_touching():
                break
            if attempt + 1 < self.MAX_RESPAWN:
                self.randomize_objects()

        # 5b) 落稳之后把物体的速度清零 —— 否则它们会带着落地余速继续滑动/滚动，
        #     球形的苹果尤其明显（一开局就在桌面上晃）。清零后是"完全静止"起始。
        self._freeze_objects()
        for _ in range(20):          # 再走一小段，确认没有残余速度
            self.step_env()
        self._freeze_objects()
        self.spawn_attempts = attempt + 1
        self.obj_init_pose = self.get_obj_pose()

        # 6) 抽一条指令
        self.set_instruction()

        # 7) 键盘/夹爪状态
        self.gripper_state = False
        self.gripper_cmd = 0.0
        self.grasp_hold = None
        self.past_chars = []

        print('DONE INITIALIZATION')
        print(f'  instruction : {self.instruction}')
        print(f'  spawn       : mode={self.spawn_mode} attempts={self.spawn_attempts}')
        for name in self.OBJECTS:
            p = self.env.get_p_body(self.BODY[name])
            tag = '目标' if name in self.target_names else '  '
            print(f'  {tag} {name:<6}: {np.round(p, 3)}')

    def gripper_reference(self):
        """World-space midpoint of the calibrated Franka fingertip pads."""
        return self.env.get_p_body('tcp_link').copy()

    def finger_touching_fruit(self):
        """夹爪手指当前是否已经碰到任意物体（用 MuJoCo 的真实接触列表判断）。"""
        model, data = self.env.model, self.env.data
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if (b1 in self._finger_body_ids and b2 in self._fruit_body_ids) or \
               (b2 in self._finger_body_ids and b1 in self._fruit_body_ids):
                return True
        return False

    def fruits_touching(self):
        """两样物体之间是否接触（正常间距下不该接触，接触说明挤在一起了）。"""
        model, data = self.env.model, self.env.data
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if (b1 in self._fruit_body_ids and b2 in self._fruit_body_ids
                    and self._object_body_owner[b1] != self._object_body_owner[b2]):
                return True
        return False

    def _compute_spawn_dz(self):
        """每个物体"原点 -> 最低点"的 z 偏移。

        物体的网格原点在底面，所以摆到 z=桌面高度就行；但杯子的模型原点在
        几何中心（mug 本地还有一层子 body），直接按桌面高度摆会有一半埋进桌子。
        这里遍历物体 body（含所有后代 body）的全部 geom，算出最低点。
        """
        model = self.env.model
        self.spawn_dz = {}
        for name in self.OBJECTS:
            root = int(model.body(self.BODY[name]).id)
            zmin = np.inf
            for b in range(model.nbody):
                # 只统计 root 自己及其后代
                cur, is_child = b, False
                while cur != 0:
                    if cur == root:
                        is_child = True
                        break
                    cur = int(model.body_parentid[cur])
                if not is_child:
                    continue
                for g in range(model.ngeom):
                    if int(model.geom_bodyid[g]) != b:
                        continue
                    gtype = int(model.geom_type[g])
                    gp = model.geom_pos[g]
                    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                        mid = int(model.geom_dataid[g])
                        a = int(model.mesh_vertadr[mid])
                        v = model.mesh_vert[a:a + int(model.mesh_vertnum[mid])]
                        R = np.zeros(9)
                        mujoco.mju_quat2Mat(R, model.geom_quat[g])
                        pts = v @ R.reshape(3, 3).T + gp
                        zmin = min(zmin, float(pts[:, 2].min()))
                    else:
                        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                            half = float(model.geom_size[g][2])
                        elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE,
                                       mujoco.mjtGeom.mjGEOM_CYLINDER):
                            half = float(model.geom_size[g][1])
                        else:
                            half = float(model.geom_size[g][0])
                        zmin = min(zmin, float(gp[2] - half))
            self.spawn_dz[name] = 0.0 if not np.isfinite(zmin) else float(zmin)
        print('  spawn dz    : ' + ', '.join(
            f'{n}={self.spawn_dz[n] * 1000:+.1f}mm' for n in self.OBJECTS))

    def _freeze_objects(self):
        """把三个物体（freejoint）的速度/加速度清零，让它们以"完全静止"开始。

        为什么需要：物体是从桌面之上 2mm 落下、再跑 100 步落稳的，
        结束时往往还剩一点速度；球形的苹果会因此一直小幅滚动，
        遥操作一开局就看到它在晃。清零后位置/姿态不变，只是不再动。
        """
        model, data = self.env.model, self.env.data
        for name in self.OBJECTS:
            bid = int(model.body(self.BODY[name]).id)
            jid = int(model.body_jntadr[bid])
            if jid < 0 or int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            adr = int(model.jnt_dofadr[jid])
            data.qvel[adr:adr + 6] = 0.0
            data.qacc[adr:adr + 6] = 0.0
        mujoco.mj_forward(model, data)

    def randomize_objects(self):
        """在各自固定区域内随机摆放两个物体，不交换区域，保留随机朝向。"""
        if not hasattr(self, 'spawn_dz'):
            self._compute_spawn_dz()

        pts, self.spawn_mode = self._sample_spawn_points()

        for name, p in zip(self.OBJECTS, pts):
            # spawn_dz 是"原点到最低点"的偏移（杯子为负，因为原点在几何中心）：
            # 物体原点应该放在桌面之上 |spawn_dz| 处，最低点才刚好贴着桌面
            q = np.array([p[0], p[1],
                          self.TABLE_TOP_Z - self.spawn_dz[name] + self.DROP_H])
            yaw = np.random.uniform(-np.pi, np.pi)
            self.env.set_p_base_body(body_name=self.BODY[name], p=q, forward=False)
            self.env.set_R_base_body(body_name=self.BODY[name],
                                     R=rpy2r(np.array([0.0, 0.0, yaw])))
        self.env.forward(increase_tick=False)

    def _pairwise_ok(self, pts):
        """三个约束：物体两两、物体到篮子中心 >= MIN_OBJECT_DIST；
        每个物体离机械臂底座 >= MIN_BASE_DIST。"""
        names = list(pts)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                a_xy = np.asarray(pts[a])[:2]
                b_xy = np.asarray(pts[b])[:2]
                if np.linalg.norm(a_xy - b_xy) <= self.MIN_OBJECT_DIST:
                    return False
        basket_xy = np.asarray(self.basket_xy, dtype=float)
        if not hasattr(self, 'base_xy'):
            self.base_xy = np.asarray(self.env.get_p_body('fr3_base')[:2], dtype=float)
        for a in names:
            a_xy = np.asarray(pts[a])[:2]
            if np.linalg.norm(a_xy - basket_xy) <= self.MIN_OBJECT_DIST:
                return False
            if np.linalg.norm(a_xy - self.base_xy) < self.MIN_BASE_DIST:
                return False
        return True

    def _min_pair_dist(self, pts):
        return min(float(np.linalg.norm(pts[a][:2] - pts[b][:2]))
                   for i, a in enumerate(self.OBJECTS)
                   for b in self.OBJECTS[i + 1:])

    def _sample_in_spot(self, spot):
        (xl, xh), (yl, yh) = spot
        return np.array([np.random.uniform(xl, xh), np.random.uniform(yl, yh),
                         self.TABLE_TOP_Z + self.DROP_H])

    def _sample_spawn_points(self):
        """按 OBJECTS 顺序在固定区域内采样，同时检查物体、篮子和底座间距。"""
        names = list(self.OBJECTS)
        if len(names) != len(self.SPAWN_SPOTS):
            raise ValueError('OBJECTS 与 SPAWN_SPOTS 数量必须一致')
        for _ in range(self.MAX_SPAWN_TRIES):
            cand = {n: self._sample_in_spot(s) for n, s in zip(names, self.SPAWN_SPOTS)}
            if self._pairwise_ok(cand):
                return [cand[n] for n in names], 'random'

        # 兜底也保持物体与区域一一对应，不能靠交换位置绕过约束。
        centers = [np.array([0.5 * (s[0][0] + s[0][1]), 0.5 * (s[1][0] + s[1][1]),
                             self.TABLE_TOP_Z + self.DROP_H]) for s in self.SPAWN_SPOTS]
        cand = dict(zip(names, centers))
        if not self._pairwise_ok(cand):
            raise RuntimeError('固定出生区域无法满足间距约束，请检查 SPAWN_SPOTS 和最小间距配置')
        return centers, 'centers'

    # ------------------------------------------------------------------ 语言指令
    def make_instruction(self, targets):
        """把一个物体列表变成一句指令（名字带颜色，如 green mug）。"""
        targets = [f for f in self.OBJECTS if f in targets]      # 固定顺序
        names = [self.DISPLAY[f] for f in targets]
        if len(targets) == 1:
            return f'Put the {names[0]} in the basket.'
        if len(targets) == len(self.OBJECTS) and len(self.OBJECTS) > 2:
            return 'Put everything in the basket.'
        return f'Put the {names[0]} and the {names[1]} in the basket.'

    def parse_instruction(self, text):
        """从指令文本反推目标物体（回放/评估时用，避免颜色/名称对不上）。"""
        t = text.lower()
        if ('all' in t) or ('everything' in t):
            return list(self.OBJECTS)
        targets = [name for name in self.OBJECTS if self.DISPLAY[name] in t]
        if len(targets) == 0:
            raise ValueError(
                f'指令里找不到物体名（apple / blue mug / green mug / everything）：{text!r}')
        return targets

    # ------------------------------------------------------------------ 任务顺序
    # 采集时的任务顺序（默认）：固定 7 条任务 -> 一轮一轮循环 -> 每轮内部打乱。
    #   为什么不用纯随机：7 条任务纯随机抽，20 条里某些任务可能只采到 1 条甚至 0 条，
    #   语言条件策略会被"采得多的那条指令"带偏，评估也不可信。
    #   为什么每轮打乱：严格按固定顺序循环，任务会和"第几条 episode"相关，
    #   之后按时间切训练/验证集就会有偏；每轮洗一次牌既均衡又和序号无关。
    TASK_CYCLE = True            # False: 退回按 instruction_weights 随机抽
    SHUFFLE_EACH_ROUND = True    # 每轮（7 条）内部重新打乱

    def all_task_targets(self):
        """所有任务的目标组合。2 个物体时是 3 条：苹果 / 绿杯 / 两个都要。"""
        objs = list(self.OBJECTS)
        singles = [(o,) for o in objs]
        pairs = [(objs[i], objs[j])
                 for i in range(len(objs)) for j in range(i + 1, len(objs))]
        out = singles + pairs
        if len(objs) > 2:            # 3 个及以上才有"全部"这条
            out = out + [tuple(objs)]
        return out

    def _refill_task_queue(self):
        """把 7 条任务装进本轮队列；开了打乱就洗一次牌。"""
        q = list(self.all_task_targets())
        if self.SHUFFLE_EACH_ROUND:
            self.task_rng.shuffle(q)
        self.task_queue = q
        self.task_round += 1

    def set_instruction(self, given=None):
        """
        不给参数：按 TASK_CYCLE 出题
            TASK_CYCLE=True ：固定 7 条列表按轮循环，每轮打乱（采集默认）
            TASK_CYCLE=False：按 instruction_weights 随机抽类别再抽成员
        给了文本：解析出目标物体（用于回放数据集里记录的 task）。
        """
        if given is None:
            if self.TASK_CYCLE:
                if not self.task_queue:
                    self._refill_task_queue()
                targets = list(self.task_queue.pop(0))
            else:
                kinds = list(self.instruction_weights.keys())
                weights = [self.instruction_weights[k] for k in kinds]
                kind = self.task_rng.choices(kinds, weights=weights)[0]
                if kind == 'single':
                    targets = [self.task_rng.choice(self.OBJECTS)]
                elif kind == 'pair':
                    targets = self.task_rng.sample(list(self.OBJECTS), 2)
                elif kind == 'all':
                    targets = list(self.OBJECTS)
                else:
                    raise ValueError(f'未知的任务类型：{kind!r}')
            self.instruction = self.make_instruction(targets)
        else:
            self.instruction = given
            targets = self.parse_instruction(given)

        self.target_names = [f for f in self.OBJECTS if f in targets]
        self.target_objects = [self.BODY[f] for f in self.target_names]
        self.obj_target = self.target_objects[0]    # 兼容 SimpleEnv2 的字段名
        return self.instruction

    # ------------------------------------------------------------------ 成功判定
    def object_in_basket(self, name):
        """物体是否已经装进篮子：水平方向在篮子内圈里，且高度在篮子范围内。"""
        p = self.env.get_p_body(self.BODY[name])
        d_xy = float(np.linalg.norm(p[:2] - self.basket_xy))
        z_ok = (self.BASKET_FLOOR_Z - 0.05) < p[2] < (self.BASKET_RIM_Z + 0.02)
        return bool(d_xy < self.IN_BASKET_R) and bool(z_ok)

    def check_success(self):
        """
        任务完成 = 所有目标物体都进了篮子。
        require_retreat=True 时还要求夹爪已张开、机械臂抬到 0.9m 以上，
        这样 episode 会在"放完最后一样、松手退出"时才结束。
        """
        for name in self.target_names:
            if not self.object_in_basket(name):
                return False
        if self.require_retreat:
            if self.get_gripper_state() > 0.15:
                return False
            if self.env.get_p_body('tcp_link')[2] < 0.9:
                return False
        return True

    def object_states(self):
        """调试/写日志用：每样物体的位置、是否在篮子里、是不是本次目标。"""
        out = {}
        for name in self.OBJECTS:
            p = self.env.get_p_body(self.BODY[name])
            out[name] = {
                'p': np.round(p, 4).tolist(),
                'in_basket': self.object_in_basket(name),
                'is_target': name in self.target_names,
            }
        return out

    # ------------------------------------------------------------------ 物体位姿（用于数据集 obj_init）
    def get_obj_pose(self):
        """返回每个物体 [x, y, z, qw, qx, qy, qz]（位置 + 完整姿态四元数）。

        以前每物体只存 [x,y,z,yaw]，丢了 roll/pitch：
          - 苹果是球，搬运/落稳过程中会绕水平轴滚动；
          - 杯子也可能轻微倾斜。
        只按 yaw 复原时回放的初始姿态和录制不一致，抓取接触随之改变
        （实测出现过"录制时抓起来了、回放里杯子掉下去"）。
        """
        out = []
        for name in self.OBJECTS:
            p, R = self.env.get_pR_body(body_name=self.BODY[name])
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
            out.extend([float(p[0]), float(p[1]), float(p[2]),
                        float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        return np.asarray(out, dtype=np.float32)

    def set_obj_pose(self, obj_init):
        """按 [x, y, z, qw, qx, qy, qz] x N 复原物体位姿（回放数据集时用）。"""
        arr = np.asarray(obj_init, dtype=np.float64).reshape(len(self.OBJECTS), 7)
        for name, row in zip(self.OBJECTS, arr):
            x, y, z, qw, qx, qy, qz = [float(v) for v in row]
            self.env.set_p_base_body(body_name=self.BODY[name],
                                     p=np.array([x, y, z]), forward=False)
            q = np.array([qw, qx, qy, qz], dtype=np.float64)
            n = np.linalg.norm(q)
            q = q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            self.env.set_R_base_body(body_name=self.BODY[name], R=R.reshape(3, 3))
        self.env.forward(increase_tick=False)
        self.obj_init_pose = self.get_obj_pose()

    # ------------------------------------------------------------------ 渲染
    def render(self, teleop=False, idx=0):
        """和 SimpleEnv2.render 基本一致，多画了当前指令目标物体的小球标记。"""
        self.env.plot_time()

        # 末端标记
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        # FR3 的 TCP Z 轴沿手指伸出方向；不再沿用 OMY 的 Y/Z 轴交换。
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95, 0.05, 0.05, 0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2,
                              rgba=[0.05, 0.95, 0.05, 0.5])

        # 目标物体上方画一个小球，方便遥操作时确认"这次要拿哪几样"
        for name in getattr(self, 'target_names', []):
            p = self.env.get_p_body(self.BODY[name])
            self.env.plot_sphere(p=p + np.array([0.0, 0.0, 0.12]), r=0.012,
                                 rgba=[0.10, 0.60, 1.00, 0.9])

        # 数据集里的相机画面（agentview / egocentric / sideview）
        if getattr(self, 'rgb_ego', None) is not None:
            rgb_ego = add_title_to_img(self.rgb_ego, text='Egocentric View', shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_ego, loc='bottom right')
        if getattr(self, 'rgb_agent', None) is not None:
            rgb_agent = add_title_to_img(self.rgb_agent, text='Agent View', shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_agent, loc='top right')
        if teleop and getattr(self, 'rgb_side', None) is not None:
            rgb_side = add_title_to_img(self.rgb_side, text='Side View', shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_side, loc='top left')

        # 固定在窗口顶部中央，避免世界坐标中的字幕挡住夹爪/物体。
        self.env.viewer_text_overlay(loc='top', text1=f'Episode {idx}')

        if teleop:
            self.env.viewer_text_overlay(text1='Key Pressed',
                                         text2='%s' % (self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',
                                         text2='%s' % (self.env.get_key_repeated_list()))

        # 指令 + 进度
        progress = '  '.join(
            f'{name}:{"OK" if self.object_in_basket(name) else "--"}'
            for name in self.target_names)
        self.env.viewer_text_overlay(text1='Language Instructions', text2=self.instruction)
        self.env.viewer_text_overlay(text1='Progress', text2=progress)

        self.env.render()
