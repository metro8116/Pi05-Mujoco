"""Evaluate Pi05 on object positions outside the original spawn regions.

Reuse script 4's tasks, policy loading, success checks, and output layout.
Keep each object's original spawn center, but sample inside a larger square
while excluding the entire original square, including its boundary. The
default outer side is 0.15 m. Validate positions again after settling and
retry invalid resets while preserving the original spacing constraints.

Inputs: The deployment options from script 4 and an optional --outer-side.
Outputs: Evaluation records, videos, scores, and square-region metadata
    in a new, empty --out directory.
Usage: python 5.position_generation_test.py --policy-path PATH --offline
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUTER_SIDE = .15
spec = importlib.util.spec_from_file_location('position_deploy_helpers', HERE / '4.deploy_pi05.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def square_regions(base_class, outer_side):
    """Read original regions from script-4 environment; keep their centers."""
    if not math.isfinite(outer_side) or outer_side <= 0:
        raise ValueError('--outer-side 必须为正的有限数值（单位：米）')
    if len(base_class.OBJECTS) != 2 or set(base_class.OBJECTS) != {'apple', 'mug_green'}:
        raise ValueError('位置泛化测试要求场景物体为 apple 和 mug_green')
    if len(base_class.SPAWN_SPOTS) != len(base_class.OBJECTS):
        raise ValueError('OBJECTS 与 SPAWN_SPOTS 数量必须一致')
    regions = {}
    for name, ((xl, xh), (yl, yh)) in zip(base_class.OBJECTS, base_class.SPAWN_SPOTS):
        if not all(math.isfinite(v) for v in (xl, xh, yl, yh)) or xl >= xh or yl >= yh:
            raise ValueError(f'{name} 的原始出生区域无效')
        if outer_side <= max(xh-xl, yh-yl):
            raise ValueError(f'--outer-side 必须大于 {name} 原区域的边长 {max(xh-xl, yh-yl):.6f} 米')
        cx, cy = (xl+xh)/2, (yl+yh)/2
        half = outer_side/2
        regions[name] = {'center_xy': [cx, cy],
                         'old_rectangle_xy': [[xl, xh], [yl, yh]],
                         'outer_rectangle_xy': [[cx-half, cx+half], [cy-half, cy+half]],
                         'outer_side_m': outer_side}
    return regions


def in_square_ring(xy, region):
    """Include the new boundary; exclude the complete old square AND its boundary."""
    xy = np.asarray(xy, dtype=float)[:2]
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return False
    old, outer = region['old_rectangle_xy'], region['outer_rectangle_xy']
    inside_outer = all(lo <= value <= hi for value, (lo, hi) in zip(xy, outer))
    inside_old = all(lo <= value <= hi for value, (lo, hi) in zip(xy, old))
    return bool(inside_outer and not inside_old)


def square_ring_environment_class(base_class, outer_side=DEFAULT_OUTER_SIDE):
    regions = square_regions(base_class, outer_side)

    class SquareRingPickPlaceEnv(base_class):
        POSITION_REGIONS = regions
        POSITION_RESET_TRIES = 10

        def _sample_spawn_points(self):
            # Uniform outer-square proposals, rejecting old-square points and
            # the original spacing violations. Never fall back to old centers.
            for _ in range(self.MAX_SPAWN_TRIES):
                candidates = {}
                for name in self.OBJECTS:
                    bounds = self.POSITION_REGIONS[name]['outer_rectangle_xy']
                    xy = [np.random.uniform(lo, hi) for lo, hi in bounds]
                    candidates[name] = np.r_[xy, self.TABLE_TOP_Z + self.DROP_H]
                if (all(in_square_ring(candidates[n], self.POSITION_REGIONS[n]) for n in self.OBJECTS)
                        and self._pairwise_ok(candidates)):
                    return [candidates[n] for n in self.OBJECTS], 'square_ring'
            raise RuntimeError('方形外环采样无法满足原间距约束；请调整 --outer-side；未回退到旧区域')

        def reset(self, seed=None):
            # A sampled object can roll back into the original training region.
            # Repeat the normal reset/settling flow until final centers qualify.
            for attempt in range(self.POSITION_RESET_TRIES):
                super().reset(seed=seed if attempt == 0 else None)
                positions = {n: np.asarray(self.env.get_p_body(self.BODY[n]), dtype=float)
                             for n in self.OBJECTS}
                if (all(in_square_ring(positions[n], self.POSITION_REGIONS[n]) for n in self.OBJECTS)
                        and self._pairwise_ok(positions)):
                    self.position_reset_attempts = attempt + 1
                    return
            raise RuntimeError('物体落稳后离开指定方形外环或违反间距，已停止以避免混入旧区域样本')

    return SquareRingPickPlaceEnv


def create_environment(*args, outer_side=DEFAULT_OUTER_SIDE, **kwargs):
    # Import MuJoCo only after the shared runner sets the rendering backend.
    from mujoco_env.pick_place_env import PickPlaceEnv
    cls = square_ring_environment_class(PickPlaceEnv, outer_side)
    print('位置泛化评测：保持原中心，仅在大正方形减去原正方形的外环生成，不交换物体。')
    for name, region in cls.POSITION_REGIONS.items():
        print(f"  {name}: center={region['center_xy']}, "
              f"outer side={100*region['outer_side_m']:.1f} cm, "
              f"exclude={region['old_rectangle_xy']}")
    return cls(*args, **kwargs)


def parser():
    ap = deploy.parser()
    ap.description = __doc__
    ap.set_defaults(out=str(HERE / 'outputs/position_generation_test'))
    ap.add_argument('--outer-side', type=float, default=DEFAULT_OUTER_SIDE,
                    help='新正方形边长（米），默认0.15；必须大于旧区域边长0.10米')
    return ap


def main():
    args = parser().parse_args()
    if not math.isfinite(args.outer_side) or args.outer_side <= 0:
        raise ValueError('--outer-side 必须为正的有限数值（单位：米）')
    output = Path(args.out).resolve()
    if not args.preflight and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'评测目录已存在且非空: {output}；请使用新的 --out，以保留旧评测结果')
    metadata = {
        'name': 'square_ring', 'outer_side_m': args.outer_side,
        'rule': 'Keep original centers; sample inside the larger square and strictly outside the old square.',
        'sampling': 'Uniform outer-square proposals; reject old-square positions and original spacing violations.',
        'settling': 'Normal placement and settling; resample if final centers leave the square rings or violate spacing.',
    }

    def factory(*env_args, **env_kwargs):
        env = create_environment(*env_args, outer_side=args.outer_side, **env_kwargs)
        metadata['regions'] = env.POSITION_REGIONS
        return env

    deploy.main(args, env_factory=factory, evaluation_metadata=metadata)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit(f'错误: {error}')
