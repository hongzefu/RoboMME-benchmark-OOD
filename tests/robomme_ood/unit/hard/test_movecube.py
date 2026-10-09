"""MoveCube 新值档（V9 只交付 xhard4）：演示段与执行段两套布局的方块／目标／杆位、移动方式，包内规格回放与自导出。

期望：区域参数取自规格里记录的 ``region``（与 header decision 的区域一致性另断言）；方块到目标距离、方块到
机械臂基座距离用实际 actor 位姿手算。
"""
from __future__ import annotations

import numpy as np
import pytest

from . import cells as C
from . import offline_scene as O

TASK = "MoveCube"


def _decision(tier):
    header, _ = O.delivered_rows(TASK, tier, 0)
    return header["sampling_config"][TASK]["decision"]


def _xy(actor):
    return actor.pose.p[0, :2].numpy().astype(np.float64)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
def test_tampered_spec_is_detected(tier):
    C.check_tamper_detected(TASK, tier)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_two_layouts_regions_and_way(tier, k):
    row, env = C.replayed(TASK, tier, k)
    dec = _decision(tier)
    layout = row["spec"]["layout"]
    # 执行段方块在演示期间停放在场外，按生产记下的执行段初始位姿（cube_init_pose_2／goal_site_2_pose_p）核对
    pairs = {"demo": (_xy(env.cube), _xy(env.goal_site)),
             "execution": (env.cube_init_pose_2.p[0, :2].numpy().astype(np.float64),
                           np.asarray(env.goal_site_2_pose_p, dtype=np.float64).reshape(-1)[:2])}
    for seg, (cube, goal) in pairs.items():
        region = layout[seg]["region"]
        declared = dec[f"{seg}_layout"][tier]["region"]
        # 规格记录的区域 ⊇ header 声明的区域（声明的每个键原样出现）
        assert all(region[key] == value for key, value in declared.items())
        # 实际 actor 位姿就是规格里冻结的方块位置与目标位置
        assert np.allclose(cube, layout[seg]["cube_pose"][:2], atol=1e-6)
        assert np.allclose(goal, layout[seg]["goal_xy"], atol=1e-6)
        # 手算：方块离目标不小于下限；方块离机械臂基座落在基座距离区间内
        assert np.linalg.norm(cube - goal) >= region["min_cube_goal_m"] - 1e-6
        base = np.asarray(region["robot_base_xy"], dtype=np.float64)
        lo, hi = region["base_dist"]
        assert lo - 1e-6 <= np.linalg.norm(cube - base) <= hi + 1e-6
    # 移动方式：三种之一，且来自生产的方式表
    assert env.way in env.ways and len(set(env.ways)) == 3
