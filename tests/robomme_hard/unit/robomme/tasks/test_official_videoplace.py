"""VideoPlaceButton 与 VideoPlaceOrder 原生三档真值表（C05、C07 语言绑定）。

演示段由下面的通用驱动按每个演示子任务的含义真实执行（拾起 → 放到该子任务的 segment 目标 → 按钮 → 放回桌面
→ 静止 → 交换 → 复位），驱动同时记下「演示里依次放过的目标」。期望：
- VideoPlaceButton：目标语言写 before → 正确目标是按按钮前最后放的那个；after → 按按钮后第一个放的那个；
- VideoPlaceOrder：目标语言写「第 k 个」→ 正确目标是演示里第 k 次放的目标（按时间序，不按空间序，按钮不占序号）；
- 放到其他目标、拿错方块 → 失败；hard 档目标盘被交换后按身份判定。
"""
from __future__ import annotations

import numpy as np
import pytest

from _official_world import OfficialWorld, find_seed, goal_text
from tests.robomme_hard.unit.robomme import official_thresholds as T

DIFFS = ("easy", "medium", "hard")
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def drive_demo(ep):
    """逐个演示子任务真实推进；返回 (演示里按时间顺序放过的目标列表, 按钮插在第几次放置之后)。"""
    env = ep.env
    cube = env.target_cube
    placed, button_after = [], None
    guard = 0
    while ep.task_index < ep.first_online_index():
        entry = env.task_list[ep.task_index]
        name = entry["name"]
        before = ep.task_index
        if name == "pick up the cube":
            ep.grasp(cube)
            ep.step()
        elif name == "drop the cube onto target":
            ep.place_on(cube, entry["segment"])
            ep.step()
            placed.append(entry["segment"])
        elif name == "drop the cube onto table":
            ep.place_on(cube, env.goal_site)
            ep.step()
        elif name == "press the button":
            ep.press(env.button)
            ep.step()
            ep.unpress(env.button)
            button_after = len(placed)
        else:  # static／NO RECORD：机器人静止、关节在复位位
            ep.step()
        guard += 1
        assert guard < 1000
        if name != "static" and name != "NO RECORD":
            assert ep.task_index == before + 1, f"演示子任务 {name} 未推进"
    ep.step()  # 越过交换在下一步开头的最终定位
    return placed, button_after


# --------------------------------------------------------------------------- VideoPlaceButton


@pytest.fixture
def vpb():
    with OfficialWorld("VideoPlaceButton") as w:
        yield w


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("lang", ["before", "after"])
def test_vpb_before_after_target(vpb, diff, lang):
    seed = find_seed("VideoPlaceButton", diff, lambda e: e.target_target_language == lang)
    ep = vpb.make(diff, seed=seed)
    env = ep.env
    assert all(lang in g for g in goal_text(env)[:2])
    placed, button_after = drive_demo(ep)
    expected = placed[button_after - 1] if lang == "before" else placed[button_after]
    assert env.target_target is expected
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, expected)
    ep.step()
    assert ep.success and not ep.fail


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("lang", ["before", "after"])
def test_vpb_reversed_target_fails(vpb, diff, lang):
    seed = find_seed("VideoPlaceButton", diff, lambda e: e.target_target_language == lang)
    ep = vpb.make(diff, seed=seed)
    env = ep.env
    placed, button_after = drive_demo(ep)
    wrong = placed[button_after] if lang == "before" else placed[button_after - 1]
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, wrong)
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", ("medium", "hard"))
def test_vpb_wrong_cube_fails(vpb, diff):
    ep = vpb.make(diff, seed=0)
    env = ep.env
    drive_demo(ep)
    ep.grasp(env.non_target_cubes[0])
    ep.step()
    assert ep.fail and not ep.success


def test_vpb_native_configs_have_no_extra_demo_placements(vpb):
    for diff in DIFFS:
        env = vpb.make(diff, seed=0).env
        assert (env.pre_flag, env.post_flag) == (0, 0)
        assert (env.swap_target_a is not None) is (diff == "hard")


def test_vpb_hard_swap_follows_identity():
    """hard：被交换的目标盘按身份判定——放到它原来的位置（现在是另一个盘）失败，放到它现在的位置成功。"""
    seed = find_seed("VideoPlaceButton", "hard",
                     lambda e: e.target_target is e.swap_target_a or e.target_target is e.swap_target_b)
    for put_on_original, ok in ((True, False), (False, True)):
        with OfficialWorld("VideoPlaceButton") as world:
            ep = world.make("hard", seed=seed)
            env = ep.env
            origin = env.target_target.xyz[:2].copy()
            drive_demo(ep)
            assert np.linalg.norm(env.target_target.xyz[:2] - origin) > T.DROP_ONTO_XY  # 原位置已不在它的放置半径内
            ep.grasp(env.target_cube)
            ep.step()
            if put_on_original:
                ep.carry(env.target_cube, *origin)
                ep.release(env.target_cube, *origin)
            else:
                ep.place_on(env.target_cube, env.target_target)
            ep.step()
            assert ep.success is ok and ep.fail is (not ok)


# --------------------------------------------------------------------------- VideoPlaceOrder


@pytest.fixture
def vpo():
    with OfficialWorld("VideoPlaceOrder") as w:
        yield w


def _vpo_case(world, diff, predicate):
    seed = find_seed("VideoPlaceOrder", diff, predicate)
    ep = world.make(diff, seed=seed)
    return ep, ep.env


@pytest.mark.parametrize("diff", DIFFS)
@pytest.mark.parametrize("which", ["first", "last"])
def test_vpo_kth_temporal_target_succeeds(vpo, diff, which):
    pred = (lambda e: e.which_in_subset == 1) if which == "first" else (
        lambda e: e.which_in_subset == len(e.which_targets_to_pick) and e.which_in_subset > 1)
    ep, env = _vpo_case(vpo, diff, pred)
    k = env.which_in_subset
    assert all(f"the {ORDINALS[k]} target" in g for g in goal_text(env))
    placed, _ = drive_demo(ep)
    assert env.target_target is placed[k - 1]
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, placed[k - 1])
    ep.step()
    assert ep.success and not ep.fail


def test_vpo_button_does_not_take_an_ordinal(vpo):
    """按钮插在第 k 次放置之前时，序号仍只数放置。"""
    ep, env = _vpo_case(vpo, "medium", lambda e: 0 < e.button_task_index // 2 < e.which_in_subset)
    placed, button_after = drive_demo(ep)
    assert 0 < button_after < env.which_in_subset
    assert env.target_target is placed[env.which_in_subset - 1]


def test_vpo_spatial_order_is_not_temporal_order(vpo):
    """按 y 坐标排第 k 的目标 ≠ 演示里第 k 次放的目标时，放到「空间第 k」→ 失败。"""
    for seed in range(64):
        ep = vpo.make("medium", seed=seed)
        env = ep.env
        k = env.which_in_subset
        spatial = sorted(env.which_targets_to_pick, key=lambda t: float(t.xyz[1]))
        if spatial[k - 1] is not env.which_targets_to_pick[k - 1]:
            break
    else:
        pytest.fail("找不到空间序与时间序不同的布局")
    placed, _ = drive_demo(ep)
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, spatial[k - 1])
    ep.step()
    assert ep.fail and not ep.success


@pytest.mark.parametrize("diff", DIFFS)
def test_vpo_other_target_fails(vpo, diff):
    ep = vpo.make(diff, seed=0)
    env = ep.env
    drive_demo(ep)
    other = next(t for t in env.targets if t is not env.target_target)
    ep.grasp(env.target_cube)
    ep.step()
    ep.place_on(env.target_cube, other)
    ep.step()
    assert ep.fail and not ep.success


def test_vpo_demo_drop_tasks_bind_their_own_target(vpo):
    """演示里每个「放到目标」子任务绑定自己的目标（闭包按值捕获）：先放到最后一个目标不推进第一个放置子任务。"""
    ep, env = _vpo_case(vpo, "easy", lambda e: len(e.which_targets_to_pick) >= 2 and e.button_task_index != 0)
    ep.grasp(env.target_cube)
    ep.step()
    idx = ep.task_index
    ep.place_on(env.target_cube, env.which_targets_to_pick[-1])
    ep.step()
    assert ep.task_index == idx
    ep.place_on(env.target_cube, env.which_targets_to_pick[0])
    ep.step()
    assert ep.task_index == idx + 1
