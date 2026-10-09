"""VPB / VPO 共用的演示落点与策略校验工具。

只在 xhard 分支被调用，原三档一次都不会走到这里。

* :func:`validate_demo_plan`：核对 ``decision`` 里 ``demo_object_count`` / ``demo_return_policy`` 的取值组合。
  两个键由环境文件自己读出（审计 ``config-map`` 按环境源码找消费点），这里只做合法性判定。
* :func:`build_home_sites`：在每个演示方块的**初始位姿**上直接调 target builder 建一个落点 actor。
* V6 `xhard1`～`xhard4` 只允许 `return_to_origin`；旧 `xhard` 规格继续支持原有放回路径。
* :func:`returned_mask` / :func:`build_goal_drop_sites` 保留副本历史策略的落点实现，供已存在的规格回放；
  不会被 V6 新值决策启用。
  不放回的方块若留在按钮后的目标台上，执行段会直接看到答案、还会与后续方块或 swap 撞台，
  所以一律挪到 goal_site 区域：离 goal_site 中心最近、避开全部方块 / 目标台 / 按钮的网格点，确定、不抽随机数。

为什么不用 ``spawn_random_target(randomize=False)``：该形参在采样循环里根本没被读，照样
``torch.rand``（会平移随机流、落点也不在指定位置）；而 BinFill 原三档正在传 ``randomize=False``，
修这个工具函数会平移 BinFill 原三档 ⇒ 按红线 N12 不就地修，另写本 xhard 专用路径。
"""

from __future__ import annotations

import torch
from mani_skill.utils.structs.pose import Pose

from .object_generation import build_gray_white_target
from .SceneGenerationError import SceneGenerationError
from . import difficulty as difficulty_utils

# 原三档的取值：只演示 1 个方块，演示完放到随机 goal_site（z 压到桌面以下隐藏）
NATIVE_DEMO_PLAN = (1, "native_random_goal_site")
# xhard 的返回策略：每个演示方块放回自己的初始位置（V5 xhard 唯一取值，逐位不变）
RETURN_TO_ORIGIN = "return_to_origin"
# V6 计划 2.10 新增：只把最后一个演示方块放回原位，其余方块落到 goal_site 区域
RETURN_LAST_ONLY = "return_last_only"
# V6 新值机制下的「不放回」：沿用原三档的策略名，语义同原三档（演示完放到隐藏 goal_site 所在处）
NO_RETURN = NATIVE_DEMO_PLAN[1]
NEWVALUE_RETURN_POLICIES = (RETURN_TO_ORIGIN, RETURN_LAST_ONLY, NO_RETURN)

# 落点 actor 的尺寸只影响（被隐藏的）可视外观，不参与任何碰撞或判定：
# is_obj_dropped_onto 只看水平距离 ≤ 0.05，solve_putonto_whenhold 只取 pose.p。
HOME_SITE_THICKNESS = 0.005


def validate_demo_plan(count, policy, difficulty: str, n_cubes: int) -> tuple[int, str]:
    """核对演示方块数与返回策略；非法组合直接抛 ``SceneGenerationError``（不静默截断）。"""
    count = int(count)
    policy = str(policy)
    is_newvalue = getattr(difficulty_utils, "is_newvalue_difficulty", None)
    is_v6_tier = (bool(is_newvalue(difficulty)) if is_newvalue is not None else
                  isinstance(difficulty, str) and difficulty.strip().lower()
                  in {"xhard1", "xhard2", "xhard3", "xhard4"})
    if is_v6_tier:
        if policy != RETURN_TO_ORIGIN:
            raise SceneGenerationError(
                f"V6 新值档只支持 demo_return_policy={RETURN_TO_ORIGIN!r}，收到 {policy!r}"
            )
        if not 1 <= count <= n_cubes:
            raise SceneGenerationError(
                f"V6 demo_object_count={count} 超出场上方块数 {n_cubes}（请求数 ≠ 可演示数）"
            )
        return count, policy
    if difficulty == "xhard":
        # V5 规格仍使用旧档名；兼容其既有 return_to_origin 路径及副本中的休眠策略。
        if policy not in NEWVALUE_RETURN_POLICIES:
            raise SceneGenerationError(
                f"xhard 只支持 demo_return_policy ∈ {NEWVALUE_RETURN_POLICIES}，收到 {policy!r}"
            )
        if not 1 <= count <= n_cubes:
            raise SceneGenerationError(
                f"xhard demo_object_count={count} 超出场上方块数 {n_cubes}（请求数 ≠ 可演示数）"
            )
        return count, policy
    else:
        if (count, policy) != NATIVE_DEMO_PLAN:
            raise SceneGenerationError(
                f"原三档只支持 demo_object_count=1 + native_random_goal_site，收到 {(count, policy)}"
            )
        return count, policy


def validate_place_sequence(steps, n_targets: int) -> dict:
    """V6 审查修复 F6 守卫：回放完整放置序列的台占用，冲突即抛 ``SceneGenerationError``。

    ``steps``：``[(cube_id, target_id), ...]``，按演示时间顺序，含正式放置与额外放置（放回原位不是台，不在其中）。
    规则：每一步把 ``cube_id`` 从它当前所在台移走、放到 ``target_id``；若 ``target_id`` 此刻被**其他**方块占着即
    「两块同台」（审查 F6/D8 的 xhard4 seed 7000500 反例），若被**自己**占着即「原地空转」（审查 N2 的 4 台反例），
    两者都判本局生成失败而不是生成冲突轨迹。返回终态占用表 ``{target_id: cube_id | None}`` 供调用方记录。
    """
    if int(n_targets) <= 0:
        raise SceneGenerationError(f"validate_place_sequence: 台数必须为正，收到 {n_targets}")
    occupancy: dict[int, int | None] = {i: None for i in range(int(n_targets))}
    location: dict[int, int] = {}
    for step_index, (cube_id, target_id) in enumerate(steps):
        cube_id, target_id = int(cube_id), int(target_id)
        if target_id not in occupancy:
            raise SceneGenerationError(f"放置序列第 {step_index} 步：台 {target_id} 越界（台数 {n_targets}）")
        holder = occupancy[target_id]
        if holder is not None and holder != cube_id:
            raise SceneGenerationError(
                f"放置序列第 {step_index} 步：方块 {cube_id} 要放到台 {target_id}，但该台仍被方块 {holder} 占着（两块同台）"
            )
        if holder == cube_id:
            raise SceneGenerationError(
                f"放置序列第 {step_index} 步：方块 {cube_id} 已在台 {target_id} 上，再次放到同一台是原地空转"
            )
        if cube_id in location:
            occupancy[location[cube_id]] = None
        occupancy[target_id] = cube_id
        location[cube_id] = target_id
    return occupancy


def build_home_sites(env, cubes, generator, name_prefix: str = "home_site"):
    """在每个方块的初始位姿上建一个隐藏的落点 actor，返回 ``(homes, checks)``。

    必须放在本场景**所有**其他 spawn 之后调用。两条验收（计划 2.14）当场自检，不过即抛错：

    1. 落点位姿逐位等于方块初始位姿（``raw_pose`` 7 个 float32 全等）；
    2. 建 actor 前后 ``generator.get_state()`` 逐字节相等（builder 不抽随机数）。

    落点登记进 ``env._hidden_objects``：传感器画面（录像器用的 base/hand camera）里看不到，
    与原三档把 goal_site 压到桌面以下同理——「演示完放哪」只体现在动作上，不在画面上多出标记。
    """
    state_before = generator.get_state().clone()
    homes = []
    pose_equal = []
    for cube in cubes:
        raw = cube.initial_pose.raw_pose.detach().clone()
        home = build_gray_white_target(
            scene=env.scene,
            radius=float(env.cube_half_size),
            thickness=HOME_SITE_THICKNESS,
            name=f"{name_prefix}_{cube.name}",
            body_type="kinematic",
            add_collision=False,
            initial_pose=Pose.create(raw),
        )
        home._home_of = cube
        same = torch.equal(home.initial_pose.raw_pose.detach().cpu(), raw.cpu())
        if same and not env.scene.gpu_sim_enabled:
            # CPU 仿真下还能直接读实体位姿；GPU 仿真在 _load_scene 阶段尚未初始化，只比 initial_pose
            same = torch.equal(home.pose.raw_pose.detach().cpu(), raw.cpu())
        if not same:
            raise SceneGenerationError(f"落点 {home.name} 的位姿与方块 {cube.name} 的初始位姿不逐位相等")
        pose_equal.append(same)
        env._hidden_objects.append(home)
        homes.append(home)
    rng_equal = torch.equal(state_before, generator.get_state())
    if not rng_equal:
        raise SceneGenerationError("建落点 actor 前后 generator 状态不一致（不许抽随机数）")
    checks = {"pose_equal": pose_equal, "rng_state_equal": rng_equal}
    return homes, checks


def home_pose_record(cubes, homes) -> dict:
    """``actions.return_pose_by_object_id`` 的内容：方块名 → 落点 raw_pose（p 三位 + q 四位）。"""
    return {
        cube.name: [float(v) for v in home.initial_pose.raw_pose[0].tolist()]
        for cube, home in zip(cubes, homes)
    }


def returned_mask(policy: str, count: int) -> list[bool]:
    """每个演示方块（按演示顺序）是否放回原位；``policy`` 须已过 :func:`validate_demo_plan`。"""
    if policy == RETURN_TO_ORIGIN:
        return [True] * count
    if policy == RETURN_LAST_ONLY:
        return [k == count - 1 for k in range(count)]
    if policy == NO_RETURN:
        return [False] * count
    raise SceneGenerationError(f"未知 demo_return_policy={policy!r}")


# 不放回落点的避让半径（中心距下限，米）。方块半边长 0.02、对角半宽约 0.028：
# 方块-方块 ≥ 0.07（两块对角相对也留 1.4 cm）；方块-目标台（半径 0.04）≥ 0.08；方块-按钮底座 ≥ 0.10。
GOAL_DROP_CLEARANCE = {"cube": 0.07, "target": 0.08, "button": 0.10, "drop": 0.07}
# 候选网格：goal_site 中心周围 ±GOAL_DROP_SEARCH_HALF 的 1 cm 网格，按离中心距离、再按 (x, y) 排序（确定、不抽随机数）
GOAL_DROP_SEARCH_HALF = 0.12
GOAL_DROP_GRID_STEP = 0.01


def goal_drop_candidates(center_xy, half: float = GOAL_DROP_SEARCH_HALF, step: float = GOAL_DROP_GRID_STEP):
    """goal_site 中心周围的候选落点，离中心由近到远；距离相同按 (x, y) 字典序，保证逐位确定。"""
    n = int(round(half / step))
    cx, cy = float(center_xy[0]), float(center_xy[1])
    pts = [(round(cx + i * step, 6), round(cy + j * step, 6)) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    return sorted(pts, key=lambda q: (round((q[0] - cx) ** 2 + (q[1] - cy) ** 2, 10), q[0], q[1]))


def plan_goal_drop_xy(center_xy, count: int, obstacles) -> list[tuple[float, float]]:
    """贪心选 ``count`` 个落点：每个取离 goal_site 中心最近、且与全部障碍及已选落点都够远的候选。

    ``obstacles``：``[(kind, (x, y)), ...]``，kind ∈ GOAL_DROP_CLEARANCE（cube / target / button）。
    方块障碍取**全部**方块的初始位置（保守：未演示的、尚未轮到的、放回原位的都停在那里）。
    找不到时抛 ``SceneGenerationError``（可重试的任务性失败），不静默放宽。
    """
    chosen: list[tuple[float, float]] = []
    for _ in range(count):
        for q in goal_drop_candidates(center_xy):
            blocked = any(
                (q[0] - x) ** 2 + (q[1] - y) ** 2 < GOAL_DROP_CLEARANCE[kind] ** 2 for kind, (x, y) in obstacles
            ) or any((q[0] - x) ** 2 + (q[1] - y) ** 2 < GOAL_DROP_CLEARANCE["drop"] ** 2 for x, y in chosen)
            if not blocked:
                chosen.append(q)
                break
        else:
            raise SceneGenerationError(f"goal_site 周围找不到第 {len(chosen) + 1} 个无碰撞的不放回落点")
    return chosen


def build_goal_drop_sites(env, cubes, goal_site, generator, obstacles, name_prefix: str = "goal_drop"):
    """给不放回的演示方块在 goal_site 区域建隐藏落点 actor，返回 ``(sites, checks)``。

    落点定义（V6 计划 2.10 / 待决 M11 的实施方案）：以 goal_site 中心为原点，按 :func:`plan_goal_drop_xy`
    贪心选离中心最近的无碰撞网格点（中心本身空着时 1 块恰落在中心，与原三档同处）；按演示顺序依次选。
    与 :func:`build_home_sites` 同一套约束：不抽随机数（前后 ``generator`` 状态逐字节相等，否则抛错）、
    登记进 ``env._hidden_objects``、``body_type=kinematic`` 无碰撞。落点 z 取方块半边长，朝向单位四元数。
    goal_site 本身在 ``_initialize_episode`` 里被压到桌面以下，这里只借它的 xy。
    """
    state_before = generator.get_state().clone()
    center = goal_site.initial_pose.raw_pose.detach().cpu()[0]
    xys = plan_goal_drop_xy((float(center[0]), float(center[1])), len(cubes), obstacles)
    z = float(env.cube_half_size)
    sites = []
    for cube, (x, y) in zip(cubes, xys):
        raw = torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        site = build_gray_white_target(
            scene=env.scene,
            radius=float(env.cube_half_size),
            thickness=HOME_SITE_THICKNESS,
            name=f"{name_prefix}_{cube.name}",
            body_type="kinematic",
            add_collision=False,
            initial_pose=Pose.create(raw),
        )
        site._goal_drop_of = cube
        env._hidden_objects.append(site)
        sites.append(site)
    rng_equal = torch.equal(state_before, generator.get_state())
    if not rng_equal:
        raise SceneGenerationError("建 goal_site 落点 actor 前后 generator 状态不一致（不许抽随机数）")
    return sites, {"rng_state_equal": rng_equal, "count": len(sites)}


def goal_drop_obstacles(env, button_xy) -> list:
    """不放回落点要避让的障碍：全部方块初始位置、全部目标台、按钮。只读 initial_pose，不抽随机数。"""
    xy = lambda actor: tuple(float(v) for v in actor.initial_pose.raw_pose.detach().cpu()[0, :2].tolist())  # noqa: E731
    obstacles = [("cube", xy(c)) for c in env.all_cubes]
    obstacles += [("target", xy(t)) for t in env.targets]
    obstacles.append(("button", (float(button_xy[0]), float(button_xy[1]))))
    return obstacles
