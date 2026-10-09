"""Demo drop-site and policy validation tools shared by VPB / VPO.

Only called from xhard branches; the original three tiers never reach here.

* :func:`validate_demo_plan`: checks the value combination of ``demo_object_count`` / ``demo_return_policy`` in ``decision``.
  Both keys are read by the env file itself (the audit ``config-map`` finds consumption points in env source); only validity is judged here.
* :func:`build_home_sites`: builds a drop-site actor directly at the **initial pose** of each demo cube via the target builder.
* V6 `xhard1`..`xhard4` only allow `return_to_origin`; old `xhard` specs still support the original put-back path.
* :func:`returned_mask` / :func:`build_goal_drop_sites` keep the drop-site implementation of the replica's historical policies for replaying existing specs;
  they are never enabled by V6 new-value decisions.
  A cube that is not put back and stays on a target stand after the button would show the answer directly in the execution segment and could collide with later cubes or swaps on that stand,
  so it is always moved to the goal_site region: the grid point closest to the goal_site center that avoids all cubes / target stands / the button; deterministic, no random draws.

Why not ``spawn_random_target(randomize=False)``: that parameter is never read in the sampling loop and it still calls
``torch.rand`` (shifting the random stream, and the drop site is not at the given position); BinFill's original three tiers do pass ``randomize=False``,
so fixing that utility would shift BinFill's original three tiers ⇒ per red line N12 it is not fixed in place; this separate xhard-only path is written instead.
"""

from __future__ import annotations

import torch
from mani_skill.utils.structs.pose import Pose

from .object_generation import build_gray_white_target
from .SceneGenerationError import SceneGenerationError
from . import difficulty as difficulty_utils

# Values of the original three tiers: demo only 1 cube, then place it on a random goal_site (z pushed below the table to hide it)
NATIVE_DEMO_PLAN = (1, "native_random_goal_site")
# xhard return policy: each demo cube goes back to its own initial position (the only V5 xhard value, bitwise unchanged)
RETURN_TO_ORIGIN = "return_to_origin"
# Added in V6 plan 2.10: only the last demo cube goes back to its original position; the others drop into the goal_site region
RETURN_LAST_ONLY = "return_last_only"
# "Do not put back" under the V6 new-value mechanism: reuses the policy name of the original three tiers, same semantics (after the demo, place where the hidden goal_site is)
NO_RETURN = NATIVE_DEMO_PLAN[1]
NEWVALUE_RETURN_POLICIES = (RETURN_TO_ORIGIN, RETURN_LAST_ONLY, NO_RETURN)

# The size of the drop-site actor only affects the (hidden) visual appearance and takes no part in any collision or check:
# is_obj_dropped_onto only looks at horizontal distance <= 0.05, solve_putonto_whenhold only uses pose.p.
HOME_SITE_THICKNESS = 0.005


def validate_demo_plan(count, policy, difficulty: str, n_cubes: int) -> tuple[int, str]:
    """Check demo cube count and return policy; an invalid combination raises ``SceneGenerationError`` directly (no silent truncation)."""
    count = int(count)
    policy = str(policy)
    is_newvalue = getattr(difficulty_utils, "is_newvalue_difficulty", None)
    is_v6_tier = (bool(is_newvalue(difficulty)) if is_newvalue is not None else
                  isinstance(difficulty, str) and difficulty.strip().lower()
                  in {"xhard1", "xhard2", "xhard3", "xhard4"})
    if is_v6_tier:
        if policy != RETURN_TO_ORIGIN:
            raise SceneGenerationError(
                f"V6 new-value tiers only support demo_return_policy={RETURN_TO_ORIGIN!r}, got {policy!r}"
            )
        if not 1 <= count <= n_cubes:
            raise SceneGenerationError(
                f"V6 demo_object_count={count} exceeds the number of cubes in the scene {n_cubes} (requested != demonstrable)"
            )
        return count, policy
    if difficulty == "xhard":
        # V5 specs still use the old tier name; keep compatibility with their existing return_to_origin path and the dormant policies in the replica.
        if policy not in NEWVALUE_RETURN_POLICIES:
            raise SceneGenerationError(
                f"xhard only supports demo_return_policy in {NEWVALUE_RETURN_POLICIES}, got {policy!r}"
            )
        if not 1 <= count <= n_cubes:
            raise SceneGenerationError(
                f"xhard demo_object_count={count} exceeds the number of cubes in the scene {n_cubes} (requested != demonstrable)"
            )
        return count, policy
    else:
        if (count, policy) != NATIVE_DEMO_PLAN:
            raise SceneGenerationError(
                f"the original three tiers only support demo_object_count=1 + native_random_goal_site, got {(count, policy)}"
            )
        return count, policy


def validate_place_sequence(steps, n_targets: int) -> dict:
    """V6 review fix F6 guard: replay the stand occupancy of the full placement sequence; raise ``SceneGenerationError`` on conflict.

    ``steps``: ``[(cube_id, target_id), ...]`` in demo time order, including formal and extra placements (putting back to the original position is not a stand and is excluded).
    Rule: each step moves ``cube_id`` off its current stand and places it on ``target_id``; if ``target_id`` is currently occupied by **another** cube, that is
    "two cubes on one stand" (review F6/D8 xhard4 seed 7000500 counterexample); if occupied by **itself**, that is "idle in place" (review N2 4-stand counterexample);
    both fail this episode's generation instead of generating a conflicting trajectory. Returns the final occupancy table ``{target_id: cube_id | None}`` for the caller to record.
    """
    if int(n_targets) <= 0:
        raise SceneGenerationError(f"validate_place_sequence: number of stands must be positive, got {n_targets}")
    occupancy: dict[int, int | None] = {i: None for i in range(int(n_targets))}
    location: dict[int, int] = {}
    for step_index, (cube_id, target_id) in enumerate(steps):
        cube_id, target_id = int(cube_id), int(target_id)
        if target_id not in occupancy:
            raise SceneGenerationError(f"placement sequence step {step_index}: stand {target_id} out of range (number of stands {n_targets})")
        holder = occupancy[target_id]
        if holder is not None and holder != cube_id:
            raise SceneGenerationError(
                f"placement sequence step {step_index}: cube {cube_id} is to be placed on stand {target_id}, but that stand is still occupied by cube {holder} (two cubes on one stand)"
            )
        if holder == cube_id:
            raise SceneGenerationError(
                f"placement sequence step {step_index}: cube {cube_id} is already on stand {target_id}; placing it on the same stand again is idle in place"
            )
        if cube_id in location:
            occupancy[location[cube_id]] = None
        occupancy[target_id] = cube_id
        location[cube_id] = target_id
    return occupancy


def build_home_sites(env, cubes, generator, name_prefix: str = "home_site"):
    """Build a hidden drop-site actor at each cube's initial pose; returns ``(homes, checks)``.

    Must be called after **all** other spawns of this scene. Two acceptance checks (plan 2.14) are self-checked on the spot; failure raises:

    1. the drop-site pose bitwise equals the cube's initial pose (all 7 float32 of ``raw_pose`` equal);
    2. ``generator.get_state()`` is byte-identical before and after building the actors (the builder draws no random numbers).

    Drop sites are registered in ``env._hidden_objects``: invisible in sensor images (the base/hand cameras used by the recorder),
    for the same reason the original three tiers push goal_site below the table -- "where it goes after the demo" shows only in actions, with no extra marker in the image.
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
            # under CPU sim the entity pose can be read directly; under GPU sim it is not initialized yet in _load_scene, so only initial_pose is compared
            same = torch.equal(home.pose.raw_pose.detach().cpu(), raw.cpu())
        if not same:
            raise SceneGenerationError(f"drop site {home.name}: pose is not bitwise equal to the initial pose of cube {cube.name} (pose mismatch)")
        pose_equal.append(same)
        env._hidden_objects.append(home)
        homes.append(home)
    rng_equal = torch.equal(state_before, generator.get_state())
    if not rng_equal:
        raise SceneGenerationError("generator state differs before and after building drop-site actors (random draws are not allowed)")
    checks = {"pose_equal": pose_equal, "rng_state_equal": rng_equal}
    return homes, checks


def home_pose_record(cubes, homes) -> dict:
    """Content of ``actions.return_pose_by_object_id``: cube name -> drop-site raw_pose (3 values of p + 4 of q)."""
    return {
        cube.name: [float(v) for v in home.initial_pose.raw_pose[0].tolist()]
        for cube, home in zip(cubes, homes)
    }


def returned_mask(policy: str, count: int) -> list[bool]:
    """Whether each demo cube (in demo order) is put back to its original position; ``policy`` must already have passed :func:`validate_demo_plan`."""
    if policy == RETURN_TO_ORIGIN:
        return [True] * count
    if policy == RETURN_LAST_ONLY:
        return [k == count - 1 for k in range(count)]
    if policy == NO_RETURN:
        return [False] * count
    raise SceneGenerationError(f"unknown demo_return_policy={policy!r}")


# Avoidance radius (lower bound on center distance, meters) for non-returned drop sites. Cube half extent 0.02, diagonal half width about 0.028:
# cube-cube >= 0.07 (1.4 cm left even diagonal to diagonal); cube-target stand (radius 0.04) >= 0.08; cube-button base >= 0.10.
GOAL_DROP_CLEARANCE = {"cube": 0.07, "target": 0.08, "button": 0.10, "drop": 0.07}
# Candidate grid: 1 cm grid within ±GOAL_DROP_SEARCH_HALF around the goal_site center, sorted by distance to center then by (x, y) (deterministic, no random draws)
GOAL_DROP_SEARCH_HALF = 0.12
GOAL_DROP_GRID_STEP = 0.01


def goal_drop_candidates(center_xy, half: float = GOAL_DROP_SEARCH_HALF, step: float = GOAL_DROP_GRID_STEP):
    """Candidate drop sites around the goal_site center, from near to far; equal distances in (x, y) lexicographic order to keep it bitwise deterministic."""
    n = int(round(half / step))
    cx, cy = float(center_xy[0]), float(center_xy[1])
    pts = [(round(cx + i * step, 6), round(cy + j * step, 6)) for i in range(-n, n + 1) for j in range(-n, n + 1)]
    return sorted(pts, key=lambda q: (round((q[0] - cx) ** 2 + (q[1] - cy) ** 2, 10), q[0], q[1]))


def plan_goal_drop_xy(center_xy, count: int, obstacles) -> list[tuple[float, float]]:
    """Greedily pick ``count`` drop sites: each is the candidate closest to the goal_site center that is far enough from all obstacles and already chosen sites.

    ``obstacles``: ``[(kind, (x, y)), ...]``, kind ∈ GOAL_DROP_CLEARANCE (cube / target / button).
    Cube obstacles use the initial positions of **all** cubes (conservative: undemonstrated cubes, cubes not yet reached and returned cubes all stay there).
    Raises ``SceneGenerationError`` (a retryable task-level failure) when none is found; no silent relaxation.
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
            raise SceneGenerationError(f"cannot find collision-free non-returned drop site #{len(chosen) + 1} around goal_site")
    return chosen


def build_goal_drop_sites(env, cubes, goal_site, generator, obstacles, name_prefix: str = "goal_drop"):
    """Build hidden drop-site actors in the goal_site region for non-returned demo cubes; returns ``(sites, checks)``.

    Drop-site definition (implementation of V6 plan 2.10 / pending M11): with the goal_site center as origin, :func:`plan_goal_drop_xy`
    greedily picks the closest collision-free grid point (if the center itself is free, one cube lands exactly at the center, same as the original three tiers); picked in demo order.
    Same constraints as :func:`build_home_sites`: no random draws (``generator`` state byte-identical before and after, otherwise raise),
    registered in ``env._hidden_objects``, ``body_type=kinematic`` with no collision. Drop-site z is the cube half extent, orientation the unit quaternion.
    goal_site itself is pushed below the table in ``_initialize_episode``; only its xy is borrowed here.
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
        raise SceneGenerationError("generator state differs before and after building goal_site drop-site actors (random draws are not allowed)")
    return sites, {"rng_state_equal": rng_equal, "count": len(sites)}


def goal_drop_obstacles(env, button_xy) -> list:
    """Obstacles that non-returned drop sites must avoid: initial positions of all cubes, all target stands, the button. Reads only initial_pose, draws no random numbers."""
    xy = lambda actor: tuple(float(v) for v in actor.initial_pose.raw_pose.detach().cpu()[0, :2].tolist())  # noqa: E731
    obstacles = [("cube", xy(c)) for c in env.all_cubes]
    obstacles += [("target", xy(t)) for t in env.targets]
    obstacles.append(("button", (float(button_xy[0]), float(button_xy[1]))))
    return obstacles
