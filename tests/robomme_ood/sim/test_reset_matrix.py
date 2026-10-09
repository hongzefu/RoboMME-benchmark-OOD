"""L4 仿真冒烟：每任务 × 每档各 1 局真实 ``make_env_for_episode`` + ``reset``，不 step。

规模：``xhard0 16 任务 × 1 局 + xhard1～5 共 43 格 × 1 局 = 59`` 次 reset；连同
``test_official_one_reset.py`` 的 1 次，每次运行 ``tests/robomme_ood/sim`` 共 ``59 + 1 = 60`` 次 reset、0 条轨迹生成。
这 60 次 reset 属用户长期授权（计划 ``1003-code-test-maintenance-todo.md`` 的 Q8），不再逐次申请。

只能这样运行（先 ``nvidia-smi`` 选空闲卡）::

    CUDA_VISIBLE_DEVICES=<空闲卡> uv run --no-sync python -m pytest tests/robomme_ood/sim --allow-sim-reset -q

不带 ``--allow-sim-reset`` 时资源守卫以 UsageError 拒绝（见 ``tests/robomme_ood/_support/resource_policy.py``）。

格的枚举与清理前同口径扫描 ``docs/validation/test-redesign-20261003/records/reset_sweep.py`` 相同：
遍历 ``BenchmarkEnvBuilder.get_task_list()``；xhard0 取官方 ``test`` 集里 hard 子集的首局
（``hard_specs.XHARD0_EPISODES[0]``，即 ``dataset="hard-verify"`` 的 episode 0；本仓构建器不再接受 ``dataset="test"``）；
新值档按 ``dataset="ood"`` 的 episode 顺序取每档首局。
收集阶段只做 CPU 上的元数据／规格读取，不构建场景、不初始化 GPU；单进程顺序跑，每格结束 ``env.close()``。

每格断言（逐项对照该扫描的实测记录 ``reset-sweep.jsonl``，59 格全部 ok）：
- wrapper 链恰为 ``FailAwareWrapper → DemonstrationWrapper → TimeLimitWrapper → OrderEnforcing → <任务类>``，
  第 0 层 ``FailAwareWrapper`` 经 ``robomme_ood.env_record_wrapper.FailAwareWrapper`` 导入，但那是 shim——
  模块别名指向官方 ``robomme.env_record_wrapper.FailAwareWrapper``，类对象就是官方类；第 1 层
  ``DemonstrationWrapper`` 是 ``robomme_ood`` 自有复制件的类；任务类是 ``robomme_ood`` 的类；
- obs 恰五个键、形状与 dtype 固定，五个列表等长；``gripper_state_list`` 对机械臂任务是 float32，
  对 ``panda_stick`` 两个任务（PatternLock、RouteStick）是 float64 全零（生产代码
  ``DemonstrationWrapper`` 对 stick 环境显式造 ``np.zeros(2, float64)``，实测记录同此）；
- info 恰七个键、``status == "ongoing"``、``task_goal`` 为 1～4 条非空字符串；
- 帧数：本局任务表里含 ``demonstration=True`` 子任务的（由 reset 后 ``unwrapped.task_list`` 推出）帧数 > 1，
  否则恰为 1；推出的有无再与实测名单 ``DEMO_TASKS_MEASURED``（9 个）交叉核对；
- xhard1～5：``spec_binding`` 的 ``mode == "replay"``、``spec_sha256`` 等于包内规格行（且等于对该行 ``spec``
  重算的哈希）、``injected_mismatch == 0``、``unused == 0``——证明规格回注真的被环境消费；
  xhard0 无规格，``mode`` 不得是 ``replay``；
- ``unwrapped`` 的 ``seed``、``difficulty`` 与规格行（xhard0 为官方 test 元数据）一致。
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder, hard_specs
from robomme_ood.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from robomme_ood.env_record_wrapper.FailAwareWrapper import FailAwareWrapper

pytestmark = pytest.mark.sim

#: 实测示教任务名单，只作交叉核对（主判据是 reset 后 ``unwrapped.task_list`` 里有无 ``demonstration=True``）。
#: 来源：清理前同口径扫描的实测记录 ``docs/validation/test-redesign-20261003/records/reset-sweep.jsonl``（59 格全 ok）：
#: 这 9 个任务在全部档帧数 > 1，其余 7 个任务恰为 1 帧。
DEMO_TASKS_MEASURED = frozenset({
    "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder", "VideoUnmask", "VideoUnmaskSwap",
    "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
})
assert len(DEMO_TASKS_MEASURED) == 9

OBS_SPEC = {
    "front_rgb_list": ((256, 256, 3), np.uint8),
    "wrist_rgb_list": ((256, 256, 3), np.uint8),
    "joint_state_list": ((7,), np.float32),
    "eef_state_list": ((6,), np.float64),
    "gripper_state_list": ((2,), np.float32),  # stick 环境另见 _expected_obs_spec
}
INFO_KEYS = {
    "elapsed_steps", "success", "fail", "simple_subgoal_online", "grounded_subgoal_online", "task_goal", "status",
}
CHAIN_NAMES = ("FailAwareWrapper", "DemonstrationWrapper", "TimeLimitWrapper", "OrderEnforcing")


def _enumerate_cells() -> list[tuple[str, str, int, str]]:
    """与 reset_sweep.py 同口径枚举 (任务, dataset, episode, 档)；只读元数据与规格，不建场景。"""
    cells: list[tuple[str, str, int, str]] = []
    for task in BenchmarkEnvBuilder.get_task_list():
        cells.append((task, "hard-verify", 0, hard_specs.XHARD0))  # hard-verify 第 0 局 = 原 episode XHARD0_EPISODES[0]
        builder = BenchmarkEnvBuilder(env_id=task, dataset="ood", action_space="joint_angle")
        seen: set[str] = set()
        for episode in range(builder.get_episode_num()):
            tier = builder.resolve_episode(episode)[1]
            if tier == hard_specs.XHARD0 or tier in seen:
                continue  # 防御：ood 不含 xhard0，xhard0 统一由上面的 hard-verify 格覆盖
            seen.add(tier)
            cells.append((task, "ood", episode, tier))
    return cells


CELLS = _enumerate_cells()
_XHARD0 = [c for c in CELLS if c[3] == hard_specs.XHARD0]
_NEW = [c for c in CELLS if c[3] != hard_specs.XHARD0]
# 枚举漂移在收集期就响亮失败：16 个 xhard0 + 恰为交付格表的 43 格
assert len(_XHARD0) == len(hard_specs.ALL_TASKS) == 16, f"xhard0 格数 {len(_XHARD0)} != 16"
assert {(t, tier) for t, _, _, tier in _NEW} == set(hard_specs.EXPECTED_CELLS) and len(_NEW) == 43, \
    f"新值档格数 {len(_NEW)} 与交付格表 EXPECTED_CELLS（43 格）不符"
assert len(CELLS) == 59


@functools.lru_cache(maxsize=None)
def _spec_rows(tier: str) -> dict[tuple[str, int], dict]:
    """包内规格行（正式局），按 (任务, candidate) 索引；独立于 builder 的读取路径。"""
    _header, rows = hard_specs.load_specs(hard_specs.packaged_specs_path(tier), check_fingerprint=False)
    return {(row["task"], int(row["candidate"])): row for row in rows if hard_specs.delivered(row)}


def _chain(env) -> list:
    layers, e = [], env
    while hasattr(e, "env"):
        layers.append(e)
        e = e.env
    return layers + [e]


def _expected_obs_spec(unwrapped) -> dict:
    spec = dict(OBS_SPEC)
    if unwrapped.robot_uids == "panda_stick":
        spec["gripper_state_list"] = ((2,), np.float64)
    return spec


@pytest.mark.parametrize(
    ("task", "dataset", "episode", "tier"),
    CELLS,
    ids=[f"{task}-{tier}-ep{episode}" for task, _ds, episode, tier in CELLS],
)
def test_reset_cell(task: str, dataset: str, episode: int, tier: str) -> None:
    builder = BenchmarkEnvBuilder(
        env_id=task, dataset=dataset, action_space="joint_angle",
        max_steps=1300 if tier == hard_specs.XHARD0 else 1600,  # xhard0 取官方默认 1300，新值档取 EXEC_CAP 1600
    )
    if tier == hard_specs.XHARD0:
        expected_seed, resolved_tier = builder.resolve_episode(episode)
        assert resolved_tier == hard_specs.XHARD0
        assert builder.resolve_identity(episode)["source_episode"] == hard_specs.XHARD0_EPISODES[0]
        expected_difficulty = "hard"  # xhard0 走官方原生 hard 分支：gym.make 收到 difficulty="hard"
        row = None
    else:
        identity = builder.resolve_identity(episode)
        assert identity["tier"] == tier
        row = _spec_rows(tier)[(task, int(identity["candidate"]))]
        assert int(row["seed"]) == identity["seed"]
        assert row["spec_sha256"] == identity["spec_sha256"] == hard_specs.spec_sha256(row["spec"])
        expected_seed, expected_difficulty = int(row["seed"]), tier

    env = builder.make_env_for_episode(episode)
    try:
        obs, info = env.reset()
        unwrapped = env.unwrapped

        # —— wrapper 链 ——
        layers = _chain(env)
        assert [type(x).__name__ for x in layers] == [*CHAIN_NAMES, task]
        assert type(layers[0]) is FailAwareWrapper  # 官方类（robomme_ood 侧只是模块别名）
        assert type(layers[1]) is DemonstrationWrapper  # robomme_ood 自有复制件
        assert layers[-1] is unwrapped
        assert type(unwrapped).__module__.startswith("robomme_ood."), type(unwrapped).__module__

        # —— obs ——
        spec = _expected_obs_spec(unwrapped)
        assert set(obs) == set(spec)
        lengths = {key: len(obs[key]) for key in spec}
        assert len(set(lengths.values())) == 1, f"五个列表不等长：{lengths}"
        n_frames = lengths["front_rgb_list"]
        assert n_frames >= 1
        for key, (shape, dtype) in spec.items():
            for i, frame in enumerate(obs[key]):
                arr = np.asarray(frame)
                assert arr.shape == shape and arr.dtype == dtype, f"{key}[{i}] 为 {arr.shape} {arr.dtype}"
        if unwrapped.robot_uids == "panda_stick":
            assert all(not np.any(np.asarray(g)) for g in obs["gripper_state_list"])

        # —— info ——
        assert set(info) == INFO_KEYS
        assert info["status"] == "ongoing"
        goals = info["task_goal"]
        assert isinstance(goals, list) and 1 <= len(goals) <= 4, goals
        assert all(isinstance(g, str) and g.strip() for g in goals), goals

        # —— 示教帧 ——
        has_demo = any(t.get("demonstration", False) for t in unwrapped.task_list)
        if has_demo:
            assert n_frames > 1, f"{task} 含示教子任务但只有 {n_frames} 帧"
        else:
            assert n_frames == 1, f"{task} 无示教子任务却有 {n_frames} 帧"
        assert has_demo == (task in DEMO_TASKS_MEASURED), f"{task} 的示教有无与实测名单不符"

        # —— 规格回注 ——
        binding = hard_specs.spec_binding(env)
        if row is None:
            assert binding.get("mode") != "replay", binding
        else:
            assert binding["available"] is True
            assert binding["mode"] == "replay", binding
            assert binding["spec_sha256"] == row["spec_sha256"]
            assert binding["injected_mismatch"] == 0, binding
            assert binding["unused"] == 0, binding

        # —— 身份 ——
        assert int(unwrapped.seed) == int(expected_seed)
        assert unwrapped.difficulty == expected_difficulty
    finally:
        env.close()
