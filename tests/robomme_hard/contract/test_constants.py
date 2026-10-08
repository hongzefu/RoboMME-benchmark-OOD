"""L1 契约：全套件唯一写业务常量字面值的地方（计划细则 4.5.2、红线 R8）。

本文件上半部是钉值（模块级常量，供 ``tests/robomme_hard/contract`` 其余用例按名导入，它们自己不再写字面值）；
下半部是用例：逐项读生产对象，与钉值比。期望值一律是手写字面量或手算结果，不在测试里复刻被测公式。

几类读不到模块级常量的值（挑战接口的重试间隔、录像器的函数局部阈值）用最小行为探针测出数值后再与钉值比，
探针方法写在各用例的 docstring 里。

按档步数查表 ``TIER_MAX_STEPS`` 已删除（1003 评估计划 1.1，原 Q16 随之了结）：评估步数上限由入口
``scripts/evaluation_ood.py`` 按数据集传 ``max_steps``（``ood`` 1800、``hard-verify`` 1300，钉值见
``DATASET_MAX_STEPS``），本文件另钉交付侧的 ``EXEC_CAP``（1600），并断言查表不再存在。

本仓只保留评估侧：xhard0 开关 ``XHARD0_IN_TEST_HARD``、规格根环境变量 ``SPECS_ROOT_ENV``、hard 包 train 元数据与
生成侧冻结口径（``_freeze``）都已删除，相应钉值随之删去；``test_no_generation_switches`` 断言它们不再存在。
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tests.robomme_hard._support.loaders import REPO

# =============================================================================
# 钉值（全套件唯一一处业务常量字面值）
# =============================================================================

#: 16 任务规范序（官方 ``_DEFAULT_TASK_LIST`` 的顺序）
TASKS = (
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap", "VideoUnmask",
    "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "PickHighlight", "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
)
N_TASKS = 16
#: 新值五档与 xhard0
NEW_TIERS = ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5")
XHARD0 = "xhard0"
#: V9 交付格表（v9 方案第一部分表 2），逐格手写
V9_CELLS = {
    ("PickXtimes", "xhard1"): 17, ("PickXtimes", "xhard2"): 17, ("PickXtimes", "xhard3"): 16,
    ("RouteStick", "xhard1"): 17, ("RouteStick", "xhard2"): 17, ("RouteStick", "xhard3"): 16,
    ("PatternLock", "xhard1"): 17, ("PatternLock", "xhard2"): 17, ("PatternLock", "xhard3"): 16,
    ("SwingXtimes", "xhard1"): 10, ("SwingXtimes", "xhard2"): 10, ("SwingXtimes", "xhard3"): 10,
    ("SwingXtimes", "xhard4"): 10, ("SwingXtimes", "xhard5"): 10,
    ("StopCube", "xhard1"): 10, ("StopCube", "xhard2"): 10, ("StopCube", "xhard3"): 10,
    ("StopCube", "xhard4"): 10, ("StopCube", "xhard5"): 10,
    ("VideoUnmask", "xhard1"): 13, ("VideoUnmask", "xhard2"): 13, ("VideoUnmask", "xhard3"): 12,
    ("VideoUnmask", "xhard4"): 12,
    ("ButtonUnmask", "xhard1"): 13, ("ButtonUnmask", "xhard2"): 13, ("ButtonUnmask", "xhard3"): 12,
    ("ButtonUnmask", "xhard4"): 12,
    ("BinFill", "xhard1"): 25, ("BinFill", "xhard2"): 25,
    ("VideoUnmaskSwap", "xhard1"): 25, ("VideoUnmaskSwap", "xhard2"): 25,
    ("ButtonUnmaskSwap", "xhard1"): 25, ("ButtonUnmaskSwap", "xhard2"): 25,
    ("VideoPlaceButton", "xhard1"): 25, ("VideoPlaceButton", "xhard2"): 25,
    ("VideoPlaceOrder", "xhard1"): 25, ("VideoPlaceOrder", "xhard2"): 25,
    ("PickHighlight", "xhard1"): 25, ("PickHighlight", "xhard2"): 25,
    ("VideoRepick", "xhard1"): 25, ("VideoRepick", "xhard2"): 25,
    ("MoveCube", "xhard4"): 50,
    ("InsertPeg", "xhard4"): 50,
}
N_CELLS = 43
PER_TASK = 50
#: 16 任务 × 50 局
TOTAL = 800
#: xhard0 每任务 12 局（官方 test 的 hard 子集，原 episode 3,7,…,47）
XHARD0_PER_TASK = 12
XHARD0_EPISODES = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
#: hard-verify：16 任务 × 12 局
TOTAL_HARD_VERIFY = 192
#: 本构建器只认这两个数据集，缺省 ood
DATASETS = ("hard-verify", "ood")
DEFAULT_DATASET = "ood"
#: 评估入口 ``scripts/evaluation_ood.py`` 按数据集传的步数上限
DATASET_MAX_STEPS = {"hard-verify": 1300, "ood": 1800}
#: 执行步上限（生成侧抽样与交付的上限）
EXEC_CAP = 1600
#: 规格回放时记录点（``source="record"``）GPU 生成与 CPU 离线的浮点容差：≤ 此值计 recorded_drift（U-13 方案甲）
RECORDED_FLOAT_TOL = 1e-5
#: 只在 xhard4 交付的任务
XHARD4_ONLY = ("InsertPeg", "MoveCube")
#: 按档 seed 偏移与 seed 公式参数
SEED_OFFSETS = {"xhard1": 16_000_000, "xhard2": 18_000_000, "xhard3": 20_000_000,
                "xhard4": 22_000_000, "xhard5": 24_000_000}
SEED_ENV_BLOCK = 100_000
SEED_EPISODE_STRIDE = 100
MAX_ATTEMPTS = 100
#: 手算 seed 样例：(任务, 档, episode, attempt, seed)；env_code 为规范序 1 起的位置
SEED_EXAMPLES = (
    ("PickXtimes", "xhard1", 3, 2, 16_100_302),   # 16e6 + 1×1e5 + 3×100 + 2
    ("StopCube", "xhard5", 0, 0, 24_200_000),     # 24e6 + 2×1e5（包内 xhard5 StopCube 第 0 行实测同值）
    ("MoveCube", "xhard4", 10, 5, 23_401_005),    # 22e6 + 14×1e5 + 10×100 + 5
    ("RouteStick", "xhard3", 999, 99, 21_699_999),  # 20e6 + 16×1e5 + 999×100 + 99
)
#: gym.make 的 runtime 四项
RUNTIME = {"obs_mode": "rgb+depth+segmentation", "control_mode": "pd_joint_pos",
           "render_mode": "rgb_array", "reward_mode": "dense"}
SPECS_SCHEMA = "hard-specs/4"
SPEC_KIND = "native-newvalue/2"
LAYOUT_RULE = {"mode": "independent"}
#: 包内五份规格（全部行 / selected 行），合计 1518 行、800 局
PACKAGED_ROWS = {"xhard1": 541, "xhard2": 551, "xhard3": 167, "xhard4": 233, "xhard5": 26}
PACKAGED_ROWS_TOTAL = 1518
PACKAGED_SELECTED = {"xhard1": 272, "xhard2": 272, "xhard3": 92, "xhard4": 144, "xhard5": 20}
#: 包内规格结果段里 error_type == exec_over_cap 的行数（实测 0）
PACKAGED_EXEC_OVER_CAP = 0
#: MoveCube xhard4 50 局 × 2 段 × 3 物体 = 300 点；运动方式 0/1/2 配额 17/17/16
MOVECUBE_POINTS = 300
MOVECUBE_WAYS = {0: 17, 1: 17, 2: 16}
#: MoveCube xhard4 的 V9 区域三个数（1002 方案第一部分 §1 已定口径 1）与改值前的 V8 区域（只用来数「落在旧区域外」）
MOVECUBE_REGION_V9 = {"r_in": 0.24, "r_out": 0.42, "base_dist": [0.31, 0.80]}
MOVECUBE_REGION_V8 = {"r_in": 0.12, "r_out": 0.20, "base_dist": [0.35, 0.76]}
#: 官方元数据：每 split 16 个文件、每文件局数
OFFICIAL_SPLIT_FILES = 16
OFFICIAL_SPLIT_EPISODES = {"train": 100, "val": 50, "test": 50}
#: 本仓已删除、不得再出现的生成侧／开关符号：(模块, 名字)
REMOVED_SYMBOLS = (
    ("hard_specs", "XHARD0_IN_TEST_HARD"), ("hard_specs", "xhard0_prefix"), ("hard_specs", "SPECS_ROOT_ENV"),
    ("hard_builder", "HARD_TRAIN_TASKS"), ("hard_builder", "HARD_METADATA_ROOT"), ("hard_builder", "_HARD_DATASETS"),
    ("hard_builder", "_ALLOWED_ACTION_SPACES"), ("hard_builder", "_override_cells"), ("hard_builder", "_root_specs"),
)
#: 数据集接口改名前的旧名（1006 改名计划 R1；构建器必须拒绝）。全仓旧名别名表只此一段，
#: ``tests/robomme_hard/static/test_official_names.py`` 对段内豁免、段外照抓。
# >>> LEGACY_NAMES
LEGACY_DATASET_NAMES = ("test-hard0", "test-hard")
# <<< LEGACY_NAMES
#: 挑战接口
CHALLENGE_MAX_STEPS_DEFAULT = 1500
CHALLENGE_ACTION_SHAPES = {"joint_angle": (8,), "ee_pose": (7,), "waypoint": (7,)}
DUMMY_POLICY_CHUNK = 10
DUMMY_POLICY_ACTION_DIM = 8
CLIENT_RETRY_SLEEP_S = 5
#: 录像器（官方与 hard 复制件同值，fail_safe_limit 除外）
RECORD_GRIPPER_CLOSE_LT = 0.03
RECORD_OVERLAY_LINE_HEIGHT = 20
RECORD_OVERLAY_PADDING = 10
RECORD_OVERLAY_MIN_HEIGHT = 50
RECORD_FK_NEGATIVE_FINGER = 0.04
RECORD_FAIL_SAFE_LIMIT = {"official": 2000, "hard": 5000}


# =============================================================================
# 公共小工具
# =============================================================================


def hard_specs():
    from robomme_hard.env_record_wrapper import hard_specs as hs

    return hs


def recording_fakes():
    """录制替身（T5 的 ``tests/robomme_hard/pipeline/recording/recording_fakes.py``，只读复用）。按路径加载成独立模块名；
    文件里有 dataclass，执行前须先登记模块名。"""
    name = "_contract_recording_fakes"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "tests" / "robomme_hard" / "pipeline" / "recording" / "recording_fakes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# 交付格表、局数与开关
# =============================================================================


def test_v9_cells_exact():
    hs = hard_specs()
    assert dict(hs.V9_CELLS) == V9_CELLS
    assert dict(hs.EXPECTED_CELLS) == V9_CELLS
    assert {name: dict(t) for name, t in hs.CELL_TABLES.items()} == {"v9": V9_CELLS}


def test_v9_cell_count_per_task_and_total():
    hs = hard_specs()
    assert len(hs.V9_CELLS) == N_CELLS
    assert sum(hs.V9_CELLS.values()) == TOTAL
    per_task = {}
    for (task, _tier), n in hs.V9_CELLS.items():
        per_task[task] = per_task.get(task, 0) + n
    assert per_task == {task: PER_TASK for task in TASKS}
    assert hs.V9_PER_TASK == PER_TASK


def test_no_generation_switches():
    """xhard0 开关、规格根环境变量、hard 包 train 元数据与生成侧入口都已删除：模块不再导出这些符号，
    ``env_metadata`` 下只有 ``ood``；hard-verify 局数 16 × 12 = 192、ood 16 × 50 = 800。"""
    from robomme_hard.env_record_wrapper import hard_builder

    modules = {"hard_specs": hard_specs(), "hard_builder": hard_builder}
    present = [f"{m}.{name}" for m, name in REMOVED_SYMBOLS if hasattr(modules[m], name)]
    assert present == []
    meta = REPO / "src" / "robomme_hard" / "env_metadata"
    assert sorted(p.name for p in meta.iterdir() if p.is_dir()) == ["ood"]
    assert N_TASKS * XHARD0_PER_TASK == TOTAL_HARD_VERIFY
    assert N_TASKS * PER_TASK == TOTAL
    assert hard_builder.OOD == DEFAULT_DATASET
    assert set(hard_builder._ALLOWED_DATASETS) == set(DATASETS)


def test_xhard0_constants():
    hs = hard_specs()
    assert hs.XHARD0 == XHARD0
    assert hs.XHARD0_PER_TASK == XHARD0_PER_TASK
    assert tuple(hs.XHARD0_EPISODES) == XHARD0_EPISODES
    assert tuple(hs.TIERS) == NEW_TIERS
    assert tuple(hs.BUILDER_TIERS) == (XHARD0, *NEW_TIERS)


def test_exec_cap_and_no_tier_table():
    """``EXEC_CAP`` 等于钉值 1600；按档步数查表 ``TIER_MAX_STEPS`` 已删除，模块与包都不再导出它。"""
    hs = hard_specs()
    assert hs.EXEC_CAP == EXEC_CAP
    assert not hasattr(hs, "TIER_MAX_STEPS")
    package = importlib.import_module("robomme_hard.env_record_wrapper")
    assert not hasattr(package, "TIER_MAX_STEPS")


def test_recorded_float_tol():
    """``hard_specs.RECORDED_FLOAT_TOL`` 与钉值相同（判定器边界行为另由 ``tests/robomme_hard/unit/hard/test_episode_spec.py`` 覆盖）。"""
    hs = hard_specs()
    assert hs.RECORDED_FLOAT_TOL == RECORDED_FLOAT_TOL
    assert isinstance(hs.RECORDED_FLOAT_TOL, float)


def test_xhard4_only():
    hs = hard_specs()
    assert tuple(hs.XHARD4_ONLY) == XHARD4_ONLY
    assert hs.xhard4_only_tasks(hs.V9_CELLS) == set(XHARD4_ONLY)


# =============================================================================
# seed 公式与按档偏移
# =============================================================================


def test_seed_offsets_and_rule():
    hs = hard_specs()
    assert {k: dict(v) for k, v in hs.TIER_SEED_OFFSETS.items()} == {"v8": SEED_OFFSETS}
    for tier, offset in SEED_OFFSETS.items():
        rule = hs.seed_rule_for(tier, "v8")
        assert rule["offset"] == offset
        assert rule["env_block"] == SEED_ENV_BLOCK and rule["episode_stride"] == SEED_EPISODE_STRIDE
    assert hs.MAX_ATTEMPTS == MAX_ATTEMPTS


@pytest.mark.parametrize("task,tier,episode,attempt,seed", SEED_EXAMPLES)
def test_seed_examples_hand_computed(task, tier, episode, attempt, seed):
    hs = hard_specs()
    assert hs.seed_for(task, episode, attempt, hs.seed_rule_for(tier, "v8")) == seed


def test_seed_rule_rejects_unknown_and_attempt_bounds():
    hs = hard_specs()
    with pytest.raises(hs.SpecsError):
        hs.seed_rule_for("xhard1", "v7")
    with pytest.raises(hs.SpecsError):
        hs.seed_rule_for(XHARD0, "v8")
    rule = hs.seed_rule_for("xhard1", "v8")
    with pytest.raises(hs.SpecsError):
        hs.seed_for("PickXtimes", 0, MAX_ATTEMPTS, rule)
    with pytest.raises(hs.SpecsError):
        hs.seed_for("PickXtimes", 0, -1, rule)
    with pytest.raises(hs.SpecsError):
        hs.seed_for("NotATask", 0, 0, rule)


def test_runtime_schema_layout():
    hs = hard_specs()
    assert hs.RUNTIME == RUNTIME
    assert hs.SCHEMA == SPECS_SCHEMA
    assert hs.LAYOUT_RULE == LAYOUT_RULE


# =============================================================================
# 16 任务名单与注册
# =============================================================================


def test_task_list_everywhere_equal():
    hs = hard_specs()
    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder as Official
    from robomme_hard.env_record_wrapper.hard_builder import BenchmarkEnvBuilder as Hard

    assert len(TASKS) == N_TASKS
    assert tuple(hs.ALL_TASKS) == TASKS
    assert tuple(Official.get_task_list()) == TASKS
    assert tuple(Hard.get_task_list()) == TASKS


def test_registered_ids_equal_task_set():
    """导入 robomme_hard 后注册表里 16 个 id 恰为任务名单，类都属于 robomme_hard（只读注册表，不建环境）。"""
    import robomme_hard
    from mani_skill.utils.registration import REGISTERED_ENVS

    assert set(robomme_hard.robomme_env.ENV_IDS) == set(TASKS)
    assert len(robomme_hard.robomme_env.ENV_IDS) == N_TASKS
    for uid in TASKS:
        assert REGISTERED_ENVS[uid].cls.__module__.startswith("robomme_hard.")


# =============================================================================
# 挑战接口（R8 移交过来的常量）
# =============================================================================


def test_phase1_eval_defaults(monkeypatch):
    mod = importlib.import_module("challenge_interface.scripts.phase1_eval")
    monkeypatch.setattr(sys, "argv", ["phase1_eval.py"])
    args = mod.parse_args()
    assert args.max_steps == CHALLENGE_MAX_STEPS_DEFAULT
    assert {k: tuple(v) for k, v in mod.EXPECTED_ACTION_SHAPES.items()} == CHALLENGE_ACTION_SHAPES


def test_dummy_policy_chunk():
    from challenge_interface.policy import DummyPolicy

    policy = DummyPolicy()
    policy.reset()
    out = policy.infer({"is_first_step": True, "front_rgb_list": [None, None]})
    assert policy.chunk_size == DUMMY_POLICY_CHUNK
    assert out["actions"].shape == (DUMMY_POLICY_CHUNK, DUMMY_POLICY_ACTION_DIM)


def test_policy_client_retry_interval(monkeypatch):
    """探针：connect 第一次抛 ConnectionRefusedError、第二次成功；替身 time.sleep 记下的间隔即重试间隔。"""
    import challenge_interface.client as client_mod
    from challenge_interface import msgpack_numpy

    calls = {"connect": 0}
    sleeps: list[float] = []

    class _Conn:
        def recv(self):
            return msgpack_numpy.packb({"meta": 1})

    def fake_connect(uri, **kwargs):
        calls["connect"] += 1
        if calls["connect"] == 1:
            raise ConnectionRefusedError
        return _Conn()

    monkeypatch.setattr(client_mod.websockets.sync.client, "connect", fake_connect)
    monkeypatch.setattr(client_mod, "time", types.SimpleNamespace(sleep=sleeps.append))
    client = client_mod.PolicyClient(host="127.0.0.1", port=1)
    assert client.get_server_metadata() == {"meta": 1}
    assert calls["connect"] == 2
    assert sleeps == [CLIENT_RETRY_SLEEP_S]


# =============================================================================
# 录像器（函数局部值，用行为探针测出）
# =============================================================================

RECORD_KINDS = ("official", "hard")


def _record_cls(kind):
    return recording_fakes().record_module(kind).RobommeRecordWrapper


@pytest.mark.parametrize("kind", RECORD_KINDS)
def test_overlay_geometry(kind):
    """探针：帧宽 21 → 可用宽度 1 像素，每个词独占一行；n 行时文字区高 = max(最小高, n×行高 + 留白)。
    由 1、4、5 行三个实测高度反推：行高 = h5 − h4，留白 = h5 − 5×行高，最小高 = h1。"""
    cls = _record_cls(kind)
    frame = np.zeros((7, 21, 3), dtype=np.uint8)

    def extra(n):
        out = cls._add_text_to_frame(None, frame, " ".join(["a"] * n))
        return out.shape[0] - frame.shape[0]

    h1, h4, h5 = extra(1), extra(4), extra(5)
    line = h5 - h4
    assert line == RECORD_OVERLAY_LINE_HEIGHT
    assert h5 - 5 * line == RECORD_OVERLAY_PADDING
    assert h1 == RECORD_OVERLAY_MIN_HEIGHT
    assert cls._add_text_to_frame(None, frame, "") is frame  # 空文本原样返回


@pytest.mark.parametrize("kind", RECORD_KINDS)
def test_fk_negative_gripper_finger(kind):
    """探针：FK 替身记录传入的完整 qpos 后抛错（被录像器吞掉返回 None）；负指令两指取固定开度，正指令取指令值。"""
    cls = _record_cls(kind)
    seen: list[np.ndarray] = []

    class _Pin:
        def compute_forward_kinematics(self, q):
            seen.append(np.asarray(q, dtype=np.float64).copy())
            raise RuntimeError("探针到此为止")

    obj = object.__new__(cls)
    obj._fk_available = True
    obj._fk_qpos_size = 9
    obj._mplib_planner = types.SimpleNamespace(pinocchio_model=_Pin())
    obj._ee_link_idx = 0
    for grip in (-1.0, -0.3, 0.02):
        assert cls._joint_action_to_ee_pose_dict(obj, np.array([0.0] * 7 + [grip])) is None
    fingers = [q[7:].tolist() for q in seen]
    assert fingers[0] == fingers[1] == [RECORD_FK_NEGATIVE_FINGER] * 2
    assert fingers[2] == [0.02, 0.02]


def _gripper_close_flag(kind, tmp_path, finger: float, monkeypatch) -> bool:
    """一局一步的真实录制：替身环境 step 之后把两指位置改成 ``finger``（float64），读 h5 的 is_gripper_close。
    （录像器在 save_video=False 时不写逐步记录，所以保持 save_video=True、只把 mp4 编码替换成空操作。）"""
    import h5py
    import torch

    rf = recording_fakes()
    cls = _record_cls(kind)
    monkeypatch.setattr(cls, "_video_write_mp4", lambda self, frames, output_path: None)
    event = rf.Event(name="pick", terminated=True, success=True)
    w, env = rf.make_wrapper(cls, tmp_path, [event])
    orig = env.step

    def step(action):
        out = orig(action)
        q = env.agent.robot.qpos.to(torch.float64).clone()
        q[0, 7:9] = finger
        env.agent.robot.qpos = q
        return out

    env.step = step
    w.reset()
    rf.drive(w, env, [event])
    w.close()
    with h5py.File(w.dataset_path, "r") as f:
        return bool(f["episode_3/timestep_0/obs/is_gripper_close"][()])


@pytest.mark.parametrize("kind", RECORD_KINDS)
def test_gripper_close_threshold(kind, tmp_path, monkeypatch):
    """探针：两指恰为阈值时判张开、阈值的前一个 double 判闭合 → 阈值恰为钉值且比较为严格小于。"""
    below = math.nextafter(RECORD_GRIPPER_CLOSE_LT, 0.0)
    assert _gripper_close_flag(kind, tmp_path / "below", below, monkeypatch) is True
    assert _gripper_close_flag(kind, tmp_path / "at", RECORD_GRIPPER_CLOSE_LT, monkeypatch) is False


def _failsafe_raises(kind, tmp_path, elapsed: int) -> bool:
    rf = recording_fakes()
    event = rf.Event(name="pick", elapsed=elapsed)
    w, env = rf.make_wrapper(_record_cls(kind), tmp_path / f"e{elapsed}", [event], save_video=False)
    w.reset()
    try:
        rf.drive(w, env, [event])
        return False
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ == "FailsafeTimeout"
        return True
    finally:
        w.h5_file.close()


@pytest.mark.parametrize("kind", RECORD_KINDS)
def test_fail_safe_limit(kind, tmp_path):
    """探针：elapsed_steps = 上限 − 1 不触发、= 上限触发（比较为 ≥）→ 上限恰为钉值。官方 2000、hard 复制件 5000。"""
    limit = RECORD_FAIL_SAFE_LIMIT[kind]
    assert _failsafe_raises(kind, tmp_path, limit - 1) is False
    assert _failsafe_raises(kind, tmp_path, limit) is True


# =============================================================================
# 外部夹具接口守卫
# =============================================================================


def test_recording_fakes_interface_used_here():
    """本文件按路径借用 T5 的录制替身；它的接口一变，上面的录像器探针会以难懂的方式报错。
    这里先把本文件用到的接口面逐项点名，变更时直接报「T5 夹具接口变更」。"""
    import dataclasses
    import inspect

    rf = recording_fakes()
    needed = ("record_module", "make_wrapper", "drive", "Event")
    missing = [name for name in needed if not hasattr(rf, name)]
    assert not missing, f"T5 夹具接口变更：recording_fakes 缺 {missing}"
    event_fields = {f.name for f in dataclasses.fields(rf.Event)}
    lost = {"name", "terminated", "success", "elapsed"} - event_fields
    assert not lost, f"T5 夹具接口变更：Event 缺字段 {sorted(lost)}"
    params = inspect.signature(rf.make_wrapper).parameters
    assert "save_video" in params, "T5 夹具接口变更：make_wrapper 不再接受 save_video"
    for kind in RECORD_KINDS:
        assert hasattr(rf.record_module(kind), "RobommeRecordWrapper"), f"T5 夹具接口变更：record_module({kind!r})"
