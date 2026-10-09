"""L1 contract: the only place in the whole suite that writes business-constant literals (plan details 4.5.2, red line R8).

The upper half of this file holds pins (module-level constants that the other tests in ``tests/robomme_ood/contract`` import by name instead of writing literals);
the lower half holds tests: each reads a production object and compares it to the pin. Expected values are always hand-written literals or hand computations; formulas under test are never re-implemented here.

Values not readable as module-level constants (the challenge interface retry interval, the recorder's function-local thresholds) are measured with minimal behavioral probes and then compared to pins;
the probe method is described in each test's docstring.

The per-tier step lookup table ``TIER_MAX_STEPS`` is removed (1003 evaluation plan 1.1, closing the original Q16): the evaluation step cap is passed per dataset by
the entry ``scripts/evaluation_ood.py`` as ``max_steps`` (``ood`` 1800, ``hard-verify`` 1300, pinned in
``DATASET_MAX_STEPS``); this file additionally pins the delivery-side ``EXEC_CAP`` (1600) and asserts the lookup table no longer exists.

This repo keeps only the evaluation side: the xhard0 switch ``XHARD0_IN_TEST_HARD``, the spec-root env var ``SPECS_ROOT_ENV``, the hard package's train metadata and
the generation-side freeze settings (``_freeze``) are all removed along with their pins; ``test_no_generation_switches`` asserts they no longer exist.
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

from tests.robomme_ood._support.loaders import REPO

# =============================================================================
# Pins (the only business-constant literals in the whole suite)
# =============================================================================

#: 16-task canonical order (the order of the official ``_DEFAULT_TASK_LIST``)
TASKS = (
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap", "VideoUnmask",
    "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "PickHighlight", "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
)
N_TASKS = 16
#: Five new-value tiers and xhard0
NEW_TIERS = ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5")
XHARD0 = "xhard0"
#: V9 delivery cell table (v9 plan part one table 2), hand-written cell by cell
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
#: 16 tasks x 50 episodes
TOTAL = 800
#: 12 xhard0 episodes per task (hard subset of the official test split, original episodes 3,7,...,47)
XHARD0_PER_TASK = 12
XHARD0_EPISODES = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
#: hard-verify: 16 tasks x 12 episodes
TOTAL_HARD_VERIFY = 192
#: This builder accepts only these two datasets; default ood
DATASETS = ("hard-verify", "ood")
DEFAULT_DATASET = "ood"
#: Step caps passed per dataset by the evaluation entry ``scripts/evaluation_ood.py``
DATASET_MAX_STEPS = {"hard-verify": 1300, "ood": 1800}
#: Execution-step cap (cap for generation-side sampling and delivery)
EXEC_CAP = 1600
#: Float tolerance between GPU generation and CPU offline at record points (``source="record"``) during spec replay: <= this counts as recorded_drift (U-13 option A)
RECORDED_FLOAT_TOL = 1e-5
#: Tasks delivered only in xhard4
XHARD4_ONLY = ("InsertPeg", "MoveCube")
#: Per-tier seed offsets and seed formula parameters
SEED_OFFSETS = {"xhard1": 16_000_000, "xhard2": 18_000_000, "xhard3": 20_000_000,
                "xhard4": 22_000_000, "xhard5": 24_000_000}
SEED_ENV_BLOCK = 100_000
SEED_EPISODE_STRIDE = 100
MAX_ATTEMPTS = 100
#: Hand-computed seed samples: (task, tier, episode, attempt, seed); env_code is the 1-based position in canonical order
SEED_EXAMPLES = (
    ("PickXtimes", "xhard1", 3, 2, 16_100_302),   # 16e6 + 1×1e5 + 3×100 + 2
    ("StopCube", "xhard5", 0, 0, 24_200_000),     # 24e6 + 2x1e5 (row 0 of packaged xhard5 StopCube measured to the same value)
    ("MoveCube", "xhard4", 10, 5, 23_401_005),    # 22e6 + 14×1e5 + 10×100 + 5
    ("RouteStick", "xhard3", 999, 99, 21_699_999),  # 20e6 + 16×1e5 + 999×100 + 99
)
#: The four runtime items of gym.make
RUNTIME = {"obs_mode": "rgb+depth+segmentation", "control_mode": "pd_joint_pos",
           "render_mode": "rgb_array", "reward_mode": "dense"}
SPECS_SCHEMA = "hard-specs/4"
SPEC_KIND = "native-newvalue/2"
LAYOUT_RULE = {"mode": "independent"}
#: Five packaged spec files (all rows / selected rows), 1518 rows and 800 episodes in total
PACKAGED_ROWS = {"xhard1": 541, "xhard2": 551, "xhard3": 167, "xhard4": 233, "xhard5": 26}
PACKAGED_ROWS_TOTAL = 1518
PACKAGED_SELECTED = {"xhard1": 272, "xhard2": 272, "xhard3": 92, "xhard4": 144, "xhard5": 20}
#: Number of rows with error_type == exec_over_cap in the result segments of the packaged specs (measured 0)
PACKAGED_EXEC_OVER_CAP = 0
#: MoveCube xhard4 50 episodes x 2 segments x 3 objects = 300 points; motion modes 0/1/2 quota 17/17/16
MOVECUBE_POINTS = 300
MOVECUBE_WAYS = {0: 17, 1: 17, 2: 16}
#: The three numbers of the MoveCube xhard4 V9 region (1002 plan part one §1 settled item 1) and the pre-change V8 region (used only to count "outside the old region")
MOVECUBE_REGION_V9 = {"r_in": 0.24, "r_out": 0.42, "base_dist": [0.31, 0.80]}
MOVECUBE_REGION_V8 = {"r_in": 0.12, "r_out": 0.20, "base_dist": [0.35, 0.76]}
#: Official metadata: 16 files per split, episode count per file
OFFICIAL_SPLIT_FILES = 16
OFFICIAL_SPLIT_EPISODES = {"train": 100, "val": 50, "test": 50}
#: Generation-side/switch symbols removed from this repo that must not reappear: (module, name)
REMOVED_SYMBOLS = (
    ("hard_specs", "XHARD0_IN_TEST_HARD"), ("hard_specs", "xhard0_prefix"), ("hard_specs", "SPECS_ROOT_ENV"),
    ("hard_builder", "HARD_TRAIN_TASKS"), ("hard_builder", "HARD_METADATA_ROOT"), ("hard_builder", "_HARD_DATASETS"),
    ("hard_builder", "_ALLOWED_ACTION_SPACES"), ("hard_builder", "_override_cells"), ("hard_builder", "_root_specs"),
)
#: Legacy names from before the dataset interface rename (1006 rename plan R1; the builder must reject them). This is the only legacy-name alias table in the repo;
#: ``tests/robomme_ood/static/test_official_names.py`` exempts this block and still catches occurrences outside it.
# >>> LEGACY_NAMES
LEGACY_DATASET_NAMES = ("test-hard0", "test-hard")
# <<< LEGACY_NAMES
#: Challenge interface
CHALLENGE_MAX_STEPS_DEFAULT = 1500
CHALLENGE_ACTION_SHAPES = {"joint_angle": (8,), "ee_pose": (7,), "waypoint": (7,)}
DUMMY_POLICY_CHUNK = 10
DUMMY_POLICY_ACTION_DIM = 8
CLIENT_RETRY_SLEEP_S = 5
#: Recorder (official and hard copy share the same values, except fail_safe_limit)
RECORD_GRIPPER_CLOSE_LT = 0.03
RECORD_OVERLAY_LINE_HEIGHT = 20
RECORD_OVERLAY_PADDING = 10
RECORD_OVERLAY_MIN_HEIGHT = 50
RECORD_FK_NEGATIVE_FINGER = 0.04
RECORD_FAIL_SAFE_LIMIT = {"official": 2000, "hard": 5000}


# =============================================================================
# Shared helpers
# =============================================================================


def hard_specs():
    from robomme_ood.env_record_wrapper import hard_specs as hs

    return hs


def recording_fakes():
    """Recording doubles (T5's ``tests/robomme_ood/pipeline/recording/recording_fakes.py``, reused read-only). Loaded by path under an independent module name;
    the file contains dataclasses, so the module name must be registered before execution."""
    name = "_contract_recording_fakes"
    if name in sys.modules:
        return sys.modules[name]
    path = REPO / "tests" / "robomme_ood" / "pipeline" / "recording" / "recording_fakes.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Delivery cell table, episode counts and switches
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
    """The xhard0 switch, spec-root env var, hard package train metadata and generation-side entries are all removed: the module no longer exports these symbols,
    ``env_metadata`` contains only ``ood``; hard-verify has 16 x 12 = 192 episodes, ood 16 x 50 = 800."""
    from robomme_ood.env_record_wrapper import hard_builder

    modules = {"hard_specs": hard_specs(), "hard_builder": hard_builder}
    present = [f"{m}.{name}" for m, name in REMOVED_SYMBOLS if hasattr(modules[m], name)]
    assert present == []
    meta = REPO / "src" / "robomme_ood" / "env_metadata"
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
    """``EXEC_CAP`` equals the pin 1600; the per-tier step lookup table ``TIER_MAX_STEPS`` is removed and exported by neither module nor package."""
    hs = hard_specs()
    assert hs.EXEC_CAP == EXEC_CAP
    assert not hasattr(hs, "TIER_MAX_STEPS")
    package = importlib.import_module("robomme_ood.env_record_wrapper")
    assert not hasattr(package, "TIER_MAX_STEPS")


def test_recorded_float_tol():
    """``hard_specs.RECORDED_FLOAT_TOL`` equals the pin (the checker's boundary behavior is covered separately by ``tests/robomme_ood/unit/hard/test_episode_spec.py``)."""
    hs = hard_specs()
    assert hs.RECORDED_FLOAT_TOL == RECORDED_FLOAT_TOL
    assert isinstance(hs.RECORDED_FLOAT_TOL, float)


def test_xhard4_only():
    hs = hard_specs()
    assert tuple(hs.XHARD4_ONLY) == XHARD4_ONLY
    assert hs.xhard4_only_tasks(hs.V9_CELLS) == set(XHARD4_ONLY)


# =============================================================================
# Seed formula and per-tier offsets
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
# 16-task list and registration
# =============================================================================


def test_task_list_everywhere_equal():
    hs = hard_specs()
    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder as Official
    from robomme_ood.env_record_wrapper.hard_builder import BenchmarkEnvBuilder as Hard

    assert len(TASKS) == N_TASKS
    assert tuple(hs.ALL_TASKS) == TASKS
    assert tuple(Official.get_task_list()) == TASKS
    assert tuple(Hard.get_task_list()) == TASKS


def test_registered_ids_equal_task_set():
    """After importing robomme_ood, the 16 ids in the registry are exactly the task list and all classes belong to robomme_ood (registry read only, no env built)."""
    import robomme_ood
    from mani_skill.utils.registration import REGISTERED_ENVS

    assert set(robomme_ood.robomme_env.ENV_IDS) == set(TASKS)
    assert len(robomme_ood.robomme_env.ENV_IDS) == N_TASKS
    for uid in TASKS:
        assert REGISTERED_ENVS[uid].cls.__module__.startswith("robomme_ood.")


# =============================================================================
# Challenge interface (constants handed over under R8)
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
    """Probe: connect raises ConnectionRefusedError the first time and succeeds the second; the interval recorded by the time.sleep double is the retry interval."""
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
# Recorder (function-local values, measured with behavioral probes)
# =============================================================================

RECORD_KINDS = ("official", "hard")


def _record_cls(kind):
    return recording_fakes().record_module(kind).RobommeRecordWrapper


@pytest.mark.parametrize("kind", RECORD_KINDS)
def test_overlay_geometry(kind):
    """Probe: frame width 21 -> usable width 1 pixel, each word on its own line; with n lines the text area height = max(min height, n x line height + padding).
    Derived from three measured heights for 1, 4 and 5 lines: line height = h5 - h4, padding = h5 - 5 x line height, min height = h1."""
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
    assert cls._add_text_to_frame(None, frame, "") is frame  # empty text returned unchanged


@pytest.mark.parametrize("kind", RECORD_KINDS)
def test_fk_negative_gripper_finger(kind):
    """Probe: the FK double records the full qpos passed in and then raises (swallowed by the recorder, returning None); a negative command gives both fingers a fixed opening, a positive one uses the command value."""
    cls = _record_cls(kind)
    seen: list[np.ndarray] = []

    class _Pin:
        def compute_forward_kinematics(self, q):
            seen.append(np.asarray(q, dtype=np.float64).copy())
            raise RuntimeError("probe stops here")

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
    """Real one-episode one-step recording: after the double env's step, set both finger positions to ``finger`` (float64) and read is_gripper_close from the h5.
    (The recorder writes no per-step records when save_video=False, so keep save_video=True and only replace mp4 encoding with a no-op.)"""
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
    """Probe: fingers exactly at the threshold are judged open, the previous double below the threshold judged closed -> the threshold is exactly the pin and the comparison is strict less-than."""
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
    """Probe: elapsed_steps = cap - 1 does not trigger, = cap triggers (comparison is >=) -> the cap is exactly the pin. Official 2000, hard copy 5000."""
    limit = RECORD_FAIL_SAFE_LIMIT[kind]
    assert _failsafe_raises(kind, tmp_path, limit - 1) is False
    assert _failsafe_raises(kind, tmp_path, limit) is True


# =============================================================================
# External fixture interface guard
# =============================================================================


def test_recording_fakes_interface_used_here():
    """This file borrows T5's recording doubles by path; if their interface changes, the recorder probes above would fail in obscure ways.
    So name every interface element this file uses up front, and report "T5 fixture interface changed" directly on change."""
    import dataclasses
    import inspect

    rf = recording_fakes()
    needed = ("record_module", "make_wrapper", "drive", "Event")
    missing = [name for name in needed if not hasattr(rf, name)]
    assert not missing, f"T5 fixture interface changed: recording_fakes missing {missing}"
    event_fields = {f.name for f in dataclasses.fields(rf.Event)}
    lost = {"name", "terminated", "success", "elapsed"} - event_fields
    assert not lost, f"T5 fixture interface changed: Event missing fields {sorted(lost)}"
    params = inspect.signature(rf.make_wrapper).parameters
    assert "save_video" in params, "T5 fixture interface changed: make_wrapper no longer accepts save_video"
    for kind in RECORD_KINDS:
        assert hasattr(rf.record_module(kind), "RobommeRecordWrapper"), f"T5 fixture interface changed: record_module({kind!r})"
