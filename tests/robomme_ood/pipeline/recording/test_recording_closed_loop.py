"""C10/C11 recording and read-back closed loop: independent event table → CPU stand-in env → real RobommeRecordWrapper.step/close
→ h5py → real EpisodeDatasetResolver and dataset_replay._build_action_sequence.

Both RecordWrapper classes, of official ``robomme`` and of the hard package's copy ``robomme_ood``, run the same closed loop (parametrized).
All expected values come from the event tables and hand computation written in this file; the logic under test is never used to generate expectations.
"""
from __future__ import annotations

import json
import math

import h5py
import numpy as np
import pytest
import torch

from recording_fakes import (
    ACTION_KEYS,
    CHOICE_POINT_YX,
    INFO_KEYS,
    INTRINSIC,
    OBS_KEYS,
    SETUP_KEYS_BASE,
    TIMESTEP_GROUPS,
    WRIST_INTRINSIC,
    EXTRINSIC,
    WRIST_EXTRINSIC,
    Event,
    action_of,
    drive,
    FINGER_CLOSED,
    FINGER_OPEN,
    front_depth,
    front_rgb,
    make_wrapper,
    read_tree,
    record_module,
    run_episode,
    tcp_xyz,
    trees_diff,
    wrist_depth,
    wrist_rgb,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*to get variables from other wrappers")

KINDS = ["official", "hard"]

W1 = dict(waypoint_p=[0.1, 0.2, 0.3], waypoint_q=[1.0, 0.0, 0.0, 0.0], waypoint_type="close", waypoint_phase_is_demo=True)
W2_CROSS = dict(waypoint_p=[9.0, 9.0, 9.0], waypoint_q=[1.0, 0.0, 0.0, 0.0], waypoint_type="open", waypoint_phase_is_demo=False)
_H = math.sqrt(0.5)
W3 = dict(waypoint_p=[0.4, -0.1, 0.25], waypoint_q=[_H, 0.0, 0.0, _H], waypoint_type="open", waypoint_phase_is_demo=False)
# By hand: W1 identity quaternion → rpy 0, close → gripper -1; W3 90° about z → yaw = π/2, open → gripper +1.
W1_ACTION = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0])
W3_ACTION = np.array([0.4, -0.1, 0.25, 0.0, 0.0, math.pi / 2, 1.0])

GROUNDED = "pick the cube at <14, 24>"  # integer mean of the segmentation square rows 10..19, cols 20..29

# Main scene event table (8 steps) and hand-written expectations per record.
MAIN_EVENTS = [
    Event(name="NO RECORD", demo=True, task_index=0),  # t=1 not recorded
    Event(name="watch", demo=True, task_index=0, waypoint=W1),  # t=2 → rec0
    Event(name="watch", demo=True, task_index=1, waypoint=W2_CROSS, choice_text="press the button"),  # t=3 → rec1
    Event(name="pick", demo=False, task_index=1, pre_demo=False, choice_text="press the button"),  # t=4 → rec2
    Event(name="pick", demo=False, task_index=2, waypoint=W3, choice_text="unknown action"),  # t=5 → rec3
    Event(name="pick", demo=False, task_index=2, choice_text="unknown action"),  # t=6 → rec4
    Event(name="NO RECORD", demo=False, task_index=2, choice_text="unknown action"),  # t=7 not recorded
    Event(name="pick", demo=False, task_index=3, choice_text="press the button", terminated=True, success=True),  # t=8 → rec5
]
NAN7 = np.full(7, np.nan)
# Each record: (from which step, simple_subgoal, is_video_demo, is_subgoal_boundary, is_completed, waypoint, choice, gripper open)
# Gripper column follows the event table: action_of(t) sends open on odd steps and close on even steps.
MAIN_EXPECTED = [
    (2, "watch", True, False, False, W1_ACTION, "", False),
    (3, "watch", True, True, False, W1_ACTION, "B", True),
    (4, "pick", False, False, False, NAN7, "B", False),  # outer layer switches to online → cache cleared
    (5, "pick", False, True, False, W3_ACTION, "", True),
    (6, "pick", False, False, False, W3_ACTION, "", False),
    (8, "pick", False, True, True, W3_ACTION, "B", False),
]
OPTIONS = [{"label": "b", "action": "press the button", "available": None}]


@pytest.fixture(autouse=True)
def no_encode(monkeypatch):
    """Daily gate does not call ffmpeg: per-episode mp4 writing is replaced with bookkeeping only (encoding itself is tested in the slow tests)."""
    written = []

    def fake_write(self, frames, output_path):
        written.append((output_path.name, len(frames)))

    for k in KINDS:
        monkeypatch.setattr(record_module(k).RobommeRecordWrapper, "_video_write_mp4", fake_write)
    return written


@pytest.fixture(params=KINDS)
def kind(request):
    return request.param


@pytest.fixture
def mod(kind):
    return record_module(kind)


@pytest.fixture
def vqa_patched(mod, monkeypatch):
    """Option builder stand-in: the recorder gets options via the module name get_vqa_options in step, and via each package's vqa_options module in close."""
    import importlib

    fake = lambda env, planner, target, env_id: [dict(o) for o in OPTIONS]  # noqa: E731
    monkeypatch.setattr(mod, "get_vqa_options", fake)
    pkg = mod.__name__.split(".")[0]
    monkeypatch.setattr(importlib.import_module(f"{pkg}.robomme_env.utils.vqa_options"), "get_vqa_options", fake)
    return fake


def _main_episode(mod, tmp_path):
    return run_episode(mod.RobommeRecordWrapper, tmp_path, MAIN_EVENTS)


def test_closed_loop_h5_matches_event_table(mod, vqa_patched, tmp_path):
    w, env, path, _ = _main_episode(mod, tmp_path)
    assert path.name == "FakeTask_ep3_seed77.h5" and path.parent.name == "hdf5_files"
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"episode_3"}
        ep = f["episode_3"]
        ts_keys = sorted(k for k in ep if k.startswith("timestep_"))
        assert set(ep.keys()) == set(ts_keys) | {"setup"}
        # Neither the reset frame nor NO RECORD steps are recorded: 8 steps → 6 records, numbered contiguously from 0
        assert sorted(ts_keys, key=lambda k: int(k.split("_")[1])) == [f"timestep_{i}" for i in range(len(MAIN_EXPECTED))]
        for i, (t, name, demo, boundary, done, wp, choice, opened) in enumerate(MAIN_EXPECTED):
            g = ep[f"timestep_{i}"]
            assert set(g.keys()) == TIMESTEP_GROUPS
            assert set(g["obs"].keys()) == OBS_KEYS
            assert set(g["action"].keys()) == ACTION_KEYS
            assert set(g["info"].keys()) == INFO_KEYS
            act = action_of(t)
            # action_t and obs_after_t are in the same record: the step index encoded in the image = t, joint reading = action_t[:7]
            np.testing.assert_array_equal(g["obs/front_rgb"][()], front_rgb(t))
            np.testing.assert_array_equal(g["obs/wrist_rgb"][()], wrist_rgb(t))
            np.testing.assert_array_equal(g["obs/front_depth"][()], front_depth(t))
            np.testing.assert_array_equal(g["obs/wrist_depth"][()], wrist_depth(t))
            assert g["obs/front_rgb"].dtype == np.uint8 and g["obs/front_depth"].dtype == np.int16
            np.testing.assert_array_equal(g["action/joint_action"][()], act)
            np.testing.assert_allclose(g["obs/joint_state"][()], act[:7].astype(np.float32))
            finger = FINGER_OPEN if opened else FINGER_CLOSED
            np.testing.assert_allclose(g["obs/gripper_state"][()], [finger, finger])
            assert bool(g["obs/is_gripper_close"][()]) is (not opened)
            np.testing.assert_allclose(g["obs/eef_state"][()], tcp_xyz(t) + [0.0, 0.0, 0.0], atol=1e-6)
            assert g["obs/eef_state"].dtype == np.float32
            np.testing.assert_array_equal(g["obs/front_camera_extrinsic"][()], EXTRINSIC)
            np.testing.assert_array_equal(g["obs/wrist_camera_extrinsic"][()], WRIST_EXTRINSIC)
            # FK unavailable (stand-in has no robot model) → eef_action is always 7-dim zeros
            np.testing.assert_array_equal(g["action/eef_action"][()], np.zeros(7))
            np.testing.assert_allclose(g["action/waypoint_action"][()], wp, atol=1e-6, equal_nan=True)
            payload = json.loads(g["action/choice_action"][()])
            assert payload == {"choice": choice, "point": CHOICE_POINT_YX}
            info = g["info"]
            assert info["simple_subgoal"][()].decode() == name
            assert info["simple_subgoal_online"][()].decode() == "online pick"
            assert info["grounded_subgoal"][()].decode() == GROUNDED
            assert info["grounded_subgoal_online"][()].decode() == GROUNDED
            assert bool(info["is_video_demo"][()]) is demo
            assert bool(info["is_subgoal_boundary"][()]) is boundary
            assert bool(info["is_completed"][()]) is done
        setup = ep["setup"]
        assert set(setup.keys()) == SETUP_KEYS_BASE  # FakeTask has no language goal → task_goal not written
        assert int(setup["seed"][()]) == 77
        assert setup["difficulty"][()].decode() == "easy"
        np.testing.assert_array_equal(setup["front_camera_intrinsic"][()], INTRINSIC)
        np.testing.assert_array_equal(setup["wrist_camera_intrinsic"][()], WRIST_INTRINSIC)
        assert json.loads(setup["available_multi_choices"][()]) == [
            {"label": "b", "action": "press the button", "need_parameter": False}
        ]
    # The actions the env receives are exactly those the caller sent, same count and order
    assert [np.asarray(a).tolist() for a in env.received_actions] == [action_of(t).tolist() for t in range(1, 9)]


def test_root_and_group_attrs_are_empty(mod, tmp_path):
    """The recorder writes no h5 attributes; any extra attribute on the root or a group is format drift (mutant M14: change root attribute)."""
    _, _, path, _ = _main_episode(mod, tmp_path)
    tree = read_tree(path)
    attrs = {k: v for k, v in tree.items() if k.endswith("@attrs") and v}
    assert attrs == {}


def test_recorded_actions_finite_and_bit_exact(mod, tmp_path):
    """joint_action/eef_action equal the input bit for bit and are all finite; waypoint is either finite 7-dim or an all-NaN sentinel (mutant M14: write non-finite actions)."""
    _, _, path, _ = _main_episode(mod, tmp_path)
    with h5py.File(path, "r") as f:
        ep = f["episode_3"]
        for i, exp in enumerate(MAIN_EXPECTED):
            g = ep[f"timestep_{i}/action"]
            ja = g["joint_action"][()]
            assert ja.dtype == np.float64 and ja.shape == (8,)
            assert np.all(np.isfinite(ja)) and ja.tobytes() == action_of(exp[0]).tobytes()
            assert np.all(np.isfinite(g["eef_action"][()]))
            wp = g["waypoint_action"][()]
            assert wp.shape == (7,) and (np.all(np.isfinite(wp)) or np.all(np.isnan(wp)))


def test_resolver_reads_recorded_episode(mod, vqa_patched, tmp_path):
    """Real write → real EpisodeDatasetResolver: sequences of the four action spaces equal the hand-computed event table results."""
    from robomme.env_record_wrapper import EpisodeDatasetResolver

    _, _, path, _ = _main_episode(mod, tmp_path)
    online = [e for e in MAIN_EXPECTED if not e[2]]  # non-demonstration records
    with EpisodeDatasetResolver("FakeTask", 3, path) as r:
        joints = _drain(r, "joint_angle")
        ees = _drain(r, "ee_pose")
        wps = _drain(r, "waypoint")
        mcs = _drain(r, "multi_choice")
    assert [j.tolist() for j in joints] == [action_of(e[0]).tolist() for e in online]
    assert [e.tolist() for e in ees] == [[0.0] * 7] * len(online)
    # waypoints of non-demonstration records: NaN, W3, W3, W3 → skip sentinel and dedupe adjacent, only W3 remains
    assert len(wps) == 1 and np.allclose(wps[0], W3_ACTION, atol=1e-6)
    # non-demonstration records that are subgoal boundaries: rec3 (empty option, filtered), rec5 (B)
    assert mcs == [{"choice": "B", "point": CHOICE_POINT_YX}]


def _drain(r, mode):
    out, i = [], 0
    while (x := r.get_step(mode, i)) is not None:
        out.append(x)
        i += 1
    return out


@pytest.fixture(scope="module")
def replay_mod():
    from replay_loader import load_replay

    return load_replay()


def test_replay_sequence_equals_resolver_on_recorded_episode(mod, vqa_patched, tmp_path, replay_mod):
    from robomme.env_record_wrapper import EpisodeDatasetResolver

    _, _, path, _ = _main_episode(mod, tmp_path)
    with h5py.File(path, "r") as f, EpisodeDatasetResolver("FakeTask", 3, path) as r:
        ep = f["episode_3"]
        for mode in ("joint_angle", "ee_pose", "waypoint", "multi_choice"):
            seq = replay_mod._build_action_sequence(ep, mode)
            res = _drain(r, mode)
            assert len(seq) == len(res), mode
            for a, b in zip(seq, res):
                if isinstance(a, dict):
                    assert a == b
                else:
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)


def test_official_and_hard_record_identical_h5(tmp_path, monkeypatch):
    """For the same event table, below both safety limits, the h5 written by official and the hard copy are identical key by key, value by value."""
    import importlib

    fake = lambda env, planner, target, env_id: [dict(o) for o in OPTIONS]  # noqa: E731
    for k in KINDS:
        m = record_module(k)
        monkeypatch.setattr(m, "get_vqa_options", fake)
        pkg = m.__name__.split(".")[0]
        monkeypatch.setattr(importlib.import_module(f"{pkg}.robomme_env.utils.vqa_options"), "get_vqa_options", fake)
    trees = {}
    for k in KINDS:
        _, _, path, _ = run_episode(record_module(k).RobommeRecordWrapper, tmp_path / k, MAIN_EVENTS)
        trees[k] = read_tree(path)
    assert trees_diff(trees["official"], trees["hard"]) == []
    # negative case: the checker can see a single difference
    other = dict(trees["hard"])
    key = "episode_3/timestep_0/action/joint_action"
    dt, shape, val = other[key]
    other[key] = (dt, shape, val + 1.0)
    assert trees_diff(trees["official"], other) == [f"value differs {key}"]
    # negative cases: missing key, attrs differ, dtype differs
    missing = dict(trees["hard"])
    del missing[key]
    assert trees_diff(trees["official"], missing) == [f"missing key {key}"]
    attrs = dict(trees["hard"])
    attrs["/@attrs"] = {"x": 1}
    assert trees_diff(trees["official"], attrs) == ["attrs differ /@attrs"]
    dtyped = dict(trees["hard"])
    dtyped[key] = ("float32", shape, val.astype(np.float32))
    assert len(trees_diff(trees["official"], dtyped)) == 1 and trees_diff(trees["official"], dtyped)[0].startswith("dtype/shape differs")


def test_hard_copy_reads_its_own_vqa_options(tmp_path, monkeypatch):
    """The only import difference between the two copies: when close writes available_multi_choices each reads its own package's vqa_options."""
    import importlib

    hard_vqa = importlib.import_module("robomme_ood.robomme_env.utils.vqa_options")
    off_vqa = importlib.import_module("robomme.robomme_env.utils.vqa_options")
    assert hard_vqa is not off_vqa
    monkeypatch.setattr(hard_vqa, "get_vqa_options", lambda *a: [{"label": "z", "action": "hard only", "available": [1]}])
    got = {}
    for k in KINDS:
        _, _, path, _ = run_episode(record_module(k).RobommeRecordWrapper, tmp_path / k, MAIN_EVENTS)
        with h5py.File(path, "r") as f:
            got[k] = json.loads(f["episode_3/setup/available_multi_choices"][()])
    assert got["hard"] == [{"label": "z", "action": "hard only", "need_parameter": True}]
    assert got["official"] == []  # FakeTask uses the official default builder: no options


# ---------------------------------------------------------------- success verdicts do not substitute for each other

SCENARIOS = {
    # name: (sequence of (terminated, truncated, success, task_index) of the event table, whether h5 is written)
    "terminated_and_success": ([(False, False, False, 0), (True, False, True, 0)], True),
    "terminated_but_failed": ([(False, False, False, 0), (True, False, False, 3)], False),
    "all_subgoals_done_not_terminated": ([(False, False, False, 1), (False, False, False, 3)], False),
    "info_success_on_truncation": ([(False, False, False, 0), (False, True, True, 0)], False),
    "info_success_not_terminated": ([(False, False, True, 0), (False, False, True, 0)], False),
    "success_then_failed_termination": ([(True, False, True, 0), (True, False, False, 0)], False),
    "failed_then_success_termination": ([(True, False, False, 0), (True, False, True, 0)], True),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_success_signals_do_not_substitute(mod, tmp_path, scenario):
    seq, written = SCENARIOS[scenario]
    events = [Event(name="pick", task_index=ti, terminated=te, truncated=tr, success=su) for te, tr, su, ti in seq]
    w, env, path, rets = run_episode(mod.RobommeRecordWrapper, tmp_path, events)
    with h5py.File(path, "r") as f:
        assert ("episode_3" in f) is written
        if written:
            done = [bool(f[f"episode_3/timestep_{i}/info/is_completed"][()]) for i in range(len(seq))]
            assert done == [ti >= 3 for *_, ti in seq]  # is_completed only looks at subgoal progress
    # the wrapper passes through the env's terminated/truncated/info unchanged
    for (te, tr, su, _), (_, _, ter, trn, info) in zip(seq, rets):
        assert bool(ter.item()) is te and bool(trn.item()) is tr and bool(info["success"].item()) is su
        assert "failsafe_elapsed_steps" not in info


def test_failed_episode_removes_preexisting_group(mod, tmp_path):
    """When the same h5 file already has an episode group with the same id: a successful episode replaces the whole group, a failed one deletes the old group."""
    ok = [Event(name="pick", terminated=True, success=True)]
    run_episode(mod.RobommeRecordWrapper, tmp_path, ok + ok)  # two records
    run_episode(mod.RobommeRecordWrapper, tmp_path, ok)  # rewrite the same file: only one left
    path = tmp_path / "out" / "hdf5_files" / "FakeTask_ep3_seed77.h5"
    with h5py.File(path, "r") as f:
        assert sorted(f["episode_3"].keys()) == ["setup", "timestep_0"]
    run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=False)])
    with h5py.File(path, "r") as f:
        assert "episode_3" not in f


def test_corrupted_h5_is_recreated(mod, tmp_path):
    out = tmp_path / "out" / "hdf5_files"
    out.mkdir(parents=True)
    (out / "FakeTask_ep3_seed77.h5").write_bytes(b"not an hdf5 file")
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=True)])
    with h5py.File(path, "r") as f:
        assert list(f.keys()) == ["episode_3"]


def test_dataset_path_forms(mod, tmp_path):
    """When dataset is an .h5 file path, the output directory is <parent>/<stem>_hdf5_files; missing dataset is rejected outright."""
    env_ev = [Event(name="pick", terminated=True, success=True)]
    from recording_fakes import FakeTaskEnv

    w = mod.RobommeRecordWrapper(FakeTaskEnv(env_ev), dataset=str(tmp_path / "x" / "rec.h5"), env_id="E", episode=1, seed=2)
    assert w.dataset_path == (tmp_path / "x" / "rec_hdf5_files" / "E_ep1_seed2.h5").resolve()
    w.close()
    with pytest.raises(ValueError):
        mod.RobommeRecordWrapper(FakeTaskEnv(env_ev), dataset=None, env_id="E", episode=1, seed=2)


# ---------------------------------------------------------------- action forms


def test_seven_dim_action_padded_and_tensor_converted(mod, tmp_path):
    """7-dim (stick) actions get a gripper placeholder -1 when written; CPU Tensors are converted to NumPy when written."""
    acts = [torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1], dtype=torch.float64)]
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=True)], actions=acts)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["episode_3/timestep_0/action/joint_action"][()], [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -1.0])


def test_none_action_written_as_string_and_resolver_skips_it(mod, tmp_path):
    from robomme.env_record_wrapper import EpisodeDatasetResolver

    events = [Event(name="pick"), Event(name="pick", terminated=True, success=True)]
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, events, actions=[None, torch.from_numpy(action_of(2))])
    with h5py.File(path, "r") as f:
        assert f["episode_3/timestep_0/action/joint_action"][()] in (b"None", "None")
    with EpisodeDatasetResolver("FakeTask", 3, path) as r:
        assert r.get_step("joint_angle", 0) is None
        np.testing.assert_array_equal(r.get_step("joint_angle", 1), action_of(2))


# ---------------------------------------------------------------- FK


def test_fk_failure_path_keeps_recording(mod, tmp_path):
    w, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=True)])
    assert w._fk_available is False and w._mplib_planner is None
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["episode_3/timestep_0/action/eef_action"][()], np.zeros(7))


class _CpuPinocchio:
    """FK stand-in: end-effector pose = first three joint values as position, identity quaternion; records the full qpos passed in."""

    def __init__(self):
        self.calls = []

    def compute_forward_kinematics(self, q):
        self.calls.append(np.asarray(q, dtype=np.float64).copy())

    def get_link_pose(self, idx):
        q = self.calls[-1]
        return np.array([q[0], q[1], q[2], 1.0, 0.0, 0.0, 0.0])


def test_fk_success_path_with_cpu_stub(mod, tmp_path, monkeypatch):
    import sapien

    pin = _CpuPinocchio()

    def fake_init(self):
        self.planner = None
        self._mplib_planner = type("P", (), {"pinocchio_model": pin})()
        self._ee_link_idx = 0
        self._robot_base_pose = sapien.Pose()
        self._fk_qpos_size = 9
        self._fk_available = True

    monkeypatch.setattr(mod.RobommeRecordWrapper, "_init_fk_planner", fake_init)
    # four gripper commands: two different positive values, two different negative values
    grips = [1.0, 0.5, -1.0, -0.5]
    acts = [np.concatenate([action_of(t + 1)[:7], [g]]) for t, g in enumerate(grips)]
    events = [Event(name="pick")] * 3 + [Event(name="pick", terminated=True, success=True)]
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, events, actions=[torch.from_numpy(a) for a in acts])
    with h5py.File(path, "r") as f:
        for i, a in enumerate(acts):
            np.testing.assert_allclose(
                f[f"episode_3/timestep_{i}/action/eef_action"][()], [a[0], a[1], a[2], 0, 0, 0, a[7]], atol=1e-6
            )
    assert [c[:7].tolist() for c in pin.calls] == [a[:7].tolist() for a in acts]
    fingers = [c[7:].tolist() for c in pin.calls]
    assert all(f[0] == f[1] for f in fingers)  # both fingers equal
    # positive command: finger position equals the command value
    assert [fingers[0][0], fingers[1][0]] == [1.0, 0.5]
    # negative command: a fixed opening independent of the command magnitude, different from any positive-command result
    assert fingers[2] == fingers[3]
    assert fingers[2][0] not in (1.0, 0.5) and fingers[2][0] >= 0.0


# ---------------------------------------------------------------- safety limit


def _probe_failsafe(cls, tmp_path, elapsed: int) -> bool:
    w, env = make_wrapper(cls, tmp_path / f"p{elapsed}", [Event(name="pick", elapsed=elapsed)])
    w.reset()
    try:
        drive(w, env, [Event(name="pick", elapsed=elapsed)])
        return False
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ == "FailsafeTimeout"
        return True
    finally:
        w.h5_file.close()


def _threshold(cls, tmp_path) -> int:
    lo, hi = 0, 1
    while not _probe_failsafe(cls, tmp_path, hi):
        lo, hi = hi, hi * 2
        assert hi < 1 << 20, "safety limit does not exist"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _probe_failsafe(cls, tmp_path, mid):
            hi = mid
        else:
            lo = mid
    return hi


def test_failsafe_threshold_differs_only_in_hard_copy(tmp_path):
    """Behaviorally measured safety limits of both classes: the hard copy is strictly larger (only relaxed, never tightened); exact values pinned by the constants layer."""
    off = _threshold(record_module("official").RobommeRecordWrapper, tmp_path / "o")
    hard = _threshold(record_module("hard").RobommeRecordWrapper, tmp_path / "h")
    assert 0 < off < hard


def test_failsafe_raises_once_then_truncates_and_drops_success(mod, tmp_path):
    """Exceeding the limit: raises FailsafeTimeout the first time; afterwards each step returns truncated=True, terminated=False, and success does not write h5."""
    lim = _threshold(mod.RobommeRecordWrapper, tmp_path / "thr")
    events = [Event(name="pick", elapsed=lim - 1), Event(name="pick", elapsed=lim),
              Event(name="pick", elapsed=lim + 1, terminated=True, success=True)]
    w, env = make_wrapper(mod.RobommeRecordWrapper, tmp_path, events)
    w.reset()
    env.events = events
    w.step(torch.from_numpy(action_of(1)))
    with pytest.raises(mod.FailsafeTimeout):
        w.step(torch.from_numpy(action_of(2)))
    _, _, ter, trn, info = w.step(torch.from_numpy(action_of(3)))
    assert bool(trn.item()) is True and bool(ter.item()) is False
    assert info["TimeLimit.truncated"] is True and info["failsafe_elapsed_steps"] == lim + 1
    w.close()
    with h5py.File(w.dataset_path, "r") as f:
        assert "episode_3" not in f
    # re-armed after reset: the same wrapper raises again when exceeding the limit
    w2, env2 = make_wrapper(mod.RobommeRecordWrapper, tmp_path / "again", events)
    for _ in range(2):
        w2.reset()
        env2.events, env2.counter = [Event(name="pick", elapsed=lim)], 0
        with pytest.raises(mod.FailsafeTimeout):
            w2.step(torch.from_numpy(action_of(1)))
    w2.h5_file.close()


# ---------------------------------------------------------------- cross-episode cache and close


def test_reset_clears_waypoint_and_boundary_caches(mod, tmp_path):
    """reset clears the waypoint cache and subgoal boundary memory: the first record of the second episode has no waypoint from the previous episode and is a boundary."""
    ep1 = [Event(name="pick", task_index=0, waypoint=W3)]
    ep2 = [Event(name="pick", task_index=0, terminated=True, success=True)]
    w, env = make_wrapper(mod.RobommeRecordWrapper, tmp_path, ep1)
    w.reset()
    drive(w, env, ep1)
    assert w._current_waypoint_action is not None
    env.events = ep2
    w.reset()
    assert w._current_waypoint_action is None and w._prev_task_index == -1
    w.buffer.clear()  # see the next test: reset does not clear buffer
    drive(w, env, ep2)
    w.close()
    with h5py.File(w.dataset_path, "r") as f:
        g = f["episode_3/timestep_0"]
        assert np.all(np.isnan(g["action/waypoint_action"][()]))
        assert bool(g["info/is_subgoal_boundary"][()]) is True


def test_reset_without_close_keeps_buffer_and_success_flag(mod, tmp_path):
    """Current-behavior record (frozen code, test only): reset does not clear buffer or episode_success;
    when the same wrapper starts a second episode without close, the previous episode's records and success flag leak into the second."""
    ep1 = [Event(name="pick", terminated=True, success=True)]
    ep2 = [Event(name="pick")]  # second episode never terminates
    w, env = make_wrapper(mod.RobommeRecordWrapper, tmp_path, ep1)
    w.reset()
    drive(w, env, ep1)
    env.events = ep2
    w.reset()
    assert w.episode_success is True  # previous episode's success flag carried into the second
    drive(w, env, ep2, first_t=2)
    assert w.episode_success is True
    w.close()
    with h5py.File(w.dataset_path, "r") as f:
        ep = f["episode_3"]
        assert sorted(k for k in ep if k.startswith("timestep_")) == ["timestep_0", "timestep_1"]
        np.testing.assert_array_equal(ep["timestep_0/action/joint_action"][()], action_of(1))


def test_second_close_after_failed_episode_is_noop(mod, tmp_path):
    w, env, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=False)])
    w.close()
    assert env.close_calls == 2
    with h5py.File(path, "r") as f:
        assert list(f.keys()) == []


def test_second_close_after_success_raises_but_keeps_file(mod, tmp_path):
    """Current-behavior record: closing a successful episode twice raises ValueError on the second close when creating a group in the closed file; the already-written h5 is unaffected."""
    w, env, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=True)])
    before = read_tree(path)
    with pytest.raises(ValueError):
        w.close()
    assert trees_diff(before, read_tree(path)) == []
    assert w.buffer == [] and w.video_frames == []


def test_save_video_false_records_no_timesteps(mod, tmp_path):
    """Current-behavior record: the per-step H5 cache is written inside the video branch, so with save_video=False a successful episode only has setup and no timestep.
    All production entries pass save_video=True (see parity/train_split_worker and official generate_dataset)."""
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=True)], save_video=False)
    with h5py.File(path, "r") as f:
        assert list(f["episode_3"].keys()) == ["setup"]
    assert not (tmp_path / "out" / "videos").exists()
