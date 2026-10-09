"""C11 read-back diff: the same hand-written h5 fixture is fed both to the real EpisodeDatasetResolver and to the upstream entry
``scripts/dataset_replay.py::_build_action_sequence``, and both are compared with hand-written expectations;
then a stand-in env drives the real ``process_episode`` to check action order, error output, terminal state and result label.

The fixture deliberately contains: timestep keys inserted out of order including 10/11 (lexicographic differs from numeric order), missing ids, demonstration steps,
adjacent and separated duplicate waypoints, NaN sentinels, waypoints of wrong shape, and various invalid choices.
Known behavioral divergences between the two sides are separate "current-behavior record" tests (the upstream file is frozen; test only, no changes).
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from robomme.env_record_wrapper import EpisodeDatasetResolver, list_episode_indices

pytestmark = pytest.mark.filterwarnings("ignore:.*to get variables from other wrappers")

MODES = ("joint_angle", "ee_pose", "waypoint", "multi_choice")
A = [0.25, 0.5, 0.75, 0.0, 0.0, 1.5, 1.0]
B = [1.25, -0.5, 0.25, 0.0, 0.5, 0.0, -1.0]
NAN = [float("nan")] * 7


def J(n):  # joint action of timestep n; multiples of 1/4, exactly representable in float32
    return [n + 0.25 * j for j in range(7)] + [1.0 if n % 2 else -1.0]


def E(n):
    return [n * 0.5, -n * 0.25, 0.5, 0.0, 0.25, 0.75, -1.0]


def C(choice, point=None, *, raw=None):
    if raw is not None:
        return raw
    d = {"choice": choice}
    if point is not None:
        d["point"] = point
    return json.dumps(d)


# (timestep id, is_video_demo, is_subgoal_boundary, waypoint, original choice_action); written in this shuffled order
ROWS = [
    (10, False, True, A, C("C")),  # missing point → option filtered
    (0, True, True, A, C("A", [1, 2])),  # demonstration step → skipped in all modes
    (2, False, False, A, C("X", [0, 0])),  # same waypoint as previous → deduped; not a boundary → no option
    (11, False, True, A, C("D", [5, 6])),
    (1, False, True, A, C("B", [3, 4])),
    (3, False, True, NAN, C("", raw="{bad json")),  # NaN sentinel → skipped; bad JSON → filtered
    (5, False, True, B, C("  ", [0, 0])),  # blank choice → filtered; id 4 missing
    (12, False, False, [1.0] * 6, C("", raw="[]")),  # wrong waypoint shape → skipped
]
ONLINE = [1, 2, 3, 5, 10, 11, 12]
EXPECTED = {
    "joint_angle": [J(n) for n in ONLINE],
    "ee_pose": [E(n) for n in ONLINE],
    # numeric order 1,2,3,5,10,11: A, (A adjacent duplicate), (NaN), B, A (differs from previous, kept), (A adjacent duplicate)
    "waypoint": [A, B, A],
    "multi_choice": [{"choice": "B", "point": [3, 4]}, {"choice": "D", "point": [5, 6]}],
}


def write_episode(path, rows=ROWS, *, episode=0, goal="pick it", joint=J):
    with h5py.File(path, "a") as f:
        ep = f.create_group(f"episode_{episode}")
        for n, demo, boundary, wp, choice in rows:
            g = ep.create_group(f"timestep_{n}")
            a = g.create_group("action")
            if joint is not None:
                a.create_dataset("joint_action", data=np.asarray(joint(n), dtype=np.float64))
            a.create_dataset("eef_action", data=np.asarray(E(n), dtype=np.float64))
            a.create_dataset("waypoint_action", data=np.asarray(wp, dtype=np.float64))
            a.create_dataset("choice_action", data=choice, dtype=h5py.special_dtype(vlen=str))
            i = g.create_group("info")
            i.create_dataset("is_video_demo", data=demo)
            i.create_dataset("is_subgoal_boundary", data=boundary)
        s = ep.create_group("setup")
        s.create_dataset("task_goal", data=np.asarray([goal], dtype=object), dtype=h5py.string_dtype("utf-8"))
    return path


@pytest.fixture
def h5_path(tmp_path):
    return write_episode(tmp_path / "record_dataset_FakeTask.h5")


@pytest.fixture(scope="module")
def replay():
    from replay_loader import load_replay

    return load_replay()


def drain(r, mode):
    out, i = [], 0
    while (x := r.get_step(mode, i)) is not None:
        out.append(x)
        i += 1
    return out


def same_sequence(got, want) -> bool:
    """Checker: item-wise equality (arrays compared bit for bit as float64, dicts compared by value)."""
    if len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if isinstance(w, dict):
            if g != w:
                return False
        elif not np.array_equal(np.asarray(g, dtype=np.float64), np.asarray(w, dtype=np.float64)):
            return False
    return True


def test_same_sequence_judge_has_teeth():
    assert same_sequence(EXPECTED["waypoint"], [A, B, A])
    assert not same_sequence(EXPECTED["waypoint"], [A, B])  # global dedupe
    assert not same_sequence(EXPECTED["waypoint"], [A, B, B])
    assert not same_sequence(EXPECTED["multi_choice"], [{"choice": "B", "point": [3, 4]}])


@pytest.mark.parametrize("mode", MODES)
def test_resolver_and_replay_agree_with_handwritten(mode, h5_path, replay):
    with EpisodeDatasetResolver("FakeTask", 0, h5_path.parent) as r:
        res = drain(r, mode)
    with h5py.File(h5_path, "r") as f:
        seq = replay._build_action_sequence(f["episode_0"], mode)
    assert same_sequence(res, EXPECTED[mode]), mode
    assert same_sequence(seq, EXPECTED[mode]), mode
    if mode != "multi_choice":
        assert all(np.asarray(x).dtype == np.float32 for x in seq)  # upstream replay converts everything to float32


def test_replay_rejects_unknown_mode(h5_path, replay):
    with h5py.File(h5_path, "r") as f, pytest.raises(ValueError):
        replay._build_action_sequence(f["episode_0"], "teleport")


def test_resolver_api_edges(h5_path, tmp_path):
    with h5py.File(h5_path, "a") as f:
        f.create_group("episode_7")
        f.create_group("not_an_episode")
    assert list_episode_indices("FakeTask", h5_path.parent) == [0, 7]
    assert list_episode_indices("Other", h5_path) == [0, 7]  # env_id ignored when a full .h5 path is passed
    with pytest.raises(FileNotFoundError):
        list_episode_indices("Missing", tmp_path)
    with pytest.raises(FileNotFoundError):
        EpisodeDatasetResolver("Missing", 0, tmp_path)
    with pytest.raises(KeyError):
        EpisodeDatasetResolver("FakeTask", 99, h5_path)
    with h5py.File(h5_path, "a"):  # file closed before raising KeyError → can be reopened in write mode
        pass
    r = EpisodeDatasetResolver("FakeTask", 0, h5_path)
    assert r.get_step("joint_angle", -1) is None
    assert r.get_step("teleport", 0) is None
    assert r.get_step("joint_angle", len(ONLINE)) is None
    assert r.get_step("multi_choice", 2) is None
    first = r.get_step("multi_choice", 0)
    first["choice"] = "mutated"
    assert r.get_step("multi_choice", 0)["choice"] == "B"  # returns a copy
    r.close()
    r.close()  # idempotent


# ---------------------------------------------------------------- known divergences (current-behavior records)


def test_divergence_missing_joint_field(tmp_path, replay):
    """Non-demonstration step missing joint_action: resolver keeps the step position and returns None; replay skips it and later indices shift forward."""
    rows = [(0, False, False, A, "{}"), (1, False, False, A, "{}")]
    p = write_episode(tmp_path / "x.h5", rows, joint=lambda n: J(n))
    with h5py.File(p, "a") as f:
        del f["episode_0/timestep_0/action/joint_action"]
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert r.get_step("joint_angle", 0) is None
        assert same_sequence([r.get_step("joint_angle", 1)], [J(1)])
    with h5py.File(p, "r") as f:
        assert same_sequence(replay._build_action_sequence(f["episode_0"], "joint_angle"), [J(1)])


def test_divergence_none_string_action(tmp_path, replay):
    """The recorder writes a None action as the string "None": resolver reads it as None, replay raises ValueError when converting to float32."""
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}")], joint=None)
    with h5py.File(p, "a") as f:
        f["episode_0/timestep_0/action"].create_dataset("joint_action", data="None", dtype=h5py.special_dtype(vlen=str))
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert r.get_step("joint_angle", 0) is None
    with h5py.File(p, "r") as f, pytest.raises(ValueError):
        replay._build_action_sequence(f["episode_0"], "joint_angle")


def test_divergence_seven_dim_joint(tmp_path, replay):
    """7-dim joint action: resolver pads to 8 dims with -1, replay keeps 7 dims (the recorder already pads when writing, so normal data never triggers this)."""
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}")], joint=lambda n: [0.5] * 7)
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert r.get_step("joint_angle", 0).tolist() == [0.5] * 7 + [-1.0]
    with h5py.File(p, "r") as f:
        assert replay._build_action_sequence(f["episode_0"], "joint_angle")[0].shape == (7,)


def test_divergence_duplicate_suffix_key(tmp_path, replay):
    """``timestep_N_dupK`` keys: the resolver's regex does not accept them, replay accepts by prefix (the recorder never produces dup keys when writing a new group)."""
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}")])
    with h5py.File(p, "a") as f:
        f.copy("episode_0/timestep_0", "episode_0/timestep_0_dup1")
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert len(drain(r, "joint_angle")) == 1
    with h5py.File(p, "r") as f:
        assert len(replay._build_action_sequence(f["episode_0"], "joint_angle")) == 2


def test_divergence_sub_float32_waypoint_change(tmp_path, replay):
    """Two waypoints that differ only below float32 precision: resolver compares in the original dtype and keeps both, replay dedupes them to one after converting to float32."""
    a2 = list(A)
    a2[0] = A[0] + 1e-12
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}"), (1, False, False, a2, "{}")])
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert len(drain(r, "waypoint")) == 2
    with h5py.File(p, "r") as f:
        assert len(replay._build_action_sequence(f["episode_0"], "waypoint")) == 1


# ---------------------------------------------------------------- real process_episode


class _ReplayEnv:
    """Stand-in env: reset gives 3 frames (first 2 count as demonstration), step returns or raises per the script."""

    def __init__(self, script, *, fail_close=False):
        self.script = list(script)
        self.fail_close = fail_close
        self.actions = []
        self.closed = 0

    @staticmethod
    def _frames(k, base):
        return {
            "front_rgb_list": [np.full((64, 64, 3), base + i, dtype=np.uint8) for i in range(k)],
            "wrist_rgb_list": [np.full((64, 64, 3), 100 + base + i, dtype=np.uint8) for i in range(k)],
        }

    def reset(self):
        return self._frames(3, 10), {}

    def step(self, action):
        i = len(self.actions)
        self.actions.append(action)
        kind, status = self.script[i] if i < len(self.script) else ("go", "ongoing")
        if kind == "raise":
            raise RuntimeError("step broken")
        obs = None if kind == "obs_none" else self._frames(1, 50 + i)
        term = kind in ("term", "obs_none")
        trunc = kind == "trunc"
        return obs, 0.0, term, trunc, {"status": status}

    def render(self):
        pass

    def close(self):
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("close broken")


@pytest.fixture
def harness(replay, monkeypatch):
    rec = {"builders": [], "episodes": [], "saved": []}

    def install(env):
        class Builder:
            def __init__(self, **kw):
                rec["builders"].append(kw)

            def make_env_for_episode(self, idx):
                rec["episodes"].append(idx)
                return env

        monkeypatch.setattr(replay, "BenchmarkEnvBuilder", Builder)
        monkeypatch.setattr(replay, "_save_video", lambda *a: rec["saved"].append(a))
        return rec

    return install


@pytest.mark.parametrize(
    "script,steps,n_frames,outcome",
    [
        ([("go", "ongoing"), ("term", "success")], 2, 3 + 2, "success"),
        ([("trunc", "timeout")], 1, 3 + 1, "timeout"),
        ([], len(ONLINE), 3 + len(ONLINE), "unknown"),  # actions exhausted without termination
        ([("go", "ongoing"), ("obs_none", "error")], 2, 3 + 1, "unknown"),  # current behavior: an error episode with obs=None is labeled unknown
        ([("raise", "")], 1, 3, "unknown"),
    ],
)
def test_process_episode_order_and_outcome(script, steps, n_frames, outcome, h5_path, replay, harness):
    env = _ReplayEnv(script)
    rec = harness(env)
    with h5py.File(h5_path, "r") as f:
        replay.process_episode(f, 0, "FakeTask", "joint_angle")
    assert rec["builders"] == [dict(env_id="FakeTask", dataset="train", action_space="joint_angle", gui_render=False)]
    assert rec["episodes"] == [0]
    assert same_sequence(env.actions, EXPECTED["joint_angle"][:steps])
    assert env.closed == 1
    (frames, task, ep, goal, got_outcome, mode), = rec["saved"]
    assert (task, ep, goal, got_outcome, mode) == ("FakeTask", 0, "pick it", outcome, "joint_angle")
    # of the 3 reset frames the first 2 get red borders; afterwards each step that returns an observation normally appends 1 frame
    assert len(frames) == n_frames
    assert [f[0, 0].tolist() == [255, 0, 0] for f in frames[:3]] == [True, True, False]
    assert all(f.shape == (64, 128, 3) for f in frames)


def test_process_episode_multi_choice_passes_dicts(h5_path, replay, harness):
    env = _ReplayEnv([])
    harness(env)
    with h5py.File(h5_path, "r") as f:
        replay.process_episode(f, 0, "FakeTask", "multi_choice")
    assert env.actions == EXPECTED["multi_choice"]


def test_process_episode_close_error_propagates(h5_path, replay, harness):
    """Current behavior: an exception from env.close propagates out of process_episode and the video is not saved."""
    env = _ReplayEnv([("term", "success")], fail_close=True)
    rec = harness(env)
    with h5py.File(h5_path, "r") as f, pytest.raises(RuntimeError):
        replay.process_episode(f, 0, "FakeTask", "joint_angle")
    assert rec["saved"] == []


@pytest.mark.slow
def test_replay_save_video_real(replay, monkeypatch, tmp_path):
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pytest.skip("Unverified: ffmpeg missing")
    import imageio

    monkeypatch.setattr(replay, "REPLAY_VIDEO_DIR", str(tmp_path / "rv"))
    frames = [np.full((64, 128, 3), i * 20, dtype=np.uint8) for i in range(5)]
    path = replay._save_video(frames, "FakeTask", 4, "pick it", "success", "ee_pose")
    assert path == tmp_path / "rv" / "ee_pose" / "success_FakeTask_ep4_pick it.mp4"
    with imageio.get_reader(str(path)) as r:
        assert sum(1 for _ in r) == 5
