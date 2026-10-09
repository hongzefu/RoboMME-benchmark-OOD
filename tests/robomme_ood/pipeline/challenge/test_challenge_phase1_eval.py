"""C14 ``challenge_interface/scripts/phase1_eval.py``: bookkeeping and verdicts of the evaluation main loop.

The environment side is always a CPU stand-in (no real scene built): replace the three module attributes ``BenchmarkEnvBuilder``, ``PolicyClient``,
``imageio`` (monkeypatch module attributes, no ``sys.modules`` injection). All expected values are hand-computed:
each episode's status, step count and action chunk length are given by a script table; inference counts, frame counts, numerator and denominator are derived item by item from the table.

Success criterion (plan C14): only a status of exactly ``success`` counts as success; ``unsuccessful``/``not_success``/
``success_pending`` and other statuses containing the word success must not count; the denominator is fixed at "number of tasks × episodes per task",
and failures, exceptions and timeouts are all in the denominator.

Four places where production deviates from this criterion (D1 substring counts as success, D2 reset wait does not re-ask, D3 exception does not close the env,
D4 empty-observation error terminal raises KeyError) were ruled not-to-fix by the user on 2026-10-04 and are locked by ``test_known_defect_D<n>_*``
tests (contracts C14-10/12/13 recorded as blocked); once the current behavior changes these tests fail, and the contracts must be updated at the same time.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from challenge_interface.policy import DummyPolicy
from challenge_interface.scripts import phase1_eval as p

from challenge_support import LOOPBACK

TASKS = ["TaskA", "TaskB"]


def _frame(v: int) -> np.ndarray:
    return np.full((4, 4, 3), v % 256, dtype=np.uint8)


def _obs(n: int, v: int) -> dict:
    return {
        "front_rgb_list": [_frame(v) for _ in range(n)],
        "wrist_rgb_list": [_frame(v) for _ in range(n)],
        "joint_state_list": [np.full(8, v, dtype=np.float32) for _ in range(n)],
    }


class FakeEnv:
    """CPU stand-in env. script: status terminal status, steps step at which it ends, raise_at step at which it raises, error_obs whether the terminal returns an empty observation."""

    def __init__(self, env_id: str, script: dict, flags: dict):
        self.env_id, self.script, self.flags = env_id, script, flags
        self.t = 0
        self.actions: list[np.ndarray] = []
        self.closed = False

    def reset(self):
        info = {
            "task_goal": [f"goal-{self.env_id}"],
            "front_camera_intrinsic": np.eye(3),
            "wrist_camera_intrinsic": np.eye(3) * 2,
        }
        return _obs(2, 0), info

    def step(self, action):
        self.t += 1
        self.actions.append(np.array(action, copy=True))
        s = self.script
        if s.get("raise_at") == self.t:
            raise RuntimeError("stand-in env failure")
        if self.t >= s["steps"]:
            if s.get("error_obs"):
                # same shape as the IK-failure return of EndeffectorDemonstrationWrapper: empty observation + status=error.
                return {}, 0.0, True, False, {"status": "error", "error_message": "ik fail"}
            status = s["status"]
            info = {"status": status, "task_goal": [f"goal-{self.env_id}"]}
            return _obs(1, self.t), 0.0, status != "timeout", status == "timeout", info
        return _obs(1, self.t), 0.0, False, False, {"status": "ongoing", "task_goal": [f"goal-{self.env_id}"]}

    def close(self):
        self.closed = True


def make_builder_cls(scripts: dict):
    """Return a stand-in BenchmarkEnvBuilder class; scripts[(env_id, episode_idx)] gives the script of each episode."""
    made: list[FakeEnv] = []
    inits: list[dict] = []

    class FakeBuilder:
        envs = made
        builders = inits

        def __init__(self, env_id, dataset, action_space, max_steps):
            self.env_id = env_id
            inits.append(dict(env_id=env_id, dataset=dataset, action_space=action_space, max_steps=max_steps))

        @staticmethod
        def get_task_list():
            return list(TASKS)

        def make_env_for_episode(self, episode_idx, **flags):
            env = FakeEnv(self.env_id, scripts[(self.env_id, episode_idx)], flags)
            env.episode_idx = episode_idx
            made.append(env)
            return env

    return FakeBuilder


class FakeClient:
    def __init__(self, chunk: int = 3, dim: int | None = None, reset_replies=None):
        """reset_replies: list of reset replies returned in order, repeating the last one when exhausted; defaults to always acknowledging."""
        # default dimension uses the production constant of the joint angle space.
        self.chunk = chunk
        self.dim = dim if dim is not None else p.EXPECTED_ACTION_SHAPES["joint_angle"][0]
        self.reset_replies = reset_replies if reset_replies is not None else [{"reset_finished": True}]
        self.reset_calls = 0
        self.infer_inputs: list[dict] = []

    def reset(self):
        self.reset_calls += 1
        return self.reset_replies[min(self.reset_calls, len(self.reset_replies)) - 1]

    def infer(self, inputs):
        self.infer_inputs.append(copy.deepcopy(inputs))
        return {"actions": np.zeros((self.chunk, self.dim), dtype=np.float32)}


@pytest.fixture
def no_video(monkeypatch):
    saved = []
    monkeypatch.setattr(p, "imageio", SimpleNamespace(mimsave=lambda path, frames, fps: saved.append((path, len(frames)))))
    return saved


def _run_episode(client, builder_cls, env_id="TaskA", ep=0, *, action_space="joint_angle", depth=False, cam=False):
    b = builder_cls(env_id=env_id, dataset="test", action_space=action_space, max_steps=99)
    return p.run_episode(client, b, ep, env_id, use_depth=depth, use_camera_params=cam, action_space=action_space)


def _run_main(monkeypatch, tmp_path, builder_cls, client, *extra):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(p, "BenchmarkEnvBuilder", builder_cls)
    monkeypatch.setattr(p, "PolicyClient", lambda host, port: client)
    monkeypatch.setattr(sys, "argv", ["phase1_eval", "--num_episodes", "2", "--team_id", "t1", *extra])
    p.main()
    out = Path(tmp_path, "challenge_results", "t1")
    metrics = out / "metrics.json"
    return (json.loads(metrics.read_text()) if metrics.exists() else None), json.loads((out / "progress.json").read_text())


# ---------------------------------------------------------------- success verdict


@pytest.mark.parametrize("status", ["fail", "failed", "timeout", "ongoing", "error", "unknown", "", None, "success_fail"])
def test_is_success_rejects_non_success_statuses(status):
    assert p._is_success(status) is False


def test_is_success_accepts_exact_success():
    assert p._is_success("success") is True


@pytest.mark.parametrize("status", ["unsuccessful", "not_success", "success_pending", "partial_success"])
def test_known_defect_D1_is_success_counts_success_substrings(status):
    """Known defect D1 (locks current behavior): ``_is_success`` judges by substring; any status containing success and not containing fail counts as success.

    Correct behavior: only a status of exactly ``success`` counts (exact match); none of these four statuses count.
    User ruled not-to-fix on 2026-10-04; this test locks current behavior and fails once it changes, as a reminder to update contract C14-10.
    """
    assert p._is_success(status) is True


# ---------------------------------------------------------------- single-episode protocol


def test_run_episode_streams_observations_and_consumes_chunks(no_video):
    # by hand: chunk length 3, terminates at step 5 → 2 inferences; the 1st carries the 2 reset frames (first step),
    # the 2nd carries the 3 frames of steps 1-3 (non-first step); env receives 5 actions in total; video 2 + 5 = 7 frames.
    B = make_builder_cls({("TaskA", 0): {"status": "success", "steps": 5}})
    c = FakeClient(chunk=3)
    outcome, frames, goal = _run_episode(c, B)
    assert outcome == "success" and goal == "goal-TaskA"
    assert c.reset_calls == 1
    assert len(c.infer_inputs) == 2
    first, second = c.infer_inputs
    assert first["is_first_step"] is True and len(first["front_rgb_list"]) == 2
    assert first["task_goal"] == ["goal-TaskA"]
    assert second["is_first_step"] is False
    assert [int(f[0, 0, 0]) for f in second["front_rgb_list"]] == [1, 2, 3]
    (env,) = B.envs
    assert len(env.actions) == 5 and all(a.shape == p.EXPECTED_ACTION_SHAPES["joint_angle"] for a in env.actions)
    assert len(frames) == 7 and frames[0].shape == (4, 8, 3)
    assert env.closed is True


@pytest.mark.parametrize("flag", [True, False])
def test_run_episode_passes_depth_and_camera_flags(no_video, flag):
    B = make_builder_cls({("TaskA", 0): {"status": "fail", "steps": 1}})
    c = FakeClient(chunk=1)
    _run_episode(c, B, depth=flag, cam=flag)
    (env,) = B.envs
    assert env.flags == {
        "include_front_depth": flag,
        "include_wrist_depth": flag,
        "include_front_camera_extrinsic": flag,
        "include_wrist_camera_extrinsic": flag,
        "include_front_camera_intrinsic": flag,
        "include_wrist_camera_intrinsic": flag,
    }
    first = c.infer_inputs[0]
    assert ("front_camera_intrinsic" in first) is flag
    if flag:
        assert first["wrist_camera_intrinsic"].tolist() == (np.eye(3) * 2).tolist()


def test_run_episode_rejects_wrong_action_shape(no_video):
    # the policy returns one more dimension than ee_pose requires; must raise immediately and not be sent into the env.
    B = make_builder_cls({("TaskA", 0): {"status": "success", "steps": 3}})
    with pytest.raises(AssertionError):
        _run_episode(FakeClient(chunk=2, dim=p.EXPECTED_ACTION_SHAPES["ee_pose"][0] + 1), B, action_space="ee_pose")
    assert B.envs[0].actions == []


def test_known_defect_D2_reset_wait_never_repolls(no_video, monkeypatch):
    """Known defect D2 (locks current behavior): when the reset reply lacks ``reset_finished``, ``run_episode`` calls
    ``client.reset()`` only once and then sleeps repeatedly in the ``while`` loop without updating the reply, never getting the acknowledgement.

    Correct behavior: re-ask reset while waiting, and raise after a bounded number of tries without starting the episode.
    User ruled not-to-fix on 2026-10-04. Observation: run the episode in a thread and replace the sleep seen by phase1_eval with a counter;
    after observing at least 20 waits the counter raises a stop signal so the thread ends in bounded time (timeouts throughout, never hangs).
    """
    import threading

    class _Stop(Exception):
        pass

    sleeps: list = []
    stop = threading.Event()
    enough = threading.Event()

    def fake_sleep(sec):
        sleeps.append(sec)
        if len(sleeps) >= 20:
            enough.set()
        if stop.is_set() or len(sleeps) >= 100000:
            raise _Stop
        if enough.is_set():
            stop.wait(0.01)

    monkeypatch.setattr(p, "time", SimpleNamespace(sleep=fake_sleep, time=lambda: 0.0))
    B = make_builder_cls({("TaskA", 0): {"status": "success", "steps": 1}})
    # acknowledge from the second reply on: if the implementation re-asks, it would get the acknowledgement the second time and start the episode.
    c = FakeClient(chunk=1, reset_replies=[{}, {"reset_finished": True}])
    result: dict = {}

    def _target():
        try:
            result["value"] = _run_episode(c, B)
        except _Stop:
            result["stopped"] = True
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    try:
        assert enough.wait(5), "no reset wait observed within 5 s"
        # within the observation window: still waiting, reset asked only once, episode not started.
        assert t.is_alive()
        assert c.reset_calls == 1
        assert B.envs == []
    finally:
        stop.set()
        t.join(5)
    assert not t.is_alive(), "thread did not end within 5 s after the stop signal"
    assert result == {"stopped": True}
    assert c.reset_calls == 1


def test_known_defect_D3_env_not_closed_when_episode_raises(no_video):
    """Known defect D3 (locks current behavior): when an episode raises mid-way (simulation failure) ``run_episode`` has no try/finally,
    so ``env.close()`` is not called.

    Correct behavior: the env is closed whether the episode ends normally or exits with an exception. User ruled not-to-fix on 2026-10-04.
    """
    B = make_builder_cls({("TaskA", 0): {"status": "success", "steps": 5, "raise_at": 2}})
    with pytest.raises(RuntimeError, match="stand-in env failure"):
        _run_episode(FakeClient(chunk=3), B)
    assert B.envs[0].closed is False


def test_known_defect_D4_error_status_empty_obs_raises_keyerror(no_video):
    """Known defect D4 (locks current behavior): on IK failure the env returns an empty observation ``{}`` + ``status=error``, and ``run_episode``
    then reads ``obs["front_rgb_list"]`` and raises KeyError, aborting the whole evaluation.

    Correct behavior: the episode ends with an ``error`` result, counts in the denominator, does not count as success, and the env is closed.
    User ruled not-to-fix on 2026-10-04.
    """
    B = make_builder_cls({("TaskA", 0): {"status": "error", "steps": 2, "error_obs": True}})
    with pytest.raises(KeyError, match="front_rgb_list"):
        _run_episode(FakeClient(chunk=3), B)
    assert B.envs[0].closed is False


# ---------------------------------------------------------------- full-run denominator


def test_main_metrics_fixed_denominator(monkeypatch, tmp_path, no_video):
    # by hand: TaskA = success, fail; TaskB = timeout, success → 2 successes, denominator 2 tasks × 2 episodes = 4.
    scripts = {
        ("TaskA", 0): {"status": "success", "steps": 2},
        ("TaskA", 1): {"status": "fail", "steps": 4},
        ("TaskB", 0): {"status": "timeout", "steps": 3},
        ("TaskB", 1): {"status": "success", "steps": 1},
    }
    B = make_builder_cls(scripts)
    metrics, progress = _run_main(monkeypatch, tmp_path, B, FakeClient(chunk=2))
    assert metrics["overall"] == {"avg_success": 0.5, "total_success": 2, "total_episodes": 4}
    assert metrics["per_task"]["TaskA"] == {"avg_success": 0.5, "success_count": 1, "num_episodes": 2}
    assert metrics["per_task"]["TaskB"] == {"avg_success": 0.5, "success_count": 1, "num_episodes": 2}
    assert progress["finished"] is True
    assert {k: {e: v["outcome"] for e, v in d.items()} for k, d in progress["completed"].items()} == {
        "TaskA": {"0": "success", "1": "fail"},
        "TaskB": {"0": "timeout", "1": "success"},
    }
    # expected max_steps comes from the real parse_args on the same argv; constant literal values are covered by tests/robomme_ood/contract.
    expected_max_steps = p.parse_args().max_steps
    assert all(b["dataset"] == "test" and b["max_steps"] == expected_max_steps for b in B.builders)
    assert len(no_video) == 4 and all(n > 0 for _, n in no_video)
    assert all(env.closed for env in B.envs)


def test_main_all_failures_gives_zero_not_shrunk_denominator(monkeypatch, tmp_path, no_video):
    scripts = {(t, e): {"status": s, "steps": 1} for (t, e), s in zip(
        [("TaskA", 0), ("TaskA", 1), ("TaskB", 0), ("TaskB", 1)], ["fail", "timeout", "fail", "unknown"])}
    metrics, _ = _run_main(monkeypatch, tmp_path, make_builder_cls(scripts), FakeClient(chunk=1))
    assert metrics["overall"] == {"avg_success": 0.0, "total_success": 0, "total_episodes": 4}


def test_main_crash_writes_no_metrics_and_resume_keeps_denominator(monkeypatch, tmp_path, no_video):
    # first round: TaskB episode 0 simulation failure → evaluation aborts and must not write (denominator-shrunk) metrics.
    bad = {
        ("TaskA", 0): {"status": "success", "steps": 1},
        ("TaskA", 1): {"status": "success", "steps": 1},
        ("TaskB", 0): {"status": "success", "steps": 3, "raise_at": 1},
        ("TaskB", 1): {"status": "success", "steps": 1},
    }
    with pytest.raises(RuntimeError):
        _run_main(monkeypatch, tmp_path, make_builder_cls(bad), FakeClient(chunk=1))
    out = tmp_path / "challenge_results" / "t1"
    assert not (out / "metrics.json").exists()
    progress = json.loads((out / "progress.json").read_text())
    assert set(progress["completed"].get("TaskA", {})) == {"0", "1"}
    assert progress["completed"].get("TaskB", {}) == {}
    # second round: resume with the same config, only the two unfinished episodes are filled in; the denominator is still 4.
    good = dict(bad)
    good[("TaskB", 0)] = {"status": "fail", "steps": 1}
    B2 = make_builder_cls(good)
    metrics, _ = _run_main(monkeypatch, tmp_path, B2, FakeClient(chunk=1))
    assert sorted((e.env_id, e.episode_idx) for e in B2.envs) == [("TaskB", 0), ("TaskB", 1)]
    assert metrics["overall"] == {"avg_success": 0.75, "total_success": 3, "total_episodes": 4}


def test_known_defect_D1_main_metrics_count_success_substrings(monkeypatch, tmp_path, no_video):
    """Known defect D1, full-run version (locks current behavior): of four episode statuses only 1 is exactly success, but the main evaluation metric
    also counts ``unsuccessful``/``not_success``/``success_pending`` as success, giving 4/4.

    Correct behavior: 1/4 (denominator 4 unchanged). User ruled not-to-fix on 2026-10-04.
    """
    scripts = {
        ("TaskA", 0): {"status": "unsuccessful", "steps": 1},
        ("TaskA", 1): {"status": "success", "steps": 1},
        ("TaskB", 0): {"status": "not_success", "steps": 1},
        ("TaskB", 1): {"status": "success_pending", "steps": 1},
    }
    metrics, _ = _run_main(monkeypatch, tmp_path, make_builder_cls(scripts), FakeClient(chunk=1))
    assert metrics["overall"] == {"avg_success": 1.0, "total_success": 4, "total_episodes": 4}


def test_main_end_to_end_over_real_websocket(monkeypatch, tmp_path, no_video, ws_server):
    """Real PolicyServer(DummyPolicy) + real PolicyClient over loopback, env is a stand-in.

    By hand: each episode terminates at step 12, action shape per the production constant EXPECTED_ACTION_SHAPES; the status table gives 3 successes / 4.
    """
    h = ws_server(DummyPolicy())
    statuses = {("TaskA", 0): "success", ("TaskA", 1): "fail", ("TaskB", 0): "success", ("TaskB", 1): "success"}
    B = make_builder_cls({k: {"status": s, "steps": 12} for k, s in statuses.items()})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(p, "BenchmarkEnvBuilder", B)
    monkeypatch.setattr(sys, "argv", [
        "phase1_eval", "--num_episodes", "2", "--team_id", "t1",
        "--host", LOOPBACK, "--port", str(h.port), "--transport", "websocket",
    ])
    p.main()
    metrics = json.loads((tmp_path / "challenge_results" / "t1" / "metrics.json").read_text())
    assert metrics["overall"] == {"avg_success": 0.75, "total_success": 3, "total_episodes": 4}
    assert all(len(env.actions) == 12 and env.actions[0].shape == p.EXPECTED_ACTION_SHAPES["joint_angle"] for env in B.envs)
    # DummyPolicy's gripper dimension is always 1.0 and unchanged after the network round trip.
    assert all(float(a[-1]) == 1.0 for env in B.envs for a in env.actions)
