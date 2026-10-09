"""C14 ``challenge_interface/scripts/phase1_eval.py``：评估主循环的记账与判定。

环境侧一律 CPU 替身（不构建真实场景）：替换模块里的 ``BenchmarkEnvBuilder``、``PolicyClient``、
``imageio`` 三个属性（monkeypatch 模块属性，不注入 ``sys.modules``）。期望值全部手算：
每局的状态、步数、动作块长由脚本表给定，推理次数、帧数、分子分母按表逐项推出。

成功判定口径（计划 C14）：只有状态恰为 ``success`` 才计成功，``unsuccessful``／``not_success``／
``success_pending`` 等含 success 字样的状态不得计成功；分母固定为「任务数 × 每任务局数」，
失败、异常、超时都在分母里。

生产现状与上述口径不符的四处（D1 子串计成功、D2 reset 等待不重新询问、D3 异常不关环境、
D4 空观测 error 终态抛 KeyError）经用户 2026-10-04 裁决不修，以 ``test_known_defect_D<n>_*``
用例锁定现状（契约 C14-10／12／13 记 blocked）；现状一旦改变这些用例即失败，届时同步改契约。
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
    """CPU 替身环境。script: status 终态、steps 第几步结束、raise_at 第几步抛错、error_obs 终态是否返回空观测。"""

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
            raise RuntimeError("替身环境故障")
        if self.t >= s["steps"]:
            if s.get("error_obs"):
                # 与 EndeffectorDemonstrationWrapper 的 IK 失败返回形态一致：空观测 + status=error。
                return {}, 0.0, True, False, {"status": "error", "error_message": "ik fail"}
            status = s["status"]
            info = {"status": status, "task_goal": [f"goal-{self.env_id}"]}
            return _obs(1, self.t), 0.0, status != "timeout", status == "timeout", info
        return _obs(1, self.t), 0.0, False, False, {"status": "ongoing", "task_goal": [f"goal-{self.env_id}"]}

    def close(self):
        self.closed = True


def make_builder_cls(scripts: dict):
    """返回替身 BenchmarkEnvBuilder 类；scripts[(env_id, episode_idx)] 给出每局脚本。"""
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
        """reset_replies：依次返回的 reset 回复列表，用完后重复最后一个；默认恒为确认。"""
        # 缺省维度取关节角空间的生产常量。
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


# ---------------------------------------------------------------- 成功判定


@pytest.mark.parametrize("status", ["fail", "failed", "timeout", "ongoing", "error", "unknown", "", None, "success_fail"])
def test_is_success_rejects_non_success_statuses(status):
    assert p._is_success(status) is False


def test_is_success_accepts_exact_success():
    assert p._is_success("success") is True


@pytest.mark.parametrize("status", ["unsuccessful", "not_success", "success_pending", "partial_success"])
def test_known_defect_D1_is_success_counts_success_substrings(status):
    """已知缺陷 D1（锁定现状）：``_is_success`` 按子串判定，含 success 且不含 fail 的状态都计成功。

    正确行为应当是：只有状态恰为 ``success`` 才计成功（精确匹配），这四种状态都不计。
    用户 2026-10-04 裁决不修；本用例锁定现状，现状一旦改变即失败，提醒同步更新契约 C14-10。
    """
    assert p._is_success(status) is True


# ---------------------------------------------------------------- 单局协议


def test_run_episode_streams_observations_and_consumes_chunks(no_video):
    # 手算：块长 3、第 5 步终止 → 推理 2 次；第 1 次带 reset 的 2 帧（首步），
    # 第 2 次带第 1～3 步的 3 帧（非首步）；环境共收 5 个动作；视频 2 + 5 = 7 帧。
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
    # 策略回的维度比 ee_pose 要求多 1，必须当场报错，不得送进环境。
    B = make_builder_cls({("TaskA", 0): {"status": "success", "steps": 3}})
    with pytest.raises(AssertionError):
        _run_episode(FakeClient(chunk=2, dim=p.EXPECTED_ACTION_SHAPES["ee_pose"][0] + 1), B, action_space="ee_pose")
    assert B.envs[0].actions == []


def test_known_defect_D2_reset_wait_never_repolls(no_video, monkeypatch):
    """已知缺陷 D2（锁定现状）：reset 回复缺 ``reset_finished`` 时，``run_episode`` 只调用一次
    ``client.reset()``，之后在 ``while`` 里反复 sleep 而不更新回复，永远等不到确认。

    正确行为应当是：等待期间重新询问 reset，并在有限次后报错且不开局。
    用户 2026-10-04 裁决不修。观察方式：在线程里跑本局，phase1_eval 看到的 sleep 换成计数器；
    观察到至少 20 次等待后令计数器抛出停止信号，让线程有限结束（全程有超时，不会挂死）。
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
    # 第二次起回复确认：若实现会重新询问，就会在第二次拿到确认并开局。
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
        assert enough.wait(5), "5 s 内未观察到 reset 等待"
        # 观察窗口内：仍在等待、只问过一次 reset、尚未开局。
        assert t.is_alive()
        assert c.reset_calls == 1
        assert B.envs == []
    finally:
        stop.set()
        t.join(5)
    assert not t.is_alive(), "停止信号发出后线程未在 5 s 内结束"
    assert result == {"stopped": True}
    assert c.reset_calls == 1


def test_known_defect_D3_env_not_closed_when_episode_raises(no_video):
    """已知缺陷 D3（锁定现状）：单局中途抛异常（仿真故障）时 ``run_episode`` 没有 try/finally，
    ``env.close()`` 不会被调用。

    正确行为应当是：无论正常结束还是异常退出，环境都要关闭。用户 2026-10-04 裁决不修。
    """
    B = make_builder_cls({("TaskA", 0): {"status": "success", "steps": 5, "raise_at": 2}})
    with pytest.raises(RuntimeError, match="替身环境故障"):
        _run_episode(FakeClient(chunk=3), B)
    assert B.envs[0].closed is False


def test_known_defect_D4_error_status_empty_obs_raises_keyerror(no_video):
    """已知缺陷 D4（锁定现状）：IK 失败时环境返回空观测 ``{}`` + ``status=error``，``run_episode``
    随即取 ``obs["front_rgb_list"]`` 抛 KeyError，整个评估中止。

    正确行为应当是：该局以 ``error`` 结果收尾、计入分母、不计成功，并关闭环境。
    用户 2026-10-04 裁决不修。
    """
    B = make_builder_cls({("TaskA", 0): {"status": "error", "steps": 2, "error_obs": True}})
    with pytest.raises(KeyError, match="front_rgb_list"):
        _run_episode(FakeClient(chunk=3), B)
    assert B.envs[0].closed is False


# ---------------------------------------------------------------- 全流程分母


def test_main_metrics_fixed_denominator(monkeypatch, tmp_path, no_video):
    # 手算：TaskA = success、fail；TaskB = timeout、success → 成功 2，分母 2 任务 × 2 局 = 4。
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
    # max_steps 的期望取自真实 parse_args 在同一 argv 下的结果，常量字面值由 tests/robomme_ood/contract 负责。
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
    # 第一轮：TaskB 第 0 局仿真故障 → 评估中止，不得写出（缩小分母的）metrics。
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
    # 第二轮：同配置续跑，只补未完成的两局；分母仍是 4。
    good = dict(bad)
    good[("TaskB", 0)] = {"status": "fail", "steps": 1}
    B2 = make_builder_cls(good)
    metrics, _ = _run_main(monkeypatch, tmp_path, B2, FakeClient(chunk=1))
    assert sorted((e.env_id, e.episode_idx) for e in B2.envs) == [("TaskB", 0), ("TaskB", 1)]
    assert metrics["overall"] == {"avg_success": 0.75, "total_success": 3, "total_episodes": 4}


def test_known_defect_D1_main_metrics_count_success_substrings(monkeypatch, tmp_path, no_video):
    """已知缺陷 D1 全流程版（锁定现状）：四局状态里只有 1 局恰为 success，但主评估指标把
    ``unsuccessful``／``not_success``／``success_pending`` 也计入成功，得 4/4。

    正确行为应当是 1/4（分母 4 不变）。用户 2026-10-04 裁决不修。
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
    """真实 PolicyServer(DummyPolicy) + 真实 PolicyClient 走回环，环境用替身。

    手算：每局 12 步终止，动作形状按生产常量 EXPECTED_ACTION_SHAPES；状态表给出 3 成功 / 4。
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
    # DummyPolicy 的夹爪维恒为 1.0，经网络往返后不变。
    assert all(float(a[-1]) == 1.0 for env in B.envs for a in env.actions)
