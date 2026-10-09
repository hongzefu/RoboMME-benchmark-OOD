"""C11 读回差分：同一份手写 h5 夹具同时喂真实 EpisodeDatasetResolver 与上游入口
``scripts/dataset_replay.py::_build_action_sequence``，两者都与手写期望比；
再用替身环境驱动真实 ``process_episode``，核动作顺序、错误输出、终态与结果标签。

夹具刻意包含：timestep 键乱序插入且含 10／11（字典序与数字序不同）、缺号、演示步、
相邻重复与隔开重复的 waypoint、NaN 哨兵、形状不对的 waypoint、各类无效 choice。
两侧已知的行为分歧单列为「现状记录」用例（上游文件冻结，只测不改）。
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


def J(n):  # 第 n 号 timestep 的关节动作；取 1/4 的整数倍，float32 可精确表示
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


# (timestep 号, is_video_demo, is_subgoal_boundary, waypoint, choice_action 原文)；按此乱序写入
ROWS = [
    (10, False, True, A, C("C")),  # 缺 point → 选项被过滤
    (0, True, True, A, C("A", [1, 2])),  # 演示步 → 全部模式都跳过
    (2, False, False, A, C("X", [0, 0])),  # 与上一条 waypoint 相同 → 去重；非边界 → 无选项
    (11, False, True, A, C("D", [5, 6])),
    (1, False, True, A, C("B", [3, 4])),
    (3, False, True, NAN, C("", raw="{bad json")),  # NaN 哨兵 → 跳过；坏 JSON → 过滤
    (5, False, True, B, C("  ", [0, 0])),  # 空白 choice → 过滤；缺 4 号
    (12, False, False, [1.0] * 6, C("", raw="[]")),  # waypoint 形状不对 → 跳过
]
ONLINE = [1, 2, 3, 5, 10, 11, 12]
EXPECTED = {
    "joint_angle": [J(n) for n in ONLINE],
    "ee_pose": [E(n) for n in ONLINE],
    # 数字序 1,2,3,5,10,11：A、(A 相邻重复)、(NaN)、B、A（与上一条不同，保留）、(A 相邻重复)
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
    """判定器：逐项相等（数组按 float64 逐位比较，字典按值比较）。"""
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
    assert not same_sequence(EXPECTED["waypoint"], [A, B])  # 全局去重
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
        assert all(np.asarray(x).dtype == np.float32 for x in seq)  # 上游回放统一转 float32


def test_replay_rejects_unknown_mode(h5_path, replay):
    with h5py.File(h5_path, "r") as f, pytest.raises(ValueError):
        replay._build_action_sequence(f["episode_0"], "teleport")


def test_resolver_api_edges(h5_path, tmp_path):
    with h5py.File(h5_path, "a") as f:
        f.create_group("episode_7")
        f.create_group("not_an_episode")
    assert list_episode_indices("FakeTask", h5_path.parent) == [0, 7]
    assert list_episode_indices("Other", h5_path) == [0, 7]  # 传完整 .h5 路径时忽略 env_id
    with pytest.raises(FileNotFoundError):
        list_episode_indices("Missing", tmp_path)
    with pytest.raises(FileNotFoundError):
        EpisodeDatasetResolver("Missing", 0, tmp_path)
    with pytest.raises(KeyError):
        EpisodeDatasetResolver("FakeTask", 99, h5_path)
    with h5py.File(h5_path, "a"):  # 抛 KeyError 前已关闭文件 → 可再以写模式打开
        pass
    r = EpisodeDatasetResolver("FakeTask", 0, h5_path)
    assert r.get_step("joint_angle", -1) is None
    assert r.get_step("teleport", 0) is None
    assert r.get_step("joint_angle", len(ONLINE)) is None
    assert r.get_step("multi_choice", 2) is None
    first = r.get_step("multi_choice", 0)
    first["choice"] = "mutated"
    assert r.get_step("multi_choice", 0)["choice"] == "B"  # 返回副本
    r.close()
    r.close()  # 幂等


# ---------------------------------------------------------------- 已知分歧（现状记录）


def test_divergence_missing_joint_field(tmp_path, replay):
    """非演示步缺 joint_action：resolver 保留该步位置并返回 None；回放直接跳过，后续下标前移。"""
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
    """录像器把 None 动作写成字符串 "None"：resolver 读成 None，回放转 float32 时抛 ValueError。"""
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}")], joint=None)
    with h5py.File(p, "a") as f:
        f["episode_0/timestep_0/action"].create_dataset("joint_action", data="None", dtype=h5py.special_dtype(vlen=str))
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert r.get_step("joint_angle", 0) is None
    with h5py.File(p, "r") as f, pytest.raises(ValueError):
        replay._build_action_sequence(f["episode_0"], "joint_angle")


def test_divergence_seven_dim_joint(tmp_path, replay):
    """7 维关节动作：resolver 补 -1 到 8 维，回放保持 7 维（录像器写入时已补齐，正常数据不触发）。"""
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}")], joint=lambda n: [0.5] * 7)
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert r.get_step("joint_angle", 0).tolist() == [0.5] * 7 + [-1.0]
    with h5py.File(p, "r") as f:
        assert replay._build_action_sequence(f["episode_0"], "joint_angle")[0].shape == (7,)


def test_divergence_duplicate_suffix_key(tmp_path, replay):
    """``timestep_N_dupK`` 键：resolver 的正则不收，回放按前缀收（录像器写新组时从不产生 dup 键）。"""
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}")])
    with h5py.File(p, "a") as f:
        f.copy("episode_0/timestep_0", "episode_0/timestep_0_dup1")
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert len(drain(r, "joint_angle")) == 1
    with h5py.File(p, "r") as f:
        assert len(replay._build_action_sequence(f["episode_0"], "joint_angle")) == 2


def test_divergence_sub_float32_waypoint_change(tmp_path, replay):
    """两条只在 float32 精度以下不同的 waypoint：resolver 按原 dtype 比较保留两条，回放转 float32 后去重成一条。"""
    a2 = list(A)
    a2[0] = A[0] + 1e-12
    p = write_episode(tmp_path / "x.h5", [(0, False, False, A, "{}"), (1, False, False, a2, "{}")])
    with EpisodeDatasetResolver("x", 0, p) as r:
        assert len(drain(r, "waypoint")) == 2
    with h5py.File(p, "r") as f:
        assert len(replay._build_action_sequence(f["episode_0"], "waypoint")) == 1


# ---------------------------------------------------------------- 真实 process_episode


class _ReplayEnv:
    """替身环境：reset 给 3 帧（前 2 帧算演示），step 按脚本返回或抛错。"""

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
            raise RuntimeError("step 坏了")
        obs = None if kind == "obs_none" else self._frames(1, 50 + i)
        term = kind in ("term", "obs_none")
        trunc = kind == "trunc"
        return obs, 0.0, term, trunc, {"status": status}

    def render(self):
        pass

    def close(self):
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("close 坏了")


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
        ([], len(ONLINE), 3 + len(ONLINE), "unknown"),  # 动作用完仍未终止
        ([("go", "ongoing"), ("obs_none", "error")], 2, 3 + 1, "unknown"),  # 现状：obs=None 的错误局被标成 unknown
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
    # reset 的 3 帧里前 2 帧加红框；之后每次正常返回观测的 step 追加 1 帧
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
    """现状：env.close 抛错直接冒出 process_episode，录像不保存。"""
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
        pytest.skip("未验证：缺 ffmpeg")
    import imageio

    monkeypatch.setattr(replay, "REPLAY_VIDEO_DIR", str(tmp_path / "rv"))
    frames = [np.full((64, 128, 3), i * 20, dtype=np.uint8) for i in range(5)]
    path = replay._save_video(frames, "FakeTask", 4, "pick it", "success", "ee_pose")
    assert path == tmp_path / "rv" / "ee_pose" / "success_FakeTask_ep4_pick it.mp4"
    with imageio.get_reader(str(path)) as r:
        assert sum(1 for _ in r) == 5
