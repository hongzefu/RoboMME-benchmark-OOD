"""C10／C11 录制与读回闭环：独立事件表 → CPU 替身环境 → 真实 RobommeRecordWrapper.step／close
→ h5py → 真实 EpisodeDatasetResolver 与 dataset_replay._build_action_sequence。

官方 ``robomme`` 与 hard 包的复制件 ``robomme_hard`` 两个 RecordWrapper 类都跑同一套闭环（参数化）。
期望值全部来自本文件手写的事件表与手算，不调用被测逻辑生成期望。
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
# 手算：W1 单位四元数 → rpy 0，close → 夹爪 -1；W3 绕 z 轴 90° → yaw = π/2，open → 夹爪 +1。
W1_ACTION = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, -1.0])
W3_ACTION = np.array([0.4, -0.1, 0.25, 0.0, 0.0, math.pi / 2, 1.0])

GROUNDED = "pick the cube at <14, 24>"  # 分割方块行 10..19、列 20..29 的整数均值

# 主场景事件表（8 次 step）与逐条记录的手写期望。
MAIN_EVENTS = [
    Event(name="NO RECORD", demo=True, task_index=0),  # t=1 不入记录
    Event(name="watch", demo=True, task_index=0, waypoint=W1),  # t=2 → rec0
    Event(name="watch", demo=True, task_index=1, waypoint=W2_CROSS, choice_text="press the button"),  # t=3 → rec1
    Event(name="pick", demo=False, task_index=1, pre_demo=False, choice_text="press the button"),  # t=4 → rec2
    Event(name="pick", demo=False, task_index=2, waypoint=W3, choice_text="unknown action"),  # t=5 → rec3
    Event(name="pick", demo=False, task_index=2, choice_text="unknown action"),  # t=6 → rec4
    Event(name="NO RECORD", demo=False, task_index=2, choice_text="unknown action"),  # t=7 不入记录
    Event(name="pick", demo=False, task_index=3, choice_text="press the button", terminated=True, success=True),  # t=8 → rec5
]
NAN7 = np.full(7, np.nan)
# 每条记录：(来自第几次 step, simple_subgoal, is_video_demo, is_subgoal_boundary, is_completed, waypoint, choice, 夹爪张开)
# 夹爪列按事件表写出：action_of(t) 奇数步发张开指令、偶数步发闭合指令。
MAIN_EXPECTED = [
    (2, "watch", True, False, False, W1_ACTION, "", False),
    (3, "watch", True, True, False, W1_ACTION, "B", True),
    (4, "pick", False, False, False, NAN7, "B", False),  # 外层切到在线 → 缓存清空
    (5, "pick", False, True, False, W3_ACTION, "", True),
    (6, "pick", False, False, False, W3_ACTION, "", False),
    (8, "pick", False, True, True, W3_ACTION, "B", False),
]
OPTIONS = [{"label": "b", "action": "press the button", "available": None}]


@pytest.fixture(autouse=True)
def no_encode(monkeypatch):
    """日常门禁不调 ffmpeg：把逐局 mp4 写出替换成只记账（编码本身在 slow 用例里测）。"""
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
    """选项构造器替身：录像器在 step 里经模块名 get_vqa_options、在 close 里经各自包的 vqa_options 模块取选项。"""
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
        # reset 帧与 NO RECORD 步都不入记录：8 次 step → 6 条，编号连续从 0 开始
        assert sorted(ts_keys, key=lambda k: int(k.split("_")[1])) == [f"timestep_{i}" for i in range(len(MAIN_EXPECTED))]
        for i, (t, name, demo, boundary, done, wp, choice, opened) in enumerate(MAIN_EXPECTED):
            g = ep[f"timestep_{i}"]
            assert set(g.keys()) == TIMESTEP_GROUPS
            assert set(g["obs"].keys()) == OBS_KEYS
            assert set(g["action"].keys()) == ACTION_KEYS
            assert set(g["info"].keys()) == INFO_KEYS
            act = action_of(t)
            # action_t 与 obs_after_t 同在一条记录：图像编码的 step 序号 = t，关节读数 = action_t[:7]
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
            # FK 不可用（替身没有机器人模型）→ eef_action 恒为 7 维零
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
        assert set(setup.keys()) == SETUP_KEYS_BASE  # FakeTask 无语言目标 → 不写 task_goal
        assert int(setup["seed"][()]) == 77
        assert setup["difficulty"][()].decode() == "easy"
        np.testing.assert_array_equal(setup["front_camera_intrinsic"][()], INTRINSIC)
        np.testing.assert_array_equal(setup["wrist_camera_intrinsic"][()], WRIST_INTRINSIC)
        assert json.loads(setup["available_multi_choices"][()]) == [
            {"label": "b", "action": "press the button", "need_parameter": False}
        ]
    # 环境收到的动作就是调用方发出的动作，次数与顺序不变
    assert [np.asarray(a).tolist() for a in env.received_actions] == [action_of(t).tolist() for t in range(1, 9)]


def test_root_and_group_attrs_are_empty(mod, tmp_path):
    """录像器不写任何 h5 属性；根或组上多出属性即是格式漂移（M14 植入：改根属性）。"""
    _, _, path, _ = _main_episode(mod, tmp_path)
    tree = read_tree(path)
    attrs = {k: v for k, v in tree.items() if k.endswith("@attrs") and v}
    assert attrs == {}


def test_recorded_actions_finite_and_bit_exact(mod, tmp_path):
    """joint_action／eef_action 逐位等于输入且全部有限；waypoint 要么有限 7 维、要么整条 NaN 哨兵（M14 植入：写入非有限动作）。"""
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
    """真实写入 → 真实 EpisodeDatasetResolver：四种动作空间的序列等于事件表手算结果。"""
    from robomme.env_record_wrapper import EpisodeDatasetResolver

    _, _, path, _ = _main_episode(mod, tmp_path)
    online = [e for e in MAIN_EXPECTED if not e[2]]  # 非演示记录
    with EpisodeDatasetResolver("FakeTask", 3, path) as r:
        joints = _drain(r, "joint_angle")
        ees = _drain(r, "ee_pose")
        wps = _drain(r, "waypoint")
        mcs = _drain(r, "multi_choice")
    assert [j.tolist() for j in joints] == [action_of(e[0]).tolist() for e in online]
    assert [e.tolist() for e in ees] == [[0.0] * 7] * len(online)
    # 非演示记录的 waypoint：NaN、W3、W3、W3 → 跳过哨兵、相邻去重后只剩 W3
    assert len(wps) == 1 and np.allclose(wps[0], W3_ACTION, atol=1e-6)
    # 非演示且为子目标边界的记录：rec3（选项为空，被过滤）、rec5（B）
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
    """同一事件表、低于两者安全上限时，官方与 hard 复制件写出的 h5 逐键逐值相同。"""
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
    # 负例：判定器能看出一处差异
    other = dict(trees["hard"])
    key = "episode_3/timestep_0/action/joint_action"
    dt, shape, val = other[key]
    other[key] = (dt, shape, val + 1.0)
    assert trees_diff(trees["official"], other) == [f"值不同 {key}"]
    # 负例：缺键、属性不同、dtype 不同
    missing = dict(trees["hard"])
    del missing[key]
    assert trees_diff(trees["official"], missing) == [f"缺键 {key}"]
    attrs = dict(trees["hard"])
    attrs["/@attrs"] = {"x": 1}
    assert trees_diff(trees["official"], attrs) == ["属性不同 /@attrs"]
    dtyped = dict(trees["hard"])
    dtyped[key] = ("float32", shape, val.astype(np.float32))
    assert len(trees_diff(trees["official"], dtyped)) == 1 and trees_diff(trees["official"], dtyped)[0].startswith("dtype/shape 不同")


def test_hard_copy_reads_its_own_vqa_options(tmp_path, monkeypatch):
    """两份复制件唯一的 import 差异：close 写 available_multi_choices 时各自读本包的 vqa_options。"""
    import importlib

    hard_vqa = importlib.import_module("robomme_hard.robomme_env.utils.vqa_options")
    off_vqa = importlib.import_module("robomme.robomme_env.utils.vqa_options")
    assert hard_vqa is not off_vqa
    monkeypatch.setattr(hard_vqa, "get_vqa_options", lambda *a: [{"label": "z", "action": "hard only", "available": [1]}])
    got = {}
    for k in KINDS:
        _, _, path, _ = run_episode(record_module(k).RobommeRecordWrapper, tmp_path / k, MAIN_EVENTS)
        with h5py.File(path, "r") as f:
            got[k] = json.loads(f["episode_3/setup/available_multi_choices"][()])
    assert got["hard"] == [{"label": "z", "action": "hard only", "need_parameter": True}]
    assert got["official"] == []  # FakeTask 走官方默认构造器：无选项


# ---------------------------------------------------------------- 成功判定互不替代

SCENARIOS = {
    # 名称: (事件表的 (terminated, truncated, success, task_index) 序列, 是否写 h5)
    "终止且成功": ([(False, False, False, 0), (True, False, True, 0)], True),
    "终止但失败": ([(False, False, False, 0), (True, False, False, 3)], False),
    "子目标全完成但未终止": ([(False, False, False, 1), (False, False, False, 3)], False),
    "截断时 info 成功": ([(False, False, False, 0), (False, True, True, 0)], False),
    "未终止时 info 成功": ([(False, False, True, 0), (False, False, True, 0)], False),
    "先成功后失败终止": ([(True, False, True, 0), (True, False, False, 0)], False),
    "先失败后成功终止": ([(True, False, False, 0), (True, False, True, 0)], True),
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
            assert done == [ti >= 3 for *_, ti in seq]  # is_completed 只看子目标进度
    # 包装器原样透传环境的 terminated／truncated／info
    for (te, tr, su, _), (_, _, ter, trn, info) in zip(seq, rets):
        assert bool(ter.item()) is te and bool(trn.item()) is tr and bool(info["success"].item()) is su
        assert "failsafe_elapsed_steps" not in info


def test_failed_episode_removes_preexisting_group(mod, tmp_path):
    """同一 h5 文件里已有同号 episode 组时：成功局整组替换，失败局删掉旧组。"""
    ok = [Event(name="pick", terminated=True, success=True)]
    run_episode(mod.RobommeRecordWrapper, tmp_path, ok + ok)  # 两条记录
    run_episode(mod.RobommeRecordWrapper, tmp_path, ok)  # 同文件重写：只剩一条
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
    """dataset 传 .h5 文件路径时，输出目录为 <父目录>/<stem>_hdf5_files；缺 dataset 直接拒绝。"""
    env_ev = [Event(name="pick", terminated=True, success=True)]
    from recording_fakes import FakeTaskEnv

    w = mod.RobommeRecordWrapper(FakeTaskEnv(env_ev), dataset=str(tmp_path / "x" / "rec.h5"), env_id="E", episode=1, seed=2)
    assert w.dataset_path == (tmp_path / "x" / "rec_hdf5_files" / "E_ep1_seed2.h5").resolve()
    w.close()
    with pytest.raises(ValueError):
        mod.RobommeRecordWrapper(FakeTaskEnv(env_ev), dataset=None, env_id="E", episode=1, seed=2)


# ---------------------------------------------------------------- 动作形态


def test_seven_dim_action_padded_and_tensor_converted(mod, tmp_path):
    """7 维（stick）动作写入时补夹爪占位 -1；CPU Tensor 转成 NumPy 写入。"""
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
    """FK 替身：末端位姿 = 前三个关节值当位置、单位四元数；记录传入的完整 qpos。"""

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
    # 四步夹爪指令：两个不同的正值、两个不同的负值
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
    assert all(f[0] == f[1] for f in fingers)  # 两指同值
    # 正指令：手指位等于指令值
    assert [fingers[0][0], fingers[1][0]] == [1.0, 0.5]
    # 负指令：取一个与指令大小无关的固定开度，且不同于任一正指令的结果
    assert fingers[2] == fingers[3]
    assert fingers[2][0] not in (1.0, 0.5) and fingers[2][0] >= 0.0


# ---------------------------------------------------------------- 安全上限


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
        assert hi < 1 << 20, "安全上限不存在"
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _probe_failsafe(cls, tmp_path, mid):
            hi = mid
        else:
            lo = mid
    return hi


def test_failsafe_threshold_differs_only_in_hard_copy(tmp_path):
    """行为测得两类的安全上限：hard 复制件严格更大（只放宽、不收紧）；具体数值由常量层钉死。"""
    off = _threshold(record_module("official").RobommeRecordWrapper, tmp_path / "o")
    hard = _threshold(record_module("hard").RobommeRecordWrapper, tmp_path / "h")
    assert 0 < off < hard


def test_failsafe_raises_once_then_truncates_and_drops_success(mod, tmp_path):
    """越过上限：第一次抛 FailsafeTimeout；之后每步返回 truncated=True、terminated=False，成功也不写 h5。"""
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
    # reset 后重新武装：同一包装器再越限会再抛一次
    w2, env2 = make_wrapper(mod.RobommeRecordWrapper, tmp_path / "again", events)
    for _ in range(2):
        w2.reset()
        env2.events, env2.counter = [Event(name="pick", elapsed=lim)], 0
        with pytest.raises(mod.FailsafeTimeout):
            w2.step(torch.from_numpy(action_of(1)))
    w2.h5_file.close()


# ---------------------------------------------------------------- 跨局缓存与关闭


def test_reset_clears_waypoint_and_boundary_caches(mod, tmp_path):
    """reset 清空 waypoint 缓存与子目标边界记忆：第二局首条记录不带上一局的 waypoint、且是边界。"""
    ep1 = [Event(name="pick", task_index=0, waypoint=W3)]
    ep2 = [Event(name="pick", task_index=0, terminated=True, success=True)]
    w, env = make_wrapper(mod.RobommeRecordWrapper, tmp_path, ep1)
    w.reset()
    drive(w, env, ep1)
    assert w._current_waypoint_action is not None
    env.events = ep2
    w.reset()
    assert w._current_waypoint_action is None and w._prev_task_index == -1
    w.buffer.clear()  # 见下一条用例：reset 不清 buffer
    drive(w, env, ep2)
    w.close()
    with h5py.File(w.dataset_path, "r") as f:
        g = f["episode_3/timestep_0"]
        assert np.all(np.isnan(g["action/waypoint_action"][()]))
        assert bool(g["info/is_subgoal_boundary"][()]) is True


def test_reset_without_close_keeps_buffer_and_success_flag(mod, tmp_path):
    """现状记录（冻结代码，只测不改）：reset 不清 buffer 与 episode_success；
    同一包装器不 close 直接开第二局时，上一局的记录与成功标志会混入第二局。"""
    ep1 = [Event(name="pick", terminated=True, success=True)]
    ep2 = [Event(name="pick")]  # 第二局从未终止
    w, env = make_wrapper(mod.RobommeRecordWrapper, tmp_path, ep1)
    w.reset()
    drive(w, env, ep1)
    env.events = ep2
    w.reset()
    assert w.episode_success is True  # 上一局的成功标志被带进第二局
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
    """现状记录：成功局 close 两次时第二次在已关闭的文件上建组而抛 ValueError；已写出的 h5 不受影响。"""
    w, env, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=True)])
    before = read_tree(path)
    with pytest.raises(ValueError):
        w.close()
    assert trees_diff(before, read_tree(path)) == []
    assert w.buffer == [] and w.video_frames == []


def test_save_video_false_records_no_timesteps(mod, tmp_path):
    """现状记录：H5 逐步缓存写在录像分支内，save_video=False 时成功局只剩 setup、没有任何 timestep。
    所有生产入口都传 save_video=True（见 parity/train_split_worker 与官方 generate_dataset）。"""
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, [Event(name="pick", terminated=True, success=True)], save_video=False)
    with h5py.File(path, "r") as f:
        assert list(f["episode_3"].keys()) == ["setup"]
    assert not (tmp_path / "out" / "videos").exists()
