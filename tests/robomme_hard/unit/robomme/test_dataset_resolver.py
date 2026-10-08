"""官方 EpisodeDatasetResolver 的纯读取逻辑（C11 resolver 部分；录制→读回闭环归 T5 的 tests/robomme_hard/pipeline/recording）。

在 tmp_path 手写微型 h5（字段名与官方 doc/h5_data_format.md 一致），期望按手写表逐项得出：
- timestep 按数字排序（timestep_10 在 timestep_9 之后）；演示帧（info/is_video_demo）一律不进在线索引；
- joint：7 维补 -1 到 8 维、"None" 字符串 → None；
- waypoint：只去「相邻重复」（A,A,B,A → A,B,A），非有限或形状不是 (7,) 的跳过且不打断相邻比较；
- multi_choice：只取 is_subgoal_boundary 为真且 JSON 合法（choice 非空字符串、带 point）的帧，按时间序；
- 越界／负下标／未知模式 → None；缺文件 FileNotFoundError；缺 episode KeyError；close 幂等。
"""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from robomme.env_record_wrapper.episode_dataset_resolver import EpisodeDatasetResolver, list_episode_indices

ENV = "PickXtimes"
WP_A = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
WP_B = [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, -1.0]


def _ts(ep, idx, *, demo=False, joint=None, eef=None, waypoint=None, boundary=None, choice=None):
    g = ep.create_group(f"timestep_{idx}")
    info = g.create_group("info")
    info["is_video_demo"] = demo
    if boundary is not None:
        info["is_subgoal_boundary"] = boundary
    act = g.create_group("action")
    if joint is not None:
        act["joint_action"] = joint
    if eef is not None:
        act["eef_action"] = np.asarray(eef, dtype=np.float64)
    if waypoint is not None:
        act["waypoint_action"] = np.asarray(waypoint, dtype=np.float64)
    if choice is not None:
        act["choice_action"] = choice


@pytest.fixture
def h5file(tmp_path):
    path = tmp_path / f"record_dataset_{ENV}.h5"
    with h5py.File(path, "w") as f:
        ep = f.create_group("episode_0")
        _ts(ep, 0, demo=True, joint=np.full(8, 9.0), waypoint=WP_B)                      # 演示帧：全部忽略
        _ts(ep, 1, joint=np.arange(7, dtype=np.float64), eef=[1, 2, 3, 4, 5, 6, -1], waypoint=WP_A,
            boundary=True, choice=json.dumps({"choice": "a", "point": [10, 20]}))
        _ts(ep, 2, joint="None", waypoint=WP_A, boundary=False, choice=json.dumps({"choice": "b", "point": [1, 2]}))
        _ts(ep, 9, joint=np.arange(8, dtype=np.float64), waypoint=[np.nan] * 7, boundary=True,
            choice=json.dumps({"choice": "", "point": [0, 0]}))                           # 空 choice 不收
        _ts(ep, 10, joint=np.arange(3, dtype=np.float64), waypoint=WP_B, boundary=True,
            choice=json.dumps({"choice": "c"}))                                           # 缺 point 不收
        _ts(ep, 11, waypoint=[1.0] * 6, boundary=True, choice="not json")                 # 形状不对、非 JSON
        _ts(ep, 12, waypoint=WP_A, boundary=True, choice=json.dumps({"choice": "c", "point": None}))
        f.create_group("episode_3")
        f.create_group("episode_10")
        f.create_group("not_an_episode")
    return path


def test_list_episode_indices_numeric_sorted(h5file):
    assert list_episode_indices(ENV, h5file.parent) == [0, 3, 10]
    assert list_episode_indices(ENV, h5file) == [0, 3, 10]  # 直接给 .h5 路径也行


def test_joint_steps_skip_demo_and_pad(h5file):
    with EpisodeDatasetResolver(ENV, 0, h5file.parent) as r:
        np.testing.assert_array_equal(r.get_step("joint_angle", 0), [0, 1, 2, 3, 4, 5, 6, -1])
        assert r.get_step("joint_angle", 1) is None                      # "None"
        np.testing.assert_array_equal(r.get_step("joint_angle", 2), np.arange(8))   # timestep_9
        np.testing.assert_array_equal(r.get_step("joint_angle", 3), [0, 1, 2, -1, -1, -1, -1, -1])  # timestep_10
        assert r.get_step("joint_angle", 4) is None                      # timestep_11 无 joint
        assert r.get_step("joint_angle", 99) is None and r.get_step("joint_angle", -1) is None


def test_ee_pose_reads_eef_action(h5file):
    with EpisodeDatasetResolver(ENV, 0, h5file.parent) as r:
        np.testing.assert_array_equal(r.get_step("ee_pose", 0), [1, 2, 3, 4, 5, 6, -1])
        assert r.get_step("ee_pose", 1) is None


def test_waypoint_adjacent_dedup_only(h5file):
    with EpisodeDatasetResolver(ENV, 0, h5file.parent) as r:
        got = [r.get_step("waypoint", i) for i in range(4)]
    # 在线帧依次：A(1) A(2) nan(9) B(10) 坏形状(11) A(12) → 相邻去重 A,B,A
    np.testing.assert_array_equal(got[0], WP_A)
    np.testing.assert_array_equal(got[1], WP_B)
    np.testing.assert_array_equal(got[2], WP_A)
    assert got[3] is None


def test_multi_choice_boundary_and_valid_json_only(h5file):
    with EpisodeDatasetResolver(ENV, 0, h5file.parent) as r:
        cmds = [r.get_step("multi_choice", i) for i in range(3)]
        first = r.get_step("multi_choice", 0)
        first["choice"] = "mutated"
        assert r.get_step("multi_choice", 0)["choice"] == "a"  # 返回副本
    assert cmds[0] == {"choice": "a", "point": [10, 20]}
    assert cmds[1] == {"choice": "c", "point": None}   # 有 point 键即可（值可为 None）
    assert cmds[2] is None


def test_unknown_mode_is_none(h5file):
    with EpisodeDatasetResolver(ENV, 0, h5file.parent) as r:
        assert r.get_step("joint", 0) is None


def test_missing_file_and_episode(tmp_path, h5file):
    with pytest.raises(FileNotFoundError):
        EpisodeDatasetResolver(ENV, 0, tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError):
        list_episode_indices(ENV, tmp_path / "nowhere")
    with pytest.raises(KeyError, match="episode_7"):
        EpisodeDatasetResolver(ENV, 7, h5file.parent)
    with h5py.File(h5file, "a"):  # 缺 episode 时已关闭文件：能以追加模式再打开
        pass


def test_close_idempotent(h5file):
    r = EpisodeDatasetResolver(ENV, 0, h5file.parent)
    r.close()
    r.close()
    assert r._h5 is None


def test_numeric_timestep_order(tmp_path):
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as f:
        ep = f.create_group("episode_0")
        for idx in (10, 2, 1):
            _ts(ep, idx, joint=np.full(8, float(idx)))
    with EpisodeDatasetResolver(ENV, 0, path) as r:
        assert [int(r.get_step("joint_angle", i)[0]) for i in range(3)] == [1, 2, 10]
