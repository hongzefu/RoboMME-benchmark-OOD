"""官方 BenchmarkEnvBuilder（C02 官方部分）：白名单、元数据解析、交给 gym.make 的参数、包装链与步数上限。

gym.make 换成记录参数的替身（只换 episode_config_resolver 模块里的 ``gym`` 名字，进程内、不落盘），
其余（DemonstrationWrapper 等四种包装层与 FailAwareWrapper）全是真实类。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robomme.env_record_wrapper import episode_config_resolver as ecr
from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder

from _official_fakes import GymMakeSpy, find_wrapper, wrapper_chain
from tests.robomme_hard.unit.robomme import official_thresholds as T

ACTION_SPACES = ("joint_angle", "ee_pose", "waypoint", "multi_choice")
SPLITS = ("train", "val", "test")
# 独立期望：四种动作空间的包装链（外 → 内），来源：官方 doc/env_format.md 的动作空间说明与 README 动作类型
CHAINS = {
    "joint_angle": ["FailAwareWrapper", "DemonstrationWrapper", "OrderEnforcing", "FakeTaskEnv"],
    "ee_pose": ["FailAwareWrapper", "EndeffectorDemonstrationWrapper", "DemonstrationWrapper", "OrderEnforcing", "FakeTaskEnv"],
    "waypoint": ["FailAwareWrapper", "MultiStepDemonstrationWrapper", "DemonstrationWrapper", "OrderEnforcing", "FakeTaskEnv"],
    "multi_choice": ["FailAwareWrapper", "OraclePlannerDemonstrationWrapper", "DemonstrationWrapper", "OrderEnforcing", "FakeTaskEnv"],
}
FLAGS = (
    "include_maniskill_obs", "include_front_depth", "include_wrist_depth", "include_front_camera_extrinsic",
    "include_wrist_camera_extrinsic", "include_available_multi_choices", "include_front_camera_intrinsic",
    "include_wrist_camera_intrinsic",
)


@pytest.fixture
def gym_spy(monkeypatch):
    spy = GymMakeSpy()
    monkeypatch.setattr(ecr, "gym", spy.namespace())
    return spy


def _metadata_tasks(split):
    return {p.name[len("record_dataset_"):-len("_metadata.json")] for p in (ecr.DATASET_ROOT / split).glob("*.json")}


# --------------------------------------------------------------------------- 白名单


@pytest.mark.parametrize("dataset", ["ood", "TEST", "", "dev"])
def test_rejects_unknown_dataset(dataset):
    with pytest.raises(ValueError, match="Unsupported dataset"):
        BenchmarkEnvBuilder("PickXtimes", dataset=dataset)


@pytest.mark.parametrize("space", ["joint", "eef", "Waypoint", ""])
def test_rejects_unknown_action_space(space):
    with pytest.raises(ValueError, match="Unsupported action_space"):
        BenchmarkEnvBuilder("PickXtimes", action_space=space)


@pytest.mark.parametrize("split", SPLITS)
def test_accepts_official_splits(split):
    assert BenchmarkEnvBuilder("PickXtimes", dataset=split).dataset == split


# --------------------------------------------------------------------------- 任务列表


def test_task_list_matches_metadata_files_and_is_a_copy():
    tasks = BenchmarkEnvBuilder.get_task_list()
    assert len(tasks) == len(set(tasks))  # 个数由下面与三个 split 元数据文件名集合相等来钉
    for split in SPLITS:
        assert set(tasks) == _metadata_tasks(split), split
    tasks.append("Injected")
    assert "Injected" not in BenchmarkEnvBuilder.get_task_list()
    assert BenchmarkEnvBuilder.get_task_list() == BenchmarkEnvBuilder.get_task_list()


# --------------------------------------------------------------------------- 元数据解析


def _write_meta(tmp_path, env_id, payload):
    p = tmp_path / f"record_dataset_{env_id}_metadata.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_metadata_resolution_rules(tmp_path):
    root = _write_meta(tmp_path, "PickXtimes", {
        "env_id": "PickXtimes",
        "records": [
            {"task": "PickXtimes", "episode": 0, "seed": 7, "difficulty": "hard"},
            {"episode": "1", "seed": "not-int", "difficulty": "easy"},   # 缺 task 用 env_id；seed 非法 → None
            {"task": "PickXtimes", "episode": None, "seed": 9},           # 无 episode → 跳过
            {"task": "PickXtimes", "episode": "x", "seed": 9},            # episode 非整数 → 跳过
            {"task": "OtherTask", "episode": 2, "seed": 5},               # 别的任务
            {"task": "PickXtimes", "episode": 3, "seed": "11"},           # 字符串数字 seed → 11
        ],
    })
    b = BenchmarkEnvBuilder("PickXtimes", override_metadata_path=root)
    assert b.resolve_episode(0) == (7, "hard")
    assert b.resolve_episode(1) == (None, "easy")
    assert b.resolve_episode(2) == (None, None)
    assert b.resolve_episode(3) == (11, None)
    assert b.resolve_episode(99) == (None, None)
    assert b.get_episode_num() == 3  # 0、1、3


def test_missing_or_corrupt_metadata_is_empty(tmp_path):
    b = BenchmarkEnvBuilder("PickXtimes", override_metadata_path=tmp_path)
    assert b.metadata_index == {} and b.get_episode_num() == 0 and b.resolve_episode(0) == (None, None)
    (tmp_path / "record_dataset_PickXtimes_metadata.json").write_text("{not json", encoding="utf-8")
    b = BenchmarkEnvBuilder("PickXtimes", override_metadata_path=tmp_path)
    assert b.metadata_index == {}


@pytest.mark.parametrize("split", SPLITS)
def test_real_metadata_episode0_seed(split):
    """真实元数据：逐任务用 json 独立读第 0 局记录，与 resolve_episode 的结果相同。"""
    for task in BenchmarkEnvBuilder.get_task_list():
        payload = json.loads((ecr.DATASET_ROOT / split / f"record_dataset_{task}_metadata.json").read_text())
        rec = next(r for r in payload["records"] if int(r["episode"]) == 0)
        seed, diff = BenchmarkEnvBuilder(task, dataset=split).resolve_episode(0)
        assert seed == int(rec["seed"]) and diff == rec.get("difficulty"), (split, task)
        assert diff in ("easy", "medium", "hard")


# --------------------------------------------------------------------------- 交给 gym.make 的参数与包装链


@pytest.mark.parametrize("space", ACTION_SPACES)
def test_make_env_kwargs_and_chain(tmp_path, gym_spy, space):
    root = _write_meta(tmp_path, "PickXtimes", {"records": [{"task": "PickXtimes", "episode": 4, "seed": 123,
                                                             "difficulty": "medium"}]})
    env = BenchmarkEnvBuilder("PickXtimes", action_space=space, override_metadata_path=root).make_env_for_episode(4)
    env_id, kwargs = gym_spy.calls[-1]
    assert env_id == "PickXtimes"
    assert kwargs == {**T.GYM_MAKE_FIXED_KWARGS, "seed": 123, "difficulty": "medium"}
    assert wrapper_chain(env) == CHAINS[space]
    assert env.unwrapped.use_demonstrationwrapper is True
    if space == "ee_pose":
        assert find_wrapper(env, "EndeffectorDemonstrationWrapper").action_repr == "rpy"
    if space == "multi_choice":
        assert find_wrapper(env, "OraclePlannerDemonstrationWrapper").env_id == "PickXtimes"


def test_make_env_without_metadata_omits_seed_and_difficulty(tmp_path, gym_spy):
    BenchmarkEnvBuilder("PickXtimes", override_metadata_path=tmp_path).make_env_for_episode(0)
    _, kwargs = gym_spy.calls[-1]
    assert "seed" not in kwargs and "difficulty" not in kwargs


def test_gui_render_switches_render_mode(tmp_path, gym_spy):
    BenchmarkEnvBuilder("PickXtimes", gui_render=True, override_metadata_path=tmp_path).make_env_for_episode(0)
    assert gym_spy.calls[-1][1]["render_mode"] == "human"


@pytest.mark.parametrize("builder_max, call_max, expected", [(10000, None, 10002), (300, None, 302), (300, 5, 7),
                                                              (300, 0, 2)])
def test_max_steps_plus_two(tmp_path, gym_spy, builder_max, call_max, expected):
    b = BenchmarkEnvBuilder("PickXtimes", max_steps=builder_max, override_metadata_path=tmp_path)
    assert b.max_steps_without_demonstration == builder_max + 2
    env = b.make_env_for_episode(0, max_steps=call_max)
    assert find_wrapper(env, "DemonstrationWrapper").max_steps_without_demonstration == expected


def _flag_combos_18():
    """全关、全开、8 个单开、8 个单关。"""
    off = dict.fromkeys(FLAGS, False)
    on = dict.fromkeys(FLAGS, True)
    out = [off, on]
    out += [{**off, f: True} for f in FLAGS]
    out += [{**on, f: False} for f in FLAGS]
    return out


def _expected_flags(space, flags):
    forced = space == "multi_choice"  # multi_choice 强制带前视相机内外参
    exp = dict(flags)
    exp["include_front_camera_extrinsic"] = flags["include_front_camera_extrinsic"] or forced
    exp["include_front_camera_intrinsic"] = flags["include_front_camera_intrinsic"] or forced
    return exp


@pytest.mark.parametrize("space", ACTION_SPACES)
@pytest.mark.parametrize("combo", range(18))
def test_include_flags_passed_through_18(tmp_path, gym_spy, space, combo):
    flags = _flag_combos_18()[combo]
    env = BenchmarkEnvBuilder("PickXtimes", action_space=space,
                              override_metadata_path=tmp_path).make_env_for_episode(0, **flags)
    demo = find_wrapper(env, "DemonstrationWrapper")
    assert {f: getattr(demo, f) for f in FLAGS} == _expected_flags(space, flags)


# 四种动作空间 × 256 全组合（含输出键核对）在 test_obs_switches.py::test_switches_all_256_through_builder（slow）


def test_override_metadata_path_wins_over_dataset(tmp_path):
    root = _write_meta(tmp_path, "PickXtimes", {"records": [{"task": "PickXtimes", "episode": 0, "seed": 1,
                                                             "difficulty": "easy"}]})
    b = BenchmarkEnvBuilder("PickXtimes", dataset="test", override_metadata_path=root)
    assert Path(b._resolve_metadata_path()) == root / "record_dataset_PickXtimes_metadata.json"
    assert b.resolve_episode(0) == (1, "easy")
