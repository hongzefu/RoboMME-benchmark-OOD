"""L1 契约：随包元数据。

- 官方 ``src/robomme/env_metadata/{train,val,test}``：每 split 恰 16 个文件（16 任务各一）、每文件局数为钉值、
  episode 恰为 0..N−1 连续、每条记录 task 与文件名一致、seed 为整数；
- test 的 hard 子集恰为原 episode 3,7,…,47（xhard0 的来源）；
- hard 包不再带 train 元数据（H1 裁剪：``env_metadata`` 下只有 ``ood``），hard 构建器对 ``dataset="train"`` 一律拒绝；
  官方 train 仍由官方构建器读，每任务 100 局。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.test_constants import (
    OFFICIAL_SPLIT_EPISODES,
    OFFICIAL_SPLIT_FILES,
    TASKS,
    XHARD0_EPISODES,
)

OFFICIAL = REPO / "src" / "robomme" / "env_metadata"
HARD_META = REPO / "src" / "robomme_ood" / "env_metadata"


def metadata_problems(path: Path, task: str, n: int) -> list[str]:
    """一个元数据文件的问题清单（空 = 合规）。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    out = []
    if not isinstance(records, list) or len(records) != n:
        return [f"局数 {len(records) if isinstance(records, list) else None} ≠ {n}"]
    episodes = [r.get("episode") for r in records]
    if sorted(episodes) != list(range(n)):
        out.append("episode 不连续")
    if any(r.get("task", payload.get("env_id")) != task for r in records):
        out.append("task 不符")
    if any(not isinstance(r.get("seed"), int) or isinstance(r.get("seed"), bool) for r in records):
        out.append("seed 不是整数")
    return out


@pytest.mark.parametrize("split", sorted(OFFICIAL_SPLIT_EPISODES))
def test_official_split(split):
    files = sorted((OFFICIAL / split).glob("*.json"))
    assert len(files) == OFFICIAL_SPLIT_FILES
    assert {p.name for p in files} == {f"record_dataset_{t}_metadata.json" for t in TASKS}
    bad = {t: p for t in TASKS
           if (p := metadata_problems(OFFICIAL / split / f"record_dataset_{t}_metadata.json", t,
                                      OFFICIAL_SPLIT_EPISODES[split]))}
    assert bad == {}


@pytest.mark.parametrize("task", TASKS)
def test_official_test_hard_subset_is_xhard0_source(task):
    records = json.loads((OFFICIAL / "test" / f"record_dataset_{task}_metadata.json").read_text(encoding="utf-8"))
    hard = sorted(int(r["episode"]) for r in records["records"] if r["difficulty"] == "hard")
    assert tuple(hard) == XHARD0_EPISODES


def test_no_hard_train_metadata():
    """hard 包只带 ood 规格，不带 train 元数据；hard 构建器拒绝 train，官方构建器读官方 train 每任务 100 局。"""
    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder as Official
    from robomme_ood.env_record_wrapper import hard_builder

    assert not (HARD_META / "train").exists()
    assert sorted(p.name for p in HARD_META.iterdir()) == ["ood"]
    assert not hasattr(hard_builder, "HARD_TRAIN_TASKS")
    for task in TASKS:
        with pytest.raises(ValueError):
            hard_builder.BenchmarkEnvBuilder(env_id=task, dataset="train")
        assert Official(env_id=task, dataset="train").get_episode_num() == OFFICIAL_SPLIT_EPISODES["train"], task


def test_metadata_problems_negative(tmp_path):
    """判定器负例：局数不符、episode 断号、task 不符、seed 为浮点都被点名。"""
    good = {"env_id": "BinFill", "records": [{"task": "BinFill", "episode": i, "seed": i} for i in range(3)]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert metadata_problems(path, "BinFill", 3) == []
    assert metadata_problems(path, "BinFill", 4) != []
    for mutate, name in ((lambda r: r[1].__setitem__("episode", 5), "episode 不连续"),
                         (lambda r: r[2].__setitem__("task", "StopCube"), "task 不符"),
                         (lambda r: r[0].__setitem__("seed", 1.0), "seed 不是整数")):
        bad = json.loads(json.dumps(good))
        mutate(bad["records"])
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert metadata_problems(path, "BinFill", 3) == [name]
