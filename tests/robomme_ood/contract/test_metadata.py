"""L1 contract: metadata shipped with the packages.

- official ``src/robomme/env_metadata/{train,val,test}``: exactly 16 files per split (one per task), pinned episode count per file,
  episodes exactly 0..N-1 contiguous, every record's task matches the file name, seeds are integers;
- the hard subset of test is exactly original episodes 3,7,...,47 (the source of xhard0);
- the hard package no longer ships train metadata (H1 trim: only ``ood`` under ``env_metadata``), and the hard builder always rejects ``dataset="train"``;
  official train is still read by the official builder, 100 episodes per task.
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
    """Problem list for one metadata file (empty = compliant)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    out = []
    if not isinstance(records, list) or len(records) != n:
        return [f"episode count {len(records) if isinstance(records, list) else None} != {n}"]
    episodes = [r.get("episode") for r in records]
    if sorted(episodes) != list(range(n)):
        out.append("episode not contiguous")
    if any(r.get("task", payload.get("env_id")) != task for r in records):
        out.append("task mismatch")
    if any(not isinstance(r.get("seed"), int) or isinstance(r.get("seed"), bool) for r in records):
        out.append("seed not an integer")
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
    """The hard package ships only ood specs, no train metadata; the hard builder rejects train, the official builder reads official train with 100 episodes per task."""
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
    """Checker negatives: wrong episode count, episode gap, task mismatch and float seed are all named."""
    good = {"env_id": "BinFill", "records": [{"task": "BinFill", "episode": i, "seed": i} for i in range(3)]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert metadata_problems(path, "BinFill", 3) == []
    assert metadata_problems(path, "BinFill", 4) != []
    for mutate, name in ((lambda r: r[1].__setitem__("episode", 5), "episode not contiguous"),
                         (lambda r: r[2].__setitem__("task", "StopCube"), "task mismatch"),
                         (lambda r: r[0].__setitem__("seed", 1.0), "seed not an integer")):
        bad = json.loads(json.dumps(good))
        mutate(bad["records"])
        path.write_text(json.dumps(bad), encoding="utf-8")
        assert metadata_problems(path, "BinFill", 3) == [name]
