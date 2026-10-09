"""L1 contract: the arguments ``robomme_ood``'s ``BenchmarkEnvBuilder(dataset="ood")`` passes to ``gym.make`` for each of the 800 episodes.

Technique: replace ``gym.make`` with a double that "records positional args and kwargs then raises a sentinel", and call the real ``make_env_for_episode`` per episode,
without building any simulation scene. Expectations come from reading the packaged specs directly with stdlib json: tier order xhard1->xhard5 as the major order, delivered rows within a tier (selected and
rollout ok) concatenated in ascending candidate order as episodes 0..49 (the contract stated in the hard_builder module docstring).

- 16 tasks x 50 episodes = 800; per-episode kwargs = four runtime items + seed + difficulty (tier name) + sampling_config
  (that task's header entry) + native_episode_spec (that row's spec), exactly these keys; ood contains no xhard0;
- omitting ``dataset`` means ood; only the two datasets ``ood``/``hard-verify`` are accepted; official ``train``/``test``/``val``, tier names
  (e.g. ``xhard1``), pre-rename legacy names and spelling variants all raise ``ValueError``; the constructor has no ``specs_root`` parameter (the spec root is packaged only);
- whole spec-root validation (empty root, old schema, edited row without re-signing) is tested directly against ``hard_specs.load_specs_root`` -- the builder no longer accepts external roots.
  Per-episode arguments of hard-verify are in ``test_builder_hard0.py``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.test_constants import (
    DATASETS,
    DEFAULT_DATASET,
    LEGACY_DATASET_NAMES,
    NEW_TIERS,
    PER_TASK,
    RUNTIME,
    TASKS,
    TOTAL,
    TOTAL_HARD_VERIFY,
    V9_CELLS,
    XHARD0,
    XHARD0_PER_TASK,
)

ROOT = REPO / "src" / "robomme_ood" / "env_metadata" / "ood"
EXPECTED_KEYS = {*RUNTIME, "seed", "difficulty", "sampling_config", "native_episode_spec"}


class Sentinel(Exception):
    """Sentinel raised by the gym.make double: proves construction was stopped at gym.make without starting simulation."""


@pytest.fixture
def recorder(monkeypatch):
    from robomme_ood.env_record_wrapper import hard_builder

    calls: list[tuple[tuple, dict]] = []

    def fake_make(*args, **kwargs):
        calls.append((args, kwargs))
        raise Sentinel

    monkeypatch.setattr(hard_builder.gym, "make", fake_make)
    return calls


def builder_cls():
    from robomme_ood.env_record_wrapper.hard_builder import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder


def expected_rows(root: Path = ROOT) -> dict[str, list[tuple[str, dict, dict]]]:
    """{task: [(tier, header, row), ...]}: read the spec files independently, sorted by tier order and ascending candidate."""
    out: dict[str, list] = {task: [] for task in TASKS}
    for tier in NEW_TIERS:
        path = root / tier / "specs.jsonl"
        if not path.is_file():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        header, rows = records[0], records[1:]
        chosen = [r for r in rows if r["selected"] and (r["rollout"] or {}).get("status") == "ok"]
        for row in sorted(chosen, key=lambda r: r["candidate"]):
            out[row["task"]].append((tier, header, row))
    return out


def capture(builder, episode: int, calls: list) -> tuple[tuple, dict]:
    before = len(calls)
    with pytest.raises(Sentinel):
        builder.make_env_for_episode(episode)
    assert len(calls) == before + 1
    return calls[-1]


def newvalue_kwargs(tier: str, header: dict, row: dict, task: str) -> dict:
    return {**RUNTIME, "seed": row["seed"], "difficulty": tier,
            "sampling_config": header["sampling_config"][task], "native_episode_spec": row["spec"]}


def test_builder_800_every_episode(recorder):
    expected = expected_rows()
    total = 0
    for task in TASKS:
        builder = builder_cls()(env_id=task, dataset="ood")
        assert builder.get_episode_num() == PER_TASK == len(expected[task])
        for episode, (tier, header, row) in enumerate(expected[task]):
            args, kwargs = capture(builder, episode, recorder)
            assert args == (task,)
            assert set(kwargs) == EXPECTED_KEYS
            assert kwargs == newvalue_kwargs(tier, header, row, task), (task, episode)
            assert builder.resolve_episode(episode) == (row["seed"], tier)
            ident = builder.resolve_identity(episode)
            assert (ident["tier"], ident["candidate"], ident["seed"], ident["spec_sha256"]) == \
                (tier, row["candidate"], row["seed"], row["spec_sha256"])
            total += 1
        with pytest.raises(KeyError):
            builder.resolve_episode(PER_TASK)
    assert total == TOTAL


def _poison(obj) -> int:
    """Corrupt a nested structure in place: add a marker key to every dict and set every numeric leaf to -999; return the number of edits (proving something was actually changed)."""
    changed = 0
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                obj[key] = -999
                changed += 1
            else:
                changed += _poison(value)
        obj["__poisoned__"] = True
        changed += 1
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                obj[i] = -999
                changed += 1
            else:
                changed += _poison(value)
    return changed


def test_known_defect_builder_kwargs_aliased_to_builder_state(recorder):
    """Lock in current behavior (known defect, registered as D5 in docs/1002-pending-decisions.md; main session decided on 2026-10-04 not to fix it this round):
    make_env_for_episode hands the original sampling_config and row["spec"] objects from the builder's internal lru_cache straight to gym.make,
    so a caller mutating the kwargs of the first build in place pollutes the second build (the spec-side SpecRecorder in the env deepcopies, sampling_config has no such protection).
    Correct behavior: hand out a copy each time, so the second build still equals the independently read spec values. When fixing, invert this test's assertion to `again == expected`."""
    builder = builder_cls()(env_id="StopCube", dataset="ood")
    tier, header, row = expected_rows()["StopCube"][0]
    expected = newvalue_kwargs(tier, header, row, "StopCube")
    _, first = capture(builder, 0, recorder)
    assert first == expected
    keys = ("sampling_config", "native_episode_spec")
    try:
        for key in keys:
            assert _poison(first[key]) > 1
        assert first["sampling_config"] != expected["sampling_config"]
        _, again = capture(builder, 0, recorder)
        # must be decided before restoring: when shared, again and first are the same object and the in-place restore in finally would wash it out too
        aliased = {key: "__poisoned__" in again[key] for key in keys}
    finally:
        # spec reading has a process-level cache (hard_builder's lru_cache): when objects are shared the cache itself is corrupted and must be restored in place,
        # otherwise later tests in the same process would read -999 (top-level object identity unchanged, contents swapped back to an independently read copy)
        for key in keys:
            first[key].clear()
            first[key].update(json.loads(json.dumps(expected[key])))
    # current behavior: both are shared (the corruption marker shows up in the second build's kwargs); if either is fixed to hand out a copy, this assertion fails as a reminder to invert it.
    assert aliased == {"sampling_config": True, "native_episode_spec": True}


def test_default_dataset_is_ood_without_xhard0(recorder):
    """Constructing without ``dataset`` means ood: episode count and per-episode arguments equal an explicit ``dataset="ood"``, and no episode is xhard0."""
    import inspect

    cls = builder_cls()
    assert inspect.signature(cls.__init__).parameters["dataset"].default == DEFAULT_DATASET
    expected = expected_rows()
    for task in TASKS:
        builder = cls(env_id=task)
        assert builder.dataset == DEFAULT_DATASET
        assert builder.get_episode_num() == PER_TASK
        tiers = {builder.resolve_episode(e)[1] for e in range(PER_TASK)}
        assert XHARD0 not in tiers and tiers <= set(NEW_TIERS)
    # compare kwargs per episode for one task, proving the default construction is ood itself
    builder = cls(env_id="StopCube")
    for episode, (tier, header, row) in enumerate(expected["StopCube"]):
        _, kwargs = capture(builder, episode, recorder)
        assert kwargs == newvalue_kwargs(tier, header, row, "StopCube")


def test_two_datasets_episode_totals():
    """Episode-count products for both datasets: hard-verify 16 tasks x 12 = 192, ood 16 tasks x 50 = 800; ood per task equals the row sum of the delivery cell table."""
    cls = builder_cls()
    totals = {dataset: 0 for dataset in DATASETS}
    for task in TASKS:
        for dataset in DATASETS:
            totals[dataset] += cls(env_id=task, dataset=dataset).get_episode_num()
        assert cls(env_id=task, dataset="ood").get_episode_num() == \
            sum(n for (t, _tier), n in V9_CELLS.items() if t == task)
    assert totals == {"hard-verify": TOTAL_HARD_VERIFY, "ood": TOTAL}


def test_no_specs_root_parameter():
    """The constructor has no ``specs_root`` parameter (H1 trim): passing it raises ``TypeError`` for both datasets; ``resolve_identity`` has no such key."""
    import inspect

    cls = builder_cls()
    assert "specs_root" not in inspect.signature(cls.__init__).parameters
    for dataset in DATASETS:
        with pytest.raises(TypeError):
            cls(env_id="StopCube", dataset=dataset, specs_root=ROOT)
    assert "specs_root" not in cls(env_id="StopCube", dataset="ood").resolve_identity(0)


def test_rejections(tmp_path):
    cls = builder_cls()
    # the three official splits and tier names: always rejected by this builder (use the official robomme builder for official behavior)
    for dataset in ("train", "test", "val", "xhard1"):
        with pytest.raises(ValueError):
            cls(env_id="StopCube", dataset=dataset)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="ood", override_metadata_path=tmp_path)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="ood", action_space="torque")
    with pytest.raises(ValueError):
        cls(env_id="NotATask", dataset="ood")
    # hard-verify: accepted (exactly 12 xhard0 episodes per task); spelling variants, pre-rename legacy names, metadata overrides and unknown tasks are all rejected
    assert cls(env_id="StopCube", dataset="hard-verify").get_episode_num() == XHARD0_PER_TASK
    assert len(LEGACY_DATASET_NAMES) == 2
    for wrong in ("hard_verify", "hard-verify0", "xhard0", "Hard-Verify", "OOD", *LEGACY_DATASET_NAMES):
        with pytest.raises(ValueError):
            cls(env_id="StopCube", dataset=wrong)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="hard-verify", override_metadata_path=tmp_path)
    with pytest.raises(ValueError):
        cls(env_id="NotATask", dataset="hard-verify")


# -- Whole spec-root validation (the builder reads only the packaged root; the validator itself is tested on tmp copies) ------------


def partial_root(tmp_path: Path, tiers=("xhard5",)) -> Path:
    root = tmp_path / "root"
    for tier in tiers:
        (root / tier).mkdir(parents=True)
        shutil.copyfile(ROOT / tier / "specs.jsonl", root / tier / "specs.jsonl")
    return root


def _xhard5_cells() -> dict:
    return {key: n for key, n in V9_CELLS.items() if key[1] == "xhard5"}


def _load(root: Path):
    from robomme_ood.env_record_wrapper import hard_specs

    return hard_specs.load_specs_root(root, _xhard5_cells(), cell_table=hard_specs.EXPECTED_CELLS)


def test_specs_root_validation_rejections(tmp_path):
    """A partial root (only xhard5) is readable as-is; an empty root, a tier file that is not /4, and one row edited without re-signing are all rejected by whole-root validation."""
    from robomme_ood.env_record_wrapper import hard_specs

    assert tuple(_load(partial_root(tmp_path / "ok"))) == ("xhard5",)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(hard_specs.SpecsError):
        _load(empty)
    # tier file is not /4
    old = tmp_path / "old"
    (old / "xhard5").mkdir(parents=True)
    lines = (ROOT / "xhard5" / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["schema"] = "hard-specs/3"
    (old / "xhard5" / "specs.jsonl").write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(hard_specs.SpecsError):
        _load(old)
    # one row edited without re-signing: whole-root validation rejects
    bad = partial_root(tmp_path / "bad")
    lines = (bad / "xhard5" / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["seed"] += 1
    lines[1] = json.dumps(row)
    (bad / "xhard5" / "specs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(hard_specs.SpecsError):
        _load(bad)
