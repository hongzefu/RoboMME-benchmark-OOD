"""L1 contract: ``BenchmarkEnvBuilder(dataset="hard-verify")`` contains only xhard0 and does not read the spec root; the per-tier step lookup table is removed.

``hard-verify`` (1003 evaluation plan 1.1) = the 12 episodes per task with ``difficulty=="hard"`` in the official test metadata (original episodes
3, 7, ..., 47), renumbered as episodes 0..11 in ascending original-episode order (the xhard0 switch ``XHARD0_IN_TEST_HARD`` was removed with the H1 trim).

Same technique as ``test_builder_800.py``: replace ``gym.make`` with a double that "records arguments then raises a sentinel", and call the real
``make_env_for_episode`` per episode without starting any simulation. Expected values come only from hand-written pins (``test_constants``) and the official
test metadata read directly with stdlib json, never from the code under test. All spec reader functions are stubbed to "fail and count on call", proving hard-verify reads no specs.

- 16 tasks x 12 episodes = 192; per-episode kwargs are exactly the four runtime items + official seed + ``difficulty="hard"``;
- ``resolve_identity`` field set and values (no candidate/spec digest, no ``specs_root``);
- a broken official hard subset (one episode missing, one extra, duplicate seed) errors at construction;
- the constructor has no ``specs_root`` parameter; passing it raises ``TypeError``;
- no .py under ``src/``, ``scripts/`` or ``tests/`` references ``TIER_MAX_STEPS`` any more (except the S2 pending-cleanup list, see below).
"""
from __future__ import annotations

import io
import json
import tokenize
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.test_constants import (
    RUNTIME,
    TASKS,
    XHARD0,
    XHARD0_EPISODES,
    XHARD0_PER_TASK,
)

OFFICIAL_TEST = REPO / "src" / "robomme" / "env_metadata" / "test"
#: hard-verify full set: 16 tasks x 12 episodes (computed by hand)
TOTAL_HARD0 = 192
N_TASKS_HARD0 = 16
#: Field set of resolve_identity for xhard0 episodes (hand-written)
IDENTITY_KEYS = {"episode", "tier", "candidate", "seed", "source_dataset", "source_episode", "spec_sha256", "source_run"}
#: Spec reader entry points stubbed to "fail on call": (module name, attribute name)
SPEC_READERS = (
    ("hard_specs", "load_specs_root"),
    ("hard_specs", "specs_root"),
    ("hard_specs", "packaged_specs_path"),
    ("hard_builder", "_packaged_specs"),
    ("hard_builder", "_ood_entries"),
)


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


@pytest.fixture
def spec_reads(monkeypatch, tmp_path):
    """Replace all spec reader entry points with stubs that "count then raise"; return the call record."""
    from robomme_ood.env_record_wrapper import hard_builder, hard_specs

    modules = {"hard_specs": hard_specs, "hard_builder": hard_builder}
    reads: list[str] = []
    for module_name, attr in SPEC_READERS:
        def boom(*_args, _name=f"{module_name}.{attr}", **_kwargs):
            reads.append(_name)
            raise AssertionError(f"hard-verify must not read specs: called {_name}")

        monkeypatch.setattr(modules[module_name], attr, boom)
    return reads


def builder_cls():
    from robomme_ood.env_record_wrapper.hard_builder import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder


def official_hard(task: str) -> list[dict]:
    """Read the official test metadata directly with stdlib json and take the hard subset in ascending original-episode order."""
    payload = json.loads((OFFICIAL_TEST / f"record_dataset_{task}_metadata.json").read_text(encoding="utf-8"))
    return sorted((r for r in payload["records"] if r["difficulty"] == "hard"), key=lambda r: int(r["episode"]))


def capture(builder, episode: int, calls: list) -> tuple[tuple, dict]:
    before = len(calls)
    with pytest.raises(Sentinel):
        builder.make_env_for_episode(episode, max_steps=1300)
    assert len(calls) == before + 1
    return calls[-1]


def test_hard_verify_every_episode(recorder, spec_reads):
    """Per-episode gym.make arguments and identity for 16 tasks; specs are never read."""
    total = 0
    tasks = 0
    for task in TASKS:
        builder = builder_cls()(env_id=task, dataset="hard-verify", action_space="joint_angle", max_steps=1300)
        assert builder.dataset == "hard-verify"
        assert builder.metadata_index == {}
        assert builder.get_episode_num() == XHARD0_PER_TASK
        hard = official_hard(task)
        assert tuple(int(r["episode"]) for r in hard) == XHARD0_EPISODES
        for episode, record in enumerate(hard):
            args, kwargs = capture(builder, episode, recorder)
            assert args == (task,)
            assert kwargs == {**RUNTIME, "seed": int(record["seed"]), "difficulty": "hard"}, (task, episode)
            assert builder.resolve_episode(episode) == (int(record["seed"]), XHARD0)
            ident = builder.resolve_identity(episode)
            assert set(ident) == IDENTITY_KEYS, ident
            assert ident == {"episode": episode, "tier": XHARD0, "candidate": None, "seed": int(record["seed"]),
                             "source_dataset": "test", "source_episode": int(record["episode"]),
                             "spec_sha256": None, "source_run": None}
            total += 1
        with pytest.raises(KeyError):
            builder.resolve_episode(XHARD0_PER_TASK)
        tasks += 1
    assert spec_reads == []
    assert (tasks, total) == (N_TASKS_HARD0, TOTAL_HARD0)
    print(f"HARD0_INTERFACE=PASS tasks={tasks} per_task={XHARD0_PER_TASK} total={total} specs_reads={len(spec_reads)}")


def test_hard_verify_rejects_specs_root(spec_reads, tmp_path):
    cls = builder_cls()
    for root in (tmp_path, str(tmp_path), REPO / "src" / "robomme_ood" / "env_metadata" / "ood"):
        with pytest.raises(TypeError, match="specs_root"):
            cls(env_id="StopCube", dataset="hard-verify", specs_root=root)
    with pytest.raises(ValueError):
        cls(env_id="StopCube", dataset="hard-verify", override_metadata_path=tmp_path)
    with pytest.raises(ValueError):
        cls(env_id="NotATask", dataset="hard-verify")
    assert spec_reads == []


def _broken_loader(monkeypatch, how: str):
    """Wrap the parent's official test metadata reader and apply the breakage given by ``how`` to BinFill's hard subset."""
    from robomme.env_record_wrapper import episode_config_resolver as official

    original = official.load_episode_metadata

    def broken(path):
        index = dict(original(path))
        hard = sorted((key for key, rec in index.items() if key[0] == "BinFill" and rec.get("difficulty") == "hard"),
                      key=lambda key: key[1])
        if not hard:  # metadata file of another task: return unchanged
            return index
        if how == "drop":
            del index[hard[-1]]
        elif how == "extra":
            easy = next(key for key, rec in index.items() if key[0] == "BinFill" and rec.get("difficulty") != "hard")
            index[easy] = dict(index[easy], difficulty="hard")
        elif how == "dup_seed":
            index[hard[1]] = dict(index[hard[1]], seed=index[hard[0]]["seed"])
        return index

    monkeypatch.setattr(official, "load_episode_metadata", broken)


@pytest.mark.parametrize("how", ["drop", "extra", "dup_seed"])
def test_hard_verify_rejects_broken_official_subset(monkeypatch, spec_reads, how):
    """The official hard subset must be exactly original episodes 3,7,...,47 with unique seeds: one missing, one extra or a duplicate seed all error at construction."""
    _broken_loader(monkeypatch, how)
    with pytest.raises(ValueError, match="xhard0"):
        builder_cls()(env_id="BinFill", dataset="hard-verify")
    # other tasks are unaffected
    assert builder_cls()(env_id="StopCube", dataset="hard-verify").get_episode_num() == XHARD0_PER_TASK
    assert spec_reads == []


# -- Per-tier step lookup table removed: no references anywhere in the repo -----------------------------------------

#: Transitional exemption list. S2 (evaluation client, manifest, report) was merged in 12.437 and cleared all lookup-table references, so the list is now empty;
#: the scan covers all .py under src/, scripts/ and tests/. The empty set is kept only so historical commits stay readable; do not add files to it.
S2_PENDING: frozenset[str] = frozenset()
_LOOKUP_NAME = "TIER_MAX_STEPS"


def _name_refs(path: Path) -> int:
    """Number of identifiers named after the lookup table in a file (NAME tokens): counts only code references, not text in strings or comments."""
    source = path.read_text(encoding="utf-8")
    return sum(1 for tok in tokenize.generate_tokens(io.StringIO(source).readline)
               if tok.type == tokenize.NAME and tok.string == _LOOKUP_NAME)


def test_no_tier_max_steps_refs():
    refs: dict[str, int] = {}
    scanned = 0
    for top in ("src", "scripts", "tests"):
        for path in sorted((REPO / top).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if rel in S2_PENDING:
                continue
            scanned += 1
            count = _name_refs(path)
            if count:
                refs[rel] = count
    assert scanned > 0
    assert refs == {}, refs
    print(f"STEP_LOOKUP=PASS tier_max_steps_refs={sum(refs.values())}")
