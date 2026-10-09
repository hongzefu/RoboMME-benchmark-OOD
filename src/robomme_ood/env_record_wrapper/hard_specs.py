"""Reading, envelope validation and re-injection binding summary for the ood four-tier specs (``env_metadata/ood/xhardN/specs.jsonl``).

Moved down from the V4 spec read/write module of the old repo's parity tooling (0927 plan part two §1.2); drawing and freezing belong to
the old repo's generation pipeline and are not shipped with this package. This module only holds pure functions needed to read and validate specs; it does not import the simulator.

Each jsonl row has two segments, "signature" and "result" (0927 plan part one §5.3; the only current format is ``schema="hard-specs/4"``):

* signature: ``task tier candidate episode seed attempt spec spec_sha256 layout_parent`` -- sealed in stage one, covered by ``identity_sha256``;
* result: ``selected tried initial_selected rollout`` -- written back in stage two, not part of ``identity_sha256``.

``delivery_sha256`` additionally seals "which episodes are the formal delivery": the sorted
``(task, tier, candidate, seed, spec_sha256, rollout.h5_sha256)``, taking only rows that are ``selected`` and have ``rollout.status=="ok"``.

``ood`` always contains only the five new-value tiers xhard1-xhard5 (16 tasks x 50 episodes = 800); xhard0 (the 12 hard episodes of the official test split)
appears only in ``hard-verify`` (16 tasks x 12 episodes = 192) and is no longer prepended to ``ood``.

``seed_for``, ``V8_SEED_OFFSETS``, ``identity_sha256`` and ``delivery_sha256`` are validation dependencies: ``load_specs`` recomputes the seed formula
and both identity hashes from them when reading the packaged specs, so they are kept (they are not generation entry points).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

#: The only current spec format (v8 plan part two §2.2 item 2, R6): the header carries ``layout_rule`` and ``exec_cap``;
#: ``delivery_per_cell``, ``select_rule`` and ``per_env`` are per-task dicts; rows carry ``layout_parent`` (always null).
#: Read/validate support for the old formats /2 (V5-V6 single tier) and /3 (V7 parent-layout derivation) was removed in maintenance plan stage 1b (W4).
SCHEMA = "hard-specs/4"
#: New-value tiers (excluding xhard0). Since v8 stage 3b repackaging these are the five tiers xhard1-xhard5 (same value as the former ``V8_TIERS``, now merged into this constant).
#: EXPECTED_CELLS, packaged_specs_path, the builder, /4 validation and spec-root reading depend on it.
TIERS = ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5")
#: Builder tier order: xhard0 (hard subset of the official test split, native branch, no re-injection) first, followed by the five new-value tiers (six entries in total).
XHARD0 = "xhard0"
BUILDER_TIERS = (XHARD0, *TIERS)
#: 12 xhard0 episodes per task = the original episodes 3,7,...,47 with difficulty=="hard" in the official test metadata (reference value only; filtering is by difficulty)
XHARD0_PER_TASK = 12
XHARD0_EPISODES = tuple(range(3, 48, 4))
#: Legacy V4/V5 single-tier name. The v5 seed rule is removed; the train-split runner in the old repo's parity tooling still uses it as the expected
#: difficulty when jobs carry no seed_rule (V9/xhard0 generation jobs all carry seed_rule or take the official-metadata branch and never hit that default), so the constant is kept.
DIFFICULTY = "xhard"
#: Re-injection binding: allowed float difference for record-only observations (SpecRecorder.record) (user U-13 option A, red line R22, not a parameter).
RECORDED_FLOAT_TOL = 1e-5

# The four runtime items match the gym.make arguments verbatim (the builder compares them verbatim when creating the env; render_mode is exempt)
RUNTIME = {
    "obs_mode": "rgb+depth+segmentation",
    "control_mode": "pd_joint_pos",
    "render_mode": "rgb_array",
    "reward_mode": "dense",
}
# seed = offset + env_code x env_block + episode x 100 + attempt; env_code is the task's 1-indexed position in the 16-task canonical order
SEED_RULE = {"offset": 4_000_000, "env_block": 100_000, "episode_stride": 100,
             "formula": "offset + env_code*env_block + episode*100 + attempt"}
#: V8 per-tier seed offsets (v8 plan part two §2.2 item 3, R9): each tier draws its layouts independently, and tier seeds are pairwise disjoint.
#: Max increment per tier 16x100000 + 999x100 + 99 < 1.7e6, below the stride 2e6; the lowest 16e6 avoids V5 (from 4e6),
#: the old V6 segment (6e6-13.7e6) and the V7/probe segment (14e6-15.7e6).
V8_SEED_OFFSETS = {"xhard1": 16_000_000, "xhard2": 18_000_000, "xhard3": 20_000_000,
                   "xhard4": 22_000_000, "xhard5": 24_000_000}
#: Per-tier offset rule families: profile -> {tier: offset}. Currently only v8 (reused by V9); the v5, v6 and v7 families are removed.
#: A per-tier offset family only accepts the tiers registered in that family; newly registered offsets must not overlap the historical segments (4e6, 6e6-15.7e6) or any registered family.
TIER_SEED_OFFSETS: dict[str, dict[str, int]] = {"v8": V8_SEED_OFFSETS}
SEED_PROFILES = tuple(TIER_SEED_OFFSETS)
MAX_ATTEMPTS = 100
#: 16-task canonical order (verbatim identical to ``ALL_TASKS`` in the old repo's generation-pipeline seed-formula module; env_code is its 1-indexed position)
ALL_TASKS = (
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap", "VideoUnmask",
    "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "PickHighlight", "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
)
#: Tasks delivered only in xhard4 (since v8 stage 3b: StopCube left after being split into five fixed-value tiers, leaving only InsertPeg and MoveCube)
XHARD4_ONLY = ("InsertPeg", "MoveCube")

# -- Execution-step cap (added in v8 plan stage 2, reused by V9; maintenance plan block R dropped the V8 prefix from the name) --
#: Execution-step cap for sampling and delivery: candidates with execution steps (excluding demo frames) > 1600 are marked exec_over_cap and backfilled; the /4 header ``exec_cap`` must equal it
EXEC_CAP = 1600


def _v9_cells() -> dict[tuple[str, str], int]:
    """The 43 new-value cells of v9 plan part one table 2 (same tier set as V8): 50 episodes per task, split evenly across the tiers the task is delivered in under V8;
    when uneven, earlier tiers get 1 more (17/17/16, 13/13/12/12); excludes xhard0."""
    cells: dict[tuple[str, str], int] = {}
    for task in ("PickXtimes", "RouteStick", "PatternLock"):
        for tier, n in zip(("xhard1", "xhard2", "xhard3"), (17, 17, 16)):
            cells[(task, tier)] = n
    for task in ("SwingXtimes", "StopCube"):
        for tier in ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5"):
            cells[(task, tier)] = 10
    for task in ("VideoUnmask", "ButtonUnmask"):
        for tier, n in zip(("xhard1", "xhard2", "xhard3", "xhard4"), (13, 13, 12, 12)):
            cells[(task, tier)] = n
    for task in ("BinFill", "VideoUnmaskSwap", "ButtonUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
                 "PickHighlight", "VideoRepick"):
        for tier in ("xhard1", "xhard2"):
            cells[(task, tier)] = 25
    for task in ("MoveCube", "InsertPeg"):
        cells[(task, "xhard4")] = 50
    return cells


#: v9 delivery cell table (v9 plan part one table 2): 43 cells, total 800, 50 per task (same cell set as the removed V8 1070-episode table; only counts differ)
V9_CELLS: dict[tuple[str, str], int] = _v9_cells()
V9_PER_TASK = 50
assert len(V9_CELLS) == 43 and sum(V9_CELLS.values()) == 800, "V9_CELLS must be the 43 cells of table 2 totalling 800"
assert all(task in ALL_TASKS and tier in TIERS for task, tier in V9_CELLS), "V9_CELLS contains an unknown task or tier"
assert all(sum(n for (t, _), n in V9_CELLS.items() if t == task) == V9_PER_TASK for task in ALL_TASKS), \
    "V9_CELLS must have exactly 50 episodes per task"
#: Delivery cell table {(task, tier): formal delivery count} (the builder asserts row counts per cell from it: cells outside have exactly 0 rows, cells inside exactly the table value).
#: v9 stage 3b repackaging (env_metadata/ood/ switched to V9 specs) and switching this line to V9_CELLS landed in the same commit (v9 plan R7), so table and package always agree.
EXPECTED_CELLS: dict[tuple[str, str], int] = V9_CELLS
#: Registered complete delivery cell tables (by version). ``resolve_cell_table`` searches EXPECTED_CELLS -> the entries of this table in order for the first one covering
#: the given sub-table, used as the per-file quota cap (``_validate_specs``) and the cell quota cap of ``load_specs_root``.
#: The V8 1070-episode table was removed in maintenance plan stage 1b (W4); only v9 remains.
CELL_TABLES: dict[str, dict[tuple[str, str], int]] = {"v9": V9_CELLS}


def xhard4_only_tasks(cells: dict[tuple[str, str], int]) -> set[str]:
    """Set of tasks that appear only in xhard4 in a cell table (the check basis for XHARD4_ONLY against a parameter cell table)."""
    return {task for task in ALL_TASKS if {t for name, t in cells if name == task} == {"xhard4"}}


assert all(xhard4_only_tasks(table) == set(XHARD4_ONLY) for table in (EXPECTED_CELLS, *CELL_TABLES.values())), \
    "XHARD4_ONLY must be exactly the tasks appearing only in xhard4 in each delivery cell table (EXPECTED_CELLS, V9_CELLS)"


def _fits(cells: dict[tuple[str, str], int], table: dict[tuple[str, str], int]) -> bool:
    return all(key in table and _is_int(n) and n <= table[key] for key, n in cells.items())


def resolve_cell_table(cells: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    """Given (sub-)cell table -> complete delivery cell table used as quota cap: take the first table, in the order ``EXPECTED_CELLS`` then ``CELL_TABLES``,
    that "contains all cells with per-cell counts <= table values"; if none covers it, return ``EXPECTED_CELLS`` (the caller's per-item check reports which cell exceeds).
    Not used when a cell table is passed explicitly."""
    for table in (EXPECTED_CELLS, *CELL_TABLES.values()):
        if _fits(cells, table):
            return table
    return EXPECTED_CELLS


def header_cell_table(header: dict[str, Any]) -> dict[tuple[str, str], int] | None:
    """Per-task quotas carried by a /4 header -> ``resolve_cell_table`` picks the quota-cap cell table (for single-file readers that do not know the cell table, e.g.
    ``_rollout`` write-back re-check); returns None for non-/4 or malformed quotas (``validate_specs`` reports the specific error)."""
    if not isinstance(header, dict) or header.get("schema") != SCHEMA:
        return None
    quota = header.get("delivery_per_cell")
    if not isinstance(quota, dict):
        return None
    return resolve_cell_table({(task, header.get("difficulty")): n for task, n in quota.items()})

# Signature: header keys and row keys that go into identity_sha256
IDENTITY_HEADER_KEYS = ("schema", "difficulty", "tasks", "per_env", "runtime", "seed_rule", "select_rule",
                        "sampling_config_sha256", "recovery_rule", "identity_source")
IDENTITY_ROW_KEYS = ("task", "tier", "candidate", "episode", "seed", "attempt", "spec_sha256")
HEADER_REQUIRED = {"record", *IDENTITY_HEADER_KEYS, "sampling_config", "run_id", "draw_stats", "provenance",
                   "delivery_per_cell", "identity_sha256", "delivery_sha256"}
#: Migration-source-only keys (written when migrating from an old snapshot; freshly drawn files may lack them)
HEADER_OPTIONAL = {"drafts_sha256", "legacy_identity_sha256", "source_files", "eval_identities_sha256",
                   "dedup_dropped", "demo_frames_out_of_band", "notes"}
ROW_KEYS = {"record", *IDENTITY_ROW_KEYS, "spec", "selected", "tried", "initial_selected", "rollout"}
ROLLOUT_STATUSES = ("ok", "failed")
#: Identity keys tabled per schema (a key table is frozen once published, so identity_sha256 of sealed files stays byte-identical);
#: /4 = base keys + header ``layout_rule``, ``exec_cap``, ``delivery_per_cell`` + row ``layout_parent``
#: (per-task quotas and the execution-step cap are signed; changing them without re-signing must fail). The /2 and /3 key tables were removed with the old formats.
IDENTITY_KEYS_BY_SCHEMA = {
    SCHEMA: (IDENTITY_HEADER_KEYS + ("layout_rule", "exec_cap", "delivery_per_cell"),
                IDENTITY_ROW_KEYS + ("layout_parent",)),
}
#: The only valid layout rule for /4: each tier draws layouts independently, no derivation
LAYOUT_RULE = {"mode": "independent"}


def _schema_keys(schema: str) -> tuple[tuple[str, ...], tuple[str, ...], set[str], set[str]]:
    if schema not in IDENTITY_KEYS_BY_SCHEMA:
        raise SpecsError(f"specs version mismatch: {schema}")
    header_keys, row_keys = IDENTITY_KEYS_BY_SCHEMA[schema]
    header_required = (HEADER_REQUIRED - set(IDENTITY_HEADER_KEYS)) | set(header_keys)
    row_required = (ROW_KEYS - set(IDENTITY_ROW_KEYS)) | set(row_keys)
    return header_keys, row_keys, header_required, row_required


class SpecsError(ValueError):
    """Spec file missing, tampered with, wrong provenance, or field set mismatch."""


# -- Basic functions -------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def spec_sha256(spec: dict[str, Any]) -> str:
    return digest(spec)


def env_code(task: str) -> int:
    if task not in ALL_TASKS:
        raise SpecsError(f"unknown env name: {task}")
    return ALL_TASKS.index(task) + 1


def seed_rule_for(difficulty: str, profile: str) -> dict[str, Any]:
    """Return the seed rule for a tier and rule family: only families registered in ``TIER_SEED_OFFSETS`` (currently v8) are accepted, offset per tier, validated against
    the tiers registered in that family. The v5 (legacy single xhard tier), v6 and v7 (same offset for four tiers) families are removed."""
    if profile not in TIER_SEED_OFFSETS:
        known = "/".join(TIER_SEED_OFFSETS)
        raise SpecsError(f"{difficulty} only supports seed rule {known} (got {profile!r})")
    offsets = TIER_SEED_OFFSETS[profile]
    if difficulty not in offsets:
        raise SpecsError(f"seed rule {profile} has no registered tier {difficulty!r}; only supports {tuple(offsets)}")
    return {**SEED_RULE, "offset": offsets[difficulty]}


def _known_seed_rule(difficulty: str, rule: dict[str, Any]) -> bool:
    for profile in SEED_PROFILES:
        try:
            if rule == seed_rule_for(difficulty, profile):
                return True
        except SpecsError:
            continue
    return False


def seed_for(task: str, episode: int, attempt: int, rule: dict[str, Any] | None = None) -> int:
    rule = SEED_RULE if rule is None else rule
    if not 0 <= int(attempt) < MAX_ATTEMPTS:
        raise SpecsError(f"attempt must be in [0, {MAX_ATTEMPTS}), got {attempt}")
    return int(rule["offset"]) + env_code(task) * int(rule["env_block"]) + int(episode) * 100 + int(attempt)


def identity_sha256(header: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Seal the signature only: header spec keys + per-row signature keys; provenance, draw_stats, run_id and the result segment are excluded. Key set chosen by the header schema."""
    header_keys, row_keys, _, _ = _schema_keys(header.get("schema"))
    ordered = sorted(rows, key=lambda r: (r["task"], int(r["candidate"])))
    return digest({
        "header": {key: header[key] for key in header_keys},
        "rows": [{key: row[key] for key in row_keys} for row in ordered],
    })


def delivered(row: dict[str, Any]) -> bool:
    rollout = row.get("rollout") or {}
    return bool(row.get("selected")) and rollout.get("status") == "ok"


def delivery_sha256(rows: list[dict[str, Any]]) -> str:
    items = sorted(
        (row["task"], row["tier"], int(row["candidate"]), int(row["seed"]), row["spec_sha256"],
         (row.get("rollout") or {}).get("h5_sha256"))
        for row in rows if delivered(row)
    )
    return digest([list(item) for item in items])


# -- Source fingerprint (provenance only, not identity; mismatch only warns) ---------------


def _package_root(name: str) -> Path:
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SpecsError(f"package not found: {name}")
    return Path(list(spec.submodule_search_locations)[0])


def hard_fingerprint() -> str:
    """Aggregate hash of relative paths and sha256 of all ``robomme_ood`` .py files."""
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}
    return digest(files)


def base_fingerprint() -> str:
    """Aggregate hash of the borrowed official shim target files (reads the currently installed robomme per the UPSTREAM.json manifest)."""
    manifest = json.loads((Path(__file__).resolve().parents[1] / "UPSTREAM.json").read_text(encoding="utf-8"))
    robomme_root = _package_root("robomme")
    files = {}
    for entry in manifest["shims"]:
        rel = entry["target_file"][len("src/robomme/"):]
        path = robomme_root / rel
        files[entry["target_file"]] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return digest(files)


# -- Read/write ---------------------------------------------------------------


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SpecsError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line, object_pairs_hook=_unique_object) for line in stream if line.strip()]
    if not records:
        raise SpecsError(f"{path} is empty")
    return records


def _exact_keys(value: dict[str, Any], required: set[str], label: str, optional: set[str] = frozenset()) -> None:
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise SpecsError(f"{label} field set mismatch: missing {sorted(missing)}, extra {sorted(extra)}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_specs(header: dict[str, Any], rows: list[dict[str, Any]],
                       expected_cells: dict[tuple[str, str], int] | None = None) -> None:
    """``hard-specs/4`` single-file validation (v8 plan part two §2.2 item 2). ``expected_cells`` is the complete delivery cell table used as quota cap,
    defaulting to ``EXPECTED_CELLS`` (v9 plan §2.1: the cell table parameter is threaded load_specs -> validate_specs -> this function):

    * header: field set; ``difficulty in TIERS``; runtime; ``seed_rule == seed_rule_for(tier, "v8")``;
      ``exec_cap == EXEC_CAP``; ``layout_rule == {"mode": "independent"}``; ``tasks`` has no duplicates and every (task, tier)
      is in the cell table; ``select_rule`` is ``{task: [distinct non-negative ints]}``, ``per_env`` is ``{task: candidate count (non-negative int)}``,
      ``delivery_per_cell`` is ``{task: positive int}``, all three with key set equal to ``tasks``; per-task quotas self-consistent:
      ``delivery_per_cell[task] <= cell_table[(task, tier)]``, ``len(select_rule[task]) == delivery_per_cell[task]``,
      ``per_env[task] ==`` row count of that task in this file, every index of ``select_rule[task]`` ``< per_env[task]``; embedded sampling_config hash self-consistent;
    * rows: field set; ``candidate``/``attempt``/``seed`` are integers and not bools (F-6); ``layout_parent is None``,
      ``spec.spec_kind == "native-newvalue/2"``; ``candidate == episode`` and
      ``0 <= episode < env_block // episode_stride``; tier, spec hash, seed formula, boolean flags, rollout.status;
      per-task selected row count <= ``delivery_per_cell[task]``;
    * both identity hashes (the signature includes exec_cap, delivery_per_cell and seed_rule; changing any without re-signing fails).
    """
    table = EXPECTED_CELLS if expected_cells is None else expected_cells
    _, _, header_required, row_required = _schema_keys(SCHEMA)
    _exact_keys(header, header_required, "specs header", HEADER_OPTIONAL)
    if header["record"] != "header":
        raise SpecsError(f"specs version mismatch: {SCHEMA}")
    tier = header["difficulty"]
    if tier not in TIERS or header["runtime"] != RUNTIME:
        raise SpecsError(f"hard-specs/4 tier or runtime mismatch: {tier!r}")
    if header["seed_rule"] != seed_rule_for(tier, "v8"):
        raise SpecsError(f"hard-specs/4 only accepts the v8 per-tier seed rule ({tier})")
    if not _is_int(header["exec_cap"]) or header["exec_cap"] != EXEC_CAP:
        raise SpecsError(f"hard-specs/4 exec_cap must be {EXEC_CAP}: {header['exec_cap']!r}")
    if header["layout_rule"] != LAYOUT_RULE:
        raise SpecsError(f"hard-specs/4 layout_rule must be {LAYOUT_RULE}: {header['layout_rule']!r}")
    tasks = header["tasks"]
    if not isinstance(tasks, list) or len(set(tasks)) != len(tasks):
        raise SpecsError(f"hard-specs/4 tasks must be a list without duplicates: {tasks!r}")
    stray = [task for task in tasks if (task, tier) not in table]
    if stray:
        raise SpecsError(f"hard-specs/4 tasks not in the {tier} delivery cells: {stray}")
    for name in ("select_rule", "per_env", "delivery_per_cell"):
        value = header[name]
        if not isinstance(value, dict) or set(value) != set(tasks):
            raise SpecsError(f"hard-specs/4 {name} must be a per-task dict whose key set equals tasks: {value!r}")
    for task in tasks:
        indices = header["select_rule"][task]
        if not isinstance(indices, list) or not all(_is_int(i) and i >= 0 for i in indices) \
                or len(set(indices)) != len(indices):
            raise SpecsError(f"hard-specs/4 select_rule[{task}] must be a list of distinct non-negative integers: {indices!r}")
        if not _is_int(header["per_env"][task]) or header["per_env"][task] < 0:
            raise SpecsError(f"hard-specs/4 per_env[{task}] must be a non-negative integer: {header['per_env'][task]!r}")
        quota = header["delivery_per_cell"][task]
        if not _is_int(quota) or quota <= 0:
            raise SpecsError(f"hard-specs/4 delivery_per_cell[{task}] must be a positive integer: {quota!r}")
        if quota > table[(task, tier)]:
            raise SpecsError(f"hard-specs/4 delivery_per_cell[{task}]={quota} exceeds the table 2 cell quota "
                             f"{table[(task, tier)]} ({tier})")
        if len(indices) != quota:
            raise SpecsError(f"hard-specs/4 select_rule[{task}] length {len(indices)} != delivery_per_cell {quota}")
        n_rows = sum(1 for row in rows if row.get("task") == task)
        if header["per_env"][task] != n_rows:
            raise SpecsError(f"hard-specs/4 per_env[{task}]={header['per_env'][task]} != row count of this task in the file {n_rows}")
        if any(i >= header["per_env"][task] for i in indices):
            raise SpecsError(f"hard-specs/4 select_rule[{task}] has an index beyond per_env={header['per_env'][task]}: "
                             f"{indices!r}")
    if header["sampling_config_sha256"] != digest(header["sampling_config"]):
        raise SpecsError("embedded sampling_config hash is not self-consistent")
    max_episode = SEED_RULE["env_block"] // SEED_RULE["episode_stride"]
    seen, selected_count = set(), {}
    for row in rows:
        _exact_keys(row, row_required, "specs row")
        # F-6: candidate/attempt/seed must be true integers (excluding bool and float), otherwise True==1, int(0.5)==0 and
        # 16000000.0==16000000 would let the later equality checks and the seed formula pass by mistake
        bad_types = {name: row[name] for name in ("candidate", "attempt", "seed") if not _is_int(row[name])}
        if bad_types:
            raise SpecsError(f"hard-specs/4 row candidate/attempt/seed must be integer (not bool or float): "
                             f"{row.get('task')} {bad_types!r}")
        key = (row["task"], int(row["candidate"]))
        if row["record"] != "spec" or key in seen or row["task"] not in tasks:
            raise SpecsError(f"duplicate or extra spec row: {key}")
        seen.add(key)
        # /4 row candidate is the drawn episode (same as how _freeze writes it); episode must not exceed env_block, otherwise cross-task seeds collide
        if not _is_int(row["episode"]) or not 0 <= row["episode"] < max_episode:
            raise SpecsError(f"hard-specs/4 row episode must be in [0, {max_episode}): {key} {row['episode']!r}")
        if row["candidate"] != row["episode"]:
            raise SpecsError(f"hard-specs/4 row candidate must equal episode: {key} {row['episode']!r}")
        if row["tier"] != tier:
            raise SpecsError(f"spec row tier does not match header: {key}")
        if row["layout_parent"] is not None or (row.get("spec") or {}).get("spec_kind") != "native-newvalue/2":
            raise SpecsError(f"hard-specs/4 row layout_parent must be null and spec must be native-newvalue/2: {key}")
        if row["spec_sha256"] != spec_sha256(row["spec"]):
            raise SpecsError(f"spec hash mismatch: {key}")
        if row["seed"] != seed_for(row["task"], row["episode"], row["attempt"], header["seed_rule"]):
            raise SpecsError(f"seed does not match formula: {key}")
        for flag in ("selected", "tried", "initial_selected"):
            if type(row[flag]) is not bool:
                raise SpecsError(f"{flag} must be a bool: {key}")
        rollout = row["rollout"]
        if rollout is not None and rollout.get("status") not in ROLLOUT_STATUSES:
            raise SpecsError(f"invalid rollout.status: {key}")
        if row["selected"]:
            selected_count[row["task"]] = selected_count.get(row["task"], 0) + 1
    over = {task: n for task, n in selected_count.items() if n > header["delivery_per_cell"][task]}
    if over:
        raise SpecsError(f"per-task selected row count exceeds delivery_per_cell: {over}")
    if identity_sha256(header, rows) != header["identity_sha256"]:
        raise SpecsError("identity_sha256 mismatch (signature or provenance was modified)")
    if delivery_sha256(rows) != header["delivery_sha256"]:
        raise SpecsError("delivery_sha256 mismatch (formal delivery set was modified)")


def validate_specs(header: dict[str, Any], rows: list[dict[str, Any]], *,
                   expected_cells: dict[tuple[str, str], int] | None = None) -> None:
    """Envelope validation: only ``hard-specs/4`` is accepted, via ``_validate_specs`` (``expected_cells`` as quota cap, default
    ``EXPECTED_CELLS``); any other schema (including the removed /2 and /3) is rejected with "specs version mismatch"."""
    schema = header.get("schema")
    if schema != SCHEMA:
        raise SpecsError(f"specs version mismatch: {schema}")
    _validate_specs(header, rows, EXPECTED_CELLS if expected_cells is None else expected_cells)


def load_specs(path: str | Path, *, expected_cells: dict[tuple[str, str], int] | None = None,
               check_fingerprint: bool = True):
    """The only read entry: returns ``(header, rows)`` (rows are all spec rows; callers take formal episodes via ``delivered``).

    ``expected_cells``: quota-cap cell table for /4 files, default ``EXPECTED_CELLS``.
    A source fingerprint (``provenance``) mismatch with the current environment only triggers ``warnings.warn``, not rejection (0927 plan §3.4).
    """
    records = read_jsonl(path)
    header, rows = records[0], records[1:]
    validate_specs(header, rows, expected_cells=expected_cells)
    if check_fingerprint:
        provenance = header.get("provenance") or {}
        try:
            current = {"hard_fingerprint": hard_fingerprint(), "base_fingerprint": base_fingerprint()}
        except Exception as exc:  # noqa: BLE001 fingerprint is supporting evidence only
            warnings.warn(f"cannot compute source fingerprint: {exc}")
            current = {}
        for key, value in current.items():
            if provenance.get(key) not in (None, value):
                warnings.warn(f"{path}: {key} does not match current source (warning only, not rejected)")
    return copy.deepcopy(header), copy.deepcopy(rows)


def load_specs_root(root: str | Path, expected_cells: dict[tuple[str, str], int], *,
                  cell_table: dict[tuple[str, str], int] | None = None,
                  check_fingerprint: bool = True) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Read a v8/v9 spec root (``<root>/<tier>/specs.jsonl``, ``hard-specs/4``), validating only the cell table given by the caller.

    ``expected_cells``: ``{(task, tier): episode count}``; keys must be a subset of the complete delivery cell table keys, values positive integers (values may be below table values,
    e.g. 1 episode per cell for smoke). Full root passes ``V9_CELLS``, smoke passes a smoke table, a shard passes its subset.
    ``cell_table``: complete delivery cell table used as quota cap; defaults to ``resolve_cell_table(expected_cells)`` (EXPECTED_CELLS ->
    first covering table in CELL_TABLES), and is passed unchanged to each ``load_specs``. Only reads tier files
    involved in ``expected_cells`` (other tiers' files are not read even if present); each goes through ``load_specs`` (/4 validation), plus:

    * each file has ``schema == "hard-specs/4"`` and ``difficulty`` equal to the directory tier name;
    * each header's ``tasks`` set equals the task set of ``expected_cells`` for that tier;
    * ``expected_cells[key] <= cell_table[key]``;
    * each cell's header ``delivery_per_cell[task]`` equals the ``expected_cells`` value;
    * each cell's ``selected`` row count equals the ``expected_cells`` value (equal, not <=). Counts ``selected`` only, ignores rollout results;
      per-cell checking of formal delivery (``delivered``: selected and rollout ok) is asserted per cell by the builder when constructing ``ood``;
    * seeds of the same task are pairwise disjoint across tiers (comparing all spec rows, not just selected).

    Returns ``{tier: (header, rows)}``: keys contain only the involved tiers, in ``TIERS`` order;
    rows are all spec rows of that tier (callers take ``selected`` or ``delivered``). Any mismatch raises ``SpecsError``.
    """
    if not isinstance(expected_cells, dict) or not expected_cells:
        raise SpecsError("expected_cells must be a non-empty {(task, tier): episode count} dict")
    table = resolve_cell_table(expected_cells) if cell_table is None else cell_table
    stray = sorted(key for key in expected_cells if key not in table)
    if stray:
        raise SpecsError(f"expected_cells contains cells outside the delivery cell table (V9_CELLS): {stray}")
    bad = {key: n for key, n in expected_cells.items() if not _is_int(n) or n <= 0}
    if bad:
        raise SpecsError(f"expected_cells episode counts must be positive integers: {bad}")
    over = {key: n for key, n in expected_cells.items() if n > table[key]}
    if over:
        raise SpecsError(f"expected_cells episode counts exceed the table 2 cell quota: {over}")
    out: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for tier in TIERS:
        want = {task for task, t in expected_cells if t == tier}
        if not want:
            continue
        path = Path(root) / tier / "specs.jsonl"
        if not path.is_file():
            raise SpecsError(f"v8 spec root is missing {path}")
        header, rows = load_specs(path, expected_cells=table, check_fingerprint=check_fingerprint)
        if header["schema"] != SCHEMA or header["difficulty"] != tier:
            raise SpecsError(f"{path}: schema must be {SCHEMA} and tier must be {tier}"
                             f" (got {header['schema']}/{header['difficulty']})")
        got_tasks = set(header["tasks"])
        if got_tasks != want:
            raise SpecsError(f"{tier} task set mismatch: missing {sorted(want - got_tasks)}, extra {sorted(got_tasks - want)}")
        for task in sorted(want):
            if header["delivery_per_cell"][task] != expected_cells[(task, tier)]:
                raise SpecsError(f"{task}/{tier} delivery_per_cell={header['delivery_per_cell'][task]} ≠ "
                                 f"expected {expected_cells[(task, tier)]}")
            got = sum(1 for row in rows if row["task"] == task and row["selected"])
            if got != expected_cells[(task, tier)]:
                raise SpecsError(f"{task}/{tier} selected row count {got} != expected {expected_cells[(task, tier)]}")
        out[tier] = (header, rows)
    seeds: dict[str, dict[str, set[int]]] = {}
    for tier, (_, rows) in out.items():
        for row in rows:
            seeds.setdefault(row["task"], {}).setdefault(tier, set()).add(int(row["seed"]))
    for task, by_tier in seeds.items():
        names = list(by_tier)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = by_tier[a] & by_tier[b]
                if common:
                    raise SpecsError(f"{task} seeds intersect between {a} and {b}: {sorted(common)[:5]}")
    return out


#: Packaged spec root: ``env_metadata/ood/xhard{1..5}/specs.jsonl`` (by TIERS). The builder reads only this and accepts no external spec root.
PACKAGED_SPECS_ROOT = Path(__file__).resolve().parents[1] / "env_metadata" / "ood"
_ANNOUNCED_ROOTS: set[str] = set()


def specs_root(override: str | Path | None = None) -> Path:
    """Spec root: explicit argument > packaged (environment variables are not read). Prints ``SPECS_ROOT=`` once when not packaged.

    The ``override`` parameter is kept for callers that read a single spec root directly (e.g. test fixtures); the builder always uses the packaged root."""
    root = Path(override).resolve() if override is not None else PACKAGED_SPECS_ROOT
    if root != PACKAGED_SPECS_ROOT and str(root) not in _ANNOUNCED_ROOTS:
        _ANNOUNCED_ROOTS.add(str(root))
        print(f"SPECS_ROOT={root}", flush=True)
    return root


def packaged_specs_path(tier: str, root: str | Path | None = None) -> Path:
    if tier not in TIERS:
        raise SpecsError(f"unknown tier {tier!r}")
    return specs_root(root) / tier / "specs.jsonl"


# -- Re-injection binding summary (the only function called by the policy repo; 0927 plan §4.2, R22) ---------------


def _max_abs_diff(a: Any, b: Any) -> float:
    """Max absolute difference between two value trees; returns inf if structures or non-numeric values differ."""
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max((_max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return math.inf
        return max((_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    return 0.0 if a == b else math.inf


def spec_binding(env) -> dict[str, Any]:
    """Read ``env.unwrapped._spec`` (``SpecRecorder``) and return the re-injection binding summary; must be called after ``reset()``.

    - ``injected_mismatch``: number of unequal entries at injection points (``value()``, trace ``source="spec"``), plus recorded points with difference > 1e-5;
    - ``recorded_drift``: number of entries at record-only observation points (``record()``, trace ``source="record"``) with difference <= ``RECORDED_FLOAT_TOL``;
    - ``unused``: number of value points present in the spec but not accessed via ``value()``/``record()`` in this episode.
    """
    recorder = getattr(getattr(env, "unwrapped", env), "_spec", None)
    if recorder is None:
        return {"available": False}
    spec_paths = {item["path"] for item in recorder.trace if item.get("source") == "spec"}
    record_paths = {item["path"] for item in recorder.trace if item.get("source") == "record"}
    injected, drift, drift_max = 0, 0, 0.0
    for item in recorder.mismatches:
        path = item["path"]
        if path in record_paths and path not in spec_paths:
            diff = _max_abs_diff(item.get("drawn"), item.get("frozen"))
            if diff <= RECORDED_FLOAT_TOL:
                drift += 1
                drift_max = max(drift_max, diff)
                continue
        injected += 1
    consumed = set(recorder.consumed_paths())
    unused = [p for p in recorder.leaf_paths() if not any(p == c or p.startswith(c + ".") for c in consumed)]
    frozen = getattr(recorder, "_frozen", None)
    # The five keys layered/layout_hit/layout_paths_expected/layout_overridden/layout_drift originally served V7 layered re-injection (removed in maintenance plan
    # stage 1b). SpecRecorder no longer has these attributes, so the getattr below always returns the default (False/0/None), byte-identical to the
    # pre-removal output for non-layered episodes; the key names are kept so the summary shape does not change (the policy repo and hard_regression reset-replay read these keys).
    layered_hit = getattr(recorder, "_layered_hit", None)
    if layered_hit is not None:
        layout_hit = len({item["path"] for item in recorder.trace if item.get("path") in layered_hit})
    else:
        layout_hit = len(getattr(recorder, "layout_paths_hit", ()) or ())
    return {
        "available": True,
        "mode": recorder.mode,
        "spec_kind": recorder.spec_kind,
        "spec_sha256": spec_sha256(frozen) if recorder.mode == "replay" else None,
        "value_points": sum(1 for item in recorder.trace if item.get("source") in ("draw", "spec")),
        "injected_mismatch": injected,
        "recorded_drift": drift,
        "recorded_max_abs": drift_max,
        "unused": len(unused),
        "layered": bool(getattr(recorder, "layered", False)),
        "layout_hit": layout_hit,
        "layout_paths_expected": len(layered_hit) if layered_hit is not None else None,
        "layout_overridden": int(getattr(recorder, "layout_overridden", 0)),
        "layout_drift": int(getattr(recorder, "layout_drift", 0)),
    }
