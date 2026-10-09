"""L1 contract: negatives for the ``hard-specs/4`` validators (``hard_specs.validate_specs``/``load_specs``/``load_specs_root``).

The base is the real packaged xhard5 spec (26 rows, 20 of them selected). Each negative changes one thing in an in-memory copy, in two classes:

- without re-signing: tamper with the signature or result segment; the two identity hashes or the spec hash must fail to match;
- semantic error after re-signing: recompute ``sampling_config_sha256``/``identity_sha256``/``delivery_sha256`` with the production signing functions,
  so all hashes are self-consistent and only the semantic error remains (tier, runtime, seed rule, execution-step cap, layout rule, quota, out of range, F-6 types, ...);
  the validator must still reject.

Also cross-tier seed intersection: it can only be produced by changing the per-tier offsets; monkeypatch the offset table in process to build a root with two tiers sharing seeds, and whole-root reading must reject.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO
from tests.robomme_ood.contract.test_constants import EXEC_CAP, V9_CELLS

ROOT = REPO / "src" / "robomme_ood" / "env_metadata" / "ood"
BASE_TIER = "xhard5"


@pytest.fixture(scope="module")
def hs():
    from robomme_ood.env_record_wrapper import hard_specs

    return hard_specs


@pytest.fixture
def base():
    lines = (ROOT / BASE_TIER / "specs.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records[0], records[1:]


def resign(hs, header, rows):
    header = copy.deepcopy(header)
    header["sampling_config_sha256"] = hs.digest(header["sampling_config"])
    header["identity_sha256"] = hs.identity_sha256(header, rows)
    header["delivery_sha256"] = hs.delivery_sha256(rows)
    return header


def spare(rows):
    return next(r for r in rows if not r["selected"] and r["rollout"] is None)


def selected(rows):
    return next(r for r in rows if r["selected"])


def test_base_and_resigned_base_pass(hs, base):
    header, rows = base
    hs.validate_specs(header, rows)
    hs.validate_specs(resign(hs, header, rows), rows)


# -- Tampering without re-signing ---------------------------------------------------------

TAMPER = {
    "row_seed_plus_1": lambda h, rs: selected(rs).__setitem__("seed", selected(rs)["seed"] + 1),
    "row_spec_changed": lambda h, rs: selected(rs)["spec"]["layout"].__setitem__("button_xy", [0.0, 0.0]),
    "spec_changed_row_hash_recomputed_only": None,  # implemented separately: spec hash self-consistent, identity mismatches
    "result_h5_sha_changed": lambda h, rs: selected(rs)["rollout"].__setitem__("h5_sha256", "0" * 64),
    "selected_flipped": lambda h, rs: spare(rs).__setitem__("selected", True),
    "header_exec_cap": lambda h, rs: h.__setitem__("exec_cap", EXEC_CAP - 100),
    "sampling_config_changed": lambda h, rs: h["sampling_config"]["StopCube"]["decision"].__setitem__("extra", 1),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_tamper_without_resign_rejected(hs, base, name):
    header, rows = copy.deepcopy(base)
    if TAMPER[name] is None:
        row = selected(rows)
        row["spec"]["layout"]["button_xy"] = [0.0, 0.0]
        row["spec_sha256"] = hs.spec_sha256(row["spec"])
    else:
        TAMPER[name](header, rows)
    with pytest.raises(hs.SpecsError):
        hs.validate_specs(header, rows)


# -- Semantic errors after re-signing -----------------------------------------------------


def _row_episode_out_of_range(hs, h, rs):
    row = spare(rs)
    limit = h["seed_rule"]["env_block"] // h["seed_rule"]["episode_stride"]
    row["candidate"] = row["episode"] = limit
    row["seed"] = hs.seed_for(row["task"], limit, row["attempt"], h["seed_rule"])


def _row_seed_formula(hs, h, rs):
    row = spare(rs)
    row["seed"] += 1  # formula no longer holds (not even after re-signing)


def _quota_over_table(hs, h, rs):
    task = "StopCube"
    over = V9_CELLS[(task, BASE_TIER)] + 1
    h["delivery_per_cell"][task] = over
    h["select_rule"][task] = list(range(over))
    h["per_env"][task] = max(h["per_env"][task], over)


SEMANTIC = {
    "tier_unknown": lambda hs, h, rs: h.__setitem__("difficulty", "xhard6"),
    "runtime_changed": lambda hs, h, rs: h["runtime"].__setitem__("control_mode", "pd_ee_delta_pose"),
    "seed_rule_offset_changed": lambda hs, h, rs: h["seed_rule"].__setitem__("offset", h["seed_rule"]["offset"] + 1),
    "exec_cap_changed": lambda hs, h, rs: h.__setitem__("exec_cap", EXEC_CAP + 1),
    "exec_cap_bool": lambda hs, h, rs: h.__setitem__("exec_cap", True),
    "layout_rule_changed": lambda hs, h, rs: h.__setitem__("layout_rule", {"mode": "derived"}),
    "tasks_duplicated": lambda hs, h, rs: h.__setitem__("tasks", h["tasks"] + h["tasks"][:1]),
    "quota_over_cell_table": _quota_over_table,
    "select_rule_length_mismatch": lambda hs, h, rs: h["select_rule"]["StopCube"].pop(),
    "per_env_mismatch": lambda hs, h, rs: h["per_env"].__setitem__("StopCube", h["per_env"]["StopCube"] + 1),
    "selected_over_quota": lambda hs, h, rs: spare(rs).__setitem__("selected", True),
    "row_tier_mismatch": lambda hs, h, rs: spare(rs).__setitem__("tier", "xhard4"),
    "layout_parent_non_null": lambda hs, h, rs: spare(rs).__setitem__("layout_parent", "xhard4/0"),
    "spec_kind_changed": None,  # implemented separately (row hash must be updated too)
    "candidate_not_equal_episode": lambda hs, h, rs: spare(rs).__setitem__("candidate", spare(rs)["candidate"] + 1000),
    "episode_out_of_range": _row_episode_out_of_range,
    "seed_violates_formula": _row_seed_formula,
    "rollout_status_invalid": lambda hs, h, rs: selected(rs)["rollout"].__setitem__("status", "maybe"),
    "bool_flag_not_bool": lambda hs, h, rs: spare(rs).__setitem__("tried", 0),
    "row_extra_key": lambda hs, h, rs: spare(rs).__setitem__("note", "x"),
    "header_missing_key": lambda hs, h, rs: h.pop("draw_stats"),
    "row_duplicated": lambda hs, h, rs: rs.append(copy.deepcopy(spare(rs))),
    # F-6: candidate/attempt/seed is a bool or float
    "F6_candidate_bool": None,
    "F6_attempt_float": lambda hs, h, rs: spare(rs).__setitem__("attempt", spare(rs)["attempt"] + 0.5),
    "F6_seed_float": lambda hs, h, rs: spare(rs).__setitem__("seed", float(spare(rs)["seed"])),
}


def _apply_semantic(hs, name, header, rows):
    if name == "spec_kind_changed":
        row = spare(rows)
        row["spec"]["spec_kind"] = "native-newvalue/1"
        row["spec_sha256"] = hs.spec_sha256(row["spec"])
    elif name == "F6_candidate_bool":
        row = next(r for r in rows if r["candidate"] == 1)
        row["candidate"] = True  # True == 1: only a type check can catch it
    else:
        SEMANTIC[name](hs, header, rows)


@pytest.mark.parametrize("name", sorted(SEMANTIC))
def test_semantic_error_after_resign_rejected(hs, base, name):
    header, rows = copy.deepcopy(base)
    _apply_semantic(hs, name, header, rows)
    try:
        header = resign(hs, header, rows)
    except (KeyError, TypeError, ValueError):
        pytest.fail(f"{name}: the signing function itself must not fail (fixture bug)")
    with pytest.raises(hs.SpecsError):
        hs.validate_specs(header, rows)


@pytest.mark.parametrize("name", ["F6_candidate_bool", "F6_attempt_float", "F6_seed_float"])
def test_f6_rejected_with_type_message(hs, base, name):
    header, rows = copy.deepcopy(base)
    _apply_semantic(hs, name, header, rows)
    header = resign(hs, header, rows)
    with pytest.raises(hs.SpecsError, match="integer"):
        hs.validate_specs(header, rows)


def test_load_specs_file_level_rejections(hs, base, tmp_path):
    """File level: empty file, duplicate fields and old schema are all rejected."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(hs.SpecsError):
        hs.load_specs(empty, check_fingerprint=False)
    header, rows = base
    dup = tmp_path / "dup.jsonl"
    text = json.dumps(header)
    dup.write_text(text[:-1] + ', "schema": "hard-specs/4"}\n' + "".join(json.dumps(r) + "\n" for r in rows),
                   encoding="utf-8")
    with pytest.raises(hs.SpecsError):
        hs.load_specs(dup, check_fingerprint=False)
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps(dict(header, schema="hard-specs/3")) + "\n" + "".join(json.dumps(r) + "\n" for r in rows),
                   encoding="utf-8")
    with pytest.raises(hs.SpecsError):
        hs.load_specs(old, check_fingerprint=False)


# -- Cross-tier seed intersection ---------------------------------------------------------


def _root_with(tmp_path: Path, tiers) -> Path:
    root = tmp_path / "root"
    for tier in tiers:
        (root / tier).mkdir(parents=True)
        shutil.copyfile(ROOT / tier / "specs.jsonl", root / tier / "specs.jsonl")
    return root


def test_cross_tier_seed_intersection_rejected(hs, tmp_path, monkeypatch):
    tiers = ("xhard4", "xhard5")
    cells = {key: n for key, n in V9_CELLS.items() if key[1] in tiers}
    root = _root_with(tmp_path, tiers)
    hs.load_specs_root(root, cells, check_fingerprint=False)  # positive: the two tiers are readable as-is
    # change xhard5's offset to xhard4's, and rewrite xhard5's seeds and signatures under the new rule (the file itself stays valid)
    monkeypatch.setitem(hs.TIER_SEED_OFFSETS["v8"], "xhard5", hs.TIER_SEED_OFFSETS["v8"]["xhard4"])
    path = root / "xhard5" / "specs.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    header, rows = records[0], records[1:]
    header["seed_rule"] = hs.seed_rule_for("xhard5", "v8")
    for row in rows:
        row["seed"] = hs.seed_for(row["task"], row["episode"], row["attempt"], header["seed_rule"])
    header = resign(hs, header, rows)
    hs.validate_specs(header, rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in [header, *rows]), encoding="utf-8")
    with pytest.raises(hs.SpecsError, match="intersect"):
        hs.load_specs_root(root, cells, check_fingerprint=False)


def test_load_specs_root_cell_table_rejections(hs, tmp_path):
    root = _root_with(tmp_path, ("xhard5",))
    cells = {key: n for key, n in V9_CELLS.items() if key[1] == "xhard5"}
    hs.load_specs_root(root, cells, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):
        hs.load_specs_root(root, {}, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # cell outside the cell table
        hs.load_specs_root(root, {("PickXtimes", "xhard5"): 1}, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # episode count differs from selected count
        hs.load_specs_root(root, {k: n - 1 for k, n in cells.items()}, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # missing tier file
        hs.load_specs_root(root, {**cells, ("StopCube", "xhard4"): V9_CELLS[("StopCube", "xhard4")]},
                           check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # episode count is a bool
        hs.load_specs_root(root, {k: True for k in cells}, check_fingerprint=False)
