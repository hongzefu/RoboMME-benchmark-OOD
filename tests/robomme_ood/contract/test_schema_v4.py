"""L1 契约：``hard-specs/4`` 校验器（``hard_specs.validate_specs``／``load_specs``／``load_specs_root``）的负例。

基底是包内真实的 xhard5 规格（26 行，其中 20 行 selected）。每个负例在内存副本上改一处，分两类：

- 不重签：篡改签或结果段，两个身份散列或规格散列必须对不上；
- 重签后语义错：用生产签名函数把 ``sampling_config_sha256``／``identity_sha256``／``delivery_sha256`` 全部重算，
  散列都自洽，只剩语义错误（档位、runtime、seed 规则、执行步上限、布局规则、配额、越界、F-6 类型……），
  校验器仍须拒绝。

另有跨档 seed 相交：只有改了按档偏移才造得出来，进程内 monkeypatch 偏移表后造两档同 seed 的根，整根读取须拒绝。
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


# ── 不重签的篡改 ─────────────────────────────────────────────────────────

TAMPER = {
    "行_seed_加1": lambda h, rs: selected(rs).__setitem__("seed", selected(rs)["seed"] + 1),
    "行_spec_改值": lambda h, rs: selected(rs)["spec"]["layout"].__setitem__("button_xy", [0.0, 0.0]),
    "spec_改值且只重算行散列": None,  # 单独实现：规格散列自洽、identity 不符
    "结果段_h5_sha_改": lambda h, rs: selected(rs)["rollout"].__setitem__("h5_sha256", "0" * 64),
    "selected_翻转": lambda h, rs: spare(rs).__setitem__("selected", True),
    "header_exec_cap": lambda h, rs: h.__setitem__("exec_cap", EXEC_CAP - 100),
    "sampling_config_改值": lambda h, rs: h["sampling_config"]["StopCube"]["decision"].__setitem__("extra", 1),
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


# ── 重签后的语义错 ───────────────────────────────────────────────────────


def _row_episode_out_of_range(hs, h, rs):
    row = spare(rs)
    limit = h["seed_rule"]["env_block"] // h["seed_rule"]["episode_stride"]
    row["candidate"] = row["episode"] = limit
    row["seed"] = hs.seed_for(row["task"], limit, row["attempt"], h["seed_rule"])


def _row_seed_formula(hs, h, rs):
    row = spare(rs)
    row["seed"] += 1  # 公式不再成立（重签后也不成立）


def _quota_over_table(hs, h, rs):
    task = "StopCube"
    over = V9_CELLS[(task, BASE_TIER)] + 1
    h["delivery_per_cell"][task] = over
    h["select_rule"][task] = list(range(over))
    h["per_env"][task] = max(h["per_env"][task], over)


SEMANTIC = {
    "档位_不认识": lambda hs, h, rs: h.__setitem__("difficulty", "xhard6"),
    "runtime_改": lambda hs, h, rs: h["runtime"].__setitem__("control_mode", "pd_ee_delta_pose"),
    "seed_rule_偏移_改": lambda hs, h, rs: h["seed_rule"].__setitem__("offset", h["seed_rule"]["offset"] + 1),
    "exec_cap_改": lambda hs, h, rs: h.__setitem__("exec_cap", EXEC_CAP + 1),
    "exec_cap_布尔": lambda hs, h, rs: h.__setitem__("exec_cap", True),
    "layout_rule_改": lambda hs, h, rs: h.__setitem__("layout_rule", {"mode": "derived"}),
    "tasks_重复": lambda hs, h, rs: h.__setitem__("tasks", h["tasks"] + h["tasks"][:1]),
    "配额_超格表": _quota_over_table,
    "select_rule_长度不符": lambda hs, h, rs: h["select_rule"]["StopCube"].pop(),
    "per_env_不符": lambda hs, h, rs: h["per_env"].__setitem__("StopCube", h["per_env"]["StopCube"] + 1),
    "selected_超配额": lambda hs, h, rs: spare(rs).__setitem__("selected", True),
    "行档位不符": lambda hs, h, rs: spare(rs).__setitem__("tier", "xhard4"),
    "layout_parent_非空": lambda hs, h, rs: spare(rs).__setitem__("layout_parent", "xhard4/0"),
    "spec_kind_改": None,  # 单独实现（需同步行散列）
    "candidate_不等于_episode": lambda hs, h, rs: spare(rs).__setitem__("candidate", spare(rs)["candidate"] + 1000),
    "episode_越界": _row_episode_out_of_range,
    "seed_不合公式": _row_seed_formula,
    "rollout_status_非法": lambda hs, h, rs: selected(rs)["rollout"].__setitem__("status", "maybe"),
    "布尔位_非布尔": lambda hs, h, rs: spare(rs).__setitem__("tried", 0),
    "行多一个键": lambda hs, h, rs: spare(rs).__setitem__("note", "x"),
    "header_缺键": lambda hs, h, rs: h.pop("draw_stats"),
    "行重复": lambda hs, h, rs: rs.append(copy.deepcopy(spare(rs))),
    # F-6：candidate／attempt／seed 为布尔或浮点
    "F6_candidate_布尔": None,
    "F6_attempt_浮点": lambda hs, h, rs: spare(rs).__setitem__("attempt", spare(rs)["attempt"] + 0.5),
    "F6_seed_浮点": lambda hs, h, rs: spare(rs).__setitem__("seed", float(spare(rs)["seed"])),
}


def _apply_semantic(hs, name, header, rows):
    if name == "spec_kind_改":
        row = spare(rows)
        row["spec"]["spec_kind"] = "native-newvalue/1"
        row["spec_sha256"] = hs.spec_sha256(row["spec"])
    elif name == "F6_candidate_布尔":
        row = next(r for r in rows if r["candidate"] == 1)
        row["candidate"] = True  # True == 1：只有类型检查能挡住
    else:
        SEMANTIC[name](hs, header, rows)


@pytest.mark.parametrize("name", sorted(SEMANTIC))
def test_semantic_error_after_resign_rejected(hs, base, name):
    header, rows = copy.deepcopy(base)
    _apply_semantic(hs, name, header, rows)
    try:
        header = resign(hs, header, rows)
    except (KeyError, TypeError, ValueError):
        pytest.fail(f"{name}：签名函数本身不应失败（夹具写错）")
    with pytest.raises(hs.SpecsError):
        hs.validate_specs(header, rows)


@pytest.mark.parametrize("name", ["F6_candidate_布尔", "F6_attempt_浮点", "F6_seed_浮点"])
def test_f6_rejected_with_type_message(hs, base, name):
    header, rows = copy.deepcopy(base)
    _apply_semantic(hs, name, header, rows)
    header = resign(hs, header, rows)
    with pytest.raises(hs.SpecsError, match="整数"):
        hs.validate_specs(header, rows)


def test_load_specs_file_level_rejections(hs, base, tmp_path):
    """文件层：空文件、重复字段、旧 schema 都拒绝。"""
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


# ── 跨档 seed 相交 ───────────────────────────────────────────────────────


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
    hs.load_specs_root(root, cells, check_fingerprint=False)  # 正例：原样两档可读
    # 把 xhard5 的偏移改成 xhard4 的，并按新规则重写 xhard5 的 seed 与签名（文件自身仍合法）
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
    with pytest.raises(hs.SpecsError, match="相交"):
        hs.load_specs_root(root, cells, check_fingerprint=False)


def test_load_specs_root_cell_table_rejections(hs, tmp_path):
    root = _root_with(tmp_path, ("xhard5",))
    cells = {key: n for key, n in V9_CELLS.items() if key[1] == "xhard5"}
    hs.load_specs_root(root, cells, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):
        hs.load_specs_root(root, {}, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # 格表外的格
        hs.load_specs_root(root, {("PickXtimes", "xhard5"): 1}, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # 局数不等于 selected 数
        hs.load_specs_root(root, {k: n - 1 for k, n in cells.items()}, check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # 缺档文件
        hs.load_specs_root(root, {**cells, ("StopCube", "xhard4"): V9_CELLS[("StopCube", "xhard4")]},
                           check_fingerprint=False)
    with pytest.raises(hs.SpecsError):  # 局数为布尔
        hs.load_specs_root(root, {k: True for k in cells}, check_fingerprint=False)
