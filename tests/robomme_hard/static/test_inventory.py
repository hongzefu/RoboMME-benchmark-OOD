"""L0：契约清单元测试（计划细则 4.3、4.5.1）——把「覆盖所有部分」与「契约真被执行」变成可机检的两条判定行。

期望全部来自独立来源，不读被测代码：
- 文件全集 = ``git ls-files src/robomme_hard scripts challenge_interface``；每个现行文件必须恰在总表
  ``tests/robomme_hard/contract/benchmark_contracts.json`` 的 ``files``（所属契约 id 列表）或 ``exempt``（豁免类别与理由）之一，
  ``files`` 里的契约 id 都必须存在于 ``entries``，不得登记已不存在的文件。以 ``/`` 结尾的键是整目录登记
  （``src/robomme/`` 冻结官方包，由 C18 上游字节守卫整体覆盖），要求该目录下确有 git 跟踪文件。
- 用例全集 = 两个子进程的实际收集结果之并（比解析 ``def`` 名准：能看到参数化、条件跳过模块与 slow 用例）：
  ``pytest --collect-only -q tests``（实测约 7 s；资源守卫在这里跳过 ``tests/robomme_hard/sim``）与
  ``pytest --collect-only -q --allow-sim-reset tests/robomme_hard/sim``（只收集、不 reset，L4 仿真冒烟条目的 nodeid 由此核对）。
- 契约 id 撞号：同一 id 出现在多份 ``contracts.delta.json`` 且两边 source 涉及的文件名不相交，视为两份不同契约，
  报冲突（总表合并时不拼接，见总表 ``merge_rule``）。

两条判定行：
- ``TEST_INVENTORY=PASS|FAIL unclassified=<n> stale=<n> exempt=<n>``
- ``TEST_CONTRACTS=PASS|FAIL entries=<n> verified=<n> conditional=<n> blocked=<n> planned=<n> missing=<n> pending=<n>``

``missing`` 是 verified 条目（以及 conditional／blocked 条目里已登记的）nodeid 去掉参数化后缀后在收集结果里找不到的个数；
``pending`` 是 planned 条目数，即尚无执行证据的覆盖缺口。

**判定行与本测试通过条件的区分**：``pending`` 不为 0 时 ``TEST_CONTRACTS`` 判定行照实写 FAIL 并列出 planned 条目——
它是给验收表（``pending=0``）看的缺口报告；但本元测试作为日常门禁只要求 ``missing=0``、``unclassified=0``、
``stale=0`` 与结构合法（状态取值、verified 有 nodeids、blocked／conditional 有说明、增量无撞号），不因 pending 失败。
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO
from tests.robomme_hard._support.resource_policy import ENV_LEDGER, ENV_MODE

TOTAL = REPO / "tests" / "robomme_hard" / "contract" / "benchmark_contracts.json"
ROOTS = ("src/robomme_hard", "scripts", "challenge_interface")
STATUSES = ("planned", "verified", "blocked", "conditional")
NOTE_KEYS = ("note",)  # blocked／conditional 的说明字段


# ---------------------------------------------------------------- 独立全集


def git_files(*paths: str) -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=REPO, capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def base_nodeid(nodeid: str) -> str:
    """去掉参数化后缀：``a.py::test_x[p]`` → ``a.py::test_x``。"""
    return nodeid.split("[", 1)[0]


def _collect(*args: str) -> set[str]:
    """一个子进程只做收集（``--collect-only`` 不执行用例，tests/robomme_hard/sim 也不会 reset），返回去参数化后的 nodeid 集合。

    子进程是独立的 pytest 会话、会自己装资源守卫：去掉父会话的守卫环境变量，免得 sitecustomize 先装一遍、
    pytest_configure 再装一遍而重复打补丁（实测递归溢出）。
    """
    env = {k: v for k, v in os.environ.items() if k not in (ENV_MODE, ENV_LEDGER)}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
        cwd=REPO, capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, f"收集失败 {args}（exit={proc.returncode}）：\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    ids = {base_nodeid(line.strip()) for line in proc.stdout.splitlines() if line.startswith("tests/") and "::" in line}
    assert ids, f"收集结果为空：{args}"
    return ids


def collect_nodeids() -> set[str]:
    """全部用例（不加 -m，slow 也在内）∪ tests/robomme_hard/sim（带 --allow-sim-reset 只收集）。"""
    return _collect("tests/robomme_hard") | _collect("--allow-sim-reset", "tests/robomme_hard/sim")


_FILE_TOKEN = re.compile(r"[\w.-]+\.(?:py|jsonl|json|sh|html|toml)\b")


def source_files(source) -> set[str]:
    """source（字符串或列表）里出现的文件名集合，只取基名；用于判断两条同 id 条目是不是同一份契约。"""
    items = source if isinstance(source, list) else [source]
    return {m.group(0) for x in items for m in _FILE_TOKEN.finditer(str(x))}


def delta_conflicts(deltas: dict[str, list[dict]]) -> list[tuple[str, str, str]]:
    """同 id 跨增量且 source 文件名不相交 → (id, 增量 A, 增量 B)。"""
    seen: dict[str, list[tuple[str, set[str]]]] = {}
    out = []
    for path, entries in sorted(deltas.items()):
        for e in entries:
            files = source_files(e.get("source", ""))
            for other_path, other_files in seen.get(e["id"], []):
                if other_path != path and not (files & other_files):
                    out.append((e["id"], other_path, path))
            seen.setdefault(e["id"], []).append((path, files))
    return sorted(out)


def load_deltas() -> dict[str, list[dict]]:
    out = {}
    for p in sorted((REPO / "tests" / "robomme_hard").rglob("contracts.delta.json")):
        out[str(p.relative_to(REPO))] = json.loads(p.read_text(encoding="utf-8"))
    assert out, "找不到任何 contracts.delta.json"
    return out


def nodeid_found(nodeid: str, collected: set[str], prefixes: set[str]) -> bool:
    """登记的 nodeid 可以是函数级、类级或文件级；能落到收集结果上即算找到。"""
    b = base_nodeid(nodeid)
    return b in collected or b in prefixes


def prefixes_of(collected: set[str]) -> set[str]:
    out = set()
    for n in collected:
        parts = n.split("::")
        for i in range(1, len(parts)):
            out.add("::".join(parts[:i]))
    return out


# ---------------------------------------------------------------- 检查函数（负例直接喂改坏的副本）


def check_inventory(total: dict, tracked: set[str]) -> dict:
    files, exempt = total.get("files", {}), total.get("exempt", {})
    entry_ids = {e["id"] for e in total.get("entries", [])}
    file_keys = [k for k in files if not k.endswith("/")]
    dir_keys = [k for k in files if k.endswith("/")]
    registered = set(file_keys) | set(exempt)
    unclassified = sorted(tracked - registered)
    stale = sorted(registered - tracked)
    stale += sorted(d for d in dir_keys if not git_files(d))
    both = sorted(set(files) & set(exempt))
    unknown = sorted({(k, c) for k, v in files.items() for c in v if c not in entry_ids})
    empty = sorted(k for k, v in files.items() if not v)
    bad_exempt = sorted(k for k, v in exempt.items()
                        if not (isinstance(v, dict) and v.get("category") and v.get("reason")))
    ok = not (unclassified or stale or both or unknown or empty or bad_exempt)
    line = (f"TEST_INVENTORY={'PASS' if ok else 'FAIL'} unclassified={len(unclassified)} "
            f"stale={len(stale)} exempt={len(exempt)}")
    return dict(ok=ok, line=line, unclassified=unclassified, stale=stale, both=both,
                unknown=unknown, empty=empty, bad_exempt=bad_exempt)


def check_contracts(total: dict, collected: set[str], deltas: dict[str, list[dict]] | None = None) -> dict:
    entries = total.get("entries", [])
    keys = total["entry_keys"]
    domains = {d["id"] for d in total["domains"]}
    prefixes = prefixes_of(collected)
    counts = Counter(e.get("status") for e in entries)
    dup = sorted(i for i, n in Counter(e["id"] for e in entries).items() if n > 1)
    missing_keys = sorted(e["id"] for e in entries if any(k not in e for k in keys))
    bad_status = sorted(e["id"] for e in entries if e.get("status") not in STATUSES)
    bad_domain = sorted(e["id"] for e in entries if e.get("domain") not in domains)
    verified_empty = sorted(e["id"] for e in entries if e.get("status") == "verified" and not e.get("nodeids"))
    undocumented = sorted(e["id"] for e in entries if e.get("status") in ("blocked", "conditional")
                          and not any(str(e.get(k, "")).strip() for k in NOTE_KEYS))
    missing = sorted((e["id"], n) for e in entries if e.get("status") != "planned"
                     for n in e.get("nodeids", []) if not nodeid_found(n, collected, prefixes))
    pending = sorted(e["id"] for e in entries if e.get("status") == "planned")
    conflicts = delta_conflicts(deltas or {})
    structural_ok = not (dup or missing_keys or bad_status or bad_domain or verified_empty or undocumented or conflicts)
    gate_ok = structural_ok and not missing
    line_ok = gate_ok and not pending
    line = (f"TEST_CONTRACTS={'PASS' if line_ok else 'FAIL'} entries={len(entries)} "
            f"verified={counts['verified']} conditional={counts['conditional']} blocked={counts['blocked']} "
            f"planned={counts['planned']} missing={len(missing)} pending={len(pending)}")
    return dict(ok=gate_ok, line_ok=line_ok, line=line, dup=dup, missing_keys=missing_keys, bad_status=bad_status,
                bad_domain=bad_domain, verified_empty=verified_empty, undocumented=undocumented,
                conflicts=conflicts, missing=missing, pending=pending)


# ---------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def total() -> dict:
    return json.loads(TOTAL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tracked() -> set[str]:
    files = git_files(*ROOTS)
    assert files, "git ls-files 为空"
    return files


@pytest.fixture(scope="module")
def collected() -> set[str]:
    return collect_nodeids()


@pytest.fixture(scope="module")
def deltas() -> dict[str, list[dict]]:
    return load_deltas()


# ---------------------------------------------------------------- 正例：真实总表


def test_inventory_every_tracked_file_classified(total, tracked):
    r = check_inventory(total, tracked)
    print(r["line"])
    assert r["ok"], {k: r[k] for k in ("unclassified", "stale", "both", "unknown", "empty", "bad_exempt")}


def test_contracts_nodeids_collected(total, collected, deltas):
    r = check_contracts(total, collected, deltas)
    print(r["line"])
    if r["pending"]:
        print("planned（覆盖缺口，只报告不判门禁）：" + ", ".join(r["pending"]))
    assert r["ok"], {k: r[k] for k in ("dup", "missing_keys", "bad_status", "bad_domain",
                                       "verified_empty", "undocumented", "conflicts", "missing")}


def test_sim_nodeids_collected_without_reset(collected):
    """L4 条目的 nodeid 来自 --allow-sim-reset 的纯收集子进程；只收集，不触发 reset。"""
    assert "tests/robomme_hard/sim/test_reset_matrix.py::test_reset_cell" in collected
    assert "tests/robomme_hard/sim/test_official_one_reset.py::test_official_reset_and_unreachable_ee_step" in collected


def test_src_robomme_registered_as_directory(total):
    """冻结官方包整目录登记一条，且挂在上游字节守卫上。"""
    assert "C18-UPSTREAM-BYTES" in total["files"].get("src/robomme/", [])


# ---------------------------------------------------------------- 负例：tmp 副本改坏后必须 FAIL


def _tmp_copy(total: dict, tmp_path: Path) -> dict:
    p = tmp_path / "benchmark_contracts.json"
    p.write_text(json.dumps(total, ensure_ascii=False), encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8"))


def test_negative_unregistered_file_fails(total, tracked, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    victim = sorted(k for k in bad["files"] if not k.endswith("/"))[0]
    del bad["files"][victim]
    r = check_inventory(bad, tracked)
    assert not r["ok"] and r["unclassified"] == [victim]
    assert r["line"].startswith("TEST_INVENTORY=FAIL unclassified=1 ")


def test_negative_stale_and_unknown_id_fail(total, tracked, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    bad["files"]["scripts/不存在的文件.py"] = ["C18-ENTRY-SET"]
    bad["files"]["scripts/run_example.py"] = ["C99-不存在"]
    bad["files"]["no/such/dir/"] = ["C18-UPSTREAM-BYTES"]
    r = check_inventory(bad, tracked)
    assert not r["ok"]
    assert "scripts/不存在的文件.py" in r["stale"] and "no/such/dir/" in r["stale"]
    assert ("scripts/run_example.py", "C99-不存在") in r["unknown"]


def test_negative_nodeid_not_collected_fails(total, collected, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    e = next(x for x in bad["entries"] if x["status"] == "verified")
    e["nodeids"] = e["nodeids"] + ["tests/robomme_hard/static/test_inventory.py::test_不存在的用例[p0]"]
    r = check_contracts(bad, collected)
    assert not r["ok"] and r["missing"] == [(e["id"], "tests/robomme_hard/static/test_inventory.py::test_不存在的用例[p0]")]
    assert r["line"].startswith("TEST_CONTRACTS=FAIL ") and " missing=1 " in r["line"]


def test_negative_verified_without_nodeids_fails(total, collected, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    e = next(x for x in bad["entries"] if x["status"] == "verified")
    e["nodeids"] = []
    r = check_contracts(bad, collected)
    assert not r["ok"] and r["verified_empty"] == [e["id"]]


def test_negative_status_and_note_rules(total, collected, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    cond = next(x for x in bad["entries"] if x["status"] == "conditional")
    cond.pop("note")
    odd = next(x for x in bad["entries"] if x["status"] == "verified")
    odd["status"] = "done"
    r = check_contracts(bad, collected)
    assert not r["ok"]
    assert r["undocumented"] == [cond["id"]] and r["bad_status"] == [odd["id"]]


def test_pending_only_fails_line_not_gate(total, collected):
    """只有 planned 条目时：判定行 FAIL、门禁通过——两者区分写在模块 docstring。"""
    only = copy.deepcopy(total)
    only["entries"] = [e for e in only["entries"] if e["status"] == "verified"]
    gap = copy.deepcopy(only["entries"][0])
    gap.update(id="T11-合成缺口", status="planned", nodeids=[])
    only["entries"].append(gap)
    r = check_contracts(only, collected)
    assert r["pending"] == ["T11-合成缺口"]
    assert r["ok"] and not r["line_ok"] and r["line"].startswith("TEST_CONTRACTS=FAIL ")


def test_negative_bad_exempt_and_both_fail(total, tracked, tmp_path):
    bad = _tmp_copy(total, tmp_path)
    no_reason = next(iter(bad["exempt"]))
    bad["exempt"][no_reason] = {"category": "说明文档"}
    in_files = next(k for k in bad["files"] if not k.endswith("/"))
    bad["exempt"][in_files] = {"category": "说明文档", "reason": "同时登记在 files"}
    r = check_inventory(bad, tracked)
    assert not r["ok"]
    assert r["bad_exempt"] == [no_reason] and r["both"] == [in_files]


def test_negative_delta_id_collision_fails(total, collected):
    """同 id 跨块且 source 不相交（两份不同契约撞号）必须报冲突；source 相交的同 id 只是同一契约的补充。"""
    fake = {
        "tests/a/contracts.delta.json": [{"id": "C16.99", "source": ["scripts/injection-dev/site_build.py"]}],
        "tests/b/contracts.delta.json": [{"id": "C16.99", "source": "scripts/parity/noise_run_gl.sh"}],
        "tests/c/contracts.delta.json": [{"id": "C08.98", "source": ["src/robomme/x/A.py", "src/robomme/x/B.py"]}],
        "tests/d/contracts.delta.json": [{"id": "C08.98", "source": "src/robomme/x/B.py、src/robomme/x/C.py"}],
    }
    assert delta_conflicts(fake) == [("C16.99", "tests/a/contracts.delta.json", "tests/b/contracts.delta.json")]
    r = check_contracts(total, collected, fake)
    assert not r["ok"] and r["conflicts"]
