"""L0：本仓与官方 ``016ac1c4`` 的上游字节、``UPSTREAM.json`` 清单与 robomme_hard 的导入边界（C01、C18）。

本仓 = 官方 RoboMME/robomme_benchmark ``016ac1c4`` + ``src/robomme_hard/`` + ``tests/robomme_hard/`` + 两个 ood 文件，
旧仓的上游守卫 ``scripts/parity/upstream_guard.py`` 随私有评估仓走，本文件把它对本仓仍有意义的几道检查改写成
不依赖该工具的独立断言：

- ``src/robomme`` 每个文件逐字节等于 ``git show 016ac1c4:<路径>`` 的 blob，工作区没有多余文件；
- 三个官方入口 ``scripts/{dataset_replay,evaluation,run_example}.py`` 与 ``016ac1c4`` 逐字节相同；
- ``UPSTREAM.json``：登记的 ``src_commit``（旧锚点 ``1fadc0ec``）下 ``src/robomme`` 的树与 ``016ac1c4`` 相同，
  ``robomme_files`` 的键与 sha256 由 git blob 独立复算一致，``vendor`` 已清空；
- 导入边界：robomme_hard 自有文件（shim 除外）的相对导入都落在包内，绝对 ``robomme.*`` 只指向 shim 目标或父类模块；
- 借用闭包：shim 目标在官方源码上的传递依赖不含任何被 robomme_hard 复制的模块。

为什么能逐字节：官方 commit 的 blob 就在本仓 git 对象库里（本仓从它分出），不联网。
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO

#: 官方锚点（完整 40 位 sha）：本仓自它分出，src/robomme 与三个官方入口逐字节等于它
OFFICIAL_COMMIT = "016ac1c4ef3df2b88488abc19db08f3de83647b5"
#: UPSTREAM.json 登记的旧锚点（旧仓对齐的上游 commit）；与 016ac1c4 在 src/robomme 上是同一棵树
MANIFEST_COMMIT = "1fadc0ec50316b60ddcfd8e82ac62ef2b70c18f9"
ENTRIES = ("dataset_replay.py", "evaluation.py", "run_example.py")
HARD = REPO / "src" / "robomme_hard"
MANIFEST = HARD / "UPSTREAM.json"


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True).stdout


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def official_blobs(commit: str) -> dict[str, bytes]:
    """官方 commit 下 src/robomme 每个文件的 blob 字节（一次 ls-tree + 一次 cat-file --batch，不经清单）。"""
    tree = _git("ls-tree", "-r", "-z", commit, "--", "src/robomme")
    entries = []
    for rec in tree.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        entries.append((path.decode(), meta.split()[2].decode()))
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=REPO, check=True, capture_output=True,
                         input="".join(f"{sha}\n" for _, sha in entries).encode()).stdout
    blobs, pos = {}, 0
    for path, sha in entries:
        nl = out.index(b"\n", pos)
        got_sha, kind, size = out[pos:nl].split()
        assert got_sha.decode() == sha and kind == b"blob"
        start = nl + 1
        blobs[path] = out[start:start + int(size)]
        pos = start + int(size) + 1  # blob 后跟一个换行
    return blobs


@pytest.fixture(scope="module")
def blobs() -> dict[str, bytes]:
    return official_blobs(OFFICIAL_COMMIT)


# ---------------------------------------------------------------- 上游字节


def byte_mismatches(root: Path, blobs: dict[str, bytes]) -> list[str]:
    """工作区（以 root 为仓根）src/robomme 与官方 blob 的差异清单：changed／extra／missing。"""
    have = {str(p.relative_to(root)) for p in (root / "src" / "robomme").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")}
    out = [f"missing:{rel}" for rel in sorted(set(blobs) - have)]
    out += [f"extra:{rel}" for rel in sorted(have - set(blobs))]
    out += [f"changed:{rel}" for rel in sorted(have & set(blobs)) if (root / rel).read_bytes() != blobs[rel]]
    return out


def test_src_robomme_bytes_equal_official(blobs):
    assert blobs
    bad = byte_mismatches(REPO, blobs)
    print(f"UPSTREAM_BYTES={'FAIL' if bad else 'PASS'} commit={OFFICIAL_COMMIT[:8]} files={len(blobs)} diff={len(bad)}")
    assert bad == []


def test_byte_mismatches_negative(blobs, tmp_path):
    """判定器负例：小副本里改 1 字节、多一个文件、少一个文件，各被点名。"""
    picked = sorted(rel for rel in blobs if rel.endswith(".py"))[:3]
    sub = {rel: blobs[rel] for rel in picked}
    for rel, data in sub.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    assert byte_mismatches(tmp_path, sub) == []
    first, second = picked[0], picked[1]
    (tmp_path / first).write_bytes(sub[first] + b"#")
    (tmp_path / "src" / "robomme" / "extra.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / second).unlink()
    assert sorted(byte_mismatches(tmp_path, sub)) == sorted(
        [f"changed:{first}", "extra:src/robomme/extra.py", f"missing:{second}"])


def test_official_entry_scripts_equal_official():
    bad = [name for name in ENTRIES
           if (REPO / "scripts" / name).read_bytes() != _git("show", f"{OFFICIAL_COMMIT}:scripts/{name}")]
    print(f"ENTRY_SCRIPTS={'FAIL' if bad else 'PASS'} commit={OFFICIAL_COMMIT[:8]} files={len(ENTRIES)} diff={len(bad)}")
    assert bad == []


# ---------------------------------------------------------------- UPSTREAM.json


def test_manifest_anchor_same_tree_as_official(manifest):
    """清单的旧锚点 1fadc0ec 与 016ac1c4 在 src/robomme 上是同一棵树：清单登记的上游字节对本仓仍然成立。"""
    assert manifest["src_commit"] == MANIFEST_COMMIT
    a = _git("rev-parse", f"{manifest['src_commit']}:src/robomme").strip()
    b = _git("rev-parse", f"{OFFICIAL_COMMIT}:src/robomme").strip()
    assert a == b


def test_manifest_files_equal_official_blobs(manifest, blobs):
    assert set(manifest["robomme_files"]) == set(blobs)
    wrong = [rel for rel, blob in blobs.items() if manifest["robomme_files"][rel] != hashlib.sha256(blob).hexdigest()]
    assert wrong == []


def test_manifest_vendor_empty(manifest):
    """官方编排脚本的 vendor 副本已随私有评估仓移走：清单 vendor 为空。"""
    assert manifest["vendor"] == {}


# ---------------------------------------------------------------- 导入边界与借用闭包


def modname(rel: str) -> str:
    parts = rel[len("src/"):-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve(cur: str, is_pkg: bool, level: int, module: str | None) -> str:
    if level == 0:
        return module or ""
    base = cur.split(".") if is_pkg else cur.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def iter_imports(mod: str, source: str, is_pkg: bool, known: set[str]):
    """产出 (lineno, 解析后的目标模块, 是否相对导入)；``from X import y`` 且 ``X.y`` 是模块时取 ``X.y``。"""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, False
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(mod, is_pkg, node.level, node.module)
            for alias in node.names:
                sub = f"{target}.{alias.name}"
                yield node.lineno, (sub if sub in known else target), node.level > 0


def hard_sources() -> dict[str, str]:
    """robomme_hard 的 .py 文件 → 源码（相对仓根路径为键）。"""
    return {str(p.relative_to(REPO)): p.read_text(encoding="utf-8")
            for p in sorted(HARD.rglob("*.py")) if "__pycache__" not in p.parts}


def abs_import_problems(sources: dict[str, str], manifest: dict) -> list[str]:
    """robomme_hard 自有文件（shim 除外）：相对导入与 ``robomme_hard.*`` 必须落在包内已有模块；
    绝对 ``robomme.*`` 只许指向 shim 目标或父类模块。"""
    shims = {s["shim"] for s in manifest["shims"]}
    allowed = {s["target_module"] for s in manifest["shims"]} | set(manifest.get("parents", ()))
    known = {modname(rel) for rel in sources}
    known_official = {modname(r) for r in manifest["robomme_files"] if r.endswith(".py")}
    bad = []
    for rel, src in sources.items():
        if rel in shims:
            continue
        for lineno, target, relative in iter_imports(modname(rel), src, rel.endswith("__init__.py"),
                                                     known | known_official):
            if relative or target == "robomme_hard" or target.startswith("robomme_hard."):
                if target not in known:
                    bad.append(f"{rel}:{lineno}:{target}")
            elif (target == "robomme" or target.startswith("robomme.")) and target not in allowed:
                bad.append(f"{rel}:{lineno}:{target}")
    return bad


def test_abs_import_boundary(manifest):
    sources = hard_sources()
    bad = abs_import_problems(sources, manifest)
    print(f"ABS_IMPORT={'FAIL' if bad else 'PASS'} files={len(sources)} unresolved={len(bad)}")
    assert bad == []


def test_abs_import_negative(manifest):
    """判定器负例：自有文件里绝对导入一个非 shim 目标的官方模块、相对导入不存在的模块，都被抓到。"""
    sources = {"src/robomme_hard/env_record_wrapper/zz_probe.py":
               "from robomme.env_record_wrapper.RecordWrapper import RobommeRecordWrapper\nfrom .no_such_mod import x\n",
               "src/robomme_hard/env_record_wrapper/__init__.py": ""}
    bad = abs_import_problems(sources, manifest)
    assert len(bad) == 2, bad


def official_edges(manifest: dict, blobs: dict[str, bytes]) -> dict[str, set[str]]:
    """官方源码（从 git blob 读）上的模块依赖边：模块 → 它导入的官方模块集合。"""
    sources = {}
    for rel in manifest["robomme_files"]:
        name = rel.rsplit("/", 1)[-1]
        if rel.endswith(".py") and " " not in name and "-" not in name:
            sources[modname(rel)] = (blobs[rel].decode("utf-8"), rel.endswith("__init__.py"))
    known = set(sources)
    return {mod: {t for _, t, _ in iter_imports(mod, src, is_pkg, known) if t in known}
            for mod, (src, is_pkg) in sources.items()}


def borrowed_hits(manifest: dict, edges: dict[str, set[str]], hard_rels: list[str]) -> list[str]:
    """shim 目标的传递闭包里出现被 robomme_hard 复制（非 shim）的官方模块 → 命中清单。"""
    shims = {s["shim"] for s in manifest["shims"]}
    copied = {"robomme" + modname(rel)[len("robomme_hard"):] for rel in hard_rels if rel not in shims}
    copied &= set(edges)
    hits = []
    for entry in manifest["shims"]:
        seen, stack = set(), [entry["target_module"]]
        while stack:
            for dep in edges.get(stack.pop(), ()):
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        if seen & copied:
            hits.append(f"{entry['target_module']}->{sorted(seen & copied)}")
    return hits


def test_borrowed_deps_closure(manifest, blobs):
    edges = official_edges(manifest, blobs)
    hits = borrowed_hits(manifest, edges, list(hard_sources()))
    print(f"BORROWED_DEPS={'FAIL' if hits else 'PASS'} shims={len(manifest['shims'])} changed_hits={len(hits)}")
    assert hits == []


def test_borrowed_deps_negative(manifest, blobs):
    """判定器负例：在某个 shim 目标的依赖边里加一条指向被复制模块（RecordWrapper）的边 → 命中。"""
    edges = {k: set(v) for k, v in official_edges(manifest, blobs).items()}
    target = manifest["shims"][0]["target_module"]
    copied = "robomme.env_record_wrapper.RecordWrapper"
    assert copied in edges
    edges[target].add(copied)
    assert borrowed_hits(manifest, edges, list(hard_sources())) != []
