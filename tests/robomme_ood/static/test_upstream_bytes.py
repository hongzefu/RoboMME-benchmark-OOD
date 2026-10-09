"""L0: upstream bytes of this repo vs. upstream ``016ac1c4``, the ``UPSTREAM.json`` manifest, and robomme_ood's import boundary (C01, C18).

This repo = upstream RoboMME/robomme_benchmark ``016ac1c4`` + ``src/robomme_ood/`` + ``tests/robomme_ood/`` + two ood files.
The old repo's upstream guard ``scripts/parity/upstream_guard.py`` moved to the private evaluation repo; this file rewrites the checks
that still matter for this repo as independent assertions that do not depend on that tool:

- every file in ``src/robomme`` is byte-identical to the blob of ``git show 016ac1c4:<path>``, with no extra files in the working tree;
- the three upstream entries ``scripts/{dataset_replay,evaluation,run_example}.py`` are byte-identical to ``016ac1c4``;
- ``UPSTREAM.json``: the ``src/robomme`` tree at the registered ``src_commit`` (old anchor ``1fadc0ec``) equals that of ``016ac1c4``,
  the keys and sha256 of ``robomme_files`` match an independent recomputation from git blobs, and ``vendor`` is empty;
- import boundary: relative imports in robomme_ood's own files (shims excluded) all resolve inside the package, and absolute ``robomme.*`` points only to shim targets or parent-class modules;
- borrow closure: the transitive dependencies of shim targets in upstream source contain no module copied by robomme_ood.

Why byte-level comparison works: the upstream commit's blobs are in this repo's git object store (this repo branched from it); no network needed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO

#: Upstream anchor (full 40-char sha): this repo branched from it; src/robomme and the three upstream entries are byte-identical to it
OFFICIAL_COMMIT = "016ac1c4ef3df2b88488abc19db08f3de83647b5"
#: Old anchor registered in UPSTREAM.json (the upstream commit the old repo was aligned to); same src/robomme tree as 016ac1c4
MANIFEST_COMMIT = "1fadc0ec50316b60ddcfd8e82ac62ef2b70c18f9"
ENTRIES = ("dataset_replay.py", "evaluation.py", "run_example.py")
HARD = REPO / "src" / "robomme_ood"
MANIFEST = HARD / "UPSTREAM.json"


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True).stdout


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def official_blobs(commit: str) -> dict[str, bytes]:
    """Blob bytes of every src/robomme file at the upstream commit (one ls-tree + one cat-file --batch, not via the manifest)."""
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
        pos = start + int(size) + 1  # a newline follows each blob
    return blobs


@pytest.fixture(scope="module")
def blobs() -> dict[str, bytes]:
    return official_blobs(OFFICIAL_COMMIT)


# ---------------------------------------------------------------- upstream bytes


def byte_mismatches(root: Path, blobs: dict[str, bytes]) -> list[str]:
    """Differences between src/robomme in the working tree (root as repo root) and upstream blobs: changed / extra / missing."""
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
    """Checker negatives: in a small copy, changing 1 byte, adding one file, removing one file are each reported."""
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
    """The manifest's old anchor 1fadc0ec and 016ac1c4 have the same src/robomme tree: upstream bytes registered in the manifest still hold for this repo."""
    assert manifest["src_commit"] == MANIFEST_COMMIT
    a = _git("rev-parse", f"{manifest['src_commit']}:src/robomme").strip()
    b = _git("rev-parse", f"{OFFICIAL_COMMIT}:src/robomme").strip()
    assert a == b


def test_manifest_files_equal_official_blobs(manifest, blobs):
    assert set(manifest["robomme_files"]) == set(blobs)
    wrong = [rel for rel, blob in blobs.items() if manifest["robomme_files"][rel] != hashlib.sha256(blob).hexdigest()]
    assert wrong == []


def test_manifest_vendor_empty(manifest):
    """The vendor copies of upstream orchestration scripts moved to the private evaluation repo: manifest vendor is empty."""
    assert manifest["vendor"] == {}


# ---------------------------------------------------------------- import boundary and borrow closure


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
    """Yields (lineno, resolved target module, is relative import); for ``from X import y`` where ``X.y`` is a module, ``X.y`` is used."""
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
    """robomme_ood .py files -> source (keyed by path relative to repo root)."""
    return {str(p.relative_to(REPO)): p.read_text(encoding="utf-8")
            for p in sorted(HARD.rglob("*.py")) if "__pycache__" not in p.parts}


def abs_import_problems(sources: dict[str, str], manifest: dict) -> list[str]:
    """robomme_ood's own files (shims excluded): relative imports and ``robomme_ood.*`` must resolve to existing modules in the package;
    absolute ``robomme.*`` may only point to shim targets or parent-class modules."""
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
            if relative or target == "robomme_ood" or target.startswith("robomme_ood."):
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
    """Checker negatives: an absolute import of a non-shim-target upstream module and a relative import of a nonexistent module in an own file are both caught."""
    sources = {"src/robomme_ood/env_record_wrapper/zz_probe.py":
               "from robomme.env_record_wrapper.RecordWrapper import RobommeRecordWrapper\nfrom .no_such_mod import x\n",
               "src/robomme_ood/env_record_wrapper/__init__.py": ""}
    bad = abs_import_problems(sources, manifest)
    assert len(bad) == 2, bad


def official_edges(manifest: dict, blobs: dict[str, bytes]) -> dict[str, set[str]]:
    """Module dependency edges in upstream source (read from git blobs): module -> set of upstream modules it imports."""
    sources = {}
    for rel in manifest["robomme_files"]:
        name = rel.rsplit("/", 1)[-1]
        if rel.endswith(".py") and " " not in name and "-" not in name:
            sources[modname(rel)] = (blobs[rel].decode("utf-8"), rel.endswith("__init__.py"))
    known = set(sources)
    return {mod: {t for _, t, _ in iter_imports(mod, src, is_pkg, known) if t in known}
            for mod, (src, is_pkg) in sources.items()}


def borrowed_hits(manifest: dict, edges: dict[str, set[str]], hard_rels: list[str]) -> list[str]:
    """Upstream modules copied (not shimmed) by robomme_ood that appear in the transitive closure of shim targets -> list of hits."""
    shims = {s["shim"] for s in manifest["shims"]}
    copied = {"robomme" + modname(rel)[len("robomme_ood"):] for rel in hard_rels if rel not in shims}
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
    """Checker negative: adding an edge from some shim target to a copied module (RecordWrapper) -> hit."""
    edges = {k: set(v) for k, v in official_edges(manifest, blobs).items()}
    target = manifest["shims"][0]["target_module"]
    copied = "robomme.env_record_wrapper.RecordWrapper"
    assert copied in edges
    edges[target].add(copied)
    assert borrowed_hits(manifest, edges, list(hard_sources())) != []
