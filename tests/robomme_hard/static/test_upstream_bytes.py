"""L0：上游守卫 ``scripts/parity/upstream_guard.py`` 的各道检查（C01 shim 部分、C18 上游字节与入口）。

正例对真实仓库跑：``src/robomme`` 与官方 ``1fadc0ec`` 逐字节相同、三个上游入口与 ``git show 1fadc0ec:scripts/<名>`` 相同、
清单自签成立。负例把守卫的模块级路径常量（``REPO``／``HARD``／``MANIFEST``／``VENDOR_DIR``）指到 ``tmp_path`` 下的
小副本，逐个造一种坏法，各自必须 FAIL；``--allow-pending`` 只把字节差异降为 PENDING 并打警告。

为什么能逐字节：``src/robomme`` 按 AGENTS.md P2 冻结，官方 commit 的 blob 就在本仓库 git 对象库里。
负例只改 tmp 副本，绝不写真实仓库；副本里的 ``git show`` 经 ``GIT_DIR`` 指回真实对象库。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO, load_script

#: 官方锚点（L0 期望）：本仓库 src/robomme 冻结所对齐的上游 commit 完整 40 位 sha
#: （2026-10-04 由 ``git rev-parse 1fadc0ec`` 与 UPSTREAM.json 的 ``src_commit`` 两边核对一致后写死）。
UPSTREAM_COMMIT = "1fadc0ec50316b60ddcfd8e82ac62ef2b70c18f9"
ENTRIES = ("dataset_replay.py", "evaluation.py", "run_example.py")


@pytest.fixture(scope="module")
def guard():
    return load_script("parity/upstream_guard.py")


@pytest.fixture(scope="module")
def real_manifest(guard):
    m = guard.load_manifest()
    m["manifest_sha256"] = "verified"
    return m


def _git(*args: str, cwd: Path = REPO) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


# ---------------------------------------------------------------- 正例：真实仓库


def test_manifest_anchor_is_official_commit(guard, real_manifest):
    assert real_manifest["src_commit"] == guard.SRC_COMMIT
    assert real_manifest["src_commit"] == UPSTREAM_COMMIT
    # 清单登记的文件集合 = 官方 commit 下 src/robomme 的文件集合（独立从 git 树取）。
    names = _git("ls-tree", "-r", "-z", "--name-only", real_manifest["src_commit"], "--", "src/robomme")
    official = {x.decode() for x in names.split(b"\0") if x}
    assert set(real_manifest["robomme_files"]) == official


def _official_blobs(commit: str) -> dict[str, bytes]:
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


def test_src_robomme_bytes_equal_official_blobs_independently(real_manifest):
    """独立证明：工作区 src/robomme 每个文件逐字节等于官方 git blob，不经 UPSTREAM.json 的 sha 中转；
    同时核清单登记的 sha 就是该 blob 的 sha256、工作区没有多余文件。"""
    blobs = _official_blobs(real_manifest["src_commit"])
    assert set(real_manifest["robomme_files"]) == set(blobs)
    have = {str(p.relative_to(REPO)) for p in (REPO / "src" / "robomme").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")}
    assert have == set(blobs)
    bad = [rel for rel, blob in blobs.items() if (REPO / rel).read_bytes() != blob]
    assert bad == []
    wrong_sha = [rel for rel, blob in blobs.items()
                 if real_manifest["robomme_files"][rel] != hashlib.sha256(blob).hexdigest()]
    assert wrong_sha == []


def test_real_repo_upstream_bytes_pass(guard, real_manifest, capsys):
    assert guard.check_upstream_bytes(real_manifest) is True
    out = capsys.readouterr().out
    assert out.startswith("UPSTREAM_BYTES=PASS") and "diff=0" in out


def test_real_repo_entry_scripts_pass(guard, real_manifest, capsys):
    assert guard.check_entry_scripts(real_manifest) is True
    assert capsys.readouterr().out.startswith("ENTRY_SCRIPTS=PASS")
    # 独立复核：三个入口逐字节等于 git show <官方>:scripts/<名>。
    for name in ENTRIES:
        assert (REPO / "scripts" / name).read_bytes() == _git("show", f"{real_manifest['src_commit']}:scripts/{name}")


def test_real_repo_vendor_shims_imports_deps_pass(guard, real_manifest, capsys):
    assert guard.check_vendor(real_manifest) is True
    assert guard.check_shims(real_manifest) is True
    assert guard.check_abs_import(real_manifest) is True
    assert guard.check_borrowed_deps(real_manifest) is True
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split("=", 1)[0] for ln in lines] == ["VENDOR_SAME", "SHIMS", "ABS_IMPORT", "BORROWED_DEPS"]
    assert all("=PASS" in ln for ln in lines)


def test_main_check_strict_by_default(guard, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["upstream_guard.py", "check"])
    assert guard.main() == 0
    assert capsys.readouterr().out.splitlines()[-1] == "UPSTREAM_GUARD=PASS"


def test_main_flags_mutually_exclusive(guard, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["upstream_guard.py", "check", "--allow-pending", "--require-upstream"])
    with pytest.raises(SystemExit) as ei:
        guard.main()
    assert ei.value.code == 2


# ---------------------------------------------------------------- 负例：tmp 小副本


def _copy(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


@pytest.fixture
def mini(guard, tmp_path, monkeypatch):
    """真实仓库守卫相关部分的 tmp 副本；守卫的路径常量全部改指过去。"""
    root = tmp_path / "repo"
    _copy(REPO / "src" / "robomme", root / "src" / "robomme")
    _copy(REPO / "src" / "robomme_hard", root / "src" / "robomme_hard")
    (root / "scripts").mkdir(parents=True)
    for name in ENTRIES:
        shutil.copy2(REPO / "scripts" / name, root / "scripts" / name)
    vendor_rel = guard.VENDOR_DIR.relative_to(REPO)
    _copy(guard.VENDOR_DIR, root / vendor_rel)
    gitdir = _git("rev-parse", "--absolute-git-dir").decode().strip()
    monkeypatch.setenv("GIT_DIR", gitdir)
    monkeypatch.setattr(guard, "REPO", root)
    monkeypatch.setattr(guard, "HARD", root / "src" / "robomme_hard")
    monkeypatch.setattr(guard, "MANIFEST", root / "src" / "robomme_hard" / "UPSTREAM.json")
    monkeypatch.setattr(guard, "VENDOR_DIR", root / vendor_rel)
    return root


def _manifest(guard) -> dict:
    m = guard.load_manifest()
    m["manifest_sha256"] = "verified"
    return m


def _resign(guard, root: Path, mutate) -> None:
    """改清单后按守卫自己的规则重签（造「签名合法但内容不对」的输入）。"""
    path = root / "src" / "robomme_hard" / "UPSTREAM.json"
    m = json.loads(path.read_text())
    m.pop("manifest_sha256")
    mutate(m)
    m["manifest_sha256"] = hashlib.sha256(guard.canonical(m).encode()).hexdigest()
    path.write_text(json.dumps(m, indent=1, ensure_ascii=False, sort_keys=True) + "\n")


def test_mini_copy_passes_everything(guard, mini, capsys):
    """副本本身必须全过，负例的 FAIL 才能归因于植入的那一处。"""
    m = _manifest(guard)
    assert guard.check_upstream_bytes(m)
    assert guard.check_entry_scripts(m)
    assert guard.check_vendor(m)
    assert guard.check_shims(m)
    assert guard.check_abs_import(m)
    assert guard.check_borrowed_deps(m)


def _flip_one_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def test_upstream_bytes_one_byte_changed_fails(guard, mini, capsys):
    _flip_one_byte(mini / "src" / "robomme" / "robomme_env" / "BinFill.py")
    assert guard.check_upstream_bytes(_manifest(guard)) is False
    out = capsys.readouterr().out
    assert "UPSTREAM_BYTES=FAIL" in out and "changed=1" in out


def test_upstream_bytes_extra_file_fails(guard, mini, capsys):
    (mini / "src" / "robomme" / "extra.py").write_text("x = 1\n")
    assert guard.check_upstream_bytes(_manifest(guard)) is False
    assert "extra=1" in capsys.readouterr().out


def test_upstream_bytes_missing_file_fails(guard, mini, capsys):
    (mini / "src" / "robomme" / "logging_utils.py").unlink()
    assert guard.check_upstream_bytes(_manifest(guard)) is False
    assert "missing=1" in capsys.readouterr().out


def test_upstream_bytes_allow_pending_passes_with_warning(guard, mini, capsys):
    _flip_one_byte(mini / "src" / "robomme" / "robomme_env" / "BinFill.py")
    assert guard.check_upstream_bytes(_manifest(guard), allow_pending=True) is True
    cap = capsys.readouterr()
    assert "UPSTREAM_BYTES=PENDING" in cap.out
    assert "--allow-pending" in cap.err and "不得作为验收证据" in cap.err


def test_main_allow_pending_does_not_excuse_entry_scripts(guard, mini, monkeypatch, capsys):
    _flip_one_byte(mini / "src" / "robomme" / "robomme_env" / "BinFill.py")
    monkeypatch.setattr(sys, "argv", ["upstream_guard.py", "check", "--allow-pending"])
    assert guard.main() == 0
    assert capsys.readouterr().out.splitlines()[-1] == "UPSTREAM_GUARD=PASS"
    with (mini / "scripts" / "evaluation.py").open("a") as fh:
        fh.write("# 多一行\n")
    assert guard.main() == 1
    assert capsys.readouterr().out.splitlines()[-1] == "UPSTREAM_GUARD=FAIL"


def test_main_strict_fails_on_byte_change(guard, mini, monkeypatch, capsys):
    _flip_one_byte(mini / "src" / "robomme" / "robomme_env" / "BinFill.py")
    monkeypatch.setattr(sys, "argv", ["upstream_guard.py", "check"])
    assert guard.main() == 1
    assert capsys.readouterr().out.splitlines()[-1] == "UPSTREAM_GUARD=FAIL"


@pytest.mark.parametrize("name", ENTRIES)
def test_entry_script_extra_line_fails(guard, mini, name, capsys):
    with (mini / "scripts" / name).open("a") as fh:
        fh.write("\n")
    assert guard.check_entry_scripts(_manifest(guard)) is False
    assert f"{name}:changed" in capsys.readouterr().out


def test_entry_script_missing_fails(guard, mini, capsys):
    (mini / "scripts" / "run_example.py").unlink()
    assert guard.check_entry_scripts(_manifest(guard)) is False
    assert "run_example.py:missing" in capsys.readouterr().out


def test_vendor_changed_fails(guard, mini, capsys):
    _flip_one_byte(mini / guard.VENDOR_DIR.relative_to(mini) / "generate_dataset.py")
    assert guard.check_vendor(_manifest(guard)) is False
    assert "VENDOR_SAME=FAIL" in capsys.readouterr().out


SHIM = Path("src/robomme_hard/env_record_wrapper/FailAwareWrapper.py")


def test_shim_wrong_target_fails(guard, mini, capsys):
    p = mini / SHIM
    p.write_text(p.read_text().replace("FailAwareWrapper\")", "MultiStepDemonstrationWrapper\")"))
    assert guard.check_shims(_manifest(guard)) is False
    assert f"{SHIM}:body" in capsys.readouterr().out


def test_shim_extra_code_lines_fail(guard, mini, capsys):
    """shim 体里塞进两行额外代码必须 FAIL。"""
    with (mini / SHIM).open("a") as fh:
        fh.write("X = 1\nY = 2\n")
    assert guard.check_shims(_manifest(guard)) is False


def test_shim_one_extra_code_line_fails(guard, mini, capsys):
    """shim 只多 1 行代码也必须 FAIL：守卫 12.393 起要求非注释行恰为两行（原判据「> 3 行」会放过这一行）。
    同时多加一行注释与空行，证明注释、空行不计入、只有代码行触发。"""
    with (mini / SHIM).open("a") as fh:
        fh.write("\n# 注释行不算代码\nX = 1\n")
    assert guard.check_shims(_manifest(guard)) is False
    out = capsys.readouterr().out
    assert out.startswith("SHIMS=FAIL") and f"{SHIM}:body" in out


def test_shim_comment_and_blank_lines_only_pass(guard, mini, capsys):
    """对照：只加注释与空行不改变 shim 判定（上一条 FAIL 确由那一行代码引起）。"""
    with (mini / SHIM).open("a") as fh:
        fh.write("\n# 只是注释\n\n")
    assert guard.check_shims(_manifest(guard)) is True
    assert capsys.readouterr().out.startswith("SHIMS=PASS")


def test_shim_missing_fails(guard, mini, capsys):
    (mini / SHIM).unlink()
    assert guard.check_shims(_manifest(guard)) is False
    assert f"{SHIM}:missing" in capsys.readouterr().out


OWN = Path("src/robomme_hard/robomme_env/utils/difficulty.py")


def test_illegal_absolute_import_fails(guard, mini, capsys):
    """自有文件绝对导入一个既非 shim 目标、也非父类模块的官方模块。"""
    with (mini / OWN).open("a") as fh:
        fh.write("\nfrom robomme.robomme_env.utils.vqa_options import get_vqa_options  # noqa\n")
    assert guard.check_abs_import(_manifest(guard)) is False
    assert "ABS_IMPORT=FAIL" in capsys.readouterr().out


def test_unresolved_relative_import_fails(guard, mini, capsys):
    with (mini / OWN).open("a") as fh:
        fh.write("\nfrom .no_such_module import x  # noqa\n")
    assert guard.check_abs_import(_manifest(guard)) is False


def test_borrowed_dep_copied_instead_of_shimmed_fails(guard, mini, capsys):
    """官方 FailAwareWrapper 依赖 robomme.logging_utils；把后者从 shim 清单拿掉（hard 包里那份就算「复制件」），
    借用闭包里出现复制件，必须 FAIL。"""
    _resign(guard, mini, lambda m: m.__setitem__(
        "shims", [s for s in m["shims"] if s["target_module"] != "robomme.logging_utils"]))
    assert guard.check_borrowed_deps(_manifest(guard)) is False
    assert "robomme.logging_utils" in capsys.readouterr().out


def test_manifest_tampered_fails(guard, mini):
    path = mini / "src" / "robomme_hard" / "UPSTREAM.json"
    m = json.loads(path.read_text())
    first = sorted(m["robomme_files"])[0]
    m["robomme_files"][first] = "0" * 64
    path.write_text(json.dumps(m, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    with pytest.raises(SystemExit) as ei:
        guard.load_manifest()
    assert "manifest_tampered" in str(ei.value)


def test_manifest_short_commit_fails_even_if_resigned(guard, mini):
    _resign(guard, mini, lambda m: m.__setitem__("src_commit", m["src_commit"][:8]))
    with pytest.raises(SystemExit) as ei:
        guard.load_manifest()
    assert "src_commit_not_40_hex" in str(ei.value)


def test_resigned_wrong_file_hash_fails_bytes(guard, mini, capsys):
    """签名合法但官方文件 sha 被改：字节检查必须 FAIL（签名只防篡改，不替代比对）。"""
    _resign(guard, mini, lambda m: m["robomme_files"].__setitem__(sorted(m["robomme_files"])[0], "0" * 64))
    assert guard.check_upstream_bytes(_manifest(guard)) is False
