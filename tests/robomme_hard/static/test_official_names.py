"""L0：官方名残留检查 ``OFFICIAL_NAMES``（1006-rename-official-names-and-stage3-eval-plan.md 第一部分一、第二部分八.2）。

仓库自造名一律换成官方名：FrameSamp+Modulation 的策略标签 ``perceptual-framesamp-modul``（标识符 ``framesamp_modul``）、
GroundSG 的 ``groundsg``、数据集接口 ``hard-verify``（原第二阶段接口）与 ``ood``（原第三阶段接口）。本测试扫描
``scripts/``、``src/robomme_hard/``、``tests/`` 下 git 跟踪的活代码，旧名出现即失败。

旧名的写法由本文件独立列出（不读被测代码），见 ``PATTERNS``；豁免（白名单）只有：

- 官方家族名形式：``mme-vla``、``mme_vla``、``MME-VLA``、``MME_VLA``、上游类名前缀 ``MMEVLAWebsocket``、``mme_vla_suite``
  （正则本身不匹配这些写法）；
- 别名表本身：``scripts/eval-official/official_defs.py`` 里 ``# >>> LEGACY_NAMES`` 与 ``# <<< LEGACY_NAMES`` 之间的行；
- 标注「历史目录名」（磁盘／NFS 真实路径）或「历史数据键」（已发布数据文件里的键）的行；
- ``XHARD0_IN_TEST_HARD`` 系列（与冻结配置互相引用，正则不匹配）；
- 本文件自身。

不扫：冻结配置 ``scripts/configs/``、vendored 官方源码 ``scripts/parity/official/``、三个上游原样入口、包内规格数据
``src/robomme_hard/env_metadata/``、上游字节清单 ``UPSTREAM.json``、说明文档 ``*.md``（现行文档由主会话另改）。

判定行：``OFFICIAL_NAMES=PASS|FAIL files=<n> hits=<n>``，失败时逐条列出 ``<路径>:<行号>: <命中>``。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO

ROOTS = ("scripts", "src/robomme_hard", "tests")
SKIP_PREFIX = ("scripts/configs/", "scripts/parity/official/", "src/robomme_hard/env_metadata/")
SKIP_FILES = {"scripts/dataset_replay.py", "scripts/evaluation.py", "scripts/run_example.py",
              "src/robomme_hard/UPSTREAM.json", "tests/robomme_hard/static/test_official_names.py"}
SKIP_SUFFIX = (".md", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mkv", ".npz", ".h5", ".pkl", ".ico", ".woff", ".woff2")
ALIAS_FILE = "scripts/eval-official/official_defs.py"
ALIAS_BEGIN, ALIAS_END = "# >>> LEGACY_NAMES", "# <<< LEGACY_NAMES"
LINE_MARKERS = ("历史目录名", "历史数据键")

#: (名称, 正则)：旧名的全部写法
PATTERNS = (
    ("mme 标签", re.compile(r"(?<![A-Za-z0-9_])mme(?![A-Za-z0-9_]|-vla)")),
    ("mme 标识符", re.compile(r"(?<![A-Za-z0-9])mme_(?!vla)|(?<=[A-Za-z0-9])_mme(?![A-Za-z0-9]|_vla)")),
    ("MME 名", re.compile(r"(?<![A-Za-z0-9_])MME(?![A-Za-z0-9_]|-VLA|-vla)")),
    ("MME 标识符", re.compile(r"(?<![A-Za-z0-9])MME_(?!VLA)|(?<=[A-Za-z0-9])_MME(?![A-Za-z0-9]|_VLA)")),
    ("MME 驼峰", re.compile(r"(?<=[a-z])(?<!Robo)MME(?!VLA)")),
    ("mmesg", re.compile(r"mmesg", re.IGNORECASE)),
    ("mmevla", re.compile(r"mmevla(?!websocket)", re.IGNORECASE)),
    ("test-hard", re.compile(r"test-hard", re.IGNORECASE)),
    ("TEST_HARD 常量", re.compile(r"(?<![A-Za-z0-9_])TEST_HARD0?(?![A-Za-z0-9_])")),
    ("test_hard0", re.compile(r"test_hard0", re.IGNORECASE)),
)


def scan_text(path: str, text: str) -> list[tuple[str, int, str]]:
    """一个文件的命中：[(路径, 行号, 命中摘要)]；按白名单跳过别名表段与标注行。"""
    hits = []
    in_alias = False
    for no, line in enumerate(text.splitlines(), 1):
        if path == ALIAS_FILE:
            if line.strip().startswith(ALIAS_BEGIN):
                in_alias = True
                continue
            if line.strip().startswith(ALIAS_END):
                in_alias = False
                continue
        if in_alias or any(m in line for m in LINE_MARKERS):
            continue
        for name, pat in PATTERNS:
            m = pat.search(line)
            if m:
                lo = max(0, m.start() - 20)
                hits.append((path, no, f"{name} …{line[lo:m.end() + 20].strip()}…"))
                break
    return hits


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--", *ROOTS], cwd=REPO, capture_output=True, text=True, check=True)
    files = []
    for f in out.stdout.split():
        if f.startswith(SKIP_PREFIX) or f in SKIP_FILES or f.lower().endswith(SKIP_SUFFIX):
            continue
        if (REPO / f).is_file():
            files.append(f)
    return sorted(files)


def scan_repo() -> tuple[int, list[tuple[str, int, str]]]:
    files = tracked_files()
    hits = []
    for f in files:
        try:
            text = (REPO / f).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits += scan_text(f, text)
    return len(files), hits


def verdict_line(n_files: int, hits: list) -> str:
    return f"OFFICIAL_NAMES={'FAIL' if hits else 'PASS'} files={n_files} hits={len(hits)}"


# ---------------------------------------------------------------- 正例：真实仓库零残留


def test_official_names_no_legacy_residue():
    n, hits = scan_repo()
    assert n > 100, f"扫描文件数异常：{n}"
    for p, no, what in hits:
        print(f"OFFICIAL_NAMES_HIT {p}:{no}: {what}")
    print(verdict_line(n, hits))
    assert not hits, "\n".join(f"{p}:{no}: {w}" for p, no, w in hits[:50])


def test_alias_block_present_and_closed():
    """别名表段必须存在、成对、且只在 official_defs.py 里出现（全仓别名表只此一份）。"""
    text = (REPO / ALIAS_FILE).read_text(encoding="utf-8")
    assert text.count(ALIAS_BEGIN) == 1 and text.count(ALIAS_END) == 1
    assert text.index(ALIAS_BEGIN) < text.index(ALIAS_END)
    others = [f for f in tracked_files() if f != ALIAS_FILE and ALIAS_BEGIN in (REPO / f).read_text(encoding="utf-8")]
    assert others == []


# ---------------------------------------------------------------- 负例：每种旧写法都被抓到，官方名与豁免不误报


@pytest.mark.parametrize("line", [
    'POLICIES = ("mme", "smvla")',
    "x = load_sibling('mme_client')",
    "--mme-variant ground-sg-oracle",
    "MME_CKPT=/x",
    "WALL_MME=1",
    "x = setup_mme_run()",
    "官方 MME 客户端",
    "route = 'mmesg/oracle/new'",
    "MMESG_ORIG_BLOCKED",
    "label mmevla",
    'dataset="test-hard"',
    "--dataset test-hard0",
    "TEST_HARD0 = 1",
    "from x import TEST_HARD",
    "def test_hard0_rejects():",
    "class FakeMMEClient:",
])
def test_scan_catches_legacy_spellings(line):
    assert scan_text("scripts/x.py", line), line


@pytest.mark.parametrize("line", [
    "third_party/mme-vla/examples/robomme/eval.py",
    "from mme_vla_suite import x",
    "MME_VLA_PY=/py",
    "_ENV_MME_VLA_PY=1; preflight_mme_vla x",
    "class MMEVLAWebsocketClientPolicy: ...",
    "MME-VLA 家族",
    "robomme_hard robomme RoboMME RoboMME-hard",
    "class FakeMMEVLAWebsocketClient(MMEVLAWebsocketClientPolicy): ...",
    "XHARD0_IN_TEST_HARD = os.environ.get('ROBOMME_HARD_XHARD0_IN_TEST_HARD')",
    "policy = 'perceptual-framesamp-modul'; mod = 'framesamp_modul_client'; v = 'groundsg'",
    'dataset="ood" or "hard-verify"',
    "def test_hard_parity_cli(): ...",
    "CKPT=/nfs/x/mmevla-ckpt/79999  # 历史目录名",
    "ids = ('mmevla',)  # 历史数据键",
])
def test_scan_ignores_official_and_whitelisted(line):
    assert scan_text("scripts/x.py", line) == [], line


def test_alias_block_whitelist_only_in_alias_file():
    block = f"{ALIAS_BEGIN}\nA = {{'mme': 1}}\n{ALIAS_END}\nB = 'mmesg'\n"
    assert [h[1] for h in scan_text(ALIAS_FILE, block)] == [4]  # 段内豁免，段外照抓
    assert len(scan_text("scripts/other.py", block)) == 2  # 别的文件里同样的段不豁免


def test_verdict_line_format():
    assert verdict_line(3, []) == "OFFICIAL_NAMES=PASS files=3 hits=0"
    assert verdict_line(3, [("a", 1, "x")]) == "OFFICIAL_NAMES=FAIL files=3 hits=1"
