"""L0：顶层入口（C18 入口白名单、生产不依赖 tests、``run_example.EPISODE_LIMITS`` 与官方元数据一致）。

- ``evaluation_ood.py`` 与官方 ``016ac1c4`` 的 ``scripts/evaluation.py``（经 ``git show`` 读，不读工作区）恰好差白名单
  四处：3 个单行 hunk（import、``dataset=DATASET``、``max_steps=DATASET_MAX_STEPS[DATASET]``）加 1 个纯插入块
  （``TASKS`` 之前的数据集选择块：``hard-verify``↔1300、``ood``↔1800，默认 ``ood``）。
- ``scripts/`` 恰好五个文件：官方三入口加 ``evaluation_ood.py``、``README_ood.md``，没有子目录。
- ``scripts/``、``challenge_interface/`` 与 ``src/robomme_ood/`` 的生产代码不 import ``tests``（AST 收集 import 语句，L0 允许）。
"""
from __future__ import annotations

import ast
import difflib
import json
import re
import subprocess
import typing
from pathlib import Path

import pytest

from tests.robomme_ood._support.loaders import REPO, load_script
from tests.robomme_ood.contract.test_constants import DATASET_MAX_STEPS, DEFAULT_DATASET

SCRIPTS = REPO / "scripts"
#: 官方 RoboMME/robomme_benchmark 的锚点 commit（本仓自它分出；完整 40 位 sha）
OFFICIAL_COMMIT = "016ac1c4ef3df2b88488abc19db08f3de83647b5"
SCRIPTS_SET = {"dataset_replay.py", "evaluation.py", "run_example.py", "evaluation_ood.py", "README_ood.md"}
PRODUCTION_DIRS = (REPO / "scripts", REPO / "challenge_interface", REPO / "src" / "robomme_ood")


# ---------------------------------------------------------------- evaluation_ood 与官方 evaluation 的差异


def single_line_hunks(old: str, new: str, *, allow_insert: bool = False):
    """逐行差异；每块必须是「一行换一行」（插入块之前行号相同，之后按插入行数平移），否则返回 None 表示形态不合。

    ``allow_insert`` 为假时返回 [(新文件行号(1 起), 旧行, 新行)]，且不允许任何插入；为真时最多允许一个纯插入块，
    返回 (单行替换列表, 插入块 (新文件起始行号(1 起), [插入的行]) 或 None)。
    """
    a, b = old.splitlines(), new.splitlines()
    out, insert = [], None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert" and allow_insert and insert is None:
            insert = (j1 + 1, b[j1:j2])
            continue
        shift = len(insert[1]) if insert else 0
        if tag != "replace" or i2 - i1 != 1 or j2 - j1 != 1 or j1 != i1 + shift:
            return None
        out.append((j1 + 1, a[i1], b[j1]))
    return (out, insert) if allow_insert else out


def _builder_call(tree: ast.AST) -> ast.Call:
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "BenchmarkEnvBuilder"]
    assert len(calls) == 1
    return calls[0]


def _official_evaluation() -> str:
    return subprocess.run(["git", "show", f"{OFFICIAL_COMMIT}:scripts/evaluation.py"], cwd=REPO, check=True,
                          capture_output=True).stdout.decode("utf-8")


def test_evaluation_ood_diff_is_three_single_line_hunks_and_dataset_block():
    old = _official_evaluation()
    new = (SCRIPTS / "evaluation_ood.py").read_text(encoding="utf-8")
    res = single_line_hunks(old, new, allow_insert=True)
    assert res is not None, "差异形态不合"
    hunks, insert = res
    assert len(hunks) == 3 and insert is not None, res
    (l1, o1, n1), (l2, o2, n2), (l3, o3, n3) = hunks
    # 1) import：只把包名换成 robomme_ood。
    assert o1 == "from robomme.env_record_wrapper import BenchmarkEnvBuilder"
    assert n1 == "from robomme_ood.env_record_wrapper import BenchmarkEnvBuilder"
    tree = ast.parse(new)
    first_def = min(n.lineno for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    assert l1 < first_def
    # 2) dataset：test → DATASET，缩进不变。
    assert o2.strip() == 'dataset="test",' and n2.strip() == 'dataset=DATASET,'
    assert o2[: len(o2) - len(o2.lstrip())] == n2[: len(n2) - len(n2.lstrip())]
    # 3) max_steps：整数字面值 → 按数据集查 DATASET_MAX_STEPS，缩进不变。
    mo = re.match(r"^(\s*)max_steps=\d+,", o3)
    mn = re.match(r"^(\s*)max_steps=DATASET_MAX_STEPS\[DATASET\],", n3)
    assert mo and mn and mo.group(1) == mn.group(1)
    # 位置：后两处恰是 BenchmarkEnvBuilder(...) 调用的 dataset／max_steps 关键字参数。
    kw = {k.arg: k for k in _builder_call(tree).keywords}
    assert kw["dataset"].value.lineno == l2
    assert isinstance(kw["dataset"].value, ast.Name) and kw["dataset"].value.id == "DATASET"
    assert kw["max_steps"].value.lineno == l3
    assert isinstance(kw["max_steps"].value, ast.Subscript)
    # 4) 插入块：只在 TASKS 之前、只含注释与两个赋值——两个接口与步数配对（hard-verify 1300、ood 1800），默认 ood。
    start, block = insert
    assert block[-1].startswith("DATASET = ") and new.splitlines()[start - 1 + len(block)].startswith("TASKS = ")
    code = [ln for ln in block if not ln.startswith("#")]
    assigns = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("\n".join(code)).body}
    assert assigns == {"DATASET_MAX_STEPS": DATASET_MAX_STEPS, "DATASET": DEFAULT_DATASET}, assigns
    print(f"BENCH_ENTRY_DIFF=PASS changes={len(hunks) + 1} unexpected=0")


def test_single_line_hunks_negatives():
    base = "a\nb\nc\n"
    assert single_line_hunks(base, base) == []
    assert single_line_hunks(base, "a\nB\nc\n") == [(2, "b", "B")]
    assert single_line_hunks(base, "a\nb\nx\nc\n") is None  # 多一行
    assert single_line_hunks(base, "a\nc\n") is None  # 少一行
    assert single_line_hunks(base, "a\nB\nC\n") is None  # 两行连成一块
    # 允许一个插入块：插入后的单行替换按插入行数平移；两个插入块、删行仍判不合
    assert single_line_hunks(base, "a\nx\ny\nb\nC\n", allow_insert=True) == ([(5, "c", "C")], (2, ["x", "y"]))
    assert single_line_hunks(base, "x\na\nb\ny\nc\n", allow_insert=True) is None
    assert single_line_hunks(base, "a\nc\n", allow_insert=True) is None


# ---------------------------------------------------------------- 入口清单


def test_scripts_has_exactly_five_files():
    """``scripts/`` 只有五个文件、没有子目录（对拍、生成、评估编排工具都在私有评估仓）。"""
    entries = [p for p in SCRIPTS.iterdir() if p.name != "__pycache__"]  # 按路径加载脚本时解释器会写字节码缓存
    assert {p.name for p in entries} == SCRIPTS_SET
    assert [p.name for p in entries if p.is_dir()] == []
    tracked = subprocess.run(["git", "ls-files", "--", "scripts"], cwd=REPO, check=True, capture_output=True,
                             text=True).stdout.split()
    assert {t.removeprefix("scripts/") for t in tracked} == SCRIPTS_SET


# ---------------------------------------------------------------- 生产代码不 import tests


def imports_of_tests(source: str) -> list[str]:
    """源码中指向 ``tests`` 包的导入（import／from-import／importlib.import_module／__import__ 字面值）。"""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name == "tests" or a.name.startswith("tests.")]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "tests" or node.module.startswith("tests."):
                hits.append(node.module)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            arg = node.args[0].value
            if name in ("import_module", "__import__") and isinstance(arg, str) \
                    and (arg == "tests" or arg.startswith("tests.")):
                hits.append(arg)
    return hits


def _production_py() -> list[Path]:
    out = []
    for d in PRODUCTION_DIRS:
        out += [p for p in d.rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(out)


def test_production_code_does_not_import_tests():
    files = _production_py()
    assert files
    bad = {str(p.relative_to(REPO)): h for p in files if (h := imports_of_tests(p.read_text(encoding="utf-8")))}
    assert bad == {}


@pytest.mark.parametrize("src", [
    "import tests\n",
    "import tests.robomme_ood._support.loaders as L\n",
    "from tests.robomme_ood._support import loaders\n",
    "from tests import conftest\n",
    "import importlib\nimportlib.import_module('tests.robomme_ood._support.loaders')\n",
    "__import__('tests')\n",
])
def test_imports_of_tests_catches(src):
    assert imports_of_tests(src)


@pytest.mark.parametrize("src", [
    "import testscenario\n",
    "from .tests import x\n",
    "from robomme import tests_helper\n",
    "s = 'import tests'\n",
])
def test_imports_of_tests_ignores_non_tests(src):
    assert imports_of_tests(src) == []


# ---------------------------------------------------------------- run_example.EPISODE_LIMITS


def test_episode_limits_match_official_metadata():
    run_example = load_script("run_example.py")
    limits = run_example.EPISODE_LIMITS
    meta_root = REPO / "src" / "robomme" / "env_metadata"
    assert set(limits) == {p.name for p in meta_root.iterdir() if p.is_dir()}
    assert set(typing.get_args(run_example.DatasetType)) == set(limits)
    tasks = set(typing.get_args(run_example.TaskID)) - {"All"}
    for split, limit in limits.items():
        files = sorted((meta_root / split).glob("record_dataset_*_metadata.json"))
        seen = set()
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            seen.add(d["env_id"])
            assert d["record_count"] == limit, (split, f.name)
            assert sorted(r["episode"] for r in d["records"]) == list(range(limit)), (split, f.name)
        assert seen == tasks, split
