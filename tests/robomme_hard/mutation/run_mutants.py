"""植入（变异）执行器：读取全部 ``tests/robomme_hard/**/mutants.json``，逐项植入语义错误，核对指定用例因之失败。

归类（详见同目录 README.md）：
- A 文本替换／数据改写：在 tmp 隔离副本里落盘改源码（每个 old 恰好命中 1 次，否则记 not_applied），受保护的
  ``src/robomme`` 与三个上游入口一律拒绝落盘（红线 R9，记 no_recipe）；
- B 进程内插件：``tests/robomme_hard/unit/hard/mutants_plugin.py``（T4_MUTANT，配 ``plugins/t4_status.py``）或
  ``plugins/mut_inproc.py``（MUT_INPROC），不落盘，在仓库本身运行（只读）；插件在植入前预校验植入点、把是否生效写进
  MUT_STATUS_FILE，未生效记 not_applied；
- C 用户裁决例外：mutants.json 里没有 expect_fail 且带 decision 的条目，计入 not_executable 并放行；
- NR 缺配方：找不到可执行做法（或无 expect_fail 又无裁决），记 no_recipe。

每项先确认原版在同环境下 expect_fail 用例全过（基线），再植入重跑：至少一个 expect_fail 用例（参数化按前缀匹配实例）
在 setup／call 阶段失败，异常类型不是导入／语法错误，异常栈顶不在植入设施（``tests/robomme_hard/mutation/``、
``tests/robomme_hard/unit/hard/mutants_plugin.py``）里，所在文件也无收集错误，才算抓到。

判定行：``TEST_MUTATION=PASS|FAIL seeded= caught= survived= not_executable= not_applied= no_recipe= baseline_fail=
repo_changed=``；通过条件 survived、baseline_fail、not_applied、no_recipe 都为 0，运行前后 git status 不变，且 seeded>0。
逐项记录写 ``artifacts/maint-regress/mutation/last_run*.jsonl``（同名 ``.meta.json`` 记仓库是否被改动）。

用法：
  UV_PROJECT_ENVIRONMENT=<主检出>/.venv uv run --no-sync python tests/robomme_hard/mutation/run_mutants.py [--jobs 4]
      [--only <块前缀或块:编号>...] [--tag <批次名>] [--list]
  … run_mutants.py --merge --tag <批次前缀>   # 只合并 last_run.<前缀>*.jsonl，写 last_run.jsonl 并打印总判定行
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.robomme_hard.mutation.plugins import mut_inproc  # noqa: E402
from tests.robomme_hard.mutation.recipes import RECIPES  # noqa: E402

OUT_DIR = REPO / "artifacts" / "maint-regress" / "mutation"
#: 隔离副本只复制这些（与各块作者自检时的副本口径一致）。
COPY_ITEMS = ("src", "scripts", "tests", "challenge_interface", "pyproject.toml")
#: 受保护：只许进程内植入，不得落盘（AGENTS.md P2／计划红线 R9）。
PROTECTED_PREFIX = "src/robomme/"
UPSTREAM_ENTRIES = {"scripts/dataset_replay.py", "scripts/evaluation.py", "scripts/run_example.py"}
#: 这些异常类型导致的失败不算抓到。
NOT_SEMANTIC = {"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError"}
#: 异常栈顶落在这些植入设施文件里的失败是插件自身出错，不算抓到。
PLUGIN_FILES_PREFIX = ("tests/robomme_hard/mutation/",)
PLUGIN_FILES = {"tests/robomme_hard/unit/hard/mutants_plugin.py"}
PYTEST_TIMEOUT = 280
T4_PLUGIN_DIR = "tests/robomme_hard/unit/hard"


# ───────────────────────────── 读取与归一 ─────────────────────────────


def load_items() -> list[dict]:
    items = []
    # 块名相对 tests/robomme_hard（与 recipes.py 的键、plugins/mut_inproc.py 的块名一致）；官方 tests/ 其余目录不参与植入。
    for path in sorted((REPO / "tests" / "robomme_hard").rglob("mutants.json")):
        rel = path.relative_to(REPO).as_posix()
        block = path.parent.relative_to(REPO / "tests" / "robomme_hard").as_posix()
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["mutants"] if isinstance(data, dict) else data
        for e in entries:
            items.append(normalize(rel, block, e))
    return items


def _is_protected(rel: str) -> bool:
    return rel.startswith(PROTECTED_PREFIX) or rel in UPSTREAM_ENTRIES


def normalize(source: str, block: str, e: dict) -> dict:
    mid = e.get("id")
    key = f"{block}:{mid}"
    expect = list(e.get("expect_fail", e.get("expected_failing")) or [])
    # 类别：A／B 可执行；C 只给「无 expect_fail 且有用户裁决」的例外（放行）；NR 为缺配方（判 FAIL）
    it = {"source": source, "block": block, "id": mid, "key": key, "target": e.get("target"), "expect_fail": expect,
          "category": "NR", "reason": None, "recipe": None}
    if not expect:
        if e.get("decision"):
            it["category"] = "C"
            it["reason"] = f"无 expect_fail 用例（用户裁决例外：{e['decision']}）"
        else:
            it["reason"] = "无 expect_fail 用例且没有用户裁决"
        return it
    method = e.get("method") or e.get("inject") or ""
    if e.get("patch"):
        recipe = {"kind": "text", "patch": e["patch"]}
    elif key in RECIPES:
        recipe = RECIPES[key]
    elif "T4_MUTANT=" in method and "mutants_plugin" in method:
        name = re.search(r"T4_MUTANT=([A-Za-z0-9_]+)", method).group(1)
        # t4_status 在 mutants_plugin 之前把替换改成「恰好命中 1 次」并写植入状态
        recipe = {"kind": "plugin", "plugin": ["mutants_plugin", "tests.robomme_hard.mutation.plugins.t4_status"],
                  "pythonpath": T4_PLUGIN_DIR, "env": {"T4_MUTANT": name}}
    elif key in mut_inproc.SUPPORTED:
        recipe = {"kind": "plugin", "plugin": ["tests.robomme_hard.mutation.plugins.mut_inproc"], "pythonpath": None,
                  "env": {"MUT_INPROC": key}}
    else:
        it["reason"] = "只有文字描述，执行器无对应配方"
        return it
    if recipe["kind"] in ("text", "transform"):
        files = [p["file"] for p in recipe.get("patch", [])] + list(recipe.get("files", []))
        bad = [f for f in files if _is_protected(f)]
        if bad:
            it["reason"] = f"R9：受保护文件不得落盘改 {bad}"
            return it
        it["category"] = "A"
    else:
        it["category"] = "B"
    it["recipe"] = recipe
    return it


# ───────────────────────────── 运行 pytest ─────────────────────────────


def _venv() -> str:
    v = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if v:
        return v
    if (REPO / ".venv").is_dir():
        return str(REPO / ".venv")
    raise SystemExit("缺 UV_PROJECT_ENVIRONMENT（worktree 里没有 .venv，须指向主检出 .venv）")


def run_pytest(root: Path, nodeids: list[str], plugins: list[str], extra_path: list[str], env_extra: dict,
               scratch: Path) -> dict:
    """在 root 下跑指定用例，返回结果记录插件写出的 JSON（另附 rc 与输出尾部）。"""
    fd, out_file = tempfile.mkstemp(suffix=".json", dir=scratch)
    os.close(fd)
    status_file = str(Path(out_file).with_suffix(".status.json"))
    pp = [str(root / "src"), str(root)] + [str(root / p) for p in extra_path]
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=_venv(), PYTHONPATH=":".join(pp), PYTHONDONTWRITEBYTECODE="1",
               MUT_OUTCOME_FILE=out_file, MUT_STATUS_FILE=status_file, **env_extra)
    for k in ("MUT_INPROC", "T4_MUTANT"):
        if k not in env_extra:
            env.pop(k, None)
    cmd = ["uv", "run", "--no-sync", "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:warnings",
           "--tb=line", "-p", "tests.robomme_hard.mutation.plugins.outcome_recorder"]
    for p in plugins:
        cmd += ["-p", p]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd + nodeids, cwd=root, env=env, capture_output=True, text=True, timeout=PYTEST_TIMEOUT)
        rc, tail = proc.returncode, (proc.stdout + proc.stderr)[-3000:]
    except subprocess.TimeoutExpired as exc:
        rc, tail = "timeout", str(exc.stdout or "")[-3000:]
    try:
        data = json.loads(Path(out_file).read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        data = {}
    Path(out_file).unlink(missing_ok=True)
    try:  # 进程内插件写的植入状态；没有该文件即 None
        status = json.loads(Path(status_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = None
    Path(status_file).unlink(missing_ok=True)
    data.update(rc=rc, tail=tail, seconds=round(time.monotonic() - t0, 1), status=status)
    return data


def _final(rec: dict) -> tuple[str, str | None, str | None]:
    """把一个用例各阶段合成最终结局：(passed|failed|skipped, 失败阶段的异常类型, 异常栈顶文件)。"""
    ph = rec.get("phases", {})
    for when in ("setup", "call", "teardown"):
        p = ph.get(when)
        if p and p["outcome"] == "failed":
            return "failed", p.get("exc"), p.get("top_file")
    if any(p["outcome"] == "skipped" for p in ph.values()):
        return "skipped", None, None
    if ph.get("call", {}).get("outcome") == "passed":
        return "passed", None, None
    return "unknown", None, None


def _matches(nodeid: str, want: str) -> bool:
    return nodeid == want or nodeid.startswith(want + "[")


def judge(data: dict, expect: list[str]) -> dict:
    """逐个 expect_fail 汇总：每个期望 → 实例结局列表。"""
    tests = data.get("tests", {})
    per = {}
    for want in expect:
        inst = [(nid, *_final(r)) for nid, r in tests.items() if _matches(nid, want)]
        per[want] = inst
    files = {w.split("::")[0] for w in expect}
    cerr = [c for c in data.get("collect_errors", []) if c["nodeid"].split("::")[0] in files or not c["nodeid"]]
    return {"per": per, "collect_errors": cerr}


def baseline_ok(j: dict) -> tuple[bool, list[str]]:
    bad = []
    if j["collect_errors"]:
        bad.append(f"收集错误 {[c['nodeid'] for c in j['collect_errors']]}")
    for want, inst in j["per"].items():
        if not inst:
            bad.append(f"{want} 未收集到")
        bad += [f"{nid} {out}" for nid, out, _, _ in inst if out != "passed"]
    return not bad, bad


def caught_by(j: dict) -> tuple[list[str], list[str]]:
    """返回 (语义失败的用例实例, 被排除的失败说明)。"""
    if j["collect_errors"]:
        return [], [f"收集错误 {[c['nodeid'] for c in j['collect_errors']]}"]
    hit, excluded = [], []
    for inst in j["per"].values():
        for nid, out, exc, top in inst:
            if out != "failed":
                continue
            if exc in NOT_SEMANTIC:
                excluded.append(f"{nid} {exc}")
            elif top is None:
                excluded.append(f"{nid} {exc} 栈顶文件不明")
            elif top.startswith(PLUGIN_FILES_PREFIX) or top in PLUGIN_FILES:
                excluded.append(f"{nid} {exc} 栈顶在植入设施 {top}")
            else:
                hit.append(nid)
    return hit, excluded


# ───────────────────────────── 隔离副本 ─────────────────────────────


def make_copy(dest: Path) -> Path:
    ign = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    for name in COPY_ITEMS:
        src = REPO / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=ign, symlinks=True)
        elif src.exists():
            shutil.copy2(src, dest / name)
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=_venv(), PYTHONPATH=f"{dest}/src:{dest}", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(["uv", "run", "--no-sync", "python", "-c", "import robomme_hard, robomme; "
                          "print(robomme_hard.__file__); print(robomme.__file__)"],
                         cwd=dest, env=env, capture_output=True, text=True, check=True).stdout.split()
    if not all(p.startswith(f"{dest}/src/") for p in out):
        raise SystemExit(f"隔离副本导入指向不对：{out}")
    return dest


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def apply_a(root: Path, recipe: dict) -> dict[str, bytes]:
    """在副本里植入，返回 {相对路径: 原字节} 供还原；old 未恰好命中 1 次时抛 ValueError（不留半改状态）。"""
    backup: dict[str, bytes] = {}
    if recipe["kind"] == "text":
        new_text: dict[str, str] = {}
        for p in recipe["patch"]:
            f = p["file"]
            if f not in backup:
                backup[f] = (root / f).read_bytes()
                new_text[f] = backup[f].decode("utf-8")
            n = new_text[f].count(p["old"])
            if n != 1:
                raise ValueError(f"{f} 植入点命中 {n} 次：{p['old'][:80]!r}")
            new_text[f] = new_text[f].replace(p["old"], p["new"])
        for f, t in new_text.items():
            (root / f).write_text(t, encoding="utf-8")
    else:
        for f in recipe["files"]:
            backup[f] = (root / f).read_bytes()
        try:
            recipe["func"](root)
        except Exception:
            restore(root, backup)
            raise
    return backup


def restore(root: Path, backup: dict[str, bytes]) -> None:
    for f, b in backup.items():
        (root / f).write_bytes(b)


# ───────────────────────────── 主流程 ─────────────────────────────


def _plugins_for(recipe: dict | None) -> tuple[list[str], list[str]]:
    if recipe is None or recipe["kind"] != "plugin":
        return [], []
    return list(recipe["plugin"]), [recipe["pythonpath"]] if recipe.get("pythonpath") else []


def _group_sig(it: dict) -> tuple:
    if it["category"] == "A":
        return ("A",)
    p, x = _plugins_for(it["recipe"])
    return ("B", tuple(p), tuple(x))


def run_baselines(items: list[dict], copy0: Path, scratch: Path, log) -> None:
    """按运行环境分组跑一次并集基线；并集里有用例不过时，对涉及条目逐项单跑复核（排除用例间互扰）。"""
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        groups.setdefault(_group_sig(it), []).append(it)

    def one(sig, its):
        root = copy0 if sig[0] == "A" else REPO
        plugins, extra = (list(sig[1]), list(sig[2])) if sig[0] == "B" else ([], [])
        union = sorted({n for it in its for n in it["expect_fail"]})
        data = run_pytest(root, union, plugins, extra, {}, scratch)
        log(f"基线 组={sig[0]}{'/' + ','.join(sig[1]) if sig[0] == 'B' else ''} 用例={len(union)} "
            f"rc={data['rc']} 用时={data['seconds']}s")
        for it in its:
            ok, bad = baseline_ok(judge(data, it["expect_fail"]))
            if not ok:  # 单跑复核
                single = run_pytest(root, it["expect_fail"], plugins, extra, {}, scratch)
                ok, bad = baseline_ok(judge(single, it["expect_fail"]))
                if not ok:
                    it["baseline_tail"] = single["tail"][-1500:]
            it["baseline"] = "pass" if ok else "fail"
            it["baseline_problems"] = bad

    # A 组用 copy0，B 组在仓库本身；彼此独立可并发
    with cf.ThreadPoolExecutor(max_workers=max(1, len(groups))) as ex:
        list(ex.map(lambda kv: one(*kv), groups.items()))


def run_one(it: dict, copies: "queue.Queue[Path]", scratch: Path, log) -> None:
    recipe = it["recipe"]
    if it["category"] == "A":
        root = copies.get()
        try:
            before = {f: _sha(root / f) for f in _files_of(recipe)}
            try:
                backup = apply_a(root, recipe)
            except ValueError as exc:
                it.update(mutant="not_applied", caught=False, reason=str(exc))
                log(f"{it['key']} 植入点不唯一：{exc}")
                return
            try:
                data = run_pytest(root, it["expect_fail"], [], [], {}, scratch)
            finally:
                restore(root, backup)
            after = {f: _sha(root / f) for f in before}
            assert after == before, f"{it['key']} 副本还原失败"
        finally:
            copies.put(root)
    else:
        plugins, extra = _plugins_for(recipe)
        data = run_pytest(REPO, it["expect_fail"], plugins, extra, dict(recipe["env"]), scratch)
        st = data.get("status")
        if not st or not st.get("applied"):
            reason = (st or {}).get("reason") or "插件未写植入状态（插件或会话异常）"
            it.update(mutant="not_applied", caught=False, reason=f"进程内植入未生效：{reason}",
                      mutant_rc=data["rc"], mutant_tail=data["tail"][-1500:])
            log(f"{it['key']} 植入未生效：{reason}")
            return
    j = judge(data, it["expect_fail"])
    hit, excluded = caught_by(j)
    it.update(mutant="applied", caught=bool(hit), failed_tests=hit, excluded_failures=excluded,
              mutant_rc=data["rc"], mutant_seconds=data["seconds"],
              mutant_outcomes={w: [[n, o, e, t] for n, o, e, t in inst] for w, inst in j["per"].items()})
    if not hit:
        it["mutant_tail"] = data["tail"][-1500:]
    log(f"{it['key']} {'CAUGHT' if hit else 'SURVIVED'} 失败={len(hit)} 用时={data['seconds']}s rc={data['rc']}")


def _files_of(recipe: dict) -> list[str]:
    return sorted({p["file"] for p in recipe.get("patch", [])} | set(recipe.get("files", [])))


def counts(rows: list[dict]) -> dict:
    c = {"seeded": 0, "caught": 0, "survived": 0, "not_executable": 0, "not_applied": 0, "no_recipe": 0,
         "baseline_fail": 0}
    for r in rows:
        if r["category"] == "C":
            c["not_executable"] += 1
        elif r["category"] == "NR":
            c["no_recipe"] += 1
        elif r.get("baseline") == "fail":
            c["baseline_fail"] += 1
        elif r.get("mutant") == "not_applied":
            c["not_applied"] += 1
        elif r.get("mutant") == "applied":
            c["seeded"] += 1
            c["caught" if r["caught"] else "survived"] += 1
    return c


def summarize(rows: list[dict], repo_changed: bool) -> str:
    """判定行。not_executable 只放行用户裁决例外；not_applied、no_recipe、仓库被改动任一出现即 FAIL。"""
    c = counts(rows)
    ok = (c["survived"] == 0 and c["baseline_fail"] == 0 and c["not_applied"] == 0 and c["no_recipe"] == 0
          and not repo_changed and c["seeded"] > 0)
    return (f"TEST_MUTATION={'PASS' if ok else 'FAIL'} seeded={c['seeded']} caught={c['caught']} "
            f"survived={c['survived']} not_executable={c['not_executable']} not_applied={c['not_applied']} "
            f"no_recipe={c['no_recipe']} baseline_fail={c['baseline_fail']} repo_changed={int(repo_changed)}")


def _row_for_file(it: dict) -> dict:
    r = {k: v for k, v in it.items() if k != "recipe"}
    rc = it.get("recipe")
    if rc is not None:
        r["recipe"] = ({"kind": "transform", "func": rc["func"].__name__, "files": rc["files"]}
                       if rc["kind"] == "transform" else rc)
    return r


def table(rows: list[dict]) -> str:
    cols = ["seeded", "caught", "survived", "not_executable", "not_applied", "no_recipe", "baseline_fail"]
    lines = ["块 | " + " | ".join(cols)]
    for blk in sorted({r["block"] for r in rows}):
        c = counts([r for r in rows if r["block"] == blk])
        lines.append(f"{blk} | " + " | ".join(str(c[x]) for x in cols))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑这些块（可写前缀，如 pipeline/ 或 unit/）或键（如 pipeline/site:M20a）")
    ap.add_argument("--jobs", type=int, default=4, help="并发 pytest 进程数（A 类每个并发一份隔离副本）")
    ap.add_argument("--tag", default=None, help="批次名；给出时写 last_run.<tag>.jsonl，否则写 last_run.jsonl")
    ap.add_argument("--list", action="store_true", help="只打印归一结果，不运行")
    ap.add_argument("--merge", action="store_true",
                    help="把 last_run.<tag>*.jsonl（须给 --tag 作前缀）合并为 last_run.jsonl 并打印总判定行")
    args = ap.parse_args(argv)

    if args.merge:
        return merge(args.tag)

    items = load_items()
    keys = [it["key"] for it in items]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit(f"同一块内编号重复：{dup}")
    if args.only:
        # 块名可写前缀（如 pipeline/ 选全部流水线块）
        items = [it for it in items if it["key"] in args.only or any(it["block"].startswith(o) for o in args.only)]
    if args.list:
        for it in items:
            print(f"{it['category']} {it['key']} {it['reason'] or (it['recipe'] or {}).get('kind')}")
        return 0

    t_start = time.monotonic()
    log = lambda s: print(f"[{time.monotonic() - t_start:6.1f}s] {s}", flush=True)  # noqa: E731
    runnable = [it for it in items if it["category"] in ("A", "B")]
    work = Path(tempfile.mkdtemp(prefix="t14-mutation-"))
    scratch = work / "outcomes"
    scratch.mkdir()
    git_before = _git_status()
    try:
        n_copies = max(1, min(args.jobs, sum(1 for it in runnable if it["category"] == "A") or 1))
        copies: "queue.Queue[Path]" = queue.Queue()
        with cf.ThreadPoolExecutor(max_workers=n_copies) as ex:
            for c in ex.map(lambda i: make_copy(work / f"copy{i}"), range(n_copies)):
                copies.put(c)
        copy0 = copies.queue[0]
        log(f"隔离副本 {n_copies} 份就绪（{work}），可执行 {len(runnable)} 项、其余 {len(items) - len(runnable)} 项")
        run_baselines(runnable, copy0, scratch, log)
        todo = [it for it in runnable if it["baseline"] == "pass"]
        for it in runnable:
            if it["baseline"] != "pass":
                log(f"{it['key']} 基线未过：{it['baseline_problems'][:3]}")
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(lambda it: run_one(it, copies, scratch, log), todo))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    git_after = _git_status()
    repo_changed = git_before is None or git_after != git_before
    if repo_changed:
        print(f"运行前后仓库 git status 不同或取不到（判 FAIL）\n前：{git_before}\n后：{git_after}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"last_run.{args.tag}" if args.tag else "last_run"
    (OUT_DIR / f"{stem}.jsonl").write_text(
        "".join(json.dumps(_row_for_file(it), ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    (OUT_DIR / f"{stem}.meta.json").write_text(json.dumps(
        {"repo_changed": repo_changed, "git_before": git_before, "git_after": git_after,
         "only": args.only, "seconds": round(time.monotonic() - t_start, 1)}, ensure_ascii=False), encoding="utf-8")
    print(table(items))
    report_lines(items)
    line = summarize(items, repo_changed)
    print(f"记录：{OUT_DIR / (stem + '.jsonl')}（用时 {time.monotonic() - t_start:.1f}s）")
    print(line)
    return 0 if "=PASS" in line else 1


def report_lines(items: list[dict]) -> None:
    for it in items:
        if it["category"] == "C":
            print(f"NOT_EXECUTABLE {it['key']}：{it.get('reason')}")
        elif it["category"] == "NR":
            print(f"NO_RECIPE {it['key']}：{it.get('reason')}")
        elif it.get("baseline") == "fail":
            print(f"BASELINE_FAIL {it['key']}：{it['baseline_problems'][:3]}")
        elif it.get("mutant") == "not_applied":
            print(f"NOT_APPLIED {it['key']}：{it.get('reason')}")
        elif not it.get("caught"):
            print(f"SURVIVED {it['key']}：{it.get('excluded_failures')}")


def merge(tag: str | None) -> int:
    """只合并本批次前缀 last_run.<tag>*.jsonl；缺项、重复或任一批次仓库被改动都判 FAIL。"""
    if not tag:
        raise SystemExit("--merge 须给 --tag <批次前缀>")
    files = sorted(OUT_DIR.glob(f"last_run.{tag}*.jsonl"))
    if not files:
        raise SystemExit(f"没有 last_run.{tag}*.jsonl")
    rows, repo_changed = [], False
    for p in files:
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        meta = p.with_name(p.name[: -len(".jsonl")] + ".meta.json")
        try:
            repo_changed |= bool(json.loads(meta.read_text(encoding="utf-8"))["repo_changed"])
        except (OSError, json.JSONDecodeError, KeyError):
            repo_changed = True  # 缺元数据按未知处理
            print(f"缺批次元数据 {meta.name}，按仓库可能被改动处理")
    keys = [r["key"] for r in rows]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit(f"批次之间条目重复：{dup}")
    missing = sorted({it["key"] for it in load_items()} - set(keys))
    (OUT_DIR / "last_run.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                            encoding="utf-8")
    (OUT_DIR / "last_run.meta.json").write_text(json.dumps(
        {"repo_changed": repo_changed, "merged": [p.name for p in files]}, ensure_ascii=False), encoding="utf-8")
    print(f"合并：{[p.name for p in files]}")
    print(table(rows))
    report_lines(rows)
    line = summarize(rows, repo_changed)
    if missing:
        print(f"批次未覆盖的条目：{missing}")
        line = line.replace("TEST_MUTATION=PASS", "TEST_MUTATION=FAIL") + f" uncovered={len(missing)}"
    print(line)
    return 0 if "=PASS" in line else 1


def _git_status() -> str | None:
    """仓库 git status（排除只追加的子代理统计文件）；取不到返回 None。"""
    try:
        proc = subprocess.run(["git", "status", "--porcelain", "--ignore-submodules=dirty", "--", ".",
                               ":!docs/subagent-stats"], cwd=REPO, capture_output=True, text=True)
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


if __name__ == "__main__":
    sys.exit(main())
