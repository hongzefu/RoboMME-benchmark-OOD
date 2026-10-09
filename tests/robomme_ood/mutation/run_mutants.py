"""Mutation runner: reads all ``tests/robomme_ood/**/mutants.json``, injects each semantic bug, and checks that the designated tests fail because of it.

Classification:
- A text replacement / data rewrite: modify source on disk in a tmp isolated copy (each old must match exactly once, else not_applied); the protected
  ``src/robomme`` and the three upstream entry scripts are never modified on disk (red line R9, recorded as no_recipe);
- B in-process plugins: ``tests/robomme_ood/unit/hard/mutants_plugin.py`` (T4_MUTANT, with ``plugins/t4_status.py``) or
  ``plugins/mut_inproc.py`` (MUT_INPROC), nothing written to disk, run in the repo itself (read-only); the plugin pre-checks the mutation point and writes whether it took effect to
  MUT_STATUS_FILE; if not in effect, recorded as not_applied;
- C user-decision exception: mutants.json entries with no expect_fail and a decision are counted as not_executable and allowed;
- NR missing recipe: no executable approach found (or no expect_fail and no decision), recorded as no_recipe.

Each item first confirms the original passes all expect_fail tests in the same environment (baseline), then reruns with the mutation: it counts as caught only if at least one expect_fail test (parametrized instances matched by prefix)
fails in the setup/call phase, the exception type is not an import/syntax error, the top stack frame is not inside the mutation tooling (``tests/robomme_ood/mutation/``,
``tests/robomme_ood/unit/hard/mutants_plugin.py``), and its file has no collection errors.

Verdict line: ``TEST_MUTATION=PASS|FAIL seeded= caught= survived= not_executable= not_applied= no_recipe= baseline_fail=
repo_changed=``; passes when survived, baseline_fail, not_applied, no_recipe are all 0, git status is unchanged before and after the run, and seeded>0.
Per-item records go to ``artifacts/maint-regress/mutation/last_run*.jsonl`` (a same-named ``.meta.json`` records whether the repo was modified).

Usage:
  UV_PROJECT_ENVIRONMENT=<main checkout>/.venv uv run --no-sync python tests/robomme_ood/mutation/run_mutants.py [--jobs 4]
      [--only <block prefix or block:id>...] [--tag <batch name>] [--list]
  ... run_mutants.py --merge --tag <batch prefix>   # merge only last_run.<prefix>*.jsonl, write last_run.jsonl and print the overall verdict line
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

from tests.robomme_ood.mutation.plugins import mut_inproc  # noqa: E402
from tests.robomme_ood.mutation.recipes import RECIPES  # noqa: E402

OUT_DIR = REPO / "artifacts" / "maint-regress" / "mutation"
#: The isolated copy only includes these (same copy scope as each block author's self-check).
COPY_ITEMS = ("src", "scripts", "tests", "challenge_interface", "pyproject.toml")
#: Protected: in-process mutation only, never written to disk (AGENTS.md P2 / plan red line R9).
PROTECTED_PREFIX = "src/robomme/"
UPSTREAM_ENTRIES = {"scripts/dataset_replay.py", "scripts/evaluation.py", "scripts/run_example.py"}
#: Failures caused by these exception types are not counted as caught.
NOT_SEMANTIC = {"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError"}
#: Failures whose top stack frame is in these mutation tooling files are errors of the plugin itself and are not counted as caught.
PLUGIN_FILES_PREFIX = ("tests/robomme_ood/mutation/",)
PLUGIN_FILES = {"tests/robomme_ood/unit/hard/mutants_plugin.py"}
PYTEST_TIMEOUT = 280
T4_PLUGIN_DIR = "tests/robomme_ood/unit/hard"


# ───────────────────────────── Load and normalize ─────────────────────────────


def load_items() -> list[dict]:
    items = []
    # Block names are relative to tests/robomme_ood (consistent with recipes.py keys and plugins/mut_inproc.py block names); other official tests/ directories are not mutated.
    for path in sorted((REPO / "tests" / "robomme_ood").rglob("mutants.json")):
        rel = path.relative_to(REPO).as_posix()
        block = path.parent.relative_to(REPO / "tests" / "robomme_ood").as_posix()
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
    # Category: A/B executable; C only for the "no expect_fail and has user decision" exception (allowed); NR is missing recipe (FAIL)
    it = {"source": source, "block": block, "id": mid, "key": key, "target": e.get("target"), "expect_fail": expect,
          "category": "NR", "reason": None, "recipe": None}
    if not expect:
        if e.get("decision"):
            it["category"] = "C"
            it["reason"] = f"no expect_fail tests (user-decision exception: {e['decision']})"
        else:
            it["reason"] = "no expect_fail tests and no user decision"
        return it
    method = e.get("method") or e.get("inject") or ""
    if e.get("patch"):
        recipe = {"kind": "text", "patch": e["patch"]}
    elif key in RECIPES:
        recipe = RECIPES[key]
    elif "T4_MUTANT=" in method and "mutants_plugin" in method:
        name = re.search(r"T4_MUTANT=([A-Za-z0-9_]+)", method).group(1)
        # t4_status changes the replacement to "exactly one match" and writes mutation status before mutants_plugin
        recipe = {"kind": "plugin", "plugin": ["mutants_plugin", "tests.robomme_ood.mutation.plugins.t4_status"],
                  "pythonpath": T4_PLUGIN_DIR, "env": {"T4_MUTANT": name}}
    elif key in mut_inproc.SUPPORTED:
        recipe = {"kind": "plugin", "plugin": ["tests.robomme_ood.mutation.plugins.mut_inproc"], "pythonpath": None,
                  "env": {"MUT_INPROC": key}}
    else:
        it["reason"] = "text description only, runner has no matching recipe"
        return it
    if recipe["kind"] in ("text", "transform"):
        files = [p["file"] for p in recipe.get("patch", [])] + list(recipe.get("files", []))
        bad = [f for f in files if _is_protected(f)]
        if bad:
            it["reason"] = f"R9: protected files must not be modified on disk {bad}"
            return it
        it["category"] = "A"
    else:
        it["category"] = "B"
    it["recipe"] = recipe
    return it


# ───────────────────────────── Run pytest ─────────────────────────────


def _venv() -> str:
    v = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if v:
        return v
    if (REPO / ".venv").is_dir():
        return str(REPO / ".venv")
    raise SystemExit("missing UV_PROJECT_ENVIRONMENT (worktree has no .venv; must point to the main checkout .venv)")


def run_pytest(root: Path, nodeids: list[str], plugins: list[str], extra_path: list[str], env_extra: dict,
               scratch: Path) -> dict:
    """Run the given tests under root and return the JSON written by the outcome recorder plugin (plus rc and output tail)."""
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
           "--tb=line", "-p", "tests.robomme_ood.mutation.plugins.outcome_recorder"]
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
    try:  # mutation status written by the in-process plugin; None if the file is absent
        status = json.loads(Path(status_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = None
    Path(status_file).unlink(missing_ok=True)
    data.update(rc=rc, tail=tail, seconds=round(time.monotonic() - t0, 1), status=status)
    return data


def _final(rec: dict) -> tuple[str, str | None, str | None]:
    """Combine a test's phases into a final outcome: (passed|failed|skipped, exception type of the failing phase, top stack frame file)."""
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
    """Summarize per expect_fail: each expectation -> list of instance outcomes."""
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
        bad.append(f"collection errors {[c['nodeid'] for c in j['collect_errors']]}")
    for want, inst in j["per"].items():
        if not inst:
            bad.append(f"{want} not collected")
        bad += [f"{nid} {out}" for nid, out, _, _ in inst if out != "passed"]
    return not bad, bad


def caught_by(j: dict) -> tuple[list[str], list[str]]:
    """Return (test instances with semantic failures, descriptions of excluded failures)."""
    if j["collect_errors"]:
        return [], [f"collection errors {[c['nodeid'] for c in j['collect_errors']]}"]
    hit, excluded = [], []
    for inst in j["per"].values():
        for nid, out, exc, top in inst:
            if out != "failed":
                continue
            if exc in NOT_SEMANTIC:
                excluded.append(f"{nid} {exc}")
            elif top is None:
                excluded.append(f"{nid} {exc} top frame file unknown")
            elif top.startswith(PLUGIN_FILES_PREFIX) or top in PLUGIN_FILES:
                excluded.append(f"{nid} {exc} top frame in mutation tooling {top}")
            else:
                hit.append(nid)
    return hit, excluded


# ───────────────────────────── Isolated copy ─────────────────────────────


def make_copy(dest: Path) -> Path:
    ign = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    for name in COPY_ITEMS:
        src = REPO / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=ign, symlinks=True)
        elif src.exists():
            shutil.copy2(src, dest / name)
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=_venv(), PYTHONPATH=f"{dest}/src:{dest}", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run(["uv", "run", "--no-sync", "python", "-c", "import robomme_ood, robomme; "
                          "print(robomme_ood.__file__); print(robomme.__file__)"],
                         cwd=dest, env=env, capture_output=True, text=True, check=True).stdout.split()
    if not all(p.startswith(f"{dest}/src/") for p in out):
        raise SystemExit(f"isolated copy imports point to the wrong place: {out}")
    return dest


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def apply_a(root: Path, recipe: dict) -> dict[str, bytes]:
    """Apply mutation in the copy, return {relative path: original bytes} for restore; raise ValueError if old does not match exactly once (no half-modified state)."""
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
                raise ValueError(f"{f} mutation point matched {n} times: {p['old'][:80]!r}")
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


# ───────────────────────────── Main flow ─────────────────────────────


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
    """Run one union baseline per runtime-environment group; if any test in the union fails, rerun the affected items individually (to rule out cross-test interference)."""
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        groups.setdefault(_group_sig(it), []).append(it)

    def one(sig, its):
        root = copy0 if sig[0] == "A" else REPO
        plugins, extra = (list(sig[1]), list(sig[2])) if sig[0] == "B" else ([], [])
        union = sorted({n for it in its for n in it["expect_fail"]})
        data = run_pytest(root, union, plugins, extra, {}, scratch)
        log(f"baseline group={sig[0]}{'/' + ','.join(sig[1]) if sig[0] == 'B' else ''} tests={len(union)} "
            f"rc={data['rc']} time={data['seconds']}s")
        for it in its:
            ok, bad = baseline_ok(judge(data, it["expect_fail"]))
            if not ok:  # individual rerun
                single = run_pytest(root, it["expect_fail"], plugins, extra, {}, scratch)
                ok, bad = baseline_ok(judge(single, it["expect_fail"]))
                if not ok:
                    it["baseline_tail"] = single["tail"][-1500:]
            it["baseline"] = "pass" if ok else "fail"
            it["baseline_problems"] = bad

    # Group A uses copy0, group B runs in the repo itself; independent of each other and can run concurrently
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
                log(f"{it['key']} mutation point not unique: {exc}")
                return
            try:
                data = run_pytest(root, it["expect_fail"], [], [], {}, scratch)
            finally:
                restore(root, backup)
            after = {f: _sha(root / f) for f in before}
            assert after == before, f"{it['key']} copy restore failed"
        finally:
            copies.put(root)
    else:
        plugins, extra = _plugins_for(recipe)
        data = run_pytest(REPO, it["expect_fail"], plugins, extra, dict(recipe["env"]), scratch)
        st = data.get("status")
        if not st or not st.get("applied"):
            reason = (st or {}).get("reason") or "plugin did not write mutation status (plugin or session error)"
            it.update(mutant="not_applied", caught=False, reason=f"in-process mutation not in effect: {reason}",
                      mutant_rc=data["rc"], mutant_tail=data["tail"][-1500:])
            log(f"{it['key']} mutation not in effect: {reason}")
            return
    j = judge(data, it["expect_fail"])
    hit, excluded = caught_by(j)
    it.update(mutant="applied", caught=bool(hit), failed_tests=hit, excluded_failures=excluded,
              mutant_rc=data["rc"], mutant_seconds=data["seconds"],
              mutant_outcomes={w: [[n, o, e, t] for n, o, e, t in inst] for w, inst in j["per"].items()})
    if not hit:
        it["mutant_tail"] = data["tail"][-1500:]
    log(f"{it['key']} {'CAUGHT' if hit else 'SURVIVED'} failures={len(hit)} time={data['seconds']}s rc={data['rc']}")


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
    """Verdict line. not_executable only allows user-decision exceptions; any not_applied, no_recipe, or repo modification means FAIL."""
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
    lines = ["block | " + " | ".join(cols)]
    for blk in sorted({r["block"] for r in rows}):
        c = counts([r for r in rows if r["block"] == blk])
        lines.append(f"{blk} | " + " | ".join(str(c[x]) for x in cols))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these blocks (prefix allowed, e.g. pipeline/ or unit/) or keys (e.g. pipeline/site:M20a)")
    ap.add_argument("--jobs", type=int, default=4, help="number of concurrent pytest processes (one isolated copy per concurrent class A job)")
    ap.add_argument("--tag", default=None, help="batch name; if given write last_run.<tag>.jsonl, else last_run.jsonl")
    ap.add_argument("--list", action="store_true", help="only print the normalized result, do not run")
    ap.add_argument("--merge", action="store_true",
                    help="merge last_run.<tag>*.jsonl (--tag required as prefix) into last_run.jsonl and print the overall verdict line")
    args = ap.parse_args(argv)

    if args.merge:
        return merge(args.tag)

    items = load_items()
    keys = [it["key"] for it in items]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit(f"duplicate ids within a block: {dup}")
    if args.only:
        # Block names may be prefixes (e.g. pipeline/ selects all pipeline blocks)
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
        log(f"isolated copies: {n_copies} ready ({work}), {len(runnable)} executable, {len(items) - len(runnable)} others")
        run_baselines(runnable, copy0, scratch, log)
        todo = [it for it in runnable if it["baseline"] == "pass"]
        for it in runnable:
            if it["baseline"] != "pass":
                log(f"{it['key']} baseline failed: {it['baseline_problems'][:3]}")
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(lambda it: run_one(it, copies, scratch, log), todo))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    git_after = _git_status()
    repo_changed = git_before is None or git_after != git_before
    if repo_changed:
        print(f"repo git status differs before/after the run or unavailable (FAIL)\nbefore: {git_before}\nafter: {git_after}")

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
    print(f"records: {OUT_DIR / (stem + '.jsonl')} (time {time.monotonic() - t_start:.1f}s)")
    print(line)
    return 0 if "=PASS" in line else 1


def report_lines(items: list[dict]) -> None:
    for it in items:
        if it["category"] == "C":
            print(f"NOT_EXECUTABLE {it['key']}: {it.get('reason')}")
        elif it["category"] == "NR":
            print(f"NO_RECIPE {it['key']}: {it.get('reason')}")
        elif it.get("baseline") == "fail":
            print(f"BASELINE_FAIL {it['key']}: {it['baseline_problems'][:3]}")
        elif it.get("mutant") == "not_applied":
            print(f"NOT_APPLIED {it['key']}: {it.get('reason')}")
        elif not it.get("caught"):
            print(f"SURVIVED {it['key']}: {it.get('excluded_failures')}")


def merge(tag: str | None) -> int:
    """Merge only this batch's last_run.<tag>*.jsonl; missing items, duplicates, or any batch with repo modification means FAIL."""
    if not tag:
        raise SystemExit("--merge requires --tag <batch prefix>")
    files = sorted(OUT_DIR.glob(f"last_run.{tag}*.jsonl"))
    if not files:
        raise SystemExit(f"no last_run.{tag}*.jsonl")
    rows, repo_changed = [], False
    for p in files:
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        meta = p.with_name(p.name[: -len(".jsonl")] + ".meta.json")
        try:
            repo_changed |= bool(json.loads(meta.read_text(encoding="utf-8"))["repo_changed"])
        except (OSError, json.JSONDecodeError, KeyError):
            repo_changed = True  # missing metadata is treated as unknown
            print(f"missing batch metadata {meta.name}, treating repo as possibly modified")
    keys = [r["key"] for r in rows]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit(f"duplicate items across batches: {dup}")
    missing = sorted({it["key"] for it in load_items()} - set(keys))
    (OUT_DIR / "last_run.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                            encoding="utf-8")
    (OUT_DIR / "last_run.meta.json").write_text(json.dumps(
        {"repo_changed": repo_changed, "merged": [p.name for p in files]}, ensure_ascii=False), encoding="utf-8")
    print(f"merging: {[p.name for p in files]}")
    print(table(rows))
    report_lines(rows)
    line = summarize(rows, repo_changed)
    if missing:
        print(f"items not covered by any batch: {missing}")
        line = line.replace("TEST_MUTATION=PASS", "TEST_MUTATION=FAIL") + f" uncovered={len(missing)}"
    print(line)
    return 0 if "=PASS" in line else 1


def _git_status() -> str | None:
    """Repo git status (excluding the append-only subagent stats file); None if unavailable."""
    try:
        proc = subprocess.run(["git", "status", "--porcelain", "--ignore-submodules=dirty", "--", ".",
                               ":!docs/subagent-stats"], cwd=REPO, capture_output=True, text=True)
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


if __name__ == "__main__":
    sys.exit(main())
