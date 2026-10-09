"""Supplementary mutation recipes: for mutants.json entries that only have a text description, turn them into machine-checkable concrete steps here.

Keys are ``<block>:<id>``, block = directory of mutants.json relative to ``tests/robomme_ood/`` (e.g. ``pipeline/challenge``). Three forms:

- ``{"kind": "text", "patch": [{"file", "old", "new"}, ...]}``: class A, literal replacement in an isolated copy; each old must match exactly once;
- ``{"kind": "transform", "func": <callable>, "files": [...]}``: class A, rewrite data files in an isolated copy with a function (e.g. re-signing spec jsonl);
  ``files`` lists the relative paths that will be rewritten; the runner backs them up and restores them;
- In-process mutations (class B) are not in this table; see ``SUPPORTED`` in ``plugins/mut_inproc.py`` and ``tests/robomme_ood/unit/hard/mutants_plugin.py``.

Each recipe copies the semantics of the corresponding mutants.json ``method`` verbatim, only pinning the text down to exact replacement points; expect_fail of each block is unchanged.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _t(*patches: tuple[str, str, str]) -> dict:
    return {"kind": "text", "patch": [{"file": f, "old": o, "new": n} for f, o, n in patches]}


# ─────────────────────────────── Challenge interface block (tests/robomme_ood/pipeline/challenge) ───────────────────────────────
_P1 = "challenge_interface/scripts/phase1_eval.py"
CHALLENGE = {
    "pipeline/challenge:M19b": _t((_P1, '    return s == "success" or ("success" in s and "fail" not in s)\n',
                                   "    return True\n")),
    "pipeline/challenge:M19c": _t(("challenge_interface/msgpack_numpy.py",
                                   '            b"data": obj.tobytes(),\n            b"dtype": obj.dtype.str,',
                                   '            b"data": obj.tobytes(),\n            b"dtype": obj.dtype.newbyteorder("=").str,')),
    "pipeline/challenge:M19d": _t(("challenge_interface/server.py", "                    self._policy.reset()\n",
                                   "                    pass\n")),
    "pipeline/challenge:M19e": _t((_P1, "total_episodes = len(task_list) * args.num_episodes",
                                   "total_episodes = len(task_list)")),
    "pipeline/challenge:M19f": _t(("challenge_interface/client.py", "if isinstance(response, str):", "if False:")),
    "pipeline/challenge:M19g": _t((_P1, 'buffer["is_first_step"] = False', 'buffer["is_first_step"] = True')),
}

# ─────────────────────────────── Contract block (tests/robomme_ood/contract): spec jsonl and builder ───────────────────────────────
_SPECS = "src/robomme_ood/env_metadata/ood/xhard1/specs.jsonl"


def _load_hs(root: Path):
    """Load hard_specs by file from the isolated copy (re-sign with the copy's own signing function)."""
    spec = importlib.util.spec_from_file_location("_mut_hs", root / "src/robomme_ood/env_record_wrapper/hard_specs.py")
    hs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hs)
    return hs


def _resign_write(path: Path, hs, records: list[dict]) -> None:
    header, rows = records[0], records[1:]
    header["sampling_config_sha256"] = hs.digest(header["sampling_config"])
    header["identity_sha256"] = hs.identity_sha256(header, rows)
    header["delivery_sha256"] = hs.delivery_sha256(rows)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [header, *rows]), encoding="utf-8")


def contract_m04(root: Path) -> None:
    """Add 1 to the seed of the first selected xhard1 row; change only that row, no re-signing."""
    path = root / _SPECS
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(i for i, ln in enumerate(lines[1:], 1) if json.loads(ln)["selected"])
    row = json.loads(lines[i])
    row["seed"] += 1
    lines[i] = json.dumps(row, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def contract_m05(root: Path) -> None:
    """Add 1 to num_repeats of the first selected PickXtimes row; recompute spec_sha256 and the three header hashes (validate_specs still passes)."""
    hs = _load_hs(root)
    path = root / _SPECS
    records = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    row = next(r for r in records[1:] if r["task"] == "PickXtimes" and r["selected"])
    row["spec"]["objects"]["num_repeats"] += 1
    row["spec_sha256"] = hs.spec_sha256(row["spec"])
    _resign_write(path, hs, records)
    hs.validate_specs(records[0], records[1:])  # after mutation the signatures and validator agree, proving it is the value check, not the signature, that catches it


def contract_m06(root: Path) -> None:
    """Set candidate==1 rows to True and attempt of candidate==2 rows to float; recompute the three header hashes."""
    hs = _load_hs(root)
    path = root / _SPECS
    records = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    row = next(r for r in records[1:] if r["candidate"] == 1)
    row["candidate"] = True
    other = next(r for r in records[1:] if r["candidate"] == 2)
    other["attempt"] = float(other["attempt"])
    _resign_write(path, hs, records)


CONTRACT = {
    "contract:M04": {"kind": "transform", "func": contract_m04, "files": [_SPECS]},
    "contract:M05": {"kind": "transform", "func": contract_m05, "files": [_SPECS]},
    "contract:M06": {"kind": "transform", "func": contract_m06, "files": [_SPECS]},
    "contract:M09": _t(("src/robomme_ood/env_record_wrapper/hard_builder.py",
                        '            "native_episode_spec": entry["row"]["spec"],\n', "")),
}

# Recipes for the eval block (pipeline/eval) and gen/site blocks (pipeline/gen, pipeline/site) moved with their tests to the private eval repo RoboMME-benchmark-OOD-eval; not in this repo.
RECIPES: dict[str, dict] = {**CHALLENGE, **CONTRACT}
