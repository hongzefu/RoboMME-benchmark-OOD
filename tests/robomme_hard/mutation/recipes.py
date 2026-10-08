"""植入配方补充表：mutants.json 只给了文字描述的条目，在这里补成可机检的具体做法。

键为 ``<块>:<编号>``，块 = mutants.json 相对 ``tests/robomme_hard/`` 的目录（如 ``pipeline/challenge``）。三种形态：

- ``{"kind": "text", "patch": [{"file", "old", "new"}, ...]}``：A 类，在隔离副本里逐字替换，每个 old 必须恰好命中 1 次；
- ``{"kind": "transform", "func": <可调用>, "files": [...]}``：A 类，在隔离副本里按函数改写数据文件（如规格 jsonl 重签）；
  ``files`` 列出会被改写的相对路径，执行器据此备份与还原；
- 进程内植入（B 类）不在本表，见 ``plugins/mut_inproc.py`` 的 ``SUPPORTED`` 与 ``tests/robomme_hard/unit/hard/mutants_plugin.py``。

每条配方的语义照抄对应 mutants.json 的 ``method``，只把文字落成精确的替换点；不改各块的 expect_fail。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _t(*patches: tuple[str, str, str]) -> dict:
    return {"kind": "text", "patch": [{"file": f, "old": o, "new": n} for f, o, n in patches]}


# ─────────────────────────────── 挑战接口块（tests/robomme_hard/pipeline/challenge） ───────────────────────────────
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

# ─────────────────────────────── 契约块（tests/robomme_hard/contract）：规格 jsonl 与 builder ───────────────────────────────
_SPECS = "src/robomme_hard/env_metadata/ood/xhard1/specs.jsonl"


def _load_hs(root: Path):
    """从隔离副本里按文件加载 hard_specs（用副本自己的签名函数重签）。"""
    spec = importlib.util.spec_from_file_location("_mut_hs", root / "src/robomme_hard/env_record_wrapper/hard_specs.py")
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
    """xhard1 第一个 selected 行 seed 加 1，只改这一行、不重签。"""
    path = root / _SPECS
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(i for i, ln in enumerate(lines[1:], 1) if json.loads(ln)["selected"])
    row = json.loads(lines[i])
    row["seed"] += 1
    lines[i] = json.dumps(row, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def contract_m05(root: Path) -> None:
    """第一个 PickXtimes selected 行 num_repeats 加 1，重算 spec_sha256 与 header 三个散列（validate_specs 仍通过）。"""
    hs = _load_hs(root)
    path = root / _SPECS
    records = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    row = next(r for r in records[1:] if r["task"] == "PickXtimes" and r["selected"])
    row["spec"]["objects"]["num_repeats"] += 1
    row["spec_sha256"] = hs.spec_sha256(row["spec"])
    _resign_write(path, hs, records)
    hs.validate_specs(records[0], records[1:])  # 植入后签名与校验器自洽，证明挡住它的是取值核对而不是签名


def contract_m06(root: Path) -> None:
    """candidate==1 的行改 True、candidate==2 的行 attempt 改浮点，重算 header 三个散列。"""
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
    "contract:M09": _t(("src/robomme_hard/env_record_wrapper/hard_builder.py",
                        '            "native_episode_spec": entry["row"]["spec"],\n', "")),
}

# 评估块（pipeline/eval）与生成／站点块（pipeline/gen、pipeline/site）的配方随测试迁到私有评估仓 RoboMME-benchmark-OOD-eval，本仓不含。
RECIPES: dict[str, dict] = {**CHALLENGE, **CONTRACT}
