"""ood 四档规格（``env_metadata/ood/xhardN/specs.jsonl``）的读取、封套校验与回注绑定摘要。

由 ``scripts/parity/v4_specs.py`` 下沉而来（0927 计划第二部分 §1.2）；抽签与冻结留在
``scripts/injection-dev/``，本模块只放评估侧与生成侧都要用的纯函数，不导入仿真。

jsonl 一行分「签」与「结果」两段（0927 计划第一部分 §5.3；现行唯一格式 ``schema="hard-specs/4"``）：

* 签：``task tier candidate episode seed attempt spec spec_sha256 layout_parent``——第一阶段封存，``identity_sha256`` 覆盖；
* 结果：``selected tried initial_selected rollout``——第二阶段回写，不进 ``identity_sha256``。

``delivery_sha256`` 另盖「哪几局是正式交付」：排序后的
``(task, tier, candidate, seed, spec_sha256, rollout.h5_sha256)``，只取 ``selected`` 且 ``rollout.status=="ok"`` 的行。

xhard0（官方 hard 12 局）是否前置在 ood 里由 ``XHARD0_IN_TEST_HARD`` 决定：V9 定稿默认关（每任务 50 局、
16 任务 800 局），环境变量 ``ROBOMME_HARD_XHARD0_IN_TEST_HARD=1`` 可恢复 12 + 50。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

#: 现行唯一规格格式（v8 方案第二部分 §2.2 第 2 条，R6）：header 带 ``layout_rule``、``exec_cap``，
#: ``delivery_per_cell``、``select_rule``、``per_env`` 为逐任务字典；行带 ``layout_parent``（恒为 null）。
#: 旧格式 /2（V5～V6 单档）与 /3（V7 母布局派生）的读写校验已于维护计划阶段 1b（W4）删除。
SCHEMA = "hard-specs/4"
#: 新值档（不含 xhard0）。v8 阶段 3b 换包起为五档 xhard1～xhard5（原 ``V8_TIERS`` 同值，已并入本常量）。
#: EXPECTED_CELLS、packaged_specs_path、builder、/4 校验与规格根读取等依赖它。
TIERS = ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5")
#: builder 档序：xhard0（官方 test 的 hard 子集，原生分支、不回注）在最前，其后新值五档（共六项）。
XHARD0 = "xhard0"
BUILDER_TIERS = (XHARD0, *TIERS)
#: xhard0 每任务 12 局＝官方 test 元数据 difficulty=="hard" 的原 episode 3,7,…,47（只作核对值，筛选按 difficulty）
XHARD0_PER_TASK = 12
#: xhard0（官方 hard 12 局）是否前置在 ood 里。V9 定稿默认关（每任务恰 50 局、16 任务 800 局）；
#: 设环境变量 ROBOMME_HARD_XHARD0_IN_TEST_HARD=1 可恢复为 12 + 50（xhard0 源码、常量与清单全部保留）。
XHARD0_IN_TEST_HARD: bool = os.environ.get("ROBOMME_HARD_XHARD0_IN_TEST_HARD", "0") == "1"


def xhard0_prefix() -> int:
    """ood 里排在新值档前面的 xhard0 局数：开关开为 XHARD0_PER_TASK，关为 0。"""
    return XHARD0_PER_TASK if XHARD0_IN_TEST_HARD else 0


XHARD0_EPISODES = tuple(range(3, 48, 4))
#: 历史 V4/V5 单档名。seed 规则 v5 已删除；``scripts/parity/train_split_runner.py`` 在 jobs 不带 seed_rule 时
#: 仍以它为期望难度（V9／xhard0 生成的 jobs 均带 seed_rule 或走官方元数据分支，不经该缺省），故保留常量。
DIFFICULTY = "xhard"
#: 回注绑定：只记录不回注的观测值（SpecRecorder.record）允许的浮点差（用户 U-13 方案甲，红线 R22，不做参数）。
RECORDED_FLOAT_TOL = 1e-5

# runtime 四项与 gym.make 参数逐字相同（builder 起环境时逐字比对，render_mode 放行）
RUNTIME = {
    "obs_mode": "rgb+depth+segmentation",
    "control_mode": "pd_joint_pos",
    "render_mode": "rgb_array",
    "reward_mode": "dense",
}
# seed = offset + env_code × env_block + episode × 100 + attempt；env_code 是任务在 16 任务规范序里的 1-indexed 位置
SEED_RULE = {"offset": 4_000_000, "env_block": 100_000, "episode_stride": 100,
             "formula": "offset + env_code*env_block + episode*100 + attempt"}
#: V8 按档 seed 偏移（v8 方案第二部分 §2.2 第 3 条，R9）：各档布局独立抽，档与档 seed 两两不交。
#: 每档最大增量 16×100000 + 999×100 + 99 < 1.7e6，小于步长 2e6；最低 16e6 不碰 V5（4e6 起）、
#: 旧 V6 段（6e6～13.7e6）与 V7／探针段（14e6～15.7e6）。
V8_SEED_OFFSETS = {"xhard1": 16_000_000, "xhard2": 18_000_000, "xhard3": 20_000_000,
                   "xhard4": 22_000_000, "xhard5": 24_000_000}
#: 按档偏移的规则族：profile → {tier: offset}。现行只有 v8（V9 沿用）；v5、v6、v7 规则族已删除。
#: 按档偏移的规则族只认本族登记的档位；新登记的偏移须与历史段（4e6、6e6～15.7e6）及已登记的族互不重叠。
TIER_SEED_OFFSETS: dict[str, dict[str, int]] = {"v8": V8_SEED_OFFSETS}
SEED_PROFILES = tuple(TIER_SEED_OFFSETS)
MAX_ATTEMPTS = 100
#: 16 任务规范序（与 scripts/injection-dev/seed_layout.py::ALL_TASKS 逐字相同；src 不反向依赖 scripts）
ALL_TASKS = (
    "PickXtimes", "StopCube", "SwingXtimes", "BinFill", "VideoUnmaskSwap", "VideoUnmask",
    "ButtonUnmaskSwap", "ButtonUnmask", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "PickHighlight", "InsertPeg", "MoveCube", "PatternLock", "RouteStick",
)
#: 只在 xhard4 交付的任务（v8 阶段 3b 起：StopCube 拆成五档定值后离开，只剩 InsertPeg、MoveCube）
XHARD4_ONLY = ("InsertPeg", "MoveCube")

# ── 执行步上限（v8 方案阶段 2 新增，V9 沿用；维护计划 R 块去掉名字里的 V8 前缀）──
#: 抽样与交付的执行步上限：执行步（不含演示帧）> 1600 的候选记 exec_over_cap 并递补；/4 header ``exec_cap`` 必须等于它
EXEC_CAP = 1600


def _v9_cells() -> dict[tuple[str, str], int]:
    """v9 方案第一部分表 2 的 43 个新值格（档集合与 V8 相同）：每任务 50 局，在该任务 V8 交付的档里平分，
    分不均时前面的档多 1 局（17／17／16、13／13／12／12），不含 xhard0。"""
    cells: dict[tuple[str, str], int] = {}
    for task in ("PickXtimes", "RouteStick", "PatternLock"):
        for tier, n in zip(("xhard1", "xhard2", "xhard3"), (17, 17, 16)):
            cells[(task, tier)] = n
    for task in ("SwingXtimes", "StopCube"):
        for tier in ("xhard1", "xhard2", "xhard3", "xhard4", "xhard5"):
            cells[(task, tier)] = 10
    for task in ("VideoUnmask", "ButtonUnmask"):
        for tier, n in zip(("xhard1", "xhard2", "xhard3", "xhard4"), (13, 13, 12, 12)):
            cells[(task, tier)] = n
    for task in ("BinFill", "VideoUnmaskSwap", "ButtonUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
                 "PickHighlight", "VideoRepick"):
        for tier in ("xhard1", "xhard2"):
            cells[(task, tier)] = 25
    for task in ("MoveCube", "InsertPeg"):
        cells[(task, "xhard4")] = 50
    return cells


#: v9 交付格表（v9 方案第一部分表 2）：43 格、合计 800、每任务 50（格集合与已删除的 V8 1070 局表相同，只有局数不同）
V9_CELLS: dict[tuple[str, str], int] = _v9_cells()
V9_PER_TASK = 50
assert len(V9_CELLS) == 43 and sum(V9_CELLS.values()) == 800, "V9_CELLS 须为表 2 的 43 格、合计 800"
assert all(task in ALL_TASKS and tier in TIERS for task, tier in V9_CELLS), "V9_CELLS 含未知任务或档位"
assert all(sum(n for (t, _), n in V9_CELLS.items() if t == task) == V9_PER_TASK for task in ALL_TASKS), \
    "V9_CELLS 每任务须恰为 50 局"
#: 交付格表 {(task, tier): 正式交付局数}（builder 按它断言每格行数，表外格恰好 0 行、表内格恰好等于表值）。
#: v9 阶段 3b 换包（env_metadata/ood/ 换为 V9 规格）与本行切到 V9_CELLS 在同一提交完成（v9 方案 R7），表与包始终一致。
EXPECTED_CELLS: dict[tuple[str, str], int] = V9_CELLS
#: 已登记的完整交付格表（按版本）。``resolve_cell_table`` 按顺序 EXPECTED_CELLS → 本表各项找第一张能覆盖
#: 给定子表的表，作为单文件配额上限（``_validate_specs``）与 ``load_specs_root`` 的格配额上限。
#: V8 的 1070 局表已于维护计划阶段 1b（W4）删除，只剩 v9。
CELL_TABLES: dict[str, dict[tuple[str, str], int]] = {"v9": V9_CELLS}


def xhard4_only_tasks(cells: dict[tuple[str, str], int]) -> set[str]:
    """格表里只在 xhard4 出现的任务集合（XHARD4_ONLY 对参数格表的核对口径）。"""
    return {task for task in ALL_TASKS if {t for name, t in cells if name == task} == {"xhard4"}}


assert all(xhard4_only_tasks(table) == set(XHARD4_ONLY) for table in (EXPECTED_CELLS, *CELL_TABLES.values())), \
    "XHARD4_ONLY 须恰为交付格表（EXPECTED_CELLS、V9_CELLS 各自）里只在 xhard4 出现的任务"


def _fits(cells: dict[tuple[str, str], int], table: dict[tuple[str, str], int]) -> bool:
    return all(key in table and _is_int(n) and n <= table[key] for key, n in cells.items())


def resolve_cell_table(cells: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    """给定（子）格表 → 用作配额上限的完整交付格表：按 ``EXPECTED_CELLS``、``CELL_TABLES`` 各项的顺序取第一张
    「含全部格且逐格局数 ≤ 表值」的表；都不覆盖时返回 ``EXPECTED_CELLS``（由调用方的逐项核对报出具体哪格超）。
    显式传入格表时不经本函数。"""
    for table in (EXPECTED_CELLS, *CELL_TABLES.values()):
        if _fits(cells, table):
            return table
    return EXPECTED_CELLS


def header_cell_table(header: dict[str, Any]) -> dict[tuple[str, str], int] | None:
    """/4 header 自带的逐任务配额 → ``resolve_cell_table`` 取配额上限格表（供不知道格表的单文件读取方用，如
    ``_rollout`` 回写复核）；非 /4 或配额形态不对返回 None（交给 ``validate_specs`` 报具体错）。"""
    if not isinstance(header, dict) or header.get("schema") != SCHEMA:
        return None
    quota = header.get("delivery_per_cell")
    if not isinstance(quota, dict):
        return None
    return resolve_cell_table({(task, header.get("difficulty")): n for task, n in quota.items()})

# 签：进 identity_sha256 的 header 键与行键
IDENTITY_HEADER_KEYS = ("schema", "difficulty", "tasks", "per_env", "runtime", "seed_rule", "select_rule",
                        "sampling_config_sha256", "recovery_rule", "identity_source")
IDENTITY_ROW_KEYS = ("task", "tier", "candidate", "episode", "seed", "attempt", "spec_sha256")
HEADER_REQUIRED = {"record", *IDENTITY_HEADER_KEYS, "sampling_config", "run_id", "draw_stats", "provenance",
                   "delivery_per_cell", "identity_sha256", "delivery_sha256"}
#: 迁移来源专有键（从旧快照迁移时写；重新抽签生成的文件可以没有）
HEADER_OPTIONAL = {"drafts_sha256", "legacy_identity_sha256", "source_files", "eval_identities_sha256",
                   "dedup_dropped", "demo_frames_out_of_band", "notes"}
ROW_KEYS = {"record", *IDENTITY_ROW_KEYS, "spec", "selected", "tried", "initial_selected", "rollout"}
ROLLOUT_STATUSES = ("ok", "failed")
#: 身份键按 schema 分表（键表一经发布即冻结，已封存文件的 identity_sha256 逐字不变）；
#: /4 = 基础键 + header ``layout_rule``、``exec_cap``、``delivery_per_cell`` + 行 ``layout_parent``
#: （逐任务配额与执行步上限进签，改了不重签必失败）。/2、/3 的键表已随旧格式删除。
IDENTITY_KEYS_BY_SCHEMA = {
    SCHEMA: (IDENTITY_HEADER_KEYS + ("layout_rule", "exec_cap", "delivery_per_cell"),
                IDENTITY_ROW_KEYS + ("layout_parent",)),
}
#: /4 唯一合法的布局规则：各档布局独立抽，不派生
LAYOUT_RULE = {"mode": "independent"}


def _schema_keys(schema: str) -> tuple[tuple[str, ...], tuple[str, ...], set[str], set[str]]:
    if schema not in IDENTITY_KEYS_BY_SCHEMA:
        raise SpecsError(f"specs 版本不符：{schema}")
    header_keys, row_keys = IDENTITY_KEYS_BY_SCHEMA[schema]
    header_required = (HEADER_REQUIRED - set(IDENTITY_HEADER_KEYS)) | set(header_keys)
    row_required = (ROW_KEYS - set(IDENTITY_ROW_KEYS)) | set(row_keys)
    return header_keys, row_keys, header_required, row_required


class SpecsError(ValueError):
    """规格文件缺失、被篡改、来源不符或字段集合不符。"""


# ── 基础函数 ─────────────────────────────────────────────────────────────


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def spec_sha256(spec: dict[str, Any]) -> str:
    return digest(spec)


def env_code(task: str) -> int:
    if task not in ALL_TASKS:
        raise SpecsError(f"未知环境名：{task}")
    return ALL_TASKS.index(task) + 1


def seed_rule_for(difficulty: str, profile: str) -> dict[str, Any]:
    """按档位与规则族给出 seed 规则：只认 ``TIER_SEED_OFFSETS`` 登记的规则族（现行 v8），按档偏移、只用本族登记的
    档位校验。v5（历史 xhard 单档）、v6、v7（四档同 offset）规则族已删除。"""
    if profile not in TIER_SEED_OFFSETS:
        known = "／".join(TIER_SEED_OFFSETS)
        raise SpecsError(f"{difficulty} 只支持 seed 规则 {known}（收到 {profile!r}）")
    offsets = TIER_SEED_OFFSETS[profile]
    if difficulty not in offsets:
        raise SpecsError(f"seed 规则 {profile} 未登记档位 {difficulty!r}，只支持 {tuple(offsets)}")
    return {**SEED_RULE, "offset": offsets[difficulty]}


def _known_seed_rule(difficulty: str, rule: dict[str, Any]) -> bool:
    for profile in SEED_PROFILES:
        try:
            if rule == seed_rule_for(difficulty, profile):
                return True
        except SpecsError:
            continue
    return False


def seed_for(task: str, episode: int, attempt: int, rule: dict[str, Any] | None = None) -> int:
    rule = SEED_RULE if rule is None else rule
    if not 0 <= int(attempt) < MAX_ATTEMPTS:
        raise SpecsError(f"attempt 必须落在 [0, {MAX_ATTEMPTS})，当前为 {attempt}")
    return int(rule["offset"]) + env_code(task) * int(rule["env_block"]) + int(episode) * 100 + int(attempt)


def identity_sha256(header: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """只盖签：header 规格键 + 每行签键；provenance、draw_stats、run_id 与结果段都不进。键集合按 header 的 schema 取。"""
    header_keys, row_keys, _, _ = _schema_keys(header.get("schema"))
    ordered = sorted(rows, key=lambda r: (r["task"], int(r["candidate"])))
    return digest({
        "header": {key: header[key] for key in header_keys},
        "rows": [{key: row[key] for key in row_keys} for row in ordered],
    })


def delivered(row: dict[str, Any]) -> bool:
    rollout = row.get("rollout") or {}
    return bool(row.get("selected")) and rollout.get("status") == "ok"


def delivery_sha256(rows: list[dict[str, Any]]) -> str:
    items = sorted(
        (row["task"], row["tier"], int(row["candidate"]), int(row["seed"]), row["spec_sha256"],
         (row.get("rollout") or {}).get("h5_sha256"))
        for row in rows if delivered(row)
    )
    return digest([list(item) for item in items])


# ── 源码指纹（只进 provenance，不进 identity；不符只警告）───────────────────


def _package_root(name: str) -> Path:
    import importlib.util

    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SpecsError(f"找不到包 {name}")
    return Path(list(spec.submodule_search_locations)[0])


def hard_fingerprint() -> str:
    """``robomme_hard`` 全部 .py 的相对路径与 sha256 的总散列。"""
    root = Path(__file__).resolve().parents[1]
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}
    return digest(files)


def base_fingerprint() -> str:
    """借用的官方 shim 目标文件的总散列（按 UPSTREAM.json 清单读当前安装的 robomme）。"""
    manifest = json.loads((Path(__file__).resolve().parents[1] / "UPSTREAM.json").read_text(encoding="utf-8"))
    robomme_root = _package_root("robomme")
    files = {}
    for entry in manifest["shims"]:
        rel = entry["target_file"][len("src/robomme/"):]
        path = robomme_root / rel
        files[entry["target_file"]] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return digest(files)


# ── 读写 ───────────────────────────────────────────────────────────────


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SpecsError(f"JSON 存在重复字段：{key}")
        result[key] = value
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line, object_pairs_hook=_unique_object) for line in stream if line.strip()]
    if not records:
        raise SpecsError(f"{path} 为空")
    return records


def _exact_keys(value: dict[str, Any], required: set[str], label: str, optional: set[str] = frozenset()) -> None:
    missing, extra = required - value.keys(), value.keys() - required - optional
    if missing or extra:
        raise SpecsError(f"{label} 字段集合不符：缺少 {sorted(missing)}，多出 {sorted(extra)}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_specs(header: dict[str, Any], rows: list[dict[str, Any]],
                       expected_cells: dict[tuple[str, str], int] | None = None) -> None:
    """``hard-specs/4`` 单文件校验（v8 方案第二部分 §2.2 第 2 条）。``expected_cells`` 是作配额上限的完整交付格表，
    缺省 ``EXPECTED_CELLS``（v9 方案 §2.1：格表参数贯通 load_specs → validate_specs → 本函数）：

    * header：字段集合；``difficulty ∈ TIERS``；runtime；``seed_rule == seed_rule_for(tier, "v8")``；
      ``exec_cap == EXEC_CAP``；``layout_rule == {"mode": "independent"}``；``tasks`` 不重复且每个 (task, tier)
      都在格表里；``select_rule`` 为 ``{task: [不重复非负整数]}``、``per_env`` 为 ``{task: 候选数（非负整数）}``、
      ``delivery_per_cell`` 为 ``{task: 正整数}``，三者键集合都等于 ``tasks``；逐任务配额自洽：
      ``delivery_per_cell[task] ≤ 格表[(task, tier)]``、``len(select_rule[task]) == delivery_per_cell[task]``、
      ``per_env[task] ==`` 本文件该任务行数、``select_rule[task]`` 每个索引 ``< per_env[task]``；内嵌 sampling_config 散列自洽；
    * 行：字段集合；``candidate``／``attempt``／``seed`` 为整数且不是布尔（F-6）；``layout_parent is None``、
      ``spec.spec_kind == "native-newvalue/2"``；``candidate == episode`` 且
      ``0 ≤ episode < env_block // episode_stride``；档位、规格散列、seed 公式、布尔位、rollout.status；
      逐任务 selected 行数 ≤ ``delivery_per_cell[task]``；
    * 两个身份散列（签含 exec_cap、delivery_per_cell、seed_rule，改任一项不重签即失败）。
    """
    table = EXPECTED_CELLS if expected_cells is None else expected_cells
    _, _, header_required, row_required = _schema_keys(SCHEMA)
    _exact_keys(header, header_required, "specs header", HEADER_OPTIONAL)
    if header["record"] != "header":
        raise SpecsError(f"specs 版本不符：{SCHEMA}")
    tier = header["difficulty"]
    if tier not in TIERS or header["runtime"] != RUNTIME:
        raise SpecsError(f"hard-specs/4 档位或 runtime 不符：{tier!r}")
    if header["seed_rule"] != seed_rule_for(tier, "v8"):
        raise SpecsError(f"hard-specs/4 只接受 v8 按档 seed 规则（{tier}）")
    if not _is_int(header["exec_cap"]) or header["exec_cap"] != EXEC_CAP:
        raise SpecsError(f"hard-specs/4 的 exec_cap 必须为 {EXEC_CAP}：{header['exec_cap']!r}")
    if header["layout_rule"] != LAYOUT_RULE:
        raise SpecsError(f"hard-specs/4 的 layout_rule 必须为 {LAYOUT_RULE}：{header['layout_rule']!r}")
    tasks = header["tasks"]
    if not isinstance(tasks, list) or len(set(tasks)) != len(tasks):
        raise SpecsError(f"hard-specs/4 的 tasks 必须是不重复列表：{tasks!r}")
    stray = [task for task in tasks if (task, tier) not in table]
    if stray:
        raise SpecsError(f"hard-specs/4 的任务不在 {tier} 交付格内：{stray}")
    for name in ("select_rule", "per_env", "delivery_per_cell"):
        value = header[name]
        if not isinstance(value, dict) or set(value) != set(tasks):
            raise SpecsError(f"hard-specs/4 的 {name} 必须是键集合等于 tasks 的逐任务字典：{value!r}")
    for task in tasks:
        indices = header["select_rule"][task]
        if not isinstance(indices, list) or not all(_is_int(i) and i >= 0 for i in indices) \
                or len(set(indices)) != len(indices):
            raise SpecsError(f"hard-specs/4 的 select_rule[{task}] 必须是不重复非负整数列表：{indices!r}")
        if not _is_int(header["per_env"][task]) or header["per_env"][task] < 0:
            raise SpecsError(f"hard-specs/4 的 per_env[{task}] 必须是非负整数：{header['per_env'][task]!r}")
        quota = header["delivery_per_cell"][task]
        if not _is_int(quota) or quota <= 0:
            raise SpecsError(f"hard-specs/4 的 delivery_per_cell[{task}] 必须是正整数：{quota!r}")
        if quota > table[(task, tier)]:
            raise SpecsError(f"hard-specs/4 的 delivery_per_cell[{task}]={quota} 超过表 2 格配额 "
                             f"{table[(task, tier)]}（{tier}）")
        if len(indices) != quota:
            raise SpecsError(f"hard-specs/4 的 select_rule[{task}] 长度 {len(indices)} ≠ delivery_per_cell {quota}")
        n_rows = sum(1 for row in rows if row.get("task") == task)
        if header["per_env"][task] != n_rows:
            raise SpecsError(f"hard-specs/4 的 per_env[{task}]={header['per_env'][task]} ≠ 本文件该任务行数 {n_rows}")
        if any(i >= header["per_env"][task] for i in indices):
            raise SpecsError(f"hard-specs/4 的 select_rule[{task}] 有索引越过 per_env={header['per_env'][task]}："
                             f"{indices!r}")
    if header["sampling_config_sha256"] != digest(header["sampling_config"]):
        raise SpecsError("内嵌 sampling_config 散列不自洽")
    max_episode = SEED_RULE["env_block"] // SEED_RULE["episode_stride"]
    seen, selected_count = set(), {}
    for row in rows:
        _exact_keys(row, row_required, "specs 行")
        # F-6：candidate／attempt／seed 必须是真整数（排除 bool 与浮点），否则 True==1、int(0.5)==0、
        # 16000000.0==16000000 会让后面的相等比较与 seed 公式误判通过
        bad_types = {name: row[name] for name in ("candidate", "attempt", "seed") if not _is_int(row[name])}
        if bad_types:
            raise SpecsError(f"hard-specs/4 行 candidate／attempt／seed 必须是整数（不得为布尔或浮点）："
                             f"{row.get('task')} {bad_types!r}")
        key = (row["task"], int(row["candidate"]))
        if row["record"] != "spec" or key in seen or row["task"] not in tasks:
            raise SpecsError(f"重复或额外的规格行：{key}")
        seen.add(key)
        # /4 行 candidate 即抽签 episode（与 _freeze 写法一致），episode 不得越过 env_block，否则跨任务 seed 会撞段
        if not _is_int(row["episode"]) or not 0 <= row["episode"] < max_episode:
            raise SpecsError(f"hard-specs/4 行 episode 必须落在 [0, {max_episode})：{key} {row['episode']!r}")
        if row["candidate"] != row["episode"]:
            raise SpecsError(f"hard-specs/4 行 candidate 必须等于 episode：{key} {row['episode']!r}")
        if row["tier"] != tier:
            raise SpecsError(f"规格行档位与 header 不符：{key}")
        if row["layout_parent"] is not None or (row.get("spec") or {}).get("spec_kind") != "native-newvalue/2":
            raise SpecsError(f"hard-specs/4 行 layout_parent 必须为 null、规格为 native-newvalue/2：{key}")
        if row["spec_sha256"] != spec_sha256(row["spec"]):
            raise SpecsError(f"规格散列不符：{key}")
        if row["seed"] != seed_for(row["task"], row["episode"], row["attempt"], header["seed_rule"]):
            raise SpecsError(f"seed 与公式不符：{key}")
        for flag in ("selected", "tried", "initial_selected"):
            if type(row[flag]) is not bool:
                raise SpecsError(f"{flag} 必须是布尔：{key}")
        rollout = row["rollout"]
        if rollout is not None and rollout.get("status") not in ROLLOUT_STATUSES:
            raise SpecsError(f"rollout.status 非法：{key}")
        if row["selected"]:
            selected_count[row["task"]] = selected_count.get(row["task"], 0) + 1
    over = {task: n for task, n in selected_count.items() if n > header["delivery_per_cell"][task]}
    if over:
        raise SpecsError(f"逐任务 selected 行数超过 delivery_per_cell：{over}")
    if identity_sha256(header, rows) != header["identity_sha256"]:
        raise SpecsError("identity_sha256 不符（签或来源被改过）")
    if delivery_sha256(rows) != header["delivery_sha256"]:
        raise SpecsError("delivery_sha256 不符（正式交付集合被改过）")


def validate_specs(header: dict[str, Any], rows: list[dict[str, Any]], *,
                   expected_cells: dict[tuple[str, str], int] | None = None) -> None:
    """封套校验：只认 ``hard-specs/4``，走 ``_validate_specs``（``expected_cells`` 作配额上限，缺省
    ``EXPECTED_CELLS``）；其余 schema（含已删除的 /2、/3）一律以「specs 版本不符」拒绝。"""
    schema = header.get("schema")
    if schema != SCHEMA:
        raise SpecsError(f"specs 版本不符：{schema}")
    _validate_specs(header, rows, EXPECTED_CELLS if expected_cells is None else expected_cells)


def load_specs(path: str | Path, *, expected_cells: dict[tuple[str, str], int] | None = None,
               check_fingerprint: bool = True):
    """唯一读取入口：返回 ``(header, rows)``（rows 为全部规格行，调用方按 ``delivered`` 取正式局）。

    ``expected_cells``：/4 文件的配额上限格表，缺省 ``EXPECTED_CELLS``。
    源码指纹（``provenance``）与当前环境不符只 ``warnings.warn``，不拒绝（0927 计划 §3.4）。
    """
    records = read_jsonl(path)
    header, rows = records[0], records[1:]
    validate_specs(header, rows, expected_cells=expected_cells)
    if check_fingerprint:
        provenance = header.get("provenance") or {}
        try:
            current = {"hard_fingerprint": hard_fingerprint(), "base_fingerprint": base_fingerprint()}
        except Exception as exc:  # noqa: BLE001 指纹只作佐证
            warnings.warn(f"无法计算源码指纹：{exc}")
            current = {}
        for key, value in current.items():
            if provenance.get(key) not in (None, value):
                warnings.warn(f"{path}：{key} 与当前源码不符（只警告，不拒绝）")
    return copy.deepcopy(header), copy.deepcopy(rows)


def load_specs_root(root: str | Path, expected_cells: dict[tuple[str, str], int], *,
                  cell_table: dict[tuple[str, str], int] | None = None,
                  check_fingerprint: bool = True) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """读 v8／v9 规格根（``<root>/<tier>/specs.jsonl``，``hard-specs/4``），只校验调用方给定的格表。

    ``expected_cells``：``{(task, tier): 局数}``，键必须是完整交付格表键的子集、值为正整数（值可小于表值，
    如冒烟每格 1 局）。完整根传 ``V9_CELLS``、冒烟传冒烟表、分片传该片子集。
    ``cell_table``：作配额上限的完整交付格表；缺省按 ``resolve_cell_table(expected_cells)`` 取（EXPECTED_CELLS →
    CELL_TABLES 第一张能覆盖的表），并原样传给逐份 ``load_specs``。只读 ``expected_cells``
    涉及的档位文件（其他档的文件即使存在也不读），逐份走 ``load_specs``（/4 校验），另查：

    * 每份 ``schema == "hard-specs/4"``、``difficulty`` 等于目录档名；
    * 每份 header ``tasks`` 的集合等于 ``expected_cells`` 在该档的任务集合；
    * ``expected_cells[key] ≤ cell_table[key]``；
    * 每格 header ``delivery_per_cell[task]`` 等于 ``expected_cells`` 的值；
    * 每格 ``selected`` 行数等于 ``expected_cells`` 的值（相等，不是 ≤）。只数 ``selected``，不看 rollout 结果；
      正式交付（``delivered``：selected 且 rollout ok）的逐格核对由 ``hard_regression.py delivery-set`` 的
      ``V8_DELIVERY_SET`` 负责；
    * 同任务跨档 seed 两两不交（比全部规格行，不只 selected）。

    返回 ``{tier: (header, rows)}``：键只含涉及的档位、按 ``TIERS`` 顺序；
    rows 为该档全部规格行（调用方按 ``selected`` 或 ``delivered`` 取）。任一不符抛 ``SpecsError``。
    """
    if not isinstance(expected_cells, dict) or not expected_cells:
        raise SpecsError("expected_cells 必须是非空 {(task, tier): 局数} 字典")
    table = resolve_cell_table(expected_cells) if cell_table is None else cell_table
    stray = sorted(key for key in expected_cells if key not in table)
    if stray:
        raise SpecsError(f"expected_cells 含交付格表（V9_CELLS）之外的格：{stray}")
    bad = {key: n for key, n in expected_cells.items() if not _is_int(n) or n <= 0}
    if bad:
        raise SpecsError(f"expected_cells 的局数必须是正整数：{bad}")
    over = {key: n for key, n in expected_cells.items() if n > table[key]}
    if over:
        raise SpecsError(f"expected_cells 的局数超过表 2 格配额：{over}")
    out: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for tier in TIERS:
        want = {task for task, t in expected_cells if t == tier}
        if not want:
            continue
        path = Path(root) / tier / "specs.jsonl"
        if not path.is_file():
            raise SpecsError(f"v8 规格根缺少 {path}")
        header, rows = load_specs(path, expected_cells=table, check_fingerprint=check_fingerprint)
        if header["schema"] != SCHEMA or header["difficulty"] != tier:
            raise SpecsError(f"{path}：schema 须为 {SCHEMA}、档位须为 {tier}"
                             f"（实为 {header['schema']}／{header['difficulty']}）")
        got_tasks = set(header["tasks"])
        if got_tasks != want:
            raise SpecsError(f"{tier} 任务集合不符：缺少 {sorted(want - got_tasks)}，多出 {sorted(got_tasks - want)}")
        for task in sorted(want):
            if header["delivery_per_cell"][task] != expected_cells[(task, tier)]:
                raise SpecsError(f"{task}/{tier} delivery_per_cell={header['delivery_per_cell'][task]} ≠ "
                                 f"期望 {expected_cells[(task, tier)]}")
            got = sum(1 for row in rows if row["task"] == task and row["selected"])
            if got != expected_cells[(task, tier)]:
                raise SpecsError(f"{task}/{tier} selected 行数 {got} ≠ 期望 {expected_cells[(task, tier)]}")
        out[tier] = (header, rows)
    seeds: dict[str, dict[str, set[int]]] = {}
    for tier, (_, rows) in out.items():
        for row in rows:
            seeds.setdefault(row["task"], {}).setdefault(tier, set()).add(int(row["seed"]))
    for task, by_tier in seeds.items():
        names = list(by_tier)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = by_tier[a] & by_tier[b]
                if common:
                    raise SpecsError(f"{task} 在 {a} 与 {b} 的 seed 相交：{sorted(common)[:5]}")
    return out


#: 规格根覆盖（0928 方案第二部分 §1.1）：设了即从该目录读 xhard{1..5}/specs.jsonl（按 TIERS），缺省读包内
SPECS_ROOT_ENV = "ROBOMME_HARD_SPECS_ROOT"
PACKAGED_SPECS_ROOT = Path(__file__).resolve().parents[1] / "env_metadata" / "ood"
_ANNOUNCED_ROOTS: set[str] = set()


def specs_root(override: str | Path | None = None) -> Path:
    """规格根：显式参数 > 环境变量 ``ROBOMME_HARD_SPECS_ROOT`` > 包内。非包内时打印一次 ``SPECS_ROOT=``。"""
    import os

    chosen = override if override is not None else os.environ.get(SPECS_ROOT_ENV)
    root = Path(chosen).resolve() if chosen else PACKAGED_SPECS_ROOT
    if root != PACKAGED_SPECS_ROOT and str(root) not in _ANNOUNCED_ROOTS:
        _ANNOUNCED_ROOTS.add(str(root))
        print(f"SPECS_ROOT={root}", flush=True)
    return root


def packaged_specs_path(tier: str, root: str | Path | None = None) -> Path:
    if tier not in TIERS:
        raise SpecsError(f"未知档位 {tier!r}")
    return specs_root(root) / tier / "specs.jsonl"


# ── 回注绑定摘要（策略仓库唯一调用的函数；0927 计划 §4.2、R22）───────────────


def _max_abs_diff(a: Any, b: Any) -> float:
    """两棵值树的最大绝对差；结构或非数值不同返回 inf。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a == b else math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max((_max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return math.inf
        return max((_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    return 0.0 if a == b else math.inf


def spec_binding(env) -> dict[str, Any]:
    """读 ``env.unwrapped._spec``（``SpecRecorder``），返回回注绑定摘要；必须在 ``reset()`` 之后调用。

    - ``injected_mismatch``：回注点（``value()``，trace ``source="spec"``）上的不等条数，外加记录点差 > 1e-5 的条数；
    - ``recorded_drift``：只记录不回注的观测点（``record()``，trace ``source="record"``）上差 ≤ ``RECORDED_FLOAT_TOL`` 的条数；
    - ``unused``：规格里有、本局没被 ``value()``／``record()`` 访问的取值点数。
    """
    recorder = getattr(getattr(env, "unwrapped", env), "_spec", None)
    if recorder is None:
        return {"available": False}
    spec_paths = {item["path"] for item in recorder.trace if item.get("source") == "spec"}
    record_paths = {item["path"] for item in recorder.trace if item.get("source") == "record"}
    injected, drift, drift_max = 0, 0, 0.0
    for item in recorder.mismatches:
        path = item["path"]
        if path in record_paths and path not in spec_paths:
            diff = _max_abs_diff(item.get("drawn"), item.get("frozen"))
            if diff <= RECORDED_FLOAT_TOL:
                drift += 1
                drift_max = max(drift_max, diff)
                continue
        injected += 1
    consumed = set(recorder.consumed_paths())
    unused = [p for p in recorder.leaf_paths() if not any(p == c or p.startswith(c + ".") for c in consumed)]
    frozen = getattr(recorder, "_frozen", None)
    # layered／layout_hit／layout_paths_expected／layout_overridden／layout_drift 五键原服务 V7 分层回注（已于维护计划
    # 阶段 1b 删除）。SpecRecorder 不再带这些属性，下面的 getattr 恒取缺省值（False／0／None），与删除前非分层局的输出逐字
    # 相同；键名保留是为了不改变摘要形态（策略仓库与 hard_regression reset-replay 读这些键）。
    layered_hit = getattr(recorder, "_layered_hit", None)
    if layered_hit is not None:
        layout_hit = len({item["path"] for item in recorder.trace if item.get("path") in layered_hit})
    else:
        layout_hit = len(getattr(recorder, "layout_paths_hit", ()) or ())
    return {
        "available": True,
        "mode": recorder.mode,
        "spec_kind": recorder.spec_kind,
        "spec_sha256": spec_sha256(frozen) if recorder.mode == "replay" else None,
        "value_points": sum(1 for item in recorder.trace if item.get("source") in ("draw", "spec")),
        "injected_mismatch": injected,
        "recorded_drift": drift,
        "recorded_max_abs": drift_max,
        "unused": len(unused),
        "layered": bool(getattr(recorder, "layered", False)),
        "layout_hit": layout_hit,
        "layout_paths_expected": len(layered_hit) if layered_hit is not None else None,
        "layout_overridden": int(getattr(recorder, "layout_overridden", 0)),
        "layout_drift": int(getattr(recorder, "layout_drift", 0)),
    }
