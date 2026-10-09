"""包内规格取值的测试侧判定器（只用标准库 json 读 ``specs.jsonl``，不依赖私有评估仓的工具）。

旧仓 ``scripts/parity/hard_regression.py``（``fd0017d6``）的 ``delivery-set``／``tier-values``／``movecube-layout``
三个子命令随私有评估仓走；它们对**包内数据本身**的断言在本仓仍须有人守护，所以把判据逐条搬到这里，数值与
口径和该工具一致：

- :func:`tier_dims`：从交付行 ``spec`` 读本局实际取值（字段与 ``hard_regression.tier_dims`` 相同）；
- :func:`delivery_set_check`：逐格数交付行（selected 且 ``rollout.status == "ok"``）与格表相等、格表外无交付、
  selected 而未成功为 0、交付行执行步不超上限；同任务跨档 seed 两两不交；布局独立（header ``layout_rule``、
  ``layout_parent`` 全空、跨档交付行浮点位置叶子不逐位相等、PatternLock 路径无 ≥ 9 节点的共同前缀）；
- :func:`movecube_layout_check`：MoveCube 每局 demo／execution 两段 × 方块、goal、抓杆点三点用源码
  ``MoveCube._in_region_u`` 判在 V9 区域内，至少一点落在旧 V8 区域外，段内 region 与源码及 V9 三个数一致；
  运动方式取最后一次 ``_initialize_episode`` 的 ``way_idx`` 计数。

本模块不写业务常量（格表、区域、配额、上限一律由调用方从 ``test_constants`` 传入）；``hard_regression`` 自身
的结构常量（位置子树、路径前缀阈值、段名与点名）照抄。不是测试文件，不被收集。
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- 读取


def read_tier(root: Path, tier: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``<root>/<tier>/specs.jsonl`` → (header, rows)；标准库 json 逐行读，不经被测的读取函数。"""
    records = [json.loads(line) for line in (Path(root) / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records or records[0].get("record") != "header":
        raise ValueError(f"{root}/{tier}/specs.jsonl：空文件或首行不是 header")
    return records[0], records[1:]


def delivered(row: dict[str, Any]) -> bool:
    """交付行：selected 且 rollout.status == "ok"（与 hard_specs.delivered 同口径，测试侧独立写）。"""
    return bool(row.get("selected")) and (row.get("rollout") or {}).get("status") == "ok"


# ---------------------------------------------------------------- 档位取值


def _actual_int(value: Any, where: str) -> int:
    """取实际整数：``{actual|placed: n}`` 取实际值；否则须为非负整数。"""
    if isinstance(value, dict):
        value = value.get("actual", value.get("placed"))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{where} 不是非负整数：{value!r}")
    return value


def tier_dims(task: str, spec: dict[str, Any]) -> dict[str, int]:
    """从 /4 行 ``spec`` 读本局实际取值（字段逐项照 ``hard_regression.tier_dims``）。"""
    objects, actions = spec.get("objects") or {}, spec.get("actions") or {}
    where = f"{task}.spec"
    if task in ("PickXtimes", "SwingXtimes"):
        return {"times" if task == "PickXtimes" else "rounds": _actual_int(objects["num_repeats"], f"{where}.objects.num_repeats"),
                "distractors": _actual_int(objects["distractor_count"], f"{where}.objects.distractor_count")}
    if task == "StopCube":
        return {"stop_time": _actual_int(actions["stop_time"], f"{where}.actions.stop_time"),
                "move_interval": _actual_int(actions["move_interval"], f"{where}.actions.move_interval")}
    if task in ("VideoUnmask", "ButtonUnmask"):
        dist = objects["distractors"]
        return {"pick": _actual_int(objects["n_picks"], f"{where}.objects.n_picks"),
                "distractor_bins": _actual_int(dist["placed"], f"{where}.objects.distractors.placed"),
                "distractor_cubes": _actual_int(dist["cube_count"], f"{where}.objects.distractors.cube_count")}
    if task in ("VideoUnmaskSwap", "ButtonUnmaskSwap"):
        return {"swap": _actual_int(objects["n_swaps"], f"{where}.objects.n_swaps"),
                "pick": _actual_int(objects["n_picks"], f"{where}.objects.n_picks"),
                "outer": _actual_int(objects["distractors"]["placed"], f"{where}.objects.distractors.placed")}
    if task == "BinFill":
        return {"put_in": sum(_actual_int(v, f"{where}.objects.target_numbers") for v in objects["target_numbers"])}
    if task == "VideoPlaceButton":
        return {"placements": _actual_int(actions["target_placement_count"], f"{where}.actions.target_placement_count")}
    if task == "VideoPlaceOrder":
        visits = objects.get("visit_counts_by_object")
        if isinstance(visits, list):
            return {"visits": sum(_actual_int(v, f"{where}.objects.visit_counts_by_object") for v in visits)}
        return {"visits": _actual_int(actions["target_placement_count"], f"{where}.actions.target_placement_count")}
    if task == "PickHighlight":
        return {"pick": _actual_int(objects["highlight_count"], f"{where}.objects.highlight_count"),
                "total": _actual_int(objects["n_cubes_spawned"], f"{where}.objects.n_cubes_spawned")}
    if task == "VideoRepick":
        return {"cubes": _actual_int(objects["cube_count"], f"{where}.objects.cube_count"),
                "swap": _actual_int(objects["n_swaps"], f"{where}.objects.n_swaps"),
                "repick": _actual_int(objects["num_repeats"], f"{where}.objects.num_repeats")}
    if task == "RouteStick":
        return {"segments": _actual_int(objects["L"], f"{where}.objects.L")}
    if task == "PatternLock":
        nodes = actions["path_nodes"]
        if not isinstance(nodes, list):
            raise ValueError(f"{where}.actions.path_nodes 不是列表")
        return {"nodes": len(nodes)}
    raise KeyError(f"取值表不含任务 {task}")


def value_ok(got: Any, want: Any) -> bool:
    """定值逐值相等；``(lo, hi)`` 为闭区间。"""
    if isinstance(want, tuple):
        return isinstance(got, int) and not isinstance(got, bool) and want[0] <= got <= want[1]
    return got == want


def row_mismatches(rows: list[dict[str, Any]], tier: str, table: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """交付行逐局实际取值与表比；读取失败与键集合不符同样记入。返回不符清单（空 = 全对）。"""
    out = []
    for row in rows:
        if row.get("task") not in table or not delivered(row):
            continue
        label = f"{row['task']}/{tier}#{row.get('candidate')}"
        try:
            got = tier_dims(row["task"], row.get("spec") or {})
        except (KeyError, TypeError, ValueError) as exc:
            out.append(f"{label}:读取失败 {exc}")
            continue
        want = table[row["task"]][tier]
        wrong = {dim: (got.get(dim), value) for dim, value in want.items() if not value_ok(got.get(dim), value)}
        if wrong or set(got) != set(want):
            out.append(f"{label}:{wrong}")
    return out


def tier_values_check(root: Path, tiers: tuple[str, ...], table: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """``tier-values`` 的等价判定：表内每格逐局比，``mismatches`` = 不符局数 + 无行的格数；缺档文件另计。"""
    mismatches, rows_checked, missing_files = [], 0, []
    seen_cells: set[tuple[str, str]] = set()
    for tier in tiers:
        if not (Path(root) / tier / "specs.jsonl").is_file():
            missing_files.append(tier)
            continue
        _, rows = read_tier(root, tier)
        chosen = [r for r in rows if r.get("task") in table and tier in table[r["task"]] and delivered(r)]
        rows_checked += len(chosen)
        seen_cells |= {(r["task"], tier) for r in chosen}
        mismatches += row_mismatches(chosen, tier, table)
    want_cells = {(task, tier) for task, by_tier in table.items() for tier in by_tier if tier in tiers}
    empty = sorted(f"{t}/{tier}:无行" for t, tier in want_cells - seen_cells if tier not in missing_files)
    return {"mismatches": mismatches + empty, "rows": rows_checked, "missing_files": missing_files}


# ---------------------------------------------------------------- 交付集、seed 隔离、布局独立

#: 位置类子树（照抄 hard_regression.POSITION_SUBTREES）
POSITION_SUBTREES = ("layout", "initializations", "actions.path_nodes", "actions.nodes", "objects.distractors.bins")
#: PatternLock 图案节点路径；两条路径共同前缀至少这么长才算照抄
PATH_NODES = "actions.path_nodes"
PATH_NODES_MIN_PREFIX = 9


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _position_leaves(tree: Any, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            out.update(_position_leaves(value, f"{prefix}.{key}"))
    elif isinstance(tree, list) and tree and all(_is_number(v) for v in tree):
        out[prefix] = tree
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            out.update(_position_leaves(value, f"{prefix}.{index}"))
    elif isinstance(tree, float):
        out[prefix] = tree
    return out


def position_leaves(spec: dict[str, Any]) -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    for path in POSITION_SUBTREES:
        node: Any = spec
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is not None:
            leaves.update(_position_leaves(node, path))
    return leaves


def _has_float(value: Any) -> bool:
    return isinstance(value, float) or (isinstance(value, list) and any(isinstance(v, float) for v in value))


def layout_overlap(items: list[tuple[str, dict[str, Any], bool]]) -> dict[str, Any]:
    """同一任务跨档位置照抄检测（口径照抄 hard_regression.layout_overlap）：剔除在全部规格行里出现 ≥ 2 次且处处
    相同的恒定叶子；不同档交付行共有的浮点叶子任一逐位相等，或 PatternLock 路径有 ≥ 9 节点的共同前缀，记一对。"""
    leaves = [(tier, row, is_delivered, position_leaves(row.get("spec") or {})) for tier, row, is_delivered in items]
    seen: dict[str, set[str]] = collections.defaultdict(set)
    count: collections.Counter = collections.Counter()
    for _, _, _, row_leaves in leaves:
        for path, value in row_leaves.items():
            seen[path].add(json.dumps(value, sort_keys=True))
            count[path] += 1
    constant = {path for path, values in seen.items() if count[path] >= 2 and len(values) == 1}
    comparable = []
    for tier, row, is_delivered, row_leaves in leaves:
        if not is_delivered:
            continue
        floats = {p: json.dumps(v) for p, v in row_leaves.items() if p not in constant and _has_float(v)}
        nodes = row_leaves.get(PATH_NODES) if PATH_NODES not in constant else None
        comparable.append((tier, row, floats, nodes if isinstance(nodes, list) else None))
    tiers = {tier for tier, *_ in comparable}
    no_position = sum(1 for _, _, floats, nodes in comparable if not floats and nodes is None) if len(tiers) > 1 else 0
    pairs, detail = 0, []
    for i, (tier_a, row_a, floats_a, nodes_a) in enumerate(comparable):
        for tier_b, row_b, floats_b, nodes_b in comparable[i + 1:]:
            if tier_a == tier_b:
                continue
            hit = next((p for p in floats_a.keys() & floats_b.keys() if floats_a[p] == floats_b[p]), None)
            if hit is None and nodes_a is not None and nodes_b is not None:
                n = min(len(nodes_a), len(nodes_b))
                if n >= PATH_NODES_MIN_PREFIX and nodes_a[:n] == nodes_b[:n]:
                    hit = PATH_NODES
            if hit is not None:
                pairs += 1
                if len(detail) < 6:
                    detail.append(f"{row_a['task']}:{tier_a}/{row_a['candidate']}={tier_b}/{row_b['candidate']}@{hit}")
    return {"pairs": pairs, "no_position": no_position, "detail": detail}


def delivery_set_check(root: Path, tiers: tuple[str, ...], cells: dict[tuple[str, str], int], *, exec_cap: int,
                       layout_rule: dict[str, Any]) -> dict[str, Any]:
    """``delivery-set`` 三行判定的等价计数（不含 ``load_specs_root`` 校验，那一项由 test_packaged_specs 覆盖）。"""
    files: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    missing_files = []
    for tier in tiers:
        if not any(t == tier for _, t in cells):
            continue
        if not (Path(root) / tier / "specs.jsonl").is_file():
            missing_files.append(tier)
            continue
        files[tier] = read_tier(root, tier)
    counted: collections.Counter = collections.Counter()
    selected_not_ok = delivered_over_cap = parent_non_null = 0
    seeds: dict[str, dict[str, set[int]]] = collections.defaultdict(lambda: collections.defaultdict(set))
    by_task: dict[str, list[tuple[str, dict[str, Any], bool]]] = collections.defaultdict(list)
    for tier, (_, rows) in files.items():
        for row in rows:
            task, ok = row["task"], delivered(row)
            seeds[task][tier].add(int(row["seed"]))
            by_task[task].append((tier, row, ok))
            parent_non_null += int(row.get("layout_parent") is not None)
            if ok:
                counted[(task, tier)] += 1
                steps = (row.get("rollout") or {}).get("exec_steps")
                delivered_over_cap += int(_is_number(steps) and steps > exec_cap)
            elif row.get("selected"):
                selected_not_ok += 1
    cell_mismatch = sorted(f"{t}/{tier}:{counted.get((t, tier), 0)}/{n}" for (t, tier), n in cells.items()
                           if counted.get((t, tier), 0) != n)
    extra_cells = sorted(f"{t}/{tier}" for (t, tier) in counted if (t, tier) not in cells)
    shared = 0
    for task, by_tier in seeds.items():
        names = sorted(by_tier)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared += len(by_tier[a] & by_tier[b])
    mode_bad = sum(header.get("layout_rule") != layout_rule for header, _ in files.values())
    pairs = no_position = 0
    detail: list[str] = []
    for task in sorted(by_task):
        r = layout_overlap(by_task[task])
        pairs += r["pairs"]
        no_position += r["no_position"]
        detail += r["detail"]
    return {"cells": len(counted), "total": sum(counted.values()), "cell_mismatch": cell_mismatch,
            "extra_cells": extra_cells, "selected_not_ok": selected_not_ok, "delivered_over_cap": delivered_over_cap,
            "missing_files": missing_files, "seed_tasks": len(seeds), "shared": shared, "mode_bad": mode_bad,
            "parent_non_null": parent_non_null, "layout_equal_pairs": pairs, "no_position": no_position,
            "layout_detail": detail}


# ---------------------------------------------------------------- MoveCube 新区域与运动方式

MOVECUBE_SEGMENTS = ("demo", "execution")
MOVECUBE_POINT_NAMES = ("cube", "goal", "grasp")


def movecube_way(spec: dict[str, Any]) -> int | None:
    """录像局的运动方式：最后一次 ``_initialize_episode`` 的 ``way_idx``（照抄 ``_freeze._movecube_way``）。"""
    inits = spec.get("initializations") if isinstance(spec, dict) else None
    if not isinstance(inits, dict) or not inits:
        return None
    last = max(inits, key=lambda k: int(k))
    way = inits[last].get("way_idx") if isinstance(inits[last], dict) else None
    return int(way) if way is not None else None


def movecube_points(spec: dict[str, Any], seg: str, module) -> dict[str, Any]:
    """一段的三个落点与该段规格 region（经源码 ``_xhard4_region`` 整理）；杆长由 ``peg_axis_extent_m`` 反算并复核。"""
    import numpy as np

    layout = spec["layout"][seg]
    region_cfg = layout["region"]
    region = module.MoveCube._xhard4_region(None, {"xhard4": {"region": region_cfg}}, f"{seg}_layout")
    extent = tuple(float(v) for v in region_cfg["peg_axis_extent_m"])
    length = 2.0 * extent[1]
    if not np.allclose(module._peg_axis_extent(length), extent, rtol=0.0, atol=1e-12):
        raise ValueError(f"{seg}.peg_axis_extent_m {extent} 与 _peg_axis_extent({length}) 不符")
    base_y, root_x, root_y = (float(v) for v in layout["peg_offsets"])
    root = module._peg_root_xy(base_y, root_x, root_y)
    grasp, _, _ = module._peg_geometry(root, float(layout["peg_yaw"]), length, extent)
    points = {"cube": np.asarray(layout["cube_pose"][:2], dtype=np.float64),
              "goal": np.asarray(layout["goal_xy"][:2], dtype=np.float64),
              "grasp": np.asarray(grasp, dtype=np.float64)}
    return {"region_cfg": region_cfg, "region": region, "points": points}


def movecube_layout_check(rows: list[dict[str, Any]], *, module, v9_region: dict[str, Any],
                          v8_region: dict[str, Any]) -> dict[str, Any]:
    """逐局两段三点 → in_region／outside_old／region_mismatch／坏行，并按运动方式计数（照抄
    ``hard_regression.movecube_layout_check``）。是否 PASS 由调用方对照钉值判断。"""
    source_cfg = module.MoveCube.config_xhard4["region"]
    in_region = outside_old = points = region_mismatch = 0
    bad_rows: list[str] = []
    out_detail: list[str] = []
    ways: collections.Counter = collections.Counter()
    for row in rows:
        label = f"{row.get('task')}/c{row.get('candidate')}/seed{row.get('seed')}"
        try:
            segs = {seg: movecube_points(row["spec"], seg, module) for seg in MOVECUBE_SEGMENTS}
        except Exception as exc:  # noqa: BLE001 缺字段、类型不对、源码校验拒绝 → 坏行
            bad_rows.append(f"{label}: {type(exc).__name__}: {exc}"[:300])
            continue
        ways[movecube_way(row["spec"])] += 1
        for seg, item in segs.items():
            cfg = item["region_cfg"]
            diff = {k for k in source_cfg if cfg.get(k) != source_cfg[k]}
            diff |= {k for k, v in v9_region.items() if cfg.get(k) != v}
            region_mismatch += int(bool(diff))
            old = module.MoveCube._xhard4_region(None, {"xhard4": {"region": {**cfg, **v8_region}}}, f"{seg}_layout")
            for name in MOVECUBE_POINT_NAMES:
                xy = item["points"][name]
                points += 1
                why = module._in_region_u(xy, item["region"])
                if why is None:
                    in_region += 1
                elif len(out_detail) < 6:
                    out_detail.append(f"{label}/{seg}/{name}:{why}")
                outside_old += int(module._in_region_u(xy, old) is not None)
    return {"episodes": len(rows), "points": points, "in_region": in_region, "outside_old": outside_old,
            "region_mismatch": region_mismatch, "bad_rows": bad_rows, "out_detail": out_detail, "ways": dict(ways)}
