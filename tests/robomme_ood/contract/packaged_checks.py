"""Test-side checkers for packaged spec values (read ``specs.jsonl`` with stdlib json only, no dependency on the private evaluation repo's tools).

The ``delivery-set``/``tier-values``/``movecube-layout`` subcommands of the old repo's ``scripts/parity/hard_regression.py`` (``fd0017d6``)
moved to the private evaluation repo; their assertions about **the packaged data itself** still need a guard in this repo, so the criteria are moved here one by one, with values and
basis identical to that tool:

- :func:`tier_dims`: read the episode's actual values from the delivered row's ``spec`` (fields identical to ``hard_regression.tier_dims``);
- :func:`delivery_set_check`: per-cell delivered row count (selected and ``rollout.status == "ok"``) equals the cell table, no delivery outside the table,
  selected-but-failed is 0, delivered rows do not exceed the execution-step cap; same-task seeds pairwise disjoint across tiers; layouts independent (header ``layout_rule``,
  ``layout_parent`` all null, float position leaves of delivered rows not bitwise equal across tiers, no PatternLock path common prefix of >= 9 nodes);
- :func:`movecube_layout_check`: for each MoveCube episode, demo/execution segments x the cube, goal and peg grasp point are checked with the source
  ``MoveCube._in_region_u`` to lie in the V9 region, at least one point outside the old V8 region, and the in-segment region consistent with the source and the three V9 numbers;
  motion modes counted from ``way_idx`` of the last ``_initialize_episode``.

This module writes no business constants (cell table, region, quota and caps are always passed in by the caller from ``test_constants``); ``hard_regression``'s own
structural constants (position subtrees, path prefix threshold, segment and point names) are copied verbatim. Not a test file; not collected.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- Reading


def read_tier(root: Path, tier: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``<root>/<tier>/specs.jsonl`` -> (header, rows); read line by line with stdlib json, not via the reader under test."""
    records = [json.loads(line) for line in (Path(root) / tier / "specs.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records or records[0].get("record") != "header":
        raise ValueError(f"{root}/{tier}/specs.jsonl: empty file or first line is not a header")
    return records[0], records[1:]


def delivered(row: dict[str, Any]) -> bool:
    """Delivered rows: selected and rollout.status == "ok" (same basis as hard_specs.delivered, written independently on the test side)."""
    return bool(row.get("selected")) and (row.get("rollout") or {}).get("status") == "ok"


# ---------------------------------------------------------------- Tier values


def _actual_int(value: Any, where: str) -> int:
    """Take the actual integer: ``{actual|placed: n}`` takes the actual value; otherwise must be a non-negative integer."""
    if isinstance(value, dict):
        value = value.get("actual", value.get("placed"))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{where} is not a non-negative integer: {value!r}")
    return value


def tier_dims(task: str, spec: dict[str, Any]) -> dict[str, int]:
    """Read the episode's actual values from a /4 row's ``spec`` (fields follow ``hard_regression.tier_dims`` item by item)."""
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
            raise ValueError(f"{where}.actions.path_nodes is not a list")
        return {"nodes": len(nodes)}
    raise KeyError(f"value table has no task {task}")


def value_ok(got: Any, want: Any) -> bool:
    """Fixed values compared for equality; ``(lo, hi)`` is a closed interval."""
    if isinstance(want, tuple):
        return isinstance(got, int) and not isinstance(got, bool) and want[0] <= got <= want[1]
    return got == want


def row_mismatches(rows: list[dict[str, Any]], tier: str, table: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Compare each delivered episode's actual values to the table; read failures and key-set mismatches are recorded too. Returns the mismatch list (empty = all correct)."""
    out = []
    for row in rows:
        if row.get("task") not in table or not delivered(row):
            continue
        label = f"{row['task']}/{tier}#{row.get('candidate')}"
        try:
            got = tier_dims(row["task"], row.get("spec") or {})
        except (KeyError, TypeError, ValueError) as exc:
            out.append(f"{label}:read failed {exc}")
            continue
        want = table[row["task"]][tier]
        wrong = {dim: (got.get(dim), value) for dim, value in want.items() if not value_ok(got.get(dim), value)}
        if wrong or set(got) != set(want):
            out.append(f"{label}:{wrong}")
    return out


def tier_values_check(root: Path, tiers: tuple[str, ...], table: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Equivalent check of ``tier-values``: compare per episode for every cell in the table; ``mismatches`` = mismatched episodes + cells with no rows; missing tier files counted separately."""
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
    empty = sorted(f"{t}/{tier}:no rows" for t, tier in want_cells - seen_cells if tier not in missing_files)
    return {"mismatches": mismatches + empty, "rows": rows_checked, "missing_files": missing_files}


# ---------------------------------------------------------------- Delivery set, seed isolation, layout independence

#: Position subtrees (copied from hard_regression.POSITION_SUBTREES)
POSITION_SUBTREES = ("layout", "initializations", "actions.path_nodes", "actions.nodes", "objects.distractors.bins")
#: PatternLock pattern node paths; two paths count as copied only if their common prefix is at least this long
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
    """Same-task cross-tier position copy detection (basis copied from hard_regression.layout_overlap): drop constant leaves that occur >= 2 times in all spec rows and are identical
    everywhere; a pair is recorded if any float leaf shared by delivered rows of different tiers is bitwise equal, or PatternLock paths share a common prefix of >= 9 nodes."""
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
    """Equivalent counts for the three verdict lines of ``delivery-set`` (excluding ``load_specs_root`` validation, which test_packaged_specs covers)."""
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


# ---------------------------------------------------------------- MoveCube new region and motion modes

MOVECUBE_SEGMENTS = ("demo", "execution")
MOVECUBE_POINT_NAMES = ("cube", "goal", "grasp")


def movecube_way(spec: dict[str, Any]) -> int | None:
    """Motion mode of the recorded episode: ``way_idx`` of the last ``_initialize_episode`` (copied from ``_freeze._movecube_way``)."""
    inits = spec.get("initializations") if isinstance(spec, dict) else None
    if not isinstance(inits, dict) or not inits:
        return None
    last = max(inits, key=lambda k: int(k))
    way = inits[last].get("way_idx") if isinstance(inits[last], dict) else None
    return int(way) if way is not None else None


def movecube_points(spec: dict[str, Any], seg: str, module) -> dict[str, Any]:
    """The three landing points of a segment and that segment's spec region (normalized via the source ``_xhard4_region``); peg length is back-computed from ``peg_axis_extent_m`` and re-checked."""
    import numpy as np

    layout = spec["layout"][seg]
    region_cfg = layout["region"]
    region = module.MoveCube._xhard4_region(None, {"xhard4": {"region": region_cfg}}, f"{seg}_layout")
    extent = tuple(float(v) for v in region_cfg["peg_axis_extent_m"])
    length = 2.0 * extent[1]
    if not np.allclose(module._peg_axis_extent(length), extent, rtol=0.0, atol=1e-12):
        raise ValueError(f"{seg}.peg_axis_extent_m {extent} does not match _peg_axis_extent({length})")
    base_y, root_x, root_y = (float(v) for v in layout["peg_offsets"])
    root = module._peg_root_xy(base_y, root_x, root_y)
    grasp, _, _ = module._peg_geometry(root, float(layout["peg_yaw"]), length, extent)
    points = {"cube": np.asarray(layout["cube_pose"][:2], dtype=np.float64),
              "goal": np.asarray(layout["goal_xy"][:2], dtype=np.float64),
              "grasp": np.asarray(grasp, dtype=np.float64)}
    return {"region_cfg": region_cfg, "region": region, "points": points}


def movecube_layout_check(rows: list[dict[str, Any]], *, module, v9_region: dict[str, Any],
                          v8_region: dict[str, Any]) -> dict[str, Any]:
    """Two segments x three points per episode -> in_region/outside_old/region_mismatch/bad rows, counted per motion mode (copied from
    ``hard_regression.movecube_layout_check``). Whether it PASSes is judged by the caller against the pins."""
    source_cfg = module.MoveCube.config_xhard4["region"]
    in_region = outside_old = points = region_mismatch = 0
    bad_rows: list[str] = []
    out_detail: list[str] = []
    ways: collections.Counter = collections.Counter()
    for row in rows:
        label = f"{row.get('task')}/c{row.get('candidate')}/seed{row.get('seed')}"
        try:
            segs = {seg: movecube_points(row["spec"], seg, module) for seg in MOVECUBE_SEGMENTS}
        except Exception as exc:  # noqa: BLE001 missing field, wrong type, source validation rejection -> bad row
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
