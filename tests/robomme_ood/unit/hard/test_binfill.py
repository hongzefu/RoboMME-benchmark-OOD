"""BinFill new-value tiers: cube count and quotas of the real offline ``_load_scene``, same-color cluster cap, packaged spec replay and self-export."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import importlib

from . import cells as C
from . import offline_scene as O

TASK = "BinFill"
BF = importlib.import_module("robomme_ood.robomme_env.BinFill")


def _cfg(tier):
    header, _ = O.delivered_rows(TASK, tier, 0)
    return header["sampling_config"][TASK]["decision"]["configs"][tier]


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_packaged_spec_replays_with_zero_mismatch(task, tier, k):
    C.check_packaged_replay(task, tier, k)


@pytest.mark.parametrize("task,tier,k", C.replay_cases(TASK))
def test_offline_export_equals_package_and_replays(task, tier, k):
    C.check_self_export(task, tier, k)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
def test_tampered_spec_is_detected(tier):
    C.check_tamper_detected(TASK, tier)


@pytest.mark.parametrize("tier", O.tiers_of(TASK))
@pytest.mark.parametrize("k", range(C.REPLAY_ROWS))
def test_cube_counts_quota_and_spacing(tier, k):
    row, env = C.replayed(TASK, tier, k)
    cfg = _cfg(tier)
    lo, hi = cfg["spawn_cubes"]
    assert lo <= len(env.all_cubes) <= hi
    by_color = {"red": env.red_cubes, "green": env.green_cubes, "blue": env.blue_cubes}
    assert len([c for c, v in by_color.items() if v]) == cfg["color"]
    assert sum(len(v) for v in by_color.values()) == len(env.all_cubes)
    targets = {c: getattr(env, f"{c}_cubes_target_number") for c in by_color}
    plo, phi = cfg["put_in_numbers"]
    assert plo <= sum(targets.values()) <= phi
    for c, cubes in by_color.items():
        assert 0 <= targets[c] <= len(cubes), c
        assert getattr(env, f"{c}_cubes_in_bin") == 0
    # by hand: necessary condition for disjoint cubes, center distance ≥ 2 × half side length
    xy = [a.pose.p[0, :2].numpy().astype(np.float64) for a in env.all_cubes]
    assert min(np.linalg.norm(a - b) for a, b in itertools.combinations(xy, 2)) >= 2 * env.cube_half_size - 1e-6
    # the language sequence lists only colors with a quota, counts match the quotas
    assert dict(env.binfill_language_sequence) == {c: n for c, n in targets.items() if n > 0}
    # cluster cap: when the fallback was not taken, the recorded largest same-color cluster does not exceed this tier's cap
    objs = row["spec"]["objects"]
    if not objs["color_mix_fallback"]:
        assert objs["color_mix_max_component"] <= cfg["color_mix"]["max_component"]


# ── pure function _max_same_color_component: hand-written small tables ───────────────────────────────


def test_same_color_component_link_is_inclusive():
    """Three same-color cubes in a row with adjacent spacing exactly link: ``<= link`` means connected → cluster size 3; slightly more than link → separate clusters."""
    link = 0.09
    xy = [(0.0, 0.0), (link, 0.0), (2 * link, 0.0)]
    assert BF._max_same_color_component(xy, ["r", "r", "r"], link) == 3
    far = [(0.0, 0.0), (link + 1e-6, 0.0), (2 * link + 2e-6, 0.0)]
    assert BF._max_same_color_component(far, ["r", "r", "r"], link) == 1


def test_same_color_component_ignores_other_colors_and_empty():
    xy = [(0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.15, 0.0)]
    # red-blue-red-red: the blue cube in the middle does not bridge two red cubes (0→2 distance 0.10 > 0.09)
    assert BF._max_same_color_component(xy, ["r", "b", "r", "r"], 0.09) == 2
    assert BF._max_same_color_component([], [], 0.09) == 0
    with pytest.raises(ValueError):
        BF._max_same_color_component(xy, ["r"], 0.09)
