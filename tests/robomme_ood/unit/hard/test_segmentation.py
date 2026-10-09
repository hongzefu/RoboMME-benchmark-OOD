"""Segmentation filtering and coordinate-filled subgoals (``utils/segmentation_utils.process_segmentation``), migrated from the old ``test_audit_fix`` with added boundaries.

Hand-written 64×64 segmentation map: each object is a 3×3 square with hand-computed center; ``<>`` in the subgoal is filled in order with each target center ``<y, x>``.
For targets marked ``_robomme_refresh_on_move_px``, when the subgoal has not switched, the center is refreshed only if it moves strictly more than the threshold (equal to the threshold does not refresh).
"""
from __future__ import annotations

import numpy as np

from robomme_ood.robomme_env.utils import segmentation_utils as SU


class _Actor:
    def __init__(self, name="obj"):
        self.name = name


def _seg(blocks):
    seg = np.zeros((64, 64), dtype=np.int32)
    for obj_id, (y, x) in blocks.items():
        seg[y - 1:y + 2, x - 1:x + 2] = obj_id
    return seg


def _run(seg, id_map, segment, sub, prev, existing=None, filled=None, color_map=None):
    return SU.process_segmentation(
        segmentation=seg, segmentation_id_map=id_map, color_map={} if color_map is None else color_map,
        current_segment=segment, current_subgoal_segment=sub, previous_subgoal_segment=prev,
        current_task_name="TASK", existing_points=existing, existing_subgoal_filled=filled,
    )


def test_switch_fills_each_placeholder_with_its_target_centre():
    a, b, c = _Actor(), _Actor(), _Actor()
    seg = _seg({1: (10, 20), 2: (40, 50), 3: (30, 30)})
    out = _run(seg, {1: a, 2: b, 3: c}, [a, b], "move <> to <>", prev=None)
    assert out["current_subgoal_segment_filled"] == "move <10, 20> to <40, 50>"
    assert out["segmentation_points"] == [[10, 20], [40, 50]]
    assert out["vis_obj_id_list"] == [1, 2]
    # filtering: keep only target ids, zero all other pixels
    assert set(np.unique(out["segmentation_result"])) == {0, 1, 2}
    assert out["updated_previous_subgoal_segment"] == "move <> to <>"


def test_single_centre_repeats_for_multiple_placeholders():
    a = _Actor()
    out = _run(_seg({1: (5, 6)}), {1: a}, a, "<> and <>", prev="x")
    assert out["current_subgoal_segment_filled"] == "<5, 6> and <5, 6>"


def test_missing_target_falls_back_to_task_name_and_flags():
    a, b = _Actor(), _Actor()
    out = _run(_seg({1: (5, 6)}), {1: a, 2: b}, [a, b], "move <> to <>", prev=None)
    assert out["no_object_flag"] is True
    assert out["current_subgoal_segment_filled"] == "TASK"


def test_table_workspace_colour_blacked_out():
    table = _Actor("table-workspace")
    colors = {7: [9, 9, 9]}
    _run(_seg({}), {7: table}, None, None, prev=None, color_map=colors)
    assert colors[7] == [0, 0, 0]


def _same_subgoal(actor_px, move_to):
    a, b = _Actor(), _Actor()
    if actor_px is not None:
        a._robomme_refresh_on_move_px = actor_px
    seg = _seg({1: move_to, 2: (5, 50)})
    return _run(seg, {1: a, 2: b}, a, "pick at <>", prev="pick at <>", existing=[[10, 10]],
                filled="pick at <10, 10>")


def test_untagged_target_keeps_cached_centre():
    out = _same_subgoal(None, (40, 40))
    assert out["segmentation_points"] == [[10, 10]] and out["current_subgoal_segment_filled"] == "pick at <10, 10>"


def test_tagged_target_refreshes_when_moved_beyond_threshold():
    out = _same_subgoal(8, (40, 40))
    assert out["segmentation_points"] == [[40, 40]] and out["current_subgoal_segment_filled"] == "pick at <40, 40>"


def test_refresh_threshold_is_strict():
    # Chebyshev distance exactly 8: no refresh; 9: refresh
    assert _same_subgoal(8, (18, 10))["segmentation_points"] == [[10, 10]]
    assert _same_subgoal(8, (19, 10))["segmentation_points"] == [[19, 10]]
