"""Mapping of official multiple-choice actions (C07 projection and nearest, C09 multi_choice wrapper).

- choice_action_mapping: world point -> pixel projection (hand-computed pinhole model), out of bounds/behind camera -> None, inverted-extrinsics fallback;
  target selection by nearest pixel and nearest 3D; nested candidates deduplicated; non-finite input -> None;
- oracle_action_matcher: exact label match, action text -> label;
- OraclePlannerDemonstrationWrapper: point in the command is [y, x], converted to [x, y] and rounded before the selector;
  label stripped and lowercased then matched exactly; no match -> empty batch; target required but no point given or no match -> ValueError;
  once selected, really runs the "collect step by step -> evaluate twice -> output" flow (solver replaced by a double calling planner.env.step).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robomme.env_record_wrapper.OraclePlannerDemonstrationWrapper import OraclePlannerDemonstrationWrapper
from robomme.robomme_env.utils import choice_action_mapping as cam
from robomme.robomme_env.utils import vqa_options
from robomme.robomme_env.utils.oracle_action_matcher import (
    find_exact_label_option_index,
    map_action_text_to_option_label,
)

from _official_fakes import FakeTaskEnv, as_made

K = [[100, 0, 50], [0, 100, 40], [0, 0, 1]]
E_ID = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
SHAPE = (100, 120)  # (H, W)


class _Actor:
    """Hashable actor double (real actors hash by object identity)."""

    def __init__(self, name, xyz):
        self.name = name
        self.pose = SimpleNamespace(p=torch.tensor([xyz], dtype=torch.float32))


def _actor(name, xyz):
    return _Actor(name, xyz)


# --------------------------------------------------------------------------- Projection


@pytest.mark.parametrize("xyz, pixel", [
    ((0.1, 0.2, 1.0), [60, 60]),        # x = 100*0.1/1 + 50, y = 100*0.2/1 + 40
    ((0.0, 0.0, 2.0), [50, 40]),        # principal point
    ((-0.25, -0.2, 1.0), [25, 20]),
    ((0.69, 0.0, 1.0), [119, 40]),      # rightmost column x = 119 (width 120)
    ((-0.5, -0.4, 1.0), [0, 0]),        # top-left corner
])
def test_projection_handcomputed(xyz, pixel):
    assert cam.project_world_to_pixel(xyz, K, E_ID, SHAPE) == pixel


@pytest.mark.parametrize("xyz", [(0, 0, 0), (2.0, 0, 1.0), (0.70, 0.0, 1.0), (0, 0.61, 1.0), (0, 0, np.nan)])
def test_projection_rejects_behind_outside_nonfinite(xyz):
    assert cam.project_world_to_pixel(xyz, K, E_ID, SHAPE) is None


def test_projection_falls_back_to_inverted_extrinsic():
    e = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -2]]  # taken as world->camera the point is behind the camera (z = -1)
    assert cam.project_world_to_pixel((0, 0, 1), K, e, SHAPE) == [50, 40]  # after inversion z = 3


@pytest.mark.parametrize("bad", [dict(intrinsic_cv=[1, 2]), dict(extrinsic_cv=[1] * 11), dict(image_shape=(0, 10)),
                                 dict(image_shape=None)])
def test_projection_rejects_bad_camera(bad):
    args = dict(world_xyz=(0, 0, 1), intrinsic_cv=K, extrinsic_cv=E_ID, image_shape=SHAPE)
    args.update(bad)
    assert cam.project_world_to_pixel(**args) is None


# --------------------------------------------------------------------------- Selection


def test_select_by_pixel_nearest_and_dedup():
    a, b, c = _actor("a", (0.1, 0.2, 1.0)), _actor("b", (0.0, 0.0, 2.0)), _actor("c", (0.0, 0.0, -1.0))
    nopose = _Actor("nopose", (0, 0, 1))
    nopose.pose = None
    res = cam.select_target_with_pixel([a, [b, a], {"k": c}, nopose], [58, 61], K, E_ID, SHAPE)
    assert res["obj"] is a and res["projected_pixel"] == [60, 60]
    assert res["match_distance"] == pytest.approx(np.hypot(2, 1))
    assert res["selection_mode"] == "nearest_pixel_projection"
    assert cam.select_target_with_pixel([a, b], [51, 40], K, E_ID, SHAPE)["obj"] is b
    assert cam.select_target_with_pixel([c], [50, 40], K, E_ID, SHAPE) is None   # the only candidate cannot be projected
    assert cam.select_target_with_pixel([a], [np.nan, 1], K, E_ID, SHAPE) is None
    assert cam.select_target_with_pixel([], [1, 1], K, E_ID, SHAPE) is None


def test_select_by_position_nearest():
    a, b = _actor("a", (0, 0, 0)), _actor("b", (1, 0, 0))
    res = cam.select_target_with_position([a, b], [0.6, 0, 0])
    assert res["obj"] is b and res["match_distance"] == pytest.approx(0.4)
    assert res["position"] == pytest.approx([1.0, 0.0, 0.0])
    assert cam.select_target_with_position([a, b], [0.4, 0, 0])["obj"] is a
    assert cam.select_target_with_position([a], [np.inf, 0, 0]) is None
    assert cam.select_target_with_position([a], [0, 0]) is None


# --------------------------------------------------------------------------- Labels


def test_exact_label_and_action_text_mapping():
    options = [{"label": "a", "action": "pick up the cube"}, {"label": "b", "action": "put it down"},
               {"label": "", "action": "no label"}]
    assert find_exact_label_option_index("b", options) == 1
    assert find_exact_label_option_index("B", options) == -1
    assert find_exact_label_option_index(None, options) == -1
    assert map_action_text_to_option_label("put it down", options) == "b"
    assert map_action_text_to_option_label("put it down ", options) is None
    assert map_action_text_to_option_label("no label", options) is None
    assert map_action_text_to_option_label(3, options) is None


# --------------------------------------------------------------------------- OraclePlanner command parsing


OPTS = [{"label": "a"}, {"label": "b", "available": []}]


@pytest.mark.parametrize("cmd, expected", [
    ({"choice": " A ", "point": [10.4, 20.6]}, (0, [21, 10])),   # [y, x] -> [x, y], rounded
    ({"choice": "b", "point": (3, 4)}, (1, [4, 3])),
    ({"choice": "b"}, (1, None)),
    ({"choice": "b", "point": [1]}, (1, None)),
    ({"choice": "b", "point": ["y", "x"]}, (1, None)),
    ({"choice": "b", "point": [np.nan, 1]}, (1, None)),
    ({"choice": "c"}, (None, None)),
    ({"choice": ""}, (None, None)),
    ({"choice": 1}, (None, None)),
    ({"point": [1, 2]}, (None, None)),
    ("a", (None, None)),
])
def test_resolve_command(cmd, expected):
    assert OraclePlannerDemonstrationWrapper._resolve_command(None, cmd, OPTS) == expected


def _oracle(monkeypatch):
    inner = FakeTaskEnv(env_id="PickXtimes")
    evals = []
    inner.evaluate = lambda solve_complete_eval=False: evals.append(solve_complete_eval) or {}
    made = as_made(inner)
    made.reset()
    w = OraclePlannerDemonstrationWrapper(made, env_id="PickXtimes", gui_render=False)
    w.planner = SimpleNamespace(env=made)  # the real planner is built only on reset; give the collection double directly here
    return w, inner, evals


def test_oracle_unmatched_choice_returns_empty_batch(monkeypatch):
    w, inner, evals = _oracle(monkeypatch)
    obs, r, term, trunc, info = w.step({"choice": "z"})
    assert obs == {} and inner.step_calls == 0 and evals == []
    assert [o["label"] for o in info["available_multi_choices"]] == ["a", "b", "c"]


def test_oracle_target_option_requires_matching_point(monkeypatch):
    w, inner, _ = _oracle(monkeypatch)
    with pytest.raises(ValueError, match="requires"):
        w.step({"choice": "a"})
    with pytest.raises(ValueError, match="could not match"):
        w.step({"choice": "a", "point": [1, 1]})  # no camera cache -> cannot project
    assert inner.step_calls == 0


def test_oracle_executes_selected_option(monkeypatch):
    w, inner, evals = _oracle(monkeypatch)
    seen = []

    def fake_solve(env, planner, target=None, **kw):
        seen.append(target)
        planner.env.step(np.zeros(8))
        planner.env.step(np.zeros(8))

    monkeypatch.setattr(vqa_options, "solve_putonto_whenhold", fake_solve)
    obs, r, term, trunc, info = w.step({"choice": "b"})
    assert seen == [inner.target] and inner.step_calls == 2
    assert all(len(v) == 2 for v in obs.values())
    assert evals == [False, True]  # normal evaluate first, then solve_complete_eval=True
    assert "available_multi_choices" in info
