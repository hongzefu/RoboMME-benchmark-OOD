"""L4 simulation smoke: one ``make`` + ``reset`` of the official ``robomme`` package (``ee_pose``, all ``include_*`` enabled), then one unreachable ee action step.

Scale: 1 reset in this file; together with ``xhard0 16 tasks × 1 episode + xhard1-5 43 cells × 1 episode = 59`` in ``test_reset_matrix.py``,
each run of ``tests/robomme_ood/sim`` does ``59 + 1 = 60`` resets and 0 trajectory generations, covered by the user's standing authorization (plan
``1003-code-test-maintenance-todo.md`` Q8). Run only like this (pick an idle GPU with ``nvidia-smi`` first)::

    CUDA_VISIBLE_DEVICES=<idle GPU> uv run --no-sync python -m pytest tests/robomme_ood/sim --allow-sim-reset -q

Official behavior must be obtained in a process that does not import ``robomme_ood`` (importing the hard package takes over the 16 environment ids), so the real call lives in
a separate subprocess in ``official_probe.py``; here we only parse its output and assert.

Assertions:
- the subprocess did not load ``robomme_ood``, the env class comes from official ``robomme.robomme_env``, and the wrapper chain is the official
  ``FailAwareWrapper → EndeffectorDemonstrationWrapper → DemonstrationWrapper → TimeLimitWrapper → OrderEnforcing → task class``;
- depth is ``(256,256,1) int16`` per frame, camera extrinsics ``(3,4) float32`` per frame, intrinsics ``(3,3) float32``, all obs lists have equal length;
- ``available_multi_choices`` is a non-empty list, each item has exactly the three keys ``label``/``action``/``need_parameter`` with types str/str/bool;
- the unreachable ee action only does IK: ``status == "error"``, the error message comes from the IK-failure branch of ``EndeffectorDemonstrationWrapper``
  (contains ``IK failed``, not an exception caught by FailAwareWrapper), obs is empty and ``terminated`` is true.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.sim

PROBE = Path(__file__).resolve().parent / "official_probe.py"
OFFICIAL_CHAIN = [
    "FailAwareWrapper", "EndeffectorDemonstrationWrapper", "DemonstrationWrapper", "TimeLimitWrapper",
    "OrderEnforcing", "PickXtimes",
]


@pytest.fixture(scope="module")
def probe() -> dict:
    proc = subprocess.run(
        [sys.executable, str(PROBE)], capture_output=True, text=True, env=dict(os.environ), timeout=600,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("PROBE_JSON=")]
    assert proc.returncode == 0 and len(lines) == 1, (
        f"official probe exit code {proc.returncode}\nstdout tail:\n{proc.stdout[-3000:]}\nstderr tail:\n{proc.stderr[-3000:]}"
    )
    return json.loads(lines[0].removeprefix("PROBE_JSON="))


def test_official_reset_and_unreachable_ee_step(probe: dict) -> None:
    # —— official package, official chain ——
    assert probe["robomme_ood_loaded"] is False
    assert probe["env_module"].startswith("robomme.robomme_env"), probe["env_module"]
    assert probe["chain"] == OFFICIAL_CHAIN
    assert probe["status"] == "ongoing"

    # —— observations: five permanent keys + optional keys with all include_* enabled ——
    expected_obs = {
        "front_rgb_list", "wrist_rgb_list", "joint_state_list", "eef_state_list", "gripper_state_list",
        "maniskill_obs", "front_depth_list", "wrist_depth_list", "front_camera_extrinsic_list",
        "wrist_camera_extrinsic_list",
    }
    assert set(probe["obs_keys"]) == expected_obs
    n = probe["n_frames"]
    assert n >= 1
    assert set(probe["obs_lengths"].values()) == {n}, probe["obs_lengths"]
    for key in ("front_depth_list", "wrist_depth_list"):
        assert probe["obs_desc"][key] == [{"shape": [256, 256, 1], "dtype": "int16"}] * n, key
    for key in ("front_camera_extrinsic_list", "wrist_camera_extrinsic_list"):
        assert probe["obs_desc"][key] == [{"shape": [3, 4], "dtype": "float32"}] * n, key
    for key in ("front_camera_intrinsic", "wrist_camera_intrinsic"):
        assert probe["intrinsic_desc"][key] == {"shape": [3, 3], "dtype": "float32"}, key

    # —— multiple choices ——
    choices = probe["available_multi_choices"]
    assert isinstance(choices, list) and choices, choices
    for opt in choices:
        assert set(opt) == {"label", "action", "need_parameter"}, opt
        assert isinstance(opt["label"], str) and opt["label"]
        assert isinstance(opt["action"], str) and opt["action"]
        assert isinstance(opt["need_parameter"], bool)

    # —— unreachable ee action: IK only, returns error ——
    step = probe["step"]
    assert step["status"] == "error", step
    assert "IK failed" in (step["error_message"] or ""), step
    assert step["obs_is_empty"] is True
    assert step["terminated"] is True and step["truncated"] is False
