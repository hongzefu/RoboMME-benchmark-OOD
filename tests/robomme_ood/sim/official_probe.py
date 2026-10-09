"""One make + reset + one unreachable ee action step on the official ``robomme`` package; called by ``test_official_one_reset.py`` in a separate subprocess.

Must run in its own process: importing ``robomme_ood`` takes over the 16 environment ids with ``override=True``, so ``gym.make`` in the same process
only gets the hard package's classes (see ``robomme_ood/__init__.py``); the other file in ``tests/robomme_ood/sim`` already imports ``robomme_ood`` at collection time.

Prints exactly one JSON line (prefixed ``PROBE_JSON=``) describing the observations; assertions live in the parent-process test. The file name does not start with ``test_`` so it is not collected.
"""
from __future__ import annotations

import json
import sys

import numpy as np

TASK = "PickXtimes"
EPISODE = 0
#: End-effector pose far outside the workspace (meters): IK is guaranteed to fail, EndeffectorDemonstrationWrapper returns status="error" directly without stepping the simulation
UNREACHABLE_ACTION = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 1.0]


def _desc(value):
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def main() -> None:
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(env_id=TASK, dataset="test", action_space="ee_pose")
    env = builder.make_env_for_episode(
        EPISODE,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )
    try:
        obs, info = env.reset()
        chain, e = [], env
        while hasattr(e, "env"):
            chain.append(type(e).__name__)
            e = e.env
        out = {
            "robomme_ood_loaded": "robomme_ood" in sys.modules,
            "env_module": type(env.unwrapped).__module__,
            "chain": chain + [type(e).__name__],
            "obs_keys": sorted(obs),
            "info_keys": sorted(info),
            "status": info.get("status"),
            "n_frames": len(obs["front_rgb_list"]),
            "obs_lengths": {k: len(v) for k, v in obs.items() if isinstance(v, list)},
            "obs_desc": {
                k: [_desc(x) for x in obs[k]]
                for k in ("front_depth_list", "wrist_depth_list", "front_camera_extrinsic_list",
                          "wrist_camera_extrinsic_list")
            },
            "intrinsic_desc": {k: _desc(info[k]) for k in ("front_camera_intrinsic", "wrist_camera_intrinsic")},
            "available_multi_choices": info.get("available_multi_choices"),
        }
        step_obs, _reward, terminated, truncated, step_info = env.step(np.asarray(UNREACHABLE_ACTION))
        out["step"] = {
            "status": step_info.get("status"),
            "error_message": step_info.get("error_message"),
            "obs_is_empty": isinstance(step_obs, dict) and not step_obs,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    finally:
        env.close()
    print("PROBE_JSON=" + json.dumps(out, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
