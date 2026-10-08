"""官方 ``robomme`` 包的 1 次 make + reset + 1 步不可达 ee 动作；由 ``test_official_one_reset.py`` 在独立子进程里调用。

必须单独起进程：``robomme_hard`` 导入时以 ``override=True`` 接管 16 个环境 id，与它同进程的 ``gym.make``
只会拿到 hard 包的类（见 ``robomme_hard/__init__.py``）；``tests/robomme_hard/sim`` 的另一个文件在收集期就导入了 ``robomme_hard``。

只输出一行 JSON（以 ``PROBE_JSON=`` 开头）描述观测，断言放在父进程测试里。文件名不以 ``test_`` 开头，不被收集。
"""
from __future__ import annotations

import json
import sys

import numpy as np

TASK = "PickXtimes"
EPISODE = 0
#: 远离工作空间的末端位姿（米）：IK 必然无解，EndeffectorDemonstrationWrapper 直接返回 status="error"，不推进仿真
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
            "robomme_hard_loaded": "robomme_hard" in sys.modules,
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
