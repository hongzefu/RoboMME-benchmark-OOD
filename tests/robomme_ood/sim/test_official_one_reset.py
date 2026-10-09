"""L4 仿真冒烟：官方 ``robomme`` 包 1 次 ``make`` + ``reset``（``ee_pose``，全部 ``include_*`` 打开），再发 1 步不可达 ee 动作。

规模：本文件 1 次 reset；连同 ``test_reset_matrix.py`` 的 ``xhard0 16 任务 × 1 局 + xhard1～5 共 43 格 × 1 局 = 59``
次，每次运行 ``tests/robomme_ood/sim`` 共 ``59 + 1 = 60`` 次 reset、0 条轨迹生成，属用户长期授权（计划
``1003-code-test-maintenance-todo.md`` 的 Q8）。只能这样运行（先 ``nvidia-smi`` 选空闲卡）::

    CUDA_VISIBLE_DEVICES=<空闲卡> uv run --no-sync python -m pytest tests/robomme_ood/sim --allow-sim-reset -q

官方行为必须在不导入 ``robomme_ood`` 的进程里取（hard 包导入即接管 16 个环境 id），因此真实调用放在
``official_probe.py`` 的独立子进程里，这里只解析其输出并断言。

断言：
- 子进程未加载 ``robomme_ood``，环境类来自官方 ``robomme.robomme_env``，包装链为官方的
  ``FailAwareWrapper → EndeffectorDemonstrationWrapper → DemonstrationWrapper → TimeLimitWrapper → OrderEnforcing → 任务类``；
- 深度图逐帧 ``(256,256,1) int16``，相机外参逐帧 ``(3,4) float32``，内参 ``(3,3) float32``，全部 obs 列表等长；
- ``available_multi_choices`` 为非空列表，每项恰有 ``label``／``action``／``need_parameter`` 三键，类型为 str／str／bool；
- 不可达 ee 动作只做 IK：``status == "error"``、错误信息来自 ``EndeffectorDemonstrationWrapper`` 的 IK 失败分支
  （含 ``IK failed``，不是 FailAwareWrapper 兜住的异常），obs 为空、``terminated`` 为真。
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
        f"官方探针退出码 {proc.returncode}\nstdout 末尾：\n{proc.stdout[-3000:]}\nstderr 末尾：\n{proc.stderr[-3000:]}"
    )
    return json.loads(lines[0].removeprefix("PROBE_JSON="))


def test_official_reset_and_unreachable_ee_step(probe: dict) -> None:
    # —— 官方包、官方链 ——
    assert probe["robomme_ood_loaded"] is False
    assert probe["env_module"].startswith("robomme.robomme_env"), probe["env_module"]
    assert probe["chain"] == OFFICIAL_CHAIN
    assert probe["status"] == "ongoing"

    # —— 观测：五个常驻键 + 全部 include_* 打开后的可选键 ——
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

    # —— 多选项 ——
    choices = probe["available_multi_choices"]
    assert isinstance(choices, list) and choices, choices
    for opt in choices:
        assert set(opt) == {"label", "action", "need_parameter"}, opt
        assert isinstance(opt["label"], str) and opt["label"]
        assert isinstance(opt["action"], str) and opt["action"]
        assert isinstance(opt["need_parameter"], bool)

    # —— 不可达 ee 动作：只做 IK，返回 error ——
    step = probe["step"]
    assert step["status"] == "error", step
    assert "IK failed" in (step["error_message"] or ""), step
    assert step["obs_is_empty"] is True
    assert step["terminated"] is True and step["truncated"] is False
