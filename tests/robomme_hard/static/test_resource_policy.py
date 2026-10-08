"""资源守卫的反证用例：被禁入口必须先被拒、且原生调用没有发生；子进程继承同一档。

每个用例把账本临时指到 tmp 文件，避免故意触发的违规让整场判失败。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.robomme_hard._support import resource_policy as rp

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def private_ledger(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(rp.ENV_LEDGER, str(path))
    return path


def _kinds(path: Path) -> list[str]:
    return [r["kind"] for r in rp.read_ledger(path)]


def test_guard_is_active_in_default_mode():
    assert os.environ.get(rp.ENV_MODE) == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_real_env_construction_is_refused(private_ledger):
    from mani_skill.envs.sapien_env import BaseEnv

    class _Probe(BaseEnv):
        pass

    with pytest.raises(rp.ResourcePolicyError):
        _Probe()
    assert _kinds(private_ledger) == ["native_reset"]


def test_gym_make_of_registered_task_is_refused(private_ledger):
    import gymnasium as gym

    import robomme_hard  # noqa: F401  注册 16 个任务

    with pytest.raises(rp.ResourcePolicyError):
        gym.make("PickXtimes", obs_mode="rgb", difficulty="easy", seed=0)
    assert "native_reset" in _kinds(private_ledger)


def test_cuda_init_and_weight_loading_are_refused(private_ledger, tmp_path):
    import torch

    assert torch.cuda.is_available() is False
    with pytest.raises(rp.ResourcePolicyError):
        torch.cuda.init()
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"\x00")
    with pytest.raises(rp.ResourcePolicyError):
        torch.load(weights)
    assert _kinds(private_ledger) == ["gpu_init", "weights"]


def test_non_loopback_network_refused_loopback_allowed(private_ledger):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        c = socket.create_connection(srv.getsockname(), timeout=2)
        c.close()
    finally:
        srv.close()
    s = socket.socket()
    try:
        with pytest.raises(rp.ResourcePolicyError):
            s.connect(("10.255.255.1", 9))
    finally:
        s.close()
    assert _kinds(private_ledger) == ["network"]


def test_subprocess_inherits_guard(private_ledger):
    code = "import torch, sys\ntry:\n    torch.load('x')\nexcept Exception as e:\n    print(type(e).__name__)\n"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert out.stdout.strip() == "ResourcePolicyError"
    assert _kinds(private_ledger) == ["weights"]


def test_sim_dir_requires_flag(tmp_path):
    env = dict(os.environ)
    env[rp.ENV_LEDGER] = str(tmp_path / "child-ledger.jsonl")
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/robomme_hard/sim", "-q", "-p", "no:cacheprovider"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 4, out.stdout + out.stderr  # pytest UsageError
    assert "--allow-sim-reset" in (out.stdout + out.stderr)


def test_ledger_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "l.jsonl"
    monkeypatch.setenv(rp.ENV_LEDGER, str(path))
    rp.record("gpu_init", "x")
    assert [json.loads(x)["kind"] for x in path.read_text().splitlines()] == ["gpu_init"]


def test_render_material_outside_offline_world_is_refused(private_ledger):
    import sapien

    with pytest.raises(rp.ResourcePolicyError):
        sapien.render.RenderMaterial()
    assert _kinds(private_ledger) == ["native_reset"]
