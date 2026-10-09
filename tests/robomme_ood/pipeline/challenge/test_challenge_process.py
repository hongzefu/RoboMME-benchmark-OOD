"""C14 slow tests: the policy server runs in a real subprocess (the subprocess inherits the resource guard via sitecustomize).

- ``PolicyServer(DummyPolicy)`` in a subprocess: handshake, inference, reset; after the server process is killed the client must report connection closed, not hang;
- the ``scripts/deploy.py`` entry (needs flask; recorded as "Unverified" if missing).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import websockets

from challenge_interface.client import PolicyClient
from challenge_interface.policy import DummyPolicy
from challenge_interface.scripts.phase1_eval import EXPECTED_ACTION_SHAPES

from challenge_support import LOOPBACK, free_port

REPO = Path(__file__).resolve().parents[4]

pytestmark = pytest.mark.slow


def _wait_port(port: int, proc: subprocess.Popen, timeout: float = 20.0) -> None:
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise AssertionError(f"server subprocess exited early rc={proc.returncode}\n{out}")
        try:
            with socket.create_connection((LOOPBACK, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"server subprocess port {port} not ready within {timeout}s")


def _spawn(code: str, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(x for x in (env.get("PYTHONPATH", ""), str(REPO)) if x)
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), str(port)],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)


SERVER_CODE = """
    import sys
    from challenge_interface.policy import DummyPolicy
    from challenge_interface.server import PolicyServer
    PolicyServer(DummyPolicy(), host="127.0.0.1", port=int(sys.argv[1]), metadata={"pid": "child"}).serve_forever()
"""


def test_subprocess_server_roundtrip_then_death_is_detected():
    port = free_port()
    proc = _spawn(SERVER_CODE, port)
    try:
        _wait_port(port, proc)
        c = PolicyClient(host=LOOPBACK, port=port)
        assert c.get_server_metadata() == {"pid": "child"}
        assert c.reset() == {"reset_finished": True}
        out = c.infer({"is_first_step": True, "front_rgb_list": [np.zeros((2, 2, 3), np.uint8)] * 3})
        assert out["actions"].shape == (DummyPolicy().chunk_size, *EXPECTED_ACTION_SHAPES["joint_angle"])
        # kill the server process: the client's next inference must report connection closed within bounded time.
        proc.send_signal(signal.SIGKILL)
        proc.wait(5)
        result = {}

        def _call():
            try:
                c.infer({"is_first_step": False, "front_rgb_list": []})
                result["ok"] = True
            except Exception as e:  # noqa: BLE001
                result["err"] = e

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(10)
        assert not t.is_alive(), "server process is dead but client inference still hangs"
        assert isinstance(result.get("err"), (websockets.ConnectionClosed, OSError)), result
    finally:
        _stop(proc)


def test_deploy_entry_serves_websocket():
    try:
        import flask  # noqa: F401
    except ImportError:
        pytest.skip("Unverified: flask missing (deploy.py imports server_http at top level; server dependency group not installed)")
    port = free_port()
    code = """
        import sys
        sys.argv = ["deploy", "--transport", "websocket", "--host", "127.0.0.1", "--port", sys.argv[1]]
        from challenge_interface.scripts import deploy
        deploy.main()
    """
    proc = _spawn(code, port)
    try:
        _wait_port(port, proc)
        c = PolicyClient(host=LOOPBACK, port=port)
        assert c.reset() == {"reset_finished": True}
        shape = c.infer({"is_first_step": True, "front_rgb_list": [0]})["actions"].shape
        assert shape == (DummyPolicy().chunk_size, *EXPECTED_ACTION_SHAPES["joint_angle"])
    finally:
        _stop(proc)
