"""Shared pieces for the challenge interface (C14) tests: loopback WebSocket server, raw fake server, recording policy.

All servers bind only to 127.0.0.1, ports are kernel-assigned (port 0) or a freshly picked free port; each fixture stops its server at test end
and confirms the thread exits; no server instance is shared across tests.
"""
from __future__ import annotations

import asyncio
import copy
import http.client
import socket
import threading
import time

import numpy as np
import pytest

from challenge_interface import server as server_mod
from challenge_interface.policy import Policy

LOOPBACK = "127.0.0.1"


def free_port() -> int:
    """Ask the kernel for a free loopback port (PolicyServer.run does not expose the actual port, so pick one first and pass it in)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOOPBACK, 0))
        return s.getsockname()[1]


class RecordingPolicy(Policy):
    """Recording policy: stores the input of every infer call (deep copy) and returns a preset output; can be configured to raise on the k-th infer."""

    def __init__(self, outputs=None, raise_on_infer: int | None = None):
        self.inputs: list[dict] = []
        self.reset_calls = 0
        self._outputs = outputs if outputs is not None else {"actions": np.zeros((1, 8), dtype=np.float32)}
        self._raise_on = raise_on_infer

    def infer(self, inputs: dict) -> dict:
        self.inputs.append(copy.deepcopy(inputs))
        if self._raise_on is not None and len(self.inputs) == self._raise_on:
            raise ValueError("policy-deliberate-error-marker-7f3a")
        return self._outputs

    def reset(self) -> None:
        self.reset_calls += 1


class WsServerHandle:
    """Run the real ``PolicyServer.run()`` in a separate event loop on a background thread."""

    def __init__(self, policy: Policy, metadata: dict | None = None, port: int | None = None):
        self.port = port if port is not None else free_port()
        self.server = server_mod.PolicyServer(policy, host=LOOPBACK, port=self.port, metadata=metadata)
        self.loop = asyncio.new_event_loop()
        self.task = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.task = self.loop.create_task(self.server.run())
        try:
            self.loop.run_until_complete(self.task)
        except asyncio.CancelledError:
            pass
        except BaseException as e:  # noqa: BLE001 -- record it for the test to judge
            self.error = e
        finally:
            self.loop.close()

    def start(self, timeout: float = 5.0) -> "WsServerHandle":
        self.thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None:
                raise self.error
            try:
                conn = http.client.HTTPConnection(LOOPBACK, self.port, timeout=0.5)
                conn.request("GET", "/healthz")
                resp = conn.getresponse()
                body = resp.read()
                conn.close()
                if resp.status == 200:
                    self.health_body = body
                    return self
            except OSError:
                time.sleep(0.01)
        raise TimeoutError(f"loopback WebSocket server {self.port} not ready within {timeout}s")

    def stop(self, timeout: float = 5.0) -> None:
        if self.task is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.task.cancel)
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "server thread did not exit in time"


@pytest.fixture
def ws_server():
    """Factory fixture: ``ws_server(policy, metadata=None)`` starts a real PolicyServer; all are stopped at test end."""
    handles: list[WsServerHandle] = []

    def _make(policy: Policy, metadata: dict | None = None, port: int | None = None) -> WsServerHandle:
        h = WsServerHandle(policy, metadata, port).start()
        handles.append(h)
        return h

    yield _make
    for h in handles:
        h.stop()


@pytest.fixture
def raw_ws_server():
    """Factory fixture: ``raw_ws_server(handler)`` starts a scripted fake WebSocket server (websockets sync API).

    Used to produce replies the real PolicyServer never gives (text frames, bad bytes, abrupt disconnects) to check client handling.
    """
    import websockets.sync.server as wss

    servers = []

    def _make(handler):
        srv = wss.serve(handler, LOOPBACK, 0, compression=None, max_size=None)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        servers.append((srv, t))
        return srv.socket.getsockname()[1]

    yield _make
    for srv, t in servers:
        srv.shutdown()
        t.join(5)
        assert not t.is_alive(), "fake server thread did not exit in time"
