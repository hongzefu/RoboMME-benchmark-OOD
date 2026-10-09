"""挑战接口（C14）测试的公共件：回环 WebSocket 服务、原始假服务、记录型策略。

所有服务只绑 127.0.0.1，端口由内核分配（端口 0）或现取空闲端口；每个夹具在用例结束时
停服并确认线程退出，用例之间不共享任何服务实例。
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
    """向内核要一个回环空闲端口（PolicyServer.run 不暴露实际端口，只能先取再传）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((LOOPBACK, 0))
        return s.getsockname()[1]


class RecordingPolicy(Policy):
    """记录型策略：保存每次 infer 收到的输入（深拷贝），返回预设输出；可配置在第 k 次 infer 抛错。"""

    def __init__(self, outputs=None, raise_on_infer: int | None = None):
        self.inputs: list[dict] = []
        self.reset_calls = 0
        self._outputs = outputs if outputs is not None else {"actions": np.zeros((1, 8), dtype=np.float32)}
        self._raise_on = raise_on_infer

    def infer(self, inputs: dict) -> dict:
        self.inputs.append(copy.deepcopy(inputs))
        if self._raise_on is not None and len(self.inputs) == self._raise_on:
            raise ValueError("策略故意抛错-标记串-7f3a")
        return self._outputs

    def reset(self) -> None:
        self.reset_calls += 1


class WsServerHandle:
    """在后台线程的独立事件循环里运行真实 ``PolicyServer.run()``。"""

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
        except BaseException as e:  # noqa: BLE001 —— 记下来交给用例判
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
        raise TimeoutError(f"回环 WebSocket 服务 {self.port} 在 {timeout}s 内未就绪")

    def stop(self, timeout: float = 5.0) -> None:
        if self.task is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.task.cancel)
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "服务线程未在限期内退出"


@pytest.fixture
def ws_server():
    """工厂夹具：``ws_server(policy, metadata=None)`` 起真实 PolicyServer，用例结束统一停服。"""
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
    """工厂夹具：``raw_ws_server(handler)`` 起一个按脚本行事的假 WebSocket 服务（websockets 同步 API）。

    用来造真实 PolicyServer 不会给出的回复（文本帧、坏字节、直接断连），检验客户端的处理。
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
        assert not t.is_alive(), "假服务线程未在限期内退出"
