"""C14 WebSocket 传输：真实 ``PolicyServer`` handler 与真实 ``PolicyClient`` 走 127.0.0.1 回环。

覆盖：握手元数据、reset／infer 协议、策略异常的错误帧与 1011 关闭、非法请求、双向断连、
连接被拒后的重试、鉴权头；客户端面对非法回复（文本帧、坏字节）必须报错而不是静默。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import websockets
import websockets.sync.client as wsc

from challenge_interface import client as client_mod
from challenge_interface import msgpack_numpy as mn
from challenge_interface.client import PolicyClient

from challenge_support import LOOPBACK, RecordingPolicy, free_port

ACTIONS = np.arange(16, dtype=np.float32).reshape(2, 8)


def _client(handle) -> PolicyClient:
    return PolicyClient(host=LOOPBACK, port=handle.port)


def test_health_check_and_metadata_handshake(ws_server):
    h = ws_server(RecordingPolicy(), metadata={"team": "t-01", "chunk": 10})
    assert h.health_body == b"OK\n"
    c = _client(h)
    assert c.get_server_metadata() == {"team": "t-01", "chunk": 10}


def test_metadata_defaults_to_empty_dict(ws_server):
    c = _client(ws_server(RecordingPolicy()))
    assert c.get_server_metadata() == {}


def test_infer_roundtrip_delivers_arrays_intact_both_ways(ws_server):
    pol = RecordingPolicy(outputs={"actions": ACTIONS})
    c = _client(ws_server(pol))
    obs = {
        "task_goal": ["目标甲"],
        "is_first_step": True,
        "front_rgb_list": [np.full((4, 4, 3), 200, dtype=np.uint8)],
        "joint_state_list": [np.array([0.1, -0.2, 0.3], dtype=">f8")],
        "gripper_state_list": [np.array(True)],
    }
    out = c.infer(obs)
    # 回到客户端的动作块与策略返回的逐元素相同、dtype 与 shape 不变。
    assert out["actions"].dtype == np.float32
    assert out["actions"].shape == (2, 8)
    assert out["actions"].tolist() == ACTIONS.tolist()
    # 服务端策略收到的观测与客户端发出的一致（含大端与 0 维 bool）。
    (got,) = pol.inputs
    assert got["task_goal"] == ["目标甲"]
    assert got["is_first_step"] is True
    assert got["front_rgb_list"][0].shape == (4, 4, 3) and int(got["front_rgb_list"][0][0, 0, 0]) == 200
    assert got["joint_state_list"][0].dtype.str == ">f8"
    assert got["joint_state_list"][0].tolist() == [0.1, -0.2, 0.3]
    assert got["gripper_state_list"][0].shape == () and bool(got["gripper_state_list"][0]) is True
    assert pol.reset_calls == 0


def test_reset_calls_policy_reset_only_and_acknowledges(ws_server):
    pol = RecordingPolicy()
    c = _client(ws_server(pol))
    assert c.reset() == {"reset_finished": True}
    assert pol.reset_calls == 1
    assert pol.inputs == []
    # reset 后同一连接继续可用，infer 照常路由到策略。
    c.infer({"reset": False, "x": 1})
    assert pol.reset_calls == 1
    assert pol.inputs == [{"reset": False, "x": 1}]


def test_policy_exception_returns_traceback_frame_then_connection_closed(ws_server):
    pol = RecordingPolicy(raise_on_infer=1)
    h = ws_server(pol)
    c = _client(h)
    with pytest.raises(RuntimeError, match="策略故意抛错-标记串-7f3a"):
        c.infer({"a": 1})
    # 服务端已按 1011 关闭该连接：同一客户端再用必须报连接关闭，不能挂住或返回旧数据。
    with pytest.raises(websockets.ConnectionClosed):
        c.infer({"a": 2})
    assert c._ws.close_code == 1011
    # 服务本身仍在：新连接照常工作（异常只结束出错的那条连接）。
    pol._raise_on = None
    c2 = _client(h)
    assert c2.reset() == {"reset_finished": True}


@pytest.mark.parametrize(
    "payload",
    [b"\xc1\xc1\xc1", mn.packb([1, 2, 3])],
    ids=["非msgpack字节", "非dict的msgpack"],
)
def test_illegal_request_gets_error_frame_and_1011_close(ws_server, payload):
    pol = RecordingPolicy()
    h = ws_server(pol)
    with wsc.connect(f"ws://{LOOPBACK}:{h.port}", compression=None, max_size=None, open_timeout=5) as ws:
        assert mn.unpackb(ws.recv(timeout=5)) == {}
        ws.send(payload)
        err = ws.recv(timeout=5)
        assert isinstance(err, str) and "Traceback" in err
        with pytest.raises(websockets.ConnectionClosed) as ei:
            ws.recv(timeout=5)
        assert ei.value.rcvd is not None and ei.value.rcvd.code == 1011
    assert pol.inputs == [] and pol.reset_calls == 0


def test_client_side_disconnect_leaves_server_serving(ws_server):
    pol = RecordingPolicy(outputs={"actions": ACTIONS})
    h = ws_server(pol)
    c1 = _client(h)
    c1.infer({"k": 1})
    c1._ws.close()
    c2 = _client(h)
    assert c2.infer({"k": 2})["actions"].tolist() == ACTIONS.tolist()
    assert [x["k"] for x in pol.inputs] == [1, 2]


def test_server_side_disconnect_raises_on_client(raw_ws_server):
    def handler(ws):
        ws.send(mn.packb({"m": 1}))
        ws.recv()
        ws.close()  # 收到请求后不回复直接断开

    port = raw_ws_server(handler)
    c = PolicyClient(host=LOOPBACK, port=port)
    with pytest.raises(websockets.ConnectionClosed):
        c.infer({"x": 1})


def test_text_reply_is_surfaced_as_runtime_error(raw_ws_server):
    def handler(ws):
        ws.send(mn.packb({}))
        ws.recv()
        ws.send("服务端错误文本")
        ws.recv()

    c = PolicyClient(host=LOOPBACK, port=raw_ws_server(handler))
    with pytest.raises(RuntimeError, match="服务端错误文本"):
        c.infer({"x": 1})


def test_garbage_binary_reply_raises_instead_of_returning(raw_ws_server):
    def handler(ws):
        ws.send(mn.packb({}))
        ws.recv()
        ws.send(b"\xc1")  # msgpack 保留字节，非法
        ws.recv()

    c = PolicyClient(host=LOOPBACK, port=raw_ws_server(handler))
    with pytest.raises(ValueError):
        c.infer({"x": 1})


def test_reset_with_non_ack_reply_is_returned_verbatim(raw_ws_server):
    """客户端 reset 不校验回复内容：服务端回 {} 时原样返回 {}，判定交给调用方（见 phase1_eval 的有限等待用例）。"""

    def handler(ws):
        ws.send(mn.packb({}))
        msg = mn.unpackb(ws.recv())
        assert msg == {"reset": True}
        ws.send(mn.packb({}))
        ws.recv()

    c = PolicyClient(host=LOOPBACK, port=raw_ws_server(handler))
    assert c.reset() == {}


def test_api_key_header_sent_only_when_given(raw_ws_server):
    seen = []

    def handler(ws):
        seen.append(ws.request.headers.get("Authorization"))
        ws.send(mn.packb({}))
        try:
            ws.recv()
        except websockets.ConnectionClosed:
            pass

    port = raw_ws_server(handler)
    a = PolicyClient(host=LOOPBACK, port=port, api_key="k-123")
    b = PolicyClient(host=LOOPBACK, port=port)
    a._ws.close()
    b._ws.close()
    assert seen == ["Api-Key k-123", None]


def test_connection_refused_waits_then_connects_when_server_appears(ws_server, monkeypatch):
    """服务未起时客户端按重试等待；这里把等待替换为「起服务」，核对只重试一次即连上。"""
    port = free_port()
    sleeps = []

    def fake_sleep(sec):
        sleeps.append(sec)
        if len(sleeps) > 3:
            raise AssertionError("重试次数超出预期，服务已起仍连不上")
        ws_server(RecordingPolicy(), metadata={"late": True}, port=port)

    # 只替换客户端模块看到的 time，不动全局 time.sleep。
    monkeypatch.setattr(client_mod, "time", SimpleNamespace(sleep=fake_sleep))
    c = PolicyClient(host=LOOPBACK, port=port)
    assert c.get_server_metadata() == {"late": True}
    assert len(sleeps) == 1 and sleeps[0] > 0
