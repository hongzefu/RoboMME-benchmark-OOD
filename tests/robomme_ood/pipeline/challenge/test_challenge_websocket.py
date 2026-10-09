"""C14 WebSocket transport: the real ``PolicyServer`` handler and the real ``PolicyClient`` over the 127.0.0.1 loopback.

Covers: handshake metadata, reset/infer protocol, error frame and 1011 close on policy exceptions, invalid requests, disconnects on both sides,
retry after connection refused, auth header; the client must raise on invalid replies (text frames, bad bytes) rather than stay silent.
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
        "task_goal": ["goal A"],
        "is_first_step": True,
        "front_rgb_list": [np.full((4, 4, 3), 200, dtype=np.uint8)],
        "joint_state_list": [np.array([0.1, -0.2, 0.3], dtype=">f8")],
        "gripper_state_list": [np.array(True)],
    }
    out = c.infer(obs)
    # the action chunk returned to the client matches the policy's return element-wise, dtype and shape unchanged.
    assert out["actions"].dtype == np.float32
    assert out["actions"].shape == (2, 8)
    assert out["actions"].tolist() == ACTIONS.tolist()
    # the observation the server-side policy receives matches what the client sent (including big-endian and 0-d bool).
    (got,) = pol.inputs
    assert got["task_goal"] == ["goal A"]
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
    # after reset the same connection stays usable, infer still routes to the policy.
    c.infer({"reset": False, "x": 1})
    assert pol.reset_calls == 1
    assert pol.inputs == [{"reset": False, "x": 1}]


def test_policy_exception_returns_traceback_frame_then_connection_closed(ws_server):
    pol = RecordingPolicy(raise_on_infer=1)
    h = ws_server(pol)
    c = _client(h)
    with pytest.raises(RuntimeError, match="policy-deliberate-error-marker-7f3a"):
        c.infer({"a": 1})
    # the server has closed the connection with 1011: reusing the same client must report connection closed, not hang or return stale data.
    with pytest.raises(websockets.ConnectionClosed):
        c.infer({"a": 2})
    assert c._ws.close_code == 1011
    # the server itself is still up: new connections work normally (the exception only ends the failing connection).
    pol._raise_on = None
    c2 = _client(h)
    assert c2.reset() == {"reset_finished": True}


@pytest.mark.parametrize(
    "payload",
    [b"\xc1\xc1\xc1", mn.packb([1, 2, 3])],
    ids=["non_msgpack_bytes", "non_dict_msgpack"],
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
        ws.close()  # disconnect right after receiving the request, without replying

    port = raw_ws_server(handler)
    c = PolicyClient(host=LOOPBACK, port=port)
    with pytest.raises(websockets.ConnectionClosed):
        c.infer({"x": 1})


def test_text_reply_is_surfaced_as_runtime_error(raw_ws_server):
    def handler(ws):
        ws.send(mn.packb({}))
        ws.recv()
        ws.send("server error text")
        ws.recv()

    c = PolicyClient(host=LOOPBACK, port=raw_ws_server(handler))
    with pytest.raises(RuntimeError, match="server error text"):
        c.infer({"x": 1})


def test_garbage_binary_reply_raises_instead_of_returning(raw_ws_server):
    def handler(ws):
        ws.send(mn.packb({}))
        ws.recv()
        ws.send(b"\xc1")  # msgpack reserved byte, invalid
        ws.recv()

    c = PolicyClient(host=LOOPBACK, port=raw_ws_server(handler))
    with pytest.raises(ValueError):
        c.infer({"x": 1})


def test_reset_with_non_ack_reply_is_returned_verbatim(raw_ws_server):
    """Client reset does not validate the reply: when the server returns {} it is returned as is, leaving the judgment to the caller (see the bounded-wait test in phase1_eval)."""

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
    """When the server is not up the client waits and retries; here the wait is replaced with "start the server" to check that exactly one retry connects."""
    port = free_port()
    sleeps = []

    def fake_sleep(sec):
        sleeps.append(sec)
        if len(sleeps) > 3:
            raise AssertionError("more retries than expected; server is up but still cannot connect")
        ws_server(RecordingPolicy(), metadata={"late": True}, port=port)

    # replace only the time seen by the client module, not the global time.sleep.
    monkeypatch.setattr(client_mod, "time", SimpleNamespace(sleep=fake_sleep))
    c = PolicyClient(host=LOOPBACK, port=port)
    assert c.get_server_metadata() == {"late": True}
    assert len(sleeps) == 1 and sleeps[0] > 0
