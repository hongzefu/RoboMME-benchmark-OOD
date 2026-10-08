"""C14 HTTP 传输。

两组：
- 真实 ``PolicyHTTPClient`` 对一个按脚本行事的 stdlib 假服务（回环）：请求路由、请求体、头、
  超时、HTTP 错误、非法回复体；
- 真实 ``PolicyHTTPServer``（Flask）走回环 + 真实客户端端到端；缺 flask 时整组记「未验证」。
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import requests

from challenge_interface import msgpack_numpy as mn
from challenge_interface.client_http import PolicyHTTPClient

from challenge_support import LOOPBACK, RecordingPolicy

ACTIONS = np.arange(14, dtype=np.float32).reshape(2, 7)


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """回环请求不得经代理（本机或集群节点可能设了 http_proxy）。"""
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def stub_http():
    """工厂夹具：``stub_http(routes)``，routes 为 {(方法, 路径): 回调(请求体, 头) -> (状态码, 回复体字节)}。"""
    started = []

    def _make(routes):
        seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 静音
                pass

            def _handle(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                seen.append((method, self.path, body, dict(self.headers)))
                fn = routes.get((method, self.path))
                if fn is None:
                    status, payload = 404, b"no route"
                else:
                    status, payload = fn(body, self.headers)
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # 超时用例里客户端已先断开，属预期

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        srv = ThreadingHTTPServer((LOOPBACK, 0), H)
        srv.daemon_threads = True
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        started.append((srv, t))
        return srv.server_address[1], seen

    yield _make
    for srv, t in started:
        srv.shutdown()
        srv.server_close()
        t.join(5)
        assert not t.is_alive()


def _meta_route(meta=None):
    return {("GET", "/metadata"): lambda b, h: (200, mn.packb(meta or {"v": 1}))}


def test_client_fetches_metadata_on_construction(stub_http):
    port, seen = stub_http(_meta_route({"team": "x"}))
    c = PolicyHTTPClient(host=LOOPBACK, port=port, timeout=5)
    assert c.get_server_metadata() == {"team": "x"}
    assert [(m, p) for m, p, *_ in seen] == [("GET", "/metadata")]


def test_infer_posts_msgpack_body_and_decodes_reply(stub_http):
    routes = _meta_route()
    routes[("POST", "/infer")] = lambda body, h: (200, mn.packb({"actions": ACTIONS}))
    port, seen = stub_http(routes)
    c = PolicyHTTPClient(host=LOOPBACK, port=port, api_key="k-9", timeout=5)
    obs = {"front_rgb_list": [np.ones((2, 2, 3), dtype=np.uint8)], "is_first_step": False}
    out = c.infer(obs)
    assert out["actions"].dtype == np.float32 and out["actions"].tolist() == ACTIONS.tolist()
    method, path, body, headers = seen[-1]
    assert (method, path) == ("POST", "/infer")
    sent = mn.unpackb(body)
    assert sent["is_first_step"] is False and sent["front_rgb_list"][0].shape == (2, 2, 3)
    assert headers["Content-Type"] == "application/msgpack"
    assert headers["Authorization"] == "Api-Key k-9"


def test_reset_posts_reset_flag_to_reset_route(stub_http):
    routes = _meta_route()
    routes[("POST", "/reset")] = lambda body, h: (200, mn.packb({"reset_finished": True}))
    port, seen = stub_http(routes)
    c = PolicyHTTPClient(host=LOOPBACK, port=port, timeout=5)
    assert c.reset() == {"reset_finished": True}
    method, path, body, headers = seen[-1]
    assert (method, path) == ("POST", "/reset")
    assert mn.unpackb(body) == {"reset": True}
    assert "Authorization" not in headers


@pytest.mark.parametrize("status", [400, 500, 503])
def test_http_error_status_raises(stub_http, status):
    routes = _meta_route()
    routes[("POST", "/infer")] = lambda body, h: (status, b"server traceback ...")
    port, _ = stub_http(routes)
    c = PolicyHTTPClient(host=LOOPBACK, port=port, timeout=5)
    with pytest.raises(requests.HTTPError):
        c.infer({"x": 1})


def test_metadata_error_fails_construction(stub_http):
    port, _ = stub_http({("GET", "/metadata"): lambda b, h: (500, b"boom")})
    with pytest.raises(requests.HTTPError):
        PolicyHTTPClient(host=LOOPBACK, port=port, timeout=5)


def test_slow_server_hits_client_timeout(stub_http):
    def slow(body, h):
        time.sleep(3.0)
        return 200, mn.packb({"actions": ACTIONS})

    routes = _meta_route()
    routes[("POST", "/infer")] = slow
    port, _ = stub_http(routes)
    c = PolicyHTTPClient(host=LOOPBACK, port=port, timeout=0.15)
    t0 = time.monotonic()
    with pytest.raises(requests.Timeout):
        c.infer({"x": 1})
    # 有限时间内结束（上限 2 s，给高负载留余量），而不是等到服务端 3 s 后回复。
    assert time.monotonic() - t0 < 2.0


def test_garbage_200_body_raises(stub_http):
    routes = _meta_route()
    routes[("POST", "/infer")] = lambda body, h: (200, b"\xc1")
    port, _ = stub_http(routes)
    c = PolicyHTTPClient(host=LOOPBACK, port=port, timeout=5)
    with pytest.raises(ValueError):
        c.infer({"x": 1})


# ---------------------------------------------------------------- 真实 Flask 服务端


def _flask_server_or_skip():
    try:
        import flask  # noqa: F401
        from werkzeug.serving import make_server  # noqa: F401
    except ImportError:
        pytest.skip("未验证：缺 flask（server 依赖组未装进当前 venv）")
    from challenge_interface.server_http import PolicyHTTPServer

    return PolicyHTTPServer


@pytest.fixture
def flask_server():
    started = []

    def _make(policy, metadata=None):
        PolicyHTTPServer = _flask_server_or_skip()
        from werkzeug.serving import make_server

        srv_obj = PolicyHTTPServer(policy, host=LOOPBACK, port=0, metadata=metadata)
        srv = make_server(LOOPBACK, 0, srv_obj._app, threaded=True)
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        t.start()
        started.append((srv, t))
        return srv.server_port

    yield _make
    for srv, t in started:
        srv.shutdown()
        t.join(5)
        assert not t.is_alive()


def test_flask_server_end_to_end(flask_server):
    pol = RecordingPolicy(outputs={"actions": ACTIONS})
    port = flask_server(pol, metadata={"m": 2})
    c = PolicyHTTPClient(host=LOOPBACK, port=port, timeout=5)
    assert c.get_server_metadata() == {"m": 2}
    assert c.reset() == {"reset_finished": True}
    out = c.infer({"a": np.array([1, 2], dtype=">i4")})
    assert out["actions"].tolist() == ACTIONS.tolist()
    assert pol.reset_calls == 1
    assert pol.inputs[0]["a"].dtype.str == ">i4"
    assert requests.get(f"http://{LOOPBACK}:{port}/healthz", timeout=5).text == "OK\n"


def test_flask_server_policy_exception_is_500_with_traceback(flask_server):
    port = flask_server(RecordingPolicy(raise_on_infer=1))
    r = requests.post(f"http://{LOOPBACK}:{port}/infer", data=mn.packb({"a": 1}), timeout=5)
    assert r.status_code == 500 and "策略故意抛错-标记串-7f3a" in r.text


def test_flask_server_empty_body_is_400(flask_server):
    pol = RecordingPolicy()
    port = flask_server(pol)
    r = requests.post(f"http://{LOOPBACK}:{port}/infer", data=b"", timeout=5)
    assert r.status_code == 400
    assert pol.inputs == []


def test_flask_server_illegal_body_is_500(flask_server):
    pol = RecordingPolicy()
    port = flask_server(pol)
    r = requests.post(f"http://{LOOPBACK}:{port}/infer", data=b"\xc1\xc1", timeout=5)
    assert r.status_code == 500
    assert pol.inputs == []
