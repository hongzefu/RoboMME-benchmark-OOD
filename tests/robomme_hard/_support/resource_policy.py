"""测试资源守卫（pytest 插件，经 pyproject addopts 的 ``-p tests.robomme_hard._support.resource_policy`` 在收集前加载）。

两档：
- 默认档（日常门禁）：拒绝真实仿真场景构建、GPU 初始化、模型权重读取与非回环网络；
  CPU 替身的 reset／step 不受影响。子进程经 sitecustomize 继承同一档。
- 仿真档（``--allow-sim-reset``）：不装守卫，只允许显式选择 ``tests/robomme_hard/sim``。

违规时当场抛 ``ResourcePolicyError``，同时写入事件账本；即使用例吞掉了异常，
会话结束时账本非空也判整场失败。末尾打印判定行
``TEST_RESOURCE=PASS|FAIL native_reset=<n> gpu_init=<n> weights=<n> network=<n> violations=<n>``。

另做三条套件纪律检查（任一不满足整场失败）：收集为空、出现 xfail（本套件不登记任何 xfail）、
出现原因不以「未验证」开头的 skip。守卫是测试执行约束，不宣称能挡住任意绕过。
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

ENV_MODE = "ROBOMME_TEST_RESOURCE_POLICY"
ENV_LEDGER = "ROBOMME_TEST_RESOURCE_LEDGER"
REPO = Path(__file__).resolve().parents[3]
SITE_DIR = Path(__file__).resolve().parent / "sitecustomize_dir"
SIM_DIR = REPO / "tests" / "robomme_hard" / "sim"

KINDS = ("native_reset", "gpu_init", "weights", "network")


class ResourcePolicyError(RuntimeError):
    """日常门禁里触碰了被禁资源。"""


# ---------------------------------------------------------------- 账本


def _ledger_path() -> Path | None:
    p = os.environ.get(ENV_LEDGER)
    return Path(p) if p else None


def record(kind: str, detail: str) -> None:
    """记一条违规（父子进程都写同一份 jsonl）。"""
    path = _ledger_path()
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "detail": detail, "pid": os.getpid()}, ensure_ascii=False) + "\n")


def violate(kind: str, detail: str):
    record(kind, detail)
    raise ResourcePolicyError(f"资源守卫拒绝 {kind}: {detail}")


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- 补丁


def _patch_mani_skill(mod) -> None:
    base = getattr(mod, "BaseEnv", None)
    if base is None or getattr(base, "_resource_policy_patched", False):
        return

    def _blocked_init(self, *a, **k):  # noqa: ANN001
        violate("native_reset", f"{type(self).__name__}.__init__ 会构建真实 SAPIEN 场景")

    base.__init__ = _blocked_init
    base._resource_policy_patched = True


def _patch_torch_cuda(mod) -> None:
    if getattr(mod, "_resource_policy_patched", False):
        return

    def _blocked(*a, **k):
        violate("gpu_init", "torch.cuda 初始化")

    for name in ("init", "_lazy_init"):
        if hasattr(mod, name):
            setattr(mod, name, _blocked)
    mod.is_available = lambda: False
    mod._resource_policy_patched = True


def _patch_torch(mod) -> None:
    if getattr(mod, "_resource_policy_patched", False):
        return

    def _blocked_load(*a, **k):
        violate("weights", "torch.load 读取权重")

    mod.load = _blocked_load
    mod._resource_policy_patched = True


def _patch_safetensors(mod) -> None:
    def _blocked(*a, **k):
        violate("weights", f"{mod.__name__} 读取权重")

    for name in ("load_file", "safe_open", "load"):
        if hasattr(mod, name):
            setattr(mod, name, _blocked)


def _patch_sapien(mod) -> None:
    """sapien 本体是 C 扩展，构造器无法可靠替换；挡场景入口，并把渲染子模块里的材质与渲染系统换成拒绝工厂
    （真实 RenderMaterial 在 CPU 下会起渲染上下文、实测可段错误；离线世界夹具在自己的上下文内另行替换并还原）。"""
    for name in ("Scene",):
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            def _blocked(self, *a, _n=name, **k):  # noqa: ANN001
                violate("native_reset", f"sapien.{_n} 构造")

            cls.__init__ = _blocked
        except (TypeError, AttributeError):
            pass
    render = getattr(mod, "render", None)
    if render is None or getattr(render, "_resource_policy_patched", False):
        return
    for name in ("RenderMaterial", "RenderSystem"):
        if getattr(render, name, None) is None:
            continue

        def _blocked_factory(*a, _n=name, **k):
            violate("native_reset", f"sapien.render.{_n} 构造")

        try:
            setattr(render, name, _blocked_factory)
        except (TypeError, AttributeError):
            pass
    try:
        render._resource_policy_patched = True
    except (TypeError, AttributeError):
        pass


PATCHES = {
    "mani_skill.envs.sapien_env": _patch_mani_skill,
    "torch.cuda": _patch_torch_cuda,
    "torch": _patch_torch,
    "safetensors.torch": _patch_safetensors,
    "safetensors": _patch_safetensors,
    "sapien": _patch_sapien,
}


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        PATCHES[self.name](module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = getattr(finder, "find_spec", lambda *a: None)(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchLoader(spec.loader, name)
                return spec
        return None


_orig_connect = socket.socket.connect
_orig_create_connection = socket.create_connection
LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


def _host_of(address) -> str | None:
    if isinstance(address, (tuple, list)) and address:
        return str(address[0])
    return None  # unix socket 等


def _guarded_connect(self, address):
    host = _host_of(address)
    if host is not None and host not in LOOPBACK and not host.startswith("127."):
        violate("network", f"connect {host}")
    return _orig_connect(self, address)


def _guarded_create_connection(address, *a, **k):
    host = _host_of(address)
    if host is not None and host not in LOOPBACK and not host.startswith("127."):
        violate("network", f"create_connection {host}")
    return _orig_create_connection(address, *a, **k)


_installed = False


def install() -> None:
    """装守卫（幂等）。已导入的模块立即打补丁，未导入的在导入时打。"""
    global _installed
    if _installed:
        return
    _installed = True
    for name, fn in PATCHES.items():
        if name in sys.modules:
            fn(sys.modules[name])
    sys.meta_path.insert(0, _PatchFinder())
    socket.socket.connect = _guarded_connect
    socket.create_connection = _guarded_create_connection
    # 渲染与 CUDA 设备一律不可见，即便绕过补丁也拿不到 GPU。
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------- pytest 钩子


def pytest_addoption(parser):
    parser.addoption(
        "--allow-sim-reset",
        action="store_true",
        default=False,
        help="仿真档：不装资源守卫，只允许运行 tests/robomme_hard/sim（每次 59 + 1 = 60 次 reset，长期授权见计划 Q8）",
    )


def _args_touch_sim(args) -> bool:
    for a in args:
        p = Path(str(a).split("::")[0])
        try:
            p = p.resolve()
        except OSError:
            continue
        if p == SIM_DIR or SIM_DIR in p.parents:
            return True
    return False


def pytest_configure(config):
    import pytest

    config._rp_skips = []
    config._rp_xfails = []
    sim = config.getoption("--allow-sim-reset")
    touches_sim = _args_touch_sim(config.args)
    if touches_sim and not sim:
        raise pytest.UsageError("tests/robomme_hard/sim 只能带 --allow-sim-reset 运行（会起真实仿真 reset）")
    if sim and not touches_sim:
        raise pytest.UsageError("--allow-sim-reset 只能与显式选择的 tests/robomme_hard/sim 一起使用")
    config._rp_mode = "sim" if sim else "cpu"
    if sim:
        return
    fd, ledger = tempfile.mkstemp(prefix="resource-ledger-", suffix=".jsonl")
    os.close(fd)
    config._rp_ledger = Path(ledger)
    os.environ[ENV_MODE] = "cpu"
    os.environ[ENV_LEDGER] = ledger
    # 子进程经 sitecustomize 继承守卫。
    old = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(x for x in (str(SITE_DIR), old) if x)
    install()


def pytest_ignore_collect(collection_path, config):
    p = Path(str(collection_path)).resolve()
    if (p == SIM_DIR or SIM_DIR in p.parents) and config._rp_mode != "sim":
        return True
    return None


def pytest_runtest_logreport(report):
    if report.skipped and hasattr(report, "wasxfail"):
        _XFAILS.append(report.nodeid)
    elif report.skipped:
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reason = reason.removeprefix("Skipped: ")
        _SKIPS.append((report.nodeid, reason))
    elif report.passed and hasattr(report, "wasxfail"):
        _XFAILS.append(report.nodeid)


_SKIPS: list[tuple[str, str]] = []
_XFAILS: list[str] = []
_PROBLEMS: list[str] = []


def pytest_sessionfinish(session, exitstatus):
    cfg = session.config
    if getattr(cfg, "_rp_mode", "cpu") == "sim":
        return
    rows = read_ledger(cfg._rp_ledger)
    counts = {k: sum(1 for r in rows if r["kind"] == k) for k in KINDS}
    bad_skips = [(n, r) for n, r in _SKIPS if not r.startswith("未验证")]
    if rows:
        _PROBLEMS.append(f"资源守卫账本有 {len(rows)} 条违规")
    if bad_skips:
        _PROBLEMS.append(f"{len(bad_skips)} 个 skip 原因不以「未验证」开头：{bad_skips[:5]}")
    if _XFAILS:
        _PROBLEMS.append(f"出现未登记的 xfail：{_XFAILS[:5]}")
    if session.testscollected == 0 and not cfg.option.collectonly:
        _PROBLEMS.append("收集为空")
    cfg._rp_counts = counts
    cfg._rp_violations = len(rows)
    if _PROBLEMS and session.exitstatus == 0:
        session.exitstatus = 1
    try:
        cfg._rp_ledger.unlink()
    except OSError:
        pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if getattr(config, "_rp_mode", "cpu") == "sim":
        terminalreporter.write_line("TEST_RESOURCE=SKIP mode=sim")
        return
    counts = getattr(config, "_rp_counts", {k: 0 for k in KINDS})
    for p in _PROBLEMS:
        terminalreporter.write_line(f"RESOURCE_POLICY_PROBLEM {p}")
    not_verified = [n for n, r in _SKIPS if r.startswith("未验证")]
    ok = not _PROBLEMS
    terminalreporter.write_line(
        f"TEST_RESOURCE={'PASS' if ok else 'FAIL'} "
        + " ".join(f"{k}={counts[k]}" for k in KINDS)
        + f" violations={getattr(config, '_rp_violations', 0)} not_verified={len(not_verified)}"
    )
