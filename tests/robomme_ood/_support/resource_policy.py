"""Test resource guard (pytest plugin, loaded before collection via pyproject addopts ``-p tests.robomme_ood._support.resource_policy``).

Two modes:
- Default mode (daily gate): rejects real simulation scene construction, GPU initialization, model weight loading and non-loopback network;
  reset/step of CPU stand-ins are unaffected. Subprocesses inherit the same mode via sitecustomize.
- Simulation mode (``--allow-sim-reset``): no guard installed; only an explicit selection of ``tests/robomme_ood/sim`` is allowed.

A violation raises ``ResourcePolicyError`` immediately and is also written to the event ledger; even if a test swallows the exception,
a non-empty ledger at session end fails the whole session. The verdict line printed at the end is
``TEST_RESOURCE=PASS|FAIL native_reset=<n> gpu_init=<n> weights=<n> network=<n> violations=<n>``.

Three extra suite-discipline checks (any failure fails the whole session): empty collection, any xfail (this suite registers none),
any skip whose reason does not start with "Unverified". The guard is a test-execution constraint and does not claim to stop arbitrary bypasses.
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
SIM_DIR = REPO / "tests" / "robomme_ood" / "sim"

KINDS = ("native_reset", "gpu_init", "weights", "network")


class ResourcePolicyError(RuntimeError):
    """A forbidden resource was touched in the daily gate."""


# ---------------------------------------------------------------- ledger


def _ledger_path() -> Path | None:
    p = os.environ.get(ENV_LEDGER)
    return Path(p) if p else None


def record(kind: str, detail: str) -> None:
    """Record one violation (parent and child processes write the same jsonl)."""
    path = _ledger_path()
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "detail": detail, "pid": os.getpid()}, ensure_ascii=False) + "\n")


def violate(kind: str, detail: str):
    record(kind, detail)
    raise ResourcePolicyError(f"resource guard rejected {kind}: {detail}")


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- patches


def _patch_mani_skill(mod) -> None:
    base = getattr(mod, "BaseEnv", None)
    if base is None or getattr(base, "_resource_policy_patched", False):
        return

    def _blocked_init(self, *a, **k):  # noqa: ANN001
        violate("native_reset", f"{type(self).__name__}.__init__ would build a real SAPIEN scene")

    base.__init__ = _blocked_init
    base._resource_policy_patched = True


def _patch_torch_cuda(mod) -> None:
    if getattr(mod, "_resource_policy_patched", False):
        return

    def _blocked(*a, **k):
        violate("gpu_init", "torch.cuda initialization")

    for name in ("init", "_lazy_init"):
        if hasattr(mod, name):
            setattr(mod, name, _blocked)
    mod.is_available = lambda: False
    mod._resource_policy_patched = True


def _patch_torch(mod) -> None:
    if getattr(mod, "_resource_policy_patched", False):
        return

    def _blocked_load(*a, **k):
        violate("weights", "torch.load reading weights")

    mod.load = _blocked_load
    mod._resource_policy_patched = True


def _patch_safetensors(mod) -> None:
    def _blocked(*a, **k):
        violate("weights", f"{mod.__name__} reading weights")

    for name in ("load_file", "safe_open", "load"):
        if hasattr(mod, name):
            setattr(mod, name, _blocked)


def _patch_sapien(mod) -> None:
    """sapien itself is a C extension and its constructors cannot be replaced reliably; block the scene entry and replace materials
    and render systems in the render submodule with rejecting factories (a real RenderMaterial on CPU starts a render context and was
    observed to segfault; the offline-world fixture does its own replacement and restore within its own context)."""
    for name in ("Scene",):
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        try:
            def _blocked(self, *a, _n=name, **k):  # noqa: ANN001
                violate("native_reset", f"sapien.{_n} construction")

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
            violate("native_reset", f"sapien.render.{_n} construction")

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
    return None  # unix sockets etc.


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
    """Install the guard (idempotent). Already-imported modules are patched immediately; others are patched on import."""
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
    # Render and CUDA devices are always hidden, so a GPU is unreachable even if the patches are bypassed.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------- pytest hooks


def pytest_addoption(parser):
    parser.addoption(
        "--allow-sim-reset",
        action="store_true",
        default=False,
        help="simulation mode: no resource guard; only tests/robomme_ood/sim may run (59 + 1 = 60 resets per run; standing authorization, see plan Q8)",
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
        raise pytest.UsageError("tests/robomme_ood/sim can only run with --allow-sim-reset (it performs real simulation resets)")
    if sim and not touches_sim:
        raise pytest.UsageError("--allow-sim-reset can only be used with an explicit selection of tests/robomme_ood/sim")
    config._rp_mode = "sim" if sim else "cpu"
    if sim:
        return
    fd, ledger = tempfile.mkstemp(prefix="resource-ledger-", suffix=".jsonl")
    os.close(fd)
    config._rp_ledger = Path(ledger)
    os.environ[ENV_MODE] = "cpu"
    os.environ[ENV_LEDGER] = ledger
    # Subprocesses inherit the guard via sitecustomize.
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
    bad_skips = [(n, r) for n, r in _SKIPS if not r.startswith("Unverified")]
    if rows:
        _PROBLEMS.append(f"resource guard ledger has {len(rows)} violations")
    if bad_skips:
        _PROBLEMS.append(f"{len(bad_skips)} skip reasons do not start with 'Unverified': {bad_skips[:5]}")
    if _XFAILS:
        _PROBLEMS.append(f"unregistered xfail found: {_XFAILS[:5]}")
    if session.testscollected == 0 and not cfg.option.collectonly:
        _PROBLEMS.append("empty collection")
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
    not_verified = [n for n, r in _SKIPS if r.startswith("Unverified")]
    ok = not _PROBLEMS
    terminalreporter.write_line(
        f"TEST_RESOURCE={'PASS' if ok else 'FAIL'} "
        + " ".join(f"{k}={counts[k]}" for k in KINDS)
        + f" violations={getattr(config, '_rp_violations', 0)} not_verified={len(not_verified)}"
    )
