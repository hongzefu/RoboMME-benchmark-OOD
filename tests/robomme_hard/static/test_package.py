"""L0（慢）：打包与来源（C18）。

构建 wheel → 在 tmp 新建 uv 环境 → 非 editable 安装（``--no-deps``，依赖借当前解释器的 site-packages 只读挂在
``PYTHONPATH`` 上；该目录里 editable 安装的 ``.pth`` 不会被处理，因此 ``robomme``／``robomme_hard`` 只能来自 tmp 环境）→
在仓库外的 cwd 里导入，核：

- 两个包的模块实际位置在 tmp 环境的 site-packages 里，tmp 环境里没有指回仓库的 ``.pth``；
- wheel 内文件集合 = git 跟踪的 ``src/robomme``、``src/robomme_hard`` 文件集合，安装后逐字节等于源码树（含规格 jsonl、
  元数据 json、``UPSTREAM.json``）；
- dist-info 的 Name／Version 与 ``pyproject.toml`` 一致，且不是 editable 安装。

网络受限：构建、建环境、安装一律 ``--offline``；装不了就 ``pytest.skip("未验证：<原因>")``，不记 PASS。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path

import pytest

from tests.robomme_hard._support.loaders import REPO
from tests.robomme_hard._support.resource_policy import SITE_DIR

pytestmark = pytest.mark.slow

PACKAGES = ("robomme", "robomme_hard")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=240, **kw)


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("未验证：PATH 里没有 uv")
    return uv


def _tracked_sources() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *(f"src/{p}" for p in PACKAGES)],
                         cwd=REPO, check=True, capture_output=True).stdout
    return sorted(x.decode() for x in out.split(b"\0") if x)


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    uv = _uv()
    tmp = tmp_path_factory.mktemp("wheel")
    dist, venv, cwd = tmp / "dist", tmp / "venv", tmp / "outside"
    cwd.mkdir()
    env = dict(os.environ)
    # 构建与安装不能被当前项目环境变量带偏到主 .venv。
    for k in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "PYTHONPATH"):
        env.pop(k, None)
    r = _run([uv, "build", "--wheel", "--offline", "--out-dir", str(dist)], cwd=REPO, env=env)
    if r.returncode != 0:
        pytest.skip(f"未验证：离线构建 wheel 失败：{r.stderr.strip()[-300:]}")
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1, wheels
    r = _run([uv, "venv", "--offline", "--python", sys.executable, str(venv)], cwd=cwd, env=env)
    if r.returncode != 0:
        pytest.skip(f"未验证：离线建 tmp 环境失败：{r.stderr.strip()[-300:]}")
    py = venv / "bin" / "python"
    r = _run([uv, "pip", "install", "--offline", "--no-deps", "--python", str(py), str(wheels[0])], cwd=cwd, env=env)
    if r.returncode != 0:
        pytest.skip(f"未验证：离线安装 wheel 失败：{r.stderr.strip()[-300:]}")
    site = Path(_run([str(py), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                     cwd=cwd, env={"PATH": env.get("PATH", "")}).stdout.strip())
    return {"wheel": wheels[0], "py": py, "site": site, "cwd": cwd, "env": env}


def test_wheel_contents_equal_tracked_sources(installed):
    with zipfile.ZipFile(installed["wheel"]) as z:
        names = {n for n in z.namelist() if ".dist-info/" not in n}
    assert names == {rel.removeprefix("src/") for rel in _tracked_sources()}


def test_installed_files_byte_identical_to_source(installed):
    site = installed["site"]
    bad = [rel for rel in _tracked_sources()
           if (site / rel.removeprefix("src/")).read_bytes() != (REPO / rel).read_bytes()]
    assert bad == []
    # 规格与元数据资源确实在包内（不是只装了 .py）。
    assert list((site / "robomme_hard" / "env_metadata" / "ood").rglob("specs.jsonl"))
    assert list((site / "robomme" / "env_metadata").rglob("*_metadata.json"))
    assert (site / "robomme_hard" / "UPSTREAM.json").is_file()


def test_dist_info_metadata_and_not_editable(installed):
    site = installed["site"]
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    infos = list(site.glob("*.dist-info"))
    mine = [d for d in infos if d.name.startswith(f"{project['name']}-")]
    assert len(mine) == 1, infos
    meta = (mine[0] / "METADATA").read_text(encoding="utf-8").splitlines()
    assert f"Name: {project['name']}" in meta
    assert f"Version: {project['version']}" in meta
    direct = mine[0] / "direct_url.json"
    if direct.exists():
        assert not json.loads(direct.read_text()).get("dir_info", {}).get("editable", False)
    # tmp 环境里没有任何 .pth 指回仓库。
    for pth in site.glob("*.pth"):
        assert str(REPO) not in pth.read_text(encoding="utf-8"), pth


def test_import_resolves_to_tmp_env_outside_repo(installed):
    deps = sysconfig.get_paths()["purelib"]  # 只借依赖；其中的 editable .pth 不经 PYTHONPATH 处理
    env = dict(installed["env"])
    env["PYTHONPATH"] = os.pathsep.join([str(SITE_DIR), deps])
    code = ("import json, robomme, robomme_hard, robomme_hard.env_record_wrapper.hard_specs as hs\n"
            "print(json.dumps({'robomme': robomme.__file__, 'robomme_hard': robomme_hard.__file__,"
            " 'hard_specs': hs.__file__}))\n")
    r = _run([str(installed["py"]), "-c", code], cwd=installed["cwd"], env=env)
    if r.returncode != 0 and "No module named" in r.stderr and "robomme" not in r.stderr.split("No module named")[-1]:
        pytest.skip(f"未验证：借用依赖导入失败：{r.stderr.strip()[-300:]}")
    assert r.returncode == 0, r.stderr[-2000:]
    files = json.loads(r.stdout.strip().splitlines()[-1])
    site = installed["site"].resolve()
    for name, f in files.items():
        p = Path(f).resolve()
        assert site in p.parents, (name, p)
        assert REPO.resolve() not in p.parents, (name, p)
