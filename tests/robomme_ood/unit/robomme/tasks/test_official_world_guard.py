"""Boundaries of the offline world itself: all replacements are restored after entering and leaving the context; doubles do not touch render materials; the resource guard's interception of BaseEnv.__init__
stays in effect outside the context (only object identity is compared; the forbidden entry is never actually called, to avoid recording a violation)."""
from __future__ import annotations

import pytest
from mani_skill.envs.sapien_env import BaseEnv

import _official_world as ow
from _official_world import OfficialWorld


def test_patches_are_restored():
    mod = ow.task_module("RouteStick")
    before = {name: getattr(mod, name) for name in ow.MODULE_PATCHES if hasattr(mod, name)}
    before["sapien"] = mod.sapien
    init, step = BaseEnv.__init__, BaseEnv.step
    with OfficialWorld("RouteStick") as world:
        assert BaseEnv.__init__ is not init and BaseEnv.step is not step
        assert mod.build_gray_white_target is ow.fake_build_disk_target
        world.make("hard", seed=0)
    assert BaseEnv.__init__ is init and BaseEnv.step is step
    for name, value in before.items():
        assert getattr(mod, name) is value, name


def _snapshot(mod):
    snap = {name: getattr(mod, name) for name in ow.MODULE_PATCHES if hasattr(mod, name)}
    snap["sapien"] = mod.sapien
    return snap, BaseEnv.__init__, BaseEnv.step


def _assert_restored(mod, snap):
    names, init, step = snap
    assert BaseEnv.__init__ is init and BaseEnv.step is step
    for name, value in names.items():
        assert getattr(mod, name) is value, name


def test_exception_inside_context_still_restores():
    mod = ow.task_module("PickXtimes")
    snap = _snapshot(mod)
    with pytest.raises(RuntimeError, match="exception inside the test"):
        with OfficialWorld("PickXtimes") as world:
            world.make("easy", seed=0)
            raise RuntimeError("exception inside the test")
    _assert_restored(mod, snap)


def test_failure_while_entering_rolls_back():
    class _Flaky(OfficialWorld):
        def _swap(self, obj, name, value):
            if obj is BaseEnv:  # error while replacing BaseEnv, after all module-level replacements are installed
                raise RuntimeError("half installed")
            super()._swap(obj, name, value)

    mod = ow.task_module("PickXtimes")
    snap = _snapshot(mod)
    world = _Flaky("PickXtimes")
    with pytest.raises(RuntimeError, match="half installed"):
        world.__enter__()
    _assert_restored(mod, snap)
    assert world._saved == []


def test_guard_still_blocks_real_init_outside_world():
    # the resource guard replaces BaseEnv.__init__ with an interceptor before collection; it must still be that after leaving the offline world
    assert BaseEnv.__init__.__name__ == "_blocked_init"
    assert getattr(BaseEnv, "_resource_policy_patched", False) is True


def test_render_material_is_never_real():
    with OfficialWorld("InsertPeg") as world:
        mod = world.module
        assert mod.sapien.render.RenderMaterial is ow._FakeMaterial
        ep = world.make("easy", seed=0)
        ep.step(3)  # InsertPeg.step creates a new material every step
