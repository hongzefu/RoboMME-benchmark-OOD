"""T4 植入自检插件：按环境变量 T4_MUTANT 在测试进程内（不落盘）改坏一处生产逻辑。

用法：PYTHONPATH 加上本目录，pytest -p mutants_plugin，T4_MUTANT=<名字>。
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import textwrap


def _redefine(mod, qualname, old, new, cls=None):
    """把 mod（或 cls）里名为 qualname 的函数源码做一次替换后在 mod 的全局里重新定义；返回 (旧对象, 新对象)。"""
    holder = cls if cls is not None else mod
    orig = getattr(holder, qualname)
    src = textwrap.dedent(inspect.getsource(orig))
    assert src.count(old) >= 1, f"植入点没找到：{qualname}: {old!r}"
    src = src.replace(old, new, 1)
    ns = dict(mod.__dict__)
    exec(compile(src, f"<mutant {mod.__name__}.{qualname}>", "exec"), ns)
    new_obj = ns[qualname]
    setattr(holder, qualname, new_obj)
    if cls is None:
        mod.__dict__[qualname] = new_obj
    return orig, new_obj


def _rebind_everywhere(orig, new_obj, name):
    for m in list(sys.modules.values()):
        if m is None or not getattr(m, "__name__", "").startswith("robomme_hard"):
            continue
        if getattr(m, name, None) is orig:
            setattr(m, name, new_obj)


def _task_modules():
    from robomme_hard.env_record_wrapper import hard_specs

    for t in hard_specs.ALL_TASKS:
        importlib.import_module(f"robomme_hard.robomme_env.{t}")


def pytest_configure(config):
    name = os.environ.get("T4_MUTANT")
    if not name:
        return
    _task_modules()
    sef = importlib.import_module("robomme_hard.robomme_env.utils.subgoal_evaluate_func")
    if name == "M11_success_one_step_early":
        o, n = _redefine(sef, "sequential_task_check", "if self.timestep == num_tasks - 1:",
                         "if self.timestep >= num_tasks - 2:")
        _rebind_everywhere(o, n, "sequential_task_check")
    elif name == "M11_failure_priority_reversed":
        o, n = _redefine(sef, "sequential_task_check", "if current_failure_func is not None:",
                         "if current_failure_func is not None and not current_task_func():")
        _rebind_everywhere(o, n, "sequential_task_check")
    elif name == "M12_swing_distance_strict":
        o, n = _redefine(sef, "is_obj_swing_onto", "if horizontal_distance <= distance_threshold and z_flag:",
                         "if horizontal_distance < distance_threshold and z_flag:")
        _rebind_everywhere(o, n, "is_obj_swing_onto")
    elif name == "M12_swing_height_inclusive":
        o, n = _redefine(sef, "is_obj_swing_onto", "z_flag=obj_pos[2]<z_threshold", "z_flag=obj_pos[2]<=z_threshold")
        _rebind_everywhere(o, n, "is_obj_swing_onto")
    elif name == "M12_recorded_drift_strict":
        hs = importlib.import_module("robomme_hard.env_record_wrapper.hard_specs")
        o, n = _redefine(hs, "spec_binding", "if diff <= RECORDED_FLOAT_TOL:", "if diff < RECORDED_FLOAT_TOL:")
        _rebind_everywhere(o, n, "spec_binding")
    elif name == "M12_vpb_before_after_swapped":
        vpb = importlib.import_module("robomme_hard.robomme_env.VideoPlaceButton")
        _redefine(vpb, "_load_scene_xhard_tail",
                  "if self.task_flag == 1:\n            return int(last_before_target",
                  "if self.task_flag != 1:\n            return int(last_before_target", cls=vpb.VideoPlaceButton)
    elif name == "Mx_replay_returns_drawn":
        es = importlib.import_module("robomme_hard.robomme_env.utils.episode_spec")
        _redefine(es, "value", "    return frozen\n", "    return drawn\n", cls=es.SpecRecorder)
    elif name == "Mx_s5_allows_undo":
        su = importlib.import_module("robomme_hard.robomme_env.utils.swap_uniform")
        o, n = _redefine(su, "_greedy_once", "if not (forbid_undo and e == last)", "if True")
    elif name == "Mx_native_button_shift":
        px = importlib.import_module("robomme_hard.robomme_env.PickXtimes")
        px.NATIVE_SAMPLING["positions"]["button"]["center_xy"] = [-0.19, 0]
    else:
        raise SystemExit(f"未知植入 {name}")
    print(f"T4_MUTANT_APPLIED={name}", flush=True)
