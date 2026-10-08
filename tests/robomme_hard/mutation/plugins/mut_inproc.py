"""进程内（不落盘）植入插件：按环境变量 MUT_INPROC=<块>:<编号> 在测试进程里改坏一处生产逻辑。

覆盖 mutants.json 里只给出文字描述、目标在受保护的 ``src/robomme``／三个上游入口（红线 R9，只许进程内）
或须在导入后改常量的植入。每项实现照抄该块作者在 mutants.json 里的描述；未设 MUT_INPROC 时本插件不做任何事。

用法：``pytest -p tests.robomme_hard.mutation.plugins.mut_inproc``，环境变量 ``MUT_INPROC=pipeline/recording:M14a``。
"""
from __future__ import annotations

import importlib
import inspect
import os
import pathlib
import textwrap
import types

KEY = os.environ.get("MUT_INPROC", "")

TASKS = ["BinFill", "PickXtimes", "SwingXtimes", "VideoRepick", "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap",
         "ButtonUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder", "PickHighlight", "StopCube", "InsertPeg",
         "MoveCube", "PatternLock", "RouteStick"]


# ───────────────────────────── 静态块（tests/robomme_hard/static）：改 pathlib 读到的字节 ─────────────────────────────


def _flip_mid(data: bytes) -> bytes:
    b = bytearray(data)
    b[len(b) // 2] ^= 0x01
    return bytes(b)


def _install_read_patch(root: pathlib.Path, key: str) -> None:
    rb, rt = pathlib.Path.read_bytes, pathlib.Path.read_text
    binfill = (root / "src/robomme/robomme_env/BinFill.py").resolve()
    if key == "M04S":
        import hashlib
        import json

        manifest = (root / "src/robomme_hard/UPSTREAM.json").resolve()

        def _man() -> bytes:
            m = json.loads(rb(manifest))
            m.pop("manifest_sha256")
            m["robomme_files"]["src/robomme/robomme_env/BinFill.py"] = hashlib.sha256(_flip_mid(rb(binfill))).hexdigest()
            m["manifest_sha256"] = hashlib.sha256(
                json.dumps(m, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            return json.dumps(m, ensure_ascii=False).encode()

        def mutate(path: pathlib.Path, data_fn):
            r = path.resolve()
            if r == binfill:
                return _flip_mid(data_fn())
            if r == manifest:
                return _man()
            return None
    else:
        target = {
            "M01": binfill,
            "M02": (root / "scripts/evaluation.py").resolve(),
            "M03": (root / "src/robomme_hard/env_record_wrapper/RecordWrapper.py").resolve(),
        }[key]

        def change(data: bytes) -> bytes:
            if key == "M01":
                return _flip_mid(data)
            if key == "M02":
                return data + b"# mutant extra line\n"
            assert data.count(b"fail_safe_limit = 5000") == 1, "M03 植入点没找到"
            return data.replace(b"fail_safe_limit = 5000", b"fail_safe_limit = 2000")

        def mutate(path: pathlib.Path, data_fn):
            return change(data_fn()) if path.resolve() == target else None

    def read_bytes(self):
        out = mutate(self, lambda: rb(self))
        return rb(self) if out is None else out

    def read_text(self, encoding=None, errors=None, newline=None):
        out = mutate(self, lambda: rb(self))
        if out is None:
            return rt(self, encoding=encoding, errors=errors)
        return out.decode(encoding or "utf-8")

    pathlib.Path.read_bytes = read_bytes
    pathlib.Path.read_text = read_text


# ───────────────────────────── 契约块（tests/robomme_hard/contract）：导入后改常量／包装函数 ─────────────────────────────


def _contract(key: str) -> None:
    from robomme_hard.env_record_wrapper import hard_builder, hard_specs as hs

    if key == "M07":
        hs.V9_CELLS[("PickXtimes", "xhard1")] += 1  # EXPECTED_CELLS／CELL_TABLES["v9"] 是同一对象
    elif key == "M10":
        orig = hard_builder._ood_entries

        def swapped(env_id, xhard0, root):
            entries = orig(env_id, xhard0, root)
            if len(entries) > 1:
                entries[0], entries[1] = entries[1], entries[0]
            return entries

        hard_builder._ood_entries = swapped
    else:
        raise KeyError(key)


# ───────────────────────────── 录制块（tests/robomme_hard/pipeline/recording） ─────────────────────────────


def _rec_mods():
    return [importlib.import_module(f"{p}.env_record_wrapper.RecordWrapper") for p in ("robomme", "robomme_hard")]


def _wrap_close(fn) -> None:
    for m in _rec_mods():
        cls = m.RobommeRecordWrapper
        orig = cls.close

        def close(self, _o=orig):
            fn(self)
            return _o(self)

        cls.close = close


def _recording(key: str) -> None:
    import numpy as np

    if key == "M14a":
        import h5py

        orig = h5py.Group.create_dataset

        def cd(self, name, *a, **k):
            if name == "is_subgoal_boundary":
                return None
            return orig(self, name, *a, **k)

        h5py.Group.create_dataset = cd
    elif key == "M14b":
        def f(self):
            if self.h5_file and self.h5_file.id.valid:
                self.h5_file.attrs["mutant"] = 1
        _wrap_close(f)
    elif key == "M14c":
        import torch

        def f(self):
            if self.buffer and self.buffer[0]["action"]["joint_action"] is not None:
                a = np.array(self.buffer[0]["action"]["joint_action"], dtype=np.float64)
                a[0] = np.nan
                self.buffer[0]["action"]["joint_action"] = torch.from_numpy(a)
        _wrap_close(f)
    elif key == "T5-K1":
        from robomme.env_record_wrapper.episode_dataset_resolver import EpisodeDatasetResolver as R
        orig = R._build_indexes

        def bi(self):
            self._timestep_indexes = sorted(self._timestep_indexes, key=str)
            return orig(self)

        R._build_indexes = bi
    elif key == "T5-K2":
        for m in _rec_mods():
            m.RobommeRecordWrapper._video_should_record = lambda self, name: self.save_video
    elif key == "T5-K3":
        for m in _rec_mods():
            cls = m.RobommeRecordWrapper
            orig = cls._video_prepare_step_frames

            def prep(self, base, *a, _o=orig):
                base[0, 0] = 255
                return _o(self, base, *a)

            cls._video_prepare_step_frames = prep
    elif key == "T5-K4":
        from tests.robomme_hard._support.loaders import load_script

        old = os.environ.get("CUDA_VISIBLE_DEVICES")
        mod = load_script("dataset_replay.py")
        # 入口脚本导入时会写 CUDA_VISIBLE_DEVICES；原先没设就删掉，设过就还原
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
        orig = mod._build_action_sequence

        def bas(ep, mode, _o=orig):
            seq = _o(ep, mode)
            if mode != "waypoint":
                return seq
            out = []
            for x in seq:
                if not any(np.array_equal(x, y) for y in out):
                    out.append(x)
            return out

        mod._build_action_sequence = bas
    elif key == "T5-K5":
        for m in _rec_mods():
            cls = m.RobommeRecordWrapper
            orig = cls.step

            def step(self, action, _o=orig):
                out = _o(self, action)
                if bool(out[4]["success"].any()):
                    self.episode_success = True
                return out

            cls.step = step
    elif key == "T5-K6":
        for m in _rec_mods():
            cls = m.RobommeRecordWrapper
            orig = cls._video_flush_episode_files

            def fl(self, success, video_prefix, filename_suffix, _o=orig):
                return _o(self, True, video_prefix, filename_suffix)

            cls._video_flush_episode_files = fl
    elif key == "T5-K7":
        for p in ("robomme", "robomme_hard"):
            cls = importlib.import_module(f"{p}.env_record_wrapper.DemonstrationWrapper").DemonstrationWrapper
            orig = cls._augment_obs_and_info

            def aug(self, obs, info, action, _o=orig):
                o, i = _o(self, obs, info, action)
                o.pop("front_depth_list", None)
                return o, i

            cls._augment_obs_and_info = aug
    elif key == "T5-K8":
        for p in ("robomme", "robomme_hard"):
            cls = importlib.import_module(f"{p}.env_record_wrapper.DemonstrationWrapper").DemonstrationWrapper
            cls._filter_no_record_from_step_batch = lambda self, b: b
    else:
        raise KeyError(key)


# ───────────────────────────── 官方任务单元块（tests/robomme_hard/unit/robomme） ─────────────────────────────


def _everywhere(name, value) -> None:
    sef = importlib.import_module("robomme.robomme_env.utils.subgoal_evaluate_func")
    setattr(sef, name, value)
    for t in TASKS:
        mod = importlib.import_module(f"robomme.robomme_env.{t}")
        if hasattr(mod, name):
            setattr(mod, name, value)


def _unit_robomme(key: str) -> None:
    sef = importlib.import_module("robomme.robomme_env.utils.subgoal_evaluate_func")
    if key == "T3-M01":  # 按钮深度严格大于 → 大于等于
        def is_button_pressed(self, obj):
            return bool(sef.get_button_depth(self, obj=obj) >= 0.005)
        _everywhere("is_button_pressed", is_button_pressed)
    elif key == "T3-M02":  # 失败条件全部失效
        sef._coerce_failure_result = lambda value: False
    elif key == "T3-M03":  # 放到目标的水平阈值 0.05 → 0.06
        def is_obj_dropped_onto(self, obj, target):
            import torch
            o, t = obj.pose.p[0], target.pose.p[0]
            d = torch.sqrt((o[0] - t[0]) ** 2 + (o[1] - t[1]) ** 2)
            return bool(d <= 0.06 and sef.is_obj_dropped(self, obj))
        _everywhere("is_obj_dropped_onto", is_obj_dropped_onto)
    elif key == "T3-M04":  # StopCube 停止窗口永远算对
        _everywhere("correct_timestep", lambda self, time_range=None, stop_timestep=None: True)
    elif key == "T3-M05":  # 交换不发生
        for t in ("VideoUnmaskSwap", "ButtonUnmaskSwap", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"):
            setattr(importlib.import_module(f"robomme.robomme_env.{t}"), "swap_flat_two_lane", lambda *a, **k: None)
    elif key == "T3-M06":  # reset 不再过滤 NO RECORD
        dw = importlib.import_module("robomme.env_record_wrapper.DemonstrationWrapper")
        dw.DemonstrationWrapper._filter_no_record_from_step_batch = lambda self, batch: batch
    elif key == "T3-M07":  # 元数据查不到
        ecr = importlib.import_module("robomme.env_record_wrapper.episode_config_resolver")
        ecr.get_episode_metadata = lambda index, task, episode: None
    elif key == "T3-M08":  # RouteStick 方向判反
        rs = importlib.import_module("robomme.robomme_env.RouteStick").RouteStick
        orig = rs.direction_fail
        flip = {"clockwise": "counterclockwise", "counterclockwise": "clockwise"}

        def direction_fail(self, judge_direction_list=None):
            if judge_direction_list is None:
                return orig(self, None)
            a, b, d = judge_direction_list
            return orig(self, [a, b, flip.get(d, d)])
        rs.direction_fail = direction_fail
    elif key == "T3-M09":  # FailAwareWrapper 不再把异常转成 status=error
        fa = importlib.import_module("robomme.env_record_wrapper.FailAwareWrapper").FailAwareWrapper
        fa.step = lambda self, action: self.env.step(action)
    elif key == "T3-M10":  # 一次调用推进两项
        orig = sef.sequential_task_check

        def sequential_task_check(self, tasks, allow_subgoal_change_this_timestep):
            before = getattr(self, "timestep", 0)
            r = orig(self, tasks, allow_subgoal_change_this_timestep)
            if not r[0] and not r[2] and getattr(self, "timestep", 0) > before:
                return orig(self, tasks, allow_subgoal_change_this_timestep)
            return r
        _everywhere("sequential_task_check", sequential_task_check)
    else:
        raise KeyError(key)


# ───────────────────────────── 包装器块（tests/robomme_hard/unit/wrappers）：按源码替换后重定义方法 ─────────────────────────────


def _func_source(module_name, owner, attr) -> str:
    mod = importlib.import_module(module_name)
    target = getattr(mod, owner) if owner else mod
    return textwrap.dedent(inspect.getsource(getattr(target, attr)))


def _mutate_func(module_name, owner, attr, repl) -> None:
    mod = importlib.import_module(module_name)
    target = getattr(mod, owner) if owner else mod
    src = _func_source(module_name, owner, attr)
    for old, new in repl:
        assert src.count(old) == 1, (attr, old, src.count(old))
        src = src.replace(old, new)
    ns = dict(mod.__dict__)
    ns["gym"] = importlib.import_module("gymnasium")
    exec(compile(src, f"<mutant {attr}>", "exec"), ns)
    tmp = ns[attr]
    # 以真实模块字典作全局，保证测试对模块属性的 monkeypatch 仍对植入后的函数生效
    if "gym" not in mod.__dict__:
        mod.__dict__["gym"] = ns["gym"]
    new = types.FunctionType(tmp.__code__, mod.__dict__, tmp.__name__, tmp.__defaults__, tmp.__closure__)
    new.__kwdefaults__ = tmp.__kwdefaults__
    setattr(target, attr, new)


_SUPER_RESET = ("super().reset(**kwargs)", "gym.Wrapper.reset(self, **kwargs)")
_SUPER_STEP = ("super().step(normalized_action)", "gym.Wrapper.step(self, normalized_action)")
_SUPER_INIT = ("super().__init__(env)", "gym.Wrapper.__init__(self, env)")
_DW = "robomme.env_record_wrapper.DemonstrationWrapper"
_SWAP_CONCAT = ("concat_step_batches([demo_batch, init_batch])", "concat_step_batches([init_batch, demo_batch])")
_ANY_EXC = ("except screw_failure_exc as exc:", "except Exception as exc:")

#: 键 → (模块, 类或 None, 方法名, [(old, new), ...])
_WRAPPERS = {
    "T10-W1": ("robomme.env_record_wrapper.EndeffectorDemonstrationWrapper", "EndeffectorDemonstrationWrapper", "step",
               [("ik_solutions[0][:7]", "ik_solutions[-1][:7]")]),
    "T10-W2": ("robomme.env_record_wrapper.MultiStepDemonstrationWrapper", "MultiStepDemonstrationWrapper", "step",
               [("waypoint_q = rpy_xyz_to_quat_wxyz_torch(rpy_t).numpy()",
                 "waypoint_q = np.array([1.0, 0.0, 0.0, 0.0])")]),
    "T10-W3": (_DW, "DemonstrationWrapper", "reset", [_SUPER_RESET, _SWAP_CONCAT]),
    "T10-W4": (_DW, "DemonstrationWrapper", "get_demonstration_trajectory",
               [("self.unwrapped.demonstration_record_traj = False", "pass")]),
    "T10-W5": (_DW, "DemonstrationWrapper", "__init__",
               [_SUPER_INIT, ("self._demo_rrt_max_attempts = 3", "self._demo_rrt_max_attempts = 2")]),
    "T10-W6": ("robomme.env_record_wrapper.OraclePlannerDemonstrationWrapper", "OraclePlannerDemonstrationWrapper",
               "_wrap_planner_with_screw_then_rrt_retry", [_ANY_EXC]),
    "T10-W7": ("robomme.robomme_env.utils.planner_denseStep", None, "close_gripper",
               [("lambda: planner.close_gripper()", "lambda: planner.open_gripper()")]),
    "T10-W8": (_DW, "DemonstrationWrapper", "_step_batch",
               [_SUPER_STEP, ("if self.current_task_demonstration == False:", "if True:")]),
    "T10-W9": ("robomme_hard.env_record_wrapper.DemonstrationWrapper", "DemonstrationWrapper", "reset",
               [_SUPER_RESET, _SWAP_CONCAT]),
    "T10-W10": ("robomme_hard.env_record_wrapper.OraclePlannerDemonstrationWrapper", "OraclePlannerDemonstrationWrapper",
                "_wrap_planner_with_screw_then_rrt_retry", [_ANY_EXC]),
}


# ───────────────────────────── 植入前预校验：植入点必须存在且（片段型）恰好命中 1 次 ─────────────────────────────


def _count_reason(where: str, text, frag) -> str | None:
    n = text.count(frag)
    return None if n == 1 else f"{where} 植入片段命中 {n} 次：{frag!r}"


def _missing(obj, *attrs) -> str | None:
    lacking = [a for a in attrs if not hasattr(obj, a)]
    return f"{getattr(obj, '__name__', obj)} 缺属性 {lacking}" if lacking else None


def _first(*reasons) -> str | None:
    return next((r for r in reasons if r), None)


def _precheck(root: pathlib.Path, block: str, key: str) -> str | None:
    """返回 None 表示可以植入；否则返回不能植入的原因（不改任何东西）。"""
    if block == "static":
        binfill = root / "src/robomme/robomme_env/BinFill.py"
        if key in ("M01", "M04S") and (not binfill.is_file() or binfill.stat().st_size == 0):
            return f"{binfill} 不存在或为空"
        if key == "M02":
            p = root / "scripts/evaluation.py"
            return None if p.is_file() else f"{p} 不存在"
        if key == "M03":
            p = root / "src/robomme_hard/env_record_wrapper/RecordWrapper.py"
            return _count_reason(str(p), p.read_bytes(), b"fail_safe_limit = 5000") if p.is_file() else f"{p} 不存在"
        if key == "M04S":
            import json

            m = json.loads((root / "src/robomme_hard/UPSTREAM.json").read_bytes())
            if "manifest_sha256" not in m or "src/robomme/robomme_env/BinFill.py" not in m.get("robomme_files", {}):
                return "UPSTREAM.json 缺 manifest_sha256 或 BinFill.py 条目"
        return None
    if block == "contract":
        from robomme_hard.env_record_wrapper import hard_builder, hard_specs as hs

        if key == "M07":
            return None if ("PickXtimes", "xhard1") in hs.V9_CELLS else "V9_CELLS 缺 (PickXtimes, xhard1)"
        return _missing(hard_builder, "_ood_entries")
    if block == "pipeline/recording":
        if key == "M14a":
            import h5py

            srcs = [inspect.getsource(m) for m in _rec_mods()]
            return _first(_missing(h5py.Group, "create_dataset"),
                          *(None if "is_subgoal_boundary" in s else "RecordWrapper 源码里没有 is_subgoal_boundary"
                            for s in srcs))
        if key == "T5-K1":
            from robomme.env_record_wrapper.episode_dataset_resolver import EpisodeDatasetResolver as R

            return _first(_missing(R, "_build_indexes"),
                          None if "_timestep_indexes" in inspect.getsource(R) else "解析器没有 _timestep_indexes")
        if key == "T5-K4":
            text = (root / "scripts/dataset_replay.py").read_text(encoding="utf-8")
            return _count_reason("scripts/dataset_replay.py", text, "def _build_action_sequence(")
        if key in ("T5-K7", "T5-K8"):
            attr = "_augment_obs_and_info" if key == "T5-K7" else "_filter_no_record_from_step_batch"
            return _first(*(_missing(importlib.import_module(f"{p}.env_record_wrapper.DemonstrationWrapper")
                                     .DemonstrationWrapper, attr) for p in ("robomme", "robomme_hard")))
        attr = {"M14b": "close", "M14c": "close", "T5-K2": "_video_should_record",
                "T5-K3": "_video_prepare_step_frames", "T5-K5": "step", "T5-K6": "_video_flush_episode_files"}[key]
        reason = _first(*(_missing(m.RobommeRecordWrapper, attr) for m in _rec_mods()))
        if reason is None and key == "T5-K6":
            for m in _rec_mods():
                params = list(inspect.signature(m.RobommeRecordWrapper._video_flush_episode_files).parameters)
                if params != ["self", "success", "video_prefix", "filename_suffix"]:
                    return f"_video_flush_episode_files 签名变了：{params}"
        return reason
    if block == "unit/robomme":
        sef = importlib.import_module("robomme.robomme_env.utils.subgoal_evaluate_func")
        if key == "T3-M01":
            return _first(_missing(sef, "get_button_depth", "is_button_pressed"),
                          _count_reason("is_button_pressed", inspect.getsource(sef.is_button_pressed), "if depth > 0.005:"))
        if key == "T3-M03":
            return _first(_missing(sef, "is_obj_dropped", "is_obj_dropped_onto"),
                          _count_reason("is_obj_dropped_onto", inspect.getsource(sef.is_obj_dropped_onto),
                                        "distance_threshold = 0.05"))
        if key == "T3-M05":
            return _first(*(_missing(importlib.import_module(f"robomme.robomme_env.{t}"), "swap_flat_two_lane")
                            for t in ("VideoUnmaskSwap", "ButtonUnmaskSwap", "VideoRepick", "VideoPlaceButton",
                                      "VideoPlaceOrder")))
        if key == "T3-M06":
            return _missing(importlib.import_module("robomme.env_record_wrapper.DemonstrationWrapper").DemonstrationWrapper,
                            "_filter_no_record_from_step_batch")
        if key == "T3-M07":
            return _missing(importlib.import_module("robomme.env_record_wrapper.episode_config_resolver"),
                            "get_episode_metadata")
        if key == "T3-M08":
            return _missing(importlib.import_module("robomme.robomme_env.RouteStick").RouteStick, "direction_fail")
        if key == "T3-M09":
            return _missing(importlib.import_module("robomme.env_record_wrapper.FailAwareWrapper").FailAwareWrapper,
                            "step")
        attr = {"T3-M02": "_coerce_failure_result", "T3-M04": "correct_timestep",
                "T3-M10": "sequential_task_check"}[key]
        return _missing(sef, attr)
    if block == "unit/wrappers":
        mod, owner, attr, repl = _WRAPPERS[key]
        src = _func_source(mod, owner, attr)
        return _first(*(_count_reason(f"{mod}.{owner}.{attr}", src, old) for old, _ in repl))
    return f"未知块 {block}"


# 每块的植入时机沿用该块作者自检时的钩子：静态／契约／录制在 pytest_configure，官方单元与包装器在 pytest_sessionstart。
CONFIGURE_BLOCKS = {"static", "contract", "pipeline/recording"}
SESSIONSTART_BLOCKS = {"unit/robomme", "unit/wrappers"}

#: 本插件能执行的全部键（执行器据此把 mutants.json 的条目归到 B 类）。
SUPPORTED = (
    {f"static:{k}" for k in ("M01", "M02", "M03", "M04S")}
    | {f"contract:{k}" for k in ("M07", "M10")}
    | {f"pipeline/recording:{k}" for k in ("M14a", "M14b", "M14c", "T5-K1", "T5-K2", "T5-K3", "T5-K4", "T5-K5",
                                          "T5-K6", "T5-K7", "T5-K8")}
    | {f"unit/robomme:T3-M{i:02d}" for i in range(1, 11)}
    | {f"unit/wrappers:{k}" for k in _WRAPPERS}
)


def _write_status(applied: bool, reason: str | None) -> None:
    """把植入是否真正生效写到 MUT_STATUS_FILE；执行器只有读到 applied=true 才会把失败计为抓到。"""
    path = os.environ.get("MUT_STATUS_FILE")
    if path:
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"key": KEY, "applied": applied, "reason": reason}, fh, ensure_ascii=False)


def _apply(config) -> None:
    block, _, key = KEY.partition(":")
    try:
        reason = _precheck(pathlib.Path(str(config.rootpath)), block, key)
    except Exception as exc:  # 预校验自身出错同样视为不能植入
        reason = f"预校验异常 {type(exc).__name__}: {exc}"
    if reason:
        # 不植入、不中断会话：用例照原版跑，执行器据状态文件记为 not_applied
        _write_status(False, reason)
        print(f"MUT_INPROC_NOT_APPLIED={KEY} {reason}", flush=True)
        return
    try:
        if block == "static":
            _install_read_patch(pathlib.Path(str(config.rootpath)), key)
        elif block == "contract":
            _contract(key)
        elif block == "pipeline/recording":
            _recording(key)
        elif block == "unit/robomme":
            _unit_robomme(key)
        elif block == "unit/wrappers":
            _mutate_func(*_WRAPPERS[key])
        else:
            raise KeyError(KEY)
    except Exception as exc:
        _write_status(False, f"植入异常 {type(exc).__name__}: {exc}")
        raise
    _write_status(True, None)
    print(f"MUT_INPROC_APPLIED={KEY}", flush=True)


def pytest_configure(config):
    if KEY:
        if KEY not in SUPPORTED:
            _write_status(False, f"未知植入 {KEY}")
            raise SystemExit(f"未知植入 {KEY}")
        if KEY.partition(":")[0] in CONFIGURE_BLOCKS:
            _apply(config)


def pytest_sessionstart(session):
    if KEY and KEY.partition(":")[0] in SESSIONSTART_BLOCKS:
        _apply(session.config)
