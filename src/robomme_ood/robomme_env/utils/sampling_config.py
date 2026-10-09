"""``sampling_config`` 的统一形态与校验（newtaskRelease-v3 步 3）。

方案第三节规定配置快照按两块组织：

``decision``
    以后允许改的参数（第二节字段表里标「是，拟修改」的那些）。**原值对拍阶段
    它也必须等于原值**（红线 R7），所以本模块在原值模式下逐键比对并拒绝任何偏差。

``native``
    维持原状的随机规则与常量，沿用各环境已有的 ``{"parameters": ..., "positions": ...}``
    结构，键名、取值域、单位、dtype 与拒绝条件一律不动。

为兼容四个早于本方案就接了 ``sampling_config`` 的环境（BinFill／RouteStick／VideoRepick／
VideoUnmaskSwap）及其既有产物与工具（``scripts/injection/``、``--extract-config`` 等），
旧格式 ``{"parameters": ..., "positions": ...}`` 继续被接受，等价于只给了 ``native``。

本模块**不抽任何随机数**，也不得在解析过程中触发随机数：调用点必须落在
``torch.Generator()`` 创建之前，在这里多抽或少抽一次会平移其后全部取值。
"""

from __future__ import annotations

import copy
import json

NATIVE_KEYS = ("parameters", "positions")


class SamplingConfigError(ValueError):
    """传入的 sampling_config 不满足固定形态或在原值模式下偏离了原值。"""


def split_sampling_config(override, native_default, decision_default):
    """把外部传入的配置拆成 ``(decision, native)`` 两块副本。

    参数
    ----
    override
        ``None`` 表示不传，直接用两个默认块；
        新格式为 ``{"decision": ..., "native": ...}``（``decision`` 可省略）；
        旧格式为 ``{"parameters": ..., "positions": ...}``，等价于只给 ``native``。
    native_default / decision_default
        本环境的原值快照，用于缺省填充与原值模式校验。

    返回的两块都是 deepcopy：gymnasium 会把 kwargs 字典的引用存进
    ``env.unwrapped.spec.kwargs``，不复制就会被跨局改写。
    """
    if override is None:
        return copy.deepcopy(decision_default), copy.deepcopy(native_default)
    if not isinstance(override, dict):
        raise SamplingConfigError("sampling_config 必须是字典")

    keys = set(override)
    if keys == set(NATIVE_KEYS):
        # 旧格式：只给了 native 块
        decision, native = copy.deepcopy(decision_default), copy.deepcopy(override)
    elif keys <= {"decision", "native"} and "native" in keys:
        native = copy.deepcopy(override["native"])
        decision = copy.deepcopy(override.get("decision", decision_default))
    else:
        raise SamplingConfigError(
            "sampling_config 必须是 {decision, native} 或旧格式 {parameters, positions}，"
            f"当前键为 {sorted(keys)}"
        )
    if not isinstance(native, dict) or set(native) != set(NATIVE_KEYS):
        raise SamplingConfigError("sampling_config.native 必须只含 parameters 与 positions")
    if not isinstance(decision, dict):
        raise SamplingConfigError("sampling_config.decision 必须是字典")
    return decision, native


# V6：活动新值键为 xhard1..xhard4；v8 加 xhard5（只有 SwingXtimes、StopCube 申报）。
# 只读 V5 投影也识别旧键 xhard，以便保持旧快照不变。
from .difficulty import ALL_NEWVALUE_TIERS

#: 活动新值族的最难档键。
XHARD4_KEY = "xhard4"
#: decision 里所有「新值」子树的键名（任意深度）；v8 起含 xhard5，否则剥不掉 Swing／StopCube 的 xhard5 子树
NEWVALUE_KEYS = frozenset(ALL_NEWVALUE_TIERS)
#: V5 冻结快照中的历史新值键；仅用于比较剥离，不作为可用难度档。
LEGACY_NEWVALUE_KEYS = frozenset({"xhard"})
_STRIP_NEWVALUE_KEYS = NEWVALUE_KEYS | LEGACY_NEWVALUE_KEYS


def _strip_xhard(node):
    """去掉活动及 V5 历史新值键，得到原三档可见部分。"""
    if isinstance(node, dict):
        return {key: _strip_xhard(value) for key, value in node.items() if key not in _STRIP_NEWVALUE_KEYS}
    if isinstance(node, list):
        return [_strip_xhard(item) for item in node]
    return node


def _xhard_shape(node, prefix="", tier=None):
    """列出新值子树的全部键路径（只看结构不看值），返回 ``{(档名, 路径)}``；档名取路径上第一个新值键。"""
    out = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            here = tier if tier is not None else (key if key in NEWVALUE_KEYS else None)
            if here is not None:
                out.add((here, path))
            out |= _xhard_shape(value, path, here)
    return out


#: xhard1/2/3 是在 xhard4 之后新加的档；缺失时仅从同层 xhard4 配置补齐。
#: v8 写死三键（原 ``NEWVALUE_DIFFICULTIES[:-1]``），xhard5 不在补齐范围内。
V6_ADDED_KEYS = frozenset({"xhard1", "xhard2", "xhard3"})


def fill_missing_newvalue(decision, decision_default):
    """旧快照兜底（V6 口径 11）：快照里缺的 **V6 新增档**（xhard1/2/3）子树从源码申报深拷贝补齐；已有的一律不动。

    只补 xhard1/2/3，且只在同一层已有 xhard4 时补；V5 快照中的历史 xhard 不会触发补齐。
    **xhard4 本身缺失的层一概不补**，保持旧快照原有行为。
    原三档可见部分不受影响。原地修改并返回 ``decision``。
    """
    if isinstance(decision, dict) and isinstance(decision_default, dict):
        for key, value in decision_default.items():
            if key in V6_ADDED_KEYS:
                # 只给「同一层已有 xhard4」的 V6 快照补齐缺档；V5 历史 xhard 不触发补齐
                if key not in decision and XHARD4_KEY in decision:
                    decision[key] = json.loads(json.dumps(value))
            elif key in decision and key not in NEWVALUE_KEYS:
                fill_missing_newvalue(decision[key], value)
            elif key in decision:
                # 已有的新值档子树（如 xhard）内部不再下钻补键：结构由 assert_native_decision 按档核对
                pass
    return decision


def assert_native_decision(decision, decision_default, task):
    """``decision`` 守卫（红线 R7；V4 步 2 分叉）。

    * **原值部分**（去掉所有新值及历史 xhard 键之后）必须与原值快照逐键相同——原三档可见的
      任何偏差都必须是显式的新用户决策，不能混在「仅随机外移」里悄悄生效。
    * **新值部分**：只允许偏离本环境源码里已申报的新值档条目，不许新增申报外的键。
    """
    left = json.dumps(_strip_xhard(decision), sort_keys=True, ensure_ascii=False)
    right = json.dumps(_strip_xhard(decision_default), sort_keys=True, ensure_ascii=False)
    if left != right:
        raise SamplingConfigError(
            f"{task}: 原值对拍模式下 decision 必须等于原值快照；收到的与原值不同"
        )
    shape = _xhard_shape(decision)
    # V6：按档核对——快照里出现的每个新值档，其键结构必须与源码申报逐一相同（值可以不同）；
    # 快照里完全没出现的档不核对；仅对已有 xhard4 的 V6 快照由 fill_missing_newvalue 补缺档。
    present = {tier for tier, _path in shape}
    declared = {item for item in _xhard_shape(decision_default) if item[0] in present}
    if shape and shape != declared:
        raise SamplingConfigError(
            f"{task}: decision 的 xhard 条目与源码申报不符："
            f"多出 {sorted(shape - declared)}，缺少 {sorted(declared - shape)}"
        )
