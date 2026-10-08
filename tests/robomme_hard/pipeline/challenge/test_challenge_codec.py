"""C14 编解码：``challenge_interface.msgpack_numpy`` 的 NumPy 往返。

期望值独立得出：输入数组由测试手工构造，往返后与原数组比较 dtype 字符串（含端序）、shape、
逐元素值与 C 序字节；不复刻 pack/unpack 的实现。
"""
from __future__ import annotations

import numpy as np
import pytest

from challenge_interface import msgpack_numpy as mn


def _roundtrip(obj):
    return mn.unpackb(mn.packb(obj))


def _base():
    return np.arange(24, dtype=np.float64).reshape(4, 6) * 0.5 - 3.0


ARRAYS = {
    "float32_c": np.arange(12, dtype=np.float32).reshape(3, 4),
    "float64_非连续切片": _base()[:, ::2],
    "float64_转置": _base().T,
    "float64_fortran": np.asfortranarray(_base()),
    "大端_f4": (np.arange(6).reshape(2, 3) + 0.25).astype(">f4"),
    "大端_i8": np.array([[-1, 2**40], [3, -(2**33)]], dtype=">i8"),
    "小端_u2": np.array([1, 65535, 256], dtype="<u2"),
    "0维": np.array(3.5, dtype=np.float32),
    "bool": np.array([[True, False, True]], dtype=bool),
    "float16_含nan与inf": np.array([1.5, -0.0, np.nan, np.inf, 65504.0], dtype=np.float16),
    "uint8_图像": (np.arange(4 * 4 * 3) % 256).astype(np.uint8).reshape(4, 4, 3),
    "空数组": np.zeros((0, 3), dtype=np.int16),
    "datetime64": np.array(["2026-10-04", "1970-01-01"], dtype="datetime64[D]"),
}


@pytest.mark.parametrize("name", list(ARRAYS))
def test_ndarray_roundtrip_preserves_dtype_shape_endianness_values(name):
    a = ARRAYS[name]
    if name.startswith("大端"):
        assert a.dtype.byteorder == ">", "夹具本身必须是大端"
    out = _roundtrip(a)
    assert isinstance(out, np.ndarray)
    # dtype 字符串含端序（'>f4' 与 '<f4' 不同），必须原样回来。
    assert out.dtype.str == a.dtype.str
    assert out.shape == a.shape
    # 值相等（NaN 视为相等）且 C 序字节完全一致。
    if a.dtype.kind == "f":
        np.testing.assert_array_equal(out, a)
    else:
        assert np.array_equal(out, a)
    assert np.ascontiguousarray(out).tobytes() == np.ascontiguousarray(a).tobytes()


def test_big_endian_values_are_not_byte_swapped_garbage():
    """负例视角：大端数组若丢了端序标记，按小端解读会得到完全不同的数；这里用手写值核对。"""
    a = np.array([1.0, 2.0, -0.5], dtype=">f4")
    out = _roundtrip(a)
    assert out.tolist() == [1.0, 2.0, -0.5]
    assert out.dtype.byteorder == ">"


def test_non_contiguous_view_roundtrip_matches_hand_values():
    a = np.arange(10, dtype=np.int32)[::3]  # 手算：0,3,6,9
    assert not a.flags.c_contiguous
    out = _roundtrip(a)
    assert out.tolist() == [0, 3, 6, 9]


@pytest.mark.parametrize(
    "scalar, expected_type, expected_value",
    [
        (np.float32(1.25), np.float32, 1.25),
        (np.float16(-2.5), np.float16, -2.5),
        (np.int64(-3), np.int64, -3),
        (np.uint8(255), np.uint8, 255),
        (np.bool_(True), np.bool_, True),
    ],
)
def test_numpy_scalar_roundtrip_keeps_type(scalar, expected_type, expected_value):
    out = _roundtrip(scalar)
    assert type(out) is expected_type
    assert out == expected_value


def test_nested_observation_structure_roundtrip():
    """观测是 dict 套 list 套数组，再混普通 Python 值；往返后逐项核对。"""
    obs = {
        "task_goal": ["把红色方块放进盒子", "备选目标"],
        "is_first_step": True,
        "front_rgb_list": [np.full((2, 2, 3), 7, dtype=np.uint8), np.full((2, 2, 3), 9, dtype=np.uint8)],
        "joint_state_list": [np.linspace(0, 1, 7, dtype=np.float32)],
        "count": 3,
    }
    out = _roundtrip(obs)
    assert out["task_goal"] == obs["task_goal"]
    assert out["is_first_step"] is True
    assert out["count"] == 3
    assert len(out["front_rgb_list"]) == 2
    assert [int(x[0, 0, 0]) for x in out["front_rgb_list"]] == [7, 9]
    assert out["joint_state_list"][0].dtype == np.float32
    assert out["joint_state_list"][0].shape == (7,)


def test_streaming_packer_matches_packb_bytes():
    obs = {"a": np.arange(5, dtype=np.int16), "b": np.float32(0.5)}
    assert mn.Packer().pack(obs) == mn.packb(obs)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1 + 2j, 3 - 1j], dtype=np.complex64),  # kind 'c'
        np.array([{"x": 1}, None], dtype=object),  # kind 'O'
        np.zeros(2, dtype=[("x", "<f4"), ("y", "<i4")]),  # kind 'V'（结构化）
        np.complex128(1 + 1j),  # 标量 kind 'c'
    ],
    ids=["complex数组", "object数组", "结构化数组", "complex标量"],
)
def test_unsupported_dtypes_raise_explicitly(bad):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mn.packb(bad)


def test_unsupported_dtype_nested_inside_dict_also_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mn.packb({"ok": np.zeros(2), "bad": [np.array([1j])]})
