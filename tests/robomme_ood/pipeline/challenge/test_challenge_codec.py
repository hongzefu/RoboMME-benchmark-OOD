"""C14 codec: NumPy round trip of ``challenge_interface.msgpack_numpy``.

Expected values are derived independently: input arrays are hand-built by the test, and after the round trip dtype string (including byte order), shape,
element values and C-order bytes are compared with the original; the pack/unpack implementation is not replicated.
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
    "float64_noncontiguous_slice": _base()[:, ::2],
    "float64_transposed": _base().T,
    "float64_fortran": np.asfortranarray(_base()),
    "bigendian_f4": (np.arange(6).reshape(2, 3) + 0.25).astype(">f4"),
    "bigendian_i8": np.array([[-1, 2**40], [3, -(2**33)]], dtype=">i8"),
    "littleendian_u2": np.array([1, 65535, 256], dtype="<u2"),
    "0d": np.array(3.5, dtype=np.float32),
    "bool": np.array([[True, False, True]], dtype=bool),
    "float16_with_nan_and_inf": np.array([1.5, -0.0, np.nan, np.inf, 65504.0], dtype=np.float16),
    "uint8_image": (np.arange(4 * 4 * 3) % 256).astype(np.uint8).reshape(4, 4, 3),
    "empty_array": np.zeros((0, 3), dtype=np.int16),
    "datetime64": np.array(["2026-10-04", "1970-01-01"], dtype="datetime64[D]"),
}


@pytest.mark.parametrize("name", list(ARRAYS))
def test_ndarray_roundtrip_preserves_dtype_shape_endianness_values(name):
    a = ARRAYS[name]
    if name.startswith("bigendian"):
        assert a.dtype.byteorder == ">", "the fixture itself must be big-endian"
    out = _roundtrip(a)
    assert isinstance(out, np.ndarray)
    # the dtype string includes byte order ('>f4' differs from '<f4') and must come back unchanged.
    assert out.dtype.str == a.dtype.str
    assert out.shape == a.shape
    # values equal (NaN treated as equal) and C-order bytes identical.
    if a.dtype.kind == "f":
        np.testing.assert_array_equal(out, a)
    else:
        assert np.array_equal(out, a)
    assert np.ascontiguousarray(out).tobytes() == np.ascontiguousarray(a).tobytes()


def test_big_endian_values_are_not_byte_swapped_garbage():
    """Negative view: if a big-endian array lost its byte-order flag, reading it as little-endian gives completely different numbers; checked here with hand-written values."""
    a = np.array([1.0, 2.0, -0.5], dtype=">f4")
    out = _roundtrip(a)
    assert out.tolist() == [1.0, 2.0, -0.5]
    assert out.dtype.byteorder == ">"


def test_non_contiguous_view_roundtrip_matches_hand_values():
    a = np.arange(10, dtype=np.int32)[::3]  # by hand: 0,3,6,9
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
    """The observation is a dict of lists of arrays mixed with plain Python values; check each item after the round trip."""
    obs = {
        "task_goal": ["put the red cube into the box", "alternative goal"],
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
        np.zeros(2, dtype=[("x", "<f4"), ("y", "<i4")]),  # kind 'V' (structured)
        np.complex128(1 + 1j),  # scalar kind 'c'
    ],
    ids=["complex_array", "object_array", "structured_array", "complex_scalar"],
)
def test_unsupported_dtypes_raise_explicitly(bad):
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mn.packb(bad)


def test_unsupported_dtype_nested_inside_dict_also_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mn.packb({"ok": np.zeros(2), "bad": [np.array([1j])]})
