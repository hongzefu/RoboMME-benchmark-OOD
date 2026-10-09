"""Unified shape and validation of ``sampling_config`` (newtaskRelease-v3 step 3).

Section 3 of the proposal specifies that the config snapshot is organized in two blocks:

``decision``
    parameters that may be changed later (those marked "yes, to be modified" in the field table of section 2). **In the original-value parity stage
    it must also equal the original value** (red line R7), so in original-value mode this module compares key by key and rejects any deviation.

``native``
    random rules and constants that stay unchanged; reuse each env's existing ``{"parameters": ..., "positions": ...}``
    structure; key names, value domains, units, dtypes and rejection conditions are all untouched.

For compatibility with the four envs that adopted ``sampling_config`` before this proposal (BinFill / RouteStick / VideoRepick /
VideoUnmaskSwap) and their existing artifacts and tools (``scripts/injection/``, ``--extract-config``, etc.),
the old format ``{"parameters": ..., "positions": ...}`` is still accepted, equivalent to giving only ``native``.

This module **draws no random numbers** and must not trigger any during parsing: the call site must come
before ``torch.Generator()`` is created; one extra or missing draw here would shift every subsequent value.
"""

from __future__ import annotations

import copy
import json

NATIVE_KEYS = ("parameters", "positions")


class SamplingConfigError(ValueError):
    """The given sampling_config does not match the fixed shape, or deviates from the original value in original-value mode."""


def split_sampling_config(override, native_default, decision_default):
    """Split the externally supplied config into two copies ``(decision, native)``.

    Parameters
    ----
    override
        ``None`` means not supplied; use the two default blocks directly;
        the new format is ``{"decision": ..., "native": ...}`` (``decision`` optional);
        the old format is ``{"parameters": ..., "positions": ...}``, equivalent to giving only ``native``.
    native_default / decision_default
        This env's original-value snapshot, used for default filling and original-value mode validation.

    Both returned blocks are deepcopies: gymnasium stores a reference to the kwargs dict in
    ``env.unwrapped.spec.kwargs``; without copying it would be mutated across episodes.
    """
    if override is None:
        return copy.deepcopy(decision_default), copy.deepcopy(native_default)
    if not isinstance(override, dict):
        raise SamplingConfigError("sampling_config must be a dict")

    keys = set(override)
    if keys == set(NATIVE_KEYS):
        # old format: only the native block given
        decision, native = copy.deepcopy(decision_default), copy.deepcopy(override)
    elif keys <= {"decision", "native"} and "native" in keys:
        native = copy.deepcopy(override["native"])
        decision = copy.deepcopy(override.get("decision", decision_default))
    else:
        raise SamplingConfigError(
            "sampling_config must be {decision, native} or the old format {parameters, positions}; "
            f"current keys are {sorted(keys)}"
        )
    if not isinstance(native, dict) or set(native) != set(NATIVE_KEYS):
        raise SamplingConfigError("sampling_config.native must contain only parameters and positions")
    if not isinstance(decision, dict):
        raise SamplingConfigError("sampling_config.decision must be a dict")
    return decision, native


# V6: active new-value keys are xhard1..xhard4; v8 adds xhard5 (declared only by SwingXtimes and StopCube).
# The read-only V5 projection also recognizes the old key xhard so old snapshots stay unchanged.
from .difficulty import ALL_NEWVALUE_TIERS

#: Key of the hardest tier in the active new-value family.
XHARD4_KEY = "xhard4"
#: Key names of all "new-value" subtrees in decision (any depth); since v8 includes xhard5, otherwise the xhard5 subtrees of Swing/StopCube cannot be stripped
NEWVALUE_KEYS = frozenset(ALL_NEWVALUE_TIERS)
#: Historical new-value key in V5 frozen snapshots; used only for comparison stripping, not as a usable difficulty tier.
LEGACY_NEWVALUE_KEYS = frozenset({"xhard"})
_STRIP_NEWVALUE_KEYS = NEWVALUE_KEYS | LEGACY_NEWVALUE_KEYS


def _strip_xhard(node):
    """Remove active and V5 historical new-value keys, giving the part visible to the original three tiers."""
    if isinstance(node, dict):
        return {key: _strip_xhard(value) for key, value in node.items() if key not in _STRIP_NEWVALUE_KEYS}
    if isinstance(node, list):
        return [_strip_xhard(item) for item in node]
    return node


def _xhard_shape(node, prefix="", tier=None):
    """List all key paths of new-value subtrees (structure only, not values); returns ``{(tier name, path)}``; the tier name is the first new-value key on the path."""
    out = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            here = tier if tier is not None else (key if key in NEWVALUE_KEYS else None)
            if here is not None:
                out.add((here, path))
            out |= _xhard_shape(value, path, here)
    return out


#: xhard1/2/3 are tiers added after xhard4; when missing they are filled only from the xhard4 config on the same level.
#: v8 hard-codes three keys (formerly ``NEWVALUE_DIFFICULTIES[:-1]``); xhard5 is outside the fill range.
V6_ADDED_KEYS = frozenset({"xhard1", "xhard2", "xhard3"})


def fill_missing_newvalue(decision, decision_default):
    """Fallback for old snapshots (V6 convention 11): subtrees of **V6-added tiers** (xhard1/2/3) missing from the snapshot are deep-copied from the source declaration; existing ones are never touched.

    Only xhard1/2/3 are filled, and only when xhard4 already exists on the same level; the historical xhard in V5 snapshots never triggers filling.
    **Levels where xhard4 itself is missing are never filled**, keeping the old snapshot's original behavior.
    The part visible to the original three tiers is unaffected. Modifies ``decision`` in place and returns it.
    """
    if isinstance(decision, dict) and isinstance(decision_default, dict):
        for key, value in decision_default.items():
            if key in V6_ADDED_KEYS:
                # only fill missing tiers for V6 snapshots that "already have xhard4 on the same level"; V5 historical xhard does not trigger filling
                if key not in decision and XHARD4_KEY in decision:
                    decision[key] = json.loads(json.dumps(value))
            elif key in decision and key not in NEWVALUE_KEYS:
                fill_missing_newvalue(decision[key], value)
            elif key in decision:
                # do not descend into existing new-value tier subtrees (e.g. xhard) to fill keys: structure is checked per tier by assert_native_decision
                pass
    return decision


def assert_native_decision(decision, decision_default, task):
    """``decision`` guard (red line R7; V4 step 2 fork).

    * **Original-value part** (after removing all new-value and historical xhard keys) must equal the original-value snapshot key by key -- any deviation visible to the original three tiers
      must be an explicit new user decision, not silently slipped in under "random offset only".
    * **New-value part**: may only deviate in new-value tier entries already declared in this env's source; adding undeclared keys is not allowed.
    """
    left = json.dumps(_strip_xhard(decision), sort_keys=True, ensure_ascii=False)
    right = json.dumps(_strip_xhard(decision_default), sort_keys=True, ensure_ascii=False)
    if left != right:
        raise SamplingConfigError(
            f"{task}: in original-value parity mode decision must equal the original-value snapshot; the received one differs from the original value"
        )
    shape = _xhard_shape(decision)
    # V6: check per tier -- for every new-value tier present in the snapshot, its key structure must match the source declaration exactly (values may differ);
    # tiers absent from the snapshot are not checked; only V6 snapshots that already have xhard4 get missing tiers filled by fill_missing_newvalue.
    present = {tier for tier, _path in shape}
    declared = {item for item in _xhard_shape(decision_default) if item[0] in present}
    if shape and shape != declared:
        raise SamplingConfigError(
            f"{task}: the xhard entries of decision do not match the source declaration: "
            f"extra {sorted(shape - declared)}, missing {sorted(declared - shape)}"
        )
