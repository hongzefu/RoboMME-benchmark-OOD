"""Read-only export and original-value replay injection of per-episode specs (newtaskRelease-v3 step 4).

The two things required by section 3 / 8.2 of the proposal are handled by one object in this module:

* **Export (path C)**: at the **original call site**, record each sampling point's result read-only, with no extra or missing draws
  and no value changed, and finally seal it into ``episode_spec``.
* **Replay injection (path D)**: the same call site still performs the original sampling (the random stream does not drift, red line R8), but **the values actually used to build the scene
  come from the frozen spec**; the original sampled result is only used for a compatibility check.

This is the core criterion of G4: the proposal explicitly rejects implementations that "redraw the same value while bypassing the spec" -- this module
structurally guarantees that the spec is really consumed by having ``value()`` return the frozen value (not the sampled value).

Relation to the existing injection channel: in the four envs that adopted ``episode_spec`` early, the branch that "skips sampling when a spec is passed"
belongs to the old injection mode and is kept as-is per red line R9, not reused as path D; this module hooks only onto the original random branch,
driven by the new ``native_episode_spec`` switch.

V6 new-value mode: specs exported for xhard1..xhard4 are tagged ``native-newvalue/2``,
the original three tiers are still tagged ``native-parity/1``; the two kinds **must not be cross-fed** (replay rejects a kind that does not match this episode's difficulty).
Every mismatch during new-value spec replay should be attributed to some ``decision`` key where possible (``value(..., decision_key=...)``);
only unattributable ones count as RNG drift; original-value specs still require zero mismatches.

The V7 shared-layout modes "derive (derive envelope)" and "layered replay (``native-layered/3``)" were removed in maintenance-plan stage 1b (W4);
only export and replay modes remain, and new-value family difficulties accept only full ``native-newvalue/2`` specs.
"""

from __future__ import annotations

import copy
from typing import Any

SPEC_KIND = "native-parity/1"
# V6 new-value tier spec version.
SPEC_KIND_NEWVALUE = "native-newvalue/2"
SPEC_KINDS = (SPEC_KIND, SPEC_KIND_NEWVALUE)


def spec_kind_for(difficulty: str | None) -> str:
    """Decide the spec kind by this episode's difficulty: new-value family tiers use the V6 spec kind, everything else (including not given) uses the original-value kind."""
    from .difficulty import is_newvalue_difficulty

    if is_newvalue_difficulty(difficulty):
        return SPEC_KIND_NEWVALUE
    return SPEC_KIND


class EpisodeSpecError(ValueError):
    """The spec's shape, version or identity does not match this episode."""


def _set_path(tree: dict, path: str, value: Any) -> None:
    node = tree
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _get_path(tree: dict, path: str):
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise EpisodeSpecError(f"spec is missing sampling point {path}")
        node = node[part]
    return node


def _plain(value: Any):
    """Convert tensor / numpy scalars into JSON-serializable native objects without changing values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


class SpecRecorder:
    """One per env instance; ``spec=None`` is export mode, otherwise original-value replay mode."""

    def __init__(self, spec: dict | None, task: str, identity: dict | None = None,
                 difficulty: str | None = None):
        self.task = task
        self.identity = dict(identity or {})
        self.mismatches: list[dict] = []
        self.trace: list[dict] = []
        # This episode's spec kind is decided by difficulty (spec_kind_for); without difficulty the behavior is byte-identical to V3.
        self.spec_kind = spec_kind_for(difficulty)
        self.difficulty = difficulty
        if spec is None:
            self.mode = "export"
            self._frozen: dict = {}
            self._document: dict = {
                "spec_kind": self.spec_kind,
                "task": task,
                "identity": self.identity,
            }
        else:
            # new-value family difficulties accept only the V6 full spec; original-value difficulties accept only native-parity/1 (original and new values must not be cross-fed)
            accepted = (self.spec_kind,)
            if not isinstance(spec, dict) or spec.get("spec_kind") not in accepted:
                raise EpisodeSpecError(
                    f"native_episode_spec requires spec_kind in {accepted}"
                    f" (got {spec.get('spec_kind') if isinstance(spec, dict) else type(spec).__name__}; "
                    "original-value and new-value specs must not be cross-fed)"
                )
            if spec.get("task") != task:
                raise EpisodeSpecError(f"spec belongs to {spec.get('task')}, cannot be used for {task}")
            self.mode = "replay"
            self._frozen = copy.deepcopy(spec)
            self._document = copy.deepcopy(spec)

    # ------------------------------------------------------------------
    @property
    def replaying(self) -> bool:
        return self.mode == "replay"

    @property
    def newvalue(self) -> bool:
        return self.spec_kind == SPEC_KIND_NEWVALUE

    def value(self, path: str, drawn: Any, decision_key: str | None = None):
        """Original call site: export mode returns the sampled value and records it; replay mode returns the frozen value and records the sampled value for the compatibility check.

        ⚠ Replay mode **always** returns the frozen value -- even if the original sampling happens to draw the same number, the sampled value must not be used,
        otherwise it becomes the "redraw the same value while bypassing the spec" that the proposal explicitly rejects.

        ``decision_key``: in new-value mode, which ``decision`` key controls this sampling point (e.g. ``number_range.xhard``);
        recorded in mismatch for attribution when replay differs; ignored in original-value mode.
        """
        plain = _plain(drawn)
        if self.mode == "export":
            _set_path(self._document, path, plain)
            self.trace.append({"path": path, "drawn": plain, "source": "draw"})
            return drawn
        frozen = _get_path(self._frozen, path)
        self.trace.append({"path": path, "drawn": plain, "frozen": frozen, "source": "spec"})
        if plain != frozen:
            # a failed compatibility check is only recorded and does not change the fact that "the frozen value is used"; the caller uses it to judge RNG drift.
            self.mismatches.append(self._mismatch(path, plain, frozen, decision_key))
        return frozen

    def _mismatch(self, path, drawn, frozen, decision_key):
        entry = {"path": path, "drawn": drawn, "frozen": frozen}
        if self.newvalue:
            # only new-value mode carries the attribution field; the original-value mismatch shape is byte-identical to V3.
            entry["decision_key"] = decision_key
        return entry

    def unattributed_mismatches(self) -> list[dict]:
        """Unattributable mismatches: all mismatches in original-value mode, those without decision_key in new-value mode."""
        if not self.newvalue:
            return list(self.mismatches)
        return [item for item in self.mismatches if not item.get("decision_key")]

    def record(self, path: str, value: Any) -> None:
        """Record derived quantities or runtime observations read-only; also checked in replay mode.

        The only difference from :meth:`value` is "does not replace the value" -- it is still counted in the trace because this path
        really was accessed in this episode; otherwise SPEC_BINDING would misjudge it as "recorded but not consumed".
        """
        plain = _plain(value)
        self.trace.append({"path": path, "value": plain, "source": "record"})
        if self.mode == "export":
            _set_path(self._document, path, plain)
            return
        try:
            frozen = _get_path(self._frozen, path)
        except EpisodeSpecError:
            _set_path(self._document, path, plain)
            return
        if plain != frozen:
            self.mismatches.append(self._mismatch(path, plain, frozen, None))

    def leaf_paths(self) -> list[str]:
        """All sampling-point paths in the spec (used in replay mode to compute the "recorded but not consumed" unused set)."""
        out: list[str] = []

        def walk(node, prefix):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}.{key}" if prefix else key)
            else:
                out.append(prefix)

        for section in ("layout", "objects", "actions", "initializations"):
            if section in self._frozen:
                walk(self._frozen[section], section)
        return out

    def consumed_paths(self) -> list[str]:
        """Sampling points actually consumed in this episode via ``value()`` / ``record()``."""
        return [item["path"] for item in self.trace]

    def to_dict(self) -> dict:
        document = copy.deepcopy(self._document)
        document.setdefault("spec_kind", self.spec_kind)
        document["task"] = self.task
        document["identity"] = self.identity
        document["provenance"] = {
            "mode": self.mode,
            "value_points": len([item for item in self.trace if item["source"] in ("draw", "spec")]),
            "mismatches": len(self.mismatches),
        }
        if self.newvalue:
            document["provenance"]["unattributed_mismatches"] = len(self.unattributed_mismatches())
        return document
