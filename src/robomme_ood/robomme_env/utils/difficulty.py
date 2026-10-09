"""Utility helpers for validating and normalizing Robomme difficulty hints."""

from __future__ import annotations

from typing import Optional


# ⚠ The whitelist is shared by all 16 tasks; whether a task supports a tier is decided by its own configs
# V6: the four new-value tiers are ordered by ascending difficulty; all reuse the same generation mechanism and read values per tier.
NATIVE_DIFFICULTIES = ("easy", "medium", "hard")
#: The four common tiers of the new-value family, in ascending difficulty.
#: ⚠ v8 (1001 proposal §2.1, R8) keeps these four tiers unchanged: VideoRepick, VideoUnmaskSwap, ButtonUnmaskSwap, etc. iterate over it as
#: "the tiers every task has"; adding xhard5 would make them read nonexistent configs or spawn a spurious xhard5 subtree.
NEWVALUE_DIFFICULTIES = ("xhard1", "xhard2", "xhard3", "xhard4")
#: Fifth tier added in v8; only SwingXtimes and StopCube have it in their own configs.
XHARD5 = "xhard5"
#: All valid new-value tiers (xhard1..xhard5); used only for global validity (VALID_DIFFICULTIES) and family checks / tier numbers.
ALL_NEWVALUE_TIERS = (*NEWVALUE_DIFFICULTIES, XHARD5)
VALID_DIFFICULTIES = set(NATIVE_DIFFICULTIES) | set(ALL_NEWVALUE_TIERS)
#: Tier number of the new-value family in the total order (xhard1=1 ... xhard5=5).
_NEWVALUE_TIER = {name: i + 1 for i, name in enumerate(ALL_NEWVALUE_TIERS)}
#: Envs with no difficulty gradient in the original release and no added tiers accept only the single new-value tier xhard4 (since v8 StopCube is extended to xhard1..5 and is no longer listed here).
NO_TIER_ENVS = ("MoveCube", "InsertPeg")


def is_newvalue_difficulty(value: Optional[str]) -> bool:
    """Family check: whether this episode's difficulty belongs to the new-value family; None and the original three tiers return False."""
    return isinstance(value, str) and value.strip().lower() in _NEWVALUE_TIER


def newvalue_tier(value: Optional[str]) -> int:
    """New-value family tier number: xhard1=1 to xhard5=5; returns 0 outside the family."""
    if not isinstance(value, str):
        return 0
    return _NEWVALUE_TIER.get(value.strip().lower(), 0)


def require_xhard4_only(difficulty: Optional[str], env_name: str) -> None:
    """Gradient-free envs (since v8 only InsertPeg and MoveCube call this) reject xhard1/2/3/5 and allow only xhard4, with no silent mapping."""
    if is_newvalue_difficulty(difficulty) and difficulty.strip().lower() != "xhard4":
        raise ValueError(
            f"{env_name} has no difficulty gradient in the original release and no added tiers; only the new-value tier 'xhard4' is supported; got {difficulty!r}"
        )


def normalize_robomme_difficulty(value: Optional[str]) -> Optional[str]:
    """Return a canonical difficulty string or ``None`` if no override was provided."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            "difficulty must be a string (got "
            f"{type(value).__name__!r})."
        )

    normalized = value.strip().lower()
    if normalized not in VALID_DIFFICULTIES:
        raise ValueError(
            "Unsupported difficulty level. Available options: "
            f"{sorted(VALID_DIFFICULTIES)}."
        )

    return normalized
