"""Utility helpers for validating and normalizing Robomme difficulty hints."""

from __future__ import annotations

from typing import Optional


# ⚠ 白名单是 16 个任务共享的；某任务是否支持某档由它自己的 configs 决定
# V6：四个新值档按难度升序排列，全部沿用同一生成机制并按档读取数值。
NATIVE_DIFFICULTIES = ("easy", "medium", "hard")
#: 新值族的四个公共档，按难度升序。
#: ⚠ v8（1001 方案 §2.1、R8）保持四档不动：VideoRepick、VideoUnmaskSwap、ButtonUnmaskSwap 等把它当
#: 「每个任务都有的档」遍历，加进 xhard5 会让它们读不存在的 configs 或凭空多出 xhard5 子树。
NEWVALUE_DIFFICULTIES = ("xhard1", "xhard2", "xhard3", "xhard4")
#: v8 新增的第五档；只有 SwingXtimes、StopCube 自己的配置里有这一档。
XHARD5 = "xhard5"
#: 全部合法新值档（xhard1～xhard5），只用于全局合法性（VALID_DIFFICULTIES）与族判断／档位号。
ALL_NEWVALUE_TIERS = (*NEWVALUE_DIFFICULTIES, XHARD5)
VALID_DIFFICULTIES = set(NATIVE_DIFFICULTIES) | set(ALL_NEWVALUE_TIERS)
#: 新值族在全序里的档位号（xhard1=1 … xhard5=5）。
_NEWVALUE_TIER = {name: i + 1 for i, name in enumerate(ALL_NEWVALUE_TIERS)}
#: 原版无梯度、不加档的环境，只认 xhard4 这一个新值档（v8 起 StopCube 已扩至 xhard1～5，不再在此列）。
NO_TIER_ENVS = ("MoveCube", "InsertPeg")


def is_newvalue_difficulty(value: Optional[str]) -> bool:
    """族判断：本局难度是否属于新值族；None 与原三档返回 False。"""
    return isinstance(value, str) and value.strip().lower() in _NEWVALUE_TIER


def newvalue_tier(value: Optional[str]) -> int:
    """新值族档位号：xhard1=1 至 xhard5=5；不在族内返回 0。"""
    if not isinstance(value, str):
        return 0
    return _NEWVALUE_TIER.get(value.strip().lower(), 0)


def require_xhard4_only(difficulty: Optional[str], env_name: str) -> None:
    """无梯度环境（v8 起只剩 InsertPeg、MoveCube 调用）拒绝 xhard1/2/3/5，只允许 xhard4，不静默映射。"""
    if is_newvalue_difficulty(difficulty) and difficulty.strip().lower() != "xhard4":
        raise ValueError(
            f"{env_name} 原版无难度梯度、不加档，只支持新值档 'xhard4'；收到 {difficulty!r}"
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
