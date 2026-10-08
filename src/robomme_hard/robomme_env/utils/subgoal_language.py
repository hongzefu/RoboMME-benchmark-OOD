from ...logging_utils import logger


# V4 E2：序数表扩到 20（与 utils/task_goal.py::num2words 的覆盖范围对齐），超出用规范英文序数兜底。
# ⚠ 前十项必须与改前逐字相同，否则原三档的 subgoal 文本变化、V0/V1 直接失败。
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
             "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
             "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth")


def _ordinal_word(idx):
    if idx < 0:
        raise ValueError(f"Invalid index: {idx}")
    if idx < len(_ORDINALS):
        return _ORDINALS[idx]
    n = idx + 1  # 序数是 1-based
    suffix = "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def get_subgoal_with_index(idx, template, **kwargs):
    return template.format(idx=_ordinal_word(idx), **kwargs)



if __name__ == "__main__":
    logger.debug(get_subgoal_with_index(0, "pick up the {idx} {color} cube", color="red"))
