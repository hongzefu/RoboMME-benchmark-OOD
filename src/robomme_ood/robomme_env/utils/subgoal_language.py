from ...logging_utils import logger


# V4 E2: ordinal table extended to 20 (aligned with the coverage of utils/task_goal.py::num2words); beyond that, fall back to canonical English ordinals.
# ⚠ The first ten entries must stay byte-identical to the pre-change version, otherwise the subgoal text of the original three tiers changes and V0/V1 fail outright.
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
             "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
             "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth")


def _ordinal_word(idx):
    if idx < 0:
        raise ValueError(f"Invalid index: {idx}")
    if idx < len(_ORDINALS):
        return _ORDINALS[idx]
    n = idx + 1  # ordinals are 1-based
    suffix = "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def get_subgoal_with_index(idx, template, **kwargs):
    return template.format(idx=_ordinal_word(idx), **kwargs)



if __name__ == "__main__":
    logger.debug(get_subgoal_with_index(0, "pick up the {idx} {color} cube", color="red"))
