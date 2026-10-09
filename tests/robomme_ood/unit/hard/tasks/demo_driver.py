"""Demonstration-segment driver for the VideoPlaceButton/VideoPlaceOrder truth tables: following the objects of the demonstration items in the task table (``segment``), performs in the CPU world
pick up, place onto target, press button, stay still, advancing through the real ``step`` to the execution segment; also records the placement events that actually happened in time order,
so tests can derive the answer independently with hand-written rules (instead of reading production's precomputed ``target_target``).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DemoLog:
    placements: list = field(default_factory=list)  # [(cube actor, landing actor, whether after the button)]
    button_at: int | None = None  # after which placement the button was pressed (placement event count)


def run_demo(w, max_steps: int = 2000) -> DemoLog:
    env = w.env
    log = DemoLog()
    w.still()
    for _ in range(max_steps):
        if w.stage >= len(env.task_list) or not env.task_list[w.stage]["demonstration"]:
            return log
        task = env.task_list[w.stage]
        name, seg = task["name"], task.get("segment")
        before = w.stage
        if name == "pick up the cube":
            w.grasp(seg)
        elif name.startswith(("drop the cube onto", "put the cube back")):
            cube = w.agent.held
            w.release_onto(cube, w.xyz(seg)[:2])
            log.placements.append((cube, seg, log.button_at is not None))
        elif name == "press the button":
            w.press(env.button)
        out = w.step()
        assert out["fail"] is False, f"demonstration segment failed: {name}"
        if name == "press the button" and w.stage > before:
            w.unpress(env.button)
            log.button_at = len(log.placements)
    raise AssertionError("demonstration segment did not finish within the step limit")


def target_placements(log: DemoLog, cube, targets):
    """Time series of the asked cube landing on a "target stand" (excluding return-to-origin/table landings): [(landing, whether after the button)]."""
    return [(t, after) for c, t, after in log.placements if c is cube and t in targets]
