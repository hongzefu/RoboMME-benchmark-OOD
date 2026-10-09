"""VideoPlaceButton／VideoPlaceOrder 真值表的演示段驱动：按任务表里演示项的对象（``segment``）在 CPU 世界里
执行抓起、放到目标、按按钮、静止，经真实 ``step`` 推进到执行段；同时把实际发生的放置事件按时间记下来，
供测试用手写规则独立推出答案（而不是读生产算好的 ``target_target``）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DemoLog:
    placements: list = field(default_factory=list)  # [(方块 actor, 落点 actor, 是否在按钮之后)]
    button_at: int | None = None  # 按钮在第几次放置之后按下（放置事件计数）


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
        assert out["fail"] is False, f"演示段失败：{name}"
        if name == "press the button" and w.stage > before:
            w.unpress(env.button)
            log.button_at = len(log.placements)
    raise AssertionError("演示段未在步数上限内走完")


def target_placements(log: DemoLog, cube, targets):
    """被问方块落到「目标台」（不含放回原位／桌面落点）的时间序列：[(落点, 是否在按钮之后)]。"""
    return [(t, after) for c, t, after in log.placements if c is cube and t in targets]
