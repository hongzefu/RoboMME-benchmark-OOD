num2words = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty"
}

num2words_2 = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth"
}

def _unmask_pick_count(self):
    """VideoUnmask / ButtonUnmask 的抓取次数。

    原三档沿用原口径（类属性 ``configs[难度]['pick']``）；xhard 优先读环境实际用来建任务表的次数
    ``xhard_pick_count``（外部 sampling_config 可改 decision 的 xhard 值，类属性不会跟着变）。
    """
    pick = self.env.unwrapped.configs[self.difficulty]['pick']
    # V6 族判断：新值族（xhard1/2/3/xhard）统一读实际次数。本模块会被单测按文件路径单独加载（无包上下文），
    # 故不用相对导入，直接对新值族档名做成员判断（与 utils/difficulty.NEWVALUE_DIFFICULTIES 同一组名字）。
    if isinstance(self.difficulty, str) and self.difficulty.strip().lower() in ("xhard1", "xhard2", "xhard3", "xhard4"):
        pick = getattr(self.env.unwrapped, "xhard_pick_count", pick)
    return pick


def _is_newvalue_tier(self):
    """V6 新值族判断（xhard1～xhard4）；本模块被单测按文件路径单独加载，故不走相对导入。"""
    difficulty = getattr(self, "difficulty", None)
    return isinstance(difficulty, str) and difficulty.strip().lower() in ("xhard1", "xhard2", "xhard3", "xhard4")


def _unmask_multi_pick_clause(color_names, pick):
    """V4 xhard（pick ≥ 3）专用：逐个列出要抓的容器；原三档（pick ≤ 2）不经过这里，文本逐字不变。"""
    parts = [f"pick up the container hiding the {color_names[0]} cube"]
    for k in range(1, pick - 1):
        parts.append(f"next pick up another container hiding the {color_names[k]} cube")
    parts.append(f"finally pick up another container hiding the {color_names[pick - 1]} cube")
    return ", ".join(parts)


def get_language_goal(self, env):
    language_goals = []

    if env == "BinFill":
        color_counts = {
            "red": getattr(self.env.unwrapped, "red_cubes_target_number", 0),
            "blue": getattr(self.env.unwrapped, "blue_cubes_target_number", 0),
            "green": getattr(self.env.unwrapped, "green_cubes_target_number", 0),
        }
        phrases = []
        for color, count in color_counts.items():
            if count <= 0:
                continue
            word = num2words.get(count, str(count))
            noun = "cube" if count == 1 else "cubes"
            phrases.append(f"{word} {color} {noun}")

        if not phrases:
            language_goals.append("put the cubes into the bin, then press the button to stop")
            language_goals.append("put the cubes into the bin and press the button to stop")
        elif len(phrases) == 1:
            language_goals.append(f"put {phrases[0]} into the bin, then press the button to stop")
            language_goals.append(f"put {phrases[0]} into the bin and press the button to stop")
        elif len(phrases) == 2:
            language_goals.append(f"put {phrases[0]} and {phrases[1]} into the bin, then press the button to stop")
            language_goals.append(f"put {phrases[0]} and {phrases[1]} into the bin and press the button to stop")
        else:
            language_goals.append(f"put {', '.join(phrases[:-1])} and {phrases[-1]} into the bin, then press the button to stop")
            language_goals.append(f"put {', '.join(phrases[:-1])} and {phrases[-1]} into the bin and press the button to stop")

    elif env == "PickXtimes":
        repeats = getattr(self.env.unwrapped, "num_repeats", 1)
        target_color = getattr(self.env.unwrapped, "target_color_name", "unknown")
        if repeats > 1:
            word = num2words.get(repeats, str(repeats))
            language_goals.append(f"pick up the {target_color} cube and place it on the target, repeating this action {word} times, then press the button to stop")
            language_goals.append(f"pick up the {target_color} cube and place it on the target, repeating this pick-and-place action {word} times, then press the button to stop")
        else:
            language_goals.append(f"pick up the {target_color} cube and place it on the target, then press the button to stop")

    elif env == "SwingXtimes":
        repeats = getattr(self.env.unwrapped, "num_repeats", 1)
        target_color = getattr(self.env.unwrapped, "target_color_name", "unknown")
        if repeats > 1:
            word = num2words.get(repeats, str(repeats))
            language_goals.append(f"pick up the {target_color} cube, move it to the top of the right-side target, then move it to the top of the left-side target, repeating this back-and-forth motion {word} times, finally press the button to stop")
            language_goals.append(f"pick up the {target_color} cube, move it to the right-side target and then to the left-side target, repeating this right-to-left swing motion {word} times, then put down the cube and press the button to stop")
        else:
            language_goals.append(f"pick up the {target_color} cube, move it to the top of the right-side target, then put it down on the left-side target, finally press the button to stop")
            language_goals.append(f"pick up the {target_color} cube, move it to the right-side target and then put it down on the left-side target, then press the button to stop")

    elif env == "VideoUnmask":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        unmask_pick = _unmask_pick_count(self)
        if unmask_pick > 2:
            language_goals.append(f"watch the video carefully, then {_unmask_multi_pick_clause(color_names, unmask_pick)}")
        elif unmask_pick > 1:
            language_goals.append(f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube")
        else:
            language_goals.append(f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube")

    elif env == "VideoUnmaskSwap":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        if self.pick_times == 2:
            language_goals.append(f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube")
        elif self.pick_times >= 3:
            # V4 xhard（pick 3）：原分支只有 1 抓／2 抓两支，3 抓会落到 1 抓文本
            cube_2_color = color_names[2]
            language_goals.append(f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube, next pick up another container hiding the {cube_1_color} cube, finally pick up another container hiding the {cube_2_color} cube")
        else:
            language_goals.append(f"watch the video carefully, then pick up the container hiding the {cube_0_color} cube")

    elif env == "ButtonUnmask":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        unmask_pick = _unmask_pick_count(self)
        if unmask_pick > 2:
            language_goals.append(f"first press the button, then {_unmask_multi_pick_clause(color_names, unmask_pick)}")
        elif unmask_pick > 1:
            language_goals.append(f"first press the button, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube")
        else:
            language_goals.append(f"first press the button, then pick up the container hiding the {cube_0_color} cube")

    elif env == "ButtonUnmaskSwap":
        color_names = getattr(self.env.unwrapped, "color_names", ["unknown", "unknown", "unknown"])
        cube_0_color = color_names[0]
        cube_1_color = color_names[1]
        if self.pick_times == 2:
            language_goals.append(f"first press both buttons on the table, then pick up the container hiding the {cube_0_color} cube, finally pick up another container hiding the {cube_1_color} cube")
        elif self.pick_times >= 3:
            # V4 xhard（pick 3）：原分支只有 1 抓／2 抓两支，3 抓会落到 1 抓文本
            cube_2_color = color_names[2]
            language_goals.append(f"first press both buttons on the table, then pick up the container hiding the {cube_0_color} cube, next pick up another container hiding the {cube_1_color} cube, finally pick up another container hiding the {cube_2_color} cube")
        else:
            language_goals.append(f"first press both buttons on the table, then pick up the container hiding the {cube_0_color} cube")

    elif env == "VideoPlaceButton":
        target_color_name = self.target_color_name
        target_target_language = self.target_target_language
        if _is_newvalue_tier(self):
            # V6 审查修复 N3/N4（用户「vpb只保留这一句」）：新四档每局只生成一句——
            # before 题用「last placed before」，after 题用「first placed after」；
            # right/immediately（双块档时序不成立）与 previously placed（额外放台时不唯一）三种说法删去。原三档四句不变。
            if target_target_language == "before":
                language_goals.append(f"watch the video carefully, then place the {target_color_name} cube on the target where it was last placed before the button was pressed")
            else:
                language_goals.append(f"watch the video carefully, then place the {target_color_name} cube on the target where it was first placed after the button was pressed")
            return language_goals

        language_goals.append(f"watch the video carefully, then place the {target_color_name} cube on the target right {target_target_language} the button was pressed")
        language_goals.append(f"watch the video carefully, and place the {target_color_name} cube on the target where it was placed immediately {target_target_language} the button was pressed")
        
        language_goals.append(f"watch the video carefully, then place the {target_color_name} cube on the target where it was previously placed {target_target_language} the button was pressed")
        if target_target_language == "before":
            language_goals.append(f"watch the video carefully, then place the {target_color_name} cube on the target where it was last placed before the button was pressed")
        else:
            language_goals.append(f"watch the video carefully, then place the {target_color_name} cube on the target where it was first placed after the button was pressed")
        

    elif env == "VideoPlaceOrder":
        target_color_name = self.target_color_name
        which_in_subset = self.which_in_subset
        num = num2words_2.get(which_in_subset, str(which_in_subset))
        language_goals.append(f"watch the video carefully, then place the {target_color_name} cube on the {num} target it was previously placed on")
        language_goals.append(f"watch the video carefully and place the {target_color_name} cube on the {num} target where it was placed")

    elif env == 'PickHighlight':
        if _is_newvalue_tier(self):
            # V6 审查修复 F1（K2）：新四档两句与新任务链（逐块抓放 + 末尾按钮）一致，并修正 highlighteted 拼写；原三档两句原样保留
            language_goals.append("first press the button, then pick up all highlighted cubes one by one, finally press the button to stop")
            language_goals.append("first press the button, then pick up every cube highlighted with a white area on the table one at a time, finally press the button again to stop")
        else:
            language_goals.append(f"first press the button, then pick up all cubes that have been highlighteted with white areas on the table")
            language_goals.append(f"first press the button, then pick up all highlighted cubes, finally press the button again to stop")

    elif env == "VideoRepick":
        num_repeats = self.num_repeats
        if num_repeats > 1:
            word = num2words.get(num_repeats, str(num_repeats))
            
            language_goals.append(f'watch the video carefully, then repeatedly pick up and put down the same block that was previously picked up for {word} times, finally put it down and press the button to stop')
            if word == "two":
                language_goals.append(f"watch the video carefully, then pick up the same cube that was previously picked up twice, and finally press the button to stop")
                language_goals.append(f"watch the video carefully, identify the cube that was picked up, then pick up and place down the same cube twice, finally press the button to stop")
            else:
                language_goals.append(f"watch the video carefully, then pick up the same cube that was previously picked up {word} times, and finally press the button to stop")
                language_goals.append(f"watch the video carefully, identify the cube that was picked up, then pick up and place down the same cube {word} times, finally press the button to stop")
            
        else:
            language_goals.append(f'watch the video carefully, then pick up the same block that was previously picked up again, finally put it down and press the button to stop')
            language_goals.append(f"watch the video carefully, then pick up the same cube that was previously picked up again, finally press the button to stop")

    elif env == "StopCube":
        repeats = getattr(self.env.unwrapped, "stop_time", 1)
        word = num2words_2.get(repeats, str(repeats))
        language_goals.append(f"press the button to stop the cube just as it reaches the target for the {word} time")
        language_goals.append(f"press the button to stop the cube exactly at the target on its {word} visit")

    elif env == "InsertPeg":
        language_goals.append(f"watch the video carefully, then grasp the same end of the same peg you've picked before and insert it into the same side of the box")
        language_goals.append(f"watch the video carefully, then grasp the same peg at the same end and insert it into the same side of the box as in the video")

    elif env == "MoveCube":
        language_goals.append(f"watch the video carefully, then move the cube to the target in the same manner as before")
        language_goals.append(f"watch the video carefully, then move the cube to the target in the same manner shown in the video")

    elif env == "PatternLock":
        language_goals.append(f"watch the video carefully, then use the stick attached to the robot to retrace the same pattern")
        language_goals.append(f"watch the video carefully, then use the stick attached to the robot to retrace the same pattern shown in the video")

    elif env == "RouteStick":
        language_goals.append(f"watch the video carefully, then use the stick attached to the robot to navigate around the sticks on the table, following the same path") 
        language_goals.append(f"watch the video carefully, then use the stick attached to the robot to navigate around the sticks on the table, following the same path shown in the video")

    return language_goals
