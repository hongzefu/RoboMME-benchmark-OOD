# OOD 评估入口说明（`scripts/evaluation_ood.py`）

本仓库在官方 RoboMME benchmark（`016ac1c4`）之上只新增环境包 `src/robomme_hard/` 与一个评估示例入口 `scripts/evaluation_ood.py`；官方三个入口 `dataset_replay.py`、`evaluation.py`、`run_example.py` 与上游逐字节相同。

## 两个数据集

| 数据集 id | 内容 | 局数 | 步数上限 |
|---|---|---|---|
| `hard-verify` | xhard0：官方 test 元数据里每任务 `difficulty=="hard"` 的 12 局（原 episode 3, 7, …, 47，编为 episode 0..11） | 16 任务 × 12 局 = 192 | 1300 |
| `ood` | xhard1～xhard5 新值局（规格在 `src/robomme_hard/env_metadata/ood/<档>/specs.jsonl`，按档、按候选序号拼成 episode 0..49） | 16 任务 × 50 局 = 800 | 1800 |

- **xhard0 = hard-verify**：数据集 id 保持 `hard-verify`，它就是官方 hard 难度的那 12 局，用来核对新环境包在官方原生场景上与官方行为一致；`xhard1`～`xhard5` 是档位名，不是合法的 `dataset` 取值。
- `hard-verify` 的 1300 与官方 `evaluation.py`（MME-VLA 实验口径）相同。
- **V9 交付集按 1600 过滤，1800 只放宽上限、不改已交付局**：`ood` 的 800 局是在 1600 步上限下生成并筛选的成功局；评估时上限放到 1800 只给策略留余量，交付的局本身（seed、规格、回注参数）一条不变。

## 运行方式

```bash
uv run python scripts/evaluation_ood.py
```

默认跑 `ood`；要跑 `hard-verify`，把脚本里的 `DATASET = "ood"` 改成 `DATASET = "hard-verify"`。脚本与官方 `evaluation.py` 一样用 `DummyModel` 演示评估循环，把自己的模型替换进 `dummy_model.predict(...)` 即可；视频落在 `runs/saved_videos/`。导入 `robomme_hard` 会以 `override=True` 接管 16 个环境 id，同一进程内要官方行为须另开只导入 `robomme` 的进程。

## 与官方 `evaluation.py` 的逐行差异

只有三处单行替换（import 改从 `robomme_hard` 导入同名 `BenchmarkEnvBuilder`、`dataset=DATASET`、`max_steps=DATASET_MAX_STEPS[DATASET]`）加一段选数据集的插入，其余逐字节相同：

```diff
--- scripts/evaluation.py
+++ scripts/evaluation_ood.py
@@ -7,7 +7,7 @@
 import imageio
 
 from pathlib import Path
-from robomme.env_record_wrapper import BenchmarkEnvBuilder
+from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder
 
 class VideoRecorder:
     BORDER_COLOR = (255, 0, 0)
@@ -72,6 +72,10 @@
         return self.base_action + noise
 
 
+# 两个数据集：hard-verify＝xhard0（官方 test 里每任务 difficulty=="hard" 的 12 局，1300 步）；
+# ood＝xhard1～5 新值局（每任务 50 局，1800 步）。改 DATASET 选择其一，默认 ood。
+DATASET_MAX_STEPS = {"hard-verify": 1300, "ood": 1800}
+DATASET = "ood"
 TASKS = BenchmarkEnvBuilder.get_task_list()
 MODEL_SEED = 7 # 7, 42, 0
 dummy_model = DummyModel(seed=MODEL_SEED)
@@ -80,9 +84,9 @@
 for task in TASKS:
     env_builder = BenchmarkEnvBuilder(
         env_id=task,
-        dataset="test",
+        dataset=DATASET,
         action_space="joint_angle", # change this to your model's action space
-        max_steps=1300,  # we set 1300 in MME-VLA experiments.
+        max_steps=DATASET_MAX_STEPS[DATASET],  # 按数据集固定（不按档查表）
     )
     episode_count = env_builder.get_episode_num()
     for episode in range(episode_count):
```

## 模型与工具在哪里

- 本仓库不含任何模型代码与生成工具。评估模型（GroundSG 等）与对拍工具在私有评估仓 `RoboMME-benchmark-OOD-eval` 里，以 submodule 方式调用本仓库的 `robomme_hard`。
- `ood` 局的生成链路（规格采样、回注、筛选）见旧仓归档 tag `archive-newtask-v9-20261008`。
