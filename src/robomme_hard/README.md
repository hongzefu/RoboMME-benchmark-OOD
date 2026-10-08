# robomme_hard：RoboMME 五档（xhard1～xhard5）环境包

## ① 一句话与对拍结论

`robomme_hard` 与官方 `robomme` 并列、分层继承：`src/robomme/` 逐字节等于官方 `1fadc0ec`（`UPSTREAM_BYTES=PASS`），本包只放差异——16 个环境类与改过／新增／传递依赖改过模块的 utils、wrapper 复制；依赖闭包干净的官方模块用 shim 借用；`BenchmarkEnvBuilder` 子类化并新增 `dataset="ood"`（V9 定稿，12.333 换包、12.341 起 xhard0 退出 ood：每任务按交付格表 `EXPECTED_CELLS`＝`V9_CELLS` 恰 50 局、43 格共 800 局，builder 每任务发 50 局；官方 hard 12 局（xhard0）默认不前置，设 `ROBOMME_HARD_XHARD0_IN_TEST_HARD=1` 可恢复为 12 + 50 = 62 局、合计 992，源码与清单保留。v8 的 1262 局包与 v7 的 1292 局包分别由 git 历史（12.332 `b462e358` 之前）与标签 `parity-anchor-v7` 保存）。

三侧对拍（A40@greatlakes，16 worker，判定为「输入绑定、结构与任务成功一致，且动作／状态／图像／帧数差异在标定容差内」，不是字节级）：

```text
PARITY_O_P=PASS tier=native compared=144 identity_equal=144 setup_equal=144 schema_equal=144 success_equal=144 both_success=144 tol=PASS action_max=0.0275/0.0413 state_max=0.0274/0.0411 image_mad=0.144/1 frames_max=0/5 sha_equal=143 frames_equal=144 binding_ok=144 shape=16x3x3
PARITY_P_H=PASS tier=native compared=144 identity_equal=144 setup_equal=144 schema_equal=144 success_equal=144 both_success=144 tol=PASS action_max=0.0275/0.0413 state_max=0.0274/0.0411 image_mad=0.144/1 frames_max=0/5 sha_equal=143 frames_equal=144 binding_ok=144 shape=16x3x3
PARITY_O_H=PASS tier=native compared=144 identity_equal=144 setup_equal=144 schema_equal=144 success_equal=144 both_success=144 tol=PASS action_max=0/0.0413 state_max=0/0.0411 image_mad=0/1 frames_max=0/5 sha_equal=144 frames_equal=144 binding_ok=144 shape=16x3x3
PARITY_P_H=PASS tier=xhard compared=165 identity_equal=165 setup_equal=165 schema_equal=165 success_equal=165 both_success=165 tol=PASS action_max=0/0.0413 state_max=0/0.0411 image_mad=0/1 frames_max=0/5 sha_equal=165 frames_equal=165 binding_ok=165 shape=13x3x3+16x3
```

O 官方（编排 `d53f21a7` + 环境源码 `1fadc0ec`）、P 修改前（tag `pre-hard-split`）、H 本包（拆包后 HEAD）。原三档 16 任务 × 3 档 × 3 局 = 144，O↔H 全部逐字节相同；xhard 为 xhard1/2/3 各 13 任务 × 3 局 + xhard4 16 任务 × 3 局 = 165，与 S4 交付全部逐字节相同。详见 `docs/validation/newtask-v6/hard-split/stage4.md`。

v7（12.237 起）另跑：上面四行以 tag `parity-anchor-v6` 为 P 侧重跑全部 PASS；xhard0 O:H 16 任务 × 12 局 = 192 局 sha 全等（VideoPlaceOrder 2 局两侧都生成失败，单列 `both_fail`）；xhard0 reset 层对拍 `XHARD0_RESET_PARITY=PASS compared=192 det_diff=0`。v7 生成与两次生成对拍见 `docs/validation/newtask-v7/`。

## ② 六档定值表与交付格（v8）

各档取值表见 [`scripts/README.md`](../../scripts/README.md) 第 3 节（V9 取值与 v8 相同，只有 MoveCube xhard4 生成区域改为圆环 r_in 0.24／r_out 0.42、base_dist [0.31, 0.80]；每格局数按 V9 每任务 50 局平分，见该节局数表），源 `docs/plans/1001-newtask-v8-xhard-gradient-plan.md` 第一部分表 1 与 `docs/plans/1002-newtask-v9-movecube-region-800-plan.md` 表 2。以下为 v8 交付格的历史要点（V9 局数已变）：新值档扩到 xhard1～xhard5，但只有 SwingXtimes、StopCube 有 xhard5（各档 10 局）；PickXtimes 交付 xhard1～3（17／17／16 局，抓放 6／7／8 次）；VideoUnmask／ButtonUnmask 交付 xhard1～4 各 20 局；BinFill、两个 Swap、VideoPlaceButton、VideoPlaceOrder、PickHighlight、VideoRepick 交付 xhard1～2 各 40 局；RouteStick、PatternLock 交付 xhard1～3（27／27／26 局，段数／节点数为区间）；MoveCube、InsertPeg 只有 xhard4 20 局。各格布局独立抽签（`layout_rule={"mode":"independent"}`，`layout_parent` 全为 null），不再共用母布局。步数上限 xhard0 1300、xhard1～5 一律 1600（`TIER_MAX_STEPS`，＝`EXEC_CAP`；抽样时已过滤执行步超过 1600 的候选）。xhard0 就是官方 hard（配置区间见该表 xhard0 列）；xhard0 桌面实测放置数（官方 test hard 12 局）：VideoUnmask 容器均值 5.42、ButtonUnmask 5.25（最大 6），Swap 两任务容器 4，PickHighlight 总块 6，PickXtimes 桌面块 3，BinFill 总块 10～12。v7（历史）xhard1～4 共用每任务 20 个母布局（xhard4 抽签、低档派生，`layout_parent` 指回母行）；v8 起各档独立抽。PickXtimes／SwingXtimes 第 4 个干扰块用橙色（`utils/xhard.py::BLOCK_DISTRACTOR_COLORS`，不动四个 Unmask／Swap 共用的三色池）。

### ②′ 历史：v6 四档定稿表（逐字搬自 `docs/plans/0925-newtask-release-v6-plan.md` 第一部分 §三；v7 起不再使用）

| 环境 | 梯度维度 | hard | xhard1 | xhard2 | xhard3 | xhard4 | 备注 |
|---|---|---|---|---|---|---|---|
| BinFill | 投入块数（总块 12） | [3,5] | 6 | 7 | 8 | 9 | 无演示段，8/9 块约 1500～1900 步，允许超 1301 |
| PickXtimes | 次数 / 干扰块 | [4,5] / 0 | [6,7] / 1 | [8,9] / 2 | [10,12] / 3 | [13,15] / 3 | 干扰颜色表若 ≥4 色则 xhard4 取 4 |
| SwingXtimes | 轮数 / 干扰块 | 3 / 0 | [4,5] / 1 | [6,7] / 2 | [8,9] / 3 | [10,11] / 3 | reset 约 96%（放圆盘采样失败，与干扰无关） |
| PickHighlight | pick / 总块 | 3 / 6 | 4 / 7 | 5 / 8 | 6 / 9 | 7 / 10 | 干扰数 = 总块 − pick，均值全程 3（不作梯度）；总块 10 时 reset 73% |
| VideoUnmask | 干扰容器 / pick | 0（15 容器）/ 2 | 8 / 2 | 10 / 3 | 13 / 3 | 15 / 3 | pick ≤3（可藏 cube 的容器只有 3 个），靠干扰数区分；内环容器数不作梯度 |
| ButtonUnmask | 干扰容器 / pick | 0 / 2 | 8 / 2 | 10 / 3 | 12 / 3 | 14 / 3 | 同上 |
| VideoUnmaskSwap | swap / pick / 外环干扰 | [2,3] / 2 / 0 | [4,5] / 2 / 4 | [6,7] / 3 / 6 | [8,9] / 3 / 8 | [10,12] / 3 / 10 | 外环含 cube 数 = 干扰一半；交换步数 xhard1 50、其余 33 |
| ButtonUnmaskSwap | swap / pick / 外环干扰 | [2,3] / 2 / 0 | 4 / 2 / 4 | 5 / 3 / 6 | [6,7] / 3 / 8 | [8,9] / 3 / 10 | 同上 |
| VideoRepick | 块数 / swap / repick | 另一种任务 | 4 / [3,4] / 2 | 5 / [5,6] / 3 | 6 / [7,8] / 4 | 7 / [9,12] / [5,6] | 4～6 块 reset 约 60%；7 块待实测 |
| PatternLock | 节点数（5×5，不重访，预算 20000） | [4,8] | [9,12] | [13,16] | [17,20] | [21,25] | 执行段 ≤ 约 850 步 |
| RouteStick | 段数 L | [4,7] | [8,10] | [11,13] | [14,16] | [17,21] | 执行段 = 50·L |
| VideoPlaceButton | 放台次数（都放回原位） | 2（放桌面） | 1 块 3 次 | 1 块 4 次 | 2 块 5 次 | 2 块 6 次 | 额外放台 = 放到无关台（原版 `additional_place` 语义），见四.9；xhard3/4 台数 4→5（`config_xhard3/4["targets"]=5`，xhard1/2 仍 4），额外放台按完整序列占用表抽取，before 题答案 = 按钮前最后一次放置的台 |
| VideoPlaceOrder | 总放台次数（都放回原位） | 1 块 v∈[2,4] | 2 块 (2,3)=5 | 2 块 (3,3)=6 | 2 块 (3,4)=7 | 2 块 (4,4)=8 | 每档总数定值，哪块多访问随机 |
| MoveCube | 不加档 | 原三档同值 | — | — | — | 圆环 U（V9：r 0.24–0.42 ∩ 离基座 0.31–0.80） | 只有 xhard4 |
| InsertPeg / StopCube | 不加档 | 原三档同值 | — | — | — | 原 xhard 改名 | 数值不动 |


## ③ 逐文件：复制／借用／子类／新增（`scripts/parity/upstream_guard.py manifest-md` 生成）

| 文件 | 做法 | 官方对应 |
|---|---|---|
| `src/robomme_hard/__init__.py` | 复制 | `src/robomme/__init__.py` |
| `src/robomme_hard/env_record_wrapper/DemonstrationWrapper.py` | 复制 | `src/robomme/env_record_wrapper/DemonstrationWrapper.py` |
| `src/robomme_hard/env_record_wrapper/EndeffectorDemonstrationWrapper.py` | 借用 shim | `robomme.env_record_wrapper.EndeffectorDemonstrationWrapper` |
| `src/robomme_hard/env_record_wrapper/FailAwareWrapper.py` | 借用 shim | `robomme.env_record_wrapper.FailAwareWrapper` |
| `src/robomme_hard/env_record_wrapper/MultiStepDemonstrationWrapper.py` | 借用 shim | `robomme.env_record_wrapper.MultiStepDemonstrationWrapper` |
| `src/robomme_hard/env_record_wrapper/OraclePlannerDemonstrationWrapper.py` | 复制 | `src/robomme/env_record_wrapper/OraclePlannerDemonstrationWrapper.py` |
| `src/robomme_hard/env_record_wrapper/RecordWrapper.py` | 复制 | `src/robomme/env_record_wrapper/RecordWrapper.py` |
| `src/robomme_hard/env_record_wrapper/__init__.py` | 复制 | `src/robomme/env_record_wrapper/__init__.py` |
| `src/robomme_hard/env_record_wrapper/episode_dataset_resolver.py` | 借用 shim | `robomme.env_record_wrapper.episode_dataset_resolver` |
| `src/robomme_hard/env_record_wrapper/hard_builder.py` | 子类 | `robomme.env_record_wrapper.episode_config_resolver.BenchmarkEnvBuilder` |
| `src/robomme_hard/env_record_wrapper/hard_specs.py` | 新增 | `—` |
| `src/robomme_hard/logging_utils.py` | 借用 shim | `robomme.logging_utils` |
| `src/robomme_hard/robomme_env/BinFill.py` | 复制 | `src/robomme/robomme_env/BinFill.py` |
| `src/robomme_hard/robomme_env/ButtonUnmask.py` | 复制 | `src/robomme/robomme_env/ButtonUnmask.py` |
| `src/robomme_hard/robomme_env/ButtonUnmaskSwap.py` | 复制 | `src/robomme/robomme_env/ButtonUnmaskSwap.py` |
| `src/robomme_hard/robomme_env/InsertPeg.py` | 复制 | `src/robomme/robomme_env/InsertPeg.py` |
| `src/robomme_hard/robomme_env/MoveCube.py` | 复制 | `src/robomme/robomme_env/MoveCube.py` |
| `src/robomme_hard/robomme_env/PatternLock.py` | 复制 | `src/robomme/robomme_env/PatternLock.py` |
| `src/robomme_hard/robomme_env/PickHighlight.py` | 复制 | `src/robomme/robomme_env/PickHighlight.py` |
| `src/robomme_hard/robomme_env/PickXtimes.py` | 复制 | `src/robomme/robomme_env/PickXtimes.py` |
| `src/robomme_hard/robomme_env/RouteStick.py` | 复制 | `src/robomme/robomme_env/RouteStick.py` |
| `src/robomme_hard/robomme_env/StopCube.py` | 复制 | `src/robomme/robomme_env/StopCube.py` |
| `src/robomme_hard/robomme_env/SwingXtimes.py` | 复制 | `src/robomme/robomme_env/SwingXtimes.py` |
| `src/robomme_hard/robomme_env/VideoPlaceButton.py` | 复制 | `src/robomme/robomme_env/VideoPlaceButton.py` |
| `src/robomme_hard/robomme_env/VideoPlaceOrder.py` | 复制 | `src/robomme/robomme_env/VideoPlaceOrder.py` |
| `src/robomme_hard/robomme_env/VideoRepick.py` | 复制 | `src/robomme/robomme_env/VideoRepick.py` |
| `src/robomme_hard/robomme_env/VideoUnmask.py` | 复制 | `src/robomme/robomme_env/VideoUnmask.py` |
| `src/robomme_hard/robomme_env/VideoUnmaskSwap.py` | 复制 | `src/robomme/robomme_env/VideoUnmaskSwap.py` |
| `src/robomme_hard/robomme_env/__init__.py` | 复制 | `src/robomme/robomme_env/__init__.py` |
| `src/robomme_hard/robomme_env/utils/SceneGenerationError.py` | 借用 shim | `robomme.robomme_env.utils.SceneGenerationError` |
| `src/robomme_hard/robomme_env/utils/__init__.py` | 复制 | `src/robomme/robomme_env/utils/__init__.py` |
| `src/robomme_hard/robomme_env/utils/adjacent.py` | 借用 shim | `robomme.robomme_env.utils.adjacent` |
| `src/robomme_hard/robomme_env/utils/bin_collision.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/choice_action_mapping.py` | 借用 shim | `robomme.robomme_env.utils.choice_action_mapping` |
| `src/robomme_hard/robomme_env/utils/constant.py` | 借用 shim | `robomme.robomme_env.utils.constant` |
| `src/robomme_hard/robomme_env/utils/difficulty.py` | 复制 | `src/robomme/robomme_env/utils/difficulty.py` |
| `src/robomme_hard/robomme_env/utils/episode_spec.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/generate_sample_action.py` | 借用 shim | `robomme.robomme_env.utils.generate_sample_action` |
| `src/robomme_hard/robomme_env/utils/object_generation.py` | 复制 | `src/robomme/robomme_env/utils/object_generation.py` |
| `src/robomme_hard/robomme_env/utils/obschange.py` | 借用 shim | `robomme.robomme_env.utils.obschange` |
| `src/robomme_hard/robomme_env/utils/oracle_action_matcher.py` | 借用 shim | `robomme.robomme_env.utils.oracle_action_matcher` |
| `src/robomme_hard/robomme_env/utils/planner_denseStep.py` | 借用 shim | `robomme.robomme_env.utils.planner_denseStep` |
| `src/robomme_hard/robomme_env/utils/planner_fail_safe.py` | 借用 shim | `robomme.robomme_env.utils.planner_fail_safe` |
| `src/robomme_hard/robomme_env/utils/reset_panda.py` | 借用 shim | `robomme.robomme_env.utils.reset_panda` |
| `src/robomme_hard/robomme_env/utils/route.py` | 复制 | `src/robomme/robomme_env/utils/route.py` |
| `src/robomme_hard/robomme_env/utils/rpy_util.py` | 借用 shim | `robomme.robomme_env.utils.rpy_util` |
| `src/robomme_hard/robomme_env/utils/sampling_config.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/save_reset_video.py` | 借用 shim | `robomme.robomme_env.utils.save_reset_video` |
| `src/robomme_hard/robomme_env/utils/segmentation_utils.py` | 复制 | `src/robomme/robomme_env/utils/segmentation_utils.py` |
| `src/robomme_hard/robomme_env/utils/statechange.py` | 借用 shim | `robomme.robomme_env.utils.statechange` |
| `src/robomme_hard/robomme_env/utils/subgoal_evaluate_func.py` | 复制 | `src/robomme/robomme_env/utils/subgoal_evaluate_func.py` |
| `src/robomme_hard/robomme_env/utils/subgoal_language.py` | 复制 | `src/robomme/robomme_env/utils/subgoal_language.py` |
| `src/robomme_hard/robomme_env/utils/subgoal_planner_func.py` | 复制 | `src/robomme/robomme_env/utils/subgoal_planner_func.py` |
| `src/robomme_hard/robomme_env/utils/swap_uniform.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/task4recovery.py` | 复制 | `src/robomme/robomme_env/utils/task4recovery.py` |
| `src/robomme_hard/robomme_env/utils/task_goal.py` | 复制 | `src/robomme/robomme_env/utils/task_goal.py` |
| `src/robomme_hard/robomme_env/utils/unmask_distractor_sampler.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/unmask_distractors.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/unmask_swap_xhard.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/vqa_options.py` | 复制 | `src/robomme/robomme_env/utils/vqa_options.py` |
| `src/robomme_hard/robomme_env/utils/xhard.py` | 新增 | `—` |
| `src/robomme_hard/robomme_env/utils/xhard_home_site.py` | 新增 | `—` |

## ④ 机制

- **`sampling_config` 两块**：每个环境的 `native_blocks(cls, release=...)` 返回 `decision`（按档位的新值取值）与 `native`（原三档参数与位置），`gym.make(..., sampling_config={"decision","native"})` 显式传入；`utils/sampling_config.py::assert_native_decision` 保证去掉新值键后与原值全等。
- **`SpecRecorder` 导出与回注**（`utils/episode_spec.py`）：`spec=None` 时导出（每个取值点在原调用点记下）；传 `native_episode_spec` 时回注——原抽样照常发生，但建场景用的值一律取冻结规格，原抽样只作兼容核验记入 `mismatches`；`record()` 只记录不回注的观测值。评估侧用 `env_record_wrapper.spec_binding(env)` 取摘要：回注点（trace `source="spec"`）必须零差，记录点（`source="record"`）浮点差 ≤ `RECORDED_FLOAT_TOL=1e-5` 计 `recorded_drift`（U-13 方案甲）。
- **jsonl 两段与两个哈希**（`env_record_wrapper/hard_specs.py`；包内 v8 为 `schema="hard-specs/4"`，经 `load_specs_root(root, EXPECTED_CELLS)` 整根校验；v7 的 `hard-specs/3`、v6 的 `hard-specs/2` 的读写校验已删除）：签 `task tier candidate episode seed attempt spec spec_sha256`（v7 起另加 `layout_parent`；/4 的 header 另签 `exec_cap` 与逐任务 `delivery_per_cell`）由 `identity_sha256` 覆盖；结果 `selected tried initial_selected rollout` 不进身份。`delivery_sha256` 盖排序后的 `(task, tier, candidate, seed, spec_sha256, rollout.h5_sha256)`（只取 `selected` 且 `rollout.status=="ok"`），锁住正式交付集合。
- **seed 偏移**：`seed = offset + env_code × 100000 + episode × 100 + attempt`。v8 按档偏移：xhard1 16e6、xhard2 18e6、xhard3 20e6、xhard4 22e6、xhard5 24e6（`hard_specs.seed_rule_for(tier, "v8")`，档与档 seed 两两不交）；v7 四档同一个 offset 14e6（规则族已删除）；v6 为 xhard4 6e6、xhard1 8e6、xhard2 10e6、xhard3 12e6。都与 train／test／val／heldout 不重叠。xhard0 用官方 test 原 seed。
- **注册表与命名空间归属**：16 个环境类 `@register_env("<id>", override=True)`，导入本包即接管 16 个 id（与导入顺序无关）；包 `__init__` 末尾断言 `REGISTERED_ENVS[uid].cls.__module__` 属于本包，并断言 16 个环境模块与 `utils` 包里本包同名可调用对象都属于本包（专挡官方 `subgoal_evaluate_func` 的星号导入回灌）。同进程要官方行为须另开只导入 `robomme` 的进程。
- **借用闭包规则**：借用资格按传递闭包判，不按单文件字节——shim 目标在官方源码上的依赖闭包不得含任何被本包复制的模块（`BORROWED_DEPS`）。复制文件里的绝对导入 `robomme.<path>`：`<path>` 在 shim 清单内的保持，是复制件的一律改成 `robomme_hard.`（`ABS_IMPORT`）。

## ⑤ 使用

```python
from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder   # 与官方唯一不同的 import
builder = BenchmarkEnvBuilder(env_id="BinFill", dataset="ood", action_space="joint_angle", max_steps=1600)  # V9 入口固定 1600
for episode in range(builder.get_episode_num()):             # BinFill 50 局：xhard1、xhard2 各 25 局
    env = builder.make_env_for_episode(episode)               # 与官方一样不传 max_steps，五档一律 1600
    obs, info = env.reset()
```

- 与官方 `dataset="test"` 的对应：`test` 每任务 50 局、三档混排、步数上限一个数；`ood` 每任务 50 局（见 ①），按 xhard1→xhard5 只排该任务交付的档、档内按 `candidate` 升序（开关 `ROBOMME_HARD_XHARD0_IN_TEST_HARD=1` 时 xhard0 12 局在前，按官方原 episode 号）；步数上限由入口 `evaluation_hard.py` 构造时写死 `max_steps=1600`、逐局不传（五档一律 1600；官方入口为 1300）；包内常量表 `TIER_MAX_STEPS = {xhard0: 1300, xhard1～xhard5: 1600}` 入口不再引用、只剩 eval-official 评估流水线在用（去留待定，`docs/1002-pending-decisions.md` B5）。上限不从 episode 或规格文件读（规格 header 只签 `exec_cap`＝1600，行里没有 `max_steps`）。`specs_root` 覆盖（或环境变量 `ROBOMME_HARD_SPECS_ROOT`）可指局部 v8／v9 根，只发存在的档；v7 `hard-specs/3` 根在换包后不再被 builder 接受。
- `builder.resolve_identity(episode)` 只读返回 `{episode, tier, candidate, seed, spec_sha256, source_run}`（开关打开时的 xhard0 行另带 `source_dataset`、`source_episode`，`candidate` 为空）；`spec_binding(env)` 须在 `reset()` 之后调用。
- `train`／`test`／`val` 行为同官方；只有四个 Unmask 任务的 `train` 元数据改读本包 `env_metadata/train`（400 条）。`override_metadata_path` 语义同官方。
- 生产命令（不随包分发，在仓库 `scripts/injection-dev/` 下，路径直跑；详见 `scripts/README.md` 第 4 节）：每档一次 `freeze_specs.py --tier <档> --seed-profile v8 --cells v9`（v8 seed 规则，V9 沿用；档内逐任务独立抽，只抽交付格）→ `generate_h5.py --mode continue --specs <规格根> --cells v9`（或 `split` → 各片 `continue` → `aggregate`）（只认 `hard-specs/4`，逐格递补、执行步超过 1600 记 `exec_over_cap` 并递补）。V9 实际只对 MoveCube 整任务重抽（`--cells v9shard1`），其余 720 局复用 V8 交付，见 `scripts/README.md` 第 4 节。

## ⑥ 红线（0927 计划第二部分 R1～R5、R11）

- R1 `src/robomme/**` 是官方 `1fadc0ec` 原样，不放任何自有文件（`UPSTREAM.json`、说明一律放本包或 `docs/`）。
- R2 `scripts/parity/official/` 四文件不改；shim 不含逻辑（≤ 3 行非注释行，守卫检查）。
- R3 `identity_sha256` 与 `delivery_sha256` 的覆盖范围不变；要变就换 `schema` 版本并重跑身份类闸门。
- R4 第二阶段回写只改 `selected`、`tried`、`rollout`；锁在 `<specs>.lock`；回写前整份文件 sha 必须与读入时相同。
- R5 同进程导入 `robomme_hard` 后 16 个 id 归它；需要官方行为另开进程；O／P 侧对拍进程不得导入 `robomme_hard`。
- R11 对拍只在 A40@greatlakes 上生成；各侧 manifest 如实记录硬件与驱动，驱动不明写 `unknown`；Ada 产物不混入 `PARITY_*`。

## ⑦ `dataset_for_parent="test"` 绕行

官方父类 `__init__` 的 `_ALLOWED_DATASETS` 只认 `train/test/val`，而官方代码不能改。子类对 `ood` 先以 `dataset="test"` 过父类校验，再把 `self.dataset` 改回 `"ood"`；父类顺手读的 test 元数据随即清空、不被使用。官方父类签名若变化属于升级流程（`UPSTREAM.json` 换锚点、重跑全部闸门）。
