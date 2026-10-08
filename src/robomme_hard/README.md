# robomme_hard：RoboMME OOD 评估环境包

## ① 一句话

`robomme_hard` 与官方 `robomme` 并列、分层继承：`src/robomme/` 是官方原样（官方文件相对官方提交零改动，由 benchmark 仓的 BENCH_UPSTREAM 闸门 `git diff --quiet 016ac1c4 HEAD -- src/robomme …` 守住；`UPSTREAM.json` 记录官方源码锚点 `1fadc0ec` 与逐文件 sha256），本包只放差异——16 个环境类与改过／新增／传递依赖改过模块的 utils、wrapper 复制；依赖闭包干净的官方模块用 shim 借用；`BenchmarkEnvBuilder` 子类化，只认 `hard-verify` 与 `ood` 两个评估数据集。

本包只做评估：生成链路（抽签、冻结规格、批量生成 h5、对拍工具）不随本仓发布，见旧仓归档 tag `archive-newtask-v9-20261008`（仓库 hongzefu/robomme_benchmark_MotionJEPA）。

## ② 包结构

```text
src/robomme_hard/
├── __init__.py                 导入即以 override=True 接管 16 个环境 id，并做注册表／命名空间归属断言、借用目标 cheap 校验
├── UPSTREAM.json               官方锚点（src_commit／src_tree）、官方逐文件 sha256、shim 清单、自签 manifest_sha256
├── robomme_env/                16 个环境类（复制）与 utils（复制／借用／新增）
├── env_record_wrapper/
│   ├── hard_builder.py         BenchmarkEnvBuilder 子类：只认 hard-verify／ood
│   ├── hard_specs.py           包内规格的读取、/4 封套校验、回注绑定摘要 spec_binding
│   └── …                       wrapper（复制或借用）
└── env_metadata/ood/xhard{1..5}/specs.jsonl   五份包内规格（hard-specs/4），ood 的唯一来源
```

## ③ 两个数据集

| dataset | 内容 | 局数 | 步数上限（入口传 `max_steps`） |
|---|---|---|---|
| `hard-verify` | xhard0＝官方 test 元数据里 `difficulty=="hard"` 的 12 局（原 episode 3, 7, …, 47），走官方原生 hard 分支，不回注 | 16 任务 × 12 局 = 192 | 1300 |
| `ood`（缺省） | 新值五档 xhard1～xhard5，读包内 `env_metadata/ood/<tier>/specs.jsonl` 的正式局（`selected` 且 `rollout.status=="ok"`），按交付格表 `EXPECTED_CELLS`（43 格）逐格断言 | 16 任务 × 50 局 = 800 | 1800 |

- 官方 `train`／`test`／`val` 与 `xhard1` 等档位名传给本构建器一律 `ValueError`；要官方行为请用官方 `robomme` 的构建器（另开进程，见 ⑥ R5）。
- `ood` 不含 xhard0，也不接受外部规格根（只读包内规格）；`override_metadata_path` 对两个数据集都不接受。
- 步数上限不由数据集给出，由评估入口 `scripts/evaluation_ood.py` 按数据集传 `max_steps`：`hard-verify` 1300、`ood` 1800。V9 交付集按 1600 过滤（规格 header 签 `exec_cap`＝1600，抽样时已过滤执行步超过 1600 的候选），1800 只放宽上限、不改已交付局。
- `ood` 每任务的档分布：PickXtimes／RouteStick／PatternLock 为 xhard1～3（17／17／16）；SwingXtimes、StopCube 为 xhard1～5 各 10；VideoUnmask、ButtonUnmask 为 xhard1～4（13／13／12／12）；BinFill、两个 Swap、VideoPlaceButton、VideoPlaceOrder、PickHighlight、VideoRepick 为 xhard1～2 各 25；MoveCube、InsertPeg 只有 xhard4 50 局。档内按 `candidate` 升序，按 xhard1→xhard5 拼接编为 episode 0..49。

## ④ 使用

```python
from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder   # 与官方唯一不同的 import
builder = BenchmarkEnvBuilder(env_id="BinFill", dataset="ood", action_space="joint_angle", max_steps=1800)
for episode in range(builder.get_episode_num()):             # BinFill 50 局：xhard1、xhard2 各 25 局
    env = builder.make_env_for_episode(episode)               # 与官方一样不传 max_steps，退回构造参数
    obs, info = env.reset()
```

- `builder.resolve_identity(episode)` 只读返回 `{episode, tier, candidate, seed, spec_sha256, source_run}`；`hard-verify` 的 xhard0 行另带 `source_dataset="test"`、`source_episode`，`candidate` 为空。
- `spec_binding(env)` 须在 `reset()` 之后调用，返回回注绑定摘要。
- 完整评估入口与命令见 [`scripts/README_ood.md`](../../scripts/README_ood.md)。

## ⑤ 逐文件：复制／借用／子类／新增

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

## ⑥ 机制

- **`sampling_config` 两块**：每个环境的 `native_blocks(cls, release=...)` 返回 `decision`（按档位的新值取值）与 `native`（原三档参数与位置），`gym.make(..., sampling_config={"decision","native"})` 显式传入；`utils/sampling_config.py::assert_native_decision` 保证去掉新值键后与原值全等。
- **`SpecRecorder` 导出与回注**（`utils/episode_spec.py`）：`gym.make(task, seed=s)` 不传 `native_episode_spec` 时进导出模式、按 seed 现场抽样（环境固有属性，与官方一致）；传 `native_episode_spec` 时回注——原抽样照常发生，但建场景用的值一律取冻结规格，原抽样只作兼容核验记入 `mismatches`；`record()` 只记录不回注的观测值。回注点（trace `source="spec"`）必须零差，记录点（`source="record"`）浮点差 ≤ `RECORDED_FLOAT_TOL=1e-5` 计 `recorded_drift`。
- **jsonl 两段与两个哈希**（`env_record_wrapper/hard_specs.py`，`schema="hard-specs/4"`，builder 经 `load_specs_root(PACKAGED_SPECS_ROOT, EXPECTED_CELLS)` 整根校验）：签 `task tier candidate episode seed attempt spec spec_sha256 layout_parent`（header 另签 `layout_rule`、`exec_cap` 与逐任务 `delivery_per_cell`）由 `identity_sha256` 覆盖；结果 `selected tried initial_selected rollout` 不进身份。`delivery_sha256` 盖排序后的 `(task, tier, candidate, seed, spec_sha256, rollout.h5_sha256)`（只取 `selected` 且 `rollout.status=="ok"`），锁住正式交付集合。
- **校验依赖（为何保留 `seed_for` 等）**：`hard_specs.py` 里的 `seed_for`、`SEED_RULE`、`V8_SEED_OFFSETS`、`seed_rule_for`、`ALL_TASKS`（env_code 的来源）、`identity_sha256`、`delivery_sha256` 不是生成入口，而是 `load_specs` 读包内规格时的校验依赖：逐行按 `seed = offset + env_code × 100000 + episode × 100 + attempt` 重算 seed（各档偏移 xhard1 16e6、xhard2 18e6、xhard3 20e6、xhard4 22e6、xhard5 24e6，档与档 seed 两两不交），再重算两个身份散列与 header 比对，任一不符即 `SpecsError`。删掉它们就无法证明包内规格未被篡改。xhard0 用官方 test 原 seed。
- **注册表与命名空间归属**：16 个环境类 `@register_env("<id>", override=True)`，导入本包即接管 16 个 id（与导入顺序无关）；包 `__init__` 末尾断言 `REGISTERED_ENVS[uid].cls.__module__` 属于本包，并断言 16 个环境模块与 `utils` 包里本包同名可调用对象都属于本包（专挡官方 `subgoal_evaluate_func` 的星号导入回灌）。
- **借用闭包规则**：借用资格按传递闭包判，不按单文件字节——shim 目标在官方源码上的依赖闭包不得含任何被本包复制的模块。复制文件里的绝对导入 `robomme.<path>`：`<path>` 在 shim 清单内的保持，是复制件的一律改成 `robomme_hard.`。
- **`UPSTREAM.json` 自签**：顶层 `manifest_sha256` 是剔掉该键后 canonical JSON（`sort_keys=True`、`ensure_ascii=False`、分隔符 `(",", ":")`）的 sha256；改任一值不重签即自签失败。

## ⑦ 红线

- R1 `src/robomme/**` 是官方原样，不放任何自有文件（`UPSTREAM.json`、说明一律放本包或 `docs/`）。
- R2 shim 不含逻辑（≤ 3 行非注释行）。
- R3 `identity_sha256` 与 `delivery_sha256` 的覆盖范围不变；五份包内 `specs.jsonl` 字节不动（sha256 见 `tests/robomme_hard/contract/packaged_specs.sha256`）。
- R5 同进程导入 `robomme_hard` 后 16 个 id 归它；需要官方行为另开只导入 `robomme` 的进程。

## ⑧ `dataset="test"` 绕行

官方父类 `__init__` 的 `_ALLOWED_DATASETS` 只认 `train/test/val`，而官方代码不能改。子类对 `ood`／`hard-verify` 先以 `dataset="test"` 过父类校验，再把 `self.dataset` 改回原值；父类顺手读的 test 元数据在 `hard-verify` 取出 xhard0 后、在 `ood` 直接清空，不再使用。官方父类签名若变化属于升级流程（`UPSTREAM.json` 换锚点、重跑全部闸门）。
