# 植入执行器（tests/robomme_ood/mutation）

计划 `1003-code-test-maintenance-todo.md` 第二部分「五、闸门总表与植入清单」的公共件：读取各测试块自己维护的
`tests/**/mutants.json`，逐项把一处**语义错误**植入生产逻辑，确认该项 `expect_fail` 里的用例因此失败，给出判定行

```
TEST_MUTATION=PASS|FAIL seeded=<n> caught=<n> survived=<n> not_executable=<n> not_applied=<n> no_recipe=<n> baseline_fail=<n> repo_changed=<0|1>
```

通过条件：`survived`、`baseline_fail`、`not_applied`、`no_recipe` 全为 0，运行前后仓库 `git status`（排除只追加的
`docs/subagent-stats`）不变（`repo_changed=0`），且 `seeded>0`。`not_executable` 只放行用户裁决例外，不影响判定。

本目录文件都不以 `test_` 开头、不在 `pyproject.toml` 的 `testpaths` 里，不进日常门禁。

## 文件

| 文件 | 作用 |
|---|---|
| `run_mutants.py` | 执行器：读取、归一、建隔离副本、跑基线与植入、汇总判定行与逐项记录 |
| `recipes.py` | 配方补充表：mutants.json 只有文字描述的 A 类条目，在这里落成精确替换点或数据改写函数 |
| `plugins/mut_inproc.py` | 进程内植入插件（`MUT_INPROC=<块>:<编号>`），覆盖受保护目录与须导入后改常量的 B 类条目；植入前预校验，状态写 `MUT_STATUS_FILE` |
| `plugins/t4_status.py` | T4 插件的配套：把 `mutants_plugin._redefine` 收紧为片段恰好命中 1 次，并写植入状态（不改 T4 插件文件） |
| `plugins/outcome_recorder.py` | 结果记录插件：把每个用例各阶段结局、异常类型与异常栈顶文件写到 `MUT_OUTCOME_FILE` |

## 三类

- **A 文本替换／数据改写**：mutants.json 自带 `patch: [{file, old, new}]`，或 `recipes.py` 里有同键配方。
  执行器把仓库的 `src`、`scripts`、`tests`、`challenge_interface`、`pyproject.toml` 复制到 tmp 隔离副本
  （每个并发一份，先打印并核对 `robomme_ood.__file__`／`robomme.__file__` 指向副本），逐字替换（每个 `old` 必须恰好命中
  1 次，否则记 `not_applied`）或调用改写函数（规格 jsonl 重签等），跑完把被改文件按原字节还原并核对 sha256。
  点名 `src/robomme/` 或三个上游入口（`scripts/dataset_replay.py`、`scripts/evaluation.py`、`scripts/run_example.py`）
  的 A 类配方一律拒绝落盘，记 `no_recipe`（红线 R9）。
- **B 进程内插件**：不落盘，在仓库本身运行（`PYTHONDONTWRITEBYTECODE=1`、`-p no:cacheprovider`，只读）。
  插件在 `pytest_configure`／`pytest_sessionstart` 阶段先预校验植入点（源码片段恰好命中 1 次、读文件型植入的字节片段
  恰好命中 1 次、被替换的属性存在、签名未变），不满足就不植入、照原版跑，并在 `MUT_STATUS_FILE` 写 `applied=false`；
  执行器只有读到 `applied=true` 才继续判定，否则记 `not_applied`。
  - T4 块沿用 `tests/robomme_ood/unit/hard/mutants_plugin.py`：`PYTHONPATH` 加 `tests/robomme_ood/unit/hard`，`-p mutants_plugin`，
    环境变量 `T4_MUTANT=<名>`（名字从 mutants.json 的 `method` 里解析），另加 `-p tests.robomme_ood.mutation.plugins.t4_status`；
  - 其余块用 `plugins/mut_inproc.py`：`-p tests.robomme_ood.mutation.plugins.mut_inproc`，`MUT_INPROC=<块>:<编号>`。
    静态块改 `pathlib` 读到的字节，契约块改导入后的常量或包装函数，录制／官方单元／包装器块替换方法；
    每项实现照抄对应 mutants.json 的描述。
- **C 用户裁决例外**：mutants.json 里没有 `expect_fail` 且带 `decision` 的条目（目前只有挑战块 M19，用户裁决不修），
  计入 `not_executable` 并放行。
- **NR 缺配方**：找不到可执行做法、R9 拒绝落盘、或没有 `expect_fail` 又没有裁决的条目，计入 `no_recipe`，判 FAIL。

## 判定口径

1. **基线**：按运行环境分组（A 在一份原版副本；B 每个插件一组，插件已加载但不设植入变量），把组内全部
   `expect_fail` 用例并集跑一遍；某项有用例不过时对该项单跑复核，仍不过记 `baseline_fail`，不再植入。
   要求每个 `expect_fail` 至少收集到一个实例且全部通过（跳过也算不过）。
2. **植入后**：至少一个 `expect_fail` 用例（函数级编号按 `编号[` 前缀匹配参数化实例）在 setup／call 阶段失败，
   且异常类型不是 `ImportError`／`ModuleNotFoundError`／`SyntaxError`／`IndentationError`，异常栈顶（抛出点）不在
   植入设施（`tests/robomme_ood/mutation/` 下或 `tests/robomme_ood/unit/hard/mutants_plugin.py`），所在文件也没有收集错误，才记抓到（caught）；
   否则记存活（survived）。存活项不得改测试凑数，交主会话裁决。
3. 资源守卫照常装着（仓库 `pyproject.toml` 的 `addopts` 在副本里同样生效）；不起仿真、不访问外网。

## 运行

worktree 里没有 `.venv`，须指向主检出的环境：

```bash
UV_PROJECT_ENVIRONMENT=/data/hongzefu/robomme_benchmark_MotionJEPANewTask/.venv \
  uv run --no-sync python tests/robomme_ood/mutation/run_mutants.py --jobs 4
```

- `--list`：只打印每项归到哪一类，不运行；
- `--only <块前缀或键>...`：只跑部分（如 `--only unit/ static`、`--only pipeline/site:M20a`），配 `--tag <批次名>`
  写 `last_run.<批次名>.jsonl`；分批跑完后 `--merge --tag <批次前缀>` 只合并 `last_run.<批次前缀>*.jsonl` 为
  `last_run.jsonl` 并打印总判定行（缺项、重复或任一批次 `repo_changed` 即 FAIL）；
- 隔离副本建在 `$TMPDIR`（默认 `/tmp`），跑完删除；
- 逐项记录写 `artifacts/maint-regress/mutation/last_run.jsonl`（不进 git）：来源文件、编号、类别、基线结果、
  植入后各用例结局、异常类型与栈顶文件、是否抓到、失败用例列表；同名 `.meta.json` 记运行前后 `git status`。

全量 108 项（A 60、B 47、C 1）在 sled-vail 上 4 并发一次跑完约 2 分钟（实测 123 s），单个 pytest 进程设 280 秒超时。

## 新增植入

在本块 `mutants.json` 加条目并写好 `expect_fail`；能写成精确替换的直接给 `patch`（非受保护文件），否则：
非受保护文件在 `recipes.py` 补同键配方，受保护文件或须导入后改的在 `plugins/mut_inproc.py` 补实现并加入 `SUPPORTED`。
跑 `--list` 确认不落在 C 类，再跑 `--only <块>` 确认基线过、植入被抓到。
