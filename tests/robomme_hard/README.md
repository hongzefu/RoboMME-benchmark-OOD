# 测试（2026-10 重构）

旧测试（`tests/lightweight`、`tests/dataset`、`tests/_shared`、两个旧夹具目录）已于 12.389 整体删除、不作基准（用户原话「旧测试完全放弃。不作为基准」）；蓝本可从 `git show 86e5a015:<路径>` 读。新结构与契约见计划 `1003-code-test-maintenance-todo.md` 第二部分「测试细则」。（以上提交与计划均在旧仓 `robomme_benchmark_MotionJEPA` 的历史里。）

本仓（官方 `016ac1c4` + `robomme_hard`）只保留评估侧测试：对拍（`scripts/parity`）、生成（`scripts/injection-dev`）、评估编排（`scripts/eval-official`）等工具及其测试随私有评估仓，本仓不测。

| 层 | 目录 | 进日常门禁 |
|---|---|---|
| L0 静态与上游 | `tests/robomme_hard/static/` | 是（wheel 安装标 `slow`） |
| L1 契约 | `tests/robomme_hard/contract/` | 是 |
| L2 单元 | `tests/robomme_hard/unit/{robomme,hard,common}/` | 是 |
| L3 流水线 | `tests/robomme_hard/pipeline/{recording,challenge}/` | 是（起 bash／ffmpeg／websocket 的标 `slow`） |
| L4 仿真冒烟 | `tests/robomme_hard/sim/` | 否，显式 `--allow-sim-reset`，每次 59 + 1 = 60 次 reset |

公共件只归主会话：`tests/robomme_hard/_support/`（资源守卫 `resource_policy.py`、按路径加载脚本 `loaders.py`）、`tests/robomme_hard/conftest.py`、`tests/robomme_hard/contract/benchmark_contracts.json`、`pyproject.toml` 的 pytest 段。

资源守卫：日常门禁里构建真实 SAPIEN 场景、初始化 CUDA、`torch.load`／safetensors 读权重、连非回环地址都会被拒并记账；子进程经 `sitecustomize` 继承。收集为空、出现 xfail、skip 原因不以「未验证」开头，整场判失败。末行判定 `TEST_RESOURCE=`。

```bash
# 日常门禁
timeout 280s uv run --no-sync python -m pytest -m 'not slow' -q --durations=20
# 慢测试
uv run --no-sync python -m pytest -m slow -q
# 仿真冒烟（先 nvidia-smi 选空闲卡）
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m pytest tests/robomme_hard/sim --allow-sim-reset -q
```
