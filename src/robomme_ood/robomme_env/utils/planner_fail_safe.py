# 借用：robomme 同名模块的别名，逻辑以官方为准；清单见 ../../UPSTREAM.json
import importlib, sys
sys.modules[__name__] = importlib.import_module("robomme.robomme_env.utils.planner_fail_safe")
