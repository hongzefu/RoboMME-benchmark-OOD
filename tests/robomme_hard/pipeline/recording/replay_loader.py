"""按路径加载上游入口 scripts/dataset_replay.py（只读加载，不改文件）。

该脚本在导入时把 ``CUDA_VISIBLE_DEVICES`` 改成 "1"；资源守卫要求测试进程里 GPU 不可见，
所以加载后立即恢复原值。torch.cuda 的初始化另由资源守卫拦截。
"""
from __future__ import annotations

import os

from tests.robomme_hard._support.loaders import load_script


def load_replay():
    key = "CUDA_VISIBLE_DEVICES"
    old = os.environ.get(key)
    try:
        return load_script("dataset_replay.py")
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old
