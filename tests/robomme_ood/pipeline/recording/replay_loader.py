"""Load the upstream entry script scripts/dataset_replay.py by path (read-only load, file not modified).

The script sets ``CUDA_VISIBLE_DEVICES`` to "1" on import; the resource guard requires the GPU to be invisible in the test process,
so the original value is restored right after loading. torch.cuda initialization is separately blocked by the resource guard.
"""
from __future__ import annotations

import os

from tests.robomme_ood._support.loaders import load_script


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
