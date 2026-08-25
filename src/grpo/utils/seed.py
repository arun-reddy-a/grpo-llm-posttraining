"""Seeding.

Worth stating what seeding does and does not buy you in an RL loop: it makes a
run reproducible *given the same sampler backend*, but vLLM and HF generation
will not produce identical tokens from the same seed, and neither will the same
backend across GPU counts or kernel versions. The point of seeding here is to
make a *bug* reproducible while you are chasing it, not to promise bitwise
determinism across environments.
"""

from __future__ import annotations

import os
import random

__all__ = ["set_seed"]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch RNGs.

    ``deterministic=True`` additionally forces deterministic cuDNN/cuBLAS
    kernels. It is off by default because it costs real throughput and some
    attention kernels have no deterministic implementation at all.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
