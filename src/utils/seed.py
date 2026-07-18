import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python/NumPy/torch (CPU+CUDA) and force deterministic cuDNN kernels.

    Call once at process start (see scripts/train.py). Deterministic cuDNN
    disables benchmark-mode kernel autotuning, trading some throughput for
    run-to-run reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: derive a per-worker seed from torch's initial seed
    so augmentation randomness is reproducible even with num_workers > 0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
