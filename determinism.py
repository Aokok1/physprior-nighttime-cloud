"""Deterministic seeding shared by every training entry point.

The re-run showed that repeating the five-seed protocol with identical seeds moved
individual seeds by up to 4 pp. The cause was not the seeds themselves — those were
set — but the surrounding configuration:

  * cuDNN was left in benchmark mode, so it re-selected convolution algorithms on
    every process start;
  * the co-teaching script never seeded the CUDA RNG, so GPU-side initialisation
    and dropout draws differed between runs;
  * the train loader shuffles off the global torch RNG, which is only reproducible
    if nothing else consumes that stream first.

`seed_everything` fixes all three. Import it and call it at the top of a run
instead of seeding by hand.
"""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed every RNG the pipeline touches and pin cuDNN to deterministic kernels.

    Args:
        seed: the run seed.
        deterministic: when True, disable cuDNN autotuning and ask torch for
            deterministic algorithms. The workspace setting must be present
            before the first CUDA handle is created for cuBLAS determinism.

    Returns:
        The seed, so callers can log it.
    """
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only keeps unsupported ops from aborting the run while still
        # surfacing them; the kernels this pipeline uses are all supported.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return seed


def loader_generator(seed: int) -> torch.Generator:
    """A dedicated generator for DataLoader shuffling.

    Giving the loader its own stream means the shuffle order no longer depends on
    how much randomness the model consumed beforehand.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
