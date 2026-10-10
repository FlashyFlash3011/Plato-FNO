"""Small helpers shared by the Hydra scripts in ``scripts/``."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from solaris.utils.logging import get_logger


def setup_run_logging(name: str):
    """Return a logger that also writes to ``<hydra run dir>/<name>.log``.

    The code logs with loguru, which Hydra's own log file never sees; without
    this sink the per-run ``.log`` files in ``outputs/`` stay empty.
    """
    log = get_logger(name)
    if HydraConfig.initialized():
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        log.add(run_dir / f"{name}.log", level="INFO", enqueue=False)
    return log


def resolve_device(requested: str) -> torch.device:
    """Use the requested device if it is a GPU and one is available, else CPU."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
