"""PyTorch AutoTune - measurement-driven training-config autotuning.

Two entry points:

* :func:`autotune` (v2.0+, recommended): measures candidate configurations
  (precision x torch.compile mode x memory format x fused optimizer) on
  YOUR model and batch, returns the fastest, and never returns a config
  that lost to your baseline.
* :func:`quick_optimize` / :class:`AutoTune` (v1.x, kept): applies a fixed
  heuristic configuration without measuring - fine when you cannot afford
  a tuning run, but expect it to leave speed on the table.
"""

__version__ = "2.0.1"

from .autotune import AutoTune, quick_optimize
from .tuner import AutotuneResult, Report, TrialConfig, TrialResult, autotune

__all__ = [
    "AutoTune",
    "AutotuneResult",
    "Report",
    "TrialConfig",
    "TrialResult",
    "autotune",
    "quick_optimize",
    "__version__",
]
