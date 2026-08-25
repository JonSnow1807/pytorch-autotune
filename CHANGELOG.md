# Changelog

## [1.0.3] - 2026-08-25 — honesty release

### Retracted

* The "4x training speedup" headline, the benchmark tables (T4 4.06x,
  ResNet-50 "ImageNet 45.2s", ViT, BERT rows), the energy-efficiency table,
  the per-GPU speedup promises (including "FP8" on H100 — nothing in this
  package uses FP8), and the "+accuracy" claims. None of them had committed
  evidence or scripts behind them. The GitHub/PyPI description claiming to
  "beat torch.compile by 79%" is retracted for the additional reason that
  this package's main mechanism *is* `torch.compile`.
* The 2024 dates on earlier entries below were wrong: the project and all
  its PyPI releases date from 2025-08-10.

### Added

* `benchmarks/measure.py` — the training-step benchmark behind every number
  now in the README (baseline fp32 eager vs the AutoTune stack vs plain
  `torch.compile`, environment printed with the results).
* Measured results (A100-SXM4-40GB, torch 2.13.0): 2.61x on a compute-bound
  ResNet-50 step, 1.25x on a small CIFAR-shape step, and an honest 0.89x
  (slower than baseline) for the no-compile configuration at that small
  shape — published because a wrapper that can lose to the baseline should
  say so.

### Fixed

* `AutoTune.benchmark()` now measures under `torch.no_grad()` with the H2D
  copy outside the timed loop (it previously timed autograd-tracked forwards
  plus a per-iteration host-to-device copy).
* `get_optimal_batch_size()` gated its backward pass on
  `hasattr(output, 'backward')`, which is true for every tensor; now checks
  `requires_grad`.
* Docstrings and printed messages no longer promise fixed speedups.

## [1.0.1] / [1.0.2] - 2025-08-10

* Packaging and README tweaks. (Entries previously dated 2024-11-10; that
  date was wrong.)

## [1.0.0] - 2025-08-10

* Initial release: the AutoTune wrapper (AMP, torch.compile, fused
  optimizers, TF32, channels-last). Claimed results retracted in 1.0.3.
