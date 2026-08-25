# Changelog

## [2.0.0] - 2026-08-25 — the real autotuner

### Added

* `autotune(model, example_batch, budget=...)` — measurement-driven search
  over precision (fp32/TF32/fp16/bf16) x `torch.compile` mode
  (off/default/reduce-overhead/max-autotune) x channels-last x fused
  optimizer, running real training steps on the caller's batch under a
  wall-clock budget. Weights are snapshotted and restored: tuning never
  trains the model. Non-finite-loss, OOM and compile-failure candidates are
  disqualified and shown in the report next to the losers and compile
  costs. **Baseline guarantee**: the measured PyTorch-default fp32 baseline
  is a candidate, and nothing that failed to beat it (>3 % noise floor) is
  ever returned.
* Config cache per (model, shapes, GPU, torch version); `cache="refresh"`.
* `benchmarks/suite.py` + committed results
  (`benchmarks/results/2026-08-25_a100-40gb_v200/`, clean clone `b4b0364`,
  locked clocks): tuned configs reach 5.84x/8.37x/6.65x vs strict-fp32
  eager, 2.71x/3.84x/6.65x vs PyTorch-default fp32, and 2.03x/3.17x/6.08x
  vs plain `torch.compile` on ResNet-50/ResNet-18/Transformer — the claims
  v1.0 fabricated, now exceeded with the baselines labelled and the data
  committed.
* Tests (baseline guarantee, disqualification paths, state restoration,
  budget skips, cache) and CPU CI (torch 2.4/2.13).

### Changed

* torch floor raised to 2.4 (the oldest version CI exercises).
* `quick_optimize`/`AutoTune` remain as the documented unmeasured
  heuristic path.

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
