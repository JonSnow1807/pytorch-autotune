# PyTorch AutoTune

**One call to apply PyTorch's built-in training optimizations — AMP, `torch.compile`, fused optimizers, TF32 and channels-last — with honest, measured expectations.**

[![PyPI version](https://badge.fury.io/py/pytorch-autotune.svg)](https://pypi.org/project/pytorch-autotune/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Honesty note (v1.0.3).** Earlier versions of this README claimed a
> universal "4x speedup", benchmark tables (including an ImageNet row and an
> energy study) with no committed evidence, and a "beats torch.compile"
> comparison that its own mechanism contradicts — this package *wraps*
> `torch.compile`. Those claims are retracted. What this package actually
> does: it saves you ~20 lines of boilerplate by applying PyTorch's own
> optimization features correctly in one call. The speedup those features
> give is real but **workload-dependent**, and every number below was
> measured with the committed script and can be reproduced.

## What it does

```python
from pytorch_autotune import quick_optimize

model, optimizer, scaler = quick_optimize(model)   # AMP + compile + fused + TF32
```

One call applies, with sensible per-GPU defaults and manual overrides:

1. **Mixed precision** (`torch.amp` autocast + `GradScaler`)
2. **`torch.compile`** (default / `reduce-overhead` / `max-autotune`)
3. **Fused optimizers** (`AdamW`/`Adam` with `fused=True`)
4. **TF32** on Ampere+, **channels-last** for CNNs, `cudnn.benchmark`

These are PyTorch's features, not this package's — the package is a
convenience wrapper. If you're comfortable setting them up yourself, you
don't need it.

## Measured results

Measured with [`benchmarks/measure.py`](https://github.com/JonSnow1807/pytorch-autotune/blob/main/benchmarks/measure.py) on an
A100-SXM4-40GB, PyTorch 2.13.0+cu129, 2026-08-25. `ms/step` is a full
training step (forward + loss + backward + optimizer), median-of-loop after
warmup, compile time excluded (it is real but one-off).

| Workload | Baseline fp32 eager | AutoTune | Speedup |
|---|---|---|---|
| ResNet-50, 224×224, batch 64 | 73.3 ms | 28.0 ms | **2.62×** |
| ResNet-18, 32×32 (CIFAR-shape), batch 128 | 8.1 ms | 6.4 ms | 1.26× |
| — same, but without `torch.compile` | 8.1 ms | 9.0 ms | **0.89× (slower!)** |
| — plain `torch.compile`, no AMP | 8.1 ms | 7.2 ms | 1.12× |

How to read this honestly:

* **Compute-bound models benefit most** (2.62× on the ResNet-50 step —
  mostly AMP's tensor cores plus compile's fusion). Small or launch-bound
  workloads benefit little, and some configurations **lose** to the fp32
  baseline (the 0.89× row) — which is exactly why you should measure your
  own workload rather than trust any package's headline.
* Against `torch.compile` alone, the full stack adds AMP's gains on top —
  it is not an alternative to `torch.compile` and does not "beat" it; it
  *uses* it.
* Older GPUs with a larger fp16-vs-fp32 throughput gap (e.g. T4) can see
  larger AMP ratios on compute-bound CNNs; no number is claimed here for
  hardware this version was not measured on.

## Installation

```bash
pip install pytorch-autotune
```

## Usage

```python
from pytorch_autotune import AutoTune

autotune = AutoTune(model, device='cuda', verbose=True)
model, optimizer, scaler = autotune.optimize(
    optimizer_name='AdamW',
    learning_rate=1e-3,
    compile_mode='default',   # or 'reduce-overhead' / 'max-autotune'
    use_amp=True,
    use_compile=True,
    use_fused=True,
)

for data, target in train_loader:
    data, target = data.cuda(), target.cuda()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        loss = criterion(model(data), target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Notes:

* The first iterations after `torch.compile` are slow (compilation); plan a
  warmup.
* Mixed precision changes numerics; validate your accuracy as you would with
  any AMP setup.
* `AutoTune.benchmark()` times forward-only inference under `no_grad` — use
  `benchmarks/measure.py` for training-step comparisons.

## Limitations

* This is a wrapper, not a tuner: settings come from a small hardware table,
  not from measuring your model. (A measurement-driven version is the
  roadmap.)
* Measured on one GPU (A100). No claims for other hardware.
* No tests or CI yet.

## Citation

```bibtex
@software{pytorch_autotune,
  title = {PyTorch AutoTune: one-call PyTorch training optimization setup},
  author = {Shrivastava, Chinmay},
  year = {2025},
  url = {https://github.com/JonSnow1807/pytorch-autotune},
  version = {1.0.3}
}
```

## Author

**Chinmay Shrivastava** — GitHub [@JonSnow1807](https://github.com/JonSnow1807)

## License

MIT — see [LICENSE](LICENSE).
