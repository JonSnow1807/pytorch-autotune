# PyTorch AutoTune

**A real autotuner: it measures candidate training configurations on *your*
model and batch — precision × `torch.compile` mode × memory format × fused
optimizer — keeps the measured winner (a 3 % noise floor breaks ties toward
the cheaper config), and is structurally incapable of returning a config
slower than your baseline.**

[![CI](https://github.com/JonSnow1807/pytorch-autotune/actions/workflows/ci.yml/badge.svg)](https://github.com/JonSnow1807/pytorch-autotune/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/pytorch-autotune.svg)](https://pypi.org/project/pytorch-autotune/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```python
from pytorch_autotune import autotune

result = autotune(model, (inputs, targets), budget="balanced")  # measures, picks
print(result.report)          # every trial: winners, losers, disqualifications
loss = result.step(x, y)      # train with the tuned setup
```

## Why measure instead of guess

No fixed configuration is right for every workload — that is a measured
fact, not a slogan. From this repo's committed benchmark run
([data](https://github.com/JonSnow1807/pytorch-autotune/tree/main/benchmarks/results/2026-08-25_a100-40gb_v200/)): `torch.compile`'s
default mode is within 4 % of the tuner's ResNet-50 pick — the raw-fastest
trial there was max-autotune, another 2.7 % quicker, which the 3 % noise
floor deliberately ignores in favour of the much cheaper compile — yet the
same default mode leaves **2.7×** on the table for small-batch ResNet-18
(CUDA-graph mode won both searches); fp16 beats
bf16 on one workload and loses on the next; on tiny models AMP *loses* to
fp32 outright. v1 of this package was a fixed-heuristic wrapper and its own
benchmarks caught it being slower than baseline at one shape. So v2
measures: every candidate runs real training steps on your actual model and
batch, and the report shows everything it tried.

**The baseline guarantee**: the first thing measured is your baseline
(PyTorch-default fp32 eager, plain optimizer). A candidate must beat the
incumbent by >3 % (noise floor) to win, so the returned config never lost
to what you already had.

## Measured results

From [`benchmarks/suite.py`](https://github.com/JonSnow1807/pytorch-autotune/blob/main/benchmarks/suite.py) — committed output in
[`benchmarks/results/2026-08-25_a100-40gb_v200/`](https://github.com/JonSnow1807/pytorch-autotune/tree/main/benchmarks/results/2026-08-25_a100-40gb_v200/)
(clean clone `b4b0364`, A100-SXM4-40GB, torch 2.13.0+cu129, SM clocks locked
at 1410 MHz, `budget="max"`). Full training steps (forward + loss + backward
+ optimizer); three baselines, all measured with the same protocol:

| workload | tuner's pick | vs strict fp32 eager¹ | vs PyTorch-default fp32² | vs plain `torch.compile`³ |
|---|---|---|---|---|
| ResNet-50, 224px, b64 | fp16 · reduce-overhead · CL · fused | **5.84×** | **2.71×** | **2.03×** |
| ResNet-18, 32px, b128 | bf16 · reduce-overhead · CL · fused | **8.37×** | **3.84×** | **3.17×** |
| Transformer 6L, s128, b32 | bf16 · max-autotune · fused | **6.65×** | **6.65×** | **6.08×** |

¹ TF32 disabled everywhere — the most generous framing, listed for
comparability with historical claims.
² What an untuned user gets (cuDNN TF32 on — PyTorch's default). **This is
the honest headline column.** One deliberate caveat: PyTorch's default
disables TF32 *matmuls*, so for matmul-dominated models this baseline is
generous to the tuner — against an informed user's one-line TF32 fix the
transformer win measures **2.55×**, not 6.65× (re-measured in
[`benchmarks/falsify.py`](https://github.com/JonSnow1807/pytorch-autotune/blob/main/benchmarks/falsify.py); committed output in the
results directory).
³ `torch.compile(model)` default mode on PyTorch-default fp32 — the "just
use torch.compile" alternative. The tuner wins because it *also* picks
precision, CUDA-graph mode, memory format and the fused optimizer — it uses
`torch.compile`, it doesn't compete with it.

Search cost: one-off, budget-capped (`"fast"` ≈ 2 min, `"balanced"` ≈ 6 min,
`"max"` ≈ 15 min — max-autotune compiles alone can take 1–4 min). Winning
configs are cached per (model, shapes, GPU, torch version); later calls skip
the search (`cache="refresh"` re-measures).

> **On this project's history.** v1.0 of this package claimed "4× speedup"
> and "beats torch.compile by 79 %" with no committed evidence; those claims
> were retracted in v1.0.3 (see CHANGELOG). The v2 rewrite was built to earn
> them instead: against the strict-fp32 baseline the old claims implicitly
> used, the measured tuner exceeds 4× on all three workloads (5.8–8.4×), and
> it exceeds plain `torch.compile` by 103–508 % — with the baselines defined,
> the mechanism explained, and the data committed. Against the fairer
> PyTorch-default baseline the honest numbers are 2.7–6.7×.

## Install

```bash
pip install pytorch-autotune
```

Requires torch ≥ 2.4 and a CUDA GPU for tuning (imports and falls back
gracefully on CPU).

## Usage

```python
from pytorch_autotune import autotune

result = autotune(
    model,                      # weights preserved: tuning never trains your model
    (example_inputs, targets),  # the shapes you actually train at
    loss_fn=None,               # default: CrossEntropyLoss (or output.mean() without targets)
    optimizer="adamw",          # "adam" / "sgd" / callable(params) -> Optimizer
    budget="balanced",          # "fast" | "balanced" | "max" | seconds as int
    cache=True,                 # True | False | "refresh"
    allow_cudagraphs=True,      # False to exclude reduce-overhead (dynamic shapes)
)

model, optimizer, scaler = result.model, result.optimizer, result.scaler
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    loss = result.step(x, y)    # or write your own loop with result.autocast()
```

`example_batch` may be a tensor, an `(input, target)` pair (integer targets
default to CrossEntropy, floating targets to MSE), or a **dict of model
kwargs**. The dict form makes HuggingFace models work with zero config —
when the output has a `.loss` (i.e. `labels` is in the batch), it is used
directly:

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}

result = autotune(model, batch, budget="balanced")
loss = result.step(batch)       # trains with the tuned setup
```

Interrupting a search with Ctrl-C keeps the best config found so far (the
report says so); winning configs export via `result.report.to_json()`, and
the printed report includes a break-even line (search cost vs per-step
saving).

The report is always honest and complete:

```
autotune report - NVIDIA A100-SXM4-40GB | torch 2.13.0+cu129 | search took 56s
baseline = PyTorch-default fp32 eager: 8.95 ms/step
config                                               status          ms/step  vs base  compile
fp32_default compile=off                             ok                 8.95    1.00x        -
tf32 compile=off channels_last fused                 ok                 8.78    1.02x        -
amp_bf16 compile=off channels_last fused             ok                 8.91    1.00x        -
amp_fp16 compile=off channels_last fused             ok                 9.35    0.96x        -
tf32 compile=default channels_last fused             ok                 6.51    1.37x      13s
amp_bf16 compile=default channels_last fused         ok                 6.27    1.43x       2s
tf32 compile=reduce-overhead channels_last fused     ok                 2.85    3.14x      11s
amp_bf16 compile=reduce-overhead channels_last fused ok                 2.33    3.83x       1s <== winner
amp_bf16 compile=max-autotune channels_last fused    ok                 2.37    3.78x       2s
amp_bf16 compile=reduce-overhead fused               ok                 2.78    3.23x      13s
amp_bf16 compile=reduce-overhead channels_last       ok                 3.01    2.97x       1s
search cost 56s; saves 6.62 ms/step -> breaks even after ~8,463 training steps.
note: reduce-overhead uses CUDA graphs - keep input shapes static.
```

(That is the committed ResNet-18 search, rendered from the committed JSON
by the shipped `Report` code — note AMP alone does *nothing* at this shape,
fp16 loses, and CUDA-graph mode is worth 2.7× over compile-default. A fixed
wrapper cannot know any of that.)

### Adversarially verified

The committed numbers were attacked before release
([`benchmarks/falsify.py`](https://github.com/JonSnow1807/pytorch-autotune/blob/main/benchmarks/falsify.py), output committed next to
the data): every winner was re-measured in a **real fresh-batch training
loop, rebuilt from scratch twice**, and reproduced its committed trial
number within −0.1 % / +1.3 % / +1.5 % (≤ 1.6 % allowing for the printed
rounding); the baselines reproduced within 1.9 %; the fp16
winner ran 60 timed-regime steps with **zero** GradScaler-skipped optimizer
steps; and the tuned config trained a learnable task to convergence
identically to the baseline. A parallel methodology audit (14 attack
findings, each adversarially verified — the full list with outcomes is
committed as
[`methodology_audit.md`](https://github.com/JonSnow1807/pytorch-autotune/blob/main/benchmarks/results/2026-08-25_a100-40gb_v200/methodology_audit.md))
produced two documentation fixes and no measurement defects.

### Caveats you should actually read

* **Winners using `reduce-overhead` need static input shapes** (CUDA
  graphs). Pass `allow_cudagraphs=False` if your batch shapes vary.
* Mixed-precision winners change numerics like any AMP setup — validate
  accuracy as you would if you enabled AMP yourself.
* The tuner optimizes the training step it can see: your dataloader,
  logging, and eval are outside its reach.
* Speedups are workload- and GPU-specific. The table above is an A100;
  your model on your GPU is what `autotune()` measures.

### Legacy heuristic mode

`quick_optimize(model)` / `AutoTune` (the v1 API) still exist: they apply a
fixed AMP+compile+fused configuration with no measurement. Fine when you
cannot afford a tuning run — but the committed data shows fixed heuristics
leave up to 2.7× unclaimed, which is why `autotune()` exists.

## Tests & CI

`pytest tests -q` — tuner behavior (baseline guarantee, disqualification of
non-finite/OOM/compile-failure candidates, state restoration, cache, budget
accounting) on GPU; CI runs the CPU-safe subset on torch 2.4 and 2.13.

## Citation

```bibtex
@software{pytorch_autotune,
  title = {PyTorch AutoTune: measurement-driven PyTorch training configuration tuning},
  author = {Shrivastava, Chinmay},
  year = {2026},
  url = {https://github.com/JonSnow1807/pytorch-autotune},
  version = {2.0.0}
}
```

## Author

**Chinmay Shrivastava** — GitHub [@JonSnow1807](https://github.com/JonSnow1807)

## License

MIT — see [LICENSE](LICENSE).
