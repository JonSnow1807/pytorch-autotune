# H100 suite run (2026-10-13, commit dd2f1a1)

`python benchmarks/suite.py --budget max` from a clean tree at `dd2f1a1`.
Hardware and software: H100 80GB HBM3, SM clocks locked at 1755 MHz, torch 2.13.0+cu129.
The JSON and the summary table are next to this file.

## Protocol

- Each reference row ran in a fresh process with its own empty Inductor,
  Triton and CUDA cache directories. The rows are strict fp32, PyTorch-default
  fp32, plain `torch.compile` and the expert config.
- `search_cold` ran the same way. `search_warm` ran next, in a new process,
  reusing the cache directories `search_cold` filled.
- Speedups use the cold search's winner and one measurement of each
  reference row.

## Cold and warm search cost

| workload | search cold | search warm | compile time in the cold search |
|---|---|---|---|
| resnet50_224px_b64 | 517 s | 54 s | 472 s, including max-autotune trials of 190 s and 154 s |
| resnet18_32px_b128 | 78 s | 16 s | 55 s |
| transformer_6l_s128_b32 | 112 s | 90 s | 102 s, including a max-autotune trial of 66 s |

The transformer's warm search was mostly cold, for these reasons:
- Stage 2 ranks precisions by eager timing, and the two runs ranked them
  differently. The cold run put fp16 ahead of bf16 (9.59 vs 9.71 ms); the warm
  run put bf16 ahead (9.56 vs 9.97 ms).
- So the warm run compiled bf16 graphs that the cold run never built:
  compile=default took 10 s and max-autotune 61 s.
- It also picked a different winner: bf16 reduce-overhead at 2.50 ms, against
  the cold run's fp16 reduce-overhead at 2.65 ms.

The stage-2 pruning decides which precisions get compiled. When two
precisions measure within noise, that choice can differ between runs.

## Measurement spread

- **Transformer expert config:** the suite row measured 5.31 ms.
  - In three more fresh processes it measured 5.21, 5.35 and 5.65 ms.
  - Measured twice in one process, after the search's earlier eager and
    compile trials, it gave 5.73 and 5.39 ms.
  - Inside the warm search it measured 6.00 ms.
  - Against these values (5.21–6.00 ms), the tuner's 2.65 ms pick is
    1.97x–2.26x faster. The table says 2.0x.
- **ResNet-50 is tighter.** The expert row (14.97 ms) agrees within 0.6% with
  the identical config measured inside the cold and warm searches (14.91 and
  14.89 ms).
- **ResNet-18:** PyTorch-default fp32 measured 5.82, 6.18 and 6.13 ms across
  the two searches and the reference row.

## Comparability

These are H100 numbers. The 2026-08-25 results come from an A100-SXM4-40GB,
so ratios from the two runs are not comparable.
