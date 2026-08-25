#!/usr/bin/env python3
"""The benchmark suite behind the README's numbers.

For each workload this runs autotune() (real search, chosen budget) and, for
reference, three fixed configurations measured with the same trial protocol:

  * strict fp32 eager (TF32 fully off) - the baseline the project's
    retracted v1.0 claims implicitly used,
  * PyTorch-default fp32 eager - what an untuned user actually gets
    (cuDNN TF32 on, matmul TF32 off),
  * plain ``torch.compile`` (default mode) on PyTorch-default fp32 - the
    "just use torch.compile" alternative.

Speedups are reported against BOTH baselines, labelled. Output: JSON (+ md
table) with git/GPU/clock metadata, suitable for committing under
benchmarks/results/.

Usage:
  python benchmarks/suite.py --budget max --out benchmarks/results/out
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    import torchvision.models as tvm
except ImportError:
    raise SystemExit("torchvision is required: pip install torchvision")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pytorch_autotune.tuner import (  # noqa: E402
    TrialConfig, _apply_precision_flags, _build_step, _time_step_fn, autotune,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_resnet50():
    torch.manual_seed(0)
    return (tvm.resnet50(num_classes=1000),
            (torch.randn(64, 3, 224, 224), torch.randint(0, 1000, (64,))))


def make_resnet18():
    torch.manual_seed(0)
    return (tvm.resnet18(num_classes=10),
            (torch.randn(128, 3, 32, 32), torch.randint(0, 10, (128,))))


def make_transformer():
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048,
                                       batch_first=True, dropout=0.0)

    class TEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.TransformerEncoder(layer, num_layers=6)
            self.head = nn.Linear(512, 100)

        def forward(self, x):
            return self.head(self.enc(x).mean(1))

    return TEnc(), (torch.randn(32, 128, 512), torch.randint(0, 100, (32,)))


WORKLOADS = {
    "resnet50_224px_b64": make_resnet50,
    "resnet18_32px_b128": make_resnet18,
    "transformer_6l_s128_b32": make_transformer,
}


def measure_fixed(model, batch, cfg: TrialConfig, strict_fp32=False):
    """Measure one fixed config with the tuner's own trial protocol."""
    data, target = batch
    model = model.to("cuda")
    data, target = data.to("cuda"), target.to("cuda")
    _apply_precision_flags(cfg)
    if strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    step = _build_step(model, cfg, data, target, nn.CrossEntropyLoss(), "adamw",
                       1e-3, "cuda")
    ms, warm_s = _time_step_fn(step, "cuda")
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return {"ms_per_step": round(ms, 2), "warmup_s": round(warm_s, 1),
            "config": cfg.to_dict(), "strict_fp32": strict_fp32}


def git_meta():
    def run(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return None
    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain"))}


def clock_state():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,persistence_mode",
             "--format=csv,noheader"], text=True)
        return out.strip().splitlines()[0]
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", default="max")
    p.add_argument("--out", default="benchmarks/out")
    p.add_argument("--workloads", nargs="*", default=list(WORKLOADS))
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    meta = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gpu_name": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "budget": args.budget,
        "gpu_clock_state": clock_state(),
        **git_meta(),
    }
    print(json.dumps(meta, indent=1))

    results = {}
    for name in args.workloads:
        print(f"\n=== {name} ===")
        make = WORKLOADS[name]

        model, batch = make()
        strict = measure_fixed(model, batch, TrialConfig(), strict_fp32=True)
        print(f"  strict fp32 eager:        {strict['ms_per_step']:8.2f} ms")
        del model; torch.cuda.empty_cache()

        model, batch = make()
        default = measure_fixed(model, batch, TrialConfig())
        print(f"  PyTorch-default fp32:     {default['ms_per_step']:8.2f} ms")
        del model; torch.cuda.empty_cache()

        model, batch = make()
        compiled = measure_fixed(model, batch, TrialConfig(compile="default"))
        print(f"  plain torch.compile fp32: {compiled['ms_per_step']:8.2f} ms")
        del model; torch.cuda.empty_cache()

        model, batch = make()
        res = autotune(model, batch, budget=args.budget, cache=False, verbose=True)
        win = res.report.winner
        del model; torch.cuda.empty_cache()

        results[name] = {
            "strict_fp32_eager": strict,
            "pytorch_default_fp32_eager": default,
            "plain_compile_default_fp32": compiled,
            "autotuned": {"ms_per_step": win.ms_per_step,
                          "config": win.config.to_dict(),
                          "compile_s": win.compile_s,
                          "search_s": round(res.report.total_s, 1)},
            "trials": [{"config": t.config.to_dict(), "status": t.status,
                        "ms_per_step": t.ms_per_step, "compile_s": t.compile_s,
                        "note": t.note} for t in res.report.trials],
            "speedup_vs_strict_fp32": round(strict["ms_per_step"] / win.ms_per_step, 2),
            "speedup_vs_pytorch_default": round(default["ms_per_step"] / win.ms_per_step, 2),
            "speedup_vs_plain_compile": round(compiled["ms_per_step"] / win.ms_per_step, 2),
        }
        r = results[name]
        print(f"  autotuned [{win.config.label}]: {win.ms_per_step:.2f} ms -> "
              f"{r['speedup_vs_strict_fp32']}x vs strict fp32, "
              f"{r['speedup_vs_pytorch_default']}x vs PyTorch default, "
              f"{r['speedup_vs_plain_compile']}x vs plain compile")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = meta["gpu_name"].lower().replace("nvidia", "").strip()
    slug = "".join(c if c.isalnum() else "-" for c in slug).strip("-")
    stamp = meta["timestamp"].replace(":", "").replace("-", "")[:13]
    base = out_dir / f"suite_{stamp}_{slug}"
    json.dump({"metadata": meta, "results": results}, open(f"{base}.json", "w"), indent=1)

    lines = ["| workload | winner | vs strict fp32 | vs PyTorch default | vs plain compile |",
             "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {TrialConfig.from_dict(r['autotuned']['config']).label} "
                     f"| {r['speedup_vs_strict_fp32']}x | {r['speedup_vs_pytorch_default']}x "
                     f"| {r['speedup_vs_plain_compile']}x |")
    open(f"{base}.md", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {base}.json\nwrote {base}.md")


if __name__ == "__main__":
    main()
