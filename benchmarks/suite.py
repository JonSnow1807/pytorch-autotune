#!/usr/bin/env python3
"""The benchmark suite behind the README's numbers.

For each workload this runs autotune() (real search, chosen budget) and, for
reference, four fixed configurations measured with the same trial protocol:

  * strict fp32 eager (TF32 fully off) - the baseline the project's
    retracted v1.0 claims implicitly used,
  * PyTorch-default fp32 eager - what an untuned user actually gets
    (cuDNN TF32 on, matmul TF32 off),
  * plain ``torch.compile`` (default mode) on PyTorch-default fp32 - the
    "just use torch.compile" alternative,
  * an expert configuration, set by hand without measuring: bf16 autocast
    (fp16 on GPUs without bf16), ``torch.compile`` default mode, fused AdamW,
    TF32 and cudnn.benchmark on, channels_last for convnets.

Every measurement runs in a fresh subprocess with its own Inductor, Triton
and CUDA cache directories. The search runs twice: cold (empty caches, what
a first run costs) and then warm (reusing the caches the cold run filled,
what a repeat run costs). Speedups use the cold search's winner.

Speedups are reported against all four references, labelled. Output: JSON
(+ md table) with git/GPU/clock metadata, suitable for committing under
benchmarks/results/.

Usage:
  python benchmarks/suite.py --budget max --out benchmarks/results/out
"""

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

try:
    import torchvision.models as tvm
except ImportError:
    raise SystemExit("torchvision is required: pip install torchvision")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pytorch_autotune.tuner import (  # noqa: E402
    TrialConfig, _apply_precision_flags, _Batch, _build_step, _is_cnn, _time_step_fn, autotune,
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


def expert_config(model, batch) -> TrialConfig:
    """The hand-set configuration an informed user would pick without measuring."""
    data, _ = batch
    return TrialConfig(
        precision="amp_bf16" if torch.cuda.is_bf16_supported() else "amp_fp16",
        compile="default",
        channels_last=_is_cnn(model) and data.dim() == 4,
        fused_opt=True,
    )


# Reference rows: name -> (config builder, strict fp32).
FIXED_ROWS = {
    "strict_fp32_eager": (lambda m, b: TrialConfig(), True),
    "pytorch_default_fp32_eager": (lambda m, b: TrialConfig(), False),
    "plain_compile_default_fp32": (lambda m, b: TrialConfig(compile="default"), False),
    "expert_config": (expert_config, False),
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
    step = _build_step(model, cfg, _Batch("pair", data=data, target=target),
                       nn.CrossEntropyLoss(), "adamw", 1e-3, "cuda")
    ms, warm_s = _time_step_fn(step, "cuda")
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return {"ms_per_step": round(ms, 2), "warmup_s": round(warm_s, 1),
            "config": cfg.to_dict(), "strict_fp32": strict_fp32}


def fixed_worker(workload: str, row: str):
    make_cfg, strict = FIXED_ROWS[row]
    model, batch = WORKLOADS[workload]()
    return measure_fixed(model, batch, make_cfg(model, batch), strict_fp32=strict)


def search_worker(workload: str, budget):
    model, batch = WORKLOADS[workload]()
    res = autotune(model, batch, budget=budget, cache=False, verbose=True)
    win = res.report.winner
    return {"ms_per_step": win.ms_per_step,
            "config": win.config.to_dict(),
            "compile_s": win.compile_s,
            "search_s": round(res.report.total_s, 1),
            "trials": [{"config": t.config.to_dict(), "status": t.status,
                        "ms_per_step": t.ms_per_step, "compile_s": t.compile_s,
                        "note": t.note} for t in res.report.trials]}


def run_worker(kind: str, workload: str, budget, cache_dir: Path, row: str = ""):
    """Run one measurement in a fresh process whose compile caches live in
    ``cache_dir`` (empty on first use, reused when passed again)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = cache_dir.with_name(f"{cache_dir.name}-result.json")
    env = dict(os.environ,
               TORCHINDUCTOR_CACHE_DIR=str(cache_dir / "inductor"),
               TRITON_CACHE_DIR=str(cache_dir / "triton"),
               CUDA_CACHE_PATH=str(cache_dir / "nv"))
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", kind,
                    "--workload", workload, "--row", row, "--budget", str(budget),
                    "--result", str(result)], env=env, check=True)
    return json.loads(result.read_text())


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


def speedup(ref, win):
    return round(ref["ms_per_step"] / win["ms_per_step"], 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", default="max")
    p.add_argument("--out", default="benchmarks/out")
    p.add_argument("--workloads", nargs="*", default=list(WORKLOADS))
    p.add_argument("--worker", choices=["fixed", "search"], help=argparse.SUPPRESS)
    p.add_argument("--workload", help=argparse.SUPPRESS)
    p.add_argument("--row", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    budget = int(args.budget) if args.budget.isdigit() else args.budget

    if args.worker:
        out = (fixed_worker(args.workload, args.row) if args.worker == "fixed"
               else search_worker(args.workload, budget))
        Path(args.result).write_text(json.dumps(out))
        return

    meta = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gpu_name": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "budget": args.budget,
        "gpu_clock_state": clock_state(),
        "cache_protocol": ("each row and search in a fresh process with its own empty "
                           "Inductor/Triton/CUDA cache dirs; search_warm reuses "
                           "search_cold's dirs"),
        **git_meta(),
    }
    print(json.dumps(meta, indent=1))

    cache_root = Path(tempfile.mkdtemp(prefix="autotune-suite-"))
    results = {}
    try:
        for name in args.workloads:
            print(f"\n=== {name} ===", flush=True)
            r = {row: run_worker("fixed", name, budget, cache_root / name / row, row=row)
                 for row in FIXED_ROWS}
            for row in FIXED_ROWS:
                print(f"  {row:28s} {r[row]['ms_per_step']:8.2f} ms", flush=True)
            cold = run_worker("search", name, budget, cache_root / name / "search")
            warm = run_worker("search", name, budget, cache_root / name / "search")
            r.update({
                "autotuned": {k: cold[k] for k in ("ms_per_step", "config", "compile_s",
                                                   "search_s")},
                "trials": cold["trials"],
                "search_cold": cold,
                "search_warm": warm,
                "speedup_vs_strict_fp32": speedup(r["strict_fp32_eager"], cold),
                "speedup_vs_pytorch_default": speedup(r["pytorch_default_fp32_eager"], cold),
                "speedup_vs_plain_compile": speedup(r["plain_compile_default_fp32"], cold),
                "speedup_vs_expert_config": speedup(r["expert_config"], cold),
            })
            results[name] = r
            label = TrialConfig.from_dict(cold["config"]).label
            print(f"  autotuned [{label}]: {cold['ms_per_step']:.2f} ms -> "
                  f"{r['speedup_vs_pytorch_default']}x vs PyTorch default, "
                  f"{r['speedup_vs_expert_config']}x vs expert config; search "
                  f"{cold['search_s']:.0f}s cold, {warm['search_s']:.0f}s warm", flush=True)
            if warm["config"] != cold["config"]:
                print(f"  warm search picked a different winner: "
                      f"[{TrialConfig.from_dict(warm['config']).label}]", flush=True)
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = meta["gpu_name"].lower().replace("nvidia", "").strip()
    slug = "".join(c if c.isalnum() else "-" for c in slug).strip("-")
    stamp = meta["timestamp"].replace(":", "").replace("-", "")[:13]
    base = out_dir / f"suite_{stamp}_{slug}"
    json.dump({"metadata": meta, "results": results}, open(f"{base}.json", "w"), indent=1)

    lines = ["| workload | winner (cold search) | vs strict fp32 | vs PyTorch default "
             "| vs plain compile | vs expert config | search cold / warm |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {TrialConfig.from_dict(r['autotuned']['config']).label} "
                     f"| {r['speedup_vs_strict_fp32']}x | {r['speedup_vs_pytorch_default']}x "
                     f"| {r['speedup_vs_plain_compile']}x | {r['speedup_vs_expert_config']}x "
                     f"| {r['search_cold']['search_s']:.0f}s / {r['search_warm']['search_s']:.0f}s |")
    open(f"{base}.md", "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {base}.json\nwrote {base}.md")


if __name__ == "__main__":
    main()
