#!/usr/bin/env python3
"""Training-step benchmark behind the README's numbers.

Times a full training step (forward + loss + backward + optimizer) for:
  1. baseline: fp32 eager, plain AdamW
  2. autotune: the full quick_optimize() stack (AMP + torch.compile +
     fused AdamW + TF32 + channels-last)
  3. autotune-nocompile: same minus torch.compile
  4. compile-only: torch.compile on fp32, plain AdamW

Warmup covers torch.compile's compilation (one-off cost, excluded from the
timed region and reported separately in wall time you will feel). Run on an
idle GPU; results are per-call averages over --iters steps.

Usage:
  python benchmarks/measure.py                      # ResNet-18, 32px, b128
  python benchmarks/measure.py --model resnet50 --res 224 --batch 64
"""

import argparse
import time

import torch
import torch.nn as nn

try:
    import torchvision.models as tvm
except ImportError:
    raise SystemExit("torchvision is required: pip install torchvision")

from pytorch_autotune import AutoTune


def train_ms(model, opt, scaler, *, batch, res, classes, iters, warmup,
             channels_last=False):
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, res, res, device="cuda")
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    y = torch.randint(0, classes, (batch,), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def fresh(name, classes):
    torch.manual_seed(0)
    return getattr(tvm, name)(num_classes=classes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18")
    p.add_argument("--res", type=int, default=32)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=15)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    kw = dict(batch=args.batch, res=args.res, classes=args.classes,
              iters=args.iters, warmup=args.warmup)
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | "
          f"{args.model} {args.res}px batch {args.batch}")

    m = fresh(args.model, args.classes).cuda()
    base = train_ms(m, torch.optim.AdamW(m.parameters(), lr=1e-3), None, **kw)
    print(f"baseline fp32 eager:        {base:7.2f} ms/step")
    del m
    torch.cuda.empty_cache()

    m = fresh(args.model, args.classes)
    m, opt, scaler = AutoTune(m, verbose=False).optimize()
    kw_c = dict(kw, warmup=max(args.warmup, 30))  # cover compilation
    at = train_ms(m, opt, scaler, channels_last=True, **kw_c)
    print(f"autotune (full stack):      {at:7.2f} ms/step   {base / at:.2f}x")
    del m
    torch.cuda.empty_cache()

    m = fresh(args.model, args.classes)
    m, opt, scaler = AutoTune(m, verbose=False).optimize(use_compile=False)
    nc = train_ms(m, opt, scaler, channels_last=True, **kw)
    print(f"autotune w/o compile:       {nc:7.2f} ms/step   {base / nc:.2f}x")
    del m
    torch.cuda.empty_cache()

    m = fresh(args.model, args.classes).cuda()
    mc = torch.compile(m)
    co = train_ms(mc, torch.optim.AdamW(m.parameters(), lr=1e-3), None, **kw_c)
    print(f"torch.compile only (fp32):  {co:7.2f} ms/step   {base / co:.2f}x")


if __name__ == "__main__":
    main()
