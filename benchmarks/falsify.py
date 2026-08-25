"""Falsification suite: attack the committed v2.0.0 numbers.

F1  Real-loop check: re-measure each committed winner in a REAL training
    loop (fresh batch every step, cycled from a pool) and compare against
    the committed trial ms. Catches same-batch/CUDA-graph flattery.
F2  Rebuild variance: the returned artifact is recompiled after the search;
    measure the rebuilt winner twice - does it match its trial?
F3  fp16 scaler-skip audit: count skipped optimizer steps in the timed
    regime for the fp16 winner (skipped steps are cheaper => would inflate).
F4  Learning check: baseline vs tuned config trained 300 steps on a
    learnable synthetic task from identical init - does the tuned config
    actually learn (loss decreases comparably)?
F5  The informed-user baseline: one-line TF32 (set_float32_matmul_precision
    plus cudnn tf32) - how much of the transformer win survives vs that?
"""
import json, statistics, sys, time
import torch, torch.nn as nn
import torchvision.models as tvm

from pytorch_autotune.tuner import TrialConfig, _apply_precision_flags, _make_optimizer

from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
DATA = json.load(open(_REPO / "benchmarks/results/2026-08-25_a100-40gb_v200/"
                      "suite_20260825T1930_a100-sxm4-40gb.json"))
RES = DATA["results"]

def make(name):
    torch.manual_seed(0)
    if name == "resnet50_224px_b64":
        return tvm.resnet50(num_classes=1000), (64, 3, 224, 224), 1000
    if name == "resnet18_32px_b128":
        return tvm.resnet18(num_classes=10), (128, 3, 32, 32), 10
    layer = nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True, dropout=0.0)
    class TEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.TransformerEncoder(layer, 6)
            self.head = nn.Linear(512, 100)
        def forward(self, x):
            return self.head(self.enc(x).mean(1))
    return TEnc(), (32, 128, 512), 100

def build(model, cfg, device="cuda"):
    _apply_precision_flags(cfg)
    torch._dynamo.reset()
    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    run = torch.compile(model, mode=None if cfg.compile == "default" else cfg.compile) \
        if cfg.compile != "off" else model
    opt = _make_optimizer("adamw", model.parameters(), 1e-3, cfg.fused_opt)
    scaler = torch.amp.GradScaler("cuda") if cfg.precision == "amp_fp16" else None
    crit = nn.CrossEntropyLoss()
    def step(x, y):
        opt.zero_grad(set_to_none=True)
        if cfg.amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=cfg.amp_dtype):
                loss = crit(run(x), y)
        else:
            loss = crit(run(x), y)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        return loss
    return step, scaler

def real_loop_ms(step, shape, classes, cfg, n=40, warm=25, pool=8):
    torch.manual_seed(7)
    xs = [torch.randn(shape, device="cuda") for _ in range(pool)]
    if cfg.channels_last:
        xs = [x.to(memory_format=torch.channels_last) for x in xs]
    ys = [torch.randint(0, classes, (shape[0],), device="cuda") for _ in range(pool)]
    for i in range(warm):
        step(xs[i % pool], ys[i % pool])
    torch.cuda.synchronize()
    chunks = []
    for c in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(n):
            loss = step(xs[(c * n + i) % pool], ys[(c * n + i) % pool])
        torch.cuda.synchronize()
        chunks.append((time.time() - t0) / n * 1000)
    assert torch.isfinite(loss).all()
    return statistics.median(chunks)

print("=" * 72)
print("F1+F2: committed trial ms vs REAL fresh-batch loop (rebuilt twice)")
verdicts = []
for name in RES:
    win = TrialConfig.from_dict(RES[name]["autotuned"]["config"])
    committed = RES[name]["autotuned"]["ms_per_step"]
    reals = []
    for rebuild in range(2):
        model, shape, classes = make(name)
        step, _ = build(model, win)
        reals.append(real_loop_ms(step, shape, classes, win))
        del model, step
        torch._dynamo.reset(); torch.cuda.empty_cache()
    worst = max(reals)
    infl = (worst - committed) / committed * 100
    ok = worst <= committed * 1.10
    verdicts.append(ok)
    print(f"  {name:26s} committed {committed:7.2f} ms | real-loop rebuilds "
          f"{reals[0]:.2f}/{reals[1]:.2f} ms | drift {infl:+.1f}% | {'OK' if ok else 'INFLATED'}")

print("\nF1b: baseline under the same real loop (is the denominator honest?)")
for name in RES:
    committed_base = RES[name]["pytorch_default_fp32_eager"]["ms_per_step"]
    model, shape, classes = make(name)
    step, _ = build(model, TrialConfig())
    real = real_loop_ms(step, shape, classes, TrialConfig())
    del model, step; torch.cuda.empty_cache()
    print(f"  {name:26s} committed {committed_base:7.2f} ms | real-loop {real:7.2f} ms "
          f"| drift {(real-committed_base)/committed_base*100:+.1f}%")

print("\nF3: fp16 scaler-skip audit (resnet50 winner)")
model, shape, classes = make("resnet50_224px_b64")
cfg = TrialConfig.from_dict(RES["resnet50_224px_b64"]["autotuned"]["config"])
step, scaler = build(model, cfg)
x = torch.randn(shape, device="cuda").to(memory_format=torch.channels_last)
y = torch.randint(0, classes, (shape[0],), device="cuda")
for _ in range(20):
    step(x, y)
skips = 0
for _ in range(60):
    s0 = scaler.get_scale()
    step(x, y)
    if scaler.get_scale() < s0:
        skips += 1
print(f"  skipped optimizer steps in 60 timed-regime steps: {skips} "
      f"({'OK - negligible' if skips <= 2 else 'CHECK: timing flattered'})")
del model, step; torch._dynamo.reset(); torch.cuda.empty_cache()

print("\nF4: does the tuned config LEARN? (resnet18, 300 steps, learnable task)")
def learnable_batches(classes=10, n_batches=20, bs=128):
    torch.manual_seed(11)
    protos = torch.randn(classes, 3, 32, 32, device="cuda") * 2
    xs, ys = [], []
    for _ in range(n_batches):
        y = torch.randint(0, classes, (bs,), device="cuda")
        x = protos[y] + 0.6 * torch.randn(bs, 3, 32, 32, device="cuda")
        xs.append(x); ys.append(y)
    return xs, ys

xs, ys = learnable_batches()
for label, cfg in [("baseline fp32", TrialConfig()),
                   ("tuned winner", TrialConfig.from_dict(RES["resnet18_32px_b128"]["autotuned"]["config"]))]:
    model, _, _ = make("resnet18_32px_b128")
    step, _ = build(model, cfg)
    first = last = None
    data = [x.to(memory_format=torch.channels_last) for x in xs] if cfg.channels_last else xs
    for i in range(300):
        loss = step(data[i % len(xs)], ys[i % len(ys)])
        if i == 0: first = loss.item()
        last = loss.item()
    print(f"  {label:14s} loss step0 {first:6.3f} -> step300 {last:6.3f} "
          f"({'LEARNS' if last < first * 0.5 else 'NOT LEARNING - CATCH'})")
    del model, step; torch._dynamo.reset(); torch.cuda.empty_cache()

print("\nF5: informed-user TF32 one-liner baseline (transformer)")
model, shape, classes = make("transformer_6l_s128_b32")
step, _ = build(model, TrialConfig(precision="tf32"))  # tf32 + benchmark, eager
real = real_loop_ms(step, shape, classes, TrialConfig(precision="tf32"))
win_ms = RES["transformer_6l_s128_b32"]["autotuned"]["ms_per_step"]
base = RES["transformer_6l_s128_b32"]["pytorch_default_fp32_eager"]["ms_per_step"]
print(f"  eager TF32 one-liner: {real:.2f} ms -> tuned wins {real/win_ms:.2f}x over it "
      f"(vs the README's {base/win_ms:.2f}x over PyTorch defaults)")
print("\nFALSIFICATION SUITE DONE")
