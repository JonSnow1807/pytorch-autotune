"""PyTorch AutoTune — quick example of the measured tuner (v2.0).

Runs a short search on a small model and prints the honest trial report.
Requires a CUDA GPU; on CPU it exits after showing the heuristic fallback.
"""

import torch
import torch.nn as nn

from pytorch_autotune import autotune, quick_optimize

model = nn.Sequential(
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10),
)

if not torch.cuda.is_available():
    print("No CUDA GPU - showing the heuristic (unmeasured) fallback only.")
    model, opt, scaler = quick_optimize(model, device="cpu", verbose=True)
    raise SystemExit(0)

x = torch.randn(256, 784)
y = torch.randint(0, 10, (256,))

# Measure candidate configs on THIS model and batch, keep the fastest.
result = autotune(model, (x, y), budget="fast", cache=False, verbose=True)

# Train with the winner (weights are your original ones - tuning does not train).
for _ in range(5):
    loss = result.step(x.cuda(), y.cuda())
print(f"\n5 training steps with the tuned config ran fine (loss {loss.item():.3f})")
print(result.summary())
