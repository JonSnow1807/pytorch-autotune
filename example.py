"""PyTorch AutoTune - Quick Example
Author: Chinmay Shrivastava
"""

from pytorch_autotune import quick_optimize
import torch.nn as nn

print("🚀 PyTorch AutoTune Example")
print("-" * 40)

# Create a simple model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

print("✅ Model created")

# Optimize with one line!
model, optimizer, scaler = quick_optimize(model, verbose=False)

print("⚡ Model optimized with AutoTune!")
print("📈 Ready for 4x faster training!")
print("\nInstall: pip install pytorch-autotune")
