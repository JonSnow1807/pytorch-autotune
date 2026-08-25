"""Smoke tests for the legacy heuristic API (kept for backward compat)."""

import pytest
import torch
import torch.nn as nn

import pytorch_autotune
from pytorch_autotune import AutoTune, quick_optimize

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_imports_and_version():
    assert pytorch_autotune.__version__.startswith("2.")
    assert callable(quick_optimize) and callable(pytorch_autotune.autotune)


@requires_cuda
def test_quick_optimize_trains():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
    model, opt, scaler = quick_optimize(model, verbose=False)
    x = torch.randn(64, 32, device="cuda")
    y = torch.randint(0, 4, (64,), device="cuda")
    crit = nn.CrossEntropyLoss()
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        loss = crit(model(x), y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).all()


@requires_cuda
def test_autotune_class_overrides():
    model = nn.Linear(16, 4)
    at = AutoTune(model, verbose=False)
    m, opt, scaler = at.optimize(use_compile=False, use_amp=False)
    assert scaler is None
    out = m(torch.randn(8, 16, device="cuda"))
    assert out.shape == (8, 4)
