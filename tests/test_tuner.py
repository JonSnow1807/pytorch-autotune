"""Tests for the measurement-driven tuner (GPU tests auto-skip without CUDA)."""

import copy

import pytest
import torch
import torch.nn as nn

from pytorch_autotune import TrialConfig, autotune
from pytorch_autotune import cache as at_cache
from pytorch_autotune.tuner import Report, TrialResult, _env_metadata

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def tiny_mlp(scale: float = 1.0) -> nn.Module:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    if scale != 1.0:
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(scale)
    return m


def batch(bs: int = 256):
    torch.manual_seed(1)
    return torch.randn(bs, 64), torch.randint(0, 10, (bs,))


# --------------------------------------------------------------------------- #
# CPU-safe tests
# --------------------------------------------------------------------------- #


def test_report_renders_without_gpu():
    r = Report(env={"gpu_name": "TestGPU", "torch": torch.__version__})
    r.baseline_ms = 10.0
    ok = TrialResult(TrialConfig(), "ok", ms_per_step=10.0)
    win = TrialResult(TrialConfig(precision="amp_bf16", compile="reduce-overhead"),
                      "ok", ms_per_step=4.0, compile_s=12.0)
    dq = TrialResult(TrialConfig(precision="amp_fp16"), "disqualified", note="non-finite loss")
    for t in (ok, win, dq):
        r.add(t)
    r.finalize(win, total_s=30.0)
    text = str(r)
    assert "disqualified" in text and "winner" in text and "non-finite" in text
    assert "CUDA graphs" in text  # static-shape warning for reduce-overhead
    assert win.speedup == pytest.approx(2.5)
    assert "baseline" in r.summary()


def test_trialconfig_roundtrip():
    cfg = TrialConfig(precision="amp_bf16", compile="max-autotune",
                      channels_last=True, fused_opt=True)
    assert TrialConfig.from_dict(cfg.to_dict()) == cfg
    assert "amp_bf16" in cfg.label and "max-autotune" in cfg.label


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(at_cache, "_CACHE_PATH", tmp_path / "cache.json")
    key = "k" * 24
    assert at_cache.load(key) is None
    at_cache.store(key, {"config": TrialConfig().to_dict(), "speedup": 1.5, "gpu": "X"})
    hit = at_cache.load(key)
    assert hit is not None and hit["speedup"] == 1.5
    at_cache.clear()
    assert at_cache.load(key) is None


def test_env_metadata_cpu_ok():
    env = _env_metadata("cpu")
    assert "torch" in env


# --------------------------------------------------------------------------- #
# GPU tests
# --------------------------------------------------------------------------- #


@requires_cuda
def test_search_returns_valid_result_and_baseline_guarantee():
    m = tiny_mlp()
    before = copy.deepcopy(m.state_dict())
    res = autotune(m, batch(), budget=90, cache=False, verbose=False,
                   compile_modes=["default"])  # keep the test fast
    # a real winner with a measured report
    assert res.config in [t.config for t in res.report.trials]
    okays = [t for t in res.report.trials if t.status == "ok"]
    assert len(okays) >= 2
    base_ms = res.report.baseline_ms
    win = res.report.winner
    assert win.ms_per_step <= base_ms * 1.03  # never worse than baseline
    # weights were restored to the pre-search state
    after = {k: v.detach().cpu() for k, v in res.model.state_dict().items()}
    for k in before:
        found = [ak for ak in after if ak.endswith(k)]  # compiled wrapper prefixes
        assert found, f"missing param {k}"
        assert torch.equal(before[k], after[found[0]]), f"weights changed: {k}"
    # the tuned step runs and is finite
    x, y = batch()
    loss = res.step(x.cuda(), y.cuda())
    assert torch.isfinite(loss).all()


@requires_cuda
def test_fp16_overflow_is_disqualified_not_returned():
    # Weights scaled so fp16 activations overflow to inf, but fp32/bf16 are
    # fine: the amp_fp16 candidate must be disqualified, and the winner must
    # not be fp16.
    m = tiny_mlp(scale=300.0)  # activations ~1e5-1e6 > fp16 max 65504
    res = autotune(m, batch(), budget=60, cache=False, verbose=False,
                   compile_modes=[])  # eager-only: precision sweep is the point
    fp16 = [t for t in res.report.trials if t.config.precision == "amp_fp16"]
    assert fp16 and all(t.status == "disqualified" for t in fp16)
    assert "non-finite" in fp16[0].note
    assert res.config.precision != "amp_fp16"


@requires_cuda
def test_budget_skips_are_recorded():
    m = tiny_mlp()
    res = autotune(m, batch(), budget=12, cache=False, verbose=False)
    # 12s cannot afford compile trials; they must appear as skipped, not
    # silently vanish.
    statuses = {t.status for t in res.report.trials}
    assert "skipped" in statuses
    assert res.report.winner.status == "ok"


@requires_cuda
def test_cache_hit_skips_search(tmp_path, monkeypatch):
    monkeypatch.setattr(at_cache, "_CACHE_PATH", tmp_path / "cache.json")
    m = tiny_mlp()
    r1 = autotune(m, batch(), budget=60, cache=True, verbose=False,
                  compile_modes=["default"])
    n_trials_first = len(r1.report.trials)
    assert n_trials_first >= 2
    m2 = tiny_mlp()
    r2 = autotune(m2, batch(), budget=60, cache=True, verbose=False,
                  compile_modes=["default"])
    assert len(r2.report.trials) == 0  # no search ran
    assert r2.config == r1.config
    loss = r2.step(*[t.cuda() for t in batch()])
    assert torch.isfinite(loss).all()


@requires_cuda
def test_allow_cudagraphs_false_excludes_reduce_overhead():
    m = tiny_mlp()
    res = autotune(m, batch(), budget=90, cache=False, verbose=False,
                   allow_cudagraphs=False, compile_modes=None)
    assert all(t.config.compile != "reduce-overhead" for t in res.report.trials)
