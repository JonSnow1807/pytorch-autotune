"""Tests for the measurement-driven tuner (GPU tests auto-skip without CUDA)."""

import copy
import functools

import pytest
import torch
import torch.nn as nn

from pytorch_autotune import TrialConfig, autotune
from pytorch_autotune import cache as at_cache
from pytorch_autotune.tuner import (
    Report, TrialResult, _env_metadata, _make_optimizer, _optimizer_id, _time_step_fn,
)

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


# --------------------------------------------------------------------------- #
# v2.0 pre-publish hardening: dict batches, cache collisions, guards, UX.
# --------------------------------------------------------------------------- #


def test_cache_key_distinguishes_architectures():
    # Two DIFFERENT architectures with the same class name, parameter count
    # and input shape must not share a cache key (this collided before the
    # structural digest was added).
    import torch.nn as nn
    m1 = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
    n1 = sum(p.numel() for p in m1.parameters())
    m2 = None
    for h in range(1, 80):
        cand = nn.Sequential(nn.Linear(100, h), nn.Linear(h, 10))
        if sum(p.numel() for p in cand.parameters()) == n1:
            m2 = cand
            break
    assert m2 is not None, "could not construct a colliding architecture"
    sig = "(4, 100):torch.float32"
    assert at_cache.make_key(m1, sig, "cpu") != at_cache.make_key(m2, sig, "cpu")


def test_parameterless_model_raises_clearly():
    class NoParams(nn.Module):
        def forward(self, x):
            return x.sum(-1)

    with pytest.raises(ValueError, match="no trainable parameters"):
        from pytorch_autotune import autotune as _at
        _at(NoParams(), torch.randn(4, 8), budget=10, cache=False, verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu")


def test_report_json_and_payback_line():
    r = Report(env={"gpu_name": "T", "torch": torch.__version__})
    r.baseline_ms = 10.0
    base = TrialResult(TrialConfig(), "ok", ms_per_step=10.0)
    win = TrialResult(TrialConfig(precision="amp_bf16"), "ok", ms_per_step=5.0)
    r.add(base)
    r.add(win)
    r.finalize(win, total_s=60.0)
    text = str(r)
    assert "breaks even after" in text and "12,000" in text  # 60s / 5ms = 12000 steps
    d = r.to_dict()
    assert d["winner"]["precision"] == "amp_bf16" and len(d["trials"]) == 2
    import json as _json
    assert _json.loads(r.to_json())["baseline_ms"] == 10.0


@requires_cuda
def test_float_targets_use_mse_default():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    x = torch.randn(64, 16)
    y = torch.randn(64, 1)
    res = autotune(m, (x, y), budget=30, cache=False, verbose=False, compile_modes=[])
    assert res.report.winner.status == "ok"
    loss = res.step(x.cuda(), y.cuda())
    assert torch.isfinite(loss).all()


@requires_cuda
def test_dict_batch_custom_module():
    # A plain module taking kwargs, output without .loss -> loss_fn required.
    class KwModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(32, 4)

        def forward(self, tokens=None, mask=None):
            return self.lin(tokens * mask.unsqueeze(-1))

    batch = {"tokens": torch.randn(16, 32), "mask": torch.ones(16)}
    with pytest.raises(RuntimeError, match="no .loss"):
        autotune(KwModel(), batch, budget=20, cache=False, verbose=False,
                 compile_modes=[])
    res = autotune(KwModel(), batch, budget=30, cache=False, verbose=False,
                   compile_modes=[], loss_fn=lambda out: out.square().mean())
    assert res.report.winner.status == "ok"
    loss = res.step({k: v.cuda() for k, v in batch.items()})
    assert torch.isfinite(loss).all()


@requires_cuda
def test_hf_style_model_end_to_end():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.BertConfig(hidden_size=64, num_hidden_layers=2,
                                  num_attention_heads=2, intermediate_size=128,
                                  vocab_size=500, num_labels=3,
                                  max_position_embeddings=64)
    model = transformers.BertForSequenceClassification(cfg)
    batch = {
        "input_ids": torch.randint(0, 500, (8, 32)),
        "attention_mask": torch.ones(8, 32, dtype=torch.long),
        "labels": torch.randint(0, 3, (8,)),
    }
    res = autotune(model, batch, budget=45, cache=False, verbose=False,
                   compile_modes=[])  # eager-only keeps the test fast
    assert res.report.winner.status == "ok"
    okays = [t for t in res.report.trials if t.status == "ok"]
    assert len(okays) >= 2
    loss = res.step({k: v.cuda() for k, v in batch.items()})
    assert torch.isfinite(loss).all()


@requires_cuda
def test_keyboard_interrupt_returns_best_so_far(monkeypatch):
    from pytorch_autotune import tuner as tmod

    calls = {"n": 0}
    orig = tmod._run_trial

    def interrupting(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # baseline + one precision trial, then Ctrl-C
            raise KeyboardInterrupt
        return orig(*args, **kwargs)

    monkeypatch.setattr(tmod, "_run_trial", interrupting)
    m = tiny_mlp()
    res = autotune(m, batch(), budget=60, cache=False, verbose=False)
    assert res.report.interrupted is True
    assert res.report.winner is not None and res.report.winner.status == "ok"
    assert "interrupted" in str(res.report)
    loss = res.step(*[t.cuda() for t in batch()])
    assert torch.isfinite(loss).all()


# --------------------------------------------------------------------------- #
# Labels describe what was measured
# --------------------------------------------------------------------------- #


def test_fused_request_is_never_dropped_silently(monkeypatch):
    params = [nn.Parameter(torch.zeros(4))]
    for spec in ("sgd", lambda ps: torch.optim.AdamW(ps)):
        with pytest.raises(ValueError, match="no fused implementation"):
            _make_optimizer(spec, params, 1e-3, fused=True)

    class AdamWWithoutFused(torch.optim.AdamW):
        def __init__(self, params, *args, fused=None, **kwargs):
            if fused:
                raise RuntimeError("fused AdamW unavailable")
            super().__init__(params, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", AdamWWithoutFused)
    with pytest.raises(RuntimeError, match="fused AdamW unavailable"):
        _make_optimizer("adamw", params, 1e-3, fused=True)


def test_cache_key_distinguishes_optimizers():
    # A cached config must not be reused for a different optimizer, including
    # factories whose names collide (every functools.partial, two lambdas).
    m = tiny_mlp()
    sig = "(4, 64):torch.float32"
    specs = ["adamw", "sgd",
             functools.partial(torch.optim.AdamW, lr=1e-3),
             functools.partial(torch.optim.SGD, lr=1e-3),
             lambda ps: torch.optim.AdamW(ps, lr=1e-3),
             lambda ps: torch.optim.AdamW(ps, lr=1e-3, amsgrad=True)]
    ids = [_optimizer_id(s, m) for s in specs]
    assert len({at_cache.make_key(m, sig, "cpu", optimizer=i) for i in ids}) == len(specs) - 1
    # partial(AdamW) and the plain AdamW lambda build the same optimizer.
    assert ids[2] == ids[4]
    # Same group options, different parameters in each group.
    by_kind = lambda ps: torch.optim.AdamW([  # noqa: E731
        {"params": [m[0].weight, m[2].weight], "weight_decay": 0.1},
        {"params": [m[0].bias, m[2].bias], "weight_decay": 0.0}])
    by_layer = lambda ps: torch.optim.AdamW([  # noqa: E731
        {"params": [m[0].weight, m[0].bias], "weight_decay": 0.1},
        {"params": [m[2].weight, m[2].bias], "weight_decay": 0.0}])
    assert _optimizer_id(by_kind, m) != _optimizer_id(by_layer, m)


@requires_cuda
@pytest.mark.parametrize("optimizer, fused_searched", [
    ("adamw", True),
    ("sgd", False),
    (lambda ps: torch.optim.AdamW(ps, lr=1e-3), False),
], ids=["adamw", "sgd", "factory"])
def test_fused_label_matches_the_optimizer(optimizer, fused_searched):
    # Only the built-in Adam/AdamW have a fused option; SGD and caller
    # factories must never produce a trial or a result labelled "fused".
    res = autotune(tiny_mlp(), batch(), optimizer=optimizer, budget=30, cache=False,
                   verbose=False, compile_modes=[])
    assert any(t.config.fused_opt for t in res.report.trials) == fused_searched
    assert res.config.fused_opt == bool(res.optimizer.defaults.get("fused"))
    assert ("fused optimizer not searched" in str(res.report)) != fused_searched


@requires_cuda
def test_fused_not_searched_when_the_fused_optimizer_fails_to_build(monkeypatch):
    class AdamWWithoutFused(torch.optim.AdamW):
        def __init__(self, params, *args, fused=None, **kwargs):
            if fused:
                raise RuntimeError("fused AdamW unavailable")
            super().__init__(params, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", AdamWWithoutFused)
    res = autotune(tiny_mlp(), batch(), budget=30, cache=False, verbose=False,
                   compile_modes=[])
    assert not any(t.config.fused_opt for t in res.report.trials)
    assert not res.config.fused_opt
    assert res.report.winner.status == "ok"
    assert "fused AdamW unavailable" in str(res.report)


# --------------------------------------------------------------------------- #
# fp16 trials must time real optimizer steps
# --------------------------------------------------------------------------- #


class _InfGradInFp16(torch.autograd.Function):
    """Identity forward; non-finite gradient whenever the activation is fp16."""

    @staticmethod
    def forward(ctx, x):
        ctx.fp16 = x.dtype == torch.float16
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * float("inf") if ctx.fp16 else g


class FiniteLossInfGradFp16(nn.Module):
    """Finite loss in every precision, but fp16 gradients overflow every step,
    so GradScaler skips every fp16 optimizer step."""

    def __init__(self):
        super().__init__()
        self.net = tiny_mlp()

    def forward(self, x):
        h = _InfGradInFp16.apply(self.net[0](x))
        return self.net[2](self.net[1](h))


@requires_cuda
def test_fp16_candidates_whose_steps_are_skipped_are_disqualified():
    res = autotune(FiniteLossInfGradFp16(), batch(), budget=60, cache=False, verbose=False,
                   compile_modes=[])
    fp16 = [t for t in res.report.trials if t.config.precision == "amp_fp16"]
    assert fp16 and all(t.status == "disqualified" for t in fp16)
    assert "skipped" in fp16[0].note
    assert res.config.precision != "amp_fp16"


class _GradTimes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.factor, None


@requires_cuda
def test_loss_scale_calibration_during_warmup_is_not_disqualified():
    # GradScaler starts at 2**16 and halves on each overflow; here the first
    # 15 steps overflow. Warmup must absorb that calibration (more than the
    # 10-step warmup minimum) instead of timing or rejecting skipped steps.
    p = nn.Parameter(torch.ones((), device="cuda"))
    opt = torch.optim.SGD([p], lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        opt.zero_grad(set_to_none=True)
        loss = _GradTimes.apply(p, float("inf") if calls["n"] <= 15 else 1.0)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    step.scaler = scaler
    _time_step_fn(step, "cuda")  # raises FloatingPointError if it timed skipped steps
    assert scaler.get_scale() < 2.0 ** 16  # the calibration skips did happen
