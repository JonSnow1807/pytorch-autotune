"""Measurement-driven autotuning: try candidate configs on YOUR model, keep
the fastest, never return something slower than your baseline.

The knob space (precision x torch.compile mode x memory format x fused
optimizer) is searched in stages ordered by measured value density, under a
wall-clock budget. Every trial is a real training step (forward + loss +
backward + optimizer) on the caller's example batch; results are reported
honestly, including losers, disqualifications and compile costs.

Design notes:
* The model's weights are snapshotted before the search and restored between
  trials and before returning, so tuning does not train the model.
* The baseline (PyTorch defaults, fp32 eager, plain optimizer) is always
  measured first and is the reference for every speedup. If nothing beats it
  by more than the noise floor, the baseline config is returned - "never
  slower than what you already had" is structural, not aspirational.
* Precision configs set process-global flags (TF32, cudnn.benchmark); the
  flags of the WINNING config are left applied on return, and recorded in
  the report.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import platform
import statistics
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["autotune", "AutotuneResult", "TrialConfig", "TrialResult", "Report"]

_PRECISIONS = ("fp32_default", "tf32", "amp_bf16", "amp_fp16")
_COMPILE_MODES = ("off", "default", "reduce-overhead", "max-autotune")

# Conservative compile-cost estimates (seconds) used only for budget
# accounting - measured costs on an A100 ranged 1-50s (default),
# 12-40s (reduce-overhead) and 60-212s (max-autotune) across three
# workloads; actuals are recorded per trial.
_COMPILE_COST_EST = {"off": 0.0, "default": 25.0, "reduce-overhead": 45.0,
                     "max-autotune": 200.0}
_NOISE_FLOOR = 1.03  # a candidate must beat the incumbent by >3% to win


@dataclasses.dataclass(frozen=True)
class TrialConfig:
    precision: str = "fp32_default"   # fp32_default | tf32 | amp_bf16 | amp_fp16
    compile: str = "off"              # off | default | reduce-overhead | max-autotune
    channels_last: bool = False
    fused_opt: bool = False

    @property
    def label(self) -> str:
        parts = [self.precision, f"compile={self.compile}"]
        if self.channels_last:
            parts.append("channels_last")
        if self.fused_opt:
            parts.append("fused")
        return " ".join(parts)

    @property
    def amp_dtype(self) -> Optional[torch.dtype]:
        return {"amp_bf16": torch.bfloat16, "amp_fp16": torch.float16}.get(self.precision)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrialConfig":
        return TrialConfig(**d)


@dataclasses.dataclass
class TrialResult:
    config: TrialConfig
    status: str                 # ok | disqualified | skipped
    ms_per_step: Optional[float] = None
    compile_s: float = 0.0
    note: str = ""
    speedup: Optional[float] = None  # vs baseline; filled by the search


class Report:
    """All trials of one autotune() run, printable as an honest table."""

    def __init__(self, env: Dict[str, Any]):
        self.env = env
        self.trials: List[TrialResult] = []
        self.baseline_ms: Optional[float] = None
        self.winner: Optional[TrialResult] = None
        self.total_s: float = 0.0

    def add(self, t: TrialResult) -> None:
        self.trials.append(t)

    def finalize(self, winner: TrialResult, total_s: float) -> None:
        self.winner = winner
        self.total_s = total_s
        for t in self.trials:
            if t.status == "ok" and self.baseline_ms:
                t.speedup = self.baseline_ms / t.ms_per_step

    def summary(self) -> str:
        w = self.winner
        if w is None:
            return "autotune: no result"
        return (f"autotune picked [{w.config.label}]: {w.ms_per_step:.2f} ms/step, "
                f"{(w.speedup or 1.0):.2f}x vs this run's measured baseline "
                f"(PyTorch-default fp32 eager, {self.baseline_ms:.2f} ms/step) "
                f"on {self.env.get('gpu_name', 'unknown GPU')}")

    def __str__(self) -> str:
        if not self.trials:
            return ("autotune report - no trials ran (config restored from cache; "
                    "pass cache='refresh' to re-measure)")
        base = f"{self.baseline_ms:.2f} ms/step" if self.baseline_ms else "n/a"
        lines = [
            f"autotune report - {self.env.get('gpu_name')} | torch {self.env.get('torch')} "
            f"| search took {self.total_s:.0f}s",
            f"baseline = PyTorch-default fp32 eager: {base}",
            f"{'config':52s} {'status':13s} {'ms/step':>9s} {'vs base':>8s} {'compile':>8s}",
        ]
        for t in self.trials:
            ms = f"{t.ms_per_step:.2f}" if t.ms_per_step is not None else "-"
            sp = f"{t.speedup:.2f}x" if t.speedup else "-"
            cs = f"{t.compile_s:.0f}s" if t.compile_s else "-"
            mark = " <== winner" if self.winner is t else ""
            note = f"  ({t.note})" if t.note and t.status != "ok" else ""
            lines.append(f"{t.config.label:52s} {t.status:13s} {ms:>9s} {sp:>8s} {cs:>8s}{mark}{note}")
        if self.winner and self.winner.config.compile == "reduce-overhead":
            lines.append("note: reduce-overhead uses CUDA graphs - keep input shapes static.")
        return "\n".join(lines)


class AutotuneResult:
    """The tuned training setup. Weights equal the model's pre-search state."""

    def __init__(self, model: nn.Module, optimizer, scaler, config: TrialConfig,
                 report: Report, loss_fn: Callable, device: str):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.report = report
        self._loss_fn = loss_fn
        self._device = device

    def autocast(self):
        dt = self.config.amp_dtype
        if dt is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(self._device.split(":")[0], dtype=dt)

    def step(self, data: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One full training step with the tuned setup; returns the loss."""
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            out = self.model(data)
            loss = self._loss_fn(out, target) if target is not None else self._loss_fn(out)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return loss

    def summary(self) -> str:
        return self.report.summary()


# --------------------------------------------------------------------------- #
# Environment / global-flag handling
# --------------------------------------------------------------------------- #

def _env_metadata(device: str) -> Dict[str, Any]:
    env: Dict[str, Any] = {"torch": torch.__version__,
                           "platform": platform.platform()}
    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name()
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=clocks.sm,persistence_mode",
                 "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
            env["gpu_clock_state"] = out.strip().splitlines()[0]
        except Exception:
            env["gpu_clock_state"] = None
    return env


def _apply_precision_flags(cfg: TrialConfig) -> None:
    if cfg.precision == "fp32_default":
        # PyTorch's own defaults: matmul TF32 off, cudnn TF32 on, no benchmark.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _is_cnn(model: nn.Module) -> bool:
    return any(isinstance(m, nn.Conv2d) for m in model.modules())


def _supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _supports_fused_opt() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7


# --------------------------------------------------------------------------- #
# Trial engine
# --------------------------------------------------------------------------- #

def _make_optimizer(optimizer: Union[str, Callable], params, lr: float, fused: bool):
    if callable(optimizer) and not isinstance(optimizer, str):
        return optimizer(params)
    name = optimizer.lower()
    if name in ("adamw", "adam"):
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        if fused:
            try:
                return cls(params, lr=lr, fused=True)
            except Exception:
                pass
        return cls(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer {optimizer!r}")


def _build_step(model: nn.Module, cfg: TrialConfig, data, target, loss_fn,
                optimizer_spec, lr: float, device: str):
    """Build (step_fn, cleanup_hint) for one candidate. May raise (e.g. a
    compile failure at build time); the caller records that as disqualified."""
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
        data = data.to(memory_format=torch.channels_last)
    run_model = model
    if cfg.compile != "off":
        mode = None if cfg.compile == "default" else cfg.compile
        run_model = torch.compile(model, mode=mode)
    opt = _make_optimizer(optimizer_spec, model.parameters(), lr, cfg.fused_opt)
    scaler = torch.amp.GradScaler(device.split(":")[0]) if cfg.precision == "amp_fp16" else None
    amp_dtype = cfg.amp_dtype

    def step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.amp.autocast(device.split(":")[0], dtype=amp_dtype):
                out = run_model(data)
                loss = loss_fn(out, target) if target is not None else loss_fn(out)
        else:
            out = run_model(data)
            loss = loss_fn(out, target) if target is not None else loss_fn(out)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        return loss

    return step


def _time_step_fn(step, device: str, min_chunk_s: float = 0.2,
                  max_warm_steps: int = 80) -> Tuple[float, float]:
    """Warm up until per-step time stabilises (this absorbs compilation and
    CUDA-graph capture; its wall time is returned as compile/warmup cost),
    then time 3 chunks and return (median ms/step, warmup_s)."""
    cuda = device.startswith("cuda")

    warm_t0 = time.time()
    recent: List[float] = []
    for i in range(max_warm_steps):
        t0 = time.time()
        loss = step()
        if cuda:
            torch.cuda.synchronize()
        recent.append(time.time() - t0)
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("non-finite loss during warmup")
        if i >= 9 and max(recent[-3:]) < 1.5 * min(recent[-3:]):
            break
    warm_s = time.time() - warm_t0

    per_step = statistics.median(recent[-3:])
    n = max(3, int(math.ceil(min_chunk_s / max(per_step, 1e-4))))
    chunk_ms: List[float] = []
    for _ in range(3):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            loss = step()
        if cuda:
            torch.cuda.synchronize()
        chunk_ms.append((time.time() - t0) / n * 1000)
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError("non-finite loss during measurement")
    return statistics.median(chunk_ms), warm_s


def _run_trial(model, cfg: TrialConfig, data, target, loss_fn, optimizer_spec,
               lr, device, state_snapshot) -> TrialResult:
    _apply_precision_flags(cfg)
    torch._dynamo.reset()
    try:
        step = _build_step(model, cfg, data, target, loss_fn, optimizer_spec, lr, device)
        ms, warm_s = _time_step_fn(step, device)
        compile_s = warm_s if cfg.compile != "off" else 0.0
        result = TrialResult(cfg, "ok", ms_per_step=ms, compile_s=compile_s)
    except FloatingPointError as e:
        result = TrialResult(cfg, "disqualified", note=str(e))
    except torch.cuda.OutOfMemoryError:
        result = TrialResult(cfg, "disqualified", note="out of memory")
    except RuntimeError as e:
        note = "out of memory" if "out of memory" in str(e) else str(e)[:100]
        result = TrialResult(cfg, "disqualified", note=note)
    except Exception as e:  # compile failures etc.
        result = TrialResult(cfg, "disqualified", note=f"{type(e).__name__}: {str(e)[:90]}")
    finally:
        # Undo trial-local mutation: weights, memory format, dynamo state.
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        model.load_state_dict(state_snapshot)
        if cfg.channels_last:
            model.to(memory_format=torch.contiguous_format)
        torch._dynamo.reset()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# The staged search
# --------------------------------------------------------------------------- #

_BUDGETS_S = {"fast": 120, "balanced": 360, "max": 900}


def autotune(
    model: nn.Module,
    example_batch,
    loss_fn: Optional[Callable] = None,
    optimizer: Union[str, Callable] = "adamw",
    lr: float = 1e-3,
    budget: Union[str, int] = "balanced",
    device: str = "cuda",
    cache: Union[bool, str] = True,
    allow_cudagraphs: bool = True,
    compile_modes: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> AutotuneResult:
    """Measure candidate training configs on ``model`` and return the fastest.

    ``example_batch`` is a tensor or an ``(input, target)`` pair with the
    shapes you will actually train at. The search runs real training steps
    on copies of the model's state (weights are restored afterwards).
    """
    budget_s = _BUDGETS_S[budget] if isinstance(budget, str) else float(budget)
    if isinstance(example_batch, (tuple, list)):
        data, target = example_batch
    else:
        data, target = example_batch, None
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss() if target is not None else (lambda out: out.float().mean())

    model = model.to(device)
    data = data.to(device)
    if target is not None:
        target = target.to(device)

    env = _env_metadata(device)
    report = Report(env)
    t_start = time.time()

    def remaining() -> float:
        return budget_s - (time.time() - t_start)

    # Cache short-circuit
    from . import cache as _cache
    cache_key = _cache.make_key(model, data, target, device) if cache else None
    if cache is True and cache_key:
        hit = _cache.load(cache_key)
        if hit is not None:
            cfg = TrialConfig.from_dict(hit["config"])
            if verbose:
                print(f"autotune: cache hit -> {cfg.label} "
                      f"(measured {hit.get('speedup', 0):.2f}x on {hit.get('gpu', '?')}); "
                      f"pass cache='refresh' to re-measure")
            return _finalize(model, cfg, report, data, target, loss_fn, optimizer, lr,
                             device, from_cache=True)

    # Weight snapshot (CPU) - trials train the model; we put it back.
    snapshot = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    cnn = _is_cnn(model) and isinstance(data, torch.Tensor) and data.dim() == 4
    fused_ok = _supports_fused_opt()

    def trial(cfg: TrialConfig, est_s: float) -> Optional[TrialResult]:
        if remaining() < est_s:
            r = TrialResult(cfg, "skipped", note=f"budget ({remaining():.0f}s left, ~{est_s:.0f}s needed)")
            report.add(r)
            return None
        r = _run_trial(model, cfg, data, target, loss_fn, optimizer, lr, device, snapshot)
        report.add(r)
        if verbose:
            ms = f"{r.ms_per_step:.2f} ms" if r.ms_per_step else r.note
            print(f"  [{r.status:12s}] {cfg.label:48s} {ms}")
        return r

    if verbose:
        print(f"autotune: searching on {env.get('gpu_name', device)} "
              f"(budget {budget_s:.0f}s)")

    # Stage 1: the baseline. Everything is measured against this.
    base_cfg = TrialConfig()
    base = trial(base_cfg, est_s=8)
    if base is None or base.status != "ok":
        raise RuntimeError(
            f"autotune: the baseline training step failed "
            f"({base.note if base else 'budget'}) - fix the model/batch first")
    report.baseline_ms = base.ms_per_step
    best = base

    def consider(r: Optional[TrialResult]) -> None:
        nonlocal best
        if r is not None and r.status == "ok" and r.ms_per_step * _NOISE_FLOOR < best.ms_per_step:
            best = r

    # Stage 2: precision sweep, no compile.
    precisions = ["tf32", "amp_bf16", "amp_fp16"] if _supports_bf16() else ["tf32", "amp_fp16"]
    stage2: Dict[str, TrialResult] = {}
    for prec in precisions:
        r = trial(TrialConfig(precision=prec, channels_last=cnn, fused_opt=fused_ok), est_s=8)
        if r is not None and r.status == "ok":
            stage2[prec] = r
            consider(r)

    # Stage 3: compile sweep on the best two eager precisions.
    modes = list(compile_modes) if compile_modes is not None else ["default", "reduce-overhead"]
    if not allow_cudagraphs and "reduce-overhead" in modes:
        modes.remove("reduce-overhead")
    ranked = sorted(stage2, key=lambda p: stage2[p].ms_per_step)[:2] or ["amp_fp16"]
    stage3_best_mode = None
    for mode in modes:
        for prec in ranked:
            r = trial(TrialConfig(precision=prec, compile=mode, channels_last=cnn,
                                  fused_opt=fused_ok),
                      est_s=_COMPILE_COST_EST.get(mode, 30) + 8)
            if r is not None and r.status == "ok" and (best is r or r.ms_per_step <= best.ms_per_step):
                stage3_best_mode = mode
            consider(r)

    # Stage 4: max-autotune on the winning precision, budget permitting.
    if compile_modes is None and best.config.precision.startswith("amp"):
        r = trial(dataclasses.replace(best.config, compile="max-autotune"),
                  est_s=_COMPILE_COST_EST["max-autotune"] + 10)
        consider(r)

    # Stage 5: micro-toggles on the winner.
    if best.config.channels_last:
        consider(trial(dataclasses.replace(best.config, channels_last=False),
                       est_s=_COMPILE_COST_EST.get(best.config.compile, 0) + 8))
    if best.config.fused_opt:
        consider(trial(dataclasses.replace(best.config, fused_opt=False),
                       est_s=_COMPILE_COST_EST.get(best.config.compile, 0) + 8))

    report.finalize(best, time.time() - t_start)
    if verbose:
        print(report)
        print(report.summary())

    if cache and cache_key and best.status == "ok":
        _cache.store(cache_key, {"config": best.config.to_dict(),
                                 "speedup": best.speedup or 1.0,
                                 "gpu": env.get("gpu_name", "?")})

    # Rebuild the winner cleanly from the snapshot and hand it back.
    model.load_state_dict(snapshot)
    return _finalize(model, best.config, report, data, target, loss_fn, optimizer,
                     lr, device)


def _finalize(model, cfg: TrialConfig, report: Report, data, target, loss_fn,
              optimizer_spec, lr, device, from_cache: bool = False) -> AutotuneResult:
    _apply_precision_flags(cfg)
    torch._dynamo.reset()
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    run_model = model
    if cfg.compile != "off":
        mode = None if cfg.compile == "default" else cfg.compile
        run_model = torch.compile(model, mode=mode)
    opt = _make_optimizer(optimizer_spec, model.parameters(), lr, cfg.fused_opt)
    scaler = torch.amp.GradScaler(device.split(":")[0]) if cfg.precision == "amp_fp16" else None
    return AutotuneResult(run_model, opt, scaler, cfg, report, loss_fn, device)
