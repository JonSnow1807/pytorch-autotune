"""Tiny on-disk cache of winning configs, keyed by (model, shapes, GPU, torch).

The cache stores only the chosen ``TrialConfig`` and its measured speedup -
never timing tables - so a hit skips the search but the report of a cached
run makes clear nothing was measured this time.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_CACHE_PATH = Path(os.environ.get(
    "PYTORCH_AUTOTUNE_CACHE",
    str(Path.home() / ".cache" / "pytorch_autotune" / "cache.json"),
))


def make_key(model, batch_signature: str, device: str, optimizer: str = "") -> str:
    from . import __version__
    # Structural digest: two different architectures must never share a key,
    # even with equal class names and parameter counts (a plain param-count
    # key demonstrably collided for two different nn.Sequentials).
    structure = ";".join(f"{name}:{tuple(p.shape)}"
                         for name, p in model.named_parameters())
    parts = [
        type(model).__name__,
        hashlib.sha256(structure.encode()).hexdigest()[:16],
        batch_signature,
        torch.cuda.get_device_name() if device.startswith("cuda") and torch.cuda.is_available() else device,
        torch.__version__,
        __version__,
        optimizer,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def _read_all() -> Dict[str, Any]:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def load(key: str) -> Optional[Dict[str, Any]]:
    return _read_all().get(key)


def store(key: str, value: Dict[str, Any]) -> None:
    try:
        entries = _read_all()
        entries[key] = value
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(entries, indent=1))
    except Exception:
        pass  # caching is best-effort; never break the caller


def clear() -> None:
    try:
        _CACHE_PATH.unlink()
    except FileNotFoundError:
        pass
