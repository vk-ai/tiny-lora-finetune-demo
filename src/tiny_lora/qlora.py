"""Optional QLoRA *concept* stub — bitsandbytes optional (skipped when absent).

Real QLoRA uses NF4 block quantization of frozen base weights + LoRA in higher
precision (peft + bitsandbytes). This module:

- If ``bitsandbytes`` is importable, exposes a thin flag helper (still trains our
  numpy toy — we do **not** pull HF Trainer).
- If absent, ``qlora_available()`` is False and callers should skip / message.

A tiny numpy fake-4bit path is also provided for offline teaching of the
memory story without requiring bnb in CI.

See: http://mccormickml.com/2024/09/14/qlora-and-4bit-quantization/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_BNB_ERR: str | None
try:
    import bitsandbytes as bnb  # noqa: F401

    _BNB_AVAILABLE = True
    _BNB_ERR = None
except ImportError as exc:  # pragma: no cover - usual CI path
    bnb = None  # type: ignore
    _BNB_AVAILABLE = False
    _BNB_ERR = str(exc)


def qlora_available() -> bool:
    """True only when bitsandbytes imports successfully."""
    return _BNB_AVAILABLE


def qlora_skip_message() -> str:
    return (
        "QLoRA optional path skipped: bitsandbytes not installed "
        f"({_BNB_ERR or 'ImportError'}). Install bitsandbytes to enable; "
        "CI uses the numpy fake-4bit teaching stub instead."
    )


def fake_quant_nf4_like(W: np.ndarray, *, n_levels: int = 16) -> np.ndarray:
    """
    Per-tensor absmax fake quant to ``n_levels`` bins (NF4-shaped teaching stub).

    Not real NF4 / bitsandbytes — shows the memory idea: store base in low
    precision, keep adapters float.
    """
    arr = np.asarray(W, dtype=np.float64)
    absmax = float(np.max(np.abs(arr))) or 1.0
    # Symmetric uniform bins as a coarse stand-in for NF4 codebook
    qmax = n_levels // 2
    q = np.round(arr / absmax * qmax)
    q = np.clip(q, -qmax, qmax - (0 if n_levels % 2 == 0 else 0))
    return (q / qmax) * absmax


@dataclass
class QLoRAReport:
    available: bool
    used_bnb: bool
    used_fake4bit: bool
    message: str
    base_nbytes_fp64: int
    base_nbytes_q4_est: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "used_bnb": self.used_bnb,
            "used_fake4bit": self.used_fake4bit,
            "message": self.message,
            "base_nbytes_fp64": self.base_nbytes_fp64,
            "base_nbytes_q4_est": self.base_nbytes_q4_est,
        }


def apply_fake4bit_to_base(W: np.ndarray) -> tuple[np.ndarray, QLoRAReport]:
    """Return fake-4bit frozen base + memory story report (always offline-safe)."""
    Wq = fake_quant_nf4_like(W)
    nbytes_fp = int(W.size * 8)
    nbytes_q4 = int(W.size * 0.5)  # 4-bit estimate
    return Wq, QLoRAReport(
        available=qlora_available(),
        used_bnb=False,
        used_fake4bit=True,
        message=(
            "numpy fake-4bit teaching stub on frozen base "
            + ("(bnb also installed)" if qlora_available() else "(bnb absent — skipped real QLoRA)")
        ),
        base_nbytes_fp64=nbytes_fp,
        base_nbytes_q4_est=nbytes_q4,
    )
