"""Toy LoRA adapter math (CPU / numpy).

For a frozen base weight W (out_features x in_features):

    classic:  scale = alpha / rank
    rslora:   scale = alpha / sqrt(rank)   # Rank-Stabilized LoRA

    y = x @ W.T + scale * x @ A.T @ B.T + b

where A is (rank x in_features), B is (out_features x rank).
A ~ N(0, 1/sqrt(in)), B = 0 so the adapter starts as a no-op.
Only A and B are trainable; W and b stay frozen.

This is a teaching stub — not Hugging Face peft / transformers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

ScaleMode = Literal["classic", "rslora"]


def lora_scale(alpha: float, rank: int, mode: ScaleMode = "classic") -> float:
    """Return LoRA multiplier: α/r (classic) or α/√r (rsLoRA)."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if mode == "classic":
        return alpha / rank
    if mode == "rslora":
        return alpha / math.sqrt(rank)
    raise ValueError(f"unknown scaling mode: {mode!r} (expected classic|rslora)")


@dataclass
class LoRALinear:
    """Linear layer with a frozen base weight and a trainable LoRA adapter."""

    in_features: int
    out_features: int
    rank: int
    alpha: float
    W: np.ndarray  # (out, in) frozen
    b: np.ndarray  # (out,) frozen
    A: np.ndarray  # (rank, in) trainable
    B: np.ndarray  # (out, rank) trainable
    scale_mode: ScaleMode = "classic"

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        rng: np.random.Generator,
        *,
        scale: float = 0.5,
        scale_mode: ScaleMode = "classic",
    ) -> "LoRALinear":
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if scale_mode not in ("classic", "rslora"):
            raise ValueError(f"unknown scaling mode: {scale_mode!r}")
        W = rng.normal(0.0, scale / np.sqrt(in_features), size=(out_features, in_features))
        b = np.zeros(out_features, dtype=np.float64)
        A = rng.normal(0.0, 1.0 / np.sqrt(in_features), size=(rank, in_features))
        B = np.zeros((out_features, rank), dtype=np.float64)
        return cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=float(alpha),
            W=W.astype(np.float64),
            b=b,
            A=A.astype(np.float64),
            B=B,
            scale_mode=scale_mode,
        )

    @property
    def scaling(self) -> float:
        return lora_scale(self.alpha, self.rank, self.scale_mode)

    def delta_W(self) -> np.ndarray:
        """Low-rank update ΔW = scale * B @ A  (out x in)."""
        return self.scaling * (self.B @ self.A)

    def effective_W(self) -> np.ndarray:
        return self.W + self.delta_W()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in) -> (batch, out)."""
        base = x @ self.W.T + self.b
        # x @ A.T -> (batch, rank); then @ B.T -> (batch, out)
        lora = self.scaling * ((x @ self.A.T) @ self.B.T)
        return base + lora

    def trainable_params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.A, self.B

    def n_trainable(self) -> int:
        return int(self.A.size + self.B.size)

    def n_frozen(self) -> int:
        return int(self.W.size + self.b.size)
