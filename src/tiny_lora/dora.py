"""DoRA teaching stub: weight-decomposed LoRA (magnitude + normalized direction).

Paper / peft: W' = m * (W0 + scale·B@A) / ||W0 + scale·B@A||
(row-wise norm over in_features; m ∈ R^{out} learnable).

Production path is ``LoraConfig(use_dora=True)`` in Hugging Face peft —
this module is a numpy toy only.
See: https://huggingface.co/docs/peft/main/en/package_reference/lora_variant_dora
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lora import MergedLinear, ScaleMode, lora_scale


@dataclass
class DoRALinear:
    """Linear with frozen W, trainable LoRA A/B, and trainable magnitude m."""

    in_features: int
    out_features: int
    rank: int
    alpha: float
    W: np.ndarray  # (out, in) frozen
    b: np.ndarray  # (out,) frozen
    A: np.ndarray  # (rank, in) trainable
    B: np.ndarray  # (out, rank) trainable
    m: np.ndarray  # (out,) trainable magnitude
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
    ) -> "DoRALinear":
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if scale_mode not in ("classic", "rslora"):
            raise ValueError(f"unknown scaling mode: {scale_mode!r}")
        W = rng.normal(0.0, scale / np.sqrt(in_features), size=(out_features, in_features))
        W = W.astype(np.float64)
        b = np.zeros(out_features, dtype=np.float64)
        A = rng.normal(0.0, 1.0 / np.sqrt(in_features), size=(rank, in_features)).astype(
            np.float64
        )
        B = np.zeros((out_features, rank), dtype=np.float64)
        # Init magnitude to ||W||_row so adapter starts ≈ base when B=0
        row_norm = np.linalg.norm(W, axis=1)
        m = np.maximum(row_norm, 1e-12)
        return cls(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=float(alpha),
            W=W,
            b=b,
            A=A,
            B=B,
            m=m.astype(np.float64),
            scale_mode=scale_mode,
        )

    @property
    def scaling(self) -> float:
        return lora_scale(self.alpha, self.rank, self.scale_mode)

    def delta_W(self) -> np.ndarray:
        return self.scaling * (self.B @ self.A)

    def direction_and_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (direction (out,in), row_norm (out,)) of W+ΔW."""
        W_prime = self.W + self.delta_W()
        row_norm = np.linalg.norm(W_prime, axis=1)
        row_norm = np.maximum(row_norm, 1e-12)
        direction = W_prime / row_norm[:, None]
        return direction, row_norm

    def effective_W(self) -> np.ndarray:
        direction, _ = self.direction_and_norm()
        return self.m[:, None] * direction

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, in) -> (batch, out)."""
        return x @ self.effective_W().T + self.b

    def merge_and_unload(self) -> MergedLinear:
        """Fold DoRA into a plain MergedLinear (assign-return; not in-place)."""
        return MergedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            W=np.asarray(self.effective_W(), dtype=np.float64).copy(),
            b=np.asarray(self.b, dtype=np.float64).copy(),
            merged_from=self.scale_mode,
            merged_rank=self.rank,
            merged_alpha=self.alpha,
        )

    def trainable_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.A, self.B, self.m

    def n_trainable(self) -> int:
        return int(self.A.size + self.B.size + self.m.size)

    def n_frozen(self) -> int:
        return int(self.W.size + self.b.size)
