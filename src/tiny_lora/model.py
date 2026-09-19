"""Tiny two-layer classifier: Linear -> ReLU -> LoRALinear (classifier head)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lora import LoRALinear, ScaleMode


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TinyClassifier:
    """Frozen hidden layer + LoRA classification head."""

    W1: np.ndarray  # (hidden, in) frozen
    b1: np.ndarray  # (hidden,) frozen
    head: LoRALinear

    @classmethod
    def create(
        cls,
        in_features: int,
        hidden_dim: int,
        n_classes: int,
        rank: int,
        alpha: float,
        rng: np.random.Generator,
        *,
        scale_mode: ScaleMode = "classic",
    ) -> "TinyClassifier":
        W1 = rng.normal(0.0, 0.5 / np.sqrt(in_features), size=(hidden_dim, in_features))
        b1 = np.zeros(hidden_dim, dtype=np.float64)
        head = LoRALinear.create(
            in_features=hidden_dim,
            out_features=n_classes,
            rank=rank,
            alpha=alpha,
            rng=rng,
            scale=0.5,
            scale_mode=scale_mode,
        )
        return cls(W1=W1.astype(np.float64), b1=b1, head=head)

    def hidden(self, x: np.ndarray) -> np.ndarray:
        return _relu(x @ self.W1.T + self.b1)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.head.forward(self.hidden(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(axis=1)

    def n_trainable(self) -> int:
        return self.head.n_trainable()

    def n_frozen(self) -> int:
        return int(self.W1.size + self.b1.size + self.head.n_frozen())
