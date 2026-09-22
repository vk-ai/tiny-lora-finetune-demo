"""Tiny two-layer classifier: Linear -> ReLU -> LoRALinear (classifier head)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lora import LoRALinear, MergedLinear, ScaleMode


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TinyClassifier:
    """Frozen hidden layer + LoRA (or merged) classification head."""

    W1: np.ndarray  # (hidden, in) frozen
    b1: np.ndarray  # (hidden,) frozen
    head: LoRALinear | MergedLinear

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

    def merge_and_unload(self) -> "TinyClassifier":
        """Return a new classifier with LoRA folded into the head weights.

        **Must assign:** ``model = model.merge_and_unload()`` — does not mutate
        this instance (PEFT assign-return lesson / peft#2032). Only valid while
        ``head`` is still a :class:`LoRALinear`.
        """
        if not isinstance(self.head, LoRALinear):
            raise TypeError(
                "merge_and_unload requires a LoRALinear head "
                f"(got {type(self.head).__name__}; already merged?)"
            )
        merged_head = self.head.merge_and_unload()
        return TinyClassifier(
            W1=np.asarray(self.W1, dtype=np.float64).copy(),
            b1=np.asarray(self.b1, dtype=np.float64).copy(),
            head=merged_head,
        )

    @property
    def mode(self) -> str:
        """``adapter`` while LoRA A/B exist; ``merged`` after merge_and_unload."""
        return "merged" if isinstance(self.head, MergedLinear) else "adapter"

    def n_trainable(self) -> int:
        return self.head.n_trainable()

    def n_frozen(self) -> int:
        return int(self.W1.size + self.b1.size + self.head.n_frozen())
