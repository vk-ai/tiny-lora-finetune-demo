"""Multiple named adapters + weighted merging (numpy teaching stub).

A :class:`MultiAdapterHead` owns one frozen base ``W, b`` and a dict of named
LoRA / DoRA adapters that *share* that base (peft ``add_adapter`` /
``set_adapter`` analogue). :meth:`MultiAdapterHead.add_weighted_adapter`
combines adapters into a new named adapter, peft ``add_weighted_adapter``-style:

- ``linear``: task arithmetic in weight space, ΔW = Σ wᵢ·ΔWᵢ. **Negative
  weights are allowed** ("subtract a skill", peft#2796). Each ΔWᵢ uses *its own*
  scaling (classic α/r or rsLoRA α/√r). The result is re-factorized with SVD at
  rank ``min(Σ rᵢ, out, in)`` (exact), or at ``svd_rank`` (lossy; the relative
  reconstruction error is reported).
- ``cat``: concatenate the factors, A = [A₁; A₂; …], B = [w₁s₁B₁ | w₂s₂B₂ | …].
  It is exact with no SVD, and the rank grows to Σ rᵢ.
- ``ties``: TIES-lite (arXiv 2306.01708) on τᵢ = wᵢ·ΔWᵢ. Keep the top
  ``density`` fraction of |τᵢ| per adapter, elect a per-entry sign from Σ τᵢ,
  then take the disjoint mean over the adapters that agree with that sign.

The merged adapter always gets **effective scaling 1** (its factors carry the
scale). This is the fix discussed for rsLoRA in peft#3761/#3449: a combined
adapter must not re-derive α/√r from the new rank. DoRA carries a learned
magnitude ``m``. This toy defines its combination as magnitude task arithmetic,
m = m₀ + Σ wᵢ·(mᵢ − m₀) with m₀ = ‖W‖_row (the DoRA init). That choice is
exact for "one adapter at weight 1.0". Upstream plans to *raise* for DoRA
because no canonical rule exists, and ``dora_policy="raise"`` does that here.

Invariant (tested): merging a single adapter at weight 1.0 reproduces its
logits for classic LoRA, rsLoRA, and DoRA, with every combination type.

Teaching stub, not Hugging Face peft or mergekit. Sources:
https://github.com/huggingface/peft/issues/3761 ·
https://github.com/huggingface/peft/issues/2796 ·
https://huggingface.co/docs/peft/v0.18.0/developer_guides/model_merging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .dora import DoRALinear
from .lora import LoRALinear, MergedLinear, ScaleMode

Combination = Literal["linear", "cat", "ties"]
Adapter = LoRALinear | DoRALinear
_VALID_COMBINATIONS = ("linear", "cat", "ties")


def _unit_scale_alpha(rank: int, mode: ScaleMode) -> float:
    """Alpha giving scaling exactly 1 for ``rank`` under ``mode``."""
    return float(rank) if mode == "classic" else float(np.sqrt(rank))


def ties_merge(task_vectors: Sequence[np.ndarray], density: float) -> np.ndarray:
    """TIES-lite: trim (top-``density`` by |τ|), elect sign (Σ τ), disjoint mean."""
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    stacked = np.stack([np.asarray(t, dtype=np.float64) for t in task_vectors])
    trimmed = np.zeros_like(stacked)
    for i, tau in enumerate(stacked):
        k = int(np.ceil(density * tau.size))
        if k >= tau.size:
            trimmed[i] = tau
            continue
        flat = np.abs(tau).ravel()
        keep = np.argpartition(flat, -k)[-k:]
        mask = np.zeros(tau.size, dtype=bool)
        mask[keep] = True
        trimmed[i] = np.where(mask.reshape(tau.shape), tau, 0.0)
    elected = np.sign(trimmed.sum(axis=0))
    agree = (np.sign(trimmed) == elected[None]) & (trimmed != 0.0)
    count = agree.sum(axis=0)
    total = np.where(agree, trimmed, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


@dataclass
class MergeInfo:
    """What :meth:`MultiAdapterHead.add_weighted_adapter` did (for reports)."""

    combination_type: str
    adapters: list[str]
    weights: list[float]
    rank: int
    density: float | None = None
    svd_rel_error: float = 0.0
    dora_magnitude: str | None = None


@dataclass
class MultiAdapterHead:
    """Frozen linear base + many named LoRA/DoRA adapters (one active at a time)."""

    W: np.ndarray  # (out, in) frozen, shared by every adapter
    b: np.ndarray  # (out,) frozen
    adapters: dict[str, Adapter] = field(default_factory=dict)
    active_adapter: str | None = None
    merge_info: dict[str, MergeInfo] = field(default_factory=dict)

    @classmethod
    def from_base(cls, W: np.ndarray, b: np.ndarray) -> "MultiAdapterHead":
        return cls(W=np.asarray(W, dtype=np.float64), b=np.asarray(b, dtype=np.float64))

    @classmethod
    def from_head(cls, head: Adapter, name: str = "default") -> "MultiAdapterHead":
        """Wrap an existing single-adapter head; it becomes adapter ``name``."""
        multi = cls.from_base(head.W, head.b)
        head.W, head.b = multi.W, multi.b  # share the frozen base
        multi.adapters[name] = head
        multi.active_adapter = name
        return multi

    @property
    def in_features(self) -> int:
        return int(self.W.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.W.shape[0])

    # ----------------------------------------------------------- registry
    def add_adapter(
        self,
        name: str,
        *,
        rank: int,
        alpha: float,
        rng: np.random.Generator,
        scale_mode: ScaleMode = "classic",
        use_dora: bool = False,
    ) -> Adapter:
        """Create a fresh adapter (B = 0 → no-op) sharing the frozen base."""
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} already exists")
        cls = DoRALinear if use_dora else LoRALinear
        adapter = cls.create(
            self.in_features, self.out_features, rank, alpha, rng, scale_mode=scale_mode
        )
        adapter.W, adapter.b = self.W, self.b
        if isinstance(adapter, DoRALinear):
            adapter.m = self.base_magnitude()
        self.adapters[name] = adapter
        if self.active_adapter is None:
            self.active_adapter = name
        return adapter

    def set_adapter(self, name: str | None) -> None:
        """Activate adapter ``name`` (``None`` → base only)."""
        if name is not None and name not in self.adapters:
            raise KeyError(f"unknown adapter {name!r}; have {sorted(self.adapters)}")
        self.active_adapter = name

    def delete_adapter(self, name: str) -> None:
        self.adapters.pop(name)
        self.merge_info.pop(name, None)
        if self.active_adapter == name:
            self.active_adapter = None

    def base_magnitude(self) -> np.ndarray:
        return np.maximum(np.linalg.norm(self.W, axis=1), 1e-12)

    # ------------------------------------------------------------ forward
    def forward(self, x: np.ndarray, adapter: str | None = "__active__") -> np.ndarray:
        name = self.active_adapter if adapter == "__active__" else adapter
        if name is None:
            return x @ self.W.T + self.b
        return self.adapters[name].forward(x)

    def delta_W(self, name: str) -> np.ndarray:
        """Weight-space update of adapter ``name`` using its own scaling."""
        return self.adapters[name].delta_W()

    def effective_W(self, name: str | None) -> np.ndarray:
        if name is None:
            return self.W.copy()
        return self.adapters[name].effective_W()

    def merge_and_unload(self, name: str | None = None) -> MergedLinear:
        """Fold adapter ``name`` (default: active) into a plain MergedLinear."""
        name = self.active_adapter if name is None else name
        if name is None:
            return MergedLinear(self.in_features, self.out_features, self.W.copy(), self.b.copy())
        return self.adapters[name].merge_and_unload()

    def n_trainable(self) -> int:
        return sum(a.n_trainable() for a in self.adapters.values())

    def n_frozen(self) -> int:
        return int(self.W.size + self.b.size)

    # -------------------------------------------------------------- merge
    def add_weighted_adapter(
        self,
        adapters: Sequence[str],
        weights: Sequence[float],
        adapter_name: str,
        *,
        combination_type: Combination = "linear",
        density: float = 0.5,
        svd_rank: int | None = None,
        dora_policy: Literal["magnitude_arithmetic", "raise"] = "magnitude_arithmetic",
    ) -> Adapter:
        """Combine named adapters into a new adapter ``adapter_name`` (not activated)."""
        adapters = list(adapters)
        weights = [float(w) for w in weights]
        if combination_type not in _VALID_COMBINATIONS:
            raise ValueError(
                f"combination_type must be one of {_VALID_COMBINATIONS}, got {combination_type!r}"
            )
        if not adapters or len(adapters) != len(weights):
            raise ValueError("need one weight per adapter (and at least one adapter)")
        if adapter_name in self.adapters:
            raise ValueError(f"adapter {adapter_name!r} already exists")
        missing = [a for a in adapters if a not in self.adapters]
        if missing:
            raise KeyError(f"unknown adapters: {missing}")
        srcs = [self.adapters[a] for a in adapters]
        kinds = {type(s) for s in srcs}
        if len(kinds) > 1:
            raise ValueError("cannot mix LoRA and DoRA adapters in one merge")
        is_dora = kinds == {DoRALinear}
        if is_dora and dora_policy == "raise":
            raise NotImplementedError(
                "DoRA magnitude has no canonical weighted combination "
                "(peft#3761 plans to raise); use dora_policy='magnitude_arithmetic'"
            )
        mode: ScaleMode = srcs[0].scale_mode
        total_rank = sum(s.rank for s in srcs)
        info = MergeInfo(combination_type, adapters, weights, rank=0)

        if combination_type == "cat":
            A = np.concatenate([s.A for s in srcs], axis=0)
            B = np.concatenate([w * s.scaling * s.B for s, w in zip(srcs, weights)], axis=1)
            rank = total_rank
        else:
            if combination_type == "linear":
                delta = sum(w * s.delta_W() for s, w in zip(srcs, weights))
            else:
                delta = ties_merge([w * s.delta_W() for s, w in zip(srcs, weights)], density)
                info.density = float(density)
            full = min(total_rank, self.out_features, self.in_features)
            rank = full if svd_rank is None else int(svd_rank)
            if rank < 1:
                raise ValueError("svd_rank must be >= 1")
            U, S, Vt = np.linalg.svd(delta, full_matrices=False)
            rank = min(rank, len(S))
            B = U[:, :rank] * S[:rank]
            A = Vt[:rank]
            approx = B @ A
            denom = float(np.linalg.norm(delta)) or 1.0
            info.svd_rel_error = float(np.linalg.norm(delta - approx) / denom)
        info.rank = int(rank)

        alpha = _unit_scale_alpha(rank, mode)  # scaling == 1: factors carry scale
        common = dict(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=int(rank),
            alpha=alpha,
            W=self.W,
            b=self.b,
            A=np.ascontiguousarray(A, dtype=np.float64),
            B=np.ascontiguousarray(B, dtype=np.float64),
            scale_mode=mode,
        )
        merged: Adapter
        if is_dora:
            m0 = self.base_magnitude()
            m = m0 + sum(w * (s.m - m0) for s, w in zip(srcs, weights))
            merged = DoRALinear(m=np.asarray(m, dtype=np.float64), **common)
            info.dora_magnitude = "m0 + Σ wᵢ (mᵢ − m0)"
        else:
            merged = LoRALinear(**common)
        assert abs(merged.scaling - 1.0) < 1e-12
        self.adapters[adapter_name] = merged
        self.merge_info[adapter_name] = info
        return merged


__all__ = ["MultiAdapterHead", "MergeInfo", "ties_merge"]
