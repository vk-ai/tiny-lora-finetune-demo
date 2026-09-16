"""Tiny synthetic multi-class dataset (no downloads)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


def make_synthetic(
    n_samples: int,
    n_features: int,
    n_classes: int,
    noise: float,
    rng: np.random.Generator,
    centroids: np.ndarray | None = None,
) -> tuple[Dataset, np.ndarray]:
    """Gaussian blobs around class centroids — sklearn-sized, CPU-instant.

    Returns (dataset, centroids). Pass the same centroids for train and test
    so both splits share one decision problem.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if centroids is None:
        centroids = rng.normal(0.0, 1.0, size=(n_classes, n_features))
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    else:
        centroids = np.asarray(centroids, dtype=np.float64)
        if centroids.shape != (n_classes, n_features):
            raise ValueError(
                f"centroids shape {centroids.shape} != ({n_classes}, {n_features})"
            )
    y = rng.integers(0, n_classes, size=n_samples)
    X = centroids[y] + rng.normal(0.0, noise, size=(n_samples, n_features))
    return Dataset(X=X.astype(np.float64), y=y.astype(np.int64)), centroids


def train_test_split(
    n_train: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    noise: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    train, centroids = make_synthetic(n_train, n_features, n_classes, noise, rng)
    test, _ = make_synthetic(
        n_test, n_features, n_classes, noise, rng, centroids=centroids
    )
    return train, test
