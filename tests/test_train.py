"""Train loop improves loss / accuracy on synthetic data."""

from __future__ import annotations

import numpy as np

from tiny_lora.config import load_config
from tiny_lora.data import train_test_split
from tiny_lora.eval import evaluate
from tiny_lora.model import TinyClassifier
from tiny_lora.train import train_lora
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_reduces_loss_and_raises_accuracy():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    # Slightly smaller for speed; still deterministic.
    cfg = {
        **cfg,
        "data": {**cfg["data"], "n_train": 128, "n_test": 48, "noise": 0.1},
        "train": {**cfg["train"], "epochs": 25, "lr": 0.08, "batch_size": 32},
        "lora": {**cfg["lora"], "rank": 4, "alpha": 8.0},
    }
    train, test = train_test_split(
        n_train=cfg["data"]["n_train"],
        n_test=cfg["data"]["n_test"],
        n_features=cfg["data"]["n_features"],
        n_classes=cfg["data"]["n_classes"],
        noise=cfg["data"]["noise"],
        seed=cfg["seed"],
    )
    rng = np.random.default_rng(cfg["seed"])
    model = TinyClassifier.create(
        in_features=cfg["data"]["n_features"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_classes=cfg["data"]["n_classes"],
        rank=cfg["lora"]["rank"],
        alpha=cfg["lora"]["alpha"],
        rng=rng,
    )
    before = evaluate(model, test)
    losses = train_lora(
        model,
        train,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        batch_size=cfg["train"]["batch_size"],
        l2=cfg["train"]["l2"],
        seed=cfg["seed"] + 1,
    )
    after = evaluate(model, test)

    assert losses[-1] < losses[0]
    assert after.loss < before.loss
    assert after.accuracy > before.accuracy
    assert after.accuracy >= 0.7  # separable toy blobs
