"""Tiny LoRA fine-tune demo — OSS / learning only (not employer production)."""

from .config import load_config
from .eval import evaluate, run_before_after
from .lora import LoRALinear, MergedLinear
from .model import TinyClassifier
from .train import train_lora

__all__ = [
    "LoRALinear",
    "MergedLinear",
    "TinyClassifier",
    "load_config",
    "evaluate",
    "run_before_after",
    "train_lora",
]

__version__ = "0.1.0"
