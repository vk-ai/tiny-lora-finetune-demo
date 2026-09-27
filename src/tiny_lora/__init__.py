"""Tiny LoRA fine-tune demo — OSS / learning only (not employer production)."""

from .config import load_config
from .dora import DoRALinear
from .eval import compare_adapters, evaluate, run_before_after
from .lora import LoRALinear, MergedLinear
from .model import TinyClassifier
from .qlora import qlora_available
from .train import train_lora

__all__ = [
    "LoRALinear",
    "DoRALinear",
    "MergedLinear",
    "TinyClassifier",
    "load_config",
    "evaluate",
    "run_before_after",
    "compare_adapters",
    "train_lora",
    "qlora_available",
]

__version__ = "0.1.0"
