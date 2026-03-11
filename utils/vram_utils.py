"""
VRAM monitoring helpers for single-GPU H100 training.
"""

import torch
import gc
from typing import Optional


def vram_stats(device: int = 0) -> dict:
    """Return current VRAM usage stats in GB."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved  = torch.cuda.memory_reserved(device) / 1e9
    total     = torch.cuda.get_device_properties(device).total_memory / 1e9
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb":  round(reserved,  2),
        "total_gb":     round(total,     2),
        "free_gb":      round(total - reserved, 2),
    }


def print_vram(label: str = "", device: int = 0):
    stats = vram_stats(device)
    if not stats:
        return
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}VRAM — "
        f"allocated: {stats['allocated_gb']:.1f}GB / "
        f"reserved: {stats['reserved_gb']:.1f}GB / "
        f"total: {stats['total_gb']:.1f}GB"
    )


def free_vram():
    """Force garbage collection and empty CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class VRAMGuard:
    """
    Context manager that prints VRAM before/after a block.

    Usage:
        with VRAMGuard("loading model"):
            model = AutoModelForCausalLM.from_pretrained(...)
    """
    def __init__(self, label: str = "", device: int = 0):
        self.label  = label
        self.device = device

    def __enter__(self):
        print_vram(f"BEFORE {self.label}", self.device)
        return self

    def __exit__(self, *args):
        print_vram(f"AFTER  {self.label}", self.device)


def assert_vram_available(required_gb: float, device: int = 0):
    """Raise RuntimeError if free VRAM is below required_gb."""
    stats = vram_stats(device)
    if stats and stats["free_gb"] < required_gb:
        raise RuntimeError(
            f"Insufficient VRAM: need {required_gb:.1f}GB, "
            f"have {stats['free_gb']:.1f}GB free."
        )