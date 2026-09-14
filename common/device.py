"""Device selection helpers shared by every part of the lab."""

from __future__ import annotations

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available device.

    Args:
        prefer_gpu: If False, always return CPU (useful for baseline timings).
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    """Human-readable description, handy to print at the top of a demo run."""
    device = device or get_device()
    if device.type == "cuda":
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / 1024**3
        return f"cuda:{index} — {name} ({total_gb:.1f} GiB)"
    return "cpu"
