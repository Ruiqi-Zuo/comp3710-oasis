"""Shared utilities for the COMP3710 Lab 2 project."""

from .device import describe_device, get_device
from .plotting import plot_curves, plot_gallery, save_fig
from .seeding import set_seed
from .timing import Timer, benchmark

__all__ = [
    "describe_device",
    "get_device",
    "plot_curves",
    "plot_gallery",
    "save_fig",
    "set_seed",
    "Timer",
    "benchmark",
]
