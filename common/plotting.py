"""Matplotlib helpers used across the lab."""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)


def save_fig(fig: plt.Figure, name: str, subdir: str = "", dpi: int = 150) -> str:
    """Save ``fig`` under ``outputs/<subdir>/<name>.png`` and return the path."""
    directory = os.path.join(OUTPUT_DIR, subdir) if subdir else OUTPUT_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_gallery(
    images: np.ndarray,
    titles: Sequence[str],
    h: int,
    w: int,
    n_row: int = 3,
    n_col: int = 4,
) -> plt.Figure:
    """Plot a grid of greyscale portraits (used for eigenfaces and samples)."""
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(min(n_row * n_col, len(images))):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    return fig


def plot_curves(history: dict, title: str = "Training history") -> plt.Figure:
    """Plot every sequence in ``history`` (e.g. loss/accuracy) against epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=label)
    ax.set_xlabel("Epoch")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return fig
