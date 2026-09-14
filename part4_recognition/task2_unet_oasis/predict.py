"""UNet inference — this must be run live on a test MRI during the demo.

    python -m part4_recognition.task2_unet_oasis.predict

The lab is explicit: you have to run inference on a test set at the demo and
visualise segmentations that justify the reported DSC.
"""

from __future__ import annotations

import torch

from common import get_device, save_fig


@torch.no_grad()
def evaluate_test_set(model, loader, device):
    """Return the per-class DSC over the whole test set, and print a table."""
    raise NotImplementedError


@torch.no_grad()
def plot_segmentations(model, loader, device, n: int = 4):
    """Three columns: input MRI, ground-truth mask, predicted mask."""
    raise NotImplementedError


def main() -> None:
    device = get_device()
    raise NotImplementedError


if __name__ == "__main__":
    main()
