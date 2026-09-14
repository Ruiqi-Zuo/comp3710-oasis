"""Unlabelled image loading for GAN training.

The lab suggests starting on MNIST or CelebA to get the training loop stable
before switching to OASIS — only OASIS results earn full marks, but a working
MNIST run is worth keeping as evidence that the implementation is sound.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

BATCH_SIZE = 128
IMAGE_SIZE = 64  # start small; scale up once training is stable


def get_dataloader(dataset: str = "oasis", batch_size: int = BATCH_SIZE) -> DataLoader:
    """Return a loader of images normalised to [-1, 1] to match a tanh generator."""
    # TODO: dispatch on `dataset` so the same training loop serves mnist/oasis.
    raise NotImplementedError
