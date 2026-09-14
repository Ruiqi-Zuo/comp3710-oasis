"""Data loading for the OASIS VAE."""

from __future__ import annotations

from torch.utils.data import DataLoader

BATCH_SIZE = 64
IMAGE_SIZE = 128


def get_dataloaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, test_loader)`` of unlabelled OASIS slices.

    Resize to IMAGE_SIZE and keep pixels in [0, 1] — a Bernoulli/BCE
    reconstruction term assumes that range, so a sigmoid output and unnormalised
    inputs must agree.
    """
    # TODO: wrap part4_recognition.oasis.OASISSlices(with_labels=False).
    raise NotImplementedError
