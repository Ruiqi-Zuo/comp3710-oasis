"""Data loading for the OASIS VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from part4_recognition.oasis import OASIS_ROOT, OASISSlices

BATCH_SIZE = 64
IMAGE_SIZE = 128
NUM_WORKERS = 4  # an A100 node on the comp3710 partition has 8 CPUs


class Resize:
    """Downsample a ``(1, H, W)`` image tensor to ``size`` x ``size``.

    Antialiased bilinear: plain bilinear sampling at a factor of two skips every
    other pixel and aliases fine structure such as the cortical folds. The
    filter has non-negative weights, so the output stays inside [0, 1]; the
    clamp only guards against floating-point round-off, which a BCE target
    must not exceed.

    A class rather than a lambda so that DataLoader workers can pickle it.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == (self.size, self.size):
            return image
        resized = F.interpolate(
            image.unsqueeze(0), size=(self.size, self.size),
            mode="bilinear", align_corners=False, antialias=True,
        )
        return resized.squeeze(0).clamp_(0.0, 1.0)


def get_datasets(image_size: int = IMAGE_SIZE, root: str = OASIS_ROOT, limit: int | None = None):
    """``(train, validate, test)`` unlabelled OASISSlices datasets."""
    transform = Resize(image_size)
    return tuple(
        OASISSlices(split, with_labels=False, transform=transform, root=root, limit=limit)
        for split in ("train", "validate", "test")
    )


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    root: str = OASIS_ROOT,
    limit: int | None = None,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, validate_loader, test_loader)`` of unlabelled OASIS slices.

    Resize to IMAGE_SIZE and keep pixels in [0, 1] — a Bernoulli/BCE
    reconstruction term assumes that range, so a sigmoid output and unnormalised
    inputs must agree.

    Three loaders rather than the scaffold's two: the validation split tracks the
    ELBO and the active latent units during training, and the test split is
    kept for the final reconstructions and manifold figures.
    """
    train, validate, test = get_datasets(image_size, root, limit)
    options = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return (
        DataLoader(train, shuffle=True, drop_last=True, **options),
        DataLoader(validate, shuffle=False, **options),
        DataLoader(test, shuffle=False, **options),
    )
