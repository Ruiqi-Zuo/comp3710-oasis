"""Paired image/mask loading for OASIS segmentation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from part4_recognition.oasis import OASIS_ROOT, OASISSlices

BATCH_SIZE = 16
IMAGE_SIZE = 256
NUM_WORKERS = 4  # an A100 node on the comp3710 partition has 8 CPUs


def get_datasets(root: str = OASIS_ROOT, limit: int | None = None):
    """``(train, validate, test)`` datasets of (image, mask) pairs at native 256x256."""
    return tuple(
        OASISSlices(split, with_labels=True, root=root, limit=limit)
        for split in ("train", "validate", "test")
    )


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    root: str = OASIS_ROOT,
    limit: int | None = None,
    num_workers: int = NUM_WORKERS,
):
    """Return ``(train_loader, validate_loader, test_loader)`` of (image, mask) pairs.

    Two things to get right or the DSC will be quietly wrong:

    * **Label remapping.** The mask PNGs store classes as greyscale levels
      (0/85/170/255). They must become contiguous class indices 0..3 before
      one-hot encoding or CrossEntropyLoss.
    * **Matched augmentation.** Any spatial transform applied to the image must
      be applied identically to its mask, and masks must be resized with
      *nearest-neighbour* interpolation — bilinear invents label values that
      do not exist.

    Here the first is handled by ``OASISSlices``, which maps grey levels through
    a lookup table that rejects anything unexpected. The resizing problem is
    avoided outright: the slices are used at their native 256x256, which the
    UNet's four poolings divide evenly. Augmentation is applied on the GPU by
    :func:`random_hflip`, to image and mask in a single indexing operation.
    """
    train, validate, test = get_datasets(root, limit)
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


def random_hflip(images: torch.Tensor, masks: torch.Tensor, p: float = 0.5):
    """Mirror a random half of the batch left-right — images and masks together.

    One boolean selection drives both tensors, so an image can never be flipped
    without its mask. A left-right mirror is a plausible brain (the hemispheres
    are near-symmetric) and leaves every tissue label meaning the same thing.
    Batched on the GPU, it costs nothing next to the forward pass.
    """
    flip = torch.rand(images.size(0), device=images.device) < p
    if flip.any():
        images = images.clone()
        masks = masks.clone()
        images[flip] = images[flip].flip(-1)
        masks[flip] = masks[flip].flip(-1)
    return images, masks
