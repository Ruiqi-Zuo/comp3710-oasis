"""Paired image/mask loading for OASIS segmentation."""

from __future__ import annotations

from torch.utils.data import DataLoader

BATCH_SIZE = 16
IMAGE_SIZE = 256


def get_dataloaders(batch_size: int = BATCH_SIZE):
    """Return ``(train_loader, validate_loader, test_loader)`` of (image, mask) pairs.

    Two things to get right or the DSC will be quietly wrong:

    * **Label remapping.** The mask PNGs store classes as greyscale levels
      (0/85/170/255). They must become contiguous class indices 0..3 before
      one-hot encoding or CrossEntropyLoss.
    * **Matched augmentation.** Any spatial transform applied to the image must
      be applied identically to its mask, and masks must be resized with
      *nearest-neighbour* interpolation — bilinear invents label values that
      do not exist.
    """
    # TODO: wrap part4_recognition.oasis.OASISSlices(with_labels=True).
    raise NotImplementedError
