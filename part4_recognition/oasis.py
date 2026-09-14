"""Shared OASIS dataset access for Tasks 1-3.

The Preprocessed OASIS data lives on Rangpur at::

    /home/groups/comp3710/OASIS/

with pre-split directories of 2D PNG slices, e.g. ``keras_png_slices_train``,
``keras_png_slices_validate``, ``keras_png_slices_test`` and the matching
``_seg_`` label directories used by the UNet task.

Check the actual directory names on the cluster before relying on the constants
below, and never copy the dataset into this repository — it stays git-ignored.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

OASIS_ROOT = os.environ.get("OASIS_ROOT", "/home/groups/comp3710/OASIS")

# Segmentation label values in the preprocessed OASIS masks:
# background, CSF, grey matter, white matter.
N_SEG_CLASSES = 4


class OASISSlices(Dataset):
    """2D OASIS slices, optionally paired with their segmentation masks.

    Args:
        split: ``"train"``, ``"validate"`` or ``"test"``.
        with_labels: If True, also return the segmentation mask (Task 2).
        transform: Optional callable applied to the image tensor.
    """

    def __init__(self, split: str = "train", with_labels: bool = False, transform=None) -> None:
        # TODO: glob the PNG paths for the split, sort them so image and mask
        # lists line up, and store them. Do not load everything into memory at
        # construction time unless you have measured that it fits.
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        """Return ``image`` or ``(image, mask)``.

        Images: float32 in [0, 1], shape (1, H, W).
        Masks:  int64 class indices, shape (H, W) — remap the raw PNG greyscale
                values (0/85/170/255) to 0..3 before use, or the one-hot
                encoding required by Task 2 will be wrong.
        """
        raise NotImplementedError
