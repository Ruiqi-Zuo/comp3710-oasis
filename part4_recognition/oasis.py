"""Shared OASIS dataset access for Tasks 1-3.

The Preprocessed OASIS data lives on Rangpur at::

    /home/groups/comp3710/OASIS/

Surveyed layout (see the repository README and ``scripts/probe_oasis.py``)::

    keras_png_slices_{train,validate,test}/case_XXX_slice_Y.nii.png
    keras_png_slices_seg_{train,validate,test}/seg_XXX_slice_Y.nii.png

256 x 256 greyscale PNGs. Masks store four labels as the grey levels 0, 85, 170
and 255. The split is by subject, so no brain appears in more than one split.

Never copy the dataset into this repository — it stays git-ignored and is read
in place on the cluster.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

OASIS_ROOT = os.environ.get("OASIS_ROOT", "/home/groups/comp3710/OASIS")

# Segmentation label values in the preprocessed OASIS masks:
# background, CSF, grey matter, white matter.
N_SEG_CLASSES = 4
MASK_GREY_LEVELS = (0, 85, 170, 255)
CLASS_NAMES = ("background", "CSF", "grey matter", "white matter")

SPLITS = ("train", "validate", "test")

# Lookup table from raw grey level to class index. Every value that is not one
# of the four label levels maps to INVALID, so a corrupt or resampled mask is
# caught at load time instead of being silently folded into a real class — which
# is exactly what ``value // 85`` would do to a stray 84 or 86.
INVALID = 255
_LABEL_LUT = np.full(256, INVALID, dtype=np.uint8)
_LABEL_LUT[list(MASK_GREY_LEVELS)] = np.arange(N_SEG_CLASSES, dtype=np.uint8)


def image_dir(split: str, root: str = OASIS_ROOT) -> str:
    return os.path.join(root, f"keras_png_slices_{split}")


def mask_dir(split: str, root: str = OASIS_ROOT) -> str:
    return os.path.join(root, f"keras_png_slices_seg_{split}")


def mask_name_for(image_name: str) -> str:
    """``case_001_slice_0.nii.png`` -> ``seg_001_slice_0.nii.png``."""
    if not image_name.startswith("case_"):
        raise ValueError(f"unexpected image file name: {image_name!r}")
    return "seg_" + image_name[len("case_"):]


def natural_key(name: str) -> list:
    """Sort ``slice_2`` before ``slice_10``, so a subject's slices stay in order.

    Plain string sorting would interleave them (0, 1, 10, 11, ..., 2), which does
    not affect image/mask pairing — that is done by name — but it scrambles any
    figure that shows consecutive slices of one brain.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def masks_to_classes(raw: np.ndarray) -> np.ndarray:
    """Map grey levels 0/85/170/255 to class indices 0..3, rejecting anything else."""
    classes = _LABEL_LUT[raw]
    if (classes == INVALID).any():
        bad = np.unique(raw[classes == INVALID])
        raise ValueError(
            f"mask contains grey levels {bad.tolist()[:10]} outside {MASK_GREY_LEVELS}"
        )
    return classes


def _available_cpus() -> int:
    """CPUs this process may use — under SLURM, the allocation, not the node."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on Windows or macOS
        return os.cpu_count() or 1


def _read_png(path: str) -> np.ndarray:
    # np.array, not np.asarray: the array PIL exposes is read-only, and
    # torch.from_numpy warns about wrapping a non-writable buffer.
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.uint8)


class OASISSlices(Dataset):
    """2D OASIS slices, optionally paired with their segmentation masks.

    Args:
        split: ``"train"``, ``"validate"`` or ``"test"``.
        with_labels: If True, also return the segmentation mask (Task 2).
        transform: Optional callable applied to the image tensor.
        root: Dataset root; defaults to ``$OASIS_ROOT`` or the Rangpur path.
        cache: Decode every PNG once at construction and keep the uint8 arrays
            in memory. Measured cost for the training split is 9664 x 256 x 256
            bytes per modality, about 0.63 GB each, so images and masks together
            are ~1.3 GB — comfortably inside a 16 GB job. Decoding 19,000 PNGs
            from the cluster's network filesystem on every epoch is far slower
            than that one-off read. Pass ``cache=False`` to decode on demand.
        limit: Keep only the first ``limit`` slices, for quick debugging runs.
    """

    def __init__(
        self,
        split: str = "train",
        with_labels: bool = False,
        transform=None,
        root: str = OASIS_ROOT,
        cache: bool = True,
        limit: int | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.split = split
        self.with_labels = with_labels
        self.transform = transform

        images_path = image_dir(split, root)
        if not os.path.isdir(images_path):
            raise FileNotFoundError(
                f"{images_path} not found — set OASIS_ROOT, or run on Rangpur"
            )

        names = sorted(
            (name for name in os.listdir(images_path) if name.endswith(".png")),
            key=natural_key,
        )
        if limit is not None:
            names = names[:limit]
        if not names:
            raise FileNotFoundError(f"no PNG slices in {images_path}")

        self.names = names
        self.image_paths = [os.path.join(images_path, name) for name in names]

        self.mask_paths: list[str] = []
        if with_labels:
            masks_path = mask_dir(split, root)
            self.mask_paths = [os.path.join(masks_path, mask_name_for(n)) for n in names]
            # Check the pairing up front: a missing mask discovered in epoch 12
            # is a far worse failure than one discovered before training starts.
            missing = [p for p in self.mask_paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of {len(names)} masks missing, e.g. {missing[0]}"
                )

        self._images: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        if cache:
            self._images = self._read_all(self.image_paths)
            if with_labels:
                self._masks = masks_to_classes(self._read_all(self.mask_paths))

    @staticmethod
    def _read_all(paths: list[str]) -> np.ndarray:
        """Decode PNGs in parallel into one ``(N, H, W)`` uint8 array.

        Threads rather than processes: PNG decompression happens in zlib, which
        releases the GIL, so threads parallelise it without copying arrays
        between processes.
        """
        with ThreadPoolExecutor(max_workers=min(16, _available_cpus())) as pool:
            return np.stack(list(pool.map(_read_png, paths)))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return ``image`` or ``(image, mask)``.

        Images: float32 in [0, 1], shape (1, H, W).
        Masks:  int64 class indices, shape (H, W) — remap the raw PNG greyscale
                values (0/85/170/255) to 0..3 before use, or the one-hot
                encoding required by Task 2 will be wrong.
        """
        if self._images is not None:
            raw_image = self._images[index]
        else:
            raw_image = _read_png(self.image_paths[index])

        image = torch.from_numpy(raw_image).float().div_(255.0).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)

        if not self.with_labels:
            return image

        if self._masks is not None:
            classes = self._masks[index]
        else:
            classes = masks_to_classes(_read_png(self.mask_paths[index]))
        return image, torch.from_numpy(classes.astype(np.int64))
