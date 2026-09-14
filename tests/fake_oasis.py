"""A tiny synthetic stand-in for the OASIS directory tree.

The real dataset may only be read in place on Rangpur, so code is developed and
smoke-tested locally against this instead. It reproduces what the loaders
depend on — directory and file names, a subject-level split, 8-bit greyscale
PNGs, and masks using the grey levels 0/85/170/255 — and draws a crude "brain"
of nested ellipses so a segmentation network has real structure to learn.

    python -m tests.fake_oasis /tmp/fake_oasis
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

# Same subject numbering scheme as the real split: train, then validate, then test.
DEFAULT_SUBJECTS = {"train": 4, "validate": 2, "test": 2}


def _slice_arrays(rng: np.random.Generator, size: int, slice_index: int, n_slices: int):
    """One synthetic slice: background, a CSF ring, grey matter, white matter core."""
    y, x = np.mgrid[0:size, 0:size]
    centre = size / 2 + rng.normal(0, size * 0.02, size=2)
    # The brain grows towards the middle slices, as a real axial stack does.
    scale = 0.55 + 0.35 * np.sin(np.pi * (slice_index + 0.5) / n_slices)
    radius = np.hypot((y - centre[0]) / (0.42 * scale), (x - centre[1]) / (0.34 * scale)) / size

    mask = np.zeros((size, size), np.uint8)
    mask[radius < 1.00] = 85     # CSF
    mask[radius < 0.90] = 170    # grey matter
    mask[radius < 0.55] = 255    # white matter

    intensity = {0: 5, 85: 60, 170: 120, 255: 200}
    image = np.zeros((size, size), np.float32)
    for level, value in intensity.items():
        image[mask == level] = value
    image += rng.normal(0, 8, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), mask


def make_fake_oasis(
    root: str,
    subjects: dict[str, int] | None = None,
    slices_per_subject: int = 4,
    size: int = 64,
    seed: int = 0,
) -> str:
    """Write the fake tree under ``root`` and return ``root``."""
    subjects = subjects or DEFAULT_SUBJECTS
    rng = np.random.default_rng(seed)
    case = 1

    for split, n_subjects in subjects.items():
        images = os.path.join(root, f"keras_png_slices_{split}")
        masks = os.path.join(root, f"keras_png_slices_seg_{split}")
        os.makedirs(images, exist_ok=True)
        os.makedirs(masks, exist_ok=True)

        for _ in range(n_subjects):
            for s in range(slices_per_subject):
                image, mask = _slice_arrays(rng, size, s, slices_per_subject)
                stem = f"{case:03d}_slice_{s}.nii.png"
                Image.fromarray(image).save(os.path.join(images, f"case_{stem}"))
                Image.fromarray(mask).save(os.path.join(masks, f"seg_{stem}"))
            case += 1

    return root


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "fake_oasis"
    print(make_fake_oasis(target))
