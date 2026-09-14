"""Survey the OASIS data on Rangpur so the Part 4 loaders match the real layout.

    python scripts/probe_oasis.py                        # default root
    python scripts/probe_oasis.py /some/other/root

Read-only: it lists directories, counts files, and opens the first image in
each image folder to report its format. Run it on a CPU node, not the login
node, and paste the whole output back.

What the Part 4 code needs to know, and this prints:
  * the directory layout and the train/validate/test split, if any
  * file format (PNG slices, NIfTI volumes, NumPy arrays)
  * image size, mode and dtype
  * for segmentation masks: the distinct pixel values, which determine how the
    labels are one-hot encoded — Task 2 requires categorical output
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/groups/comp3710"
MAX_DEPTH = 4
MAX_DIRS_LISTED = 80
MAX_DIRS_INSPECTED = 16
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff",
                    ".nii", ".nii.gz", ".npy", ".npz")


def extension(name: str) -> str:
    return ".nii.gz" if name.lower().endswith(".nii.gz") else os.path.splitext(name)[1].lower()


def describe_array(array: np.ndarray) -> str:
    text = (f"shape={array.shape} dtype={array.dtype} "
            f"min={array.min()} max={array.max()}")
    distinct = np.unique(array)
    if distinct.size <= 16:
        text += f" distinct values={distinct.tolist()}  <- looks like a label mask"
    else:
        text += f" ({distinct.size} distinct values)"
    return text


def inspect(path: str) -> str:
    ext = extension(path)
    try:
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            from PIL import Image
            with Image.open(path) as image:
                return f"PIL mode={image.mode} size={image.size}  " + describe_array(np.asarray(image))
        if ext == ".npy":
            return describe_array(np.load(path, mmap_mode="r")[:])
        if ext == ".npz":
            with np.load(path) as archive:
                return "npz keys=" + ", ".join(
                    f"{key}:{archive[key].shape}/{archive[key].dtype}" for key in archive.files
                )
        if ext in (".nii", ".nii.gz"):
            try:
                import nibabel
            except ImportError:
                return "NIfTI volume (install nibabel to inspect: pip install nibabel)"
            volume = nibabel.load(path)
            return (f"NIfTI shape={volume.shape} dtype={volume.get_data_dtype()} "
                    f"voxel size={volume.header.get_zooms()}")
    except Exception as error:  # report and keep surveying
        return f"could not open: {type(error).__name__}: {error}"
    return "unrecognised format"


MASK_SAMPLE = 300


def mask_statistics(dirpath: str, images: list[str]) -> str | None:
    """Pixel-value census over a spread of masks, or None if these are not masks.

    One file is not enough to count the classes: sorted by name, the first
    slices are often at the edge of the brain and contain only background and
    one or two tissues. Sampling evenly across the whole folder catches every
    label, and the pixel shares show how badly the classes are imbalanced — which
    decides whether plain cross entropy is usable or Dice loss is needed.
    """
    from PIL import Image

    first = np.asarray(Image.open(os.path.join(dirpath, images[0])))
    if np.unique(first).size > 16:
        return None

    picks = np.unique(np.linspace(0, len(images) - 1, min(MASK_SAMPLE, len(images))).astype(int))
    counts: Counter = Counter()
    for index in picks:
        values, freq = np.unique(
            np.asarray(Image.open(os.path.join(dirpath, images[index]))), return_counts=True
        )
        counts.update(dict(zip(values.tolist(), freq.tolist())))

    total = sum(counts.values())
    shares = ", ".join(f"{value}: {count / total:.2%}" for value, count in sorted(counts.items()))
    return f"mask census over {len(picks)} files -> {len(counts)} labels {{{shares}}}"


def main() -> None:
    print(f"root: {ROOT}")
    if not os.path.isdir(ROOT):
        print("  does not exist or is not readable")
        return

    listed = 0
    inspected = 0
    extensions: Counter = Counter()
    image_dirs: list[tuple[str, list[str]]] = []

    print(f"\nDirectories (depth <= {MAX_DEPTH}), with file counts:")
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames.sort()
        depth = os.path.relpath(dirpath, ROOT).count(os.sep) + (dirpath != ROOT)
        if depth >= MAX_DEPTH:
            dirnames[:] = []

        for name in filenames:
            extensions[extension(name)] += 1

        images = sorted(f for f in filenames if f.lower().endswith(IMAGE_EXTENSIONS))
        if images:
            image_dirs.append((dirpath, images))

        if listed < MAX_DIRS_LISTED:
            indent = "  " * depth
            label = os.path.basename(dirpath) or dirpath
            print(f"  {indent}{label}/  {len(filenames)} files, {len(dirnames)} subdirs")
            listed += 1
        elif listed == MAX_DIRS_LISTED:
            print("  ... (listing truncated)")
            listed += 1

    print("\nFile types:")
    for ext, count in extensions.most_common(12):
        print(f"  {ext or '(none)':<10}{count}")

    print("\nFirst files in each image folder:")
    for dirpath, images in image_dirs[:MAX_DIRS_INSPECTED]:
        inspected += 1
        print(f"\n  {dirpath}  ({len(images)} images)")
        print(f"    e.g. {images[0]}" + (f", {images[1]}" if len(images) > 1 else "")
              + (f" ... {images[-1]}" if len(images) > 2 else ""))
        print(f"    {inspect(os.path.join(dirpath, images[0]))}")
        if images[0].lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            census = mask_statistics(dirpath, images)
            if census:
                print(f"    {census}")
    if len(image_dirs) > MAX_DIRS_INSPECTED:
        print(f"\n  ... {len(image_dirs) - MAX_DIRS_INSPECTED} more image folders not inspected")


if __name__ == "__main__":
    main()
