"""Tests for the shared OASIS loader, on synthetic data.

    python -m pytest tests          # if pytest is installed
    python -m tests.test_oasis      # otherwise
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

from part4_recognition.oasis import (
    N_SEG_CLASSES,
    OASISSlices,
    mask_dir,
    mask_name_for,
    masks_to_classes,
    natural_key,
)
from tests.fake_oasis import make_fake_oasis


def _fresh_root(slices_per_subject: int = 12) -> str:
    root = tempfile.mkdtemp(prefix="fake_oasis_")
    return make_fake_oasis(root, slices_per_subject=slices_per_subject)


def test_lengths_and_pairing():
    root = _fresh_root()
    try:
        for split, n_subjects in (("train", 4), ("validate", 2), ("test", 2)):
            dataset = OASISSlices(split, with_labels=True, root=root)
            assert len(dataset) == n_subjects * 12
            for image_path, mask_path in zip(dataset.image_paths, dataset.mask_paths):
                assert os.path.basename(mask_path) == mask_name_for(os.path.basename(image_path))
    finally:
        shutil.rmtree(root)


def test_natural_order_keeps_slices_consecutive():
    names = ["case_001_slice_10.nii.png", "case_001_slice_2.nii.png", "case_001_slice_1.nii.png"]
    assert sorted(names, key=natural_key) == [
        "case_001_slice_1.nii.png", "case_001_slice_2.nii.png", "case_001_slice_10.nii.png",
    ]


def test_tensor_contract():
    root = _fresh_root()
    try:
        image, mask = OASISSlices("train", with_labels=True, root=root)[5]
        assert image.dtype == torch.float32 and image.shape == (1, 64, 64)
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert mask.dtype == torch.int64 and mask.shape == (64, 64)
        assert set(mask.unique().tolist()) <= set(range(N_SEG_CLASSES))

        unlabelled = OASISSlices("train", with_labels=False, root=root)[5]
        assert torch.equal(unlabelled, image)
    finally:
        shutil.rmtree(root)


def test_labels_match_raw_grey_levels():
    root = _fresh_root()
    try:
        dataset = OASISSlices("test", with_labels=True, root=root)
        raw = np.asarray(Image.open(dataset.mask_paths[3]))
        _, mask = dataset[3]
        assert np.array_equal(mask.numpy(), raw // 85)
    finally:
        shutil.rmtree(root)


def test_cache_and_lazy_agree():
    root = _fresh_root()
    try:
        cached = OASISSlices("validate", with_labels=True, root=root, cache=True)
        lazy = OASISSlices("validate", with_labels=True, root=root, cache=False)
        for index in (0, 7, len(cached) - 1):
            assert torch.equal(cached[index][0], lazy[index][0])
            assert torch.equal(cached[index][1], lazy[index][1])
    finally:
        shutil.rmtree(root)


def test_limit_and_transform():
    root = _fresh_root()
    try:
        dataset = OASISSlices("train", root=root, limit=5, transform=lambda x: x * 0 + 0.5)
        assert len(dataset) == 5
        assert torch.all(dataset[0] == 0.5)
    finally:
        shutil.rmtree(root)


def test_missing_mask_is_caught_at_construction():
    root = _fresh_root()
    try:
        victim = sorted(os.listdir(mask_dir("train", root)))[0]
        os.remove(os.path.join(mask_dir("train", root), victim))
        try:
            OASISSlices("train", with_labels=True, root=root)
        except FileNotFoundError as error:
            assert "1 of 48 masks missing" in str(error)
        else:
            raise AssertionError("a missing mask was not reported")
    finally:
        shutil.rmtree(root)


def test_invalid_grey_level_is_rejected():
    raw = np.array([[0, 85], [170, 86]], dtype=np.uint8)
    try:
        masks_to_classes(raw)
    except ValueError as error:
        assert "86" in str(error)
    else:
        raise AssertionError("grey level 86 was silently accepted")
    assert masks_to_classes(np.array([[0, 85, 170, 255]], np.uint8)).tolist() == [[0, 1, 2, 3]]


def test_unknown_split_is_rejected():
    try:
        OASISSlices("training", root=tempfile.gettempdir())
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid split name was accepted")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} passed")
