"""Tests for the Task 2 UNet, the Dice metric and the Dice loss.

    python -m pytest tests          # if pytest is installed
    python -m tests.test_unet       # otherwise
"""

from __future__ import annotations

import math

import torch

from part4_recognition.task2_unet_oasis.modules import (
    N_CLASSES,
    CEDiceLoss,
    DiceLoss,
    UNet,
    dice_coefficient,
    dice_counts,
    dice_from_counts,
    one_hot,
)


def _confident_logits(target: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Logits whose argmax is ``target`` with near-certain softmax."""
    return one_hot(target, n_classes) * 20.0 - 10.0


def test_output_is_categorical_and_full_resolution():
    model = UNet(in_channels=1, n_classes=4, base=8)
    logits = model(torch.rand(2, 1, 256, 256))
    assert logits.shape == (2, 4, 256, 256)
    assert torch.allclose(logits.softmax(dim=1).sum(dim=1), torch.ones(2, 256, 256), atol=1e-5)


def test_rejects_sizes_not_divisible_by_16():
    try:
        UNet(base=8)(torch.rand(1, 1, 100, 100))
    except ValueError:
        pass
    else:
        raise AssertionError("a 100x100 input should be rejected")


def test_skip_connections_carry_fine_detail():
    """Removing the finest skip must change the output: the decoder depends on it."""
    torch.manual_seed(0)
    model = UNet(base=8).eval()
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        reference = model(x)
        original = model.decoders[-1].forward

        def without_skip(features):
            half = features.shape[1] // 2
            features = features.clone()
            features[:, :half] = 0          # the skip is concatenated first
            return original(features)

        model.decoders[-1].forward = without_skip
        ablated = model(x)
    assert not torch.allclose(reference, ablated)


def test_dice_is_one_for_a_perfect_prediction():
    target = torch.randint(0, N_CLASSES, (3, 32, 32))
    assert torch.allclose(dice_coefficient(_confident_logits(target), target), torch.ones(N_CLASSES))


def test_dice_matches_a_hand_computed_case():
    # 1x4 image. Truth: [0, 1, 1, 1]; prediction: [0, 1, 0, 0].
    target = torch.tensor([[[0, 1, 1, 1]]])
    predicted = torch.tensor([[[0, 1, 0, 0]]])
    dice = dice_coefficient(_confident_logits(predicted, 2), target)
    # class 0: |X|=3, |Y|=1, overlap 1 -> 2/4.  class 1: |X|=1, |Y|=3, overlap 1 -> 2/4.
    assert torch.allclose(dice, torch.tensor([0.5, 0.5]), atol=1e-5)


def test_counts_accumulated_over_batches_equal_one_big_batch():
    """Dataset-level Dice must not depend on how the data was batched."""
    torch.manual_seed(1)
    target = torch.randint(0, N_CLASSES, (10, 16, 16))
    logits = torch.randn(10, N_CLASSES, 16, 16)

    whole = dice_coefficient(logits, target)

    totals = [torch.zeros(N_CLASSES) for _ in range(3)]
    for start in range(0, 10, 3):
        for total, part in zip(totals, dice_counts(logits[start:start + 3], target[start:start + 3])):
            total += part
    assert torch.allclose(dice_from_counts(*totals), whole, atol=1e-6)


def test_absent_class_does_not_produce_nan():
    target = torch.zeros(2, 8, 8, dtype=torch.long)       # only background present
    dice = dice_coefficient(_confident_logits(target), target)
    assert torch.isfinite(dice).all()


def test_dice_loss_bounds_and_gradient():
    target = torch.randint(0, N_CLASSES, (2, 16, 16))
    assert DiceLoss()(_confident_logits(target), target).item() < 1e-3

    logits = torch.randn(2, N_CLASSES, 16, 16, requires_grad=True)
    loss = DiceLoss()(logits, target)
    assert 0.0 < loss.item() < 1.0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_dice_loss_weights_classes_equally():
    """Missing a class that is 1% of the pixels costs half the loss, not 1%.

    Pixel accuracy would still read 99%. The image is 100x100 rather than tiny
    because the smoothing term is +1 in numerator and denominator: on a
    4-pixel class it alone lifts a total miss to Dice 1/5, while on the ~59,000
    CSF pixels in a real batch of 16 it is negligible.
    """
    target = torch.zeros(1, 100, 100, dtype=torch.long)
    target[0, :10, :10] = 1                                 # class 1: 100 px, 1%
    miss_small = torch.zeros_like(target)                   # predict background everywhere
    loss = DiceLoss(n_classes=2)(_confident_logits(miss_small, 2), target)
    # background Dice = 19801/19901 ~ 0.995, class 1 Dice = 1/101 ~ 0.010 -> loss ~ 0.4975
    assert math.isclose(loss.item(), 1 - (19801 / 19901 + 1 / 101) / 2, abs_tol=1e-4)


def test_flip_augmentation_keeps_image_and_mask_aligned():
    """After augmentation every mask must still be the mask of its own image."""
    from part4_recognition.task2_unet_oasis.dataset import random_hflip

    torch.manual_seed(3)
    images = torch.rand(32, 1, 16, 16)
    masks = (images.squeeze(1) * N_CLASSES).long().clamp(max=N_CLASSES - 1)   # mask = f(image)
    images_before, masks_before = images.clone(), masks.clone()

    flipped_images, flipped_masks = random_hflip(images, masks)

    rederived = (flipped_images.squeeze(1) * N_CLASSES).long().clamp(max=N_CLASSES - 1)
    assert torch.equal(rederived, flipped_masks)
    changed = (flipped_images != images).flatten(1).any(dim=1)
    assert 0 < changed.sum() < 32                      # some flipped, some not
    # The caller's tensors must not be modified in place.
    assert torch.equal(images, images_before) and torch.equal(masks, masks_before)


def test_combined_loss_is_the_sum_of_its_parts():
    target = torch.randint(0, N_CLASSES, (2, 16, 16))
    logits = torch.randn(2, N_CLASSES, 16, 16)
    total, ce, dice = CEDiceLoss()(logits, target)
    assert math.isclose(total.item(), ce.item() + dice.item(), rel_tol=1e-6)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} passed")
