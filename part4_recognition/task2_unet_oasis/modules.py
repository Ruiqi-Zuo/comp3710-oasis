"""UNet, plus the Dice loss and metric.

UNet is an encoder-decoder with **skip connections**: each decoder stage is
concatenated with the encoder feature map at the same resolution. Downsampling
gives context ("what"), the skips restore the spatial detail ("where") that
pooling destroyed. Being able to explain this is explicitly required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 4  # background, CSF, grey matter, white matter


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, the repeating unit of UNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UNet(nn.Module):
    """UNet with four down/up stages.

    Args:
        n_classes: Number of segmentation classes. The output has this many
            channels — the task requires **categorical (one-hot) output**, not
            a single-channel label map.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = N_CLASSES, base: int = 32) -> None:
        super().__init__()
        # TODO:
        #   encoder: DoubleConv at base, 2*base, 4*base, 8*base, MaxPool2d between
        #   bottleneck: 16*base
        #   decoder: ConvTranspose2d upsample, concatenate the skip, DoubleConv
        #   head: Conv2d(base, n_classes, kernel_size=1) -> logits
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-class logits of shape ``(N, n_classes, H, W)``."""
        raise NotImplementedError


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-class Dice similarity coefficient.

    DSC = 2|X and Y| / (|X| + |Y|)

    Returns one value per class. The task requires **> 0.9 for all labels**, so
    always report the per-class vector, never just the mean — background alone
    can score above 0.95 and hide a failing tissue class.
    """
    # TODO: argmax or softmax the logits, one-hot the target, reduce over
    # batch and spatial dims per class.
    raise NotImplementedError


class DiceLoss(nn.Module):
    """Soft Dice loss, usually combined with cross-entropy.

    Use the softmax probabilities (not argmax) so the loss stays
    differentiable. A common and effective choice is ``CE + DiceLoss``: CE gives
    stable early gradients, Dice directly optimises the reported metric and
    handles the class imbalance of a mostly-background image.
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
