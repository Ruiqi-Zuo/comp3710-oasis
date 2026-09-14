"""UNet, plus the Dice loss and metric.

UNet is an encoder-decoder with **skip connections**: each decoder stage is
concatenated with the encoder feature map at the same resolution. Downsampling
gives context ("what"), the skips restore the spatial detail ("where") that
pooling destroyed. Being able to explain this is explicitly required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4  # background, CSF, grey matter, white matter
DEPTH = 4      # four 2x poolings: 256 -> 16 at the bottleneck


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, the repeating unit of UNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            # No bias: BatchNorm immediately follows and has its own shift.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """UNet with four down/up stages.

    Args:
        n_classes: Number of segmentation classes. The output has this many
            channels — the task requires **categorical (one-hot) output**, not
            a single-channel label map.

    ``padding=1`` throughout keeps every feature map the same size as its skip
    partner, so the skips concatenate directly. The original paper used
    unpadded convolutions and had to crop the encoder maps; with a 256x256 input
    and four poolings everything divides evenly, so no cropping is needed.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = N_CLASSES, base: int = 32) -> None:
        super().__init__()
        widths = [base * 2 ** level for level in range(DEPTH)]   # 32, 64, 128, 256
        self.divisor = 2 ** DEPTH

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoders.append(DoubleConv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)  # 512 at 16x16

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = widths[-1] * 2
        for width in reversed(widths):
            self.upsamplers.append(nn.ConvTranspose2d(previous, width, kernel_size=2, stride=2))
            # After concatenating the skip there are 2 * width channels to fuse.
            self.decoders.append(DoubleConv(width * 2, width))
            previous = width

        # 1x1 conv: a per-pixel linear classifier over the final feature vector.
        self.head = nn.Conv2d(widths[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-class logits of shape ``(N, n_classes, H, W)``."""
        if x.shape[-1] % self.divisor or x.shape[-2] % self.divisor:
            raise ValueError(f"input H and W must be divisible by {self.divisor}, got {tuple(x.shape[-2:])}")

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)          # full-resolution detail, kept for the decoder
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = upsample(x)
            x = decoder(torch.cat([skip, x], dim=1))

        return self.head(x)


def one_hot(target: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    """``(N, H, W)`` class indices -> ``(N, C, H, W)`` one-hot float tensor."""
    return F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()


@torch.no_grad()
def dice_counts(logits: torch.Tensor, target: torch.Tensor, n_classes: int = N_CLASSES):
    """Per-class ``(intersection, prediction size, target size)`` for one batch.

    Returned as raw counts so they can be **summed over a whole dataset** before
    dividing. Averaging a per-image Dice instead is unstable here: an edge slice
    with no white matter in either the prediction or the mask gives 0/0 for that
    class, and whatever convention fills the gap then dominates the mean.
    """
    prediction = one_hot(logits.argmax(dim=1), n_classes)
    truth = one_hot(target, n_classes)
    dims = (0, 2, 3)
    return (
        (prediction * truth).sum(dims),
        prediction.sum(dims),
        truth.sum(dims),
    )


def dice_from_counts(intersection, predicted, actual, eps: float = 1e-6) -> torch.Tensor:
    return (2 * intersection + eps) / (predicted + actual + eps)


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-class Dice similarity coefficient.

    DSC = 2|X and Y| / (|X| + |Y|)

    Returns one value per class. The task requires **> 0.9 for all labels**, so
    always report the per-class vector, never just the mean — background alone
    can score above 0.95 and hide a failing tissue class.

    Hard Dice: the prediction is the argmax label, one-hot encoded. This is the
    metric reported; the soft version in :class:`DiceLoss` is only its
    differentiable stand-in for training.
    """
    return dice_from_counts(*dice_counts(logits, target, logits.shape[1]), eps=eps)


class DiceLoss(nn.Module):
    """Soft Dice loss, usually combined with cross-entropy.

    Use the softmax probabilities (not argmax) so the loss stays
    differentiable. A common and effective choice is ``CE + DiceLoss``: CE gives
    stable early gradients, Dice directly optimises the reported metric and
    handles the class imbalance of a mostly-background image.

    The Dice is computed per class over the whole batch and then averaged with
    equal weight per class. That equal weighting is what counters the imbalance:
    CSF covers under 6% of the pixels but contributes a quarter of this loss.
    """

    def __init__(self, n_classes: int = N_CLASSES, smooth: float = 1.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # float32 even under autocast: summing 16 x 65,536 probabilities per
        # class in fp16 loses precision, and the loss is cheap either way.
        probabilities = logits.float().softmax(dim=1)
        truth = one_hot(target, self.n_classes)
        dims = (0, 2, 3)
        intersection = (probabilities * truth).sum(dims)
        cardinality = probabilities.sum(dims) + truth.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    """Cross entropy plus soft Dice, returned together with both parts for logging.

    ``nn.CrossEntropyLoss`` takes class indices, but it computes exactly the
    categorical cross entropy against the one-hot target — an index is simply a
    sparse encoding of a one-hot vector. The Dice term one-hot encodes the
    target explicitly.
    """

    def __init__(self, n_classes: int = N_CLASSES, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss(n_classes)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        ce = self.cross_entropy(logits.float(), target)
        dice = self.dice(logits, target)
        return ce + self.dice_weight * dice, ce, dice
