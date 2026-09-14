"""Train the OASIS UNet.

    python -m part4_recognition.task2_unet_oasis.train
"""

from __future__ import annotations

import torch

from common import get_device, plot_curves, save_fig, set_seed

EPOCHS = 40
LEARNING_RATE = 1e-3


def main() -> None:
    set_seed(42)
    device = get_device()
    # TODO:
    #   1. loaders + UNet(n_classes=4).to(device)
    #   2. loss = CrossEntropyLoss() + DiceLoss(); Adam; cosine or plateau schedule
    #   3. per epoch: log per-class validation DSC, not just the mean
    #   4. keep the best checkpoint by worst-class DSC — the requirement is
    #      "> 0.9 for ALL labels", so the worst class is the metric that matters
    #   5. save outputs/part4/unet.pth and the DSC curves
    raise NotImplementedError


if __name__ == "__main__":
    main()
