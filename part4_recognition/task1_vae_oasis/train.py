"""Train the OASIS VAE.

    python -m part4_recognition.task1_vae_oasis.train
"""

from __future__ import annotations

import torch

from common import get_device, plot_curves, save_fig, set_seed

EPOCHS = 50
LEARNING_RATE = 1e-3


def main() -> None:
    set_seed(42)
    device = get_device()
    # TODO:
    #   1. loaders + VAE(latent_dim).to(device), Adam optimiser
    #   2. per epoch: accumulate total / reconstruction / KL loss separately so
    #      you can show posterior collapse if it happens
    #   3. save reconstruction grids every few epochs (input vs output) as
    #      evidence of training
    #   4. save the checkpoint to outputs/part4/vae.pth and the loss curves
    raise NotImplementedError


if __name__ == "__main__":
    main()
