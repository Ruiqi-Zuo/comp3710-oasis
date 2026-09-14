"""Train the OASIS GAN.

    python -m part4_recognition.task3_gan_oasis.train

Warning from the lab sheet: GAN convergence is chaotic. Attempt this only if
you are confident, and get it working on MNIST first.

Evidence you must keep, because it is part of the mark:
  * generator and discriminator loss curves
  * a sample grid saved at fixed intervals from a **fixed** latent batch, so
    progress across epochs is comparable
  * the final samples, which must look like plausible and *distinct* brains

Mode collapse (every sample converging to the same image) must be fully
resolved. Detect it by watching the pixel-wise variance across a sample batch,
not by eyeballing one image.
"""

from __future__ import annotations

import torch

from common import get_device, plot_curves, save_fig, set_seed

EPOCHS = 100
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)


def main() -> None:
    set_seed(42)
    device = get_device()
    # TODO:
    #   1. G, D with init_weights; two Adam optimisers
    #   2. per batch: update D on real + fake, then update G
    #   3. fixed_noise = torch.randn(64, LATENT_DIM, device=device) created once
    #   4. log D(real) and D(fake) as well as the losses — losses alone say very
    #      little about GAN health
    #   5. save checkpoints and sample grids to outputs/part4/gan/
    raise NotImplementedError


if __name__ == "__main__":
    main()
