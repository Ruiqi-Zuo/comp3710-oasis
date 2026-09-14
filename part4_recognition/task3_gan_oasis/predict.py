"""Generate brains from the trained GAN.

    python -m part4_recognition.task3_gan_oasis.predict
"""

from __future__ import annotations

import torch

from common import get_device, save_fig


@torch.no_grad()
def sample_grid(generator, device, n: int = 64):
    """Generate a grid of independent samples."""
    raise NotImplementedError


@torch.no_grad()
def latent_interpolation(generator, device, steps: int = 10):
    """Interpolate between two latent vectors.

    Smooth anatomical transitions are good evidence that the generator learned a
    real manifold rather than memorising a handful of outputs.
    """
    raise NotImplementedError


def main() -> None:
    device = get_device()
    raise NotImplementedError


if __name__ == "__main__":
    main()
