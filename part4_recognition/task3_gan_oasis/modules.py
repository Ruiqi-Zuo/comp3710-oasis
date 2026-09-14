"""DCGAN generator and discriminator.

DCGAN conventions that matter for stability — worth citing at the demo:
  * strided convolutions instead of pooling, in both networks
  * BatchNorm everywhere except the generator output and discriminator input
  * ReLU in the generator (tanh at the output), LeakyReLU(0.2) in the
    discriminator
  * Adam with lr 2e-4 and betas (0.5, 0.999)

If DCGAN will not converge, WGAN-GP is the usual next step: it replaces the
saturating loss with a Wasserstein estimate plus a gradient penalty, which is
markedly more robust to mode collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LATENT_DIM = 128


class Generator(nn.Module):
    """Maps a latent vector to an image via ConvTranspose2d upsampling."""

    def __init__(self, latent_dim: int = LATENT_DIM, base: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        # TODO: project z to (base*8, 4, 4), then upsample to the target size,
        # finishing with Tanh.
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Discriminator(nn.Module):
    """Maps an image to a single realism score (logit)."""

    def __init__(self, base: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def init_weights(module: nn.Module) -> None:
    """DCGAN initialisation: N(0, 0.02) for conv weights, N(1, 0.02) for BN."""
    raise NotImplementedError
