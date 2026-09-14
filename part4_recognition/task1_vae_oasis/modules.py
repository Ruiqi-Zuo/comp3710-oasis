"""Convolutional VAE.

The encoder outputs the mean and log-variance of an approximate posterior
q(z|x); the reparameterisation trick (z = mu + sigma * eps) keeps the sample
differentiable so gradients flow back through the encoder. Be ready to explain
why sampling z directly would break training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 2  # 2 makes the manifold directly plottable; raise it and use t-SNE/UMAP
IMAGE_SIZE = 128
CHANNELS = (32, 64, 128, 256)  # four stride-2 stages: 128 -> 64 -> 32 -> 16 -> 8
HIDDEN = 512


def _down(in_channels: int, out_channels: int) -> nn.Sequential:
    """Halve the resolution: 4x4 stride-2 conv, BatchNorm, LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up(in_channels: int, out_channels: int) -> nn.Sequential:
    """Double the resolution: 4x4 stride-2 transposed conv, BatchNorm, ReLU.

    Kernel 4 with stride 2 divides evenly, so every output pixel receives the
    same number of kernel taps. Kernel 3 with stride 2 would not, and produces
    the checkerboard artefacts transposed convolutions are known for.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Strided conv stack producing ``(mu, logvar)``."""

    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        if image_size % 2 ** len(CHANNELS):
            raise ValueError(f"image_size must be divisible by {2 ** len(CHANNELS)}")

        stages, previous = [], 1
        for width in CHANNELS:
            stages.append(_down(previous, width))
            previous = width
        self.features = nn.Sequential(*stages, nn.Flatten())

        n_flat = CHANNELS[-1] * (image_size // 2 ** len(CHANNELS)) ** 2
        self.hidden = nn.Sequential(nn.Linear(n_flat, HIDDEN), nn.LeakyReLU(0.2, inplace=True))
        self.mu = nn.Linear(HIDDEN, latent_dim)
        self.logvar = nn.Linear(HIDDEN, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(self.features(x))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """Mirror of the encoder using ConvTranspose2d, ending in a sigmoid."""

    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.start = image_size // 2 ** len(CHANNELS)

        self.project = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, CHANNELS[-1] * self.start ** 2),
            nn.ReLU(inplace=True),
        )

        widths = list(reversed(CHANNELS))                      # 256, 128, 64, 32
        stages = [_up(a, b) for a, b in zip(widths, widths[1:])]
        # The last stage goes straight to one channel with no BatchNorm or ReLU:
        # it produces the pixel values, which the sigmoid then maps into [0, 1].
        stages.append(nn.ConvTranspose2d(widths[-1], 1, kernel_size=4, stride=2, padding=1))
        self.upsample = nn.Sequential(*stages)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.project(z).view(-1, CHANNELS[-1], self.start, self.start)
        return torch.sigmoid(self.upsample(h))


class VAE(nn.Module):
    """Encoder + reparameterisation + decoder."""

    def __init__(self, latent_dim: int = LATENT_DIM, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, with eps ~ N(0, I).

        logvar rather than sigma is predicted because it is unconstrained in
        sign, which keeps the network output stable and the exponential keeps
        sigma positive.
        """
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):
        """Return ``(reconstruction, mu, logvar)``."""
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """ELBO: reconstruction term + beta * KL divergence.

    KL for a diagonal Gaussian against N(0, I) has a closed form:
        -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Sum (do not average) over pixels and latent dims within a sample, then
    average over the batch, or the two terms will be mis-weighted relative to
    each other and the latent space will collapse or go unused.

    To see why: a 128x128 image has 16,384 pixels and the latent has 2 dims.
    Averaging the reconstruction over pixels shrinks it by four orders of
    magnitude relative to the KL, so the optimiser satisfies the prior and
    ignores the image — posterior collapse by bookkeeping error.

    Returns ``(total, reconstruction, kl)``, each averaged over the batch, so the
    two terms can be logged separately.
    """
    batch = x.size(0)
    reconstruction = F.binary_cross_entropy(recon_x, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return reconstruction + beta * kl, reconstruction, kl
