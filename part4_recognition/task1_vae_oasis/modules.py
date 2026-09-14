"""Convolutional VAE.

The encoder outputs the mean and log-variance of an approximate posterior
q(z|x); the reparameterisation trick (z = mu + sigma * eps) keeps the sample
differentiable so gradients flow back through the encoder. Be ready to explain
why sampling z directly would break training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LATENT_DIM = 2  # 2 makes the manifold directly plottable; raise it and use UMAP


class Encoder(nn.Module):
    """Strided conv stack producing ``(mu, logvar)``."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        # TODO: Conv2d stack with stride 2, then two Linear heads.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Decoder(nn.Module):
    """Mirror of the encoder using ConvTranspose2d, ending in a sigmoid."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VAE(nn.Module):
    """Encoder + reparameterisation + decoder."""

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, with eps ~ N(0, I).

        logvar rather than sigma is predicted because it is unconstrained in
        sign, which keeps the network output stable and the exponential keeps
        sigma positive.
        """
        # TODO: std = torch.exp(0.5 * logvar); return mu + std * torch.randn_like(std)
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """Return ``(reconstruction, mu, logvar)``."""
        raise NotImplementedError


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """ELBO: reconstruction term + beta * KL divergence.

    KL for a diagonal Gaussian against N(0, I) has a closed form:
        -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Sum (do not average) over pixels and latent dims within a sample, then
    average over the batch, or the two terms will be mis-weighted relative to
    each other and the latent space will collapse or go unused.
    """
    # TODO: implement; return (total, recon_term, kl_term) for separate logging.
    raise NotImplementedError
