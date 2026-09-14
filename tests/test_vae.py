"""Tests for the Task 1 VAE — shapes, and the properties of the ELBO.

    python -m pytest tests          # if pytest is installed
    python -m tests.test_vae        # otherwise
"""

from __future__ import annotations

import math

import torch

from part4_recognition.task1_vae_oasis.modules import VAE, vae_loss


def test_shapes_and_output_range():
    model = VAE(latent_dim=2, image_size=128)
    x = torch.rand(3, 1, 128, 128)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == (3, 2)
    assert 0.0 <= recon.min() and recon.max() <= 1.0


def test_other_sizes_and_latent_dims():
    for size, latent in ((64, 16), (256, 32)):
        recon, mu, _ = VAE(latent_dim=latent, image_size=size)(torch.rand(2, 1, size, size))
        assert recon.shape == (2, 1, size, size) and mu.shape == (2, latent)


def test_rejects_size_not_divisible_by_16():
    try:
        VAE(image_size=100)
    except ValueError:
        pass
    else:
        raise AssertionError("image_size=100 should be rejected")


def test_kl_is_zero_at_the_prior():
    """q(z|x) = N(0, I) is the prior itself, so the KL must be exactly zero."""
    x = torch.rand(4, 1, 8, 8)
    _, _, kl = vae_loss(x, x, torch.zeros(4, 2), torch.zeros(4, 2))
    assert kl.abs().item() < 1e-6


def test_kl_matches_closed_form():
    """One latent dim with mu=1, logvar=0: KL = 0.5 * mu^2 = 0.5 per sample."""
    x = torch.rand(5, 1, 8, 8)
    _, _, kl = vae_loss(x, x, torch.ones(5, 1), torch.zeros(5, 1))
    assert math.isclose(kl.item(), 0.5, rel_tol=1e-6)


def test_reconstruction_is_summed_over_pixels_not_averaged():
    """Doubling the image area should double the per-sample reconstruction term."""
    torch.manual_seed(0)
    small_x, small_r = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    big_x, big_r = small_x.repeat(1, 1, 1, 2), small_r.repeat(1, 1, 1, 2)
    zeros = torch.zeros(2, 2)
    _, small, _ = vae_loss(small_r, small_x, zeros, zeros)
    _, big, _ = vae_loss(big_r, big_x, zeros, zeros)
    assert math.isclose(big.item(), 2 * small.item(), rel_tol=1e-5)


def test_beta_scales_only_the_kl():
    x, r = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    mu, logvar = torch.randn(2, 2), torch.randn(2, 2)
    total_1, recon_1, kl_1 = vae_loss(r, x, mu, logvar, beta=1.0)
    total_0, recon_0, _ = vae_loss(r, x, mu, logvar, beta=0.0)
    assert torch.allclose(recon_1, recon_0) and torch.allclose(total_0, recon_0)
    assert torch.allclose(total_1, recon_1 + kl_1)


def test_gradients_reach_the_encoder_through_the_sample():
    """The point of reparameterisation: mu and logvar receive gradients."""
    model = VAE(latent_dim=2, image_size=64)
    x = torch.rand(2, 1, 64, 64)
    recon, mu, logvar = model(x)
    total, _, _ = vae_loss(recon, x, mu, logvar)
    total.backward()
    for head in (model.encoder.mu, model.encoder.logvar):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"{len(tests)} passed")
