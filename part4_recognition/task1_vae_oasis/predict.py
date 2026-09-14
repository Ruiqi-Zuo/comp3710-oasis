"""VAE inference and manifold visualisation — this is what earns the mark.

    python -m part4_recognition.task1_vae_oasis.predict

Full marks require the trained model **and** a visualisation of the manifold it
learned. Two accepted routes:

  * 2D latent space: decode a grid of z values over the inverse CDF of the
    Gaussian prior (``scipy.stats.norm.ppf``) and tile the outputs. Sampling a
    uniform grid over raw z is a common mistake — it over-samples the tails.
  * Higher-dimensional latent space: encode the test set and reduce the mu
    vectors to 2D with UMAP (or t-SNE), then scatter them.

This script does both where they apply. The embedding is coloured by slice
position in the axial stack, which is a continuous anatomical variable the model
was never told about: a well-organised manifold shows it as a smooth gradient.
t-SNE is used for the higher-dimensional case because scikit-learn is already a
dependency, while UMAP pulls in numba and its NumPy version pins.
"""

from __future__ import annotations

import argparse
import re

import matplotlib

matplotlib.use("Agg")  # headless-safe: Rangpur has no display

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from scipy.stats import norm

from common import describe_device, get_device, save_fig
from part4_recognition.oasis import OASIS_ROOT
from part4_recognition.task1_vae_oasis.dataset import get_dataloaders
from part4_recognition.task1_vae_oasis.modules import VAE
from part4_recognition.task1_vae_oasis.train import CHECKPOINT

#: Grid quantiles. 0.05-0.95 rather than 0-1: the inverse CDF is infinite at the
#: ends, and those extreme codes are almost never produced by the encoder.
GRID_QUANTILES = (0.05, 0.95)


def load_model(device):
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = VAE(checkpoint["latent_dim"], checkpoint["image_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def slice_numbers(names: list[str]) -> np.ndarray:
    """``case_441_slice_17.nii.png`` -> 17."""
    return np.array([int(re.search(r"slice_(\d+)", name).group(1)) for name in names])


@torch.no_grad()
def encode_means(model, loader, device) -> np.ndarray:
    """Posterior means for every image in ``loader``, in loader order."""
    return torch.cat([model.encoder(images.to(device))[0].cpu() for images in loader]).numpy()


@torch.no_grad()
def plot_reconstructions(model, loader, device, n: int = 8) -> str:
    """Input slices above their reconstructions — the sanity check to show first.

    Decodes the posterior *mean* rather than a sample, which is the model's best
    single reconstruction and what should be judged.
    """
    dataset = loader.dataset
    picks = np.linspace(0, len(dataset) - 1, n).astype(int)
    images = torch.stack([dataset[i] for i in picks]).to(device)
    recon = model.decoder(model.encoder(images)[0])

    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.8))
    for column, (original, rebuilt) in enumerate(zip(images.cpu(), recon.cpu())):
        for row, image in enumerate((original, rebuilt)):
            axes[row, column].imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_xticks(())
            axes[row, column].set_yticks(())
    axes[0, 0].set_ylabel("test input", fontsize=10)
    axes[1, 0].set_ylabel("reconstruction", fontsize=10)
    fig.suptitle("VAE reconstructions of held-out test slices", fontsize=12)
    fig.tight_layout()
    path = save_fig(fig, "vae_reconstructions", "part4")
    plt.close(fig)
    return path


@torch.no_grad()
def plot_manifold(model, device, n: int = 20, image_size: int = 128) -> str:
    """Decode an n x n grid of latent codes into one large tiled image."""
    quantiles = np.linspace(*GRID_QUANTILES, n)
    values = norm.ppf(quantiles)

    # Row 0 is the top of the figure, so it takes the *largest* z2: up is positive.
    z1, z2 = np.meshgrid(values, values[::-1])
    codes = torch.tensor(np.stack([z1.ravel(), z2.ravel()], axis=1),
                         dtype=torch.float32, device=device)
    tiles = model.decoder(codes).cpu().squeeze(1).numpy()

    canvas = tiles.reshape(n, n, image_size, image_size).transpose(0, 2, 1, 3)
    canvas = canvas.reshape(n * image_size, n * image_size)

    fig, ax = plt.subplots(figsize=(11, 11))
    extent = (values[0], values[-1], values[0], values[-1])
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, extent=extent)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(
        f"Learned manifold: {n}x{n} grid decoded over the Gaussian prior's quantiles "
        f"{GRID_QUANTILES[0]:.2f}-{GRID_QUANTILES[1]:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    path = save_fig(fig, "vae_manifold", "part4")
    plt.close(fig)
    return path


def plot_latent_embedding(means: np.ndarray, slices: np.ndarray) -> str:
    """Scatter the test-set posterior means, coloured by slice position.

    Plotted directly for a 2D latent space, through t-SNE otherwise. For 2D, the
    region covered by the manifold grid is outlined, which ties the two figures
    together: the grid should span the part of the space the encoder uses.
    """
    latent_dim = means.shape[1]
    if latent_dim == 2:
        points, method = means, "posterior means"
    else:
        from sklearn.manifold import TSNE
        perplexity = min(30.0, (len(means) - 1) / 3)
        points = TSNE(n_components=2, perplexity=perplexity, init="pca",
                      random_state=0).fit_transform(means)
        method = f"t-SNE of {latent_dim}-D posterior means"

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=slices, cmap="viridis", s=9, alpha=0.8)
    fig.colorbar(scatter, ax=ax, label="slice index in the axial stack")

    if latent_dim == 2:
        low, high = norm.ppf(GRID_QUANTILES)
        ax.add_patch(Rectangle((low, low), high - low, high - low, fill=False,
                               ls="--", lw=1.0, ec="crimson", label="manifold grid extent"))
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")
        ax.legend(fontsize=9, loc="upper right")
    else:
        ax.set_xticks(())
        ax.set_yticks(())

    ax.set_title(f"Test set in latent space — {method}", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = save_fig(fig, "vae_latent_embedding", "part4")
    plt.close(fig)
    return path


@torch.no_grad()
def plot_prior_samples(model, device, n: int = 8, seed: int = 0) -> str:
    """Brains decoded from z ~ N(0, I) — what the model generates unprompted."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n * 2, model.latent_dim, generator=generator).to(device)
    samples = model.decoder(z).cpu().squeeze(1).numpy()

    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.6))
    for ax, image in zip(axes.flat, samples):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks(())
        ax.set_yticks(())
    fig.suptitle("Samples decoded from the prior, z ~ N(0, I)", fontsize=12)
    fig.tight_layout()
    path = save_fig(fig, "vae_samples", "part4")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE manifold visualisation")
    parser.add_argument("--root", default=OASIS_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grid", type=int, default=20, help="manifold grid size")
    args = parser.parse_args()

    device = get_device()
    print("Device:", describe_device(device))
    model, checkpoint = load_model(device)
    print(f"Loaded {CHECKPOINT}: latent {checkpoint['latent_dim']}, "
          f"{checkpoint['image_size']}px, {checkpoint['epochs']} epochs")

    _, _, test_loader = get_dataloaders(
        image_size=checkpoint["image_size"], root=args.root, limit=args.limit, num_workers=0
    )
    print(f"Test set: {len(test_loader.dataset)} slices")

    print("\nSaved:")
    print(" ", plot_reconstructions(model, test_loader, device))

    means = encode_means(model, test_loader, device)
    slices = slice_numbers(test_loader.dataset.names)
    print(" ", plot_latent_embedding(means, slices))

    if checkpoint["latent_dim"] == 2:
        print(" ", plot_manifold(model, device, args.grid, checkpoint["image_size"]))
        # How much of the prior's typical region the encoder actually uses.
        low, high = norm.ppf(GRID_QUANTILES)
        inside = np.all((means >= low) & (means <= high), axis=1).mean()
        print(f"  {inside:.1%} of test posterior means fall inside the grid extent")
    else:
        print("  (latent grid skipped: it needs a 2D latent space)")

    print(" ", plot_prior_samples(model, device))


if __name__ == "__main__":
    main()
