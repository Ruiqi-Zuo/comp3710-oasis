"""Train the OASIS VAE.

    python -m part4_recognition.task1_vae_oasis.train
    python -m part4_recognition.task1_vae_oasis.train --latent-dim 16
    python -m part4_recognition.task1_vae_oasis.train --epochs 2 --limit 256   # smoke test

Two guards against posterior collapse, which is the failure a VAE demo is most
likely to be questioned on:

  * **KL warm-up.** beta rises linearly from 0 to its target over the first
    ``--warmup-epochs``. Early on the decoder is useless, so the cheapest way to
    lower the loss is to match the prior and ignore the input; letting the
    reconstruction term establish itself first prevents that.
  * **Active units.** Each epoch counts the latent dimensions whose posterior
    mean actually varies across the validation set (variance > 0.01, following
    Burda et al., 2016). A collapsed dimension carries no information about the
    input, so collapse shows up here as a number, epochs before it is obvious
    in the reconstructions.

Mixed precision is deliberately not used: ``F.binary_cross_entropy`` is not
autocast-safe, and this model is small enough that fp32 costs little.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless-safe: Rangpur has no display

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import Timer, describe_device, get_device, save_fig, set_seed
from part4_recognition.oasis import OASIS_ROOT
from part4_recognition.task1_vae_oasis.dataset import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_WORKERS,
    get_dataloaders,
)
from part4_recognition.task1_vae_oasis.modules import LATENT_DIM, VAE, vae_loss

EPOCHS = 50
LEARNING_RATE = 1e-3
BETA = 1.0
WARMUP_EPOCHS = 10
ACTIVE_UNIT_THRESHOLD = 0.01
PROGRESS_EVERY = 5

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(REPO_ROOT, "outputs", "part4", "vae.pth")


def beta_at(step: int, steps_per_epoch: int, beta: float, warmup_epochs: int) -> float:
    """Linear KL warm-up, measured in optimiser steps for a smooth ramp."""
    if warmup_epochs <= 0:
        return beta
    return beta * min(1.0, step / (warmup_epochs * steps_per_epoch))


def train_one_epoch(model, loader, optimiser, device, step, steps_per_epoch, args):
    """One pass over the training set. Returns per-sample means and the new step."""
    model.train()
    totals = {"recon": 0.0, "kl": 0.0}
    n_seen = 0

    for images in loader:
        images = images.to(device, non_blocking=True)
        beta = beta_at(step, steps_per_epoch, args.beta, args.warmup_epochs)

        recon, mu, logvar = model(images)
        loss, reconstruction, kl = vae_loss(recon, images, mu, logvar, beta=beta)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        batch = images.size(0)
        totals["recon"] += reconstruction.item() * batch
        totals["kl"] += kl.item() * batch
        n_seen += batch
        step += 1

    return {key: value / n_seen for key, value in totals.items()}, step, beta


@torch.no_grad()
def evaluate(model, loader, device, beta: float):
    """Validation ELBO terms and the number of active latent units.

    ``model.eval()`` switches BatchNorm to its running statistics, but the
    forward pass still samples z; the reconstruction term is therefore a
    one-sample Monte Carlo estimate, which is the standard way to report it.
    """
    model.eval()
    totals = {"recon": 0.0, "kl": 0.0}
    n_seen = 0
    means = []

    for images in loader:
        images = images.to(device, non_blocking=True)
        recon, mu, logvar = model(images)
        _, reconstruction, kl = vae_loss(recon, images, mu, logvar, beta=beta)
        batch = images.size(0)
        totals["recon"] += reconstruction.item() * batch
        totals["kl"] += kl.item() * batch
        n_seen += batch
        means.append(mu.cpu())

    variance = torch.cat(means).var(dim=0)
    active = int((variance > ACTIVE_UNIT_THRESHOLD).sum())
    result = {key: value / n_seen for key, value in totals.items()}
    result["active_units"] = active
    return result


@torch.no_grad()
def save_progress_grid(model, fixed_batch, epoch: int, n: int = 8) -> str:
    """Inputs above reconstructions, from the same fixed batch every time.

    Using one fixed batch for the whole run is what makes the grids comparable
    across epochs — this is the "evidence of training" the task asks for.
    """
    model.eval()
    recon = model.decoder(model.encoder(fixed_batch[:n])[0])  # decode the mean, no sampling
    pairs = torch.cat([fixed_batch[:n], recon]).cpu().squeeze(1).numpy()

    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    for ax, image in zip(axes.flat, pairs):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks(())
        ax.set_yticks(())
    axes[0, 0].set_ylabel("input", fontsize=10)
    axes[1, 0].set_ylabel("reconstruction", fontsize=10)
    fig.suptitle(f"VAE reconstructions — epoch {epoch}", fontsize=12)
    fig.tight_layout()
    path = save_fig(fig, f"epoch_{epoch:03d}", "part4/vae_progress")
    plt.close(fig)
    return path


def plot_history(history: dict, args) -> str:
    """Reconstruction, KL, and active units against epoch."""
    epochs = np.arange(1, len(history["train_recon"]) + 1)
    fig, (ax_r, ax_k, ax_a) = plt.subplots(1, 3, figsize=(16, 4.5))

    ax_r.plot(epochs, history["train_recon"], label="train")
    ax_r.plot(epochs, history["val_recon"], label="validate")
    ax_r.set_title("Reconstruction (BCE, summed over pixels)")
    ax_r.set_xlabel("epoch")
    ax_r.grid(alpha=0.3)
    ax_r.legend(fontsize=9)

    ax_k.plot(epochs, history["train_kl"], label="train")
    ax_k.plot(epochs, history["val_kl"], label="validate")
    if args.warmup_epochs > 0:
        ax_k.axvspan(0.5, min(args.warmup_epochs, len(epochs)) + 0.5, color="0.9",
                     label="beta warm-up")
    ax_k.set_title("KL divergence (summed over latent dims)")
    ax_k.set_xlabel("epoch")
    ax_k.grid(alpha=0.3)
    ax_k.legend(fontsize=9)

    ax_a.step(epochs, history["active_units"], where="mid", color="C2")
    ax_a.set_ylim(-0.2, args.latent_dim + 0.2)
    ax_a.set_title(f"Active latent units (of {args.latent_dim})")
    ax_a.set_xlabel("epoch")
    ax_a.grid(alpha=0.3)

    fig.suptitle(
        f"VAE on OASIS — latent {args.latent_dim}, beta {args.beta}, "
        f"{args.warmup_epochs}-epoch warm-up", fontsize=13,
    )
    fig.tight_layout()
    path = save_fig(fig, "vae_loss", "part4")
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE on OASIS brain slices")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--root", default=OASIS_ROOT, help="OASIS dataset root")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N slices of each split (smoke tests)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = get_device()
    print("Device:", describe_device(device))

    with Timer("load") as load_timer:
        train_loader, val_loader, _ = get_dataloaders(
            args.batch_size, args.image_size, args.root, args.limit, args.num_workers
        )
    print(f"Data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} validate "
          f"slices at {args.image_size}x{args.image_size}, loaded in {load_timer.seconds:.1f} s")

    model = VAE(args.latent_dim, args.image_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: VAE, latent {args.latent_dim}, {n_params:,} parameters")
    print(f"Loss: BCE + beta*KL, beta {args.beta} after a {args.warmup_epochs}-epoch warm-up\n")

    fixed_batch = next(iter(val_loader))[:8].to(device)
    steps_per_epoch = len(train_loader)
    history = {key: [] for key in
               ("train_recon", "train_kl", "val_recon", "val_kl", "active_units", "beta")}
    step = 0

    with Timer("training") as timer:
        for epoch in range(1, args.epochs + 1):
            with Timer() as epoch_timer:
                train_stats, step, beta = train_one_epoch(
                    model, train_loader, optimiser, device, step, steps_per_epoch, args
                )
                # Validation uses the target beta, not the warm-up value, so the
                # validation ELBO means the same thing in every epoch.
                val_stats = evaluate(model, val_loader, device, beta=args.beta)

            history["train_recon"].append(train_stats["recon"])
            history["train_kl"].append(train_stats["kl"])
            history["val_recon"].append(val_stats["recon"])
            history["val_kl"].append(val_stats["kl"])
            history["active_units"].append(val_stats["active_units"])
            history["beta"].append(beta)

            print(f"  epoch {epoch:>3}/{args.epochs}  beta {beta:.2f}  "
                  f"recon {train_stats['recon']:9.1f} / {val_stats['recon']:9.1f}  "
                  f"KL {train_stats['kl']:6.2f} / {val_stats['kl']:6.2f}  "
                  f"active {val_stats['active_units']}/{args.latent_dim}  "
                  f"{epoch_timer.seconds:.1f} s")

            if epoch == 1 or epoch % PROGRESS_EVERY == 0 or epoch == args.epochs:
                save_progress_grid(model, fixed_batch, epoch)

    final = history["val_recon"][-1] + args.beta * history["val_kl"][-1]
    print(f"\nTraining time: {timer.seconds:.1f} s")
    print(f"Final validation: -ELBO {final:.1f} = recon {history['val_recon'][-1]:.1f} "
          f"+ {args.beta} x KL {history['val_kl'][-1]:.2f}; "
          f"active units {history['active_units'][-1]}/{args.latent_dim}")
    if history["active_units"][-1] == 0:
        print("  WARNING: no active latent units — the posterior has collapsed.")

    print(" ", plot_history(history, args))
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "image_size": args.image_size,
            "beta": args.beta,
            "epochs": args.epochs,
            "history": history,
        },
        CHECKPOINT,
    )
    print(" ", CHECKPOINT)


if __name__ == "__main__":
    main()
