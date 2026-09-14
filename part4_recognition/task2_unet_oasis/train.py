"""Train the OASIS UNet.

    python -m part4_recognition.task2_unet_oasis.train
    python -m part4_recognition.task2_unet_oasis.train --epochs 2 --limit 64   # smoke test

Protocol, which is what makes the reported DSC trustworthy:

  * training slices train; validation slices choose the checkpoint; test slices
    are touched once, by ``predict.py``, after training is over;
  * the checkpoint kept is the one with the best **worst-class** validation
    DSC, because the requirement is "> 0.9 for ALL labels" — a checkpoint with a
    higher mean but a weaker CSF score is the worse model for this task;
  * DSC is accumulated over the whole validation set per class, not averaged
    per slice (see ``modules.dice_counts``).
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
from part4_recognition.oasis import CLASS_NAMES, N_SEG_CLASSES, OASIS_ROOT
from part4_recognition.task2_unet_oasis.dataset import (
    BATCH_SIZE,
    NUM_WORKERS,
    get_dataloaders,
    random_hflip,
)
from part4_recognition.task2_unet_oasis.modules import (
    CEDiceLoss,
    UNet,
    dice_counts,
    dice_from_counts,
)

EPOCHS = 40
LEARNING_RATE = 1e-3
TARGET_DSC = 0.9

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(REPO_ROOT, "outputs", "part4", "unet.pth")


def autocast_dtype(device) -> torch.dtype | None:
    """bf16 where supported, else fp16, else nothing on CPU.

    bf16 has fp32's exponent range, so gradients cannot underflow and no
    GradScaler is needed. The A100 supports it; fp16 with a scaler is the
    fallback for older GPUs.
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def train_one_epoch(model, loader, criterion, optimiser, scheduler, scaler, device, dtype, hflip):
    model.train()
    sums = {"loss": 0.0, "ce": 0.0, "dice": 0.0}
    n_seen = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if hflip:
            images, masks = random_hflip(images, masks)
        images = images.contiguous(memory_format=torch.channels_last)

        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            logits = model(images)
        loss, ce, dice = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        batch = images.size(0)
        sums["loss"] += loss.item() * batch
        sums["ce"] += ce.item() * batch
        sums["dice"] += dice.item() * batch
        n_seen += batch

    return {key: value / n_seen for key, value in sums.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, dtype):
    """Validation loss and per-class DSC accumulated over the whole loader."""
    model.eval()
    totals = [torch.zeros(N_SEG_CLASSES, device=device) for _ in range(3)]
    loss_sum = 0.0
    n_seen = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            logits = model(images)
        loss, _, _ = criterion(logits, masks)
        loss_sum += loss.item() * images.size(0)
        n_seen += images.size(0)
        for total, part in zip(totals, dice_counts(logits.float(), masks)):
            total += part

    return loss_sum / n_seen, dice_from_counts(*totals).cpu().numpy()


def plot_history(history: dict, best_epoch: int) -> str:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    dsc = np.array(history["val_dsc"])
    fig, (ax_loss, ax_dsc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validate")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross entropy + soft Dice")
    ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=9)

    for index, name in enumerate(CLASS_NAMES):
        ax_dsc.plot(epochs, dsc[:, index], label=name)
    ax_dsc.plot(epochs, dsc.min(axis=1), color="black", ls=":", lw=1.2, label="worst class")
    ax_dsc.axhline(TARGET_DSC, color="crimson", ls="--", lw=1.0, label=f"target {TARGET_DSC}")
    ax_dsc.axvline(best_epoch, color="0.5", ls="-.", lw=1.0, label=f"kept: epoch {best_epoch}")
    ax_dsc.set_xlabel("epoch")
    ax_dsc.set_ylabel("validation DSC")
    ax_dsc.set_title("Per-class validation Dice")
    ax_dsc.set_ylim(max(0.0, min(0.5, float(dsc.min()) - 0.05)), 1.0)
    ax_dsc.grid(alpha=0.3)
    ax_dsc.legend(fontsize=8, loc="lower right")

    fig.suptitle("UNet segmentation of OASIS", fontsize=13)
    fig.tight_layout()
    path = save_fig(fig, "unet_training", "part4")
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet to segment OASIS brain slices")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--base", type=int, default=32, help="width of the first UNet stage")
    parser.add_argument("--no-hflip", action="store_true", help="disable flip augmentation")
    parser.add_argument("--root", default=OASIS_ROOT)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N slices of each split (smoke tests)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = get_device()
    dtype = autocast_dtype(device)
    torch.backends.cudnn.benchmark = True
    print("Device:", describe_device(device))

    with Timer() as load_timer:
        train_loader, val_loader, _ = get_dataloaders(
            args.batch_size, args.root, args.limit, args.num_workers
        )
    print(f"Data: {len(train_loader.dataset)} train / {len(val_loader.dataset)} validate "
          f"slices, loaded in {load_timer.seconds:.1f} s")

    model = UNet(n_classes=N_SEG_CLASSES, base=args.base).to(device, memory_format=torch.channels_last)
    criterion = CEDiceLoss(N_SEG_CLASSES)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs * len(train_loader)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    n_params = sum(p.numel() for p in model.parameters())
    precision = {torch.bfloat16: "bf16", torch.float16: "fp16 + GradScaler", None: "fp32"}[dtype]
    print(f"Model: UNet base {args.base}, {n_params:,} parameters, {precision}")
    print(f"Loss: cross entropy + soft Dice; hflip {'off' if args.no_hflip else 'on'}\n")

    header = "  ".join(f"{name[:5]:>5}" for name in CLASS_NAMES)
    print(f"  {'epoch':>9}  {'train':>6}  {'val':>6}   {header}   worst")

    history = {"train_loss": [], "val_loss": [], "val_dsc": []}
    best_worst, best_epoch = -1.0, 0

    with Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            with Timer() as epoch_timer:
                train_stats = train_one_epoch(
                    model, train_loader, criterion, optimiser, scheduler, scaler,
                    device, dtype, hflip=not args.no_hflip,
                )
                val_loss, val_dsc = evaluate(model, val_loader, criterion, device, dtype)

            history["train_loss"].append(train_stats["loss"])
            history["val_loss"].append(val_loss)
            history["val_dsc"].append(val_dsc.tolist())

            worst = float(val_dsc.min())
            marker = ""
            if worst > best_worst:
                best_worst, best_epoch = worst, epoch
                os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "base": args.base,
                        "epoch": epoch,
                        "val_dsc": val_dsc.tolist(),
                    },
                    CHECKPOINT,
                )
                marker = "  *saved"

            scores = "  ".join(f"{value:.3f}" for value in val_dsc)
            print(f"  {epoch:>4}/{args.epochs:<4}  {train_stats['loss']:.4f}  {val_loss:.4f}   "
                  f"{scores}   {worst:.3f}  {epoch_timer.seconds:.0f} s{marker}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    checkpoint["history"] = history
    torch.save(checkpoint, CHECKPOINT)

    kept = np.array(history["val_dsc"][best_epoch - 1])
    print(f"\nTraining time: {timer.seconds:.1f} s")
    print(f"Kept epoch {best_epoch}: validation DSC per class")
    for name, value in zip(CLASS_NAMES, kept):
        print(f"  {name:<13} {value:.4f}  {'PASS' if value > TARGET_DSC else 'FAIL'}")
    print(f"  {'worst':<13} {kept.min():.4f}  "
          f"{'all labels > 0.9' if kept.min() > TARGET_DSC else 'target NOT met'}")
    print(" ", plot_history(history, best_epoch))
    print(" ", CHECKPOINT)


if __name__ == "__main__":
    main()
