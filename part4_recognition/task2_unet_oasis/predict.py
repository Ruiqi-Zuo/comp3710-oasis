"""UNet inference — this must be run live on a test MRI during the demo.

    python -m part4_recognition.task2_unet_oasis.predict

The lab is explicit: you have to run inference on a test set at the demo and
visualise segmentations that justify the reported DSC.

This script reports three levels of evidence:

  * per-class DSC over the whole test set — the number the requirement is about;
  * the same per test subject, so one well-segmented majority cannot hide a
    brain the model handles badly;
  * figures of individual slices with their own per-class DSC, including a map
    of exactly which pixels disagree with the ground truth.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless-safe: Rangpur has no display

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from common import Timer, describe_device, get_device, save_fig
from part4_recognition.oasis import CLASS_NAMES, N_SEG_CLASSES, OASIS_ROOT
from part4_recognition.task2_unet_oasis.dataset import get_dataloaders
from part4_recognition.task2_unet_oasis.modules import UNet, dice_counts, dice_from_counts
from part4_recognition.task2_unet_oasis.train import CHECKPOINT, TARGET_DSC, autocast_dtype

LABEL_COLOURS = ListedColormap(["black", "#4fa3e0", "#9e9e9e", "#f5f5f5"])


def load_model(device):
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model = UNet(n_classes=N_SEG_CLASSES, base=checkpoint["base"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def subject_of(name: str) -> str:
    """``case_441_slice_17.nii.png`` -> ``441``."""
    return re.search(r"case_(\d+)_slice", name).group(1)


@torch.no_grad()
def predict_all(model, loader, device):
    """Predicted label maps for the whole loader, plus the inference time."""
    dtype = autocast_dtype(device)
    predictions = []
    with Timer() as timer:
        for images, _ in loader:
            with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                logits = model(images.to(device, non_blocking=True))
            predictions.append(logits.float().argmax(dim=1).cpu())
    return torch.cat(predictions), timer.seconds


@torch.no_grad()
def evaluate_test_set(predictions: torch.Tensor, loader):
    """Return the per-class DSC over the whole test set, and per subject.

    DSC counts are summed over every slice before dividing, which is the same
    definition used for validation during training.
    """
    dataset = loader.dataset
    masks = torch.stack([dataset[i][1] for i in range(len(dataset))])

    # dice_counts takes logits and argmaxes them; one-hot predictions argmax to themselves.
    as_logits = torch.nn.functional.one_hot(predictions, N_SEG_CLASSES).permute(0, 3, 1, 2).float()
    overall = dice_from_counts(*dice_counts(as_logits, masks)).numpy()

    per_subject_counts = defaultdict(lambda: [torch.zeros(N_SEG_CLASSES) for _ in range(3)])
    for index, name in enumerate(dataset.names):
        counts = dice_counts(as_logits[index:index + 1], masks[index:index + 1])
        for total, part in zip(per_subject_counts[subject_of(name)], counts):
            total += part
    per_subject = {
        subject: dice_from_counts(*counts).numpy()
        for subject, counts in sorted(per_subject_counts.items())
    }
    return overall, per_subject, masks


def slice_dsc_label(prediction: torch.Tensor, mask: torch.Tensor) -> str:
    """Per-class DSC for a single slice, with n/a where a class is absent from both."""
    parts = []
    for index, name in enumerate(("bg", "CSF", "GM", "WM")):
        predicted = prediction == index
        actual = mask == index
        denominator = predicted.sum() + actual.sum()
        if denominator == 0:
            parts.append(f"{name} n/a")
        else:
            parts.append(f"{name} {2 * (predicted & actual).sum().item() / denominator.item():.3f}")
    return "  ".join(parts)


def choose_slices(dataset, masks: torch.Tensor, n: int) -> list[int]:
    """One slice from each of ``n`` evenly spread test subjects.

    From each subject, the slice with the most brain tissue: edge slices that are
    almost entirely background would make the figure look better than the model
    is, not demonstrate it.
    """
    by_subject = defaultdict(list)
    for index, name in enumerate(dataset.names):
        by_subject[subject_of(name)].append(index)
    subjects = sorted(by_subject)
    picks = [subjects[i] for i in np.linspace(0, len(subjects) - 1, min(n, len(subjects))).astype(int)]
    tissue = (masks > 0).flatten(1).sum(dim=1)
    return [max(by_subject[s], key=lambda i: tissue[i].item()) for s in picks]


def plot_segmentations(dataset, predictions, masks, indices) -> str:
    """Four columns: input MRI, ground truth, prediction, and where they disagree."""
    fig, axes = plt.subplots(len(indices), 4, figsize=(13, 3.35 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, index in enumerate(indices):
        image = dataset[index][0].squeeze(0).numpy()
        truth = masks[index].numpy()
        predicted = predictions[index].numpy()
        wrong = predicted != truth

        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(truth, cmap=LABEL_COLOURS, vmin=0, vmax=N_SEG_CLASSES - 1, interpolation="nearest")
        axes[row, 2].imshow(predicted, cmap=LABEL_COLOURS, vmin=0, vmax=N_SEG_CLASSES - 1, interpolation="nearest")
        axes[row, 3].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(np.ma.masked_where(~wrong, wrong), cmap=ListedColormap(["red"]),
                            alpha=0.9, interpolation="nearest")

        axes[row, 0].set_ylabel(dataset.names[index].replace(".nii.png", ""), fontsize=9)
        axes[row, 2].set_title(slice_dsc_label(predictions[index], masks[index]), fontsize=8)
        axes[row, 3].set_title(f"{wrong.sum()} of {wrong.size} pixels differ "
                               f"({wrong.mean():.2%})", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks(())
            ax.set_yticks(())

    for ax, title in zip(axes[0], ("input MRI", "ground truth", "prediction", "disagreement")):
        ax.set_title(f"{title}\n{ax.get_title()}" if ax.get_title() else title, fontsize=10)

    legend = [Patch(facecolor=LABEL_COLOURS(i), edgecolor="0.4", label=name)
              for i, name in enumerate(CLASS_NAMES)]
    legend.append(Patch(facecolor="red", label="prediction != truth"))
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("UNet segmentation of held-out test subjects", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    path = save_fig(fig, "unet_segmentations", "part4")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="UNet inference on the OASIS test set")
    parser.add_argument("--root", default=OASIS_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n", type=int, default=4, help="slices to plot")
    args = parser.parse_args()

    device = get_device()
    print("Device:", describe_device(device))
    model, checkpoint = load_model(device)
    kept = "  ".join(f"{v:.4f}" for v in checkpoint["val_dsc"])
    print(f"Loaded {CHECKPOINT}\n  epoch {checkpoint['epoch']}, validation DSC {kept}")

    _, _, test_loader = get_dataloaders(root=args.root, limit=args.limit, num_workers=0)
    dataset = test_loader.dataset
    print(f"Test set: {len(dataset)} slices from "
          f"{len({subject_of(n) for n in dataset.names})} subjects")

    predictions, seconds = predict_all(model, test_loader, device)
    print(f"Inference: {seconds:.2f} s ({len(dataset) / seconds:,.0f} slices/s)")

    overall, per_subject, masks = evaluate_test_set(predictions, test_loader)

    print("\nTest-set DSC per class (counts summed over all slices)")
    for name, value in zip(CLASS_NAMES, overall):
        print(f"  {name:<13} {value:.4f}  {'PASS' if value > TARGET_DSC else 'FAIL'}")
    verdict = "all labels > 0.9" if overall.min() > TARGET_DSC else "target NOT met"
    print(f"  {'worst':<13} {overall.min():.4f}  {verdict}")

    print("\nPer test subject (worst class first)")
    header = "  ".join(f"{name[:5]:>6}" for name in CLASS_NAMES)
    print(f"  {'subject':>7}  {header}   worst")
    for subject, scores in sorted(per_subject.items(), key=lambda item: item[1].min()):
        print(f"  {subject:>7}  " + "  ".join(f"{v:6.4f}" for v in scores) + f"   {scores.min():.4f}")
    failing = [s for s, scores in per_subject.items() if scores.min() <= TARGET_DSC]
    print(f"  {len(per_subject) - len(failing)} of {len(per_subject)} subjects have every class > 0.9")

    indices = choose_slices(dataset, masks, args.n)
    print("\nSaved", plot_segmentations(dataset, predictions, masks, indices))


if __name__ == "__main__":
    main()
