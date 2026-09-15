# Task 2 — UNet segmentation of OASIS (lifts the cap to 5 of 7 marks)

UNet-based segmentation of brain MR images from the Preprocessed OASIS dataset.

## Requirements

* **DSC > 0.9 for all labels** — validated, not asserted.
* **Categorical (one-hot) output.** A non-categorical output loses marks even
  if the accuracy target is met.
* Visualise segmentation results to justify the DSC scores.
* **Run inference live on a test set during the demonstration.**
* Explain every element of training, the method, and the accuracies obtained.

## Method

### Data

Image/mask pairs at their native 256x256: 9664 training, 1120 validation and
544 test slices, split by subject (17 test subjects). Mask grey levels
0/85/170/255 are mapped to classes 0-3 through a lookup table that raises on any
other value.

Because the slices are never resized, a mask is never interpolated, so no
invented labels can appear. The only augmentation is a random left-right flip,
applied on the GPU to image and mask through one shared selection.

### Model

```
encoder   1 -> 32 -> 64 -> 128 -> 256      DoubleConv at 256, 128, 64, 32 px
          (DoubleConv = [conv3x3, BN, ReLU] x2; 2x2 max pool between stages)
bottleneck 256 -> 512                       at 16x16
decoder   2x2 transposed conv up, concatenate the encoder map, DoubleConv
          512 -> 256 -> 128 -> 64 -> 32     back to 256x256
head      1x1 conv, 32 -> 4                 per-pixel class logits
```

7.76 million parameters. Padded convolutions keep each decoder map the same
size as its skip partner, so the skips concatenate without cropping.

**Categorical output.** The network outputs four channels per pixel, one per
class; softmax turns them into a categorical distribution. The Dice loss is
computed against the one-hot encoded mask. `nn.CrossEntropyLoss` is given
class indices, but an index is only a sparse encoding of a one-hot vector — it
computes exactly the categorical cross entropy against the one-hot target.

### Loss

Cross entropy plus soft Dice. Cross entropy gives well-behaved per-pixel
gradients from the first step, when Dice is near zero and uninformative. Soft
Dice is computed per class over the batch and averaged **with equal weight per
class**, which is what counters the imbalance: background is 72% of the pixels
and CSF under 6%, but each carries a quarter of the Dice term.

### Metric

Hard Dice on the argmax prediction, with intersection and sizes **summed over
every slice in the split before dividing**. Averaging a per-slice Dice would
hit 0/0 on edge slices where a tissue is absent from both prediction and truth,
and whatever convention fills that gap would then distort the mean.
`tests/test_unet.py` checks that the accumulated result does not depend on how
the slices were batched.

### Training protocol

Adam at 1e-3 with a per-step cosine schedule, batch 16, 40 epochs, bf16
autocast. Every epoch is scored on the validation split, and the checkpoint
kept is the one with the best **worst-class** validation DSC — the requirement
is about all labels, so the weakest class is the quantity to maximise. The test
split is used exactly once, by `predict.py`, after training.

## Run

```bash
# on Rangpur: trains, then evaluates the test set and writes the figures
sbatch -J unet scripts/oasis.slurm unet

# the live demo, on the saved checkpoint
python -m part4_recognition.task2_unet_oasis.predict

# locally, against the synthetic tree (no real data needed)
python -m tests.fake_oasis ~/fake_oasis
python -m part4_recognition.task2_unet_oasis.train --root ~/fake_oasis --epochs 6 --base 16
python -m part4_recognition.task2_unet_oasis.predict --root ~/fake_oasis
```

## Results

Trained on Rangpur (NVIDIA A100-PCIE-40GB). Test set, DSC counts summed over
all 544 slices of the 17 held-out subjects:

| Label | DSC | |
|-------|-----|---|
| Background | 0.9993 | pass |
| CSF | **0.9658** | pass — lowest |
| Grey matter | 0.9666 | pass |
| White matter | 0.9799 | pass |

**Every label exceeds 0.9, and so does every label for every one of the 17 test
subjects.**

### Per subject

| Label | Mean over subjects | SD | Lowest | Highest | Worst class for |
|---|---|---|---|---|---|
| Background | 0.9993 | 0.0002 | 0.9989 | 0.9997 | 0 subjects |
| CSF | 0.9591 | **0.0221** | **0.9060** (subject 448) | 0.9843 | 6 subjects |
| Grey matter | 0.9662 | 0.0075 | 0.9485 (subject 449) | 0.9765 | **11 subjects** |
| White matter | 0.9798 | 0.0059 | 0.9686 | 0.9872 | 0 subjects |

"CSF is the hardest class" is only half true here, and the per-subject view is
what shows it:

* **Grey matter is the usual weak point** — the worst class for 11 of the 17
  brains — but it fails mildly and consistently (SD 0.0075, never below 0.948).
* **CSF is the least reliable.** Its spread across brains is three times that of
  grey matter, and it produces the two lowest scores anywhere in the table
  (0.9060 and 0.9145). It is the worst class less often than grey matter, but
  when it is, it is by the widest margin.
* **Pooled CSF (0.9658) is higher than the per-subject mean (0.9591).** Pooling
  counts weights each brain roughly by its CSF volume, so the brains with the
  most CSF are also the ones where CSF is segmented best. The closest call in
  the whole test set, subject 448 at 0.906, is therefore invisible in the pooled
  figure — which is exactly why the per-subject table is reported.

| Artefact | Path |
|----------|------|
| Loss and per-class validation DSC curves | `outputs/part4/unet_training.png` |
| Test segmentations with disagreement maps | `outputs/part4/unet_segmentations.png` |

## Questions to prepare

* **What do the skip connections do?** Pooling discards spatial precision. The
  skips concatenate the encoder feature map at each resolution into the decoder,
  so the network keeps deep semantic context *and* the fine boundary detail
  needed for a sharp mask. Without them the output is a blurry blob.
  `tests/test_unet.py` shows the output changes when the finest skip is removed.
* **Why Dice rather than plain accuracy?** The image is mostly background;
  pixel accuracy can exceed 95% while a small structure is missed entirely.
  Dice measures overlap per class and is far more sensitive to that failure.
  With 72% background, predicting background everywhere already scores 72%
  pixel accuracy.
* **Why combine CE with Dice?** CE gives well-behaved per-pixel gradients early
  in training when Dice is near zero and uninformative; Dice then optimises the
  metric you actually report.
* **Why nearest-neighbour interpolation for masks?** Bilinear resizing averages
  label indices and creates classes that do not exist. This implementation
  sidesteps the problem by never resizing at all.
* **Which class is hardest, and why?** It depends what "hardest" means, and the
  measured answer differs from the usual one. CSF has the lowest pooled DSC and
  by far the largest spread across brains — it is thin, low-contrast and the
  smallest class, so a brain with little of it is easy to get proportionally
  wrong. But grey matter is the worst class for more subjects (11 of 17): it
  sits between CSF and white matter, so every one of its boundaries is a
  tissue-tissue boundary. Report both rather than hiding either in the mean.
* **Why keep the checkpoint by the worst class rather than the mean?** The mean
  is dominated by background and the large tissues. A later epoch can raise the
  mean while CSF slips below 0.9, and that epoch fails the requirement.
* **Why sum Dice counts over the dataset instead of averaging per slice?**
  Per-slice Dice is undefined (0/0) for a class absent from a slice, which
  happens on every edge slice. Summing first gives one well-defined number per
  class.
* **How is the output "categorical"?** Four output channels, one probability
  per class per pixel after softmax, trained against one-hot targets. A
  single-channel output regressing the label value would not be.
* **Does a high test DSC mean it generalises?** The split is by subject, so the
  test brains were never seen in training. The per-subject table in
  `predict.py` shows whether the score holds for every one of them, not only on
  average.

## Attribution

Developed with AI assistance (Claude, by Anthropic), as permitted by the course.
UNet follows Ronneberger, Fischer and Brox, *U-Net: Convolutional Networks for
Biomedical Image Segmentation*, MICCAI 2015, with padded convolutions and batch
normalisation. Soft Dice loss as in Milletari, Navab and Ahmadi, *V-Net*, 3DV 2016.
