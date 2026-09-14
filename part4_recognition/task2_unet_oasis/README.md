# Task 2 — UNet segmentation of OASIS (lifts the cap to 5 of 7 marks)

UNet-based segmentation of brain MR images from the Preprocessed OASIS dataset.

## Requirements

* **DSC > 0.9 for all labels** — validated, not asserted.
* **Categorical (one-hot) output.** A non-categorical output loses marks even
  if the accuracy target is met.
* Visualise segmentation results to justify the DSC scores.
* **Run inference live on a test set during the demonstration.**
* Explain every element of training, the method, and the accuracies obtained.

## Run

```bash
python -m part4_recognition.task2_unet_oasis.train
python -m part4_recognition.task2_unet_oasis.predict
```

## Results

| Label | DSC |
|-------|-----|
| Background | |
| CSF | |
| Grey matter | |
| White matter | |

All four must exceed 0.9.

## Questions to prepare

* **What do the skip connections do?** Pooling discards spatial precision. The
  skips concatenate the encoder feature map at each resolution into the decoder,
  so the network keeps deep semantic context *and* the fine boundary detail
  needed for a sharp mask. Without them the output is a blurry blob.
* **Why Dice rather than plain accuracy?** The image is mostly background;
  pixel accuracy can exceed 95% while a small structure is missed entirely.
  Dice measures overlap per class and is far more sensitive to that failure.
* **Why combine CE with Dice?** CE gives well-behaved per-pixel gradients early
  in training when Dice is near zero and uninformative; Dice then optimises the
  metric you actually report.
* **Why nearest-neighbour interpolation for masks?** Bilinear resizing averages
  label indices and creates classes that do not exist.
* **Which class is hardest, and why?** Usually CSF — thin, low-contrast, and the
  smallest by volume. Report it honestly rather than hiding it in the mean.
