# Task 3 — GAN brain generation (lifts the cap to 7 of 7 marks)

Realistic brain generation from the Preprocessed OASIS dataset using a GAN.

## Requirements

* Images must be suitably realistic — your instructor judges this, so check in
  early on whether your results qualify for full or partial marks.
* Provide evidence of training: generated images over time, loss plots.
* Results must look like **unique** brains. Mode collapse must be fully resolved.
* Full marks only for OASIS results; partial marks for MNIST/CelebA.

## Warning from the lab sheet

GANs have very chaotic convergence and are difficult to train. Attempt this only
if you are confident with deep learning. Start with MNIST, then CelebA, then
OASIS.

## Run

```bash
python -m part4_recognition.task3_gan_oasis.train
python -m part4_recognition.task3_gan_oasis.predict
```

## Results

| Artefact | Path |
|----------|------|
| Loss curves (G and D) | `outputs/part4/gan/losses.png` |
| Sample grids over epochs | `outputs/part4/gan/samples_epoch_*.png` |
| Final samples | `outputs/part4/gan/final_samples.png` |
| Latent interpolation | `outputs/part4/gan/interpolation.png` |

## Questions to prepare

* **What is the minimax game?** D is trained to separate real from generated
  images; G is trained to fool D. At the theoretical optimum G matches the data
  distribution and D outputs 0.5 everywhere.
* **How do you detect mode collapse?** Sample variance across a batch drops and
  the grid fills with near-identical images. Fixes: minibatch discrimination,
  one-sided label smoothing, spectral norm, or switching to WGAN-GP.
* **Why can loss curves look fine while the output is bad?** The losses are
  relative to an adversary that is itself changing — they measure the balance of
  the game, not sample quality. Judge with samples, and with FID if you have it.
* **How do you show the samples are not memorised training images?** Latent
  interpolation should be smooth, and you can compare each sample against its
  nearest training neighbour.
