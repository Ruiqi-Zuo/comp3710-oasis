# Task 1 — Variational Autoencoder on OASIS (max 3 of 7 marks)

Construct a VAE of brain MR images from the Preprocessed OASIS dataset.

## Requirements

* Train the model.
* **Visualise the resulting manifold** — either decode a 2D latent grid, or
  reduce a higher-dimensional latent space with UMAP.

Both parts are needed for full marks; a trained model with no manifold plot is
an incomplete answer.

## Method

### Data

The 9664 training slices, downsampled from 256x256 to 128x128 with antialiased
bilinear filtering and kept in [0, 1]. Plain bilinear sampling at a factor of
two skips every other pixel and aliases fine structure such as the cortical
folds. The 1120 validation slices track the loss during training; the 544 test
slices are held back for the figures. The split is by subject, so every figure
is drawn from brains the model never saw.

### Model

```
encoder  1x128x128 -> [4x4 conv, stride 2, BN, LeakyReLU] x4 -> 256x8x8
         -> Linear(16384 -> 512) -> mu, logvar  (Linear(512 -> latent) each)

decoder  z -> Linear(latent -> 512) -> Linear(512 -> 16384) -> 256x8x8
         -> [4x4 transposed conv, stride 2, BN, ReLU] x3
         -> 4x4 transposed conv -> 1x128x128 -> sigmoid
```

The latent space is 2-dimensional by default, so the manifold can be decoded
and plotted directly. `--latent-dim` raises it, in which case the embedding is
drawn with t-SNE instead.

Transposed convolutions use kernel 4 with stride 2. A kernel that divides the
stride evenly gives every output pixel the same number of kernel taps; kernel 3
with stride 2 does not, and produces checkerboard artefacts.

### Loss

The negative ELBO: binary cross entropy **summed over pixels**, plus beta times
the closed-form KL divergence to N(0, I) **summed over latent dimensions**, both
then averaged over the batch.

The summing matters. A 128x128 image has 16,384 pixels and the latent has 2
dimensions; averaging the reconstruction over pixels would shrink it by four
orders of magnitude relative to the KL, and the cheapest way to lower the loss
would become ignoring the image entirely. `tests/test_vae.py` pins this down.

### Guarding against posterior collapse

* **KL warm-up.** beta rises linearly from 0 to 1 over the first 10 epochs, so
  the decoder learns to use the latent code before the prior starts pulling the
  posterior onto itself.
* **Active units.** Each epoch counts the latent dimensions whose posterior mean
  varies across the validation set by more than 0.01 (Burda et al., 2016). A
  collapsed dimension carries no information, so collapse appears here as a
  number, before it is visible in the reconstructions.

### Training

Adam at 1e-3, batch 64, 50 epochs, fp32. Mixed precision is not used:
`F.binary_cross_entropy` is not autocast-safe, and the model is small.

## Run

```bash
# on Rangpur: trains, then writes all figures
sbatch -J vae scripts/oasis.slurm vae

# or directly
python -m part4_recognition.task1_vae_oasis.train
python -m part4_recognition.task1_vae_oasis.predict

# locally, against the synthetic tree (no real data needed)
python -m tests.fake_oasis ~/fake_oasis
python -m part4_recognition.task1_vae_oasis.train --root ~/fake_oasis --image-size 64 --epochs 3
python -m part4_recognition.task1_vae_oasis.predict --root ~/fake_oasis --grid 10
```

## Results

Trained on Rangpur (NVIDIA A100-PCIE-40GB), 50 epochs, latent dimension 2.

| Metric | Value |
|---|---|
| Final validation -ELBO | 4253.3 |
| of which reconstruction (BCE, summed over 16,384 pixels) | 4246.7 |
| of which KL | 6.54 |
| Training reconstruction, final epoch | 4149.2 |
| Active latent units | **2 / 2** at every epoch after the warm-up |
| Test posterior means inside the grid extent | 65.3% |
| Training time (A100) | 98.2 s (about 1.8 s per epoch) |

**No posterior collapse.** Both latent dimensions stay active and the KL settles
at about 6.5 nats rather than decaying towards zero.

**Converged, not overfitting.** Training reconstruction improved by under 0.2%
over the last 15 epochs (4157.6 to 4149.2) while validation held flat around
4240, a gap of 2.3%. With two latent dimensions the bottleneck, not the number
of epochs, limits the reconstruction.

**The grid covers less of the data than the prior would suggest.** The manifold
grid spans the central 90% of the prior on each axis, so if the posterior means
followed N(0, I), 0.9 x 0.9 = 81% of them would fall inside it. Only 65.3% do:
the encoder spreads its means more widely than the prior, and about a third of
the test slices are encoded outside the plotted region.

| Artefact | Path |
|----------|------|
| Loss curves (reconstruction / KL / active units) | `outputs/part4/vae_loss.png` |
| Reconstruction grids during training | `outputs/part4/vae_progress/epoch_*.png` |
| Test reconstructions | `outputs/part4/vae_reconstructions.png` |
| Latent manifold (2D grid) | `outputs/part4/vae_manifold.png` |
| Test set in latent space, coloured by slice | `outputs/part4/vae_latent_embedding.png` |
| Samples from the prior | `outputs/part4/vae_samples.png` |

## Questions to prepare

* **Why is it *variational*?** The encoder outputs a distribution over latent
  codes rather than a point, and training maximises a lower bound (the ELBO) on
  the data likelihood.
* **What does the KL term do?** It pulls each posterior towards N(0, I), which
  keeps the latent space continuous and sampleable. Without it you have a plain
  autoencoder with a latent space full of holes — decoding a random z gives
  noise.
* **Why the reparameterisation trick?** Sampling is not differentiable; moving
  the randomness into an input eps makes mu and sigma trainable by backprop.
* **What is posterior collapse?** The KL term drives the latents to be ignored
  and the decoder outputs an average brain. Watch for the KL going to zero — a
  beta warm-up is the usual fix. Here it is also tracked directly, as the
  active-unit count.
* **What should the manifold show?** Smooth, gradual anatomical change across
  the grid. Abrupt jumps or repeated identical tiles mean the latent space did
  not organise.
* **Why is the grid spaced by quantiles rather than evenly?** The prior is
  Gaussian, so most of the probability mass sits near zero. An even grid over z
  spends most of its tiles in the tails, where the encoder rarely puts anything
  and the decoder has learnt little.
* **Why colour the embedding by slice position?** It is a continuous anatomical
  variable the model is never given. If the embedding shows it as a smooth
  gradient, the latent space has organised around real anatomy rather than
  arbitrary features.
* **Why are the reconstructions blurry?** Two reasons. A 2D bottleneck cannot
  hold the detail of a 128x128 slice. And a pixel-wise likelihood averages over
  what it is uncertain about, which produces blur rather than choosing one sharp
  answer — a known property of VAEs, and the motivation for GANs in Task 3.

## Attribution

Developed with AI assistance (Claude, by Anthropic), as permitted by the course.
Active units as a collapse diagnostic follow Burda, Grosse and Salakhutdinov,
*Importance Weighted Autoencoders*, ICLR 2016.
