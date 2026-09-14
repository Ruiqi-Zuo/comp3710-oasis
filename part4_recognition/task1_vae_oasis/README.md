# Task 1 — Variational Autoencoder on OASIS (max 3 of 7 marks)

Construct a VAE of brain MR images from the Preprocessed OASIS dataset.

## Requirements

* Train the model.
* **Visualise the resulting manifold** — either decode a 2D latent grid, or
  reduce a higher-dimensional latent space with UMAP.

Both parts are needed for full marks; a trained model with no manifold plot is
an incomplete answer.

## Run

```bash
python -m part4_recognition.task1_vae_oasis.train
python -m part4_recognition.task1_vae_oasis.predict
```

## Results

| Artefact | Path |
|----------|------|
| Loss curves (total / reconstruction / KL) | `outputs/part4/vae_loss.png` |
| Reconstructions | `outputs/part4/vae_reconstructions.png` |
| Latent manifold | `outputs/part4/vae_manifold.png` |

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
  beta warm-up is the usual fix.
* **What should the manifold show?** Smooth, gradual anatomical change across
  the grid. Abrupt jumps or repeated identical tiles mean the latent space did
  not organise.
