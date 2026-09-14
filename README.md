# COMP3710 Lab 2 Part 4 — Brain MRI recognition on OASIS

Deep learning on the Preprocessed OASIS brain MRI dataset, for Part 4 of
COMP3710 Lab 2 at the University of Queensland.

| Task | Model | Goal | Status |
|------|-------|------|--------|
| [1](part4_recognition/task1_vae_oasis/README.md) | Variational autoencoder | Train, and visualise the learned latent manifold | not started |
| [2](part4_recognition/task2_unet_oasis/README.md) | UNet | Segment brain tissue with DSC > 0.9 for **every** label | not started |
| [3](part4_recognition/task3_gan_oasis/README.md) | GAN | Generate realistic, distinct brains | deferred |

Tasks 1 and 2 together are the "medium" tier (up to 5 of 7 marks). Task 3 will
be decided once Task 2 meets its target. Marking criteria and difficulty tiers
are in [part4_recognition/README.md](part4_recognition/README.md).

## Data

The dataset lives read-only on the Rangpur cluster and is read in place — it is
never copied into this repository or a home directory.

Surveyed with `scripts/probe_oasis.py`:

| | |
|---|---|
| Location | `/home/groups/comp3710/OASIS/` |
| Images | `keras_png_slices_{train,validate,test}/case_XXX_slice_Y.nii.png` |
| Masks | `keras_png_slices_seg_{train,validate,test}/seg_XXX_slice_Y.nii.png` |
| Split | 9664 train / 1120 validate / 544 test slices |
| Format | 256 x 256 greyscale PNG, uint8 |
| Labels | 4 classes stored as pixel values 0, 85, 170, 255 (class index = value / 85) |

The split is **by subject** — cases 001–401 train, 402–440 validate, 441–457
test — so no brain contributes slices to more than one split.

Pixel share of each label, from 300 masks sampled across the training set:

| Label value | Class index | Share of pixels |
|---|---|---|
| 0 | 0 | 72.27% |
| 85 | 1 | 5.61% |
| 170 | 2 | 11.41% |
| 255 | 3 | 10.71% |

The smallest class covers under 6% of pixels while background covers over 70%.
That imbalance is why Task 2 cannot rely on cross entropy alone.

## Layout

```
.
├── common/                 # device selection, seeding, CUDA-safe timing, plotting
├── part4_recognition/
│   ├── oasis.py            # shared OASIS dataset access
│   ├── task1_vae_oasis/    # VAE + latent manifold
│   ├── task2_unet_oasis/   # UNet segmentation
│   └── task3_gan_oasis/    # GAN brain generation
├── scripts/                # SLURM job script and the dataset survey tool
├── tests/                  # tests on a synthetic OASIS tree (no real data needed)
├── data/                   # git-ignored
└── outputs/                # checkpoints, figures, logs (git-ignored)
```

Each task follows the COMP3710 report convention: `modules.py` (model),
`dataset.py` (data), `train.py` (training), `predict.py` (inference), and a
`README.md` covering the problem, method and results.

## Tests

The real dataset cannot leave Rangpur, so the loaders are tested against a small
synthetic tree with the same file names, split structure and label encoding
(`tests/fake_oasis.py`). These run anywhere, in a few seconds:

```bash
python -m pytest tests        # or, without pytest:
python -m tests.test_oasis
```

## Running on Rangpur

One-time environment, inside a CPU node (`srun --partition=cpu --time=00:30:00 --pty bash`):

```bash
source $HOME/miniconda3/bin/activate
conda create -n torch python=3.11 pip -y     # skip if it already exists
conda activate torch
pip install --no-cache-dir -r requirements.txt
```

Then from the repository root:

```bash
sbatch scripts/oasis.slurm         # edit the last lines to choose a task
squeue --me
tail -f outputs/slurm-<JOBID>.out  # substitute the number sbatch printed
```

All commands assume the repository root as the working directory, so that
`common` and `part4_recognition` import as packages.

## Attribution

This project was developed with AI assistance (Claude, by Anthropic), which the
course permits provided it is cited. Commits made with that assistance carry a
`Co-Authored-By` trailer. Sources for any adapted code are cited in the README of
the task that uses them.
