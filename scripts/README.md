# Scripts

| Script | Purpose |
|---|---|
| `probe_oasis.py` | Survey a dataset directory: layout, file formats, image sizes, and a pixel census of label masks. Read-only. |
| `oasis.slurm` | SLURM job that trains one Part 4 task on a GPU node and then runs its predict script. The task is a command-line argument. |

## probe_oasis.py

```bash
srun --partition=cpu --time=00:05:00 python scripts/probe_oasis.py /home/groups/comp3710/OASIS
```

Run it on a compute node, not the login node. `srun` inherits the activated conda
environment, so no interactive shell is needed.

## oasis.slurm

Submit from the repository root — the job uses relative paths:

```bash
sbatch -J vae  scripts/oasis.slurm vae
sbatch -J unet scripts/oasis.slurm unet
```

The task is passed as an argument, so nothing needs editing on the cluster.
`-J` only names the job in `squeue`. With no argument or an unknown one, the job
prints its usage and exits immediately.

The job writes to `outputs/`, which must already exist when the job is
submitted: SLURM does not create the directory, and a missing one makes the job
fail without writing any log. The repository ships `outputs/.gitkeep` for that
reason.

### Why the job requests look the way they do

A job the `comp3710` partition will not admit is **not rejected** — it is
accepted and then sits in the queue forever with reason `(PartitionConfig)`.
Two things cause that here, both found by submitting one-variable test jobs:

* **`--account=comp3710` is mandatory.** The partition has
  `AllowAccounts=comp3710`, and jobs are not charged to that account by
  default. Without the flag, even a bare `sbatch -p comp3710 --gres=gpu:1`
  never starts.
* **No `--mem`.** The partition registers about 1 MB of memory per node
  (`TRES=...,mem=10M` across 10 nodes), so memory is not a real schedulable
  resource and a request for it can only make the job unsatisfiable.

Each node is one A100 with 8 CPUs, so `--cpus-per-task` should stay at 8 or
below. After submitting, `squeue --me` should show `Priority`, `Resources` or
`RUNNING`; `PartitionConfig` means a request is still wrong. Inspect the limits
with `scontrol show partition comp3710`.
