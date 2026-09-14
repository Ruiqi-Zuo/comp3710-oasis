# Scripts

| Script | Purpose |
|---|---|
| `probe_oasis.py` | Survey a dataset directory: layout, file formats, image sizes, and a pixel census of label masks. Read-only. |
| `oasis.slurm` | SLURM job that runs one Part 4 task on a GPU node. Choose the task by editing its last lines. |

## probe_oasis.py

```bash
srun --partition=cpu --time=00:05:00 python scripts/probe_oasis.py /home/groups/comp3710/OASIS
```

Run it on a compute node, not the login node. `srun` inherits the activated conda
environment, so no interactive shell is needed.

## oasis.slurm

Submit from the repository root — the job uses relative paths:

```bash
sbatch scripts/oasis.slurm
```

The job writes to `outputs/`, which must already exist when the job is
submitted: SLURM does not create the directory, and a missing one makes the job
fail without writing any log. The repository ships `outputs/.gitkeep` for that
reason.

The partition, CPU, memory and time requests must fit the limits of the
`comp3710` partition. A job that exceeds them is not rejected; it sits in the
queue forever with reason `(PartitionConfig)`. Check the limits with
`scontrol show partition comp3710`.
