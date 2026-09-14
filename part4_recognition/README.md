# Part 4 — Recognition (8 marks)

| Section | Task | Marks |
|---------|------|-------|
| 4.1 | edX short course: *Version Control for Teams using Git* | 1 |
| 4.4 | Recognition tasks below | 7 |

## Difficulty tiers

| Tier | Tasks | Max marks (of 7) |
|------|-------|------------------|
| Easy | Task 1 only | 3 |
| Medium | Tasks 1 and 2 | 5 |
| Hard | All three | 7 |

Medium is the recommended target — the report/project assessment later in the
course builds directly on these tasks.

## Hard requirement

> You must create a GitHub project in your own account for the tasks below with
> relevant commit logs **in addition to** your demonstration to receive ANY
> marks for this part.

The demonstrator may ask you to log into the account during the demo.

## Marking criteria

1. Code works, tasks complete, hosted correctly on GitHub, results presented and
   explained thoroughly — 1-5 marks depending on tasks completed.
2. Good project practice: README, documentation, meaningful commit messages — 1 mark.
3. Code commented, structured and designed for use by others — 1 mark.

That is **2 of 7 marks for engineering practice alone**, independent of model
quality. Commit as you go with real messages; do not squash the whole lab into
one "final" commit.

## Data

Preprocessed OASIS brain MRI, on Rangpur at `/home/groups/comp3710/`.
`oasis.py` holds the shared loading code; set `OASIS_ROOT` to override the path
when testing locally on a subset.

## Task layout

Each task directory follows the COMP3710 report convention:

* `modules.py` — model components
* `dataset.py` — data loading and preprocessing
* `train.py` — training, validation, plots
* `predict.py` — inference demonstration
* `README.md` — problem, method, results, usage
