# Running the precision study at PIC

All jobs run on one L40S (gpu09-gpu15) so every measurement shares its
rounding behaviour. Only the account holder submits jobs.

## Sync and install (once per code change)

From the laptop, sync the worktree (code only; results stay at PIC until
synced back):

    git -C <worktree> rev-parse HEAD > <worktree>/GIT_SHA
    rsync -a --exclude __pycache__ --exclude .venv --exclude .git --exclude .claude \
        --exclude prototypes --exclude 'experiments/precision_study/runs' \
        --exclude 'experiments/precision_study/cache' \
        <worktree>/ oballo@ui.pic.es:/data/cvcqml/scratch/oballo/iqp-precision/tree/

`GIT_SHA` records the synced commit in every run's provenance, since the tree
at PIC is not a git checkout.

On ui.pic.es, the first time:

    bash /data/cvcqml/scratch/oballo/iqp-precision/tree/experiments/precision_study/pic/setup.sh

## Smoke run

    cd /data/cvcqml/scratch/oballo/iqp-precision/tree/experiments/precision_study/pic
    mkdir -p jobs && condor_submit smoke.sub

It prepares the teacher caches for 12 and 64 qubits, screens 12 qubits, and
trains `plans/smoke.yaml` (36 short runs: 7 recipe-variant runs and 2 fp64 baselines
per size and mask arm). Logs, GPU samples and timings go to
`runs/logs/`; runs to `runs/smoke/`; screening to `runs/smoke-screening/`.

## Bring results home

    rsync -a oballo@ui.pic.es:/data/cvcqml/scratch/oballo/iqp-precision/tree/experiments/precision_study/runs/ \
        <worktree>/experiments/precision_study/runs/
