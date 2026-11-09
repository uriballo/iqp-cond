#!/bin/bash
# HTCondor job for the precision study on one L40S.
# Usage: job.sh CPUS RUN_ID smoke
#        job.sh CPUS RUN_ID prepare PLAN
#        job.sh CPUS RUN_ID train PLAN INDEX
#        job.sh CPUS RUN_ID screen ARGS...   (arguments of screening.py)
set -uo pipefail
CPUS=$1 RUN=$2 MODE=$3
shift 3
ROOT=${ROOT:-/data/cvcqml/scratch/$USER/iqp-precision}
PY=$ROOT/env-gpu/bin/python
STUDY=$ROOT/tree/experiments/precision_study
LOGS=$STUDY/runs/logs
mkdir -p "$LOGS"
exec >>"$LOGS/$MODE-$RUN.log" 2>&1

# XLA sizes its thread pool from the affinity mask; use only the requested cores.
if [ "$(nproc)" -gt "$CPUS" ] && [ -z "${IQP_PINNED:-}" ]; then
    cores=$($PY -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:$CPUS])))")
    IQP_PINNED="pinned to $cores" exec taskset -c "$cores" "$0" "$CPUS" "$RUN" "$MODE" "$@"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$ROOT/tree:$ROOT/tree/src
cd "$ROOT/tree"
echo "== $(date -Is) host $HOSTNAME, mode $MODE $*, run $RUN, ${IQP_PINNED:-affinity from HTCondor}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
$PY -c "import jax, sys; print('jax', jax.__version__, jax.devices()); sys.exit(jax.default_backend() != 'gpu')" \
    || { echo "no GPU visible to JAX; stopping"; exit 1; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used --format=csv,nounits -lms 1000 \
    > "$LOGS/gpu-$MODE-$RUN.csv" &
SAMPLER=$!

PLANS=experiments/precision_study/plans
case $MODE in
smoke)
    echo "== $(date -Is) stage 1: prepare teacher caches" &&
    $PY -m experiments.precision_study.matrix $PLANS/smoke.yaml --prepare &&
    echo "== $(date -Is) stage 2: screening at 12 qubits" &&
    $PY -m experiments.precision_study.screening $STUDY/runs/smoke-screening \
        --n-qubits 12 --samples 1000 16000 --draws 16 --plan $PLANS/smoke.yaml &&
    echo "== $(date -Is) stage 3: training matrix" &&
    $PY -m experiments.precision_study.matrix $PLANS/smoke.yaml --all ;;
prepare) $PY -m experiments.precision_study.matrix "$1" --prepare ;;
train) $PY -m experiments.precision_study.matrix "$1" --index "$2" ;;
screen) $PY -m experiments.precision_study.screening "$@" ;;
*) echo "unknown mode $MODE"; false ;;
esac
status=$?
kill $SAMPLER 2>/dev/null
echo "== $(date -Is) exit $status"
exit $status
