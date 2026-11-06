#!/bin/bash
# Seed replicates: two extra seeds per configuration.
#
# Every architecture conclusion so far rests on a single run per configuration, so differences of
# 1-3% cannot be separated from seed noise. Two extra seeds per configuration give a mean and a
# spread for the two claims that matter: the local-frame gain (B4_localframe vs B4_global, same
# 1850 proteins, same parameters, only the edge-feature frame differs) and the headline number
# (B5_localframe on 4254 proteins).
#
# Three streams x 4 torch threads = 12 of 15 cores, leaving headroom for the pocket sweep.

cd "$(cd "$(dirname "$0")/.." && pwd)"   # repository root
PY=${GGNN_PYTHON:-python3}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 GGNN_THREADS=4 PYTHONWARNINGS=ignore

run () {  # tag split graphdir seed
  echo "[$(date +%H:%M)] start $1"
  $PY code/run_clean.py --train-split "$2" --graph-dir "$3" --seed "$4" \
      --loss combined_ft --tag "$1" --epochs 200 --patience 25 \
      > "logs/$1.log" 2>&1
  echo "[$(date +%H:%M)] done  $1 (exit $?)"
}

( run B4_localframe_s1 B4_train data_localframe/combined 1
  run B4_localframe_s2 B4_train data_localframe/combined 2 ) > logs/_streamA.log 2>&1 &

( run B4_global_s1     B4_train data_relabeled/combined  1
  run B4_global_s2     B4_train data_relabeled/combined  2 ) > logs/_streamB.log 2>&1 &

( run B5_localframe_s1 B5_train data_localframe/combined 1
  run B5_localframe_s2 B5_train data_localframe/combined 2 ) > logs/_streamC.log 2>&1 &

wait
echo "[$(date +%H:%M)] ALL SEED RUNS COMPLETE"
