#!/bin/bash
# Retraining on artifact-free labels.
#
# The decisive run is B5_strict_lf: the current 0.4141 vs P2Rank's 0.4703 comes from a model that
# was TRAINED on labels containing glycerol, acetate and selenomethionine sites, so it does not
# measure the architecture. Stream B repeats the local-vs-global frame comparison under the new
# labels, to check that the paper's main technical claim survives the label correction.
cd /Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run
P=/Users/trongnhannguyenvu/.claude-science/conda/envs/ggnn/bin/python
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 GGNN_THREADS=5 PYTHONWARNINGS=ignore

run () {  # tag split graphdir seed
  echo "[$(date +%H:%M)] start $1"
  $P code/run_clean.py --train-split "$2" --graph-dir "$3" --seed "$4" \
     --loss combined_ft --tag "$1" --epochs 200 --patience 25 > "logs/$1.log" 2>&1
  echo "[$(date +%H:%M)] done $1"
}

( run B5_strict_lf    B5_train data_strict/combined        42
  run B5_strict_lf_s1 B5_train data_strict/combined         1 ) &
( run B4_strict_lf    B4_train data_strict/combined        42
  run B4_strict_gl    B4_train data_strict_global/combined 42 ) &
( run B5_strict_gl    B5_train data_strict_global/combined 42 ) &
wait
echo "[$(date +%H:%M)] STRICT QUEUE COMPLETE"
