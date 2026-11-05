#!/usr/bin/env python3
"""Check every headline number of the GGNN 2025 manuscript against the saved
result files in results/.

This script needs nothing but the Python standard library and the files that
ship with this repository: no PyTorch, no graph data, no model weights, no
network access. Clone the repository and run it.

    python code/check_reported_numbers.py

It exits 0 if every reported value matches its source file, 1 otherwise.
"""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
checks = []


def check(label, reported, actual, tol=5e-5):
    ok = actual is not None and abs(reported - actual) <= tol
    checks.append((ok, label, reported, actual))


def pct_change(new, old):
    return (new - old) / old * 100.0


# ---- Table 2: one model, three splits -------------------------------------
run = json.loads((RES / "B5_strict_lf.json").read_text())
ev = run["evaluations"]
TABLE2 = {
    "validation": dict(auprc=0.6696, auroc=0.9306, f1=0.6205,
                       mcc=0.5985, precision=0.6543, recall=0.5900),
    "T_leaky":    dict(auprc=0.8207, auroc=0.9776, f1=0.7582,
                       mcc=0.7437, precision=0.6949, recall=0.8342),
    "T_clean":    dict(auprc=0.4611, auroc=0.8647, f1=0.4458,
                       mcc=0.4291, precision=0.5671, recall=0.3673),
}
for split, row in TABLE2.items():
    for metric, reported in row.items():
        check(f"Table 2  {split:10s} {metric}", reported, ev[split][metric])

check("Table 2  n validation", 302, ev["validation"]["n_proteins"], 0)
check("Table 2  n homologous", 316, ev["T_leaky"]["n_proteins"], 0)
check("Table 2  n disjoint", 244, ev["T_clean"]["n_proteins"], 0)

# ---- Section 4.3: the homology cost ---------------------------------------
for metric, reported in [("auprc", -43.8), ("auroc", -11.5),
                         ("f1", -41.2), ("mcc", -42.3)]:
    check(f"Sec 4.3  change {metric} (%)",
          reported, pct_change(ev["T_clean"][metric], ev["T_leaky"][metric]), 0.05)

factor = pct_change(ev["T_clean"]["auprc"], ev["T_leaky"]["auprc"]) / \
         pct_change(ev["T_clean"]["auroc"], ev["T_leaky"]["auroc"])
check("Sec 4.3  factor AUPRC/AUROC", 3.8, factor, 0.05)

# ---- Configuration --------------------------------------------------------
check("Sec 3.5  trainable parameters", 282017, run["n_parameters"], 0)
check("Sec 3.8  locked threshold", 0.9087636471, run["locked_threshold"], 1e-9)
check("Sec 3.7  selected epoch", 131, run["best_epoch"], 0)
check("Sec 3.7  epochs logged", 156, len(run["history"]), 0)
check("Sec 4.1  training structures", 3412, run["n_train"], 0)

# ---- Table 4: matched three-method comparison, strict labels --------------
ver = json.loads((RES / "verify_strict_comparison.json").read_text())
pooled = ver["pooled"]
for method, auprc, auroc in [("GGNN", 0.4611, 0.8647),
                             ("P2Rank", 0.4703, 0.8800),
                             ("Fpocket", 0.1739, 0.6777)]:
    check(f"Table 4  {method:8s} AUPRC", auprc, pooled[method]["auprc"])
    check(f"Table 4  {method:8s} AUROC", auroc, pooled[method]["auroc"])

check("Sec 4.5  gap to P2Rank AUPRC", 0.0092,
      pooled["P2Rank"]["auprc"] - pooled["GGNN"]["auprc"])
check("Sec 4.5  gap to P2Rank AUROC", 0.0152,
      pooled["P2Rank"]["auroc"] - pooled["GGNN"]["auroc"])
check("Sec 4.5  labels verified", 0, len(ver["label_mismatches"]), 0)

# ---- Section 4.4 / Table 3: the edge-feature frame ------------------------
frame = json.loads((RES / "frame_paired_test.json").read_text())
for i, reported in enumerate([0.0350, 0.0543, 0.0574]):
    check(f"Sec 4.4  paired difference seed {frame['seeds'][i]}",
          reported, frame["diff"][i])
check("Sec 4.4  mean difference", 0.0489, frame["mean"])
check("Sec 4.4  standard deviation", 0.0121, frame["sd"])
check("Sec 4.4  t statistic", 6.99, frame["t"], 0.005)
check("Sec 4.4  p value", 0.020, frame["p"], 0.0005)
check("Sec 4.4  CI lower", 0.0188, frame["ci_lo"])
check("Sec 4.4  CI upper", 0.0790, frame["ci_hi"])

global_mean = sum(frame["global_"]) / 3
check("Sec 4.4  relative gain (%)", 11.6, frame["mean"] / global_mean * 100, 0.05)

seeds = {r["config"]: r for r in csv.DictReader((RES / "seed_replicates.csv").open())}
check("Table 3  global AUPRC mean", 0.4224, float(seeds["B4_global"]["auprc_mean"]))
check("Table 3  global AUPRC s.d.", 0.0207, float(seeds["B4_global"]["auprc_sd"]))
check("Table 3  local  AUPRC mean", 0.4713, float(seeds["B4_localframe"]["auprc_mean"]))
check("Table 3  local  AUPRC s.d.", 0.0094, float(seeds["B4_localframe"]["auprc_sd"]))
check("Sec 4.4  homology gap, global (%)", 41.4,
      float(seeds["B4_global"]["homology_gap_mean"]) * 100, 0.05)
check("Sec 4.4  homology gap, local (%)", 36.5,
      float(seeds["B4_localframe"]["homology_gap_mean"]) * 100, 0.05)

# data scaling: 1,850 -> 4,254 structures, local frame throughout
check("Sec 4.4  data-scaling gain (%)", 4.9,
      pct_change(float(seeds["B5_localframe"]["auprc_mean"]),
                 float(seeds["B4_localframe"]["auprc_mean"])), 0.05)

# ---- Table 5: pocket localisation (permissive labels) ---------------------
pocket = {r["approach"]: r for r in csv.DictReader((RES / "pocket_level_results.csv").open())}
tuned = pocket["GGNN residue clustering (validation-tuned)"]
check("Table 5  GGNN Top-1 (%)", 19.9, float(tuned["top1_dca4"]) * 100, 0.05)
check("Table 5  GGNN Top-3 (%)", 22.0, float(tuned["top3_dca4"]) * 100, 0.05)
check("Table 5  GGNN median distance", 8.14, float(tuned["median_dca1"]), 0.005)
check("Table 5  Fpocket Top-1 (%)", 56.9,
      float(pocket["Fpocket native ranking"]["top1_dca4"]) * 100, 0.05)
check("Table 5  P2Rank Top-1 (%)", 58.1,
      float(pocket["P2Rank"]["top1_dca4"]) * 100, 0.05)

# ---- Supplementary S10: error analysis ------------------------------------
tc = ev["T_clean"]
for name, reported in [("tp", 3257), ("fp", 2486), ("fn", 5611), ("tn", 134958)]:
    check(f"Sec S10  {name}", reported, tc[name], 0)
check("Sec S10  missed positives (%)", 63.3,
      tc["fn"] / (tc["tp"] + tc["fn"]) * 100, 0.05)

# ---- report ---------------------------------------------------------------
failed = [c for c in checks if not c[0]]
width = max(len(c[1]) for c in checks)
for ok, label, reported, actual in checks:
    if not ok:
        print(f"FAIL  {label:<{width}}  reported={reported!r}  file={actual!r}")
print(f"\n{len(checks) - len(failed)}/{len(checks)} reported values match the saved result files.")
if failed:
    print(f"{len(failed)} MISMATCH(ES) — see above.")
    sys.exit(1)
print("All headline numbers in the manuscript are reproduced from results/.")
sys.exit(0)
