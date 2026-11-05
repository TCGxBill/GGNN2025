# Reproducibility — GGNN 2025 (v2)

This repository accompanies the homology-controlled revision of GGNN 2025. The
revision followed an audit of the original evaluation pipeline, which found a
random rather than sequence-clustered partition, a decision threshold selected on
the test set, and a label rule that discarded every ligand except the largest
heteroatom group. **The v1 instructions that previously occupied this file
described that superseded pipeline and have been withdrawn.** The original code
and results are preserved unchanged under `legacy_v1/` so that the audit itself
remains checkable.

Reproduction is possible at three levels. Level 1 works from a bare clone with no
dependencies at all; levels 2 and 3 need data that is too large to distribute
through Git.

---

## Level 1 — verify every reported number

**Requires:** a clone of this repository and Python 3.8 or newer. No PyTorch, no
model weights, no graph data, no network access.

```bash
git clone https://github.com/TCGxBill/GGNN2025.git
cd GGNN2025
python code/check_reported_numbers.py
```

The script re-derives 67 values — every table row and every percentage change in
the manuscript — directly from the JSON and CSV files in `results/`, and exits
non-zero if any of them disagrees. Expected output:

```
67/67 reported values match the saved result files.
All headline numbers in the manuscript are reproduced from results/.
```

What it covers: the three-split table and its four percentage changes, the
parameter count, the locked threshold, the selected epoch, the matched
three-method comparison under strict labels, the paired seed test for the
edge-feature frame with its t statistic and confidence interval, the two
seed-replicate configurations, the data-scaling gain, the pocket-localisation
table, and the confusion counts behind the error analysis.

This level establishes that the manuscript reports what the run artifacts
contain. It does not re-execute the model.

---

## Level 2 — recompute the metrics from the checkpoint

**Requires:** the processed graph tensors (`data_strict/combined/`, roughly 6 GB,
not distributed here) plus PyTorch and scikit-learn.

```bash
python verification/verify.py
```

This performs an independent CPU forward pass of `checkpoints/B5_strict_lf.pth`
over all three splits, rederives the validation-selected threshold, and writes
`verification/verification.json`. The model class is vendored under
`verification/copied_code/` so that the verification path does not depend on the
rest of `src/`.

Companion checks, each writing its own JSON record:

| Script | What it establishes |
|---|---|
| `verification/check_data.py` | labels recomputed from raw HETATM coordinates agree on all 244 clean-test proteins; ligand-group and amino-acid counts |
| `verification/check_configuration.py` | graph dimensions, the constant DSSP fallback slots, the frame determinant of −1 |
| `verification/verify_seeds.py` | six frame-comparison checkpoints reproduce their clean-test metrics |
| `verification/check_original_figure.py` | the preserved permissive-label checkpoint reproduces AUPRC 0.4963 and 49/246 pocket Top-1 |

The records produced by these scripts are committed, so their outputs can be
inspected without running them.

---

## Level 3 — rebuild the pipeline from structures

**Requires:** the raw PDB structures and the external benchmark archives, neither
of which is distributed here. Expect several hours on CPU per training run.

The pipeline, in order:

```bash
# 1. Cluster all sequences and assign whole clusters to a single split
mmseqs easy-cluster <fasta> <prefix> <tmp> --min-seq-id 0.3 -c 0.5 --cov-mode 0
#    Split manifests produced by this step are committed in splits/

# 2. Apply the strict ligand-label rule (excludes additives, ions, modified residues)
python code/relabel_strict.py

# 3. Rewrite edge directions into the per-residue local backbone frame
python code/localize_frames.py

# 4. Train
python code/run_clean.py --tag B5_strict_lf --train-split B5_train \
    --loss combined_ft --seed 42 --epochs 200 --patience 25

# 5. Score the baselines on the identical split and labels
python code/run_fpocket.py
python code/baselines_on_split.py
python code/verify_strict_comparison.py

# 6. External benchmarks, pocket localisation, figures
python code/eval_benchmarks.py
python code/pocket_eval.py
python code/make_figures.py        # writes figures/
```

`run_clean.py` honours `GGNN_THREADS` for CPU thread count and mirrors the best
checkpoint to `checkpoints/<tag>.partial.pth` whenever validation AUPRC improves,
so an interrupted run is recoverable.

---

## The protocol in one paragraph

Sequences are clustered with MMseqs2 `easy-cluster` at 30% identity with coverage
0.5, and whole clusters are assigned to a single split, so no test or validation
protein shares a cluster with any training protein. A residue is positive when its
stored coordinate lies within 6 Å of an atom of a retained candidate ligand;
water, monatomic ions, crystallisation additives and modified amino acids are
excluded by the list in the supplementary material. Checkpoint selection and the
operating threshold are both determined on validation — by AUPRC and by F1
respectively — and the threshold is then held fixed for every test evaluation.

Splits: 3,412 training, 302 validation, 244 cluster-disjoint test, 316 homologous
test proteins. Seed 42. Locked threshold 0.9087636471.

---

## What is not in this repository

- **Processed graph tensors and raw structures.** Too large for Git. Levels 2 and
  3 cannot be run from a clone alone.
- **The manuscript PDF and LaTeX source.** Held back deliberately; `paper/` is
  listed in `.gitignore`. The study record is the Zenodo deposit,
  [10.5281/zenodo.18061054](https://doi.org/10.5281/zenodo.18061054).
- **Score dumps** (`*.npz`) behind the baseline comparison, omitted for size. The
  aggregated metrics they produce are in `results/`.

---

## Known gaps

These are stated rather than papered over.

1. **Figure 1 has no generating script.** `code/make_figures.py` regenerates the
   other four figures from the result files; the original evaluation-audit figure
   does not have its plotting code on disk.
2. **No matched end-to-end timing comparison exists.** The architecture runs on
   CPU, but per-protein latency and peak memory were never measured under
   controlled conditions across methods, so no speedup is claimed anywhere.
3. **The strict-label headline run is a single seed.** The edge-feature frame
   comparison is replicated across three seeds; the headline run is not.
4. **Baseline metrics were recalculated from stored residue scores,** not from a
   fresh execution of the Fpocket or P2Rank binaries.
5. **The split-manifest check does not rerun MMseqs2.** It confirms that no
   cluster identifier is shared between training, validation and clean test.

---

## Software

```
Python >= 3.10
torch==2.4.0
torch-geometric==2.5.3
numpy==2.2.5
scipy==1.15.3
scikit-learn==1.6.1
pandas==2.2.3
biopython==1.86
pyyaml==6.0.2
matplotlib==3.10.3
```

Level 1 needs none of these — only the standard library.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

External tools: MMseqs2 for clustering, Fpocket and P2Rank 2.4.1 for the
baselines. DSSP is optional; note that in the committed runs DSSP was unavailable
for every graph, so three secondary-structure slots are zero and the relative
accessibility slot is 0.5 throughout.

---

## Contact

For reproducibility questions, please open an issue at
<https://github.com/TCGxBill/GGNN2025/issues>.
