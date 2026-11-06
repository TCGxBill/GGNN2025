# GGNN 2025: Protein-Ligand Binding Site Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18061054.svg)](https://doi.org/10.5281/zenodo.18061054)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A compact geometric graph neural network that scores residues for ligand binding
from structure alone. It has 282,017 trainable parameters, 1.1 MB on disk, CPU-only,
with no protein language model and no sequence alignment.

**This repository is the homology-controlled revision (v2).** An audit of the
original pipeline found three defects: a random rather than sequence-clustered
partition, a decision threshold selected by maximising F1 on the test set, and a
label rule that kept only the largest heteroatom group per structure. All three
are corrected here, and the model is re-evaluated under the corrected protocol.
The original code and results are preserved unchanged under `legacy_v1/` so the
audit itself remains checkable.

> The v1 README and reproducibility guide advertised state-of-the-art accuracy
> using numbers measured on a leaky split. **Those claims are withdrawn.** They
> are superseded by the results below.

---

## Results under the corrected protocol

On 244 cluster-disjoint test proteins, with identical labels and identical
metrics for all three methods:

| Method | Parameters | AUPRC | AUROC |
|---|---|---:|---:|
| P2Rank 2.4.1 | random forest | 0.4703 | 0.8800 |
| GGNN 2025 | 282,017 | 0.4611 | 0.8647 |
| Fpocket | 0 | 0.1739 | 0.6777 |

GGNN 2025 is **behind** P2Rank, and the paired bootstrap interval contains zero,
so the two are statistically indistinguishable on this test set. No superiority
claim is made.

One model, one validation-selected threshold (τ = 0.9088), one evaluation code
path, three splits:

| Split | AUPRC | AUROC | F1 | MCC |
|---|---:|---:|---:|---:|
| Validation (302) | 0.6696 | 0.9306 | 0.6205 | 0.5985 |
| Test, homologous (316) | 0.8207 | 0.9776 | 0.7582 | 0.7437 |
| Test, cluster-disjoint (244) | 0.4611 | 0.8647 | 0.4458 | 0.4291 |
| **Change, homologous → disjoint** | **−43.8%** | **−11.5%** | −41.2% | −42.3% |

AUROC loses 11.5% of its value under homology control while AUPRC loses 43.8%,
a factor of 3.8. This is why AUPRC is the primary metric here.

**Edge-feature frame.** Expressing edge directions in a per-residue local
backbone frame instead of the global PDB coordinate frame is worth **+11.6%
AUPRC (p = 0.020, three paired seeds)** at zero parameter cost. For scale,
growing the training pool by 130% is worth +4.9%.

**Residue ranking is not pocket localisation.** Under permissive labels GGNN has
the best residue AUPRC of the three methods and the worst pocket Top-1 (19.9%
against 56.9% for Fpocket and 58.1% for P2Rank). GGNN 2025 is a residue-level
scorer; for a docking box, use a geometric method.

---

## Verify every reported number

Needs nothing but a clone and the Python standard library. No PyTorch, no
weights, no graph data, no network:

```bash
git clone https://github.com/TCGxBill/GGNN2025.git
cd GGNN2025
python code/check_reported_numbers.py
```

```
67/67 reported values match the saved result files.
```

The script re-derives every table row and every percentage change in the
manuscript directly from `results/`, and exits non-zero on any disagreement.

---

## Reproduction, in three levels

Level 1 works from a bare clone. Levels 2 and 3 need data that is too large to
distribute through Git.

### Level 1: verify the reported numbers

`python code/check_reported_numbers.py`, as above. This establishes that the
manuscript reports what the run artifacts contain. It does not re-execute the
model.

### Level 2: recompute metrics from the checkpoint

**Requires** the processed graph tensors (`data_strict/combined/`, roughly 6 GB,
not distributed here), PyTorch and scikit-learn.

```bash
python verification/verify.py
```

An independent CPU forward pass of `checkpoints/B5_strict_lf.pth` over all three
splits, rederiving the validation-selected threshold and writing
`verification/verification.json`. The model class is vendored under
`verification/copied_code/` so this path does not depend on the rest of `src/`.

| Script | What it establishes |
|---|---|
| `verification/check_data.py` | labels recomputed from raw HETATM coordinates agree on all 244 clean-test proteins |
| `verification/check_configuration.py` | graph dimensions, the constant DSSP fallback slots, frame determinant −1 |
| `verification/verify_seeds.py` | six frame-comparison checkpoints reproduce their clean-test metrics |
| `verification/check_original_figure.py` | the permissive-label checkpoint reproduces AUPRC 0.4963 and 49/246 pocket Top-1 |

Their output records are committed, so the results can be inspected without
running them.

### Level 3: rebuild from structures

**Requires** the raw PDB structures and external benchmark archives, neither
distributed here. Several hours on CPU per training run.

```bash
# 1. Cluster sequences; whole clusters go to a single split
mmseqs easy-cluster <fasta> <prefix> <tmp> --min-seq-id 0.3 -c 0.5 --cov-mode 0
#    The resulting manifests are committed in splits/

# 2. Strict ligand labels (excludes additives, ions, modified residues)
python code/relabel_strict.py

# 3. Rewrite edge directions into the per-residue local backbone frame
python code/localize_frames.py

# 4. Train
python code/run_clean.py --tag B5_strict_lf --train-split B5_train \
    --loss combined_ft --seed 42 --epochs 200 --patience 25

# 5. Baselines on the identical split and labels
python code/run_fpocket.py
python code/baselines_on_split.py
python code/verify_strict_comparison.py

# 6. External benchmarks, pocket localisation, figures
python code/eval_benchmarks.py
python code/pocket_eval.py
python code/make_figures.py        # writes figures/
```

### Paths and environment variables

No script contains an absolute path. Each resolves the repository root from its
own location, and both roots can be redirected:

| Variable | Default | Purpose |
|---|---|---|
| `GGNN_SRC` | repository root | where raw structures and processed graphs are read from |
| `GGNN_OUT` | repository root | where results, splits and checkpoints are written |
| `GGNN_PYTHON` | `python3` | interpreter used by the batch shell scripts |
| `GGNN_THREADS` | all cores | CPU threads for training and inference |
| `FPOCKET_BIN` | `fpocket` | path to the Fpocket executable |

If your data tree is separate from the repository, point `GGNN_SRC` and
`GGNN_OUT` at it:

```bash
export GGNN_SRC=/path/to/data-tree GGNN_OUT=/path/to/run-tree
```

---

## Protocol

Sequences are clustered with MMseqs2 `easy-cluster` at 30% identity with coverage
0.5, and whole clusters are assigned to a single split, so no test or validation
protein shares a cluster with any training protein. A residue is positive when
its stored coordinate lies within 6 Å of an atom of a retained candidate ligand;
water, monatomic ions, crystallisation additives and modified amino acids are
excluded. Checkpoint selection and the operating threshold are both determined on
validation, by AUPRC and by F1 respectively, and the threshold is then held
fixed for every test evaluation.

Splits: 3,412 training, 302 validation, 244 cluster-disjoint test, 316 homologous
test proteins. Seed 42. Locked threshold 0.9087636471.

**Architecture.** Three GATv2 blocks of width 256 → 128 → 64, four attention
heads, edge-conditioned attention over 20-dimensional edge features, residual
connections, batch normalisation with ELU, dropout 0.15. Mean- and max-pooled
global context projected to 32 dimensions and broadcast back to every node. Node
features are 32-dimensional; note that in the committed runs DSSP was unavailable
for every graph, so three secondary-structure slots are zero and the relative
accessibility slot is 0.5 throughout. Objective: 0.3 weighted BCE + 0.7 focal
Tversky (α = 0.3, β = 0.7, γ = 1.33, positive weight 12).

---

## Repository layout

```
paper/          manuscript (not published here; see Zenodo)
results/        all run outputs as JSON and CSV
verification/   independent checks and their recorded outputs
splits/         cluster-assignment manifests
code/           training, evaluation, baseline and figure scripts
figures/        the four manuscript figures (PDF + 400 dpi PNG)
checkpoints/    B5_strict_lf.pth, the strict-label model (1.1 MB)
src/            model, data and training modules
legacy_v1/      all superseded v1 material, preserved unchanged
```

## What is not in this repository

- **Processed graph tensors and raw structures.** Too large for Git; levels 2 and
  3 cannot run from a clone alone.
- **The manuscript PDF and LaTeX source.** Held back deliberately; `paper/` is
  listed in `.gitignore`. The study record is the Zenodo deposit below.
- **Score dumps** (`*.npz`) behind the baseline comparison, omitted for size. The
  metrics they produce are in `results/`.

## Known gaps

1. The original evaluation-audit figure has no generating script;
   `code/make_figures.py` regenerates the other four from the result files.
2. No matched end-to-end timing comparison exists. The model runs on CPU, but
   per-protein latency and peak memory were never measured under controlled
   conditions across methods, so no speedup is claimed anywhere.
3. The strict-label headline run is a single seed. The frame comparison is
   replicated across three.
4. Baseline metrics were recalculated from stored residue scores, not from a
   fresh execution of the Fpocket or P2Rank binaries.
5. The split-manifest check confirms no shared cluster identifiers; it does not
   rerun MMseqs2.

---

## Software

```
Python >= 3.10
torch==2.4.0            scipy==1.15.3        biopython==1.86
torch-geometric==2.5.3  scikit-learn==1.6.1  pyyaml==6.0.2
numpy==2.2.5            pandas==2.2.3        matplotlib==3.10.3
```

Level 1 needs none of these, only the standard library.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

External tools: MMseqs2 for clustering, Fpocket and P2Rank 2.4.1 for the
baselines. DSSP is optional.

---

## Citation

```bibtex
@misc{nguyen2026ggnn,
  title  = {GGNN 2025: A Lightweight Geometric Graph Neural Network
            for Protein--Ligand Binding Site Prediction},
  author = {Nguyen, Nhan Vu Trong and Nguyen, Linh Hoang Anh},
  year   = {2026},
  doi    = {10.5281/zenodo.18061054},
  note   = {Homology-controlled revision}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Dataset providers for Holo4K, scPDB, PDBbind, SC6K, COACH420, Binding MOAD,
DUD-E and CryptoBench; the PyTorch Geometric team; the authors of MMseqs2,
Fpocket and P2Rank. AI assistants were used as development and analysis tools,
including for the audit of the original pipeline; experimental design,
measurements and conclusions are the authors' own.
