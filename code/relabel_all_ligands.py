#!/usr/bin/env python3
"""
Relabel every pre-built graph using ALL candidate ligands instead of only the largest one.

The original src/data/preprocessor.py::_extract_ligand_from_pdb keeps a single HETATM residue
(the one with most atoms) and labels residues within 6 A of it. On this dataset that discards
58.8% of ligand atoms and leaves 54% of true binding residues labelled negative.

Node features, edges and coordinates are untouched - only `y` is recomputed, against the graph's
own `pos` tensor, so no residue-ordering assumption is needed. Writes to a new tree; the source
data/processed/combined is not modified.
"""
import collections, glob, os, sys
import os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GGNN_SRC", ROOT))
OUT = Path(OUT) / "data_relabeled"
EXCLUDE = {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "MN",
           "FE", "SO4", "PO4"}
CUTOFF = 6.0          # same ligand-proximity rule the original code uses
MIN_LIGAND_ATOMS = 3  # same minimum-size rule


def ligand_coords(pdb_path):
    """All candidate ligands, not just the largest."""
    groups = collections.defaultdict(list)
    for line in open(pdb_path, errors="ignore"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("HETATM"):
            rn = line[17:20].strip().upper()
            if rn in EXCLUDE:
                continue
            key = (line[21], line[22:27], rn)
            groups[key].append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    keep = [v for v in groups.values() if len(v) >= MIN_LIGAND_ATOMS]
    if not keep:
        return None, 0
    return np.array([p for v in keep for p in v], dtype=np.float32), len(keep)


def main():
    pdb = {}
    for d in ("holo4k", "joined", "scpdb", "pdbbind_refined", "sc6k", "coach420",
              "cryptobench", "dude_diverse", "moad_quality", "difficult_cases"):
        for f in glob.glob(str(SRC / "data/raw" / d / "*.pdb")):
            pdb.setdefault(Path(f).stem, f)

    trees = [("combined/train", "combined/train"), ("combined/val", "combined/val"),
             ("combined/test", "combined/test")]
    trees += [(b, b) for b in ("scpdb", "pdbbind_refined", "sc6k", "coach420",
                               "cryptobench", "dude_diverse", "moad_quality", "difficult_cases")]

    tot = relabelled = skipped = 0
    pos_before = pos_after = residues = 0
    for src_rel, dst_rel in trees:
        (OUT / dst_rel).mkdir(parents=True, exist_ok=True)
        for p in sorted((SRC / "data/processed" / src_rel).glob("*.pt")):
            tot += 1
            try:
                g = torch.load(p, weights_only=False)
            except Exception:
                skipped += 1
                continue
            f = pdb.get(p.stem)
            if g is None or f is None or not hasattr(g, "pos"):
                skipped += 1
                continue
            L, n_lig = ligand_coords(f)
            if L is None:
                skipped += 1
                continue
            P = g.pos.numpy()
            d = np.linalg.norm(P[:, None, :] - L[None, :, :], axis=2).min(1)
            y_new = torch.tensor((d <= CUTOFF).astype(np.float32))
            pos_before += int(g.y.sum()); pos_after += int(y_new.sum()); residues += len(P)
            g.y = y_new
            g.n_ligands = n_lig
            torch.save(g, OUT / dst_rel / p.name)
            relabelled += 1
        print(f"  {dst_rel:22s} done", flush=True)

    print(f"\ngraphs seen {tot} | relabelled {relabelled} | skipped {skipped}")
    print(f"positive rate  before {100*pos_before/residues:.2f}%  ->  after {100*pos_after/residues:.2f}%")
    print(f"binding residues recovered: {pos_after-pos_before:,}")


if __name__ == "__main__":
    main()
