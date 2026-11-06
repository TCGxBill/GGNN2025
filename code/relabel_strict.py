#!/usr/bin/env python3
"""
Rebuild y for every graph using the artifact-free ligand definition, writing data_strict/.

Identical to relabel_all_ligands.py except for the ligand filter: crystallisation additives,
cryoprotectants, buffers and modified amino acids are no longer treated as ligands. Only `y` is
touched; coordinates, node features and edge features are copied unchanged from data_localframe/,
so the strict run differs from B5_localframe in labels alone.

Graphs left with no positive residue are still written; the training driver skips empty-label
graphs via the split files produced here.
"""
import collections, json, os, shutil, sys
import os
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GGNN_SRC", ROOT))
OUT = Path(os.environ.get("GGNN_OUT", ROOT))
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(OUT / "code"))
from pocket_eval import EXCLUDE
from strict_labels import MODRES, ARTIFACT, CUTOFF

SRCDIR = OUT / os.environ.get("RELABEL_SRC", "data_localframe")
DST = OUT / os.environ.get("RELABEL_DST", "data_strict")
PDB_DIRS = [SRC / "data/raw/holo4k", SRC / "data/raw/joined",
            OUT / "expansion2/pdb", SRC / "data/raw/biolip_sample"]
for d in (SRC / "external_data/p2rank-datasets").glob("*"):
    if d.is_dir():
        PDB_DIRS.append(d)


# Graph filenames from the two expansion rounds carry a provenance prefix that the structure files
# on disk do not (RCSB_1DEL.pt <- 1DEL.pdb). Without this map the first pass silently dropped all
# 2,881 expansion graphs as "no PDB found", which would have cost 46% of the training set.
PREFIX_DIRS = {
    "RCSB_":         [OUT / "expansion2/pdb"],
    "EXP_p2holo4k_": [SRC / "external_data/p2rank-datasets/holo4k", SRC / "data/raw/holo4k"],
    "EXP_chen11_":   [SRC / "external_data/p2rank-datasets/chen11"],
    "EXP_biolip_":   [SRC / "data/raw/biolip_sample"],
    "EXP_fptrain_":  [SRC / "external_data/p2rank-datasets/fptrain"],
}


def find_pdb(code):
    for d in PDB_DIRS:
        for ext in (".pdb", ".ent", ".cif"):
            p = d / f"{code}{ext}"
            if p.exists():
                return p
    for pre, dirs in PREFIX_DIRS.items():
        if not code.startswith(pre):
            continue
        stem = code[len(pre):]
        for d in dirs:
            for cand in (stem, stem.upper(), stem.lower()):
                for ext in (".pdb", ".ent", ".cif"):
                    p = d / f"{cand}{ext}"
                    if p.exists():
                        return p
    return None


def ligand_atoms(path):
    g = collections.defaultdict(list)
    for line in open(path, errors="ignore"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("HETATM"):
            rn = line[17:20].strip().upper()
            if rn in EXCLUDE or rn in MODRES or rn in ARTIFACT:
                continue
            try:
                g[(line[21], line[22:27], rn)].append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    return [a for v in g.values() if len(v) >= 3 for a in v]


def main():
    stats = collections.Counter()
    empty = collections.defaultdict(list)
    for sub in sorted(p for p in SRCDIR.iterdir() if p.is_dir()):
        for split in sorted(p for p in sub.iterdir() if p.is_dir()):
            dst = DST / sub.name / split.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in split.glob("*.pt"):
                code = f.stem
                out = dst / f.name
                if out.exists():
                    stats["skip"] += 1
                    continue
                g = torch.load(f, weights_only=False)
                p = find_pdb(code)
                if p is None:
                    stats["no_pdb"] += 1
                    empty[f"{sub.name}/{split.name}"].append(code)
                    continue
                A = ligand_atoms(p)
                if not A:
                    stats["no_ligand"] += 1
                    empty[f"{sub.name}/{split.name}"].append(code)
                    continue
                pos = g.pos.numpy()
                d = np.linalg.norm(pos[:, None, :] - np.array(A, np.float32)[None, :, :], axis=2).min(1)
                y = torch.tensor((d <= CUTOFF).astype(np.int64))
                if y.sum() == 0:
                    stats["zero_pos"] += 1
                    empty[f"{sub.name}/{split.name}"].append(code)
                    continue
                g.y = y
                torch.save(g, out)
                stats["written"] += 1
                stats["pos"] += int(y.sum()); stats["res"] += len(y)
            print(f"  {sub.name}/{split.name}: {stats['written']} written", flush=True)
    print(dict(stats))
    print(f"positive rate under strict labels: {100*stats['pos']/max(1,stats['res']):.2f}%")
    json.dump({k: v for k, v in empty.items()},
              open(OUT / "results/strict_dropped.json", "w"), indent=1)


if __name__ == "__main__":
    main()
