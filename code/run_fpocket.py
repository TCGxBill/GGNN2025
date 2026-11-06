#!/usr/bin/env python3
"""
Run Fpocket on the T_clean proteins and convert its pockets into per-residue scores, so it can be
compared with GGNN on the same ground truth (all-ligand labels) and the same metrics.

Fpocket ranks pockets; it does not emit per-residue probabilities. A residue's score is taken from
the best-ranked pocket that contains it:  score = 1 - (rank-1)/n_pockets, and 0 if no pocket
contains it. This preserves Fpocket's own ordering, which is what AUPRC measures.

Ground truth comes from the relabelled graphs (all candidate ligands, 6 A), identical to what GGNN
is scored against. Residues are matched to the graph by Ca coordinate.
"""
import json, os, re, shutil, subprocess, sys, tempfile
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GGNN_SRC", ROOT))
OUT = Path(os.environ.get("GGNN_OUT", ROOT))


def pdb_path(code):
    for d in ("holo4k", "joined"):
        p = SRC / "data/raw" / d / f"{code}.pdb"
        if p.exists():
            return p
    return None


def ca_coords(path):
    """Ca coordinates in file order, with their (chain, resseq) key."""
    out, seen = [], set()
    for line in open(path, errors="ignore"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            k = (line[21], line[22:27])
            if k in seen:
                continue
            seen.add(k)
            out.append((k, (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    return out


def run_one(path, workdir):
    """Return {(chain,resseq): best pocket rank}, 1-based, or None on failure."""
    tmp = Path(workdir) / path.stem
    tmp.mkdir(parents=True, exist_ok=True)
    local = tmp / path.name
    shutil.copy(path, local)
    r = subprocess.run(["fpocket", "-f", str(local)], capture_output=True, text=True, timeout=600)
    outdir = tmp / f"{path.stem}_out"
    pockets = outdir / "pockets"
    if not pockets.exists():
        return None
    best = {}
    for f in pockets.glob("pocket*_atm.pdb"):
        m = re.match(r"pocket(\d+)_atm", f.stem)
        if not m:
            continue
        rank = int(m.group(1)) + 1          # fpocket numbers from 0, ranked by score
        for line in open(f, errors="ignore"):
            if line.startswith("ATOM"):
                k = (line[21], line[22:27])
                if k not in best or rank < best[k]:
                    best[k] = rank
    return best


def main():
    sp = json.load(open(OUT / "splits/splits_clean.json"))
    gdir = OUT / "data_relabeled/combined"
    gidx = {}
    for s in ("train", "val", "test"):
        for p in (gdir / s).glob("*.pt"):
            gidx.setdefault(p.stem, p)

    scores, labels, per_protein, missing = [], [], {}, 0
    with tempfile.TemporaryDirectory() as wd:
        for i, code in enumerate(sp["T_clean"], 1):
            gp, pp = gidx.get(code), pdb_path(code)
            if gp is None or pp is None:
                missing += 1
                continue
            g = torch.load(gp, weights_only=False)
            y = g.y.numpy().ravel()
            cas = ca_coords(pp)
            if len(cas) != len(y):          # preprocessor filtered residues; align by coordinate
                P = g.pos.numpy()
                keymap = {}
                for k, c in cas:
                    keymap[tuple(np.round(c, 2))] = k
                keys = [keymap.get(tuple(np.round(c, 2))) for c in P]
            else:
                keys = [k for k, _ in cas]
            best = run_one(pp, wd)
            if best is None:
                missing += 1
                continue
            npk = max(best.values()) if best else 1
            s = np.array([0.0 if (k is None or k not in best)
                          else 1.0 - (best[k] - 1) / max(npk, 1) for k in keys])
            scores.append(s); labels.append(y)
            per_protein[code] = dict(n=int(len(y)), pos=int(y.sum()), pockets=int(npk))
            if i % 25 == 0:
                print(f"  {i}/{len(sp['T_clean'])}  ok={len(scores)}", flush=True)

    y = np.concatenate(labels); s = np.concatenate(scores)
    np.savez(OUT / "results/fpocket_T_clean.npz", y=y, s=s)
    json.dump(per_protein, open(OUT / "results/fpocket_per_protein.json", "w"))
    print(f"\nproteins scored {len(scores)} | skipped {missing} | residues {len(y)}")


if __name__ == "__main__":
    main()
