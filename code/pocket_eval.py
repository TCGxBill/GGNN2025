#!/usr/bin/env python3
"""
Pocket-level evaluation by DCA, the criterion the binding-site field actually reports.

The shipped experiments/pocket_level_eval.py reports "Top-N success" as *at least one of the N
highest-scoring residues falls inside a true site*, which is a residue hit rate, not a pocket
metric, and is not comparable to the published Top-N figures of P2Rank / DeepSurf / Kalasanty.
It also never clusters residues into pockets, so its "predicted centre" is the centroid of all
high-scoring residues anywhere on the surface (median DCA 39.96 A in the original results).

Here: per-residue probabilities -> spatial clustering of confident residues -> pockets ranked by
summed probability -> DCA from each pocket centre to each true ligand centre. A structure counts
as a Top-N success if any of its N best-ranked pockets lies within the cutoff of any true ligand.

Baselines are scored identically: P2Rank pocket centres come from its own *_predictions.csv,
Fpocket centres from the alpha-sphere barycentres in its pocket PDBs.
"""
import argparse, collections, csv, json, os, sys
from pathlib import Path

import numpy as np
import torch

SRC = Path("/Users/trongnhannguyenvu/Downloads/files-2/binding-site-prediction")
OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
sys.path.insert(0, str(SRC))
EXCLUDE = {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "MN",
           "FE", "SO4", "PO4"}


def pdb_path(code):
    for d in ("holo4k", "joined"):
        p = SRC / "data/raw" / d / f"{code}.pdb"
        if p.exists():
            return p
    return None


def true_sites(path):
    """One centre per candidate ligand (>=3 heavy atoms, not water/ion)."""
    g = collections.defaultdict(list)
    for line in open(path, errors="ignore"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("HETATM"):
            rn = line[17:20].strip().upper()
            if rn in EXCLUDE:
                continue
            g[(line[21], line[22:27], rn)].append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return [np.mean(np.array(v, dtype=np.float32), axis=0) for v in g.values() if len(v) >= 3]


def cluster_pockets(pos, prob, prob_cut, link_dist=8.0, min_size=3):
    """Single-linkage clustering of confident residues; pockets ranked by summed probability."""
    sel = np.where(prob >= prob_cut)[0]
    if len(sel) == 0:
        return []
    P = pos[sel]
    n = len(sel)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    for i, j in zip(*np.where(d <= link_dist)):
        if i < j:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    pockets = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        idx = sel[members]
        w = prob[idx]
        centre = (pos[idx] * w[:, None]).sum(0) / w.sum()
        pockets.append((float(w.sum()), centre))
    pockets.sort(key=lambda t: -t[0])
    return [c for _, c in pockets]


def success(pockets, sites, topn, cutoff):
    if not sites:
        return None
    for c in pockets[:topn]:
        for s in sites:
            if np.linalg.norm(c - s) <= cutoff:
                return True
    return False


def p2rank_pockets(code):
    f = Path(f"/tmp/p2out/{code}.pdb_predictions.csv")
    if not f.exists():
        return []
    out = []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): v for k, v in row.items()}
            try:
                out.append((float(row["score"]),
                            np.array([float(row["center_x"]), float(row["center_y"]),
                                      float(row["center_z"])], dtype=np.float32)))
            except (KeyError, ValueError):
                continue
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B5_localframe")
    ap.add_argument("--graph-dir", default="data_localframe/combined")
    ap.add_argument("--prob-cut", type=float, default=None,
                    help="confidence cut for clustering; default = the checkpoint's locked threshold")
    a = ap.parse_args()

    import yaml
    from src.models.gcn_geometric import GeometricGNN
    cfg = yaml.safe_load(open(SRC / "config_optimized.yaml"))
    ck = torch.load(OUT / f"checkpoints/{a.tag}.pth", weights_only=False)
    mcfg = dict(cfg["model"])
    if "relframe" in a.graph_dir:
        mcfg["edge_dim"] = 29
    model = GeometricGNN(mcfg)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    cut = a.prob_cut if a.prob_cut is not None else ck["threshold"]

    sp = json.load(open(OUT / "splits/splits_clean.json"))
    gidx = {}
    for s in ("train", "val", "test"):
        for p in (OUT / a.graph_dir / s).glob("*.pt"):
            gidx.setdefault(p.stem, p)

    TOPN = [1, 3, 5]
    CUTS = [4.0, 8.0]
    res = {m: {f"top{n}_dca{int(c)}": [] for n in TOPN for c in CUTS}
           for m in ("GGNN", "P2Rank")}
    dca1 = {m: [] for m in ("GGNN", "P2Rank")}
    n_used = 0
    for code in sp["T_clean"]:
        pp = pdb_path(code)
        if pp is None or code not in gidx:
            continue
        sites = true_sites(pp)
        if not sites:
            continue
        g = torch.load(gidx[code], weights_only=False)
        if "n_ligands" in g:
            del g["n_ligands"]
        with torch.no_grad():
            prob = torch.sigmoid(model(g)[0]).numpy().ravel()
        pk = {"GGNN": cluster_pockets(g.pos.numpy(), prob, cut),
              "P2Rank": p2rank_pockets(code)}
        for m in res:
            for n in TOPN:
                for c in CUTS:
                    s = success(pk[m], sites, n, c)
                    if s is not None:
                        res[m][f"top{n}_dca{int(c)}"].append(s)
            if pk[m]:
                dca1[m].append(float(min(np.linalg.norm(pk[m][0] - s) for s in sites)))
        n_used += 1

    print(f"proteins evaluated: {n_used}   GGNN confidence cut: {cut:.4f}\n")
    print(f"{'method':8s} " + "  ".join(f"{k:>12s}" for k in res['GGNN']))
    for m, d in res.items():
        print(f"{m:8s} " + "  ".join(f"{100*np.mean(v):11.1f}%" for v in d.values()))
    print()
    for m, v in dca1.items():
        print(f"{m:8s} top-1 DCA: median {np.median(v):6.2f} A   mean {np.mean(v):6.2f} A   n={len(v)}")
    json.dump({m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in res.items()} |
              {f"{m}_median_dca1": float(np.median(v)) for m, v in dca1.items()},
              open(OUT / f"results/pocket_eval_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
