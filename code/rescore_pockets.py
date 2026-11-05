#!/usr/bin/env python3
"""
GGNN as a re-scoring function over geometric cavity candidates.

Clustering the network's own confident residues saturates at Top-1 DCA <= 4 A = 19.9% (validation
sweep: every high-scoring setting uses the widest linkage, mean 1.8 pockets per structure -- the
confident region is one blob, so there is nothing to rank). But Top-1 DCA <= 8 A was already ~50%:
the network finds the right region and misplaces the centre. That is expected, because a C-alpha
centroid is the centroid of the residues *lining* a cavity, not the cavity itself.

Fpocket was assumed to have the opposite failure -- good alpha-sphere cavity centres, poor ranking
-- so the experiment lets Fpocket enumerate cavities and lets GGNN rank them, scoring each cavity
by an aggregate of the network's residue probabilities over the graph nodes lining it.

RESULT: the premise was wrong and the experiment is a negative result. Fpocket's own ranking is
already strong at this criterion (val Top-1 DCA <= 4 A = 62.4%, T_clean 56.9%) -- its weak residue
level AUPRC of 0.1902 does not predict its pocket-level behaviour. The best re-scoring setting
found on validation (60.1%) is below Fpocket's native ranking on that same set, and on T_clean it
is far below (42.3% vs 56.9%). Re-scoring helps only deeper in the list (Top-5 DCA <= 4 A: 74.4%
vs 71.5%; Top-5 DCA <= 8 A: 88.6% vs 83.3%), i.e. it pulls true cavities into the top five but
cannot pick the best one. Kept and reported as a negative result; the oracle ceiling of 76.5%
(how often Fpocket's cavity set contains the true site at all) caps any re-scoring of it.

Aggregation, lining radius and candidate depth are selected on validation; T_clean is scored once.
Reported alongside: Fpocket's own ranking on the identical candidate sets (the control that
isolates the contribution of the re-scoring), and the oracle ceiling (is the true pocket in the
candidate set at all?).
"""
import itertools, json, os, re, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import yaml

SRC = Path("/Users/trongnhannguyenvu/Downloads/files-2/binding-site-prediction")
OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(OUT / "code"))
from pocket_eval import true_sites, pdb_path

torch.set_num_threads(3)
FPOCKET = "/Users/trongnhannguyenvu/.claude-science/conda/envs/baselines/bin/fpocket"
WORK = Path("/tmp/fp_rescore")
TAG, GDIR = "B5_localframe", "data_localframe/combined"


def fpocket_pockets(code):
    """Return [(fpocket_rank, centre, alpha_sphere_coords)] in fpocket's own ranked order."""
    d = WORK / f"{code}_out"
    if not d.exists():
        src = pdb_path(code)
        if src is None:
            return []
        tmp = WORK / f"{code}.pdb"
        shutil.copy(src, tmp)
        subprocess.run([FPOCKET, "-f", str(tmp)], capture_output=True, timeout=300)
        tmp.unlink(missing_ok=True)
    pd = d / "pockets"
    if not pd.exists():
        return []
    out = []
    for f in pd.glob("pocket*_vert.pqr"):
        m = re.match(r"pocket(\d+)_vert", f.name)
        if not m:
            continue
        pts = []
        for line in open(f, errors="ignore"):
            if line.startswith(("ATOM", "HETATM")):
                try:
                    pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    continue
        if not pts:
            continue
        A = np.array(pts, dtype=np.float32)
        out.append((int(m.group(1)), A.mean(0), A))
    out.sort(key=lambda t: t[0])          # fpocket ranks pocket0 best
    return out


def build_cache(codes, model, idx):
    cache = []
    for code in codes:
        pp = pdb_path(code)
        if pp is None or code not in idx:
            continue
        sites = true_sites(pp)
        if not sites:
            continue
        pk = fpocket_pockets(code)
        if not pk:
            continue
        g = torch.load(idx[code], weights_only=False)
        if "n_ligands" in g:
            del g["n_ligands"]
        with torch.no_grad():
            p = torch.sigmoid(model(g)[0]).numpy().ravel()
        pos = g.pos.numpy()
        # distance from every alpha sphere of every pocket to every residue C-alpha
        dists = [np.linalg.norm(pos[None, :, :] - A[:, None, :], axis=2).min(0) for _, _, A in pk]
        cache.append(dict(code=code, prob=p, sites=sites,
                          centres=[c for _, c, _ in pk], dmin=dists))
    return cache


def rank(entry, agg, radius, depth):
    scores = []
    for d in entry["dmin"][:depth]:
        m = d <= radius
        v = entry["prob"][m]
        if v.size == 0:
            scores.append(-1.0)
        elif agg == "mean":
            scores.append(float(v.mean()))
        elif agg == "max":
            scores.append(float(v.max()))
        elif agg == "sum":
            scores.append(float(v.sum()))
        else:                                    # mean of the 10 most confident lining residues
            scores.append(float(np.sort(v)[-10:].mean()))
    order = np.argsort(-np.array(scores))
    return [entry["centres"][i] for i in order]


def hit(centres, sites, n, cut):
    return any(np.linalg.norm(c - s) <= cut for c in centres[:n] for s in sites)


def evaluate(cache, ranker):
    out = {f"top{n}_dca{int(c)}": [] for n in (1, 3, 5) for c in (4.0, 8.0)}
    d1 = []
    for e in cache:
        C = ranker(e)
        if not C:
            continue
        for n in (1, 3, 5):
            for c in (4.0, 8.0):
                out[f"top{n}_dca{int(c)}"].append(hit(C, e["sites"], n, c))
        d1.append(float(min(np.linalg.norm(C[0] - s) for s in e["sites"])))
    r = {k: float(np.mean(v)) for k, v in out.items()}
    r["median_dca1"] = float(np.median(d1))
    return r


def main():
    WORK.mkdir(exist_ok=True)
    cfg = yaml.safe_load(open(SRC / "config_optimized.yaml"))
    from src.models.gcn_geometric import GeometricGNN
    ck = torch.load(OUT / f"checkpoints/{TAG}.pth", weights_only=False)
    model = GeometricGNN(dict(cfg["model"]))
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    idx = {}
    for s in ("train", "val", "test"):
        for p in (OUT / GDIR / s).glob("*.pt"):
            idx.setdefault(p.stem, p)
    sp = json.load(open(OUT / "splits/splits_clean.json"))

    val = build_cache(sp["val"], model, idx)
    print(f"validation structures with fpocket cavities + ligand: {len(val)}", flush=True)
    print(f"mean cavities per structure: {np.mean([len(e['centres']) for e in val]):.1f}", flush=True)

    base = evaluate(val, lambda e: e["centres"])
    print(f"\nfpocket native ranking (val): top1_4={100*base['top1_dca4']:.1f}%  "
          f"top3_4={100*base['top3_dca4']:.1f}%  med={base['median_dca1']:.2f}", flush=True)
    for D in (5, 10, 20, 999):
        orc = np.mean([hit(e["centres"][:D], e["sites"], D, 4.0) for e in val])
        print(f"  oracle ceiling, best of first {D:3d} cavities: {100*orc:5.1f}%", flush=True)

    rows = []
    for agg, radius, depth in itertools.product(("mean", "max", "sum", "top10"),
                                                (5.0, 6.5, 8.0), (10, 20, 999)):
        r = evaluate(val, lambda e, a=agg, rr=radius, d=depth: rank(e, a, rr, d))
        rows.append(dict(agg=agg, radius=radius, depth=depth, **r))
        print(f"{agg:6s} r={radius:<4} depth={depth:<4} "
              f"top1_4={100*r['top1_dca4']:5.1f}%  top3_4={100*r['top3_dca4']:5.1f}%  "
              f"med={r['median_dca1']:5.2f}", flush=True)

    rows.sort(key=lambda r: (-r["top1_dca4"], -r["top3_dca4"]))
    best = rows[0]
    print(f"\nselected on validation: agg={best['agg']} radius={best['radius']} "
          f"depth={best['depth']}  val top1_dca4={100*best['top1_dca4']:.1f}%", flush=True)

    tc = build_cache(sp["T_clean"], model, idx)
    fin = evaluate(tc, lambda e: rank(e, best["agg"], best["radius"], best["depth"]))
    fpk = evaluate(tc, lambda e: e["centres"])
    print(f"\nT_clean ({len(tc)} structures)")
    print(f"  fpocket native : top1_4={100*fpk['top1_dca4']:5.1f}%  top3_4={100*fpk['top3_dca4']:5.1f}%  med={fpk['median_dca1']:5.2f}")
    print(f"  GGNN re-scored : top1_4={100*fin['top1_dca4']:5.1f}%  top3_4={100*fin['top3_dca4']:5.1f}%  med={fin['median_dca1']:5.2f}")
    for k in ("top1_dca8", "top5_dca4", "top5_dca8"):
        print(f"    {k:11s} fpocket {100*fpk[k]:5.1f}%   rescored {100*fin[k]:5.1f}%")

    json.dump(dict(val_grid=rows, val_fpocket_native=base, selected=best,
                   T_clean_rescored=fin, T_clean_fpocket_native=fpk),
              open(OUT / "results/rescore_pockets.json", "w"), indent=1)


if __name__ == "__main__":
    main()
