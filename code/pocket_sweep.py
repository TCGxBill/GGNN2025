#!/usr/bin/env python3
"""
Tune the residue->pocket clustering on the VALIDATION split, then apply the winner once to T_clean.

The first pocket evaluation used the model's locked operating threshold as the confidence cut and
8 A single-linkage, and produced Top-3 == Top-5 == Top-1 == 17.5%: almost every structure yielded a
single pocket, so the ranking had nothing to rank. That is a post-processing failure, not a model
failure -- Top-1 DCA <= 8 A was already 49.6%, i.e. the centre was in the right region half the time.

Two cut modes are swept, because a fixed probability cut cannot adapt to protein size:
  abs   -- residues with p >= c
  frac  -- the top c fraction of residues by p (minimum 5)
Selection is on validation Top-1 DCA <= 4 A, the quantity the field reports. T_clean is scored once,
at the end, with the selected setting.
"""
import collections, itertools, json, os, sys
from pathlib import Path

import numpy as np
import torch
import yaml

SRC = Path("/Users/trongnhannguyenvu/Downloads/files-2/binding-site-prediction")
OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(OUT / "code"))
from pocket_eval import true_sites, pdb_path, success, p2rank_pockets

torch.set_num_threads(3)
TAG, GDIR = "B5_localframe", "data_localframe/combined"


def cluster(pos, prob, mode, c, link, min_size):
    if mode == "abs":
        sel = np.where(prob >= c)[0]
    else:
        k = max(5, int(round(c * len(prob))))
        sel = np.argsort(-prob)[:k]
    if len(sel) == 0:
        return []
    P, n = pos[sel], len(sel)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    for i, j in zip(*np.where(d <= link)):
        if i < j:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    g = collections.defaultdict(list)
    for i in range(n):
        g[find(i)].append(i)
    pk = []
    for mem in g.values():
        if len(mem) < min_size:
            continue
        idx = sel[mem]
        w = prob[idx]
        pk.append((float(w.sum()), (pos[idx] * w[:, None]).sum(0) / w.sum()))
    pk.sort(key=lambda t: -t[0])
    return [c_ for _, c_ in pk]


def score_set(codes, model, idx):
    """Cache probabilities + coordinates + true sites once; the sweep then costs no forward passes."""
    cache = []
    for code in codes:
        pp = pdb_path(code)
        if pp is None or code not in idx:
            continue
        sites = true_sites(pp)
        if not sites:
            continue
        g = torch.load(idx[code], weights_only=False)
        if "n_ligands" in g:
            del g["n_ligands"]
        with torch.no_grad():
            p = torch.sigmoid(model(g)[0]).numpy().ravel()
        cache.append((code, g.pos.numpy(), p, sites))
    return cache


def evaluate(cache, mode, c, link, min_size):
    out = {f"top{n}_dca{int(cu)}": [] for n in (1, 3, 5) for cu in (4.0, 8.0)}
    d1, npk = [], []
    for _, pos, p, sites in cache:
        pk = cluster(pos, p, mode, c, link, min_size)
        npk.append(len(pk))
        for n in (1, 3, 5):
            for cu in (4.0, 8.0):
                out[f"top{n}_dca{int(cu)}"].append(bool(success(pk, sites, n, cu)))
        if pk:
            d1.append(float(min(np.linalg.norm(pk[0] - s) for s in sites)))
    r = {k: float(np.mean(v)) for k, v in out.items()}
    r["median_dca1"] = float(np.median(d1)) if d1 else float("nan")
    r["mean_n_pockets"] = float(np.mean(npk))
    return r


def main():
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

    val = score_set(sp["val"], model, idx)
    print(f"validation structures with a usable ligand: {len(val)}", flush=True)

    grid = ([("abs", c) for c in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)] +
            [("frac", c) for c in (0.02, 0.05, 0.08, 0.12, 0.20)])
    rows = []
    for (mode, c), link, ms in itertools.product(grid, (4.5, 6.0, 8.0), (3, 5, 8)):
        r = evaluate(val, mode, c, link, ms)
        rows.append(dict(mode=mode, cut=c, link=link, min_size=ms, **r))
        print(f"{mode:5s} cut={c:<5} link={link:<4} min={ms}  "
              f"top1_4={100*r['top1_dca4']:5.1f}%  top3_4={100*r['top3_dca4']:5.1f}%  "
              f"med={r['median_dca1']:5.2f}  npk={r['mean_n_pockets']:.1f}", flush=True)

    rows.sort(key=lambda r: (-r["top1_dca4"], -r["top3_dca4"]))
    best = rows[0]
    print(f"\nselected on validation: {best['mode']} cut={best['cut']} "
          f"link={best['link']} min_size={best['min_size']}  "
          f"val top1_dca4={100*best['top1_dca4']:.1f}%", flush=True)

    tc = score_set(sp["T_clean"], model, idx)
    final = evaluate(tc, best["mode"], best["cut"], best["link"], best["min_size"])
    print(f"\nT_clean with selected setting ({len(tc)} structures):")
    for k, v in final.items():
        print(f"  {k:16s} {100*v:6.1f}%" if k.startswith("top") else f"  {k:16s} {v:6.2f}")

    json.dump(dict(grid=rows, selected=best, T_clean=final),
              open(OUT / "results/pocket_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
