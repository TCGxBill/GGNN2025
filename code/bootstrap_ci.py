#!/usr/bin/env python3
"""
Protein-level bootstrap confidence intervals for the three-method comparison.

Resampling residues treats the ~146,000 residues of T_clean as independent draws, which they are
not -- residues within one structure share a fold, a ligand and a label process -- and yields
intervals of about +/-0.002 that nobody should believe. The resampling unit here is the protein:
246 structures are drawn with replacement, their residues pooled, and AUPRC/AUROC recomputed.

Paired differences (GGNN - P2Rank) are computed within each bootstrap replicate, so the interval
on the difference accounts for the two methods seeing the same structures.
"""
import json, sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GGNN_SRC", ROOT))
OUT = Path(os.environ.get("GGNN_OUT", ROOT))
sys.path.insert(0, str(SRC))
torch.set_num_threads(2)
TAG, GDIR, NBOOT = "B5_localframe", "data_localframe/combined", 2000


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

    ys, gg, lens = [], [], []
    for code in sp["T_clean"]:
        if code not in idx:
            continue
        g = torch.load(idx[code], weights_only=False)
        if "n_ligands" in g:
            del g["n_ligands"]
        with torch.no_grad():
            p = torch.sigmoid(model(g)[0]).numpy().ravel()
        y = g.y.numpy().ravel()
        ys.append(y); gg.append(p); lens.append(len(y))

    total = sum(lens)
    fp = np.load(OUT / "results/fpocket_T_clean.npz")
    p2 = np.load(OUT / "results/p2rank_T_clean.npz")
    assert len(fp["y"]) == total, f"fpocket length {len(fp['y'])} != GGNN {total}"
    assert np.array_equal(fp["y"], np.concatenate(ys)), "fpocket label order differs from GGNN"
    assert np.array_equal(p2["y"], np.concatenate(ys)), "p2rank label order differs from GGNN"

    cuts = np.cumsum([0] + lens)
    per = {"GGNN": gg,
           "P2Rank": [p2["s"][cuts[i]:cuts[i + 1]] for i in range(len(lens))],
           "Fpocket": [fp["s"][cuts[i]:cuts[i + 1]] for i in range(len(lens))]}
    n = len(lens)
    print(f"proteins {n}   residues {total}   positives {int(np.concatenate(ys).sum())}", flush=True)

    rng = np.random.default_rng(0)
    boot = {m: {"auprc": [], "auroc": []} for m in per}
    diff = {"auprc": [], "auroc": []}
    for b in range(NBOOT):
        sel = rng.integers(0, n, n)
        Y = np.concatenate([ys[i] for i in sel])
        if Y.min() == Y.max():
            continue
        cur = {}
        for m, arr in per.items():
            S = np.concatenate([arr[i] for i in sel])
            a, r = average_precision_score(Y, S), roc_auc_score(Y, S)
            boot[m]["auprc"].append(a); boot[m]["auroc"].append(r)
            cur[m] = (a, r)
        diff["auprc"].append(cur["GGNN"][0] - cur["P2Rank"][0])
        diff["auroc"].append(cur["GGNN"][1] - cur["P2Rank"][1])
        if (b + 1) % 500 == 0:
            print(f"  {b+1}/{NBOOT}", flush=True)

    Yall = np.concatenate(ys)
    res = {}
    for m, arr in per.items():
        S = np.concatenate(arr)
        res[m] = {
            "auprc": float(average_precision_score(Yall, S)),
            "auprc_lo": float(np.percentile(boot[m]["auprc"], 2.5)),
            "auprc_hi": float(np.percentile(boot[m]["auprc"], 97.5)),
            "auroc": float(roc_auc_score(Yall, S)),
            "auroc_lo": float(np.percentile(boot[m]["auroc"], 2.5)),
            "auroc_hi": float(np.percentile(boot[m]["auroc"], 97.5)),
        }
        print(f"{m:8s} AUPRC {res[m]['auprc']:.4f} [{res[m]['auprc_lo']:.4f}, {res[m]['auprc_hi']:.4f}]   "
              f"AUROC {res[m]['auroc']:.4f} [{res[m]['auroc_lo']:.4f}, {res[m]['auroc_hi']:.4f}]")

    for k in ("auprc", "auroc"):
        d = np.array(diff[k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac = float((d > 0).mean())
        res[f"GGNN_minus_P2Rank_{k}"] = {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
                                         "frac_positive": frac}
        print(f"GGNN - P2Rank {k.upper():5s}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"GGNN higher in {100*frac:.1f}% of replicates")

    json.dump(res, open(OUT / "results/bootstrap_ci.json", "w"), indent=1)


if __name__ == "__main__":
    main()
