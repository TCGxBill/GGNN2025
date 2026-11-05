#!/usr/bin/env python3
"""
Verify the GGNN-vs-P2Rank comparison under strict (artifact-free) labels.

The headline pair -- GGNN 0.4611 against P2Rank 0.4703 -- was assembled from two different
code paths: the GGNN number comes from run_clean.py evaluating B5_strict_lf on data_strict
graphs, the P2Rank number from strict_labels.py recomputing labels on the fly. If those two
label constructions disagree anywhere, the comparison is not like-for-like and the sign of a
0.009 difference means nothing.

This script puts all three methods on ONE label vector, built once, and reports:
  1. label agreement between data_strict/*.pt y and a fresh strict_y() recomputation
  2. pooled AUPRC / AUROC for GGNN, P2Rank, Fpocket on identical residues
  3. per-protein mean AUPRC (the size-unbiased statistic; pooled is dominated by big proteins)
  4. paired protein-level bootstrap CI on GGNN - P2Rank, 2000 replicates
  5. the same comparison with the 17 T_clean proteins that share a 30% identity cluster with
     chen11 (P2Rank's training set) removed -- P2Rank's own homology control

Nothing here is retrained. Writes results/verify_strict_comparison.json.
"""
import collections, json, sys
from pathlib import Path
import numpy as np, torch, yaml
from sklearn.metrics import average_precision_score, roc_auc_score

SRC = Path("/Users/trongnhannguyenvu/Downloads/files-2/binding-site-prediction")
OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(OUT / "code"))
from strict_labels import strict_y
torch.set_num_threads(8)

CKPT = OUT / "checkpoints/B5_strict_lf.pth"


def main():
    cfg = yaml.safe_load(open(SRC / "config_optimized.yaml"))
    from src.models.gcn_geometric import GeometricGNN
    ck = torch.load(CKPT, weights_only=False)
    model = GeometricGNN(dict(cfg["model"])); model.load_state_dict(ck["model_state_dict"]); model.eval()
    print(f"checkpoint {CKPT.name}  locked_threshold={ck.get('threshold')}")

    # graph index over the tree the baselines were aligned to (same residue order as data_strict)
    idx_lf, idx_st = {}, {}
    for s in ("train", "val", "test"):
        for q in (OUT / "data_localframe/combined" / s).glob("*.pt"):
            idx_lf.setdefault(q.stem, q)
        for q in (OUT / "data_strict/combined" / s).glob("*.pt"):
            idx_st.setdefault(q.stem, q)
    sp = json.load(open(OUT / "splits/splits_clean.json"))

    codes, lens, y_old, y_new, gg, kept, has_strict_graph = [], [], [], [], [], [], []
    mismatch = []
    for code in sp["T_clean"]:
        if code not in idx_lf:
            continue
        g = torch.load(idx_lf[code], weights_only=False)
        if "n_ligands" in g:
            del g["n_ligands"]
        pos = g.pos.numpy()
        yn = strict_y(code, pos)
        with torch.no_grad():
            p = torch.sigmoid(model(g)[0]).numpy().ravel()
        # cross-check against the stored strict graph the model was trained/evaluated on
        if code in idx_st:
            gs = torch.load(idx_st[code], weights_only=False)
            ys = gs.y.numpy().ravel()
            if yn is None or len(ys) != len(yn) or not np.array_equal(ys, yn):
                mismatch.append(code)
            has_strict_graph.append(True)
        else:
            has_strict_graph.append(False)
        codes.append(code); lens.append(len(pos)); gg.append(p)
        y_old.append(g.y.numpy().ravel())
        y_new.append(yn if yn is not None else np.zeros(len(pos), dtype=np.int64))
        kept.append(yn is not None)

    fp = np.load(OUT / "results/fpocket_T_clean.npz"); p2 = np.load(OUT / "results/p2rank_T_clean.npz")
    assert np.array_equal(fp["y"], np.concatenate(y_old)), "fpocket label order mismatch"
    assert np.array_equal(p2["y"], np.concatenate(y_old)), "p2rank label order mismatch"

    keep = np.concatenate([np.full(l, k) for l, k in zip(lens, kept)])
    YN = np.concatenate(y_new)
    scores = {"GGNN": np.concatenate(gg), "P2Rank": p2["s"], "Fpocket": fp["s"]}

    print(f"\nproteins in T_clean loop      {len(codes)}")
    print(f"proteins with a genuine ligand {sum(kept)}")
    print(f"proteins present in data_strict {sum(has_strict_graph)}")
    print(f"label mismatches vs data_strict {len(mismatch)} {mismatch[:10]}")
    print(f"residues scored {int(keep.sum())}   prevalence {100*YN[keep].mean():.3f}%")

    out = {"n_proteins": int(sum(kept)), "n_residues": int(keep.sum()),
           "prevalence": float(YN[keep].mean()), "label_mismatches": mismatch,
           "checkpoint": CKPT.name, "pooled": {}, "per_protein": {}}

    print(f"\n{'method':9s} {'AUPRC':>8s} {'AUROC':>8s} {'perProtAUPRC':>13s} {'sd':>7s}")
    # per-protein slices
    cuts = np.cumsum([0] + lens)
    prot_slices = [(codes[i], slice(cuts[i], cuts[i+1])) for i in range(len(codes)) if kept[i]]
    per_prot = {}
    for m, s in scores.items():
        a = average_precision_score(YN[keep], s[keep]); r = roc_auc_score(YN[keep], s[keep])
        vals = []
        for c, sl in prot_slices:
            yy, ss = YN[sl], s[sl]
            vals.append(average_precision_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan)
        vals = np.array(vals, float)
        per_prot[m] = vals
        out["pooled"][m] = {"auprc": float(a), "auroc": float(r)}
        out["per_protein"][m] = {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1)),
                                 "n": int(np.isfinite(vals).sum())}
        print(f"{m:9s} {a:8.4f} {r:8.4f} {np.nanmean(vals):13.4f} {np.nanstd(vals, ddof=1):7.4f}")

    # paired protein-level bootstrap, GGNN - P2Rank
    rng = np.random.default_rng(0)
    nP = len(prot_slices)
    idx_arr = [sl for _, sl in prot_slices]
    yl = [YN[sl] for sl in idx_arr]
    gl = [scores["GGNN"][sl] for sl in idx_arr]
    pl = [scores["P2Rank"][sl] for sl in idx_arr]
    fl = [scores["Fpocket"][sl] for sl in idx_arr]

    def boot(a_list, b_list, metric):
        d = []
        for _ in range(2000):
            pick = rng.integers(0, nP, nP)
            Y = np.concatenate([yl[i] for i in pick])
            A = np.concatenate([a_list[i] for i in pick])
            B = np.concatenate([b_list[i] for i in pick])
            if Y.sum() == 0 or Y.sum() == len(Y):
                continue
            d.append(metric(Y, A) - metric(Y, B))
        d = np.array(d)
        return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                    hi=float(np.percentile(d, 97.5)), frac_positive=float((d > 0).mean()))

    out["bootstrap"] = {}
    for name, blist in (("P2Rank", pl), ("Fpocket", fl)):
        for mn, mf in (("auprc", average_precision_score), ("auroc", roc_auc_score)):
            k = f"GGNN_minus_{name}_{mn}"
            out["bootstrap"][k] = boot(gl, blist, mf)
            b = out["bootstrap"][k]
            print(f"{k:28s} {b['mean']:+.4f}  95% CI [{b['lo']:+.4f}, {b['hi']:+.4f}]  "
                  f"P(>0)={b['frac_positive']:.3f}")

    # chen11 control: drop T_clean proteins sharing a 30% cluster with P2Rank's training set
    import csv
    capath = OUT / "splits/cluster_assignment.csv"
    cl = {}
    with open(capath) as fh:
        for row in csv.DictReader(fh):
            cl[row.get("protein") or row.get("code") or row.get("id")] = row.get("cluster")
    chen_clusters = {c for k, c in cl.items() if k.startswith("EXP_chen11_")}
    drop = {c for c in codes if cl.get(c) in chen_clusters}
    print(f"\nchen11-sharing T_clean proteins: {len(drop)}")
    sel = [i for i, (c, _) in enumerate(prot_slices) if c not in drop]
    if sel and len(sel) < nP:
        Y = np.concatenate([yl[i] for i in sel])
        out["chen11_control"] = {"n_dropped": len(drop), "n_proteins": len(sel)}
        for m, lst in (("GGNN", gl), ("P2Rank", pl)):
            A = np.concatenate([lst[i] for i in sel])
            out["chen11_control"][m] = {"auprc": float(average_precision_score(Y, A)),
                                        "auroc": float(roc_auc_score(Y, A))}
            print(f"  {m:7s} AUPRC {out['chen11_control'][m]['auprc']:.4f}  "
                  f"AUROC {out['chen11_control'][m]['auroc']:.4f}")

    json.dump(out, open(OUT / "results/verify_strict_comparison.json", "w"), indent=1)
    print("\nwrote results/verify_strict_comparison.json")


if __name__ == "__main__":
    main()
