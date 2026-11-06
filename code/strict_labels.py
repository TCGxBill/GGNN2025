#!/usr/bin/env python3
"""
Re-evaluate all three methods against labels that exclude crystallisation artifacts and modified
residues.

The all-ligand relabelling of section 4 fixed the original pipeline's "keep only the largest HETATM"
bug, but it kept every HETATM group with >=3 heavy atoms. Inspecting what that admits on T_clean:
glycerol (104 instances), selenomethionine (93), acetate (90), ethylene glycol (34), isopropanol,
cacodylate. Glycerol and ethylene glycol are cryoprotectants, acetate and cacodylate are buffers,
and selenomethionine is not a ligand at all -- it is a methionine substitution inside the
polypeptide chain, deposited as HETATM. Residues "contacting" these are not binding-site residues.

Counted on T_clean: 24.6% of the ligand instances feeding the labels are crystallisation additives
and 8.8% are modified residues -- a third of the positive signal is spurious.

This also biases the baseline comparison. P2Rank's training labels follow the BioLiP convention,
which excludes these groups, so scoring it against labels that include glycerol sites penalises it
for not predicting something it was never trained to predict.

Labels are recomputed here without retraining; all three methods are rescored on identical labels.
"""
import collections, json, sys
import numpy as np, torch, yaml
import os
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GGNN_SRC", ROOT))
OUT = Path(os.environ.get("GGNN_OUT", ROOT))
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(OUT / "code"))
from pocket_eval import pdb_path, EXCLUDE
torch.set_num_threads(6)

MODRES = {'MSE','SEP','TPO','PTR','CSO','CME','KCX','LLP','PCA','MLY','M3L','ALY','HYP',
          'CAS','CSD','OCS','CSX','SMC','MHO','NEP','SNN','TYS','ABA','AIB','DAL','FME'}
ARTIFACT = {'GOL','EDO','PEG','PG4','PGE','1PE','2PE','P6G','7PE','XPE','ACT','ACY','MES',
            'TRS','EPE','DMS','IPA','MPD','FMT','CIT','TLA','BME','DTT','CAC','MLI','BCT',
            'AZI','NO3','SCN','IMD','BU3','POP','MRD','PE4','PEU','SIN','TAR','MAE','OXL',
            'GLY','ACE','NH2','UNX','UNL','DOD','PGO','BU1','ETE','P33','MYR','OLC','LDA'}
CUTOFF = 6.0


def strict_y(code, pos):
    """Positive if within CUTOFF of a genuine biological ligand (>=3 heavy atoms)."""
    p = pdb_path(code)
    if p is None:
        return None
    g = collections.defaultdict(list)
    for line in open(p, errors="ignore"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("HETATM"):
            rn = line[17:20].strip().upper()
            if rn in EXCLUDE or rn in MODRES or rn in ARTIFACT:
                continue
            g[(line[21], line[22:27], rn)].append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    atoms = [a for v in g.values() if len(v) >= 3 for a in v]
    if not atoms:
        return None
    A = np.array(atoms, dtype=np.float32)
    d = np.linalg.norm(pos[:, None, :] - A[None, :, :], axis=2).min(1)
    return (d <= CUTOFF).astype(np.int64)


def main():
    cfg = yaml.safe_load(open(SRC / "config_optimized.yaml"))
    from src.models.gcn_geometric import GeometricGNN
    ck = torch.load(OUT / "checkpoints/B5_localframe.pth", weights_only=False)
    model = GeometricGNN(dict(cfg["model"])); model.load_state_dict(ck["model_state_dict"]); model.eval()

    idx = {}
    for s in ("train", "val", "test"):
        for q in (OUT / "data_localframe/combined" / s).glob("*.pt"):
            idx.setdefault(q.stem, q)
    sp = json.load(open(OUT / "splits/splits_clean.json"))

    y_old, y_new, gg, lens, kept = [], [], [], [], []
    for code in sp["T_clean"]:
        if code not in idx:
            continue
        g = torch.load(idx[code], weights_only=False)
        if "n_ligands" in g:
            del g["n_ligands"]
        pos = g.pos.numpy()
        yn = strict_y(code, pos)
        with torch.no_grad():
            p = torch.sigmoid(model(g)[0]).numpy().ravel()
        y_old.append(g.y.numpy().ravel()); gg.append(p); lens.append(len(pos))
        y_new.append(yn if yn is not None else np.zeros(len(pos), dtype=np.int64))
        kept.append(yn is not None)

    cuts = np.cumsum([0] + lens)
    fp = np.load(OUT / "results/fpocket_T_clean.npz"); p2 = np.load(OUT / "results/p2rank_T_clean.npz")
    assert np.array_equal(fp["y"], np.concatenate(y_old)), "label order mismatch"

    keep = np.concatenate([np.full(l, k) for l, k in zip(lens, kept)])
    scores = {"GGNN": np.concatenate(gg), "P2Rank": p2["s"], "Fpocket": fp["s"]}
    YO, YN = np.concatenate(y_old), np.concatenate(y_new)

    n_drop = sum(1 for k in kept if not k)
    print(f"proteins retained {sum(kept)} of {len(kept)}  (dropped {n_drop}: no genuine ligand left)")
    print(f"positives  all-ligand labels {int(YO.sum()):6d} ({100*YO.mean():.2f}%)   "
          f"strict labels {int(YN[keep].sum()):6d} ({100*YN[keep].mean():.2f}%)")
    print(f"positives removed: {int(YO[keep].sum()-YN[keep].sum())} "
          f"({100*(YO[keep].sum()-YN[keep].sum())/YO[keep].sum():.1f}% of them)\n")

    print(f"{'method':9s} {'AUPRC(all-lig)':>15s} {'AUPRC(strict)':>14s} {'delta':>8s}   "
          f"{'AUROC(all)':>11s} {'AUROC(strict)':>14s}")
    res = {}
    for m, s in scores.items():
        a0 = average_precision_score(YO, s); a1 = average_precision_score(YN[keep], s[keep])
        r0 = roc_auc_score(YO, s);           r1 = roc_auc_score(YN[keep], s[keep])
        res[m] = dict(auprc_all=a0, auprc_strict=a1, auroc_all=r0, auroc_strict=r1)
        print(f"{m:9s} {a0:15.4f} {a1:14.4f} {100*(a1-a0)/a0:+7.1f}%   {r0:11.4f} {r1:14.4f}")

    json.dump(res, open(OUT / "results/strict_label_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
