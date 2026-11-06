#!/usr/bin/env python3
"""
Score GGNN 2025, P2Rank and Fpocket on ONE split, against the strict labels stored in
data_strict/, and report pooled metrics plus paired protein-level bootstrap intervals.

Why this exists. The earlier baseline numbers covered only T_clean, and the P2Rank
predictions they were built from lived in /tmp and are gone. More importantly, a user
querying a protein that *does* have close homologues with solved structures gets the
T_leaky number, not the T_clean one -- so both regimes are deployment-relevant and both
need all three methods on them.

Read the comparison carefully in the homologous regime: GGNN saw homologues of those
proteins during training and P2Rank did not, so any GGNN lead there is a measure of
training exposure, not of architecture. The honest reading is the SIZE OF THE GAP
between regimes, per method.

    python code/baselines_on_split.py --split T_leaky
    python code/baselines_on_split.py --split T_clean

Writes results/baselines_<split>.json and results/baselines_<split>.npz.
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, tempfile
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GGNN_SRC", ROOT))
OUT = Path(os.environ.get("GGNN_OUT", ROOT))
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(OUT / "code"))
from relabel_strict import find_pdb

CONDA = Path.home() / ".claude-science/conda/envs/baselines"
FPOCKET = CONDA / "bin/fpocket"
JAVA_HOME = CONDA / "lib/jvm"
PRANK = OUT / "baselines/p2rank_2.4.1/prank"
CKPT = OUT / "checkpoints/B5_strict_lf.pth"
torch.set_num_threads(8)


# ---------------------------------------------------------------- residue keys

def pdb_key(line):
    """(chain, resnum, insertion code) -- the insertion code MUST be kept.

    Dropping it merges residues such as 100A/100B onto one key, which silently
    mis-assigns baseline scores for a fraction of a percent of residues. That is
    small enough to miss and large enough to shift AUPRC in the third decimal.
    """
    try:
        return (line[21], int(line[22:26]), line[26].strip())
    except ValueError:
        return None


def label_key(chain, label):
    """Same key from a P2Rank residue_label such as '100' or '100A'."""
    m = re.match(r"\s*(-?\d+)\s*([A-Za-z]?)\s*$", str(label))
    return (chain, int(m.group(1)), m.group(2).strip()) if m else None


def ca_keys(path):
    """Residue key for each Calpha record, in file order, de-duplicated."""
    out, seen = [], set()
    for line in open(path, errors="ignore"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            k = pdb_key(line)
            if k is None or k in seen:
                continue
            seen.add(k)
            out.append((k, (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    return out


def align_keys(path, pos):
    """Map each graph residue to its (chain, resseq) key; fall back to coordinates."""
    cas = ca_keys(path)
    if len(cas) == len(pos):
        return [k for k, _ in cas]
    keymap = {tuple(np.round(c, 2)): k for k, c in cas}
    return [keymap.get(tuple(np.round(c, 2))) for c in pos]


# ---------------------------------------------------------------- baselines

def fpocket_ranks(path, workdir):
    """{(chain,resseq): best (lowest) pocket rank}, 1-based."""
    tmp = Path(workdir) / path.stem
    tmp.mkdir(parents=True, exist_ok=True)
    local = tmp / path.name
    shutil.copy(path, local)
    try:
        subprocess.run([str(FPOCKET), "-f", str(local)], capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    pockets = tmp / f"{path.stem}_out" / "pockets"
    if not pockets.exists():
        return None
    best = {}
    for f in pockets.glob("pocket*_atm.pdb"):
        m = re.match(r"pocket(\d+)_atm", f.stem)
        if not m:
            continue
        rank = int(m.group(1)) + 1
        for line in open(f, errors="ignore"):
            if line.startswith("ATOM"):
                k = pdb_key(line)
                if k is None:
                    continue
                if k not in best or rank < best[k]:
                    best[k] = rank
    return best


def run_p2rank(paths, outdir):
    """Batch mode: one JVM for the whole split. Returns the output directory."""
    ds = outdir / "split.ds"
    ds.write_text("\n".join(str(p) for p in paths) + "\n")
    env = dict(os.environ, JAVA_HOME=str(JAVA_HOME),
               PATH=f"{JAVA_HOME}/bin:{os.environ.get('PATH','')}")
    r = subprocess.run([str(PRANK), "predict", str(ds), "-o", str(outdir), "-threads", "6"],
                       capture_output=True, text=True, cwd=str(PRANK.parent), env=env)
    print("   p2rank exit", r.returncode, r.stdout.strip().splitlines()[-1][:90] if r.stdout else "")
    return outdir


def p2rank_residue_scores(outdir, code, path):
    """{(chain,resseq): probability} from P2Rank's own per-residue output."""
    for cand in (outdir / f"{path.name}_residues.csv", outdir / f"{code}.pdb_residues.csv"):
        if cand.exists():
            out = {}
            with open(cand) as fh:
                for row in csv.DictReader(fh):
                    row = {k.strip(): (v.strip() if isinstance(v, str) else v)
                           for k, v in row.items()}
                    try:
                        k = label_key(row["chain"], row["residue_label"])
                        if k is not None:
                            out[k] = float(row["probability"])
                    except (KeyError, ValueError):
                        continue
            return out
    return None


# ---------------------------------------------------------------- metrics

def boot(yl, al, bl, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    nP = len(yl); d = []
    for _ in range(n):
        pick = rng.integers(0, nP, nP)
        Y = np.concatenate([yl[i] for i in pick])
        if Y.sum() == 0 or Y.sum() == len(Y):
            continue
        d.append(metric(Y, np.concatenate([al[i] for i in pick]))
                 - metric(Y, np.concatenate([bl[i] for i in pick])))
    d = np.array(d)
    return dict(mean=float(d.mean()), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), frac_positive=float((d > 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="T_leaky")
    a = ap.parse_args()

    sp = json.load(open(OUT / "splits/splits_clean.json"))
    gidx = {}
    for s in ("train", "val", "test"):
        for q in (OUT / "data_strict/combined" / s).glob("*.pt"):
            gidx.setdefault(q.stem, q)
    codes = [c for c in sp[a.split] if c in gidx]
    paths = {c: find_pdb(c) for c in codes}
    codes = [c for c in codes if paths[c] is not None]
    print(f"{a.split}: {len(codes)} proteins with a strict graph and a PDB file", flush=True)

    cfg = yaml.safe_load(open(SRC / "config_optimized.yaml"))
    from src.models.gcn_geometric import GeometricGNN
    ck = torch.load(CKPT, weights_only=False)
    model = GeometricGNN(dict(cfg["model"])); model.load_state_dict(ck["model_state_dict"]); model.eval()

    with tempfile.TemporaryDirectory() as wd:
        p2dir = Path(wd) / "p2"; p2dir.mkdir()
        print("running P2Rank (batch)...", flush=True)
        run_p2rank([paths[c] for c in codes], p2dir)

        print("running Fpocket + GGNN per protein...", flush=True)
        Y, G, P, F, keep_codes, dropped = [], [], [], [], [], []
        for i, c in enumerate(codes, 1):
            g = torch.load(gidx[c], weights_only=False)
            if "n_ligands" in g:
                del g["n_ligands"]
            y = g.y.numpy().ravel()
            keys = align_keys(paths[c], g.pos.numpy())
            p2 = p2rank_residue_scores(p2dir, c, paths[c])
            fr = fpocket_ranks(paths[c], wd)
            if p2 is None or fr is None:
                dropped.append((c, "p2rank" if p2 is None else "fpocket")); continue
            npk = max(fr.values()) if fr else 1
            fs = np.array([0.0 if (k is None or k not in fr)
                           else 1.0 - (fr[k] - 1) / max(npk, 1) for k in keys])
            ps = np.array([0.0 if (k is None or k not in p2) else p2[k] for k in keys])
            with torch.no_grad():
                gs = torch.sigmoid(model(g)[0]).numpy().ravel()
            Y.append(y); G.append(gs); P.append(ps); F.append(fs); keep_codes.append(c)
            if i % 40 == 0:
                print(f"   {i}/{len(codes)} ok={len(Y)}", flush=True)

    yv, gv, pv, fv = (np.concatenate(x) for x in (Y, G, P, F))
    res = {"split": a.split, "checkpoint": CKPT.name, "n_proteins": len(Y),
           "n_residues": int(len(yv)), "prevalence": float(yv.mean()),
           "dropped": dropped, "pooled": {}, "per_protein": {}}
    print(f"\nscored {len(Y)} proteins, {len(yv)} residues, prevalence {100*yv.mean():.3f}%")
    if dropped:
        print("dropped:", dropped[:10])

    print(f"\n{'method':9s} {'AUPRC':>8s} {'AUROC':>8s} {'perProtAUPRC':>13s}")
    for name, v, lst in (("GGNN", gv, G), ("P2Rank", pv, P), ("Fpocket", fv, F)):
        au, ar = average_precision_score(yv, v), roc_auc_score(yv, v)
        pp = [average_precision_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan
              for yy, ss in zip(Y, lst)]
        res["pooled"][name] = {"auprc": float(au), "auroc": float(ar)}
        res["per_protein"][name] = {"mean": float(np.nanmean(pp)), "sd": float(np.nanstd(pp, ddof=1))}
        print(f"{name:9s} {au:8.4f} {ar:8.4f} {np.nanmean(pp):13.4f}")

    res["bootstrap"] = {}
    for nm, lst in (("P2Rank", P), ("Fpocket", F)):
        for mn, mf in (("auprc", average_precision_score), ("auroc", roc_auc_score)):
            k = f"GGNN_minus_{nm}_{mn}"
            res["bootstrap"][k] = boot(Y, G, lst, mf)
            b = res["bootstrap"][k]
            print(f"{k:28s} {b['mean']:+.4f}  95% CI [{b['lo']:+.4f}, {b['hi']:+.4f}]  "
                  f"P(>0)={b['frac_positive']:.3f}")

    json.dump(res, open(OUT / f"results/baselines_{a.split}.json", "w"), indent=1)
    np.savez(OUT / f"results/baselines_{a.split}.npz", y=yv, ggnn=gv, p2rank=pv, fpocket=fv,
             lens=np.array([len(x) for x in Y]), codes=np.array(keep_codes))
    print(f"\nwrote results/baselines_{a.split}.json")


if __name__ == "__main__":
    main()
