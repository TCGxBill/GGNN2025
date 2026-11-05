#!/usr/bin/env python3
"""
Evaluate a checkpoint on the eight external benchmarks, stratified by maximum sequence
identity to the model's OWN training set. Uses the validation-locked threshold stored in
the checkpoint. Read-only w.r.t. the original repository.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

SRC = Path("/Users/trongnhannguyenvu/Downloads/files-2/binding-site-prediction")
OUT = Path("/Users/trongnhannguyenvu/Downloads/files-2/ggnn-homology-run")
MMSEQS = SRC / "mmseqs/bin/mmseqs"
sys.path.insert(0, str(SRC))

import numpy as np, torch, yaml
from torch_geometric.loader import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef)
from src.models.gcn_geometric import GeometricGNN

AA = "ARNDCQEGHILKMFPSTWYV"
BENCH = ["scpdb", "pdbbind_refined", "sc6k", "coach420",
         "cryptobench", "dude_diverse", "moad_quality", "difficult_cases"]
CUTOFFS = [0.9, 0.5, 0.3]


def seq_of(graph):
    oh = graph.x[:, :20]
    idx = oh.argmax(dim=1).tolist()
    ok = (oh.sum(dim=1) > 0).tolist()
    return "".join(AA[i] if v else "X" for i, v in zip(idx, ok))


def max_identity(queries, targets, min_cov=0.5, max_evalue=1e-3, threads=4):
    """{query_id: max fractional identity to any target}, coverage of the shorter sequence."""
    with tempfile.TemporaryDirectory() as tmp:
        q, t, o = f"{tmp}/q.fa", f"{tmp}/t.fa", f"{tmp}/hits.tsv"
        qn = {f"q{i}": n for i, (n, _) in enumerate(queries)}
        for path, recs, pre in ((q, queries, "q"), (t, targets, "t")):
            with open(path, "w") as fh:
                for i, (_, s) in enumerate(recs):
                    fh.write(f">{pre}{i}\n{s}\n")
        subprocess.run([str(MMSEQS), "easy-search", q, t, o, f"{tmp}/w",
                        "--exhaustive-search", "1", "-e", "10", "--min-seq-id", "0.0",
                        "--format-output", "query,target,fident,alnlen,qlen,tlen,evalue",
                        "--threads", str(threads), "-v", "1"], check=True)
        best = {n: 0.0 for n, _ in queries}
        for line in open(o):
            qid, _, fid, aln, ql, tl, ev = line.rstrip("\n").split("\t")
            if float(ev) > max_evalue:
                continue
            if int(aln) / min(int(ql), int(tl)) < min_cov:
                continue
            name = qn[qid]
            best[name] = max(best[name], float(fid))
    return best


def score(model, graphs, thr):
    P, Y = [], []
    model.eval()
    with torch.no_grad():
        for data in DataLoader(graphs, batch_size=32, shuffle=False):
            out, _ = model(data)
            P.append(torch.sigmoid(out).numpy().ravel())
            Y.append(data.y.numpy().ravel())
    y, p = np.concatenate(Y), np.concatenate(P)
    pred = (p >= thr).astype(int)
    return dict(n_proteins=len(graphs), n_residues=int(len(y)), prevalence=float(y.mean()),
                auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                mcc=float(matthews_corrcoef(y, pred)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--graph-dir", default=None, help="alternative processed-graph tree")
    a = ap.parse_args()

    ck = torch.load(OUT / f"checkpoints/{a.tag}.pth", weights_only=False)
    thr = ck["threshold"]
    cfg = yaml.safe_load(open(SRC / "config_optimized.yaml"))
    model = GeometricGNN(cfg["model"])
    model.load_state_dict(ck["model_state_dict"])

    splits = json.load(open(OUT / "splits/splits_clean.json"))
    train_tag = json.load(open(OUT / f"results/{a.tag}.json"))["train_split"]

    base = Path(a.graph_dir) if a.graph_dir else (SRC / "data/processed/combined")
    idx = {}
    for s in ("train", "val", "test"):
        for p in (base / s).glob("*.pt"):
            idx.setdefault(p.stem, p)

    train_seqs = []
    for i in splits[train_tag]:
        if i in idx:
            g = torch.load(idx[i], weights_only=False)
            train_seqs.append((i, seq_of(g)))
    print(f"training sequences: {len(train_seqs)}", flush=True)

    results = {"tag": a.tag, "threshold": thr, "train_split": train_tag, "benchmarks": {}}
    for bm in BENCH:
        d = (Path(a.graph_dir).parent / bm) if a.graph_dir else (SRC / "data/processed" / bm)
        graphs, names = [], []
        for p in sorted(d.glob("*.pt")):
            try:
                g = torch.load(p, weights_only=False)
            except Exception:
                continue
            if g is None or not hasattr(g, "y") or g.y is None or g.x.shape[0] < 5:
                continue
            if "n_ligands" in g:
                del g["n_ligands"]
            if float(g.y.sum()) == 0:
                continue
            graphs.append(g); names.append(p.stem)
        if not graphs:
            continue
        ident = max_identity([(n, seq_of(g)) for n, g in zip(names, graphs)], train_seqs)
        entry = {"all": score(model, graphs, thr),
                 "max_identity": {n: round(ident[n], 3) for n in names}}
        for c in CUTOFFS:
            sel = [g for n, g in zip(names, graphs) if ident[n] < c]
            entry[f"id<{c}"] = score(model, sel, thr) if len(sel) >= 3 else None
        results["benchmarks"][bm] = entry
        line = " ".join(f"{k}:{(entry[k]['auprc'] if entry.get(k) else float('nan')):.3f}"
                        for k in ["all", "id<0.9", "id<0.5", "id<0.3"])
        print(f"{bm:18s} n={len(graphs):4d}  AUPRC  {line}", flush=True)

    json.dump(results, open(OUT / f"results/{a.tag}_benchmarks.json", "w"), indent=1)
    print("written", OUT / f"results/{a.tag}_benchmarks.json")


if __name__ == "__main__":
    main()
