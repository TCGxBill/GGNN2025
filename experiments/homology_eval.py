#!/usr/bin/env python3
"""
Homology-aware evaluation (reviewer request, SLSB 2026).

Removes every benchmark protein whose sequence identity to any training/validation
protein is >= cutoff (MMseqs2, coverage of the shorter sequence >= 0.5), then reports
AUPRC as the primary metric alongside AUROC, F1 and MCC at the validation-locked threshold.

Sequences are recovered from the one-hot block of the saved graphs, so no PDB files are needed.

Requires MMseqs2 on PATH (macOS: brew install mmseqs2).

Usage:
    python experiments/homology_eval.py
    python experiments/homology_eval.py --cutoffs 0.3 0.5 0.9 --n_boot 1000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             f1_score, precision_score, recall_score, matthews_corrcoef)

from src.models.gcn_geometric import GeometricGNN

AA = 'ARNDCQEGHILKMFPSTWYV'  # same order as ProteinPreprocessor._extract_node_features

BENCHMARKS = [
    ('combined/test', 'Combined Test'),
    ('scpdb', 'scPDB'),
    ('pdbbind_refined', 'PDBbind Refined'),
    ('sc6k', 'SC6K'),
    ('coach420', 'COACH420'),
    ('cryptobench', 'CryptoBench'),
    ('dude_diverse', 'DUD-E Diverse'),
    ('moad_quality', 'Binding MOAD'),
    ('difficult_cases', 'Difficult Cases'),
]
EXTERNAL = ['scPDB', 'PDBbind Refined', 'SC6K', 'COACH420', 'CryptoBench', 'DUD-E Diverse']


def load_graphs(path):
    graphs = {}
    for f in sorted(Path(path).glob('*.pt')):
        try:
            graphs[f.stem] = torch.load(f, weights_only=False)
        except Exception as e:
            print(f'  skip {f.name}: {e}')
    return graphs


def to_sequence(graph):
    onehot = graph.x[:, :20]
    idx = onehot.argmax(dim=1).tolist()
    valid = (onehot.sum(dim=1) > 0).tolist()
    return ''.join(AA[i] if v else 'X' for i, v in zip(idx, valid))


def write_fasta(records, path):
    with open(path, 'w') as f:
        for name, seq in records:
            f.write(f'>{name}\n{seq}\n')


def max_identity(queries, targets, min_cov=0.5, max_evalue=1e-3, threads=0):
    """queries/targets: list of (id, seq). Returns {query_id: (max_identity, best_target)}."""
    if shutil.which('mmseqs') is None:
        sys.exit('MMseqs2 not found. Install: brew install mmseqs2  (or conda install -c bioconda mmseqs2)')

    #New tranfer
    with tempfile.TemporaryDirectory() as tmp:
        q, t, out = f'{tmp}/q.fa', f'{tmp}/t.fa', f'{tmp}/hits.tsv'
        # MMseqs2 parses FASTA identifiers at whitespace. Use unique, opaque
        # identifiers and restore the original names when reading its output.
        query_names = {f'q{i}': name for i, (name, _) in enumerate(queries)}
        target_names = {f't{i}': name for i, (name, _) in enumerate(targets)}
        write_fasta([(f'q{i}', seq) for i, (_, seq) in enumerate(queries)], q)
        write_fasta([(f't{i}', seq) for i, (_, seq) in enumerate(targets)], t)
        cmd = ['mmseqs', 'easy-search', q, t, out, f'{tmp}/work',
               '--exhaustive-search', '1', '-e', '10', '--min-seq-id', '0.0',
               '--format-output', 'query,target,fident,alnlen,qlen,tlen,evalue',
               '-v', '1']
        if threads:
            cmd += ['--threads', str(threads)]
        subprocess.run(cmd, check=True)

        best = {name: (0.0, None) for name, _ in queries}
        with open(out) as f:
            for line in f:
                qid, tid, fid, aln, ql, tl, ev = line.rstrip('\n').split('\t')
                qid, tid = query_names[qid], target_names[tid]
                if float(ev) > max_evalue:
                    continue
                if int(aln) / min(int(ql), int(tl)) < min_cov:
                    continue
                if float(fid) > best[qid][0]:
                    best[qid] = (float(fid), tid)
    return best


@torch.no_grad()
def predict(model, graphs):
    preds = {}
    for name, g in graphs.items():
        out, _ = model(g)
        preds[name] = (torch.sigmoid(out).numpy().ravel(), g.y.numpy().ravel())
    return preds


def concat(preds, names):
    p = np.concatenate([preds[n][0] for n in names])
    y = np.concatenate([preds[n][1] for n in names])
    return p, y


def locked_threshold(val_preds):
    p, y = concat(val_preds, list(val_preds))
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def metrics(preds, names, thr, n_boot, rng):
    p, y = concat(preds, names)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    b = (p >= thr).astype(int)
    res = {
        'n_proteins': len(names),
        'n_residues': int(len(y)),
        'prevalence': float(y.mean()),
        'auprc': float(average_precision_score(y, p)),
        'auroc': float(roc_auc_score(y, p)),
        'f1': float(f1_score(y, b, zero_division=0)),
        'precision': float(precision_score(y, b, zero_division=0)),
        'recall': float(recall_score(y, b, zero_division=0)),
        'mcc': float(matthews_corrcoef(y, b)),
    }
    res['auprc_over_baseline'] = res['auprc'] / res['prevalence']

    if n_boot:
        # protein-level bootstrap: residues of one protein are not independent
        aps, aucs = [], []
        names = np.array(names)
        for _ in range(n_boot):
            s = rng.choice(names, len(names), replace=True)
            bp, by = concat(preds, s)
            if 0 < by.sum() < len(by):
                aps.append(average_precision_score(by, bp))
                aucs.append(roc_auc_score(by, bp))
        res['auprc_ci95'] = [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))]
        res['auroc_ci95'] = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed')
    ap.add_argument('--config', default='config_optimized.yaml')
    ap.add_argument('--checkpoint', default='checkpoints_optimized/best_model.pth')
    ap.add_argument('--cutoffs', type=float, nargs='+', default=[0.3, 0.5, 0.9])
    ap.add_argument('--min_cov', type=float, default=0.5)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='results_optimized/homology_benchmark_results.json')
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    data = Path(args.data)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model = GeometricGNN(config['model'])
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    train = load_graphs(data / 'combined/train')
    val = load_graphs(data / 'combined/val')
    if not train or not val:
        sys.exit(f'No training graphs under {data}/combined/{{train,val}}. Run scripts/preprocess_all.py first.')
    print(f'Reference (train+val): {len(train) + len(val)} proteins')

    thr = locked_threshold(predict(model, val))
    print(f'Validation-locked threshold: {thr:.4f}')

    reference = [(f'ref|{k}', to_sequence(g)) for k, g in {**train, **val}.items()]
    ref_pdb = {k[:4].upper() for k in {**train, **val}}

    benches = {}
    for sub, name in BENCHMARKS:
        g = load_graphs(data / sub)
        if g:
            benches[name] = g
        else:
            print(f'  {name}: not found, skipped')

    queries = [(f'{b}|{k}', to_sequence(g)) for b, gs in benches.items() for k, g in gs.items()]
    print(f'Searching {len(queries)} benchmark proteins against reference with MMseqs2...')
    ident = max_identity(queries, reference, args.min_cov, threads=args.threads)

    results = {'threshold': thr, 'min_cov': args.min_cov, 'cutoffs': args.cutoffs, 'benchmarks': {}}
    for bname, gs in benches.items():
        preds = predict(model, gs)
        ids = {k: ident[f'{bname}|{k}'][0] for k in gs}
        pdb_hit = {k for k in gs if k[:4].upper() in ref_pdb}
        entry = {
            'max_identity': {k: round(v, 3) for k, v in ids.items()},
            'all': metrics(preds, list(gs), thr, 0, rng),
            'pdb_id_filter': metrics(preds, [k for k in gs if k not in pdb_hit], thr, 0, rng),
        }
        for c in args.cutoffs:
            keep = [k for k in gs if k not in pdb_hit and ids[k] < c]
            entry[f'id<{c:g}'] = metrics(preds, keep, thr, args.n_boot, rng) if keep else None
        results['benchmarks'][bname] = entry

    for c in args.cutoffs:
        key = f'id<{c:g}'
        print(f'\nSequence identity < {c:.0%} to train+val (coverage >= {args.min_cov:.0%})')
        print(f'{"Benchmark":<17}{"Kept":>9}{"Prev":>8}{"AUPRC":>8}{"95% CI":>16}{"AUROC":>8}{"F1":>7}{"MCC":>7}')
        ext = []
        for bname, e in results['benchmarks'].items():
            total = e['all']['n_proteins'] if e['all'] else 0
            m = e[key]
            if m is None:
                print(f'{bname:<17}{0:>4}/{total:<4}{"n/a":>8}')
                continue
            ci = m.get('auprc_ci95', [np.nan, np.nan])
            print(f'{bname:<17}{m["n_proteins"]:>4}/{total:<4}{m["prevalence"]:>8.3f}{m["auprc"]:>8.3f}'
                  f'   [{ci[0]:.3f}-{ci[1]:.3f}]{m["auroc"]:>8.3f}{m["f1"]:>7.3f}{m["mcc"]:>7.3f}')
            if bname in EXTERNAL:
                ext.append(m)
        if ext:
            results[f'mean_external_{key}'] = {
                'n_benchmarks': len(ext),
                'auprc': float(np.mean([m['auprc'] for m in ext])),
                'auroc': float(np.mean([m['auroc'] for m in ext])),
                'f1': float(np.mean([m['f1'] for m in ext])),
                'mcc': float(np.mean([m['mcc'] for m in ext])),
            }
            r = results[f'mean_external_{key}']
            print(f'{"Mean external":<17}{"":>17}{r["auprc"]:>8.3f}{"":>16}{r["auroc"]:>8.3f}{r["f1"]:>7.3f}{r["mcc"]:>7.3f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
