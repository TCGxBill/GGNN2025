#!/usr/bin/env python3
"""
Predict binding residues for a single PDB with a given checkpoint.

Each model runs with its OWN codebase via --root, so the old 29-dim model and
the new 32-dim model both work:

  new:  python scripts/predict_single.py --pdb 1hsg.pdb \
            --model runs/run1_new/best_model.pth
  old:  python scripts/predict_single.py --pdb 1hsg.pdb \
            --model backups_29/checkpoints_optimized/best_model.pth \
            --root backups_29

Writes a JSON with per-residue probabilities so old vs new can be diffed.
"""
import sys
import os
import argparse
import json
import yaml
import numpy as np
import torch


def load_model(model_path, config, device):
    from src.models.gcn_geometric import GeometricGNN
    model = GeometricGNN(config['model']).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    epoch = ckpt.get('epoch') if isinstance(ckpt, dict) else None
    return model, n_params, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--root', default='.', help='Project root that contains src/ and config (use backups_29 for old model)')
    ap.add_argument('--config', default=None, help='Config path (default: <root>/config_optimized.yaml)')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--top', type=int, default=None, help='Keep only top-N residues by probability')
    ap.add_argument('--out', default=None)
    ap.add_argument('--dist', type=float, default=6.0, help='Distance threshold; must match training preprocessing (preprocess_all uses 6.0)')
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    config_path = args.config or os.path.join(root, 'config_optimized.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.data.preprocessor import ProteinPreprocessor
    from src.data.graph_builder import ProteinGraphBuilder

    device = torch.device('cpu')
    gcfg = {'distance_threshold': args.dist, 'k_neighbors': 15}
    pre = ProteinPreprocessor(gcfg)
    gb = ProteinGraphBuilder(gcfg)

    data = pre.process_pdb(args.pdb)
    if data is None:
        print('Failed to process PDB'); sys.exit(1)
    residues = data['residues']
    labels = data['labels'] if data['labels'] is not None else np.zeros(len(residues), dtype=np.float32)
    graph = gb.build_graph(data['node_features'], data['coordinates'], labels)

    model, n_params, epoch = load_model(args.model, config, device)
    print(f"Model: {n_params:,} params | input_dim {config['model']['input_dim']} | ckpt epoch {epoch}")
    print(f"Protein: {os.path.basename(args.pdb)} | {len(residues)} residues")

    with torch.no_grad():
        out = model(graph.to(device))
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.sigmoid(out).squeeze(-1).cpu().numpy()

    items = []
    for i, res in enumerate(residues):
        items.append({
            'chain': res.get_parent().id,
            'resname': res.get_resname(),
            'resnum': int(res.get_id()[1]),
            'prob': float(probs[i]),
        })
    items.sort(key=lambda x: x['prob'], reverse=True)

    binding = [it for it in items if it['prob'] >= args.threshold]
    if args.top is not None:
        binding = binding[:args.top]

    print(f"\nBinding residues (>= {args.threshold}): {len(binding)} of {len(residues)}")
    for it in binding:
        print(f"  {it['resname']} {it['resnum']} (chain {it['chain']})  {it['prob']*100:.1f}%")

    result = {
        'pdb': os.path.basename(args.pdb),
        'model': args.model,
        'n_params': n_params,
        'input_dim': config['model']['input_dim'],
        'threshold': args.threshold,
        'n_residues': len(residues),
        'n_binding': len(binding),
        'binding': [f"{it['resname']}{it['resnum']}_{it['chain']}" for it in binding],
        'all_probs': items,
    }
    out_path = args.out or f"pred_{os.path.splitext(os.path.basename(args.pdb))[0]}_{n_params}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
