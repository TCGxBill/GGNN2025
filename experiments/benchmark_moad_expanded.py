#!/usr/bin/env python3
"""
Run benchmark on expanded MOAD dataset
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, matthews_corrcoef
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gcn_geometric import GeometricGNN


def evaluate_moad_expanded():
    print("=" * 60)
    print("EXPANDED MOAD BENCHMARK")
    print("=" * 60)
    
    # Load model
    with open('config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = GeometricGNN(config['model'])
    checkpoint = torch.load('checkpoints_optimized/best_model.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Combine both MOAD directories
    moad_dirs = [
        'data/processed/moad_quality',
        'data/processed/moad_expanded'
    ]
    
    all_labels = []
    all_scores = []
    n_proteins = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0
    
    with torch.no_grad():
        for moad_dir in moad_dirs:
            if not os.path.exists(moad_dir):
                continue
                
            for pt_file in os.listdir(moad_dir):
                if not pt_file.endswith('.pt'):
                    continue
                    
                try:
                    data = torch.load(os.path.join(moad_dir, pt_file), weights_only=False)
                    output = model(data)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    scores = torch.sigmoid(output).numpy().flatten()
                    labels = data.y.numpy().flatten()
                    
                    all_labels.extend(labels)
                    all_scores.extend(scores)
                    n_proteins += 1
                    
                    # Pocket-level
                    true_binding = set(np.where(labels == 1)[0])
                    if true_binding:
                        sorted_idx = np.argsort(scores)[::-1]
                        for n, count_var in [(1, 'top1'), (3, 'top3'), (5, 'top5'), (10, 'top10')]:
                            top_n = set(sorted_idx[:n])
                            if top_n & true_binding:
                                if n == 1: top1_hits += 1
                                elif n == 3: top3_hits += 1
                                elif n == 5: top5_hits += 1
                                elif n == 10: top10_hits += 1
                                
                except Exception as e:
                    print(f"  Error: {pt_file}: {e}")
    
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_scores)
    
    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    y_pred = (y_scores >= best_threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    prauc = auc(recall, precision)
    
    print(f"\n=== RESULTS: EXPANDED MOAD ({n_proteins} proteins) ===")
    print(f"  Residues: {len(y_true):,}")
    print(f"  Binding sites: {int(y_true.sum()):,} ({y_true.mean()*100:.2f}%)")
    print()
    print("  Residue-level:")
    print(f"    AUC:    {auc_score:.3f}")
    print(f"    PR-AUC: {prauc:.3f}")
    print(f"    F1:     {best_f1:.3f}")
    print(f"    MCC:    {mcc:.3f}")
    print()
    print("  Pocket-level:")
    print(f"    Top-1:  {top1_hits}/{n_proteins} = {top1_hits/n_proteins*100:.1f}%")
    print(f"    Top-3:  {top3_hits}/{n_proteins} = {top3_hits/n_proteins*100:.1f}%")
    print(f"    Top-5:  {top5_hits}/{n_proteins} = {top5_hits/n_proteins*100:.1f}%")
    print(f"    Top-10: {top10_hits}/{n_proteins} = {top10_hits/n_proteins*100:.1f}%")
    
    # Save results
    results = {
        'n_proteins': n_proteins,
        'n_residues': len(y_true),
        'n_binding': int(y_true.sum()),
        'auc': float(auc_score),
        'prauc': float(prauc),
        'f1': float(best_f1),
        'mcc': float(mcc),
        'top1': top1_hits/n_proteins,
        'top3': top3_hits/n_proteins,
        'top5': top5_hits/n_proteins,
        'top10': top10_hits/n_proteins
    }
    
    with open('results_optimized/moad_expanded_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: results_optimized/moad_expanded_benchmark.json")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    evaluate_moad_expanded()
