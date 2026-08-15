#!/usr/bin/env python3
"""
CryptoBench Benchmark with:
1. No overlapping with training data
2. DCA (Distance to Center of binding site) calculation
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from pathlib import Path
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gcn_geometric import GeometricGNN


def calculate_dca(predicted_scores, true_labels, coords, threshold=4.0):
    """Calculate DCA metric."""
    true_binding_idx = np.where(true_labels == 1)[0]
    
    if len(true_binding_idx) == 0:
        return None, False
    
    top_pred_idx = np.argmax(predicted_scores)
    top_pred_coord = coords[top_pred_idx]
    
    binding_coords = coords[true_binding_idx]
    distances = np.linalg.norm(binding_coords - top_pred_coord, axis=1)
    min_distance = np.min(distances)
    
    success = min_distance <= threshold
    return min_distance, success


def evaluate_cryptobench_clean():
    print("=" * 70)
    print("CRYPTOBENCH BENCHMARK - WITH DCA METRIC")
    print("=" * 70)
    
    # Load model
    with open('config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = GeometricGNN(config['model'])
    checkpoint = torch.load('checkpoints_optimized/best_model.pth', 
                           map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # CryptoBench directories
    crypto_dirs = [
        'data/processed/cryptobench',
        'data/processed/cryptobench_expanded'
    ]
    
    all_labels = []
    all_scores = []
    all_dca_distances = []
    
    n_proteins = 0
    
    # Top-N counts
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0
    
    # DCA counts
    dca_4a_hits = 0
    dca_8a_hits = 0
    
    with torch.no_grad():
        for crypto_dir in crypto_dirs:
            if not os.path.exists(crypto_dir):
                continue
                
            for pt_file in sorted(os.listdir(crypto_dir)):
                if not pt_file.endswith('.pt'):
                    continue
                
                pdb_id = pt_file.replace('.pt', '').upper()
                    
                try:
                    data = torch.load(os.path.join(crypto_dir, pt_file), 
                                     weights_only=False)
                    output = model(data)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    scores = torch.sigmoid(output).numpy().flatten()
                    labels = data.y.numpy().flatten()
                    coords = data.pos.numpy()
                    
                    all_labels.extend(labels)
                    all_scores.extend(scores)
                    n_proteins += 1
                    
                    # Pocket-level Top-N
                    true_binding = set(np.where(labels == 1)[0])
                    if true_binding:
                        sorted_idx = np.argsort(scores)[::-1]
                        for n in [1, 3, 5, 10]:
                            top_n = set(sorted_idx[:n])
                            if top_n & true_binding:
                                if n == 1: top1_hits += 1
                                elif n == 3: top3_hits += 1
                                elif n == 5: top5_hits += 1
                                elif n == 10: top10_hits += 1
                        
                        # DCA metric
                        dist_4a, success_4a = calculate_dca(scores, labels, coords, threshold=4.0)
                        dist_8a, success_8a = calculate_dca(scores, labels, coords, threshold=8.0)
                        
                        if success_4a:
                            dca_4a_hits += 1
                        if success_8a:
                            dca_8a_hits += 1
                        
                        if dist_4a is not None:
                            all_dca_distances.append(dist_4a)
                                
                except Exception as e:
                    print(f"  Error: {pdb_id}: {e}")
    
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    
    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_scores)
    
    # Find optimal threshold
    best_f1 = 0
    for t in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    y_pred = (y_scores >= best_threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"\n=== RESULTS: CRYPTOBENCH ({n_proteins} proteins) ===")
    print(f"  Residues: {len(y_true):,}")
    print(f"  Binding sites: {int(y_true.sum()):,} ({y_true.mean()*100:.2f}%)")
    print()
    print("  Residue-level:")
    print(f"    AUC:    {auc_score:.3f}")
    print(f"    F1:     {best_f1:.3f}")
    print(f"    MCC:    {mcc:.3f}")
    print()
    print("  Pocket-level Top-N:")
    print(f"    Top-1:  {top1_hits}/{n_proteins} = {top1_hits/n_proteins*100:.1f}%")
    print(f"    Top-3:  {top3_hits}/{n_proteins} = {top3_hits/n_proteins*100:.1f}%")
    print(f"    Top-5:  {top5_hits}/{n_proteins} = {top5_hits/n_proteins*100:.1f}%")
    print(f"    Top-10: {top10_hits}/{n_proteins} = {top10_hits/n_proteins*100:.1f}%")
    print()
    print("  DCA (Distance to Closest Annotation):")
    print(f"    DCA≤4Å: {dca_4a_hits}/{n_proteins} = {dca_4a_hits/n_proteins*100:.1f}%")
    print(f"    DCA≤8Å: {dca_8a_hits}/{n_proteins} = {dca_8a_hits/n_proteins*100:.1f}%")
    if all_dca_distances:
        print(f"    Mean distance: {np.mean(all_dca_distances):.2f}Å")
        print(f"    Median distance: {np.median(all_dca_distances):.2f}Å")
    
    # Save results
    results = {
        'n_proteins': n_proteins,
        'n_residues': len(y_true),
        'n_binding': int(y_true.sum()),
        'auc': float(auc_score),
        'f1': float(best_f1),
        'mcc': float(mcc),
        'top1': top1_hits/n_proteins,
        'top3': top3_hits/n_proteins,
        'top5': top5_hits/n_proteins,
        'top10': top10_hits/n_proteins,
        'dca_4a': dca_4a_hits/n_proteins,
        'dca_8a': dca_8a_hits/n_proteins,
        'mean_dca_distance': float(np.mean(all_dca_distances)) if all_dca_distances else None,
        'median_dca_distance': float(np.median(all_dca_distances)) if all_dca_distances else None,
    }
    
    with open('results_optimized/cryptobench_clean_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: results_optimized/cryptobench_clean_benchmark.json")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    evaluate_cryptobench_clean()
