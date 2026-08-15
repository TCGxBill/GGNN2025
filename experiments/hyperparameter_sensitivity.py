#!/usr/bin/env python3
"""
EMPIRICAL Hyperparameter Sensitivity Analysis for GGNN 2025
Actually rebuilds graphs and evaluates - no hardcoded estimates
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from pathlib import Path
import yaml
from tqdm import tqdm
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def rebuild_graph(data, k_neighbors=None, distance_threshold=None):
    """Rebuild graph with new connectivity, keeping edge_attr compatible with model."""
    pos = data.pos  # Cα coordinates
    n = pos.shape[0]
    
    if n < 5:
        return None
    
    # Calculate pairwise distances
    dist_matrix = torch.cdist(pos, pos)
    
    # Build new edge_index based on k_neighbors or threshold
    if k_neighbors is not None:
        # K-NN graph
        _, indices = torch.topk(dist_matrix, k=min(k_neighbors+1, n), largest=False)
        row = torch.arange(n).unsqueeze(1).expand(-1, min(k_neighbors+1, n)).flatten()
        col = indices.flatten()
        # Remove self-loops
        mask = row != col
        row, col = row[mask], col[mask]
    elif distance_threshold is not None:
        # Threshold-based graph
        row, col = torch.where((dist_matrix < distance_threshold) & (dist_matrix > 0))
    else:
        return data
    
    if len(row) == 0:
        return None
    
    edge_index = torch.stack([row, col])
    
    # Build edge_attr with same dimensions as original (20-dim)
    # [distance(1), direction(3), distance_bins(16)] = 20
    distances = dist_matrix[row, col].unsqueeze(1)  # [E, 1]
    
    # Normalized direction vectors
    direction = pos[col] - pos[row]  # [E, 3]
    direction = direction / (distances + 1e-8)
    
    # Distance bins (16 bins from 0-20Å)
    bins = torch.linspace(0, 20, 17)
    bin_idx = torch.bucketize(distances.squeeze(), bins) - 1
    bin_idx = bin_idx.clamp(0, 15)
    bin_onehot = torch.zeros(len(distances), 16)
    bin_onehot.scatter_(1, bin_idx.unsqueeze(1), 1.0)
    
    edge_attr = torch.cat([distances, direction, bin_onehot], dim=1)  # [E, 20]
    
    # Create new data object
    new_data = Data(
        x=data.x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=data.y,
        pos=pos
    )
    
    return new_data


def evaluate_with_rebuild(model, data_dir, k_neighbors=None, distance_threshold=None, max_samples=100):
    """Evaluate model with rebuilt graphs."""
    model.eval()
    all_labels = []
    all_scores = []
    success_count = 0
    
    pt_files = list(Path(data_dir).glob("*.pt"))[:max_samples]
    
    with torch.no_grad():
        for pt_file in tqdm(pt_files, desc=f"k={k_neighbors} t={distance_threshold}", leave=False):
            try:
                data = torch.load(pt_file, weights_only=False)
                
                # Rebuild graph
                new_data = rebuild_graph(data, k_neighbors=k_neighbors, distance_threshold=distance_threshold)
                
                if new_data is None or new_data.edge_index.shape[1] == 0:
                    continue
                
                output = model(new_data)
                if isinstance(output, tuple):
                    output = output[0]
                
                scores = torch.sigmoid(output).numpy().flatten()
                labels = new_data.y.numpy().flatten()
                
                if len(scores) != len(labels):
                    continue
                
                all_labels.extend(labels)
                all_scores.extend(scores)
                success_count += 1
            except Exception as e:
                continue
    
    if len(all_labels) < 100:
        return None
    
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred = (y_scores >= 0.5).astype(int)
    
    # Full metrics
    from sklearn.metrics import precision_score, recall_score
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'auc': float(roc_auc_score(y_true, y_scores)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'n_proteins': success_count,
        'n_residues': len(y_true)
    }


def main():
    print("=" * 70)
    print("EMPIRICAL HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("(Actually testing, not estimated!)")
    print("=" * 70)
    
    # Load model
    with open('config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from models.gcn_geometric import GeometricGNN
    model = GeometricGNN(config['model'])
    ckpt = torch.load('checkpoints_optimized/best_model.pth', 
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    results = {'k_neighbors': {}, 'distance_threshold': {}}
    data_dir = 'data/processed/combined/val'
    max_samples = 150  # Use more samples for robust results
    
    # ======== k_neighbors Sensitivity ========
    print("\n[1] k_neighbors Sensitivity (EMPIRICAL)")
    print("-" * 50)
    
    k_values = [10, 15, 20, 30]
    for k in k_values:
        print(f"\nTesting k={k}...")
        metrics = evaluate_with_rebuild(model, data_dir, k_neighbors=k, max_samples=max_samples)
        if metrics:
            results['k_neighbors'][str(k)] = metrics
            print(f"  k={k}: AUC={metrics['auc']:.4f}, MCC={metrics['mcc']:.4f} ({metrics['n_proteins']} proteins)")
        else:
            print(f"  k={k}: FAILED")
    
    # ======== Distance Threshold Sensitivity ========
    print("\n[2] Distance Threshold Sensitivity (EMPIRICAL)")
    print("-" * 50)
    
    threshold_values = [6.0, 8.0, 10.0, 12.0]
    for t in threshold_values:
        print(f"\nTesting threshold={t}Å...")
        metrics = evaluate_with_rebuild(model, data_dir, distance_threshold=t, max_samples=max_samples)
        if metrics:
            results['distance_threshold'][str(t)] = metrics
            print(f"  {t}Å: AUC={metrics['auc']:.4f}, MCC={metrics['mcc']:.4f} ({metrics['n_proteins']} proteins)")
        else:
            print(f"  {t}Å: FAILED")
    
    # ======== Summary Tables ========
    print("\n" + "=" * 70)
    print("SUMMARY TABLES")
    print("=" * 70)
    
    # k_neighbors table
    if results['k_neighbors']:
        print("\nk-neighbors sensitivity:")
        print(f"{'k':<10} {'AUC':<10} {'MCC':<10} {'n_proteins':<10}")
        print("-" * 40)
        baseline_auc = results['k_neighbors'].get('15', {}).get('auc', 0)
        for k in sorted(results['k_neighbors'].keys(), key=int):
            m = results['k_neighbors'][k]
            delta = (m['auc'] - baseline_auc) * 100 if baseline_auc else 0
            marker = ' *' if k == '15' else ''
            print(f"{k:<10} {m['auc']:.4f}     {m['mcc']:.4f}     {m['n_proteins']:<10}{marker}")
    
    # threshold table
    if results['distance_threshold']:
        print("\nDistance threshold sensitivity:")
        print(f"{'Threshold':<12} {'AUC':<10} {'MCC':<10} {'n_proteins':<10}")
        print("-" * 42)
        baseline_auc_t = results['distance_threshold'].get('8.0', {}).get('auc', 0)
        for t in sorted(results['distance_threshold'].keys(), key=float):
            m = results['distance_threshold'][t]
            delta = (m['auc'] - baseline_auc_t) * 100 if baseline_auc_t else 0
            marker = ' *' if t == '8.0' else ''
            print(f"{t}Å         {m['auc']:.4f}     {m['mcc']:.4f}     {m['n_proteins']:<10}{marker}")
    
    # Save results
    output_file = 'results_optimized/hyperparameter_sensitivity.json'
    os.makedirs('results_optimized', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
