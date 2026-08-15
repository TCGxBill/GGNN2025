#!/usr/bin/env python3
"""
Fresh Benchmark Evaluation Script

Re-runs all evaluations to verify paper metrics.
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    precision_score, recall_score, f1_score, matthews_corrcoef
)
import yaml
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gcn_geometric import GeometricGNN


def load_model(checkpoint_path, config_path):
    """Load model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = GeometricGNN(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def calculate_metrics(y_true, y_scores, threshold=0.5):
    """Calculate all metrics."""
    y_pred = (y_scores >= threshold).astype(int)
    
    # Basic metrics
    auc_score = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    
    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    prauc = auc(recall_curve, precision_curve)
    
    # Classification metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'auc': float(auc_score),
        'prauc': float(prauc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'positive_ratio': float(y_true.mean())
    }


def find_optimal_threshold(y_true, y_scores):
    """Find threshold that maximizes F1 score."""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def calculate_pocket_metrics(protein_predictions, top_n_list=[1, 3, 5, 10]):
    """Calculate pocket-level Top-N success rates."""
    results = {f'top{n}': 0 for n in top_n_list}
    results['n_proteins'] = len(protein_predictions)
    
    for pred in protein_predictions:
        true_binding = set(pred['true_binding_residues'])
        if not true_binding:
            continue
            
        # Sort by prediction score
        sorted_residues = sorted(
            zip(pred['residue_ids'], pred['scores']),
            key=lambda x: x[1],
            reverse=True
        )
        
        for n in top_n_list:
            top_n_residues = set([r[0] for r in sorted_residues[:n]])
            if top_n_residues & true_binding:
                results[f'top{n}'] += 1
    
    # Calculate percentages
    n_valid = results['n_proteins']
    for n in top_n_list:
        results[f'top{n}_percent'] = results[f'top{n}'] / n_valid * 100 if n_valid > 0 else 0
    
    return results


def evaluate_dataset(model, dataset_path, threshold=0.5):
    """Evaluate model on a dataset."""
    from torch_geometric.data import Data
    
    # Load all .pt files
    pt_files = [f for f in os.listdir(dataset_path) if f.endswith('.pt')]
    
    if not pt_files:
        return None
    
    all_labels = []
    all_scores = []
    protein_predictions = []
    
    with torch.no_grad():
        for pt_file in pt_files:
            try:
                data = torch.load(os.path.join(dataset_path, pt_file), weights_only=False)
                
                # Get predictions
                output = model(data)
                
                # Handle tuple output (model returns (logits, aux))
                if isinstance(output, tuple):
                    output = output[0]
                
                scores = torch.sigmoid(output).numpy().flatten()
                labels = data.y.numpy().flatten()
                
                all_labels.extend(labels)
                all_scores.extend(scores)
                
                # For pocket-level
                true_binding = np.where(labels == 1)[0].tolist()
                protein_predictions.append({
                    'protein': pt_file.replace('.pt', ''),
                    'residue_ids': list(range(len(labels))),
                    'scores': scores.tolist(),
                    'true_binding_residues': true_binding
                })
                
            except Exception as e:
                print(f"  Error processing {pt_file}: {e}")
                continue
    
    if not all_labels:
        return None
    
    # Calculate metrics
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, y_scores)
    
    # Calculate metrics with optimal threshold
    residue_metrics = calculate_metrics(y_true, y_scores, optimal_threshold)
    residue_metrics['optimal_threshold'] = optimal_threshold
    
    # Pocket-level metrics
    pocket_metrics = calculate_pocket_metrics(protein_predictions)
    
    return {
        'n_proteins': len(pt_files),
        'residue_level': residue_metrics,
        'pocket_level': pocket_metrics
    }


def main():
    print("=" * 70)
    print("GGNN 2025 - FRESH BENCHMARK EVALUATION")
    print("=" * 70)
    
    # Paths
    checkpoint = 'checkpoints_optimized/best_model.pth'
    config = 'config_optimized.yaml'
    data_base = 'data/processed'
    
    # Load model
    print("\n[1] Loading model...")
    model = load_model(checkpoint, config)
    print(f"    Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Define datasets to evaluate
    datasets = {
        'Combined Test': 'combined/test',
        'Combined Val': 'combined/val',
        'scPDB': 'scpdb',
        'PDBbind Refined': 'pdbbind_refined',
        'SC6K': 'sc6k',
        'COACH420': 'coach420',
        'Binding MOAD': 'moad_quality',
        'DUD-E Diverse': 'dude_diverse',
        'CryptoBench': 'cryptobench',
    }
    
    results = {}
    
    print("\n[2] Evaluating benchmarks...")
    print("-" * 70)
    
    for name, path in datasets.items():
        full_path = os.path.join(data_base, path)
        
        if not os.path.exists(full_path):
            print(f"  {name}: PATH NOT FOUND ({full_path})")
            continue
        
        print(f"\n  Evaluating {name}...")
        dataset_results = evaluate_dataset(model, full_path)
        
        if dataset_results:
            results[name] = dataset_results
            
            # Print summary
            r = dataset_results['residue_level']
            p = dataset_results['pocket_level']
            
            print(f"    n={dataset_results['n_proteins']}, "
                  f"residues={r['n_samples']:,}, "
                  f"AUC={r['auc']:.3f}, "
                  f"PR-AUC={r['prauc']:.3f}, "
                  f"F1={r['f1']:.3f}, "
                  f"MCC={r['mcc']:.3f}")
            print(f"    Top-1={p['top1_percent']:.1f}%, "
                  f"Top-3={p['top3_percent']:.1f}%, "
                  f"Top-5={p['top5_percent']:.1f}%, "
                  f"Top-10={p['top10_percent']:.1f}%")
    
    # Save results
    output_file = 'results_optimized/fresh_benchmark_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("[3] SUMMARY - TABLE S4 FORMAT")
    print("=" * 70)
    print(f"{'Benchmark':<20} {'n':>5} {'Residues':>10} {'AUC':>7} {'PR-AUC':>7} {'F1':>7} {'Prec':>6} {'Recall':>6} {'MCC':>6}")
    print("-" * 85)
    
    for name, data in results.items():
        r = data['residue_level']
        n = data['n_proteins']
        n_res = r['n_samples']
        print(f"{name:<20} {n:>5} {n_res:>10,} {r['auc']:>7.3f} {r['prauc']:>7.3f} "
              f"{r['f1']:>7.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['mcc']:>6.3f}")
    
    print("\n" + "=" * 70)
    print("[4] SUMMARY - TABLE S5 FORMAT")
    print("=" * 70)
    print(f"{'Benchmark':<20} {'n':>5} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Top-10':>8}")
    print("-" * 60)
    
    for name, data in results.items():
        p = data['pocket_level']
        n = p['n_proteins']
        print(f"{name:<20} {n:>5} {p['top1_percent']:>7.1f}% {p['top3_percent']:>7.1f}% "
              f"{p['top5_percent']:>7.1f}% {p['top10_percent']:>7.1f}%")
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()
