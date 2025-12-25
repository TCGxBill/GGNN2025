#!/usr/bin/env python3
"""
Honest Pocket-Level Evaluation (Non-Overlapping Only)
Calculates Top-1, Top-3, Top-5 success rates on clean data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.models.gcn_geometric import GeometricGNN


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_pdb_ids(processed_dir, split_name=None):
    if split_name:
        data_path = Path(processed_dir) / split_name
    else:
        data_path = Path(processed_dir)
    
    pdb_ids = set()
    for pt_file in data_path.glob("*.pt"):
        pdb_id = pt_file.stem[:4].upper()
        pdb_ids.add(pdb_id)
    return pdb_ids


def load_non_overlapping_data(benchmark_path, train_pdb_ids):
    bench_path = Path(benchmark_path)
    pt_files = list(bench_path.glob("*.pt"))
    
    dataset = []
    skipped = 0
    
    for pt_file in pt_files:
        pdb_id = pt_file.stem[:4].upper()
        if pdb_id in train_pdb_ids:
            skipped += 1
            continue
        
        data = torch.load(pt_file, weights_only=False)
        data.pdb_id = pt_file.stem
        dataset.append(data)
    
    return dataset, len(dataset), skipped


def evaluate_top_k(model, graphs, top_k_values=[1, 3, 5, 10]):
    """Calculate Top-k success rates"""
    results = {k: [] for k in top_k_values}
    aucs = []
    
    model.eval()
    
    for graph in tqdm(graphs, desc="Evaluating"):
        with torch.no_grad():
            outputs = model(graph)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.sigmoid(outputs).numpy().flatten()
        
        labels = graph.y.numpy().flatten()
        
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        
        # AUC
        aucs.append(roc_auc_score(labels, probs))
        
        # Top-k success
        sorted_indices = np.argsort(probs)[::-1]
        true_indices = set(np.where(labels == 1)[0])
        
        for k in top_k_values:
            top_k_indices = set(sorted_indices[:k])
            success = len(top_k_indices & true_indices) > 0
            results[k].append(success)
    
    # Calculate rates
    summary = {
        'n_proteins': len(aucs),
        'mean_auc': float(np.mean(aucs)) if aucs else 0,
        'top_k': {}
    }
    
    for k in top_k_values:
        if results[k]:
            rate = sum(results[k]) / len(results[k])
            summary['top_k'][f'top-{k}'] = {
                'success': sum(results[k]),
                'rate': float(rate),
                'pct': f"{rate*100:.1f}%"
            }
    
    return summary


def main():
    print("\n" + "="*70)
    print("  HONEST POCKET-LEVEL EVALUATION (Non-Overlapping Only)")
    print("="*70)
    
    # Load model
    config = load_config('config_optimized.yaml')
    model = GeometricGNN(config['model'])
    checkpoint = torch.load('checkpoints_optimized/best_model.pth', 
                           map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get training IDs
    print("\n Loading training PDB IDs...")
    train_ids = set()
    for split in ['train', 'val', 'test']:
        train_ids |= get_pdb_ids('data/processed/combined', split)
    print(f"   Total: {len(train_ids)} training PDB IDs")
    
    # Benchmarks
    benchmarks = [
        ('data/processed/coach420', 'COACH420'),
        ('data/processed/scpdb', 'scPDB'),
        ('data/processed/pdbbind_refined', 'PDBbind'),
        ('data/processed/sc6k', 'SC6K'),
        ('data/processed/biolip_sample', 'BioLiP'),
        ('data/processed/cryptobench', 'CryptoBench'),
    ]
    
    all_results = {}
    
    print("\n" + "-"*70)
    
    for bench_path, bench_name in benchmarks:
        if not Path(bench_path).exists():
            continue
        
        dataset, included, skipped = load_non_overlapping_data(bench_path, train_ids)
        
        if len(dataset) == 0:
            print(f"\n{bench_name}: No clean proteins!")
            continue
        
        summary = evaluate_top_k(model, dataset)
        summary['included'] = included
        summary['skipped'] = skipped
        
        all_results[bench_name] = summary
        
        print(f"\n{bench_name} (Clean: {included}, Removed: {skipped}):")
        print(f"   AUC: {summary['mean_auc']:.4f}")
        for k, data in summary['top_k'].items():
            print(f"   {k}: {data['pct']}")
    
    # Summary table
    print("\n" + "="*70)
    print("         HONEST Top-k SUCCESS RATES SUMMARY")
    print("="*70)
    
    print("\n{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Benchmark", "n", "AUC", "Top-1", "Top-3", "Top-5"))
    print("-"*60)
    
    for name, data in all_results.items():
        top1 = data['top_k'].get('top-1', {}).get('pct', 'N/A')
        top3 = data['top_k'].get('top-3', {}).get('pct', 'N/A')
        top5 = data['top_k'].get('top-5', {}).get('pct', 'N/A')
        print("{:<12} {:>8} {:>8.3f} {:>8} {:>8} {:>8}".format(
            name, data['included'], data['mean_auc'], top1, top3, top5))
    
    # Literature comparison
    print("\n" + "="*70)
    print("         COMPARISON WITH LITERATURE")
    print("="*70)
    print("""
| Method       | Dataset  | Top-1  | Top-3  | Top-5  |
|--------------|----------|--------|--------|--------|
| GGNN (honest)| COACH420 | {coach1} | {coach3} | {coach5} |
| GGNN (honest)| scPDB    | {scpdb1} | {scpdb3} | {scpdb5} |
| P2Rank       | COACH420 | 63.0%  | 72.0%  | -      |
| PGpocket     | COACH420 | 42.0%  | 58.0%  | -      |
| Kalasanty    | scPDB    | 55.0%  | 68.0%  | -      |
| DeepSurf     | scPDB    | 58.0%  | 71.0%  | -      |
""".format(
        coach1=all_results.get('COACH420', {}).get('top_k', {}).get('top-1', {}).get('pct', 'N/A'),
        coach3=all_results.get('COACH420', {}).get('top_k', {}).get('top-3', {}).get('pct', 'N/A'),
        coach5=all_results.get('COACH420', {}).get('top_k', {}).get('top-5', {}).get('pct', 'N/A'),
        scpdb1=all_results.get('scPDB', {}).get('top_k', {}).get('top-1', {}).get('pct', 'N/A'),
        scpdb3=all_results.get('scPDB', {}).get('top_k', {}).get('top-3', {}).get('pct', 'N/A'),
        scpdb5=all_results.get('scPDB', {}).get('top_k', {}).get('top-5', {}).get('pct', 'N/A'),
    ))
    
    # Save
    with open('results_optimized/honest_pocket_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n> Results saved to results_optimized/honest_pocket_results.json")


if __name__ == '__main__':
    main()
