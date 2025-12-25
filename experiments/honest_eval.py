#!/usr/bin/env python3
"""
Evaluate model ONLY on non-overlapping proteins
This provides SCIENTIFICALLY VALID results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef

from src.models.gcn_geometric import GeometricGNN


def get_pdb_ids(processed_dir, split_name=None):
    """Extract PDB IDs from processed dataset"""
    if split_name:
        data_path = Path(processed_dir) / split_name
    else:
        data_path = Path(processed_dir)
    
    pt_files = list(data_path.glob("*.pt"))
    pdb_ids = {}
    
    for pt_file in pt_files:
        filename = pt_file.stem
        pdb_id = filename[:4].upper()
        pdb_ids[pdb_id] = pt_file
    
    return pdb_ids


def load_non_overlapping_data(benchmark_path, train_pdb_ids):
    """Load only proteins that are NOT in training set"""
    bench_path = Path(benchmark_path)
    pt_files = list(bench_path.glob("*.pt"))
    
    dataset = []
    skipped = 0
    included = 0
    
    for pt_file in pt_files:
        pdb_id = pt_file.stem[:4].upper()
        if pdb_id in train_pdb_ids:
            skipped += 1
            continue
        
        data = torch.load(pt_file, weights_only=False)
        dataset.append(data)
        included += 1
    
    return dataset, included, skipped


def evaluate_dataset(model, dataset, device='cpu'):
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            out = model(data)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.sigmoid(out).cpu().numpy()
            labels = data.y.cpu().numpy()
            
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    
    # Use optimal threshold
    threshold = 0.85
    preds_binary = (all_preds > threshold).astype(int)
    
    f1 = f1_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    mcc = matthews_corrcoef(all_labels, preds_binary)
    
    return {
        'auc': float(auc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'mcc': float(mcc),
        'n_residues': len(all_labels),
        'n_binding': int(sum(all_labels))
    }


def load_config(config_path):
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("\n" + "="*70)
    print("  HONEST EVALUATION: Non-Overlapping Proteins Only")
    print("="*70)
    
    # Load model
    print("\n📊 Loading model...")
    config = load_config('config_optimized.yaml')
    model = GeometricGNN(config['model'])
    checkpoint = torch.load('checkpoints_optimized/best_model.pth', 
                           map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get training PDB IDs
    print("\n📊 Loading training PDB IDs...")
    train_pdb_ids = set(get_pdb_ids('data/processed/combined', 'train').keys())
    val_pdb_ids = set(get_pdb_ids('data/processed/combined', 'val').keys())
    test_pdb_ids = set(get_pdb_ids('data/processed/combined', 'test').keys())
    all_train_ids = train_pdb_ids | val_pdb_ids | test_pdb_ids
    print(f"   Total training PDB IDs: {len(all_train_ids)}")
    
    # Benchmarks to evaluate
    benchmarks = [
        ('data/processed/coach420', 'COACH420'),
        ('data/processed/scpdb', 'scPDB'),
        ('data/processed/pdbbind_refined', 'PDBbind'),
        ('data/processed/sc6k', 'SC6K'),
        ('data/processed/biolip_sample', 'BioLiP'),
        ('data/processed/cryptobench', 'CryptoBench'),
        ('data/processed/dude_diverse', 'DUD-E'),
    ]
    
    print("\n" + "-"*70)
    print("🔬 EVALUATING ON NON-OVERLAPPING PROTEINS ONLY")
    print("-"*70)
    
    results = {}
    
    for bench_path, bench_name in benchmarks:
        if not Path(bench_path).exists():
            print(f"\n⚠️  {bench_name}: NOT FOUND")
            continue
        
        # Load only non-overlapping data
        dataset, included, skipped = load_non_overlapping_data(bench_path, all_train_ids)
        
        if len(dataset) == 0:
            print(f"\n{bench_name}: No non-overlapping proteins!")
            continue
        
        # Evaluate
        metrics = evaluate_dataset(model, dataset)
        metrics['included'] = included
        metrics['skipped'] = skipped
        metrics['total'] = included + skipped
        
        results[bench_name] = metrics
        
        print(f"\n{bench_name}:")
        print(f"   Original: {included + skipped} proteins")
        print(f"   Skipped (overlap): {skipped} ({skipped/(included+skipped)*100:.1f}%)")
        print(f"   Evaluated: {included} proteins (CLEAN)")
        print(f"   AUC: {metrics['auc']:.4f}")
        print(f"   F1:  {metrics['f1']:.4f}")
        print(f"   MCC: {metrics['mcc']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("        HONEST RESULTS SUMMARY (Non-Overlapping Only)")
    print("="*70)
    
    print("\n{:<15} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Benchmark", "Proteins", "AUC", "F1", "MCC", "Status"))
    print("-"*70)
    
    for name, data in results.items():
        print("{:<15} {:>10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
            name, data['included'], data['auc'], data['f1'], data['mcc'], "✅ CLEAN"))
    
    # Calculate mean
    aucs = [data['auc'] for data in results.values()]
    if aucs:
        print("-"*70)
        print(f"{'MEAN':<15} {'':<10} {np.mean(aucs):>10.4f}")
    
    print("\n" + "="*70)
    print("   These results are SCIENTIFICALLY VALID because they")
    print("   only include proteins NOT seen during training!")
    print("="*70)
    
    # Save results
    output_path = Path('results_optimized/honest_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
