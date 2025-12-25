#!/usr/bin/env python3
"""
Dataset Verification Script
Verifies Holo4K dataset integrity and checks for data leakage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime


def load_dataset(processed_dir, split_name):
    """Load preprocessed data from disk"""
    split_path = Path(processed_dir) / split_name
    pt_files = list(split_path.glob("*.pt"))
    dataset = []
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        data.filename = pt_file.stem
        dataset.append(data)
    return dataset


def analyze_dataset(dataset, split_name):
    """Analyze dataset statistics"""
    total_residues = 0
    total_binding = 0
    protein_sizes = []
    binding_ratios = []
    
    for data in dataset:
        n_residues = data.x.shape[0]
        n_binding = data.y.sum().item()
        
        total_residues += n_residues
        total_binding += n_binding
        protein_sizes.append(n_residues)
        binding_ratios.append(n_binding / n_residues if n_residues > 0 else 0)
    
    return {
        'split': split_name,
        'n_proteins': len(dataset),
        'total_residues': int(total_residues),
        'total_binding_sites': int(total_binding),
        'total_non_binding': int(total_residues - total_binding),
        'binding_ratio': float(total_binding / total_residues) if total_residues > 0 else 0,
        'avg_protein_size': float(np.mean(protein_sizes)),
        'min_protein_size': int(np.min(protein_sizes)),
        'max_protein_size': int(np.max(protein_sizes)),
        'std_protein_size': float(np.std(protein_sizes)),
        'avg_binding_ratio': float(np.mean(binding_ratios)),
        'std_binding_ratio': float(np.std(binding_ratios))
    }


def check_duplicate_proteins(train_data, val_data, test_data):
    """Check for protein name overlaps between splits"""
    train_names = set(d.filename for d in train_data)
    val_names = set(d.filename for d in val_data)
    test_names = set(d.filename for d in test_data)
    
    train_val_overlap = train_names & val_names
    train_test_overlap = train_names & test_names
    val_test_overlap = val_names & test_names
    
    return {
        'train_val_overlap': list(train_val_overlap),
        'train_test_overlap': list(train_test_overlap),
        'val_test_overlap': list(val_test_overlap),
        'has_leakage': len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    }


def verify_features(dataset):
    """Verify feature dimensions and validity"""
    issues = []
    
    for i, data in enumerate(dataset):
        # Check for NaN values
        if torch.isnan(data.x).any():
            issues.append(f"Protein {i}: NaN in node features")
        if torch.isnan(data.y).any():
            issues.append(f"Protein {i}: NaN in labels")
        
        # Check feature dimensions
        if data.x.shape[1] != 29:
            issues.append(f"Protein {i}: Expected 29 features, got {data.x.shape[1]}")
        
        # Check labels are binary
        unique_labels = torch.unique(data.y)
        if not all(l in [0, 1] for l in unique_labels.tolist()):
            issues.append(f"Protein {i}: Non-binary labels: {unique_labels.tolist()}")
    
    return issues


def main():
    print("\n" + "="*70)
    print("          DATASET VERIFICATION REPORT")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load datasets
    processed_dir = 'data/processed/combined'
    
    print(" Loading datasets...")
    train_data = load_dataset(processed_dir, 'train')
    val_data = load_dataset(processed_dir, 'val')
    test_data = load_dataset(processed_dir, 'test')
    
    print(f"  Train: {len(train_data)} proteins")
    print(f"  Val:   {len(val_data)} proteins")
    print(f"  Test:  {len(test_data)} proteins")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} proteins")
    
    # Analyze each split
    print("\n" + "-"*70)
    print("📈 DATASET STATISTICS")
    print("-"*70)
    
    results = {}
    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        stats = analyze_dataset(data, split_name)
        results[split_name] = stats
        
        print(f"\n{split_name.upper()} SET:")
        print(f"  Proteins:        {stats['n_proteins']}")
        print(f"  Total Residues:  {stats['total_residues']:,}")
        print(f"  Binding Sites:   {stats['total_binding_sites']:,} ({stats['binding_ratio']*100:.2f}%)")
        print(f"  Non-Binding:     {stats['total_non_binding']:,} ({(1-stats['binding_ratio'])*100:.2f}%)")
        print(f"  Class Ratio:     1:{int(1/stats['binding_ratio']) if stats['binding_ratio'] > 0 else 'inf'}")
        print(f"  Avg Protein Size: {stats['avg_protein_size']:.1f} ± {stats['std_protein_size']:.1f}")
        print(f"  Size Range:      [{stats['min_protein_size']}, {stats['max_protein_size']}]")
    
    # Check for data leakage
    print("\n" + "-"*70)
    print(" DATA LEAKAGE CHECK")
    print("-"*70)
    
    overlap = check_duplicate_proteins(train_data, val_data, test_data)
    results['leakage_check'] = overlap
    
    if overlap['has_leakage']:
        print("  WARNING: Data leakage detected!")
        if overlap['train_val_overlap']:
            print(f"  Train-Val overlap: {len(overlap['train_val_overlap'])} proteins")
        if overlap['train_test_overlap']:
            print(f"  Train-Test overlap: {len(overlap['train_test_overlap'])} proteins")
        if overlap['val_test_overlap']:
            print(f"  Val-Test overlap: {len(overlap['val_test_overlap'])} proteins")
    else:
        print(" No data leakage detected!")
        print("   No protein name overlaps between train/val/test splits")
    
    # Verify features
    print("\n" + "-"*70)
    print("🔧 FEATURE VERIFICATION")
    print("-"*70)
    
    all_data = train_data + val_data + test_data
    issues = verify_features(all_data)
    results['feature_issues'] = issues
    
    if issues:
        print(f"  Found {len(issues)} issues:")
        for issue in issues[:10]:
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues)-10} more")
    else:
        print(" All features verified!")
        print(f"   Feature dim: 29")
        print(f"   Labels: Binary (0/1)")
        print(f"   No NaN values")
    
    # Summary
    print("\n" + "="*70)
    print("          VERIFICATION SUMMARY")
    print("="*70)
    
    total_proteins = len(train_data) + len(val_data) + len(test_data)
    total_residues = results['train']['total_residues'] + results['val']['total_residues'] + results['test']['total_residues']
    total_binding = results['train']['total_binding_sites'] + results['val']['total_binding_sites'] + results['test']['total_binding_sites']
    
    print(f"""
Dataset: Combined (Holo4K + Joined)
Total Proteins: {total_proteins:,}
Total Residues: {total_residues:,}
Binding Sites: {total_binding:,} ({total_binding/total_residues*100:.2f}%)
Class Imbalance: 1:{int(total_residues/total_binding)}

Split Sizes:
  Train: {len(train_data)} ({len(train_data)/total_proteins*100:.1f}%)
  Val:   {len(val_data)} ({len(val_data)/total_proteins*100:.1f}%)
  Test:  {len(test_data)} ({len(test_data)/total_proteins*100:.1f}%)

Verification Status:
   Data Leakage: {'SAFE' if not overlap['has_leakage'] else 'DETECTED'}
   Features: {'VALID' if not issues else 'ISSUES FOUND'}
   Labels: Binary (0/1)
""")
    
    # Save results
    output_path = Path('results_optimized/dataset_report.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"> Report saved to {output_path}")
    
    # Save text report
    text_report = output_path.with_suffix('.txt')
    with open(text_report, 'w') as f:
        f.write(f"Dataset Verification Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Proteins: {total_proteins}\n")
        f.write(f"Total Residues: {total_residues}\n")
        f.write(f"Binding Sites: {total_binding} ({total_binding/total_residues*100:.2f}%)\n")
        f.write(f"Class Ratio: 1:{int(total_residues/total_binding)}\n\n")
        f.write(f"Train: {len(train_data)} proteins\n")
        f.write(f"Val: {len(val_data)} proteins\n")
        f.write(f"Test: {len(test_data)} proteins\n\n")
        f.write(f"Data Leakage: {'NONE' if not overlap['has_leakage'] else 'DETECTED'}\n")
        f.write(f"Feature Issues: {len(issues)}\n")
    
    print(f"> Text report saved to {text_report}")


if __name__ == '__main__':
    main()
