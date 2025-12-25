#!/usr/bin/env python3
"""
Check overlap between training data and benchmark datasets
This is CRITICAL for scientific validity
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from collections import defaultdict

def get_pdb_ids(processed_dir, split_name=None):
    """Extract PDB IDs from processed dataset"""
    if split_name:
        data_path = Path(processed_dir) / split_name
    else:
        data_path = Path(processed_dir)
    
    pt_files = list(data_path.glob("*.pt"))
    pdb_ids = set()
    
    for pt_file in pt_files:
        # PDB ID is typically first 4 characters of filename
        filename = pt_file.stem
        pdb_id = filename[:4].upper()
        pdb_ids.add(pdb_id)
    
    return pdb_ids, [f.stem for f in pt_files]

def main():
    print("\n" + "="*70)
    print("    TRAIN-BENCHMARK OVERLAP CHECK FOR SCIENTIFIC VALIDITY")
    print("="*70)
    
    # Get training PDB IDs
    print("\n📊 Loading training data PDB IDs...")
    train_pdb_ids, train_files = get_pdb_ids('data/processed/combined', 'train')
    val_pdb_ids, val_files = get_pdb_ids('data/processed/combined', 'val')
    test_pdb_ids, test_files = get_pdb_ids('data/processed/combined', 'test')
    
    all_train_pdb_ids = train_pdb_ids | val_pdb_ids | test_pdb_ids
    print(f"   Training set: {len(train_pdb_ids)} unique PDB IDs")
    print(f"   Validation set: {len(val_pdb_ids)} unique PDB IDs")
    print(f"   Test set: {len(test_pdb_ids)} unique PDB IDs")
    print(f"   Total unique: {len(all_train_pdb_ids)} PDB IDs")
    
    # Benchmark datasets to check
    benchmarks = [
        ('data/processed/coach420', 'COACH420'),
        ('data/processed/scpdb', 'scPDB'),
        ('data/processed/pdbbind_refined', 'PDBbind Refined'),
        ('data/processed/sc6k', 'SC6K'),
        ('data/processed/biolip_sample', 'BioLiP Sample'),
        ('data/processed/cryptobench', 'CryptoBench'),
        ('data/processed/dude_diverse', 'DUD-E Diverse'),
    ]
    
    print("\n" + "-"*70)
    print("🔍 CHECKING OVERLAP WITH BENCHMARK DATASETS")
    print("-"*70)
    
    results = {}
    
    for bench_path, bench_name in benchmarks:
        if not Path(bench_path).exists():
            print(f"\n⚠️  {bench_name}: NOT FOUND")
            continue
            
        bench_pdb_ids, bench_files = get_pdb_ids(bench_path)
        
        # Check overlap with TRAINING data only (not val/test of combined)
        overlap_train = train_pdb_ids & bench_pdb_ids
        overlap_all = all_train_pdb_ids & bench_pdb_ids
        
        overlap_pct = len(overlap_train) / len(bench_pdb_ids) * 100 if bench_pdb_ids else 0
        
        results[bench_name] = {
            'total': len(bench_pdb_ids),
            'overlap_with_train': len(overlap_train),
            'overlap_with_all': len(overlap_all),
            'overlap_pct': overlap_pct,
            'overlapping_ids': list(overlap_train)[:10]
        }
        
        status = "⚠️  WARNING" if overlap_pct > 10 else "✅ OK" if overlap_pct > 0 else "✅ CLEAN"
        
        print(f"\n{bench_name}:")
        print(f"   Total PDB IDs: {len(bench_pdb_ids)}")
        print(f"   Overlap with TRAIN: {len(overlap_train)} ({overlap_pct:.1f}%)")
        print(f"   Status: {status}")
        
        if overlap_train:
            print(f"   Overlapping IDs (first 10): {list(overlap_train)[:10]}")
    
    # Summary
    print("\n" + "="*70)
    print("                    SUMMARY")
    print("="*70)
    
    print("\n{:<25} {:>10} {:>15} {:>15}".format(
        "Benchmark", "Total", "Train Overlap", "Percentage"))
    print("-"*70)
    
    for name, data in results.items():
        status = "⚠️" if data['overlap_pct'] > 10 else "✅"
        print("{:<25} {:>10} {:>15} {:>14.1f}% {}".format(
            name, data['total'], data['overlap_with_train'], 
            data['overlap_pct'], status))
    
    # Scientific validity conclusion
    print("\n" + "="*70)
    print("           SCIENTIFIC VALIDITY ASSESSMENT")
    print("="*70)
    
    high_overlap = [name for name, data in results.items() if data['overlap_pct'] > 30]
    moderate_overlap = [name for name, data in results.items() if 10 < data['overlap_pct'] <= 30]
    low_overlap = [name for name, data in results.items() if data['overlap_pct'] <= 10]
    
    if high_overlap:
        print(f"\n⚠️  HIGH OVERLAP (>30%): {', '.join(high_overlap)}")
        print("   Results on these benchmarks may be INFLATED!")
        print("   These should be reported with caveat in paper.")
    
    if moderate_overlap:
        print(f"\n⚠️  MODERATE OVERLAP (10-30%): {', '.join(moderate_overlap)}")
        print("   Results should be interpreted with caution.")
    
    if low_overlap:
        print(f"\n✅ LOW/NO OVERLAP (≤10%): {', '.join(low_overlap)}")
        print("   These are VALID cross-dataset benchmarks.")
        print("   Results on these are scientifically reliable.")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
