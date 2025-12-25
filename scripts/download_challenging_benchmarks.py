#!/usr/bin/env python3
"""
Download Additional Challenging Benchmark Datasets

New challenging benchmarks:
1. LIGYSIS (2024) - 30,000 proteins, most comprehensive
2. CryptoBench (2024) - Cryptic binding sites (very challenging)
3. DUD-E - Drug targets with decoys
4. BioLiP - Biologically relevant ligand-protein interactions
5. Binding MOAD - High-quality binding data
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ============================================================
# CHALLENGING BENCHMARK DATASETS
# ============================================================

# PDB IDs from various challenging benchmarks
# These are curated sets for testing generalization

# From CryptoBench - cryptic binding sites (very challenging)
CRYPTOBENCH_IDS = [
    # These proteins have significant conformational changes
    "1ypc", "3eml", "3gqz", "4djp", "2v4j", "4g34", "4j72", "2wvt",
    "3l3n", "4mgd", "2rku", "3d4z", "1w8l", "4aou", "3t08", "4otw",
    "2hue", "3p3h", "2p33", "3qgy", "4ypm", "3lpj", "4llb", "4m7p",
]

# From DUD-E diverse subset - drug targets
DUDE_DIVERSE = [
    # Kinases
    "1m17", "1t46", "2src", "3eqg", "3pp0",
    # GPCRs related
    "3pbl", "3uon", "4djh", "4dkl", "4grv",
    # Proteases
    "1hiv", "1hvr", "1msn", "1ohr", "2qhq",
    # Nuclear receptors
    "1nq7", "1yow", "2aa2", "2gv5", "2q70",
    # Enzymes
    "1b38", "1e7a", "1gpk", "1m2z", "1xd0",
]

# From recent literature (2023-2024) - difficult cases
DIFFICULT_CASES = [
    # Large proteins
    "6m17", "6vsb", "6xlu", "7bv2", "7jm2",
    # Allosteric sites
    "4ea2", "5f1a", "6dbb", "6dir", "6n4b",
    # Multi-domain
    "1ghq", "3lpo", "4dkq", "5c28", "6hax",
    # Membrane-associated
    "4k5y", "5c1m", "5d3d", "6d26", "6n7r",
]

# Binding MOAD high-quality subset
MOAD_QUALITY = [
    "1a28", "1a4w", "1abe", "1abf", "1acl",
    "1acy", "1adb", "1add", "1adl", "1aec",
    "1af2", "1aha", "1ai5", "1aif", "1aj7",
]


def download_pdb(pdb_id, output_dir, max_retries=3):
    """Download PDB file from RCSB"""
    output_file = output_dir / f"{pdb_id.lower()}.pdb"
    if output_file.exists():
        return pdb_id, "exists", output_file.stat().st_size
    
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_file, 'w') as f:
                    f.write(response.text)
                return pdb_id, "success", len(response.text)
            elif response.status_code == 404:
                return pdb_id, "not_found", 0
        except Exception as e:
            if attempt == max_retries - 1:
                return pdb_id, "error", str(e)
    
    return pdb_id, "error", "Max retries exceeded"


def download_dataset(pdb_ids, dataset_name, output_base, max_workers=10):
    """Download all PDBs for a dataset"""
    output_dir = Path(output_base) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {dataset_name} ({len(pdb_ids)} structures)...")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_size = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_pdb, pdb_id, output_dir): pdb_id 
                   for pdb_id in pdb_ids}
        
        for future in tqdm(as_completed(futures), total=len(pdb_ids), desc=f"  {dataset_name}"):
            pdb_id, status, size = future.result()
            if status == "success":
                success_count += 1
                total_size += size if isinstance(size, int) else 0
            elif status == "exists":
                skip_count += 1
                total_size += size if isinstance(size, int) else 0
            else:
                fail_count += 1
    
    print(f"  Downloaded: {success_count}, Existing: {skip_count}, Failed: {fail_count}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'downloaded': datetime.now().isoformat(),
        'total_requested': len(pdb_ids),
        'success': success_count + skip_count,
        'failed': fail_count,
        'pdb_ids': pdb_ids,
        'challenge_level': 'high'
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return success_count + skip_count, fail_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download challenging benchmark datasets")
    parser.add_argument('--dataset', type=str, 
                       choices=['cryptobench', 'dude', 'difficult', 'moad', 'all'],
                       default='all', help='Dataset to download')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel downloads')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("       CHALLENGING BENCHMARK DATASETS DOWNLOADER")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Available challenging benchmarks:")
    print("   - CryptoBench (2024): Cryptic binding sites - VERY challenging")
    print("   - DUD-E Diverse: Drug targets with diverse chemotypes")
    print("   - Difficult Cases: Large/allosteric/multi-domain proteins")
    print("   - MOAD Quality: High-quality curated binding data")
    
    datasets = {
        'cryptobench': ('cryptobench', CRYPTOBENCH_IDS),
        'dude': ('dude_diverse', DUDE_DIVERSE),
        'difficult': ('difficult_cases', DIFFICULT_CASES),
        'moad': ('moad_quality', MOAD_QUALITY)
    }
    
    if args.dataset == 'all':
        to_download = list(datasets.keys())
    else:
        to_download = [args.dataset]
    
    total_success = 0
    total_fail = 0
    
    for ds_key in to_download:
        ds_name, pdb_ids = datasets[ds_key]
        success, fail = download_dataset(pdb_ids, ds_name, args.output, args.workers)
        total_success += success
        total_fail += fail
    
    print(f"\n" + "="*70)
    print(f"Total: {total_success} downloaded, {total_fail} failed")
    print("="*70)
    
    print("\nNext steps:")
    print("  1. python scripts/preprocess_all.py --dataset cryptobench --output data/processed/cryptobench")
    print("  2. python experiments/comprehensive_eval.py (update script to include new datasets)")


if __name__ == '__main__':
    main()
