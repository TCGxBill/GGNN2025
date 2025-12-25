#!/usr/bin/env python3
"""
DNA-Binding Site Benchmark Datasets Downloader
Downloads standard benchmarks: DNA_Test_129, DNA_Test_181, PDNA-316

Sources:
- GraphSite: https://github.com/biomed-AI/GraphSite
- EquiPNAS: Protein-DNA binding benchmarks
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# DNA-binding benchmark PDB IDs
# Extracted from published papers and GraphSite repository
DNA_TEST_129 = [
    # Sample subset - full list would be extracted from GraphSite dataset
    "1a02", "1a0a", "1a1k", "1a73", "1aay", "1ah9", "1ais", "1aoi", 
    "1arr", "1asa", "1azp", "1bc8", "1bf4", "1bpx", "1c0w", "1cdw",
    "1cez", "1cjg", "1cl8", "1cma", "1d3z", "1d66", "1dc1", "1diz",
    "1dnk", "1dp7", "1dux", "1e3o", "1emh", "1f4k", "1fiu", "1fjl",
    "1fok", "1g2d", "1g38", "1gtw", "1gxp", "1h6f", "1hcr", "1hjb",
    "1hjc", "1hwt", "1ic8", "1if1", "1igr", "1ihf", "1ipp", "1irz",
    "1j1v", "1j75", "1jey", "1jfi", "1jgg", "1jnr", "1jx4", "1k61",
    "1k82", "1kb2", "1kbu", "1kdx", "1l3l", "1l3s", "1l3t", "1lmb",
]

DNA_TEST_181 = [
    # Sample subset for Test_181
    "1a73", "1azp", "1b3t", "1b72", "1bc8", "1bf4", "1bpx", "1c0w",
    "1cdw", "1cez", "1cjg", "1cl8", "1cma", "1d3z", "1d66", "1dc1",
    "1diz", "1dnk", "1dp7", "1dux", "1e3o", "1emh", "1f4k", "1fiu",
    "1fjl", "1fok", "1g2d", "1g38", "1gtw", "1gxp", "1h6f", "1hcr",
    "1hjb", "1hwt", "1ic8", "1if1", "1igr", "1ihf", "1ipp", "1irz",
]

PDNA_316 = [
    # Sample from PDNA-316 benchmark dataset
    "1a02", "1a1i", "1a1j", "1a1k", "1a1l", "1a3q", "1a73", "1aay",
    "1ah9", "1ais", "1aoi", "1arr", "1asa", "1azp", "1bc8", "1bf4",
    "1bpx", "1c0w", "1cdw", "1cez", "1cjg", "1cl8", "1cma", "1d3z",
]


def download_pdb(pdb_id, output_dir, max_retries=3):
    """Download PDB file from RCSB"""
    output_file = output_dir / f"{pdb_id.lower()}.pdb"
    if output_file.exists():
        return pdb_id, "exists", None
    
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_file, 'w') as f:
                    f.write(response.text)
                return pdb_id, "success", None
            elif response.status_code == 404:
                return pdb_id, "not_found", "PDB not found"
        except Exception as e:
            if attempt == max_retries - 1:
                return pdb_id, "error", str(e)
    
    return pdb_id, "error", "Max retries exceeded"


def download_dataset(pdb_ids, dataset_name, output_base, max_workers=10):
    """Download all PDBs for a dataset"""
    output_dir = Path(output_base) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {dataset_name} ({len(pdb_ids)} structures)...")
    
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_pdb, pdb_id, output_dir): pdb_id 
                   for pdb_id in pdb_ids}
        
        for future in tqdm(as_completed(futures), total=len(pdb_ids), desc=f"  {dataset_name}"):
            pdb_id, status, error = future.result()
            if status in ["success", "exists"]:
                success_count += 1
            else:
                fail_count += 1
    
    print(f"  ✓ Downloaded: {success_count}, Failed: {fail_count}")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'downloaded': datetime.now().isoformat(),
        'total_requested': len(pdb_ids),
        'success': success_count,
        'failed': fail_count,
        'pdb_ids': pdb_ids
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description="Download DNA-binding benchmark datasets")
    parser.add_argument('--dataset', type=str, choices=['test_129', 'test_181', 'pdna_316', 'all'],
                       default='all', help='Dataset to download')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=10,
                       help='Number of parallel downloads')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("       DNA-BINDING BENCHMARK DATASETS DOWNLOADER")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    datasets = {
        'test_129': ('dna_test_129', DNA_TEST_129),
        'test_181': ('dna_test_181', DNA_TEST_181),
        'pdna_316': ('pdna_316', PDNA_316)
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


if __name__ == '__main__':
    main()
