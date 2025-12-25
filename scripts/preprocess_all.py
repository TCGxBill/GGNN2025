#!/usr/bin/env python3
"""Simple preprocessing script"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import torch

from src.data.preprocessor import ProteinPreprocessor
from src.data.graph_builder import ProteinGraphBuilder


# Global initialization for workers
def process_single(pdb_file):
    # Re-import locally to ensure worker has access if needed, though top-level imports should work.
    # Initialize preprocessor/graph_builder here to avoid pickling complex objects
    config = {'distance_threshold': 6.0, 'k_neighbors': 15}
    preprocessor = ProteinPreprocessor(config)
    graph_builder = ProteinGraphBuilder(config)
    
    try:
        data = preprocessor.process_pdb(str(pdb_file))
        if data is None: return None
        
        if 'labels' not in data or data['labels'] is None:
            return None
            
        graph = graph_builder.build_graph(
            data['node_features'],
            data['coordinates'],
            data['labels']
        )
        
        return {
            'graph': graph,
            'id': Path(pdb_file).stem,
            'n_res': data['num_residues']
        }
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='demo', choices=['demo', 'pdbbind', 'coach420', 'custom', 'holo4k', 'joined', 'combined', 'scpdb', 'pdbbind_refined', 'sc6k', 'all_benchmarks', 'dna_test_129', 'dna_test_181', 'pdna_316', 'cryptobench', 'dude_diverse', 'difficult_cases', 'moad_quality'])
    parser.add_argument('--input', default='data/raw')
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    print(f"Preprocessing {args.dataset} dataset...")
    
    # Logic to find files
    input_base = Path(args.input)
    pdb_files = []
    
    if args.dataset == 'combined':
        for ds in ['holo4k', 'joined']:
            ds_path = input_base / ds
            if ds_path.exists():
                ds_files = list(ds_path.glob("**/*.pdb"))
                print(f"Found {len(ds_files)} files in {ds}")
                pdb_files.extend(ds_files)
            else:
                print(f"Warning: Dataset {ds} not found at {ds_path}")
    elif args.dataset == 'all_benchmarks':
        # Combine all benchmark datasets for comprehensive evaluation
        for ds in ['scpdb', 'pdbbind_refined', 'sc6k', 'coach420']:
            ds_path = input_base / ds
            if ds_path.exists():
                ds_files = list(ds_path.glob("**/*.pdb"))
                print(f"Found {len(ds_files)} files in {ds}")
                pdb_files.extend(ds_files)
            else:
                print(f"Warning: Dataset {ds} not found at {ds_path}")
    elif args.dataset == 'custom':
        # Custom uses --input directly
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            sys.exit(1)
        pdb_files = list(input_path.glob("**/*.pdb"))
    else:
        input_path = input_base / args.dataset
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            print(f"Run: python scripts/download_benchmark_datasets.py --dataset {args.dataset}")
            sys.exit(1)
        pdb_files = list(input_path.glob("**/*.pdb"))

    if args.max_samples:
        pdb_files = pdb_files[:args.max_samples]
    
    print(f"Total PDB files to process: {len(pdb_files)}")
    
    # Process (Parallel)
    import concurrent.futures
    print(f"Starting parallel processing with {os.cpu_count()} cores...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single, pdb_files), total=len(pdb_files), desc='Processing'))
        
    # Filter None results
    processed = [r for r in results if r is not None]
    
    print(f"Processed: {len(processed)}/{len(pdb_files)}")
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.split:
        n = len(processed)
        indices = np.random.permutation(n)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        
        splits = {
            'train': indices[:n_train].tolist(),
            'val': indices[n_train:n_train+n_val].tolist(),
            'test': indices[n_train+n_val:].tolist()
        }
        
        for split_name, split_indices in splits.items():
            split_path = output_path / split_name
            split_path.mkdir(exist_ok=True)
            for idx in tqdm(split_indices, desc=f'Saving {split_name} (Serial)'):
                data = processed[idx]
                torch.save(data['graph'], split_path / f"{data['id']}.pt")
        
        with open(output_path / 'split_indices.json', 'w') as f:
            json.dump(splits, f)
        
        print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    else:
        for data in tqdm(processed, desc='Saving'):
            torch.save(data['graph'], output_path / f"{data['id']}.pt")
    
    # Metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'n_proteins': len(processed),
            'n_residues': sum(d['n_res'] for d in processed),
            'split': args.split
        }, f)
    
    print(f"Done. Saved to {args.output}")
    print(f"Next: python scripts/train.py --config config_m2.yaml")


if __name__ == "__main__":
    np.random.seed(42)
    main()