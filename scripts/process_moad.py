#!/usr/bin/env python3
"""
Process MOAD PDB files with proper ligand binding labels.

Uses ligand coordinates from each PDB to label binding site residues.
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocessor import ProteinPreprocessor
from data.graph_builder import ProteinGraphBuilder


def process_moad_pdbs(input_dir, output_dir, config, csv_file=None):
    """Process MOAD PDB files using ligand info."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ligand mapping from CSV if available
    ligand_mapping = {}
    if csv_file and os.path.exists(csv_file):
        import pandas as pd
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            pdb_id = row['pdbFile'].lower()
            if pdb_id not in ligand_mapping:
                ligand_mapping[pdb_id] = row['ligand_id']
    
    # Initialize preprocessor
    preprocessor = ProteinPreprocessor(config)
    
    # Get PDB files
    pdb_files = list(Path(input_dir).glob("*.pdb"))
    
    print(f"\nProcessing {len(pdb_files)} MOAD PDB files...")
    
    successful = 0
    failed = []
    
    for pdb_file in tqdm(pdb_files, desc="MOAD"):
        pdb_id = pdb_file.stem.lower()
        output_file = os.path.join(output_dir, f"{pdb_id.upper()}.pt")
        
        if os.path.exists(output_file):
            successful += 1
            continue
        
        try:
            # Process PDB - preprocessor auto-extracts ligand
            data = preprocessor.process_pdb(str(pdb_file))
            
            if data is None:
                failed.append(pdb_id)
                continue
            
            # Check that we have binding sites labeled
            if 'labels' not in data or data['labels'] is None:
                failed.append(pdb_id)
                continue
            
            labels = data['labels']
            if isinstance(labels, np.ndarray):
                n_binding = labels.sum()
            else:
                n_binding = sum(labels)
            
            if n_binding == 0:
                # No binding sites found - skip
                failed.append(pdb_id)
                continue
            
            # Build graph - build_graph expects (node_features, coordinates, labels)
            graph_builder = ProteinGraphBuilder(config.get('model', {}))
            node_features = data.get('node_features')
            coordinates = data.get('coordinates')
            labels_array = data.get('labels')
            
            if node_features is None or coordinates is None:
                failed.append(pdb_id)
                continue
            
            graph_data = graph_builder.build_graph(node_features, coordinates, labels_array)
            
            if graph_data is not None:
                torch.save(graph_data, output_file)
                successful += 1
            else:
                failed.append(pdb_id)
                
        except Exception as e:
            failed.append(pdb_id)
    
    print(f"MOAD: {successful}/{len(pdb_files)} processed successfully")
    if failed:
        print(f"  Failed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    return successful


def main():
    print("=" * 60)
    print("MOAD BENCHMARK PROCESSING")
    print("=" * 60)
    
    # Load config
    config_file = "config_optimized.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'model': {'distance_threshold': 8.0}, 'distance_threshold': 6.0}
    
    # Process MOAD PDBs
    input_dir = "external_data/moad_new"
    output_dir = "data/processed/moad_expanded"
    csv_file = "external_data/binding_moad_full.csv"
    
    if os.path.exists(input_dir):
        count = process_moad_pdbs(input_dir, output_dir, config, csv_file)
        print(f"\nTotal processed: {count}")
    else:
        print(f"Input directory not found: {input_dir}")
    
    # Show final count
    pt_files = list(Path(output_dir).glob("*.pt")) if Path(output_dir).exists() else []
    print(f"\nFinal MOAD expanded: {len(pt_files)} proteins")
    
    # Also show current moad_quality
    moad_quality = list(Path("data/processed/moad_quality").glob("*.pt"))
    print(f"Original moad_quality: {len(moad_quality)} proteins")
    print(f"Total MOAD: {len(pt_files) + len(moad_quality)} proteins")


if __name__ == "__main__":
    main()
