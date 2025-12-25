import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessor import ProteinPreprocessor
from src.data.graph_builder import ProteinGraphBuilder
from src.models.gcn_geometric import GeometricGNN

def create_dummy_pdb(filename):
    """Create a minimal PDB file with protein and ligand"""
    with open(filename, 'w') as f:
        # Protein (Alanine dipeptide-ish)
        f.write("ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00  0.00           C\n")
        f.write("ATOM      3  C   ALA A   1      12.000  11.458  10.000  1.00  0.00           C\n")
        f.write("ATOM      4  O   ALA A   1      11.500  12.500  10.000  1.00  0.00           O\n")
        f.write("ATOM      5  N   ALA A   2      13.500  11.500  10.000  1.00  0.00           N\n")
        f.write("ATOM      6  CA  ALA A   2      14.000  13.000  10.000  1.00  0.00           C\n")
        f.write("ATOM      7  N   ALA A   3      15.500  13.500  10.000  1.00  0.00           N\n")
        f.write("ATOM      8  CA  ALA A   3      16.000  15.000  10.000  1.00  0.00           C\n")
        
        # Ligand (LIG close to residue 2)
        f.write("HETATM    9  C1  LIG A 100      14.000  13.000  11.000  1.00  0.00           C\n")
        f.write("HETATM   10  C2  LIG A 100      14.000  13.000  12.000  1.00  0.00           C\n")
        f.write("HETATM   11  C3  LIG A 100      14.000  13.000  13.000  1.00  0.00           C\n")
        f.write("END\n")

def test_pipeline():
    print("Running pipeline test...")
    
    # 1. Setup
    config = {
        'distance_threshold': 6.0,
        'k_neighbors': 5,
        'input_dim': 29,
        'hidden_dims': [64, 32],
        'output_dim': 1,
        'dropout': 0.1,
        'edge_dim': 20
    }
    
    pdb_file = 'test_protein.pdb'
    create_dummy_pdb(pdb_file)
    
    try:
        # 2. Preprocessing
        print("Testing Preprocessor...")
        preprocessor = ProteinPreprocessor(config)
        data = preprocessor.process_pdb(pdb_file)
        
        assert data is not None, "Preprocessing returned None"
        print(f"  Num residues: {data['num_residues']}")
        print(f"  Node features shape: {data['node_features'].shape}")
        
        # Check automatic labels
        assert data['labels'] is not None, "Labels were not generated automatically"
        print(f"  Labels shape: {data['labels'].shape}")
        print(f"  Positive labels: {np.sum(data['labels'])}")
        assert np.sum(data['labels']) > 0, "No binding sites detected (should detect near LIG)"

        # 3. Graph Building
        print("Testing GraphBuilder...")
        builder = ProteinGraphBuilder(config)
        graph = builder.build_graph(
            data['node_features'],
            data['coordinates'],
            data['labels']
        )
        
        print(f"  Graph nodes: {graph.x.shape}")
        print(f"  Graph edges: {graph.edge_index.shape}")
        
        # Assert input dimension matches config expectation
        assert graph.x.shape[1] == 29, f"Expected 29 input features, got {graph.x.shape[1]}"
        
        # 4. Model Forward Pass
        print("Testing Model initialization and forward pass...")
        model = GeometricGNN(config)
        output, attn = model(graph)
        
        print(f"  Output shape: {output.shape}")
        assert output.shape == (graph.x.shape[0], 1), "Output shape mismatch"
        
        print("SUCCESS: Full pipeline verified!")
        
    finally:
        if os.path.exists(pdb_file):
            os.remove(pdb_file)

if __name__ == "__main__":
    test_pipeline()
