#!/usr/bin/env python3
"""Test GGNN on difficult proteins"""

import torch
import requests
import tempfile
from pathlib import Path

# Difficult test cases
DIFFICULT_PROTEINS = {
    # Cryptic binding sites
    "4N49": "Cap-specific mRNA methyltransferase (cryptic)",
    "1RTC": "Ricin (cryptic pocket)",
    
    # Allosteric sites  
    "3K5V": "PDK1 allosteric site",
    "2HYY": "Bcr-Abl allosteric",
    
    # Large flexible proteins
    "1HSG": "HIV-1 Protease",
    "4HJO": "Kinase with multiple pockets",
    
    # Small/difficult pockets
    "1M17": "EGFR kinase",
    "3ERT": "Estrogen receptor",
}

def download_pdb(pdb_id, output_dir):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        path = Path(output_dir) / f"{pdb_id.lower()}.pdb"
        path.write_text(response.text)
        return path
    return None

def main():
    import sys
    sys.path.insert(0, '.')
    
    from src.models.gcn_geometric import GeometricGNN
    from src.data.preprocessor import ProteinPreprocessor
    from src.data.graph_builder import ProteinGraphBuilder
    import yaml
    
    # Load model
    with open('config_optimized.yaml') as f:
        config = yaml.safe_load(f)
    
    model = GeometricGNN(config['model'])
    ckpt = torch.load('checkpoints_optimized/best_model.pth', 
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    preprocessor = ProteinPreprocessor(config)
    graph_builder = ProteinGraphBuilder(config.get('model', {}))
    
    print("="*60)
    print("DIFFICULT PROTEIN TEST")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for pdb_id, desc in DIFFICULT_PROTEINS.items():
            pdb_file = download_pdb(pdb_id, tmpdir)
            if not pdb_file:
                print(f"{pdb_id}: Download failed")
                continue
            
            data = preprocessor.process_pdb(str(pdb_file))
            if data is None:
                print(f"{pdb_id}: Processing failed")
                continue
            
            graph = graph_builder.build_graph(
                data['node_features'],
                data['coordinates'],
                data.get('labels')
            )
            
            with torch.no_grad():
                out, _ = model(graph)
                probs = torch.sigmoid(out).squeeze().numpy()
            
            n_predicted = (probs > 0.5).sum()
            n_high_conf = (probs > 0.85).sum()
            
            print(f"\n{pdb_id}: {desc}")
            print(f"  Residues: {len(probs)}")
            print(f"  Predicted binding (>0.5): {n_predicted}")
            print(f"  High confidence (>0.85): {n_high_conf}")
            print(f"  Max prob: {probs.max():.3f}")
            
            if data.get('labels') is not None:
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(data['labels'], probs)
                    print(f"  AUC: {auc:.3f}")
                except:
                    pass

if __name__ == "__main__":
    main()
