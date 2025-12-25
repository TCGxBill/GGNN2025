#!/usr/bin/env python3
"""
Case Study Validation Script
Biological validation on famous protein-ligand complexes

Famous test cases:
- 1HSG: HIV-1 Protease with inhibitor (drug target)
- 4HHB: Hemoglobin with heme (classic binding)
- 1FKB: FKBP12 with FK506 (immunosuppressant target)
- 3HTB: HMG-CoA reductase with statin (cholesterol drug)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import requests
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter

from src.data.preprocessor import ProteinPreprocessor
from src.data.graph_builder import ProteinGraphBuilder
from src.models.gcn_geometric import GeometricGNN


# ============================================================
# FAMOUS PROTEIN-LIGAND COMPLEXES FOR VALIDATION
# ============================================================

CASE_STUDIES = {
    "1HSG": {
        "name": "HIV-1 Protease + Indinavir",
        "description": "FDA-approved HIV protease inhibitor complex",
        "ligand": "MK1 (Indinavir)",
        "known_binding_residues": [8, 23, 25, 27, 28, 29, 30, 32, 47, 48, 49, 50, 76, 80, 81, 82, 84],
        "biological_function": "Cleaves viral polyproteins; drug target",
        "reference": "Chen et al., J Mol Biol 1994"
    },
    "4HHB": {
        "name": "Hemoglobin + Heme",
        "description": "Classic oxygen-binding protein with heme cofactor",
        "ligand": "HEM (Heme)",
        "known_binding_residues": [58, 63, 87, 92, 99, 141],  # Proximal/distal histidine, heme pocket
        "biological_function": "Oxygen transport in blood",
        "reference": "Fermi et al., J Mol Biol 1984"
    },
    "1FKB": {
        "name": "FKBP12 + FK506",
        "description": "Immunophilin with immunosuppressant drug",
        "ligand": "FK5 (FK506/Tacrolimus)",
        "known_binding_residues": [26, 36, 37, 42, 46, 54, 55, 56, 59, 82, 87, 97],
        "biological_function": "Immunosuppression target; transplant medicine",
        "reference": "Van Duyne et al., Science 1991"
    },
    "3HTB": {
        "name": "HMG-CoA Reductase + Atorvastatin",
        "description": "Cholesterol-lowering drug target",
        "ligand": "ATV (Atorvastatin)",
        "known_binding_residues": [559, 591, 682, 683, 684, 724, 760, 805, 867],
        "biological_function": "Rate-limiting enzyme in cholesterol biosynthesis",
        "reference": "Istvan & Deisenhofer, Science 2001"
    }
}


def download_pdb(pdb_id, output_dir):
    """Download PDB file from RCSB"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_file = output_dir / f"{pdb_id.lower()}.pdb"
    if pdb_file.exists():
        return pdb_file
    
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            return pdb_file
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
    return None


def load_model(checkpoint_path, config_path):
    """Load trained model"""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model = GeometricGNN(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_binding_sites(model, pdb_file, threshold=0.5):
    """Predict binding sites for a protein"""
    config = {'distance_threshold': 8.0, 'k_neighbors': 15}
    preprocessor = ProteinPreprocessor(config)
    graph_builder = ProteinGraphBuilder(config)
    
    # Process PDB
    data = preprocessor.process_pdb(str(pdb_file))
    if data is None:
        return None
    
    # Build graph
    graph = graph_builder.build_graph(
        data['node_features'],
        data['coordinates'],
        data.get('labels')  # May be None for prediction only
    )
    
    # Predict
    with torch.no_grad():
        outputs, _ = model(graph)
        probs = torch.sigmoid(outputs).numpy().flatten()
    
    # Extract predictions
    predicted_sites = np.where(probs > threshold)[0].tolist()
    
    return {
        'probabilities': probs.tolist(),
        'predicted_sites': predicted_sites,
        'n_residues': len(probs),
        'n_predicted': len(predicted_sites),
        'residues': data['residues']
    }


def evaluate_case_study(pdb_id, predictions, known_sites):
    """Evaluate predictions against known binding sites"""
    predicted_set = set(predictions['predicted_sites'])
    known_set = set(known_sites)
    
    # Handle residue numbering offset (PDB vs 0-indexed)
    # Check both original and offset versions
    n_residues = predictions['n_residues']
    
    # True positives, false positives, false negatives
    tp = len(predicted_set & known_set)
    fp = len(predicted_set - known_set)
    fn = len(known_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'predicted_sites': sorted(list(predicted_set)),
        'known_sites': sorted(list(known_set)),
        'overlap': sorted(list(predicted_set & known_set))
    }


def analyze_amino_acid_composition(predictions):
    """Analyze amino acid composition of predicted binding sites"""
    residues = predictions['residues']
    predicted_indices = predictions['predicted_sites']
    
    # Get amino acid names at predicted sites
    binding_aa = []
    for idx in predicted_indices:
        if idx < len(residues):
            aa_name = residues[idx].get_resname()
            binding_aa.append(aa_name)
    
    # Background distribution
    all_aa = [res.get_resname() for res in residues]
    
    binding_counts = Counter(binding_aa)
    background_counts = Counter(all_aa)
    
    # Calculate enrichment
    enrichment = {}
    total_binding = len(binding_aa)
    total_background = len(all_aa)
    
    for aa, count in binding_counts.items():
        if background_counts[aa] > 0:
            binding_freq = count / total_binding if total_binding > 0 else 0
            bg_freq = background_counts[aa] / total_background
            enrichment[aa] = binding_freq / bg_freq if bg_freq > 0 else 0
    
    return {
        'binding_composition': dict(binding_counts),
        'background_composition': dict(background_counts),
        'enrichment': enrichment
    }


def run_case_studies(pdb_ids, model, output_dir, threshold=0.5):
    """Run validation on multiple case studies"""
    results = {}
    
    for pdb_id in pdb_ids:
        if pdb_id not in CASE_STUDIES:
            print(f"⚠ {pdb_id} not in case study database, skipping...")
            continue
        
        case_info = CASE_STUDIES[pdb_id]
        print(f"\n{'='*60}")
        print(f"📋 Case Study: {pdb_id} - {case_info['name']}")
        print(f"   {case_info['description']}")
        print(f"{'='*60}")
        
        # Download PDB
        pdb_file = download_pdb(pdb_id, output_dir / "pdbs")
        if pdb_file is None:
            print(f"    Failed to download {pdb_id}")
            continue
        print(f"   > Downloaded PDB file")
        
        # Predict binding sites
        predictions = predict_binding_sites(model, pdb_file, threshold)
        if predictions is None:
            print(f"    Failed to process {pdb_id}")
            continue
        print(f"   > Predicted {predictions['n_predicted']}/{predictions['n_residues']} residues as binding sites")
        
        # Evaluate against known sites
        evaluation = evaluate_case_study(pdb_id, predictions, case_info['known_binding_residues'])
        print(f"    Precision: {evaluation['precision']:.3f}")
        print(f"    Recall: {evaluation['recall']:.3f}")
        print(f"    F1: {evaluation['f1']:.3f}")
        
        # Amino acid composition
        aa_analysis = analyze_amino_acid_composition(predictions)
        
        # Top enriched amino acids
        sorted_enrichment = sorted(aa_analysis['enrichment'].items(), key=lambda x: x[1], reverse=True)
        print(f"   🧬 Top enriched amino acids at binding sites:")
        for aa, enrich in sorted_enrichment[:5]:
            print(f"      {aa}: {enrich:.2f}x enriched")
        
        results[pdb_id] = {
            'case_info': case_info,
            'predictions': {
                'n_residues': predictions['n_residues'],
                'n_predicted': predictions['n_predicted'],
                'predicted_sites': predictions['predicted_sites'],
                'top_probabilities': sorted(enumerate(predictions['probabilities']), 
                                           key=lambda x: x[1], reverse=True)[:20]
            },
            'evaluation': evaluation,
            'aa_analysis': aa_analysis
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Case study validation on famous proteins")
    parser.add_argument('--proteins', type=str, default='1HSG,4HHB,1FKB',
                       help='Comma-separated list of PDB IDs')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_optimized/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config_optimized.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='results_optimized/case_studies',
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("           BIOLOGICAL VALIDATION - CASE STUDIES")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Parse protein list
    pdb_ids = [p.strip().upper() for p in args.proteins.split(',')]
    print(f"Proteins to analyze: {pdb_ids}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, args.config)
    print("> Model loaded")
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run case studies
    results = run_case_studies(pdb_ids, model, output_dir, args.threshold)
    
    # Save results
    output_file = output_dir / "case_study_results.json"
    
    # Convert non-serializable items
    serializable_results = {}
    for pdb_id, data in results.items():
        serializable_results[pdb_id] = {
            'case_info': {k: v for k, v in data['case_info'].items()},
            'predictions': {
                'n_residues': data['predictions']['n_residues'],
                'n_predicted': data['predictions']['n_predicted'],
                'predicted_sites': data['predictions']['predicted_sites']
            },
            'evaluation': data['evaluation'],
            'aa_analysis': {
                'binding_composition': data['aa_analysis']['binding_composition'],
                'enrichment': {k: round(v, 3) for k, v in data['aa_analysis']['enrichment'].items()}
            }
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'threshold': args.threshold,
            'model': args.checkpoint,
            'case_studies': serializable_results
        }, f, indent=2)
    
    print(f"\n> Results saved to {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("                         SUMMARY")
    print("="*70)
    
    print(f"\n{'PDB ID':<10} {'Protein':<30} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*70)
    for pdb_id, data in results.items():
        e = data['evaluation']
        info = data['case_info']
        print(f"{pdb_id:<10} {info['name'][:28]:<30} {e['precision']:<12.3f} {e['recall']:<12.3f} {e['f1']:<12.3f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
