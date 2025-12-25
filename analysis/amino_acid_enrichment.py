#!/usr/bin/env python3
"""
Amino Acid Enrichment Analysis
Analyze amino acid composition at predicted binding sites vs background

This script provides:
1. Amino acid frequency at binding sites
2. Enrichment/depletion analysis
3. Statistical significance (chi-square)
4. Visualization-ready output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import torch
from scipy import stats

from torch_geometric.loader import DataLoader


# Standard amino acid properties
AA_PROPERTIES = {
    'hydrophobic': ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'TYR', 'VAL'],
    'polar': ['ASN', 'CYS', 'GLN', 'SER', 'THR'],
    'charged_positive': ['ARG', 'HIS', 'LYS'],
    'charged_negative': ['ASP', 'GLU'],
    'special': ['GLY', 'PRO'],
    'aromatic': ['PHE', 'TRP', 'TYR', 'HIS']
}

AA_ONE_LETTER = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def load_dataset_graphs(processed_dir, split='test', max_samples=None):
    """Load preprocessed graph data"""
    processed_path = Path(processed_dir)
    
    if split:
        data_path = processed_path / split
    else:
        data_path = processed_path
    
    if not data_path.exists():
        return []
    
    pt_files = list(data_path.glob("*.pt"))
    if max_samples:
        pt_files = pt_files[:max_samples]
    
    graphs = []
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, weights_only=False)
            graphs.append(data)
        except:
            pass
    
    return graphs


def analyze_binding_site_composition(graphs):
    """Analyze amino acid composition at binding sites vs non-binding sites"""
    
    # We need to decode amino acid from node features
    # Node features: first 20 dims are one-hot amino acid encoding
    aa_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    
    binding_aa = []
    non_binding_aa = []
    all_aa = []
    
    for graph in graphs:
        x = graph.x.numpy()  # Node features
        y = graph.y.numpy()  # Labels
        
        for i, (features, label) in enumerate(zip(x, y)):
            # Decode amino acid from one-hot (first 20 features)
            aa_idx = np.argmax(features[:20])
            aa_name = aa_list[aa_idx]
            all_aa.append(aa_name)
            
            if label == 1:
                binding_aa.append(aa_name)
            else:
                non_binding_aa.append(aa_name)
    
    return binding_aa, non_binding_aa, all_aa


def calculate_enrichment(binding_aa, non_binding_aa):
    """Calculate enrichment of each amino acid at binding sites"""
    
    binding_counts = Counter(binding_aa)
    non_binding_counts = Counter(non_binding_aa)
    
    total_binding = sum(binding_counts.values())
    total_non_binding = sum(non_binding_counts.values())
    
    enrichment = {}
    p_values = {}
    
    # All amino acids
    all_aa = set(binding_counts.keys()) | set(non_binding_counts.keys())
    
    for aa in all_aa:
        bind_count = binding_counts.get(aa, 0)
        non_bind_count = non_binding_counts.get(aa, 0)
        
        # Frequency
        bind_freq = bind_count / total_binding if total_binding > 0 else 0
        non_bind_freq = non_bind_count / total_non_binding if total_non_binding > 0 else 0
        
        # Enrichment ratio
        if non_bind_freq > 0:
            enrichment[aa] = bind_freq / non_bind_freq
        else:
            enrichment[aa] = float('inf') if bind_freq > 0 else 1.0
        
        # Chi-square test for significance
        observed = [bind_count, non_bind_count]
        expected_bind = (bind_count + non_bind_count) * total_binding / (total_binding + total_non_binding)
        expected_non_bind = (bind_count + non_bind_count) * total_non_binding / (total_binding + total_non_binding)
        expected = [expected_bind, expected_non_bind]
        
        if expected_bind > 5 and expected_non_bind > 5:
            try:
                chi2, p = stats.chisquare(observed, expected)
                p_values[aa] = p
            except:
                p_values[aa] = 1.0
        else:
            p_values[aa] = 1.0
    
    return enrichment, p_values, binding_counts, non_binding_counts


def categorize_by_property(enrichment):
    """Categorize enrichment by amino acid properties"""
    category_enrichment = {}
    
    for category, aa_list in AA_PROPERTIES.items():
        values = [enrichment.get(aa, 1.0) for aa in aa_list if aa in enrichment]
        if values:
            category_enrichment[category] = np.mean(values)
    
    return category_enrichment


def main():
    parser = argparse.ArgumentParser(description="Amino acid enrichment analysis")
    parser.add_argument('--data_dir', type=str, default='data/processed/combined',
                       help='Path to processed data')
    parser.add_argument('--split', type=str, default='test',
                       help='Data split to analyze')
    parser.add_argument('--output', type=str, default='results_optimized/aa_enrichment.json',
                       help='Output file')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to analyze')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("       AMINO ACID ENRICHMENT ANALYSIS AT BINDING SITES")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    print(f"Loading data from {args.data_dir}/{args.split}...")
    graphs = load_dataset_graphs(args.data_dir, args.split, args.max_samples)
    print(f"> Loaded {len(graphs)} protein graphs")
    
    # Analyze composition
    print("\nAnalyzing amino acid composition...")
    binding_aa, non_binding_aa, all_aa = analyze_binding_site_composition(graphs)
    print(f"   Total residues: {len(all_aa)}")
    print(f"   Binding site residues: {len(binding_aa)} ({len(binding_aa)/len(all_aa)*100:.1f}%)")
    print(f"   Non-binding residues: {len(non_binding_aa)}")
    
    # Calculate enrichment
    print("\nCalculating enrichment...")
    enrichment, p_values, bind_counts, non_bind_counts = calculate_enrichment(binding_aa, non_binding_aa)
    
    # Sort by enrichment
    sorted_enrichment = sorted(enrichment.items(), key=lambda x: x[1], reverse=True)
    
    # Display results
    print("\n" + "="*70)
    print("                    ENRICHMENT RESULTS")
    print("="*70)
    print(f"\n{'Amino Acid':<12} {'1-Letter':<10} {'Binding':<12} {'Non-Binding':<12} {'Enrichment':<12} {'Significant':<12}")
    print("-"*70)
    
    significant_enriched = []
    significant_depleted = []
    
    for aa, enrich in sorted_enrichment:
        one_letter = AA_ONE_LETTER.get(aa, '?')
        p = p_values.get(aa, 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        
        if p < 0.05:
            if enrich > 1:
                significant_enriched.append((aa, enrich, p))
            else:
                significant_depleted.append((aa, enrich, p))
        
        print(f"{aa:<12} {one_letter:<10} {bind_counts.get(aa, 0):<12} {non_bind_counts.get(aa, 0):<12} {enrich:<12.2f} {sig:<12}")
    
    # Category analysis
    print("\n" + "="*70)
    print("              ENRICHMENT BY AMINO ACID CATEGORY")
    print("="*70)
    
    category_enrichment = categorize_by_property(enrichment)
    for category, avg_enrich in sorted(category_enrichment.items(), key=lambda x: x[1], reverse=True):
        status = "ENRICHED" if avg_enrich > 1.1 else "DEPLETED" if avg_enrich < 0.9 else "NEUTRAL"
        print(f"   {category.replace('_', ' ').title():<25} {avg_enrich:.2f}x  [{status}]")
    
    # Summary
    print("\n" + "="*70)
    print("                         SUMMARY")
    print("="*70)
    
    print("\n🔺 Significantly ENRICHED at binding sites:")
    for aa, enrich, p in significant_enriched[:5]:
        print(f"   {aa} ({AA_ONE_LETTER.get(aa, '?')}): {enrich:.2f}x (p={p:.2e})")
    
    print("\n🔻 Significantly DEPLETED at binding sites:")
    for aa, enrich, p in significant_depleted[:5]:
        print(f"   {aa} ({AA_ONE_LETTER.get(aa, '?')}): {enrich:.2f}x (p={p:.2e})")
    
    # Biological interpretation
    print("\n📋 Biological Interpretation:")
    
    # Check if aromatic/hydrophobic are enriched (common in binding sites)
    aromatic_enrich = np.mean([enrichment.get(aa, 1.0) for aa in AA_PROPERTIES['aromatic']])
    hydrophobic_enrich = np.mean([enrichment.get(aa, 1.0) for aa in AA_PROPERTIES['hydrophobic']])
    charged_pos_enrich = np.mean([enrichment.get(aa, 1.0) for aa in AA_PROPERTIES['charged_positive']])
    
    if aromatic_enrich > 1.1:
        print(f"   > Aromatic residues are ENRICHED ({aromatic_enrich:.2f}x) - expected for π-stacking with ligands")
    if hydrophobic_enrich > 1.0:
        print(f"   > Hydrophobic residues are ENRICHED ({hydrophobic_enrich:.2f}x) - expected for binding pocket lining")
    if charged_pos_enrich > 1.0:
        print(f"   > Positively charged residues are ENRICHED ({charged_pos_enrich:.2f}x) - common for ligand recognition")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'generated': datetime.now().isoformat(),
        'data_source': f"{args.data_dir}/{args.split}",
        'n_proteins': len(graphs),
        'n_total_residues': len(all_aa),
        'n_binding_residues': len(binding_aa),
        'binding_ratio': len(binding_aa) / len(all_aa) if len(all_aa) > 0 else 0,
        'enrichment': {aa: round(e, 3) for aa, e in enrichment.items()},
        'p_values': {aa: float(p) for aa, p in p_values.items()},
        'binding_counts': dict(bind_counts),
        'non_binding_counts': dict(non_bind_counts),
        'category_enrichment': {k: round(v, 3) for k, v in category_enrichment.items()},
        'significant_enriched': [(aa, round(e, 3), float(p)) for aa, e, p in significant_enriched],
        'significant_depleted': [(aa, round(e, 3), float(p)) for aa, e, p in significant_depleted]
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n> Results saved to {output_path}")


if __name__ == '__main__':
    main()
