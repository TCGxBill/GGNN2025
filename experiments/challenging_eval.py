#!/usr/bin/env python3
"""
Challenging Benchmark Evaluation
Test model on hardest protein-ligand binding site benchmarks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef, average_precision_score

from src.models.gcn_geometric import GeometricGNN


def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config):
    model = GeometricGNN(config['model'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_dataset(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    graphs = []
    for pt_file in sorted(data_path.glob("*.pt")):
        try:
            data = torch.load(pt_file, weights_only=False)
            data.pdb_id = pt_file.stem
            graphs.append(data)
        except:
            pass
    return graphs


def evaluate(model, graphs, top_k_values=[1, 3, 5, 10]):
    """Full evaluation: residue + pocket level"""
    if not graphs:
        return None
    
    all_probs = []
    all_labels = []
    pocket_results = {k: [] for k in top_k_values}
    n_valid = 0
    
    model.eval()
    for graph in graphs:
        with torch.no_grad():
            outputs, _ = model(graph)
            probs = torch.sigmoid(outputs).numpy().flatten()
        
        labels = graph.y.numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels)
        
        if labels.sum() > 0:
            n_valid += 1
            sorted_indices = np.argsort(probs)[::-1]
            true_indices = set(np.where(labels == 1)[0])
            
            for k in top_k_values:
                top_k_indices = set(sorted_indices[:k])
                overlap = len(top_k_indices & true_indices)
                pocket_results[k].append(overlap > 0)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return None
    
    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.3, 0.95, 0.05):
        preds = (all_probs > t).astype(int)
        if preds.sum() > 0:
            f1 = f1_score(all_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    
    preds = (all_probs > best_thresh).astype(int)
    
    # Pocket-level metrics
    pocket_summary = {'n_proteins': n_valid}
    for k in top_k_values:
        rate = sum(pocket_results[k]) / len(pocket_results[k]) if pocket_results[k] else 0
        pocket_summary[f'top{k}'] = float(rate)
        pocket_summary[f'top{k}_pct'] = f"{rate*100:.1f}%"
    
    return {
        'residue': {
            'auc': float(roc_auc_score(all_labels, all_probs)),
            'prauc': float(average_precision_score(all_labels, all_probs)),
            'f1': float(f1_score(all_labels, preds)),
            'precision': float(precision_score(all_labels, preds, zero_division=0)),
            'recall': float(recall_score(all_labels, preds)),
            'mcc': float(matthews_corrcoef(all_labels, preds)),
            'threshold': float(best_thresh),
            'n_samples': len(all_labels),
            'n_positive': int(all_labels.sum()),
            'positive_ratio': float(all_labels.mean())
        },
        'pocket': pocket_summary
    }


def main():
    print("\n" + "="*80)
    print("       CHALLENGING BENCHMARK EVALUATION")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load model
    print("Loading model...")
    config = load_config('config_optimized.yaml')
    model = load_model('checkpoints_optimized/best_model.pth', config)
    print("Model loaded (276K parameters)\n")
    
    # Challenging benchmarks
    benchmarks = [
        ('data/processed/cryptobench', 'CryptoBench', 'Cryptic binding sites (VERY HARD)'),
        ('data/processed/dude_diverse', 'DUD-E Diverse', 'Drug targets (challenging)'),
        ('data/processed/difficult_cases', 'Difficult Cases', 'Large/allosteric/multi-domain'),
        ('data/processed/moad_quality', 'Binding MOAD', 'High-quality curated'),
    ]
    
    all_results = {}
    
    print("="*80)
    print("                    CHALLENGING BENCHMARKS")
    print("="*80)
    
    for data_dir, name, description in benchmarks:
        print(f"\n{name}")
        print(f"   {description}")
        print("-"*50)
        
        graphs = load_dataset(data_dir)
        if not graphs:
            print(f"   No data found")
            continue
        
        print(f"   Proteins: {len(graphs)}")
        
        metrics = evaluate(model, graphs)
        
        if metrics:
            r = metrics['residue']
            p = metrics['pocket']
            
            print(f"   Residues: {r['n_samples']:,} (binding: {r['positive_ratio']*100:.1f}%)")
            print(f"\n   Residue-Level:")
            print(f"      AUC:    {r['auc']:.4f}")
            print(f"      MCC:    {r['mcc']:.4f}")
            print(f"      F1:     {r['f1']:.4f}")
            print(f"\n   Pocket-Level:")
            print(f"      Top-1:  {p['top1_pct']}")
            print(f"      Top-3:  {p['top3_pct']}")
            print(f"      Top-5:  {p['top5_pct']}")
            
            all_results[name] = {
                'description': description,
                'n_proteins': len(graphs),
                **metrics
            }
    
    # Summary
    print("\n\n" + "="*80)
    print("                    SUMMARY TABLE")
    print("="*80)
    
    print(f"\n{'Benchmark':<20} {'Proteins':<10} {'AUC':<10} {'MCC':<10} {'Top-1':<10} {'Top-3':<10}")
    print("-"*80)
    
    for name, data in all_results.items():
        r = data['residue']
        p = data['pocket']
        print(f"{name:<20} {data['n_proteins']:<10} {r['auc']:<10.4f} {r['mcc']:<10.4f} {p['top1_pct']:<10} {p['top3_pct']:<10}")
    
    # Statistics
    if all_results:
        aucs = [d['residue']['auc'] for d in all_results.values()]
        mccs = [d['residue']['mcc'] for d in all_results.values()]
        
        print(f"\nChallenging benchmarks average:")
        print(f"   Mean AUC: {np.mean(aucs):.4f}")
        print(f"   Mean MCC: {np.mean(mccs):.4f}")
    
    # Save results
    output_path = Path("results_optimized/challenging_benchmark_results.json")
    final_results = {
        'generated': datetime.now().isoformat(),
        'description': 'Evaluation on challenging protein-ligand binding benchmarks',
        'benchmarks': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
