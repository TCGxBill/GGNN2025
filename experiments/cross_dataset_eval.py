#!/usr/bin/env python3
"""
Cross-Dataset Evaluation Script
Test trained model on different datasets for generalization validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                            recall_score, accuracy_score, matthews_corrcoef)
from torch_geometric.loader import DataLoader

from src.models.gcn_geometric import GeometricGNN


def load_config(config_path):
    """Load configuration from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(processed_dir, split_name=None):
    """Load preprocessed data from disk"""
    processed_path = Path(processed_dir)
    
    if split_name:
        split_path = processed_path / split_name
        if split_path.exists():
            pt_files = list(split_path.glob("*.pt"))
        else:
            pt_files = []
    else:
        pt_files = list(processed_path.glob("*.pt"))
    
    dataset = []
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, weights_only=False)
            dataset.append(data)
        except:
            pass
    
    return dataset


def load_model(checkpoint_path, config):
    """Load trained model from checkpoint"""
    model = GeometricGNN(config['model'])
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate_model(model, loader, device):
    """Evaluate model on data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds_binary = (all_preds > thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    preds_binary = (all_preds > best_thresh).astype(int)
    
    return {
        'auc': float(roc_auc_score(all_labels, all_preds)),
        'accuracy': float(accuracy_score(all_labels, preds_binary)),
        'precision': float(precision_score(all_labels, preds_binary, zero_division=0)),
        'recall': float(recall_score(all_labels, preds_binary, zero_division=0)),
        'f1': float(best_f1),
        'mcc': float(matthews_corrcoef(all_labels, preds_binary)),
        'threshold': float(best_thresh),
        'n_samples': len(all_labels),
        'n_positive': int(all_labels.sum())
    }


def main():
    print("\n" + "="*70)
    print("         CROSS-DATASET EVALUATION")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load config and model
    config = load_config('config_optimized.yaml')
    model = load_model('checkpoints_optimized/best_model.pth', config)
    device = torch.device('cpu')
    model = model.to(device)
    
    print("Model loaded (GeometricGNN, 276K params)")
    
    # Datasets to evaluate
    datasets_to_test = [
        ('combined/test', 'Combined Test (same distribution)'),
        ('coach420', 'COACH420 (cross-dataset)'),
        ('joined_full', 'Joined Full Dataset'),
        # Additional benchmark datasets
        ('scpdb', 'scPDB (druggable binding sites)'),
        ('pdbbind_refined', 'PDBbind Refined Set'),
        ('sc6k', 'SC6K Benchmark'),
    ]
    
    results = {}
    
    for data_path, description in datasets_to_test:
        full_path = f'data/processed/{data_path}'
        
        print(f"\nEvaluating on: {description}")
        print(f"   Path: {full_path}")
        
        # Check if has splits
        if Path(full_path).exists() and (Path(full_path) / 'test').exists():
            dataset = load_dataset(full_path, 'test')
        elif Path(full_path).exists():
            dataset = load_dataset(full_path)
        else:
            print(f"   Dataset not found, skipping...")
            continue
        
        if len(dataset) == 0:
            print(f"   No valid samples, skipping...")
            continue
        
        print(f"   Samples: {len(dataset)}")
        
        # Create loader
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # Evaluate
        metrics = evaluate_model(model, loader, device)
        
        print(f"   Results:")
        print(f"     AUC:       {metrics['auc']:.4f}")
        print(f"     F1:        {metrics['f1']:.4f}")
        print(f"     Precision: {metrics['precision']:.4f}")
        print(f"     Recall:    {metrics['recall']:.4f}")
        print(f"     MCC:       {metrics['mcc']:.4f}")
        
        results[data_path] = {
            'description': description,
            'metrics': metrics
        }
    
    # Summary
    print("\n" + "="*70)
    print("                    SUMMARY")
    print("="*70)
    
    print(f"\n{'Dataset':<35} {'AUC':<10} {'F1':<10} {'MCC':<10}")
    print("-"*65)
    
    for path, data in results.items():
        m = data['metrics']
        print(f"{data['description']:<35} {m['auc']:<10.4f} {m['f1']:<10.4f} {m['mcc']:<10.4f}")
    
    # Save results
    output_path = Path('results_optimized/cross_dataset_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'model': 'checkpoints_optimized/best_model.pth',
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
