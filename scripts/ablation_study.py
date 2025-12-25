#!/usr/bin/env python3
"""
Ablation Study Script for Binding Site Prediction
Analyzes contribution of each component: features, architecture, loss function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import copy
import yaml
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef

from src.models.gcn_geometric import GeometricGNN
from src.training.trainer import BindingSiteTrainer


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_processed_dataset(processed_dir, split_name):
    """Load preprocessed data from disk"""
    processed_path = Path(processed_dir)
    split_path = processed_path / split_name
    pt_files = list(split_path.glob("*.pt"))
    dataset = [torch.load(pt_file, weights_only=False) for pt_file in pt_files]
    return dataset


def evaluate_model(model, loader, device):
    """Quick evaluation on data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            # Handle tuple output (some models return (logits, features))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    
    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds_binary = (all_preds > thresh).astype(int)
        f1 = f1_score(all_labels, preds_binary, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    preds_binary = (all_preds > best_thresh).astype(int)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    mcc = matthews_corrcoef(all_labels, preds_binary)
    
    return {
        'auc': auc,
        'f1': best_f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'threshold': best_thresh
    }


def train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs=10):
    """Train a model and return test metrics"""
    model = GeometricGNN(config['model'])
    
    training_config = {**config['training'], **config.get('logging', {}), **config.get('paths', {})}
    training_config['num_epochs'] = epochs
    training_config['checkpoint_dir'] = './checkpoints_ablation'
    training_config['results_dir'] = './results_ablation'
    training_config['tensorboard_dir'] = './runs_ablation'
    
    trainer = BindingSiteTrainer(model, training_config, device=device)
    trainer.train(train_loader, val_loader, epochs)
    
    # Evaluate on test set
    metrics = evaluate_model(model, test_loader, device)
    return metrics


def run_architecture_ablation(base_config, train_loader, val_loader, test_loader, device, epochs=10):
    """Ablation study on architecture components"""
    results = {}
    
    print("\n📊 Architecture Ablation Study")
    print("-" * 50)
    
    # 1. Full model (baseline)
    print("  [1/4] Full Model (baseline)...")
    config = copy.deepcopy(base_config)
    results['full_model'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['full_model']['auc']:.4f}, F1: {results['full_model']['f1']:.4f}")
    
    # 2. Without attention
    print("  [2/4] Without Attention...")
    config = copy.deepcopy(base_config)
    config['model']['use_attention'] = False
    results['no_attention'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['no_attention']['auc']:.4f}, F1: {results['no_attention']['f1']:.4f}")
    
    # 3. Without geometric encoding
    print("  [3/4] Without Geometric Encoding...")
    config = copy.deepcopy(base_config)
    config['model']['use_geometric'] = False
    results['no_geometric'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['no_geometric']['auc']:.4f}, F1: {results['no_geometric']['f1']:.4f}")
    
    # 4. Fewer layers (2 instead of 3)
    print("  [4/4] Fewer Layers (2)...")
    config = copy.deepcopy(base_config)
    config['model']['num_layers'] = 2
    config['model']['hidden_dims'] = [128, 64]
    results['fewer_layers'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['fewer_layers']['auc']:.4f}, F1: {results['fewer_layers']['f1']:.4f}")
    
    return results


def run_loss_ablation(base_config, train_loader, val_loader, test_loader, device, epochs=10):
    """Ablation study on loss functions"""
    results = {}
    
    print("\n📊 Loss Function Ablation Study")
    print("-" * 50)
    
    # 1. Combined Loss (baseline - 30% BCE + 70% Dice)
    print("  [1/4] Combined Loss (30% BCE + 70% Dice)...")
    config = copy.deepcopy(base_config)
    config['training']['loss_fn'] = 'combined'
    config['training']['bce_weight'] = 0.3
    config['training']['dice_weight'] = 0.7
    results['combined_loss'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['combined_loss']['auc']:.4f}, F1: {results['combined_loss']['f1']:.4f}")
    
    # 2. Pure BCE Loss
    print("  [2/4] Pure BCE Loss...")
    config = copy.deepcopy(base_config)
    config['training']['loss_fn'] = 'bce'
    results['bce_only'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['bce_only']['auc']:.4f}, F1: {results['bce_only']['f1']:.4f}")
    
    # 3. Weighted BCE Loss
    print("  [3/4] Weighted BCE Loss (pos_weight=12)...")
    config = copy.deepcopy(base_config)
    config['training']['loss_fn'] = 'weighted_bce'
    config['training']['pos_weight'] = 12.0
    results['weighted_bce'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['weighted_bce']['auc']:.4f}, F1: {results['weighted_bce']['f1']:.4f}")
    
    # 4. Pure Dice Loss
    print("  [4/4] Pure Dice Loss...")
    config = copy.deepcopy(base_config)
    config['training']['loss_fn'] = 'dice'
    results['dice_only'] = train_ablation_model(config, train_loader, val_loader, test_loader, device, epochs)
    print(f"       AUC: {results['dice_only']['auc']:.4f}, F1: {results['dice_only']['f1']:.4f}")
    
    return results


def save_results(results, output_path):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--config', type=str, default='config_optimized.yaml',
                       help='Base config file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Epochs for each ablation experiment')
    parser.add_argument('--output_dir', type=str, default='results_optimized',
                       help='Output directory for results')
    parser.add_argument('--study', type=str, default='all',
                       choices=['all', 'architecture', 'loss'],
                       help='Which ablation study to run')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("          ABLATION STUDY")
    print("="*60)
    
    # Load config
    config = load_config(args.config)
    set_seed(42)
    
    device = torch.device('cpu')
    print(f"✓ Using device: {device}")
    
    # Load data
    print("\n📊 Loading dataset...")
    processed_dir = config['data'].get('processed_dir', 'data/processed/combined')
    train_dataset = load_processed_dataset(processed_dir, 'train')
    val_dataset = load_processed_dataset(processed_dir, 'val')
    test_dataset = load_processed_dataset(processed_dir, 'test')
    
    # Use smaller subset for ablation (faster)
    max_samples = min(500, len(train_dataset))
    train_dataset = train_dataset[:max_samples]
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    batch_size = config['training'].get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_results = {}
    
    # Run selected ablation studies
    if args.study in ['all', 'architecture']:
        arch_results = run_architecture_ablation(config, train_loader, val_loader, test_loader, device, args.epochs)
        all_results['architecture'] = arch_results
    
    if args.study in ['all', 'loss']:
        loss_results = run_loss_ablation(config, train_loader, val_loader, test_loader, device, args.epochs)
        all_results['loss_function'] = loss_results
    
    # Save results
    output_path = Path(args.output_dir) / 'ablation_results.json'
    save_results(all_results, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("          ABLATION STUDY COMPLETE")
    print("="*60)
    
    for study_name, study_results in all_results.items():
        print(f"\n{study_name.upper()}:")
        print("-" * 40)
        for exp_name, metrics in study_results.items():
            print(f"  {exp_name:20s}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}")


if __name__ == '__main__':
    main()
