#!/usr/bin/env python3
"""
Main Training Script for Binding Site Prediction
Usage: python scripts/train.py --config config.yaml
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader

# Import project modules
from src.models.gcn_geometric import GeometricGNN
from src.training.trainer import BindingSiteTrainer
from src.evaluation.metrics import evaluate_model


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_processed_dataset(processed_dir, split_name=None):
    """Load preprocessed data from disk"""
    processed_path = Path(processed_dir)
    
    if split_name:
        # Load specific split
        split_path = processed_path / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Split not found: {split_path}")
        pt_files = list(split_path.glob("*.pt"))
    else:
        # Load all data
        pt_files = list(processed_path.glob("*.pt"))
    
    dataset = []
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        dataset.append(data)
    
    return dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train/val/test"""
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]
    
    return train_dataset, val_dataset, test_dataset


def main(args):
    """Main training function"""
    
    print("\n" + "="*70)
    print(" "*15 + "BINDING SITE PREDICTION TRAINING")
    print("="*70 + "\n")
    
    # Load configuration
    config = load_config(args.config)
    print(f"> Configuration loaded from {args.config}")
    
    # Set random seed
    seed = args.seed if args.seed is not None else config['project']['random_seed']
    set_seed(seed)
    print(f"> Random seed set to {seed}")
    
    # Set device
    config_device = config.get('device', 'cpu')
    if config_device == 'cpu':
        device = torch.device('cpu')
    elif config_device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config_device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        # Auto-detect
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            # device = torch.device('mps') 
            # Disable MPS auto-detect for now due to performance issues
            device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    
    print(f"> Using device: {device}")
    
    # Load real dataset
    print("\n Loading dataset...")
    processed_dir = config['data'].get('processed_dir', 'data/processed')
    
    # Check if data exists
    if not Path(processed_dir).exists():
        print(f"Error: Processed data not found at {processed_dir}")
        print("Run: python scripts/preprocess_all.py --dataset pdbbind --split")
        sys.exit(1)
    
    # Load train/val/test splits
    try:
        train_dataset = load_processed_dataset(processed_dir, 'train')
        val_dataset = load_processed_dataset(processed_dir, 'val')
        test_dataset = load_processed_dataset(processed_dir, 'test')
    except FileNotFoundError:
        print("Error: Train/val/test splits not found")
        print("Run: python scripts/preprocess_all.py --dataset pdbbind --split")
        sys.exit(1)
    
    print(f"> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"> Data loaders created (batch_size={batch_size})")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model_config = config['model']
    model = GeometricGNN(model_config)
    
    # Load checkpoint if resuming
    if args.resume and Path(args.resume).exists():
        print(f"\n📥 Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"> Loaded model state from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("> Loaded model state from checkpoint")
        # Reduce learning rate for fine-tuning
        config['training']['learning_rate'] = config['training']['learning_rate'] * 0.5
        print(f"> Reduced learning rate to {config['training']['learning_rate']} for fine-tuning")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"> Model: {model_config['type']}")
    print(f"> Parameters: {num_params:,}")
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    training_config = {**config['training'], **config['logging'], **config['paths']}
    trainer = BindingSiteTrainer(model, training_config, device=device)
    
    print(f"> Optimizer: {config['training']['optimizer']}")
    print(f"> Learning rate: {config['training']['learning_rate']}")
    print(f"> Loss function: {config['training']['loss_fn']}")
    
    # Train model
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    num_epochs = config['training']['num_epochs']
    history = trainer.train(train_loader, val_loader, num_epochs)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70 + "\n")
    
    checkpoint_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
    trainer.load_checkpoint(checkpoint_path)
    
    test_metrics = evaluate_model(trainer.model, test_loader, device=device)
    
    print("\nTest Set Results:")
    print("-" * 50)
    print(f"AUC:           {test_metrics['auc']:.4f}")
    print(f"Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Precision:     {test_metrics['precision']:.4f}")
    print(f"Recall:        {test_metrics['recall']:.4f}")
    print(f"F1-Score:      {test_metrics['f1']:.4f}")
    print(f"MCC:           {test_metrics['mcc']:.4f}")
    print("-" * 50)
    
    # Save results
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    import json
    results_path = results_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        # Remove non-serializable items
        serializable_metrics = {
            k: v for k, v in test_metrics.items() 
            if not k.endswith('_curve')
        }
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\n> Results saved to {results_path}")
    
    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train binding site prediction model"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples (for testing)'
    )
    
    parser.add_argument(
        '--num_residues',
        type=int,
        default=None,
        help='Number of residues per protein (for testing)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\n Error: {str(e)}")
        raise