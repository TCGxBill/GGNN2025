#!/usr/bin/env python3
"""
Random Forest Baseline for Binding Site Prediction
Simple ML baseline for comparison with GNN model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                            recall_score, accuracy_score, matthews_corrcoef,
                            precision_recall_curve, roc_curve)
import argparse


def load_dataset(processed_dir, split_name):
    """Load preprocessed data from disk"""
    split_path = Path(processed_dir) / split_name
    pt_files = list(split_path.glob("*.pt"))
    dataset = []
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        dataset.append(data)
    return dataset


def extract_features(dataset, max_samples=None):
    """Extract node features and labels from graph dataset"""
    all_features = []
    all_labels = []
    
    for i, data in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # Get node features (29-dim)
        features = data.x.numpy()
        labels = data.y.numpy()
        
        all_features.append(features)
        all_labels.append(labels)
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y


def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes F1 score"""
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.1, 0.9, 0.02):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1


def evaluate(y_true, y_proba, threshold=0.5):
    """Calculate all metrics"""
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_proba),
        'threshold': threshold
    }


def main():
    parser = argparse.ArgumentParser(description='Random Forest Baseline')
    parser.add_argument('--data', type=str, default='data/processed/combined',
                       help='Path to processed data directory')
    parser.add_argument('--output', type=str, default='results_optimized/rf_baseline.json',
                       help='Output file for results')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees in forest')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max proteins to use (for faster testing)')
    parser.add_argument('--class_weight', type=str, default='balanced',
                       help='Class weight strategy')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("     RANDOM FOREST BASELINE FOR BINDING SITE PREDICTION")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    print("Loading datasets...")
    train_data = load_dataset(args.data, 'train')
    val_data = load_dataset(args.data, 'val')
    test_data = load_dataset(args.data, 'test')
    
    print(f"  Train: {len(train_data)} proteins")
    print(f"  Val:   {len(val_data)} proteins")
    print(f"  Test:  {len(test_data)} proteins")
    
    # Extract features
    print("\nExtracting features...")
    X_train, y_train = extract_features(train_data, args.max_samples)
    X_val, y_val = extract_features(val_data)
    X_test, y_test = extract_features(test_data)
    
    print(f"  Train: {X_train.shape[0]:,} residues, {y_train.sum():,.0f} binding ({y_train.mean()*100:.2f}%)")
    print(f"  Val:   {X_val.shape[0]:,} residues, {y_val.sum():,.0f} binding ({y_val.mean()*100:.2f}%)")
    print(f"  Test:  {X_test.shape[0]:,} residues, {y_test.sum():,.0f} binding ({y_test.mean()*100:.2f}%)")
    
    # Train Random Forest
    print(f"\nTraining Random Forest (n_estimators={args.n_estimators})...")
    
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight=args.class_weight,
        n_jobs=-1,
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    rf.fit(X_train, y_train)
    print("Training complete!")
    
    # Predictions
    print("\nEvaluating...")
    y_val_proba = rf.predict_proba(X_val)[:, 1]
    y_test_proba = rf.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold on validation set
    opt_thresh, opt_f1 = find_optimal_threshold(y_val, y_val_proba)
    print(f"  Optimal threshold: {opt_thresh:.3f} (F1={opt_f1:.4f} on val)")
    
    # Evaluate on test set
    val_metrics = evaluate(y_val, y_val_proba, opt_thresh)
    test_metrics = evaluate(y_test, y_test_proba, opt_thresh)
    
    # Print results
    print("\n" + "-"*70)
    print("RESULTS")
    print("-"*70)
    
    print("\nVALIDATION SET:")
    print(f"  AUC:       {val_metrics['auc']:.4f}")
    print(f"  F1:        {val_metrics['f1']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  MCC:       {val_metrics['mcc']:.4f}")
    
    print("\nTEST SET:")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")
    
    # Feature importance
    print("\nTOP 10 FEATURE IMPORTANCE:")
    feature_names = [
        'AA_A', 'AA_C', 'AA_D', 'AA_E', 'AA_F', 'AA_G', 'AA_H', 'AA_I', 'AA_K', 'AA_L',
        'AA_M', 'AA_N', 'AA_P', 'AA_Q', 'AA_R', 'AA_S', 'AA_T', 'AA_V', 'AA_W', 'AA_Y',
        'SS_Helix', 'SS_Sheet', 'SS_Coil', 'SASA', 'Hydro', 'Charge', 'Polar', 
        'Geo_Angle', 'Geo_Dihedral'
    ]
    
    importance = rf.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    for i in range(min(10, len(indices))):
        idx = indices[i]
        name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        print(f"  {i+1}. {name}: {importance[idx]:.4f}")
    
    # Save results
    results = {
        'method': 'Random Forest',
        'n_estimators': args.n_estimators,
        'class_weight': args.class_weight,
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'optimal_threshold': opt_thresh,
        'validation': val_metrics,
        'test': test_metrics,
        'feature_importance': {
            feature_names[i] if i < len(feature_names) else f"feat_{i}": float(importance[i])
            for i in indices[:15]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Comparison summary
    print("\n" + "="*70)
    print("          COMPARISON WITH GNN MODEL")
    print("="*70)
    print(f"""
                Random Forest    GeometricGNN    Improvement
    AUC         {test_metrics['auc']:.4f}           0.9487          +{(0.9487 - test_metrics['auc'])*100:.1f}%
    F1          {test_metrics['f1']:.4f}           0.6297          +{(0.6297 - test_metrics['f1'])*100:.1f}%
    Precision   {test_metrics['precision']:.4f}           0.6370          +{(0.6370 - test_metrics['precision'])*100:.1f}%
    Recall      {test_metrics['recall']:.4f}           0.6226          +{(0.6226 - test_metrics['recall'])*100:.1f}%
    MCC         {test_metrics['mcc']:.4f}           0.6174          +{(0.6174 - test_metrics['mcc'])*100:.1f}%
    """)


if __name__ == '__main__':
    main()
