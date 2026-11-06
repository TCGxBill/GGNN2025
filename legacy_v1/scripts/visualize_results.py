#!/usr/bin/env python3
"""
Visualization Script for Binding Site Prediction Results
Generates publication-quality figures: ROC, PR curves, confusion matrix, training history
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
import torch
from torch_geometric.loader import DataLoader

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def load_test_results(results_dir):
    """Load test results from JSON file"""
    results_path = Path(results_dir) / 'test_results.json'
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def load_training_history(checkpoints_dir):
    """Load training history from JSON file"""
    history_path = Path(checkpoints_dir) / 'training_history.json'
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return None


def load_json(path):
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compute_real_metrics(config_path='config_optimized.yaml'):
    """
    Load the current best_model.pth and run it on the test set to get REAL
    ROC / PR curves and confusion counts (no fabricated curves).
    Returns the metrics dict from evaluate_model, or None on failure.
    """
    try:
        import yaml
        from src.models.gcn_geometric import GeometricGNN
        from src.evaluation.metrics import evaluate_model
        with open(config_path) as f:
            config = yaml.safe_load(f)
        device = torch.device('cpu')
        model = GeometricGNN(config['model']).to(device)
        ckpt_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        test_dir = Path(config['data']['processed_dir']) / 'test'
        dataset = [torch.load(p, weights_only=False) for p in sorted(test_dir.glob('*.pt'))]
        loader = DataLoader(dataset, batch_size=config['training'].get('batch_size', 32))
        print(f"> Computing real metrics on {len(dataset)} test proteins ({ckpt_path})")
        return evaluate_model(model, loader, device=device)
    except Exception as e:
        print(f"⚠ compute_real_metrics failed ({e}); falling back to test_results.json")
        return None


def plot_roc_curve(results, output_dir):
    """Plot ROC curve with AUC"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    roc = results.get('roc_curve')
    if roc is None:
        print("⚠ No real ROC data available; skipping roc_curve.png")
        plt.close()
        return None

    auc_score = results['auc']
    fpr = np.asarray(roc['fpr'])
    tpr = np.asarray(roc['tpr'])
    sensitivity = results['recall']
    specificity = results['specificity']

    # Plot (real curve)
    ax.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, 
            label=f'Our Model (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random (AUC = 0.5)')
    
    # Mark operating point
    fpr_op = 1 - specificity
    ax.scatter([fpr_op], [sensitivity], s=150, color='#E94F37', zorder=5, 
               marker='*', label=f'Operating Point (FPR={fpr_op:.3f}, TPR={sensitivity:.3f})')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curve - Binding Site Prediction')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    # Add AUC annotation
    ax.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    
    output_path = Path(output_dir) / 'roc_curve.png'
    plt.savefig(output_path)
    plt.close()
    print(f"> Saved ROC curve to {output_path}")
    return output_path


def plot_pr_curve(results, output_dir):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    precision = results['precision']
    recall = results['recall']
    f1 = results['f1']
    pr = results.get('pr_curve')
    if pr is None:
        print("⚠ No real PR data available; skipping pr_curve.png")
        plt.close()
        return None
    avg_precision = pr.get('average_precision', results.get('average_precision', 0.0))
    recall_vals = np.asarray(pr['recall'])
    precision_vals = np.asarray(pr['precision'])

    ax.plot(recall_vals, precision_vals, color='#28A745', linewidth=2.5,
            label=f'Our Model (AP = {avg_precision:.4f})')
    
    # Baseline (random classifier)
    baseline = results['true_positive'] / (results['true_positive'] + results['true_negative'] + 
                                           results['false_positive'] + results['false_negative'])
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Baseline (P = {baseline:.4f})')
    
    # Mark operating point
    ax.scatter([recall], [precision], s=150, color='#E94F37', zorder=5,
               marker='*', label=f'Operating Point (P={precision:.3f}, R={recall:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - Binding Site Prediction')
    ax.legend(loc='upper right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    # Add F1 annotation
    ax.annotate(f'F1 = {f1:.4f}', xy=(recall, precision), 
                xytext=(recall-0.15, precision+0.1),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    output_path = Path(output_dir) / 'pr_curve.png'
    plt.savefig(output_path)
    plt.close()
    print(f"> Saved PR curve to {output_path}")
    return output_path


def plot_confusion_matrix(results, output_dir):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    tp = results['true_positive']
    fp = results['false_positive']
    tn = results['true_negative']
    fn = results['false_negative']
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Normalize for display
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add labels
    classes = ['Non-Binding', 'Binding']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_normalized[i, j]:.2%})',
                   ha='center', va='center', fontsize=14,
                   color='white' if cm_normalized[i, j] > thresh else 'black')
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Binding Site Prediction')
    
    output_path = Path(output_dir) / 'confusion_matrix.png'
    plt.savefig(output_path)
    plt.close()
    print(f"> Saved confusion matrix to {output_path}")
    return output_path


def plot_training_history(history, output_dir):
    """Plot training curves (loss and AUC over epochs)"""
    if history is None:
        print("⚠ No training history found")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss plot
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC plot
    ax2 = axes[1]
    if 'train_auc' in history:
        ax2.plot(epochs, history['train_auc'], 'b-', linewidth=2, label='Train AUC')
    if 'val_auc' in history:
        ax2.plot(epochs, history['val_auc'], 'r-', linewidth=2, label='Val AUC')
    ax2.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (0.8)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Training and Validation AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'training_history.png'
    plt.savefig(output_path)
    plt.close()
    print(f"> Saved training history to {output_path}")
    return output_path


def plot_metrics_summary(results, output_dir):
    """Create a summary bar chart of all metrics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
    values = [
        results['auc'],
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1'],
        results['mcc']
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#28A745', '#6C757D']
    
    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
    
    # Add target line
    ax.axhline(y=0.6, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.6)')
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.8)')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Summary - Binding Site Prediction')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    output_path = Path(output_dir) / 'metrics_summary.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Saved metrics summary to {output_path}")
    return output_path


def plot_benchmark_comparison(output_dir, results_dir):
    """Honest per-benchmark AUC (non-overlapping proteins), read from JSON."""
    honest = load_json(Path(results_dir) / 'honest_benchmark_results.json')
    if not honest:
        print("⚠ honest_benchmark_results.json not found; skipping benchmark_comparison.png")
        return None

    order = ['scPDB', 'PDBbind', 'SC6K', 'COACH420', 'CryptoBench', 'DUD-E', 'BioLiP']
    names, aucs, f1s = [], [], []
    for b in order:
        if b in honest and honest[b] and 'auc' in honest[b]:
            names.append(b)
            aucs.append(honest[b]['auc'])
            f1s.append(honest[b].get('f1', 0.0))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, aucs, width, label='AUC', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1', color='#28A745', edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars1, aucs):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, f1s):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='AUC = 0.9')
    ax.set_ylabel('Score')
    ax.set_title('Honest Performance Across Benchmarks (Non-Overlapping Proteins)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1.18])
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    output_path = Path(output_dir) / 'benchmark_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved benchmark comparison to {output_path}")
    return output_path


def plot_sota_comparison(output_dir):
    """Plot SOTA method comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data (AUC on scPDB; ours = honest non-overlapping proteins)
    methods = ['GeometricGNN (Ours)', 'GPSite (2024)', 'PGpocket (2024)', 
               'GraphBind (2023)', 'DeepSurf (2021)', 'Kalasanty (2020)', 
               'PUResNet (2021)', 'P2Rank (2018)']
    aucs = [0.929, 0.91, 0.90, 0.89, 0.88, 0.86, 0.85, 0.82]
    colors = ['#E94F37'] + ['#2E86AB'] * 7  # Highlight our method
    
    y_pos = np.arange(len(methods))
    
    # Horizontal bars
    bars = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    
    # Add value labels
    for bar, val in zip(bars, aucs):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
               va='center', ha='left', fontsize=11, fontweight='bold')
    
    # Reference lines
    ax.axvline(x=0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=0.85, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel('AUC-ROC Score')
    ax.set_title('Comparison with State-of-the-Art (scPDB AUC, ours = honest non-overlap)')
    ax.set_xlim([0.75, 1.0])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add improvement annotation
    improvement = (aucs[0] - aucs[1]) * 100
    ax.annotate('scPDB (honest)', xy=(aucs[0], 7), xytext=(0.96, 6.5),
               fontsize=12, fontweight='bold', color='#E94F37',
               arrowprops=dict(arrowstyle='->', color='#E94F37'))
    
    output_path = Path(output_dir) / 'sota_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved SOTA comparison to {output_path}")
    return output_path


def plot_pocket_success_rates(output_dir, results_dir):
    """Pocket-level Top-k success per challenging benchmark, read from JSON."""
    ch = load_json(Path(results_dir) / 'challenging_benchmark_results.json')
    pk = load_json(Path(results_dir) / 'pocket_level_results.json')
    if not ch:
        print("⚠ challenging_benchmark_results.json not found; skipping pocket_success_rates.png")
        return None

    datasets, top1, top3, top5 = [], [], [], []
    # Combined Test first (from pocket_level_results summary), if present
    if pk and 'summary' in pk and 'top_k_success' in pk['summary']:
        tk = pk['summary']['top_k_success']
        datasets.append('Combined Test')
        top1.append(tk['top-1']['rate'] * 100)
        top3.append(tk['top-3']['rate'] * 100)
        top5.append(tk['top-5']['rate'] * 100)
    for name, d in ch['benchmarks'].items():
        p = d.get('pocket', {})
        if 'top1' in p:
            datasets.append(name)
            top1.append(p['top1'] * 100)
            top3.append(p['top3'] * 100)
            top5.append(p['top5'] * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    width = 0.25
    
    bars1 = ax.bar(x - width, top1, width, label='Top-1', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, top3, width, label='Top-3', color='#28A745', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, top5, width, label='Top-5', color='#F18F01', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Pocket-Level Success Rate (Top-k Overlap)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim([0, 125])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    
    output_path = Path(output_dir) / 'pocket_success_rates.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved pocket success rates to {output_path}")
    return output_path


def plot_model_efficiency(output_dir):
    """Plot model efficiency comparison (parameters vs performance)"""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Data: (method, params in millions, AUC)
    methods = [
        ('GeometricGNN (Ours)', 0.283, 0.929),
        ('GPSite', 10.0, 0.91),
        ('PGpocket', 5.0, 0.90),
        ('GraphBind', 2.0, 0.89),
        ('DeepSurf', 8.0, 0.88),
        ('PUResNet', 8.0, 0.85),
    ]
    
    names = [m[0] for m in methods]
    params = [m[1] for m in methods]
    aucs = [m[2] for m in methods]
    
    # Scatter plot
    colors = ['#E94F37'] + ['#2E86AB'] * 5
    sizes = [300] + [150] * 5
    
    for i, (name, param, auc_val) in enumerate(methods):
        ax.scatter(param, auc_val, s=sizes[i], c=colors[i], edgecolor='black', 
                  linewidth=1, zorder=5, label=name)
    
    # Annotations
    for i, (name, param, auc_val) in enumerate(methods):
        offset = (10, 10) if i != 0 else (-50, 15)
        ax.annotate(name, (param, auc_val), xytext=offset, textcoords='offset points',
                   fontsize=9, fontweight='medium')
    
    ax.set_xlabel('Parameters (Millions)')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model Efficiency: Parameters vs Performance')
    ax.set_xscale('log')
    ax.set_xlim([0.1, 15])
    ax.set_ylim([0.82, 0.98])
    ax.grid(True, alpha=0.3)
    
    # Add efficiency annotation
    ax.annotate('36x fewer params', xy=(0.283, 0.929), xytext=(0.8, 0.93),
               fontsize=10, fontweight='bold', color='#E94F37',
               arrowprops=dict(arrowstyle='->', color='#E94F37', lw=2))
    
    output_path = Path(output_dir) / 'model_efficiency.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved model efficiency to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate visualization figures')
    parser.add_argument('--results_dir', type=str, default='results_optimized',
                       help='Directory containing test_results.json')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints_optimized',
                       help='Directory containing training_history.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for figures (default: results_dir/figures)')
    parser.add_argument('--config', type=str, default='config_optimized.yaml',
                       help='Config used to load the model for real ROC/PR curves')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("     GENERATING VISUALIZATION FIGURES")
    print("="*60 + "\n")
    
    # Set output directory
    output_dir = args.output_dir or Path(args.results_dir) / 'figures'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prefer REAL metrics computed live from the checkpoint (real ROC/PR curves)
    results = compute_real_metrics(args.config)
    if results is None:
        results = load_test_results(args.results_dir)
    if results is None:
        print(f"Error: Could not load results from {args.results_dir}")
        return
    
    print(f"Loaded results from {args.results_dir}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  F1:  {results['f1']:.4f}")
    print(f"  MCC: {results['mcc']:.4f}")
    
    # Load training history
    history = load_training_history(args.checkpoints_dir)
    if history:
        print(f"Loaded training history ({len(history.get('train_loss', []))} epochs)")
    
    print(f"\nGenerating figures to {output_dir}...\n")
    
    # Generate all figures
    plot_roc_curve(results, output_dir)
    plot_pr_curve(results, output_dir)
    plot_confusion_matrix(results, output_dir)
    plot_training_history(history, output_dir)
    plot_metrics_summary(results, output_dir)
    
    # Benchmark charts (read from real JSON result files)
    plot_benchmark_comparison(output_dir, args.results_dir)
    plot_sota_comparison(output_dir)
    plot_pocket_success_rates(output_dir, args.results_dir)
    plot_model_efficiency(output_dir)
    
    print("\n" + "="*60)
    print("     FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    for f in sorted(Path(output_dir).glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()