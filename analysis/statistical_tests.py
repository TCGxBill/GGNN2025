#!/usr/bin/env python3
"""
Statistical Significance Tests for GGNN 2025 Paper

Implements:
- Paired t-test between methods
- Wilcoxon signed-rank test (non-parametric)
- Bootstrap 95% confidence intervals
- McNemar's test for binary predictions
"""

import os
import sys
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
import torch
from pathlib import Path
from tqdm import tqdm
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def bootstrap_ci(y_true, y_scores, metric_fn, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval for a metric."""
    np.random.seed(42)
    n = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]
        
        try:
            score = metric_fn(y_true_boot, y_scores_boot)
            bootstrap_scores.append(score)
        except:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_scores, alpha/2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'ci': ci
    }


def wilcoxon_test(scores1, scores2):
    """Perform Wilcoxon signed-rank test."""
    statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='greater')
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'significant_0.001': p_value < 0.001
    }


def paired_ttest(scores1, scores2):
    """Perform paired t-test."""
    statistic, p_value = stats.ttest_rel(scores1, scores2)
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
        'significant_0.001': p_value < 0.001
    }


def mcnemar_test(pred1, pred2, y_true):
    """
    McNemar's test for comparing binary classifiers.
    Tests if two classifiers have the same error rate.
    """
    # Create contingency table
    # b = pred1 correct, pred2 wrong
    # c = pred1 wrong, pred2 correct
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)
    
    b = np.sum(correct1 & ~correct2)  # pred1 right, pred2 wrong
    c = np.sum(~correct1 & correct2)  # pred1 wrong, pred2 right
    
    # McNemar's test with continuity correction
    if b + c == 0:
        return {'statistic': 0, 'p_value': 1.0, 'significant_0.05': False}
    
    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        'b': int(b),
        'c': int(c),
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant_0.05': p_value < 0.05
    }


def evaluate_per_protein(model, data_dir, threshold=0.5):
    """Evaluate model and return per-protein metrics."""
    from models.gcn_geometric import GeometricGNN
    
    model.eval()
    per_protein_results = []
    
    pt_files = list(Path(data_dir).glob("*.pt"))
    
    with torch.no_grad():
        for pt_file in tqdm(pt_files, desc=f"Evaluating {data_dir}"):
            try:
                data = torch.load(pt_file, weights_only=False)
                output = model(data)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                scores = torch.sigmoid(output).numpy().flatten()
                labels = data.y.numpy().flatten()
                preds = (scores >= threshold).astype(int)
                
                # Calculate per-protein metrics
                auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
                f1 = f1_score(labels, preds, zero_division=0)
                mcc = matthews_corrcoef(labels, preds)
                
                per_protein_results.append({
                    'pdb_id': pt_file.stem,
                    'auc': auc,
                    'f1': f1,
                    'mcc': mcc,
                    'n_residues': len(labels),
                    'n_binding': int(labels.sum()),
                    'labels': labels,
                    'scores': scores,
                    'preds': preds
                })
            except Exception as e:
                continue
    
    return per_protein_results


def main():
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    
    # Load model
    with open('config_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from models.gcn_geometric import GeometricGNN
    model = GeometricGNN(config['model'])
    ckpt = torch.load('checkpoints_optimized/best_model.pth', 
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    results = {}
    
    # Evaluate on test set
    print("\n[1] Evaluating on Combined Test Set...")
    test_results = evaluate_per_protein(model, 'data/processed/combined/test')
    
    if test_results:
        # Collect all predictions
        all_labels = np.concatenate([r['labels'] for r in test_results])
        all_scores = np.concatenate([r['scores'] for r in test_results])
        all_preds = np.concatenate([r['preds'] for r in test_results])
        
        # Per-protein AUC scores
        auc_scores = np.array([r['auc'] for r in test_results])
        mcc_scores = np.array([r['mcc'] for r in test_results])
        
        print(f"  Proteins: {len(test_results)}")
        print(f"  Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
        print(f"  Mean MCC: {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}")
        
        # Bootstrap CI for AUC
        print("\n[2] Bootstrap Confidence Intervals (n=1000)...")
        auc_ci = bootstrap_ci(all_labels, all_scores, roc_auc_score, n_bootstrap=1000)
        print(f"  AUC: {auc_ci['mean']:.4f} (95% CI: [{auc_ci['ci_lower']:.4f}, {auc_ci['ci_upper']:.4f}])")
        
        # MCC CI (need binary threshold)
        def mcc_metric(y_true, y_scores):
            return matthews_corrcoef(y_true, (y_scores >= 0.5).astype(int))
        
        mcc_ci = bootstrap_ci(all_labels, all_scores, mcc_metric, n_bootstrap=1000)
        print(f"  MCC: {mcc_ci['mean']:.4f} (95% CI: [{mcc_ci['ci_lower']:.4f}, {mcc_ci['ci_upper']:.4f}])")
        
        results['test_set'] = {
            'n_proteins': len(test_results),
            'n_residues': len(all_labels),
            'auc': {
                'mean': float(np.mean(auc_scores)),
                'std': float(np.std(auc_scores)),
                'bootstrap_ci': auc_ci
            },
            'mcc': {
                'mean': float(np.mean(mcc_scores)),
                'std': float(np.std(mcc_scores)),
                'bootstrap_ci': mcc_ci
            }
        }
    
    # Compare with random baseline
    print("\n[3] Comparison with Random Baseline...")
    np.random.seed(42)
    random_scores = np.random.rand(len(all_scores))
    random_auc = roc_auc_score(all_labels, random_scores)
    
    # Create per-protein random AUCs for comparison
    random_aucs = []
    for r in test_results:
        rand_scores = np.random.rand(len(r['labels']))
        try:
            rand_auc = roc_auc_score(r['labels'], rand_scores)
        except:
            rand_auc = 0.5
        random_aucs.append(rand_auc)
    random_aucs = np.array(random_aucs)
    
    # Wilcoxon test: GGNN vs Random
    wilcox_result = wilcoxon_test(auc_scores, random_aucs)
    print(f"  Random AUC: {random_auc:.4f}")
    print(f"  Wilcoxon test (GGNN > Random): p = {wilcox_result['p_value']:.2e}")
    print(f"    Significant at α=0.001: {'***' if wilcox_result['significant_0.001'] else 'No'}")
    
    # Paired t-test
    ttest_result = paired_ttest(auc_scores, random_aucs)
    print(f"  Paired t-test: p = {ttest_result['p_value']:.2e}")
    
    results['vs_random'] = {
        'random_auc': float(random_auc),
        'wilcoxon_test': wilcox_result,
        'paired_ttest': ttest_result
    }
    
    # McNemar's test vs Random
    print("\n[4] McNemar's Test (Binary Predictions)...")
    random_preds = (random_scores >= 0.5).astype(int)
    mcnemar_result = mcnemar_test(all_preds, random_preds, all_labels.astype(int))
    print(f"  McNemar statistic: {mcnemar_result['statistic']:.2f}")
    print(f"  p-value: {mcnemar_result['p_value']:.2e}")
    print(f"  Significant at α=0.05: {'Yes' if mcnemar_result['significant_0.05'] else 'No'}")
    
    results['mcnemar_vs_random'] = mcnemar_result
    
    # Per-benchmark CI
    print("\n[5] Per-Benchmark Confidence Intervals...")
    benchmarks = {
        'scpdb': 'data/processed/scpdb',
        'pdbbind': 'data/processed/pdbbind_refined',
        'coach420': 'data/processed/coach420',
        'moad': ['data/processed/moad_quality', 'data/processed/moad_expanded'],
        'cryptobench': ['data/processed/cryptobench', 'data/processed/cryptobench_expanded']
    }
    
    results['benchmarks'] = {}
    for name, paths in benchmarks.items():
        if isinstance(paths, str):
            paths = [paths]
        
        all_labels_bm = []
        all_scores_bm = []
        
        for path in paths:
            if not os.path.exists(path):
                continue
            bm_results = evaluate_per_protein(model, path)
            for r in bm_results:
                all_labels_bm.extend(r['labels'])
                all_scores_bm.extend(r['scores'])
        
        if all_labels_bm:
            all_labels_bm = np.array(all_labels_bm)
            all_scores_bm = np.array(all_scores_bm)
            
            auc = roc_auc_score(all_labels_bm, all_scores_bm)
            auc_ci = bootstrap_ci(all_labels_bm, all_scores_bm, roc_auc_score, n_bootstrap=500)
            
            print(f"  {name}: AUC = {auc:.4f} (95% CI: [{auc_ci['ci_lower']:.4f}, {auc_ci['ci_upper']:.4f}])")
            
            results['benchmarks'][name] = {
                'auc': float(auc),
                'ci_lower': auc_ci['ci_lower'],
                'ci_upper': auc_ci['ci_upper']
            }
    
    # Save results
    output_file = 'results_optimized/statistical_tests.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print("=" * 70)
    
    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    if 'test_set' in results:
        auc_data = results['test_set']['auc']
        mcc_data = results['test_set']['mcc']
        print(f"Combined Test Set (n={results['test_set']['n_proteins']}):")
        print(f"  AUC = {auc_data['bootstrap_ci']['mean']:.3f} (95% CI: {auc_data['bootstrap_ci']['ci_lower']:.3f}-{auc_data['bootstrap_ci']['ci_upper']:.3f})")
        print(f"  MCC = {mcc_data['bootstrap_ci']['mean']:.3f} (95% CI: {mcc_data['bootstrap_ci']['ci_lower']:.3f}-{mcc_data['bootstrap_ci']['ci_upper']:.3f})")
        print(f"\nStatistical Significance vs Random:")
        print(f"  Wilcoxon p < {results['vs_random']['wilcoxon_test']['p_value']:.0e}")
        print(f"  Paired t-test p < {results['vs_random']['paired_ttest']['p_value']:.0e}")


if __name__ == "__main__":
    main()
