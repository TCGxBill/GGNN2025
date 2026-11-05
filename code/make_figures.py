#!/usr/bin/env python3
"""
Regenerate every GGNN 2025 figure from the result files, as vector PDF + 400 dpi PNG.

The PNGs shipped before this script were produced in a sandbox session and the generating
code was lost; nothing on disk could reproduce them, and one of them (metric inversion)
carried permissive-label numbers next to a strict-label table. Every figure here names the
result file it reads and the label regime it belongs to, so the provenance is checkable.

Outputs land in figures_v2/.  Run: python code/make_figures.py
"""
import csv, json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R, OUTD = ROOT/"results", ROOT/"figures"
OUTD.mkdir(exist_ok=True)
GG, P2, FP = "#1b4f72", "#b0b7bd", "#c9ced3"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "figure.dpi": 120,
                     "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})

V    = json.load(open(R/"verify_strict_comparison.json"))   # strict labels, B5_strict_lf
B    = json.load(open(R/"B5_strict_lf.json"))               # strict labels, final checkpoint
seed = list(csv.DictReader(open(R/"seed_replicates.csv")))  # PERMISSIVE labels
inv  = list(csv.DictReader(open(R/"metric_inversion.csv"))) # PERMISSIVE labels, B5_localframe


def save(fig, name):
    fig.savefig(OUTD/f"{name}.pdf")
    fig.savefig(OUTD/f"{name}.png", dpi=400)
    plt.close(fig)
    print("wrote", name)


def fig_strict_comparison():
    """Strict labels. Pooled AUPRC + paired bootstrap CIs. THE headline comparison."""
    meth, key, cols = ["Fpocket", "P2Rank", "GGNN 2025"], ["Fpocket", "P2Rank", "GGNN"], [FP, P2, GG]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
    vals, y = [V["pooled"][k]["auprc"] for k in key], np.arange(3)
    ax1.hlines(y, 0, vals, color=cols, lw=5)
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax1.plot(v, i, "o", ms=8, color=c)
        ax1.text(v + 0.015, i, f"{v:.4f}", va="center", fontsize=7)
    ax1.set_yticks(y); ax1.set_yticklabels(meth); ax1.set_xlim(0, 0.62)
    ax1.set_xlabel("Residue-level AUPRC  (higher = better)")
    ax1.set_title("Same 244 proteins, same artifact-free labels", loc="left")

    pairs = [("GGNN_minus_P2Rank_auprc", "vs P2Rank\nAUPRC"), ("GGNN_minus_P2Rank_auroc", "vs P2Rank\nAUROC"),
             ("GGNN_minus_Fpocket_auprc", "vs Fpocket\nAUPRC"), ("GGNN_minus_Fpocket_auroc", "vs Fpocket\nAUROC")]
    yy = np.arange(len(pairs))[::-1]
    for (k, lab), ypos in zip(pairs, yy):
        b = V["bootstrap"][k]
        c = "#7a7f85" if b["lo"] < 0 < b["hi"] else GG      # grey = interval crosses zero
        ax2.plot([b["lo"], b["hi"]], [ypos, ypos], lw=2.4, color=c, solid_capstyle="round")
        ax2.plot(b["mean"], ypos, "o", ms=7, color=c)
        ax2.text(b["hi"] + 0.014, ypos, f"{b['mean']:+.4f}", va="center", fontsize=7, color=c)
    ax2.axvline(0, color="#444", lw=0.9, ls=(0, (4, 2)))
    ax2.set_yticks(yy); ax2.set_yticklabels([l for _, l in pairs], fontsize=7)
    ax2.set_xlabel("Difference (GGNN 2025 − comparator), 95% CI")
    ax2.set_title("Paired bootstrap over proteins (2,000 replicates)", loc="left")
    ax2.set_xlim(-0.12, 0.44)
    fig.tight_layout(); save(fig, "fig_strict_comparison")


def fig_leakage_cost():
    """Strict labels. Same checkpoint on the leaky and the cluster-disjoint test set."""
    leaky = [B["evaluations"]["T_leaky"]["auroc"], B["evaluations"]["T_leaky"]["auprc"]]
    clean = [B["evaluations"]["T_clean"]["auroc"], B["evaluations"]["T_clean"]["auprc"]]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for i in range(2):
        drop = 100 * (leaky[i] - clean[i]) / leaky[i]
        ax.plot([i, i], [clean[i], leaky[i]], color="#9aa0a6", lw=2, zorder=1)
        ax.plot(i, leaky[i], "o", ms=9, color="#c0392b", zorder=2)
        ax.plot(i, clean[i], "o", ms=9, color=GG, zorder=2)
        ax.text(i + 0.10, (leaky[i] + clean[i]) / 2, f"−{drop:.1f}%", va="center",
                fontsize=8.5, color="#c0392b", fontweight="bold")
        ax.text(i - 0.10, leaky[i], f"{leaky[i]:.4f}", ha="right", va="center", fontsize=7)
        ax.text(i - 0.10, clean[i], f"{clean[i]:.4f}", ha="right", va="center", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["AUROC", "AUPRC"]); ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.40, 1.02); ax.set_ylabel("Score")
    ax.set_title("AUROC survives homology control; AUPRC does not", loc="left")
    ax.plot([], [], "o", color="#c0392b", label="test set with close training homologues (316 proteins)")
    ax.plot([], [], "o", color=GG, label="cluster-disjoint test set (244 proteins)")
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(); save(fig, "fig_leakage_cost")


def fig_seed_replicates():
    """PERMISSIVE labels, B4/B5 localframe. Three seeds per configuration."""
    LBL = {"B4_global": "Global frame\n(1,809 proteins)", "B4_localframe": "Local frame\n(1,809 proteins)",
           "B5_localframe": "Local frame\n(3,412 proteins)"}
    x = np.arange(len(seed))
    mu = np.array([float(r["auprc_mean"]) for r in seed]); sd = np.array([float(r["auprc_sd"]) for r in seed])
    mn = np.array([float(r["auprc_min"]) for r in seed]);  mx = np.array([float(r["auprc_max"]) for r in seed])
    cols = ["#9aa0a6", GG, GG]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for i in range(len(seed)):
        ax.plot([x[i], x[i]], [mn[i], mx[i]], lw=1.2, color=cols[i], alpha=0.55)
        ax.errorbar(x[i], mu[i], yerr=sd[i], fmt="o", ms=8, color=cols[i], capsize=4, lw=1.6)
        ax.text(x[i] + 0.12, mu[i], f"{mu[i]:.4f}\n± {sd[i]:.4f}", va="center", fontsize=6.8)
    ax.annotate("", xy=(0.97, mu[1] - 0.005), xytext=(0.03, mu[0] + 0.005),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))
    ax.text(0.5, mu[1] + 0.004, "+11.6%, p = 0.020", ha="center", fontsize=7.5, color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels([LBL[r["config"]] for r in seed], fontsize=7)
    ax.set_xlim(-0.5, 2.7); ax.set_ylabel("Clean-test AUPRC")
    ax.set_title("Local frames beat more data, at zero parameter cost", loc="left")
    fig.tight_layout(); save(fig, "fig_seed_replicates_v2")


def fig_metric_inversion():
    """PERMISSIVE labels, 246 proteins, B5_localframe. Kept separate from the strict table."""
    nm  = [r["method"] for r in inv]
    res = [float(r["res_auprc"]) for r in inv]
    lo  = [float(r["res_auprc"]) - float(r["res_auprc_lo"]) for r in inv]
    hi  = [float(r["res_auprc_hi"]) - float(r["res_auprc"]) for r in inv]
    pk  = [100 * float(r["pocket_top1_dca4"]) for r in inv]
    cols, xx = [FP, P2, GG], np.arange(3)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    axa.bar(xx, res, color=cols, width=0.6)
    axa.errorbar(xx, res, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=3, lw=1)
    for i, v in enumerate(res): axa.text(i, v + 0.035, f"{v:.3f}", ha="center", fontsize=7)
    axa.set_xticks(xx); axa.set_xticklabels(nm, fontsize=7); axa.set_ylim(0, 0.62)
    axa.set_ylabel("Residue AUPRC"); axa.set_title("Ranking residues", loc="left")
    axb.bar(xx, pk, color=cols, width=0.6)
    for i, v in enumerate(pk): axb.text(i, v + 3.5, f"{v:.1f}%", ha="center", fontsize=7)
    axb.set_xticks(xx); axb.set_xticklabels(nm, fontsize=7); axb.set_ylim(0, 75)
    axb.set_ylabel("Pocket Top-1 success (DCA ≤ 4 Å)"); axb.set_title("Locating pockets", loc="left")
    fig.suptitle("The ranking inverts between the two tasks — permissive labels, 246 proteins",
                 x=0.01, ha="left", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); save(fig, "fig_metric_inversion_v2")


if __name__ == "__main__":
    fig_strict_comparison()
    fig_leakage_cost()
    fig_seed_replicates()
    fig_metric_inversion()
