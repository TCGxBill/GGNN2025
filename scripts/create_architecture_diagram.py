#!/usr/bin/env python3
"""
GGNN 2025 Architecture Diagram - v2 Style (as shown in user's image)
With dropout fixed to 0.15
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 300

# Colors matching v2
BLUE_DARK = '#1a365d'
BLUE_MED = '#2c5282'
BLUE_LIGHT = '#4299e1'
ORANGE = '#ed8936'
GRAY = '#718096'
GRAY_LIGHT = '#e2e8f0'
WHITE = '#ffffff'
GREEN = '#48bb78'

def box(ax, x, y, w, h, text, color, text_color=WHITE, fontsize=9, subtext=None, subtext_size=7):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                          facecolor=color, edgecolor=BLUE_DARK, linewidth=1)
    ax.add_patch(rect)
    shadow = FancyBboxPatch((x+0.02, y-0.02), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                            facecolor='#00000010', edgecolor='none', zorder=0)
    ax.add_patch(shadow)
    if subtext:
        ax.text(x+w/2, y+h*0.65, text, ha='center', va='center', fontsize=fontsize, 
                color=text_color, fontweight='bold')
        ax.text(x+w/2, y+h*0.3, subtext, ha='center', va='center', fontsize=subtext_size, 
                color=text_color)
    else:
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize, 
                color=text_color, fontweight='bold')

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def main():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 7.6, 'GGNN 2025 Architecture', fontsize=14, ha='center', fontweight='bold', color=BLUE_DARK)
    
    # Section headers
    ax.text(1.5, 7.0, 'Input', fontsize=11, ha='center', fontweight='bold', color=BLUE_DARK)
    ax.text(6.0, 7.0, 'GGNN 2025 Model (283K params)', fontsize=11, ha='center', fontweight='bold', color=BLUE_DARK)
    ax.text(11.5, 7.0, 'Output', fontsize=11, ha='center', fontweight='bold', color=BLUE_DARK)
    
    # Separators
    ax.axvline(x=3.2, ymin=0.08, ymax=0.85, color=GRAY_LIGHT, linestyle='--', linewidth=1)
    ax.axvline(x=9.0, ymin=0.08, ymax=0.85, color=GRAY_LIGHT, linestyle='--', linewidth=1)
    
    # === INPUT ===
    box(ax, 0.3, 5.5, 2.4, 0.8, '3D Protein', GRAY_LIGHT, BLUE_DARK, 9, 'Structure', 7)
    box(ax, 0.3, 4.0, 2.4, 1.0, 'Graph', GREEN, WHITE, 9, 'Construction', 7)
    ax.text(1.5, 3.5, 'K-NN (k=15)\nDist. thresh 8Å', fontsize=7, ha='center', color=GRAY)
    box(ax, 0.3, 1.8, 2.4, 0.9, 'Node Features', GREEN, WHITE, 8, '(32-dim)', 7)
    ax.text(1.5, 1.3, 'AA type, DSSP, Hydrophobicity,\nSASA, HSE, Density', fontsize=6, ha='center', color=GRAY)
    box(ax, 0.3, 0.3, 2.4, 0.9, 'Edge Features', ORANGE, WHITE, 8, '(20-dim)', 7)
    ax.text(1.5, -0.2, 'Distance, Direction,\n16 distance bins', fontsize=6, ha='center', color=GRAY)
    
    # Input arrows
    arrow(ax, 1.5, 5.5, 1.5, 5.0)
    arrow(ax, 2.7, 4.5, 3.5, 4.5)
    arrow(ax, 2.7, 2.2, 3.5, 3.5)
    arrow(ax, 2.7, 0.75, 3.5, 2.5)
    
    # === MODEL ===
    # GATv2 Layer 1
    box(ax, 3.5, 5.2, 2.2, 1.2, 'GATv2 Layer 1', BLUE_MED, WHITE, 9, '32 → 256', 8)
    ax.text(5.9, 5.8, '4 heads', fontsize=7, ha='left', color=GRAY)
    
    # GATv2 Layer 2
    box(ax, 3.5, 3.2, 2.2, 1.2, 'GATv2 Layer 2', BLUE_MED, WHITE, 9, '256 → 128', 8)
    ax.text(5.9, 3.8, '4 heads', fontsize=7, ha='left', color=GRAY)
    
    # GATv2 Layer 3
    box(ax, 3.5, 1.2, 2.2, 1.2, 'GATv2 Layer 3', BLUE_MED, WHITE, 9, '128 → 64', 8)
    ax.text(5.9, 1.8, '4 heads', fontsize=7, ha='left', color=GRAY)
    
    # Dropout (FIXED: 0.15)
    ax.text(6.0, 4.7, 'Dropout\n(0.15)', fontsize=7, ha='center', color=GRAY)
    ax.text(6.0, 2.7, 'Dropout\n(0.15)', fontsize=7, ha='center', color=GRAY)
    
    # Residual connections
    ax.annotate('', xy=(5.7, 4.4), xytext=(5.7, 5.8),
               arrowprops=dict(arrowstyle='->', color=GREEN, lw=1, 
                              connectionstyle="arc3,rad=-0.3", linestyle='--'))
    ax.annotate('', xy=(5.7, 2.4), xytext=(5.7, 3.8),
               arrowprops=dict(arrowstyle='->', color=GREEN, lw=1,
                              connectionstyle="arc3,rad=-0.3", linestyle='--'))
    ax.text(6.3, 5.1, 'Residual', fontsize=6, color=GREEN, rotation=90)
    ax.text(6.3, 3.1, 'Residual', fontsize=6, color=GREEN, rotation=90)
    
    # Layer arrows
    arrow(ax, 4.6, 5.2, 4.6, 4.4)
    arrow(ax, 4.6, 3.2, 4.6, 2.4)
    
    # Edge arrows to layers
    arrow(ax, 3.0, 5.8, 3.5, 5.8, ORANGE, 1)
    arrow(ax, 3.0, 3.8, 3.5, 3.8, ORANGE, 1)
    arrow(ax, 3.0, 1.8, 3.5, 1.8, ORANGE, 1)
    
    # Combined Loss
    box(ax, 7.0, 5.0, 2.0, 1.3, 'Combined Loss', ORANGE, WHITE, 9, 'L = 0.3·BCE + 0.7·Dice', 7)
    arrow(ax, 5.7, 5.0, 7.0, 5.6)
    
    # Global-Context Readout
    box(ax, 7.0, 1.0, 2.0, 0.9, 'Global-Context', BLUE_LIGHT, WHITE, 8, 'Readout (32-dim)', 7)
    arrow(ax, 5.7, 1.2, 7.0, 1.45)
    
    # Classification Head
    box(ax, 7.0, 2.5, 2.0, 1.5, 'Classification', BLUE_DARK, WHITE, 9, 'Head\n96 → 1', 8)
    arrow(ax, 5.7, 1.8, 7.0, 3.0)
    arrow(ax, 8.0, 1.9, 8.0, 2.5)
    
    # === OUTPUT ===
    box(ax, 10.0, 2.5, 1.8, 1.0, 'Sigmoid', GRAY, WHITE, 9, 'Activation', 7)
    box(ax, 10.0, 0.8, 1.8, 1.2, 'Per-residue', GREEN, WHITE, 9, 'Binding\nProbability', 7)
    
    arrow(ax, 9.0, 3.25, 10.0, 3.0)
    arrow(ax, 10.9, 2.5, 10.9, 2.0)
    
    # Performance box
    ax.text(12.5, 3.5, 'Performance:', fontsize=9, ha='left', fontweight='bold', color=BLUE_DARK)
    ax.text(12.5, 3.0, 'AUC: 0.967', fontsize=8, ha='left', color=GRAY)
    ax.text(12.5, 2.6, 'MCC: 0.627', fontsize=8, ha='left', color=GRAY)
    ax.text(12.5, 2.2, 'Inference: 14ms', fontsize=8, ha='left', color=GRAY)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=BLUE_MED, label='GATv2 Layers'),
        mpatches.Patch(color=GREEN, label='Node Features / Residual'),
        mpatches.Patch(color=ORANGE, label='Edge Features / Loss'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=7, frameon=True, fancybox=True, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig('paper/figures/architecture_v2.png', dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    fig.savefig('paper/figures/architecture_v2.pdf', bbox_inches='tight', facecolor='white', pad_inches=0.1)
    print(" paper/figures/architecture_v2.png")
    print(" paper/figures/architecture_v2.pdf")
    plt.close()

if __name__ == "__main__":
    main()
