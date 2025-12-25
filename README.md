# GGNN 2025: Protein-Ligand Binding Site Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Geometric Graph Neural Network for predicting protein-ligand binding sites at the residue level. Achieves **state-of-the-art performance** with only **276K parameters** - 36x smaller than pLM-based methods.

## Key Results

| Benchmark | Type | AUC | Top-1 | Status |
|-----------|------|-----|-------|--------|
| Combined Test | In-distribution | **0.949** | **68.8%** | SOTA |
| scPDB | Druggable sites | **0.941** | **68.7%** | SOTA |
| Binding MOAD | High-quality | **0.980** | **100%** | SOTA |
| CryptoBench | Cryptic sites | **0.855** | 50.0% | Competitive |
| DUD-E | Drug targets | **0.865** | 56.0% | Strong |
| SC6K | Surface cavities | **0.921** | 63.5% | Strong |

### Comparison with Published Methods

| Method | Year | AUC | Top-1 | Params |
|--------|------|-----|-------|--------|
| **GGNN 2025 (Ours)** | **2025** | **0.949** | **68.8%** | **276K** |
| GPSite | 2024 | 0.91 | - | 10M+ |
| PGpocket | 2024 | 0.90 | 42.0% | ~5M |
| GraphBind | 2023 | 0.89 | - | ~2M |
| DeepSurf | 2021 | 0.88 | 58.0% | ~8M |
| P2Rank | 2018 | 0.82 | 63.0% | N/A |

---

## Installation

```bash
# Clone repository
git clone https://github.com/TCGxBill/GGNN2025.git
cd GGNN2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- BioPython 1.81+
- NumPy, SciPy, Pandas
- scikit-learn 1.3+

---

## Reproducing Results

### Step 1: Download Data

Download the Holo4K dataset (or use your own PDB files):

```bash
# Create data directories
mkdir -p data/raw/holo4k

# Place PDB files in data/raw/holo4k/
# Each PDB should have binding site annotations
```

### Step 2: Preprocess Data

```bash
# Preprocess combined dataset with train/val/test split
python scripts/preprocess_all.py \
    --dataset combined \
    --input data/raw \
    --output data/processed/combined \
    --split \
    --train_ratio 0.8 \
    --val_ratio 0.1

# Output structure:
# data/processed/combined/
#   ├── train/         (3449 proteins)
#   ├── val/           (431 proteins)
#   ├── test/          (432 proteins)
#   └── metadata.json
```

### Step 3: Train Model

```bash
# Train with optimized configuration
python scripts/train.py --config config_optimized.yaml

# Training parameters:
# - Batch size: 32
# - Learning rate: 0.0008
# - Loss: Combined (30% BCE + 70% Dice)
# - Epochs: 40 (early stopping at ~32)
# - Device: CPU or MPS (Apple Silicon)
```

Training takes approximately 4 hours on Apple M2.

### Step 4: Evaluate Model

```bash
# Run comprehensive evaluation
python experiments/comprehensive_eval.py

# Results saved to results_optimized/
```

---

## Quick Inference

```python
import torch
from src.models.gcn_geometric import GeometricGNN
from src.data.preprocessor import ProteinPreprocessor
from src.data.graph_builder import ProteinGraphBuilder

# Load pre-trained model
model = GeometricGNN.from_config("config_optimized.yaml")
checkpoint = torch.load("checkpoints_optimized/best_model.pth", 
                        map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process a protein
preprocessor = ProteinPreprocessor({'distance_threshold': 8.0, 'k_neighbors': 15})
graph_builder = ProteinGraphBuilder({'distance_threshold': 8.0, 'k_neighbors': 15})

data = preprocessor.process_pdb("your_protein.pdb")
graph = graph_builder.build_graph(
    data['node_features'], 
    data['coordinates'], 
    data['labels']
)

# Predict binding sites
with torch.no_grad():
    predictions = torch.sigmoid(model(graph))
    binding_residues = (predictions > 0.85).squeeze().numpy()

print(f"Predicted binding residues: {binding_residues.sum()}")
```

---

## Project Structure

```
GGNN2025/
├── src/
│   ├── models/gcn_geometric.py       # GATv2 model architecture
│   ├── data/
│   │   ├── preprocessor.py           # PDB parsing, feature extraction
│   │   └── graph_builder.py          # Graph construction
│   ├── training/trainer.py           # Training loop, Combined Loss
│   └── evaluation/metrics.py         # AUC, MCC, F1 metrics
├── scripts/
│   ├── train.py                      # Main training script
│   ├── preprocess_all.py             # Data preprocessing
│   ├── visualize_results.py          # Generate figures
│   └── download_benchmark_datasets.py
├── experiments/
│   ├── comprehensive_eval.py         # Multi-benchmark evaluation
│   ├── cross_dataset_eval.py         # Cross-dataset testing
│   ├── pocket_level_eval.py          # Pocket success rates
│   └── challenging_eval.py           # Cryptic/difficult sites
├── analysis/
│   ├── amino_acid_enrichment.py      # AA composition analysis
│   └── case_study_validation.py      # Biological validation
├── checkpoints_optimized/
│   └── best_model.pth                # Pre-trained weights (3.4MB)
├── results_optimized/
│   ├── figures/                      # Publication figures (9 PNG)
│   └── *.json                        # Evaluation results
├── tests/test_full_pipeline.py
├── config_optimized.yaml             # Training configuration
├── requirements.txt                  # Dependencies
├── paper/
│   ├── GGNN2025.tex                  # Paper manuscript
│   └── GGNN2025.pdf                  # Compiled PDF
├── LICENSE                           # MIT License
└── README.md
```

---

## Model Architecture

**GGNN 2025** uses GATv2 (Graph Attention Network v2) with:

| Component | Configuration |
|-----------|---------------|
| Layers | 3 GATv2 layers |
| Hidden dims | 256 → 128 → 64 |
| Attention heads | 4 |
| Edge features | 20-dim (distance + direction) |
| Node features | 29-dim |
| Dropout | 0.15 |
| Parameters | 276,097 |

### Combined Loss Function

```python
L = 0.3 * BCE_weighted + 0.7 * Dice
# BCE pos_weight = 12.0 (for class imbalance 1:31)
```

---

## Citation

```bibtex
@article{nguyen2025ggnn,
  title={GGNN 2025: A Lightweight Geometric Graph Neural Network 
         for Protein-Ligand Binding Site Prediction},
  author={Nguyen, Vu Trong Nhan},
  journal={bioRxiv},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Holo4K dataset providers
- PyTorch Geometric team
- AI assistants: Claude (Anthropic), Grok (xAI), Antigravity (Google DeepMind)
- Hardware: Apple MacBook Air M2

---

*Last updated: December 25, 2025*
