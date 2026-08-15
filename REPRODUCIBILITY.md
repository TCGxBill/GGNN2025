# Reproducibility Guide for GGNN 2025

## Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores (Intel/AMD x64)
- **RAM**: 8 GB
- **Storage**: 2 GB for code + data

### Recommended (for training)
- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **GPU**: NVIDIA GPU with 4+ GB VRAM (optional, CPU training supported)

### Hardware Used in Paper
- **CPU**: Apple M1/M2 or Intel Core i7
- **RAM**: 8 GB
- **Training Device**: CPU (no GPU required)
- **Training Time**: ~2-4 hours on CPU

## Software Requirements

### Python Version
```
Python 3.10+ (tested on 3.10, 3.11, 3.12, 3.13)
```

### Core Dependencies (exact versions)
```
torch==2.4.0
torch-geometric==2.5.3
torch-scatter==2.1.2
torch-sparse==0.6.18
numpy==2.2.5
scipy==1.15.3
scikit-learn==1.6.1
pandas==2.2.3
biopython==1.86
pyyaml==6.0.2
tqdm==4.68.2
matplotlib==3.10.3
seaborn==0.13.3
```

### Installation
```bash
# Clone repository
git clone https://github.com/TCGxBill/GGNN2025.git
cd GGNN2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Random Seeds

All experiments use the following random seeds for reproducibility:

```python
RANDOM_SEED = 42

# Set in code:
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

## Data Preprocessing

### Dataset Preparation
```bash
# Download and preprocess datasets (automatic)
python scripts/download_data.py
python scripts/preprocess.py
```

### Expected Data Structure
```
data/
├── raw/
│   ├── scpdb/
│   ├── pdbbind/
│   └── ...
└── processed/
    ├── combined/
    │   ├── train/  (3449 .pt files)
    │   ├── val/    (431 .pt files)
    │   └── test/   (432 .pt files)
    ├── scpdb/      (150 .pt files)
    ├── coach420/   (419 .pt files)
    └── ...
```

## Training

### Command
```bash
python train.py --config config_optimized.yaml
```

### Expected Training Metrics
| Epoch | Train Loss | Val Loss | Val AUC |
|-------|------------|----------|---------|
| 1     | 0.45       | 0.42     | 0.85    |
| 10    | 0.25       | 0.28     | 0.91    |
| 50    | 0.18       | 0.22     | 0.94    |
| 100   | 0.15       | 0.20     | 0.949   |

### Early Stopping
- Patience: 20 epochs
- Min delta: 0.001
- Monitor: Validation AUC

## Evaluation

### Run Full Benchmark
```bash
python experiments/fresh_benchmark.py
```

### Expected Results
| Benchmark | AUC | MCC | Top-1 |
|-----------|-----|-----|-------|
| Combined Test | 0.949 | 0.617 | 68.8% |
| scPDB | 0.941 | 0.603 | 68.7% |
| COACH420 | 0.852 | 0.384 | 44.6% |
| MOAD (57) | 0.919 | 0.433 | 66.7% |
| CryptoBench (59) | 0.838 | 0.353 | 44.1% |

## Checkpoints

### Pre-trained Model
```
checkpoints_optimized/best_model.pth
```

### Loading Checkpoint
```python
import torch
from src.models.gcn_geometric import GeometricGNN

# Load config
with open('config_optimized.yaml') as f:
    config = yaml.safe_load(f)

# Create model and load weights
model = GeometricGNN(config['model'])
ckpt = torch.load('checkpoints_optimized/best_model.pth', 
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

## Inference

### Single Protein Prediction
```python
from src.data.preprocessor import ProteinPreprocessor
from src.data.graph_builder import ProteinGraphBuilder

# Preprocess PDB file
preprocessor = ProteinPreprocessor(config)
data = preprocessor.process_pdb('protein.pdb')

# Build graph
graph_builder = ProteinGraphBuilder(config['model'])
graph = graph_builder.build_graph(
    data['node_features'],
    data['coordinates'],
    data.get('labels')
)

# Predict
with torch.no_grad():
    output, _ = model(graph)
    probs = torch.sigmoid(output).numpy()
```

### Expected Inference Time
- **Per protein (CPU)**: 50-200 ms
- **Throughput**: 5-20 proteins/second

## Troubleshooting

### DSSP Not Found
```bash
# Install DSSP (optional, for secondary structure)
conda install -c salilab dssp
# or: brew install dssp (Mac)
```

### CUDA Out of Memory
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
```

### Import Errors
```bash
# Ensure you're in the project root
cd GGNN2025
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

## Contact

For reproducibility issues, please open a GitHub issue:
https://github.com/TCGxBill/GGNN2025/issues
