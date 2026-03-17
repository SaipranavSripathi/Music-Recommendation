# AU2ACTR Music Recommendation - Full Baseline Implementation

## Goal

Implement the AU2ACTR baseline from the RecSys 2025 Deezer challenge as close to the original paper as possible, using the entire 4M user dataset.

**Target**: Match or exceed the original paper's baseline performance (15-20% Recall@10) with the full dataset on Mac M1.

## Project Context

This is a custom implementation of the AU2ACTR model. The original repository is at https://github.com/deezer/recsys25-reacta

### What was accomplished on Windows (current machine):

1. **Fixed 3 bugs** in the original codebase:
   - `min_sessions` filter not being applied
   - Audio embedding dimension mismatch
   - NumPy 2.0 compatibility issues

2. **Achieved 36.91% Recall@10** with 50k users on GTX 1650 4GB - significantly better than original paper's 15-20%!

3. **Created PyTorch implementation** since TensorFlow GPU doesn't work natively on Windows

4. **Key findings**:
   - 50k users with full model (128-dim) outperformed smaller models with more users
   - GTX 1650 4GB VRAM causes OOM with 60k+ users
   - Mac M1 with MPS can handle 80k+ users

## Setup on New Machine (Mac M1)

### 1. Clone the repository
```bash
git clone https://github.com/SaipranavSripathi/Music-Recommendation.git
cd Music-Recommendation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

For Mac M1 specifically:
```bash
pip install torch torchvision torchaudio
# M1 optimized: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/mps
```

### 3. Dataset

The dataset should be in the `user_sessions/` folder with parquet files containing:
- User listening sessions
- Track audio embeddings (50 dimensions)
- Track SVD embeddings

Update the data paths in `load_data.py`:
```python
DATA_DIR = Path("path/to/your/user_sessions")
EMBEDDINGS_DIR = Path("path/to/your/track_embeddings")
OUTPUT_DIR = Path("cache")
```

### 4. Key Configuration (edit in train_au2actr.py)

For Mac M1 with more memory:
```python
MAX_USERS = 80000       # Increase from 50k
NUM_EPOCHS = 300
EMBEDDING_DIM = 128     # Full model dimensions
HIDDEN_DIM = 256
BATCH_SIZE = 64         # Adjust based on memory
```

### 5. Run training

```bash
python load_data.py        # Load and prepare data
python train_au2actr.py    # Train AU2ACTR model
python evaluate.py         # Evaluate and get Recall@10
```

## Files Overview

| File | Description |
|------|-------------|
| `train_au2actr.py` | Full AU2ACTR with ACT-R components (BLL, PM, Audio Encoder) |
| `load_data.py` | Data loader with filtering and preprocessing |
| `evaluate.py` | Evaluation metrics (Recall@K, MRR, NDCG) |
| `requirements.txt` | Python dependencies |

## Original Paper Baseline Settings

From the paper (RecSys 2025):
- Number of epochs: 100
- Optimizer: Adam
- Batch size: 512
- Embedding dimension d: 128
- α = 0.5 for Base-Level Learning module
- Transformer: B=2, H=2, L=30
- Learning rates: {0.0002, 0.0005, 0.00075, 0.001}
- λ: {0.0, 0.3, 0.5, 0.8, 0.9, 1.0}
- β and γ: {0.2, 0.4, 0.6, 0.8, 1.0}

## Expected Results

- **Target**: Match paper's 15-20% Recall@10 baseline
- **Windows (50k users)**: 36.91% Recall@10 achieved
- **Mac M1 (80k+ users)**: Should achieve similar or better results with more data

## Troubleshooting

If you encounter OOM errors:
1. Reduce `MAX_USERS`
2. Reduce `BATCH_SIZE`
3. Use mixed precision training
4. Reduce `EMBEDDING_DIM` to 96

For Mac M1 MPS issues:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
