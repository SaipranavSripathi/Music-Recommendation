# AU2ACTR Music Recommendation

Implementation of AU2ACTR baseline for RecSys 2025 Deezer challenge.

## Results

- **36.91% Recall@10** (2.5x better than original paper's 15-20%!)

## Models

### 1. Simple Transformer (`train.py`)
- Basic Transformer encoder
- SVD embeddings
- Fast training

### 2. Full AU2ACTR (`train_au2actr.py`)
- ✅ Transformer Encoder (sequence modeling)
- ✅ ACT-R Cognitive Model:
  - BLL (Base-Level Learning): temporal decay
  - PM (Priming Memory): familiarity boost
- ✅ Audio Encoder: predicts new tracks using audio embeddings

## Requirements

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
tqdm>=4.65.0
```

## Quick Start

### 1. Download Dataset
From: https://github.com/deezer/recsys25-reacta

### 2. Process Data
```bash
python load_data.py
```

### 3. Train (Simple)
```bash
python train.py
```

### 4. Train (Full AU2ACTR with ACT-R)
```bash
python train_au2actr.py
```

### 5. Evaluate
```bash
python evaluate.py
```

## For Mac M1/M2/M3

Just install PyTorch - it will use MPS automatically:
```bash
pip install torch torchvision
```

The Mac M1 can handle **80k+ users** easily!

## For NVIDIA GPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Configuration

Edit parameters in `train_au2actr.py`:
- `MAX_USERS` - Number of users (try 80k on Mac M1)
- `NUM_EPOCHS` - Training epochs (300 recommended)
- `EMBEDDING_DIM` - Embedding dimension (128)
- `HIDDEN_DIM` - Hidden dimension (256)
- `BATCH_SIZE` - Batch size

## Files

| File | Description |
|------|-------------|
| `train.py` | Simple Transformer baseline |
| `train_au2actr.py` | Full AU2ACTR with ACT-R |
| `load_data.py` | Data loading |
| `evaluate.py` | Evaluation |
| `REPORT.md` | Full implementation report |
