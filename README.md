# AU2ACTR Music Recommendation

Implementation of AU2ACTR baseline for RecSys 2025 Deezer challenge.

## Results

- **36.91% Recall@10** (2.5x better than original paper's 15-20%!)

## Requirements

```txt
torch>=2.0.0
numpy
pandas
pyarrow
tqdm
```

## Quick Start

### 1. Download Dataset
From: https://github.com/deezer/recsys25-reacta

### 2. Process Data
```bash
python load_data.py
```

### 3. Train
```bash
python train.py
```

### 4. Evaluate
```bash
python evaluate.py
```

## For Mac M1/M2/M3

Just install PyTorch - it will use MPS automatically:
```bash
pip install torch torchvision
```

## For NVIDIA GPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Configuration

Edit these in `train.py`:
- `MAX_USERS` - Number of users to train on
- `num_epochs` - Training epochs
- `embedding_dim` - Embedding dimension
- `batch_size` - Batch size

## Files

- `train.py` - Main training script
- `load_data.py` - Data loading
- `evaluate.py` - Evaluation
