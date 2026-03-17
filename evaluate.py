"""
Evaluation script for AU2ACTR Model

Computes Recall@K, MRR, and NDCG metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pickle
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CACHE_DIR = Path("cache")
MODEL_PATH = CACHE_DIR / "au2actr_best.pt"


def load_model_and_data():
    """Load trained model and test data"""
    cache_file = CACHE_DIR / f"data_50k.pkl"
    
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    
    from train_au2actr import AU2ACTRModel, SessionDataset, collate_fn
    
    model = AU2ACTRModel(
        num_tracks=data['num_tracks'],
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_blocks=4,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    user_sessions = data['user_sessions']
    track2id = data['track2id']
    
    dataset = SessionDataset(
        user_sessions, track2id,
        seq_len=10,
        tracks_per_session=5,
        min_sessions=5
    )
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(
        val_dataset, batch_size=64,
        num_workers=0, collate_fn=collate_fn
    )
    
    return model, val_loader, data['num_tracks']


def compute_recall_at_k(predictions, targets, k=10):
    """Compute Recall@K"""
    _, top_k_preds = predictions.topk(k, dim=1)
    targets = targets.unsqueeze(1).expand_as(top_k_preds)
    hits = (top_k_preds == targets).any(dim=1).sum().item()
    return hits / len(targets)


def compute_mrr(predictions, targets):
    """Compute Mean Reciprocal Rank"""
    sorted_preds = predictions.argsort(dim=1, descending=True)
    targets = targets.unsqueeze(1).expand_as(sorted_preds)
    ranks = (sorted_preds == targets).nonzero(as_tuple=True)[1] + 1
    return (1.0 / ranks).mean().item()


def compute_ndcg(predictions, targets, k=10):
    """Compute NDCG@K"""
    _, top_k_preds = predictions.topk(k, dim=1)
    targets = targets.unsqueeze(1).expand_as(top_k_preds)
    
    dcg = ((top_k_preds == targets).float() / torch.log2(torch.arange(2, k+2, device=predictions.device))).sum(dim=1)
    
    ideal = torch.ones_like(targets)
    idcg = ((ideal == targets).float() / torch.log2(torch.arange(2, k+2, device=predictions.device))).sum(dim=1)
    
    ndcg = dcg / (idcg + 1e-8)
    return ndcg.mean().item()


def evaluate():
    """Run evaluation"""
    print("Loading model and data...")
    model, val_loader, num_tracks = load_model_and_data()
    
    print(f"Evaluating on {len(val_loader.dataset)} samples...")
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets, contexts in val_loader:
            inputs = inputs.to(device)
            contexts = contexts.to(device)
            mask = (inputs == 0)
            
            outputs = model(inputs, mask, contexts)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets)
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    recall_5 = compute_recall_at_k(predictions, targets, k=5)
    recall_10 = compute_recall_at_k(predictions, targets, k=10)
    recall_20 = compute_recall_at_k(predictions, targets, k=20)
    
    mrr = compute_mrr(predictions, targets)
    
    ndcg_10 = compute_ndcg(predictions, targets, k=10)
    
    print(f"Recall@5:  {recall_5*100:.2f}%")
    print(f"Recall@10: {recall_10*100:.2f}%")
    print(f"Recall@20: {recall_20*100:.2f}%")
    print(f"MRR:       {mrr*100:.2f}%")
    print(f"NDCG@10:   {ndcg_10*100:.2f}%")
    print("="*50)
    
    results = {
        'recall@5': recall_5,
        'recall@10': recall_10,
        'recall@20': recall_20,
        'mrr': mrr,
        'ndcg@10': ndcg_10
    }
    
    with open(CACHE_DIR / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    return results


if __name__ == "__main__":
    evaluate()
