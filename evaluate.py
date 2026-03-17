"""
Evaluate AU2ACTR PyTorch Model
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
DATA_PATH = "D:/CS274/deezer-recsys25/deezer-recsys25"
CACHE_DIR = Path("cache/deezer/min5sess")

with open(CACHE_DIR / "user_sessions.pkl", "rb") as f:
    user_sessions = pickle.load(f)

print(f"Users: {len(user_sessions)}")

# Get all tracks
all_tracks = set()
for sessions in user_sessions.values():
    for session in sessions:
        all_tracks.update(session['track_ids'])

track_list = sorted(all_tracks)
track2id = {t: i+1 for i, t in enumerate(track_list)}
num_tracks = len(track_list) + 1
print(f"Tracks: {num_tracks}")

# Load embeddings
emb_df = pd.read_parquet(Path(DATA_PATH) / "track_embeddings_small.parquet")
embedding_dim = 128
embedding_matrix = np.zeros((num_tracks, embedding_dim), dtype=np.float32)

for _, row in emb_df.iterrows():
    track_id = row['track_id']
    if track_id in track2id:
        svd_emb = row['svd']
        if isinstance(svd_emb, dict):
            if 'list' in svd_emb:
                emb_list = svd_emb['list']
                emb_values = [x['item'] for x in emb_list[:embedding_dim]]
            else:
                emb_values = list(svd_emb.values())[:embedding_dim]
        else:
            emb_values = svd_emb[:embedding_dim] if len(svd_emb) >= embedding_dim else svd_emb
        
        if len(emb_values) < embedding_dim:
            emb_values = list(emb_values) + [0.0] * (embedding_dim - len(emb_values))
        else:
            emb_values = emb_values[:embedding_dim]
        embedding_matrix[track2id[track_id]] = np.array(emb_values, dtype=np.float32)


class AU2ACTRModel(nn.Module):
    def __init__(self, num_tracks, embedding_dim=128, hidden_dim=256, num_heads=2, num_blocks=2, dropout=0.1):
        super().__init__()
        
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim, padding_idx=0)
        self.track_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        self.pos_embedding = nn.Embedding(100, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, enable_nested_tensor=False)
        
        self.output = nn.Linear(embedding_dim, num_tracks)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        embeds = self.track_embedding(x)
        embeds += self.pos_embedding(torch.arange(seq_len, device=x.device)).unsqueeze(0)
        
        if mask is not None:
            output = self.transformer(embeds, src_key_padding_mask=mask)
        else:
            output = self.transformer(embeds)
        
        last_output = output[:, -1, :]
        
        return self.output(last_output)


class SessionDataset(Dataset):
    def __init__(self, user_sessions, track2id, seq_len=10, tracks_per_session=5, min_sessions=5):
        self.samples = []
        self.track2id = track2id
        self.seq_len = seq_len
        self.tracks_per_session = tracks_per_session
        
        for user_id, sessions in user_sessions.items():
            if len(sessions) < min_sessions + 1:
                continue
            
            for i in range(len(sessions) - seq_len):
                input_tracks = []
                for sess in sessions[i:i+seq_len]:
                    tracks = [self.track2id.get(t, 0) for t in sess['track_ids'][:self.tracks_per_session]]
                    input_tracks.append(tracks)
                
                target_session = sessions[i+seq_len]
                target_tracks = [self.track2id.get(t, 0) for t in target_session['track_ids'][:5]]
                
                flat_input = []
                for sess_tracks in input_tracks:
                    flat_input.extend(sess_tracks)
                
                flat_len = seq_len * self.tracks_per_session
                if len(flat_input) > flat_len:
                    flat_input = flat_input[:flat_len]
                else:
                    flat_input = flat_input + [0] * (flat_len - len(flat_input))
                
                self.samples.append((flat_input, target_tracks))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_tracks, target_tracks = self.samples[idx]
        return (
            torch.tensor(input_tracks, dtype=torch.long),
            torch.tensor(target_tracks[0], dtype=torch.long)
        )


# Create dataset
print("\nCreating dataset...")
dataset = SessionDataset(user_sessions, track2id, seq_len=10, tracks_per_session=5)
print(f"Total samples: {len(dataset)}")

# Split
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Train: {train_size}, Val: {val_size}")

val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)

# Load model
model = AU2ACTRModel(
    num_tracks=num_tracks,
    embedding_dim=128,
    hidden_dim=256,
    num_heads=2,
    num_blocks=2,
    dropout=0.1
).to(device)

model.load_state_dict(torch.load("au2actr_pytorch_best.pt"))
model.eval()
print("Model loaded!")

# Evaluate
print("\nEvaluating...")
correct = 0
total = 0
ndcg_scores = []

k = 10
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Top-k predictions
        _, top_preds = outputs.topk(k, dim=1)
        targets = targets.to(device)
        
        for i in range(len(targets)):
            if targets[i].item() in top_preds[i]:
                correct += 1
            total += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(val_loader)}")

accuracy = correct / total if total > 0 else 0
print(f"\n{'='*50}")
print(f"Top-{k} Accuracy: {accuracy:.4f}")
print(f"Correct: {correct}, Total: {total}")
