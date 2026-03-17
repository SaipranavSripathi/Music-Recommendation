"""
AU2ACTR PyTorch Implementation - Full Dataset with GPU
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration - Edit these paths as needed
DATA_PATH = "D:/CS274/deezer-recsys25/deezer-recsys25"  # Path to dataset
CACHE_DIR = Path("cache/deezer/min5sess")  # Cache directory

print("\nLoading user sessions...")
with open(CACHE_DIR / "user_sessions.pkl", "rb") as f:
    user_sessions = pickle.load(f)

print(f"Users: {len(user_sessions)}")

# Get all unique tracks
all_tracks = set()
for sessions in user_sessions.values():
    for session in sessions:
        all_tracks.update(session['track_ids'])

track_list = sorted(all_tracks)
track2id = {t: i+1 for i, t in enumerate(track_list)}
num_tracks = len(track_list) + 1
print(f"Tracks: {num_tracks}")

# Load embeddings from parquet
print("Loading embeddings...")
emb_df = pd.read_parquet(Path(DATA_PATH) / "track_embeddings_small.parquet")

# Create embedding matrix for SVD embeddings
embedding_dim = 128
embedding_matrix = np.random.randn(num_tracks, embedding_dim).astype(np.float32) * 0.01  # Random init for unknown tracks

for _, row in emb_df.iterrows():
    track_id = row['track_id']
    if track_id in track2id:
        svd_emb = row['svd']
        # Handle different formats
        if isinstance(svd_emb, dict):
            if 'list' in svd_emb:
                # Extract actual values from list of dicts
                emb_list = svd_emb['list']
                emb_values = [x['item'] for x in emb_list[:embedding_dim]]
            else:
                emb_values = list(svd_emb.values())[:embedding_dim]
        elif isinstance(svd_emb, str):
            svd_emb = eval(svd_emb)
            emb_values = list(svd_emb.values())[:embedding_dim] if isinstance(svd_emb, dict) else svd_emb[:embedding_dim]
        else:
            emb_values = svd_emb[:embedding_dim]
        
        # Ensure we have exactly embedding_dim values
        if len(emb_values) < embedding_dim:
            emb_values = list(emb_values) + [0.0] * (embedding_dim - len(emb_values))
        else:
            emb_values = emb_values[:embedding_dim]
        
        idx = track2id[track_id]
        if idx < num_tracks:
            embedding_matrix[idx] = np.array(emb_values, dtype=np.float32)

print(f"Embedding matrix shape: {embedding_matrix.shape}")


class AU2ACTRModel(nn.Module):
    def __init__(self, num_tracks, embedding_dim=128, hidden_dim=512, num_heads=4, num_blocks=4, dropout=0.1):
        super().__init__()
        
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        
        # Track embeddings
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim, padding_idx=0)
        self.track_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(100, embedding_dim)
        
        # Transformer encoder (matching original AU2ACTR architecture)
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
        
        # Output - predict next track
        self.output = nn.Linear(embedding_dim, num_tracks)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Embeddings
        embeds = self.track_embedding(x)
        embeds += self.pos_embedding(torch.arange(seq_len, device=x.device)).unsqueeze(0)
        
        # Transformer
        if mask is not None:
            output = self.transformer(embeds, src_key_padding_mask=mask)
        else:
            output = self.transformer(embeds)
        
        # Use last hidden state
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
            
            # Use sliding window
            for i in range(len(sessions) - seq_len):
                # Input: sequence of sessions
                input_tracks = []
                for sess in sessions[i:i+seq_len]:
                    tracks = [self.track2id.get(t, 0) for t in sess['track_ids'][:self.tracks_per_session]]
                    input_tracks.append(tracks)
                
                # Target: first track of next session
                target_session = sessions[i+seq_len]
                target_tracks = [self.track2id.get(t, 0) for t in target_session['track_ids'][:5]]
                
                # Flatten
                flat_input = []
                for sess_tracks in input_tracks:
                    flat_input.extend(sess_tracks)
                
                # Pad
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
            torch.tensor(target_tracks[0], dtype=torch.long)  # First track as main target
        )


# Create dataset
print("\nCreating dataset...")
dataset = SessionDataset(user_sessions, track2id, seq_len=10, tracks_per_session=5)
print(f"Total samples: {len(dataset)}")

# Split data
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"Train: {train_size}, Val: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=24, num_workers=0, pin_memory=True)

# Create model - smaller for 4GB GPU
model = AU2ACTRModel(
    num_tracks=num_tracks,
    embedding_dim=128,
    hidden_dim=256,
    num_heads=2,
    num_blocks=2,
    dropout=0.1
).to(device)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)
criterion = nn.CrossEntropyLoss()

print("\nTraining on GPU...")
num_epochs = 250
best_val_acc = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    scheduler.step()
    avg_loss = total_loss / num_batches
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            targets = targets.to(device)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total if total > 0 else 0
    
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), "au2actr_pytorch_best.pt")
        print(f"  New best model saved! Accuracy: {accuracy:.4f}")
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}, Best: {best_val_acc:.4f}")

print(f"\n{'='*50}")
print(f"Training complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Model saved to: au2actr_pytorch_best.pt")
