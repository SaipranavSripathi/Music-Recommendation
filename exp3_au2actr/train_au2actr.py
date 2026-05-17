"""
AU2ACTR Complete PyTorch Implementation
With ACT-R cognitive model components (BLL, PM, Spread) and Audio Encoder
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
import math
import gc

print(f"PyTorch: {torch.__version__}")

# Check for MPS (Mac) or CUDA (NVIDIA)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using: Apple MPS (M1/M2/M3)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using: NVIDIA GPU")
else:
    device = torch.device("cpu")
    print("Using: CPU")

# ============= CONFIGURATION =============
DATA_PATH = "D:/CS274/deezer-recsys25/deezer-recsys25"  # Change this
CACHE_DIR = Path("cache/deezer/min5sess")
MAX_USERS = 80000  # Increase on Mac M1!
NUM_EPOCHS = 300
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
SEQ_LEN = 10
TRACKS_PER_SESSION = 5
LEARNING_RATE = 0.001

# ============= DATA LOADING =============

print("\n[1/5] Loading user sessions...")
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

# Load SVD embeddings
print("[2/5] Loading SVD embeddings...")
emb_df = pd.read_parquet(Path(DATA_PATH) / "track_embeddings_small.parquet")
embedding_matrix = np.random.randn(num_tracks, EMBEDDING_DIM).astype(np.float32) * 0.01

for _, row in emb_df.iterrows():
    track_id = row['track_id']
    if track_id in track2id:
        svd_emb = row['svd']
        if isinstance(svd_emb, dict) and 'list' in svd_emb:
            emb_values = [x['item'] for x in svd_emb['list'][:EMBEDDING_DIM]]
        else:
            continue
        if len(emb_values) < EMBEDDING_DIM:
            emb_values = emb_values + [0.0] * (EMBEDDING_DIM - len(emb_values))
        embedding_matrix[track2id[track_id]] = np.array(emb_values[:EMBEDDING_DIM], dtype=np.float32)

# Load Audio embeddings
print("[3/5] Loading audio embeddings...")
audio_matrix = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01

for _, row in emb_df.iterrows():
    track_id = row['track_id']
    if track_id in track2id:
        audio_emb = row['audio']
        if isinstance(audio_emb, dict) and 'list' in audio_emb:
            emb_values = [x['item'] for x in audio_emb['list'][:128]]
        else:
            continue
        if len(emb_values) < 128:
            emb_values = emb_values + [0.0] * (128 - len(emb_values))
        audio_matrix[track2id[track_id]] = np.array(emb_values[:128], dtype=np.float32)

print(f"SVD matrix: {embedding_matrix.shape}, Audio matrix: {audio_matrix.shape}")


# ============= MODEL COMPONENTS =============

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class AudioEncoder(nn.Module):
    """Audio encoder for predicting new tracks"""
    def __init__(self, audio_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, audio_emb):
        return self.net(audio_emb)


class ACTRCognitive(nn.Module):
    """
    ACT-R Cognitive Model Components:
    - BLL: Base-Level Learning (temporal decay)
    - PM: Priming Memory (familiarity)
    - Spread: Association spreading
    """
    def __init__(self, embedding_dim, use_bll=True, use_pm=True, use_spread=False):
        super().__init__()
        self.use_bll = use_bll
        self.use_pm = use_pm
        self.use_spread = use_spread
        
        # PM: Priming memory - learns familiarity weights
        if use_pm:
            self.pm_weights = nn.Parameter(torch.ones(embedding_dim) * 0.1)
    
    def compute_bll(self, time_stamps, decay=0.5):
        """
        Base-Level Learning: items learned based on recency
        More recent = higher activation
        """
        if not self.use_bll or len(time_stamps) < 2:
            return None
        
        # Simple recency weights (exponential decay)
        times = torch.tensor(time_stamps, dtype=torch.float32)
        times = times - times.min()
        if times.max() > 0:
            weights = torch.exp(-decay * times / times.max())
        else:
            weights = torch.ones_like(times)
        return weights
    
    def compute_pm(self, item_embeddings):
        """
        Priming Memory: familiarity boost for seen items
        """
        if not self.use_pm:
            return None
        
        # Simple attention-like familiarity
        familiarity = torch.mean(item_embeddings * self.pm_weights, dim=-1)
        return torch.sigmoid(familiarity)
    
    def forward(self, embeddings, time_context=None):
        outputs = []
        
        # BLL
        if self.use_bll and time_context is not None:
            bll_weights = self.compute_bll(time_context)
            if bll_weights is not None:
                bll_weights = bll_weights.to(embeddings.device)
                weighted = embeddings * bll_weights.view(-1, 1)
                outputs.append(weighted)
        
        # PM
        if self.use_pm:
            pm_boost = self.compute_pm(embeddings)
            if pm_boost is not None:
                pm_boost = pm_boost.to(embeddings.device)
                boosted = embeddings * pm_boost.view(-1, 1)
                outputs.append(boosted)
        
        if outputs:
            return torch.stack(outputs).mean(dim=0)
        return embeddings


class AU2ACTRModel(nn.Module):
    """
    Complete AU2ACTR Model with:
    - Transformer Encoder (sequence modeling)
    - ACT-R Cognitive Model (BLL, PM)
    - Audio Encoder (new track prediction)
    """
    def __init__(self, num_tracks, embedding_dim=128, hidden_dim=256, 
                 num_heads=2, num_blocks=2, dropout=0.1,
                 use_bll=True, use_pm=True, use_audio=True):
        super().__init__()
        
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        self.use_audio = use_audio
        
        # Track embeddings (SVD)
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim, padding_idx=0)
        self.track_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_blocks, 
            enable_nested_tensor=False
        )
        
        # ACT-R Cognitive Model
        self.actr = ACTRCognitive(embedding_dim, use_bll=use_bll, use_pm=use_pm)
        
        # Audio Encoder (for new track prediction)
        if use_audio:
            self.audio_encoder = AudioEncoder(audio_dim=128, hidden_dim=256, output_dim=embedding_dim)
            self.audio_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(audio_matrix), freeze=False
            )
        
        # Output layers
        self.output = nn.Linear(embedding_dim, num_tracks)
        
        # Audio prediction head
        if use_audio:
            self.audio_predict = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_tracks)
            )
    
    def forward(self, x, return_embeddings=False):
        batch_size, seq_len = x.shape
        
        # Embeddings
        embeds = self.track_embedding(x)
        embeds = self.pos_encoding(embeds)
        
        # Transformer
        output = self.transformer(embeds)
        
        # Use last hidden state
        last_output = output[:, -1, :]
        
        # Apply ACT-R
        actr_output = self.actr(last_output)
        
        # Final output
        logits = self.output(actr_output)
        
        if return_embeddings:
            return logits, last_output, actr_output
        
        return logits
    
    def predict_with_audio(self, x):
        """Predict with audio encoder for new tracks"""
        logits, hidden, actr_hidden = self.forward(x, return_embeddings=True)
        
        # Get audio embeddings for all tracks
        audio_all = self.audio_encoder(self.audio_embeddings.weight)
        
        # Compute similarity
        scores = torch.matmul(actr_hidden, audio_all.T)
        
        # Combine with item predictions
        final_scores = logits + 0.3 * scores
        
        return final_scores


# ============= DATASET =============

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
                # Input sessions
                input_tracks = []
                for sess in sessions[i:i+seq_len]:
                    tracks = [self.track2id.get(t, 0) for t in sess['track_ids'][:tracks_per_session]]
                    input_tracks.append(tracks)
                
                # Target
                target_session = sessions[i+seq_len]
                target_tracks = [self.track2id.get(t, 0) for t in target_session['track_ids'][:5]]
                
                # Flatten
                flat_input = []
                for sess_tracks in input_tracks:
                    flat_input.extend(sess_tracks)
                
                flat_len = seq_len * tracks_per_session
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


# ============= TRAINING =============

print("\n[4/5] Creating dataset...")
dataset = SessionDataset(user_sessions, track2id, SEQ_LEN, TRACKS_PER_SESSION)
print(f"Total samples: {len(dataset)}")

# Clear memory
del user_sessions
gc.collect()

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

# Create model
model = AU2ACTRModel(
    num_tracks=num_tracks,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=2,
    num_blocks=2,
    dropout=0.1,
    use_bll=True,
    use_pm=True,
    use_audio=True
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

print("\n[5/5] Training...")
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 500 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    scheduler.step()
    
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
        torch.save(model.state_dict(), "au2actr_best.pt")
        print(f"  New best! Accuracy: {accuracy:.4f}")
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {accuracy:.4f}, Best: {best_val_acc:.4f}")

print(f"\n{'='*50}")
print(f"Training complete!")
print(f"Best validation accuracy: {best_val_acc:.4f}")
