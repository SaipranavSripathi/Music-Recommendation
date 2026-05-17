"""
AU2ACTR - Efficient Full Dataset Implementation
With optimized data loading using multiprocessing
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import datetime

print(f"PyTorch: {torch.__version__}")

# Device
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
DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")
MIN_SESSIONS = 50  # Practical threshold
MAX_USERS = 100000  # Limit users for practical training
SEQ_LEN = 30
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
BATCH_SIZE = 512
NUM_EPOCHS = 50  # Reduced for faster iteration
LEARNING_RATE = 0.001
NUM_BLOCKS = 2
NUM_HEADS = 2

# ACT-R parameters
ALPHA = 0.5
LAMBDA_AUENC = 0.6
FLATTEN_ACTR = 0.5

def process_session_file(args):
    """Process a single parquet file - for parallel execution"""
    fname, target_set, data_dir = args
    try:
        df = pd.read_parquet(data_dir / fname)
        df = df[df['user_id'].isin(target_set)]
        
        if len(df) == 0:
            return {}
        
        df = df.sort_values('ts')
        user_sessions = defaultdict(list)
        
        for (user_id, session_id), group in df.groupby(['user_id', 'session_id']):
            tracks = sorted(set(group['track_id'].tolist()))
            ts = group['ts'].iloc[0]
            dt = datetime.datetime.fromtimestamp(ts)
            
            session_data = {
                'session_id': session_id,
                'context': {
                    'ts': ts,
                    'day_of_week': dt.weekday(),
                    'hour_of_day': dt.hour
                },
                'track_ids': tracks
            }
            user_sessions[user_id].append(session_data)
        
        return dict(user_sessions)
    except Exception as e:
        print(f"Error: {fname} - {e}")
        return {}


print("\n[1/7] Sampling users with enough sessions...")
start_time = time.time()

DATA_DIR = DATA_PATH / "user_sessions"
PARQUET_FILES = sorted(os.listdir(DATA_DIR))

# First pass: sample users with enough sessions
# Process first N files to find eligible users
SAMPLE_FILES = 50  # Use first 50 files to sample users

print(f"Processing first {SAMPLE_FILES} files to find eligible users...")

user_counts = defaultdict(int)

for i, fname in enumerate(PARQUET_FILES[:SAMPLE_FILES]):
    if i % 10 == 0:
        print(f"  Scanning: {i}/{SAMPLE_FILES}")
    try:
        df = pd.read_parquet(DATA_DIR / fname, columns=['user_id', 'session_id'])
        for (user_id, session_id), _ in df.groupby(['user_id', 'session_id']):
            user_counts[user_id] += 1
    except Exception as e:
        pass

print(f"Users found: {len(user_counts)}")

# Filter users with MIN_SESSIONS+
eligible_users = {uid for uid, cnt in user_counts.items() if cnt >= MIN_SESSIONS}
print(f"Users with {MIN_SESSIONS}+ sessions: {len(eligible_users)}")

# Sample MAX_USERS
if len(eligible_users) > MAX_USERS:
    import random
    eligible_users = set(random.sample(list(eligible_users), MAX_USERS))
    print(f"Sampled to: {len(eligible_users)} users")

del user_counts
gc.collect()

# Second pass: load sessions for eligible users
print("\n[2/7] Loading user sessions...")
user_sessions = defaultdict(list)

for i, fname in enumerate(PARQUET_FILES):
    if i % 20 == 0:
        elapsed = time.time() - start_time
        print(f"  Loading: {i}/{len(PARQUET_FILES)} ({elapsed:.1f}s)")
    
    try:
        df = pd.read_parquet(DATA_DIR / fname)
        df = df[df['user_id'].isin(eligible_users)]
        
        if len(df) == 0:
            continue
        
        df = df.sort_values('ts')
        
        for (user_id, session_id), group in df.groupby(['user_id', 'session_id']):
            tracks = sorted(set(group['track_id'].tolist()))
            ts = group['ts'].iloc[0]
            dt = datetime.datetime.fromtimestamp(ts)
            
            session_data = {
                'session_id': session_id,
                'context': {'ts': ts, 'day_of_week': dt.weekday(), 'hour_of_day': dt.hour},
                'track_ids': tracks
            }
            user_sessions[user_id].append(session_data)
    except Exception as e:
        print(f"Error: {fname}: {e}")

# Sort sessions
for user_id in user_sessions:
    user_sessions[user_id] = sorted(user_sessions[user_id], key=lambda x: x['context']['ts'])

# Filter again
filtered_sessions = {uid: sess for uid, sess in user_sessions.items() if len(sess) >= MIN_SESSIONS}
print(f"Final users: {len(filtered_sessions)}")

del user_sessions
gc.collect()

# Build track vocabulary
print("\n[3/7] Building track vocabulary...")
all_tracks = set()
for sessions in filtered_sessions.values():
    for session in sessions:
        all_tracks.update(session['track_ids'])

track_list = sorted(all_tracks)
track2id = {t: i+1 for i, t in enumerate(track_list)}
num_tracks = len(track_list) + 1
print(f"Tracks: {num_tracks}")

del all_tracks
gc.collect()

# Save to cache
session_cache_path = CACHE_DIR / "deezer" / f"min{MIN_SESSIONS}sess"
os.makedirs(session_cache_path, exist_ok=True)
with open(session_cache_path / "user_sessions.pkl", "wb") as f:
    pickle.dump(dict(filtered_sessions), f)
print("Saved to cache")

# Load embeddings
print("\n[4/7] Loading track embeddings...")
emb_files = sorted((DATA_PATH / "track_embeddings").glob("*.parquet"))

svd_embeddings = {}
audio_embeddings = {}

for f in emb_files:
    print(f"  Loading {f.name}...")
    df = pd.read_parquet(f)
    for _, row in df.iterrows():
        track_id = row['track_id']
        if track_id not in track2id:
            continue
        
        svd_emb = row['svd']
        if isinstance(svd_emb, list):
            svd_embeddings[track_id] = np.array(svd_emb[:EMBEDDING_DIM], dtype=np.float32)
        
        audio_emb = row['audio']
        if isinstance(audio_emb, list):
            audio_embeddings[track_id] = np.array(audio_emb[:128], dtype=np.float32)

print(f"SVD: {len(svd_embeddings)}, Audio: {len(audio_embeddings)}")

# Create embedding matrices
embedding_matrix = np.random.randn(num_tracks, EMBEDDING_DIM).astype(np.float32) * 0.01
audio_matrix = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01

for track_id, tid in track2id.items():
    if track_id in svd_embeddings:
        embedding_matrix[tid] = svd_embeddings[track_id]
    if track_id in audio_embeddings:
        audio_matrix[tid] = audio_embeddings[track_id]

print(f"Matrices: SVD {embedding_matrix.shape}, Audio {audio_matrix.shape}")


# ============= MODEL =============

class PositionalEncoding(nn.Module):
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
    def __init__(self, embedding_dim, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.pm_weights = nn.Parameter(torch.ones(embedding_dim) * 0.1)
    
    def forward(self, embeddings):
        # PM - Priming Memory
        familiarity = torch.mean(embeddings * self.pm_weights, dim=-1, keepdim=True)
        pm_boost = torch.sigmoid(familiarity)
        return embeddings * pm_boost


class AU2ACTRModel(nn.Module):
    def __init__(self, num_tracks, embedding_dim=128, hidden_dim=256, num_heads=2, num_blocks=2):
        super().__init__()
        
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim, padding_idx=0)
        self.track_embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=200)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, enable_nested_tensor=False)
        
        self.actr = ACTRCognitive(embedding_dim, alpha=ALPHA)
        self.audio_encoder = AudioEncoder(128, 256, embedding_dim)
        self.audio_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(audio_matrix), freeze=False)
        
        self.output = nn.Linear(embedding_dim, num_tracks)
        self.audio_predict = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tracks)
        )
        
        with torch.no_grad():
            self.track_embedding.weight.copy_(torch.from_numpy(embedding_matrix))
            self.audio_embeddings.weight.copy_(torch.from_numpy(audio_matrix))
    
    def forward(self, x, return_embeddings=False):
        embeds = self.track_embedding(x)
        embeds = self.pos_encoding(embeds)
        output = self.transformer(embeds)
        last_output = output[:, -1, :]
        actr_output = self.actr(last_output)
        logits = self.output(actr_output)
        
        if return_embeddings:
            return logits, last_output, actr_output
        return logits
    
    def predict_with_audio(self, x):
        logits, hidden, actr_hidden = self.forward(x, return_embeddings=True)
        audio_all = self.audio_encoder(self.audio_embeddings.weight)
        scores = torch.matmul(actr_hidden, audio_all.T)
        return logits + LAMBDA_AUENC * scores


# ============= DATASET =============

class SessionDataset(Dataset):
    def __init__(self, user_sessions, track2id, seq_len=30, min_sessions=50):
        self.samples = []
        self.track2id = track2id
        self.seq_len = seq_len
        
        for user_id, sessions in user_sessions.items():
            if len(sessions) < min_sessions + 10:
                continue
            
            train_sessions = sessions[:-10]  # Last 10 for val/test
            
            for i in range(len(train_sessions) - seq_len):
                input_tracks = []
                for sess in train_sessions[i:i+seq_len]:
                    tracks = [self.track2id.get(t, 0) for t in sess['track_ids'][:5]]
                    input_tracks.append(tracks)
                
                target_session = train_sessions[i+seq_len]
                target_track = self.track2id.get(target_session['track_ids'][0], 0)
                
                if target_track == 0:
                    continue
                
                flat_input = []
                for sess_tracks in input_tracks:
                    flat_input.extend(sess_tracks)
                
                flat_len = seq_len * 5
                if len(flat_input) > flat_len:
                    flat_input = flat_input[:flat_len]
                else:
                    flat_input = flat_input + [0] * (flat_len - len(flat_input))
                
                self.samples.append((flat_input, target_track))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_tracks, target_track = self.samples[idx]
        return torch.tensor(input_tracks, dtype=torch.long), torch.tensor(target_track, dtype=torch.long)


# ============= TRAINING =============

print("\n[5/7] Creating dataset...")
dataset = SessionDataset(filtered_sessions, track2id, SEQ_LEN, MIN_SESSIONS)
print(f"Total samples: {len(dataset)}")

del filtered_sessions
gc.collect()

if len(dataset) == 0:
    print("ERROR: No training samples!")
    exit(1)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

print("\n[6/7] Creating model...")
model = AU2ACTRModel(num_tracks, EMBEDDING_DIM, HIDDEN_DIM, NUM_HEADS, NUM_BLOCKS).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

print("\n[7/7] Training...")
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model.predict_with_audio(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 200 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    scheduler.step()
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model.predict_with_audio(inputs)
            predictions = outputs.argmax(dim=1)
            targets = targets.to(device)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    accuracy = correct / total if total > 0 else 0
    
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), CACHE_DIR / "au2actr_best.pt")
        print(f"  New best! Acc: {accuracy:.4f}")
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {accuracy:.4f}, Best: {best_val_acc:.4f}")

print(f"\n{'='*50}")
print(f"Training complete! Best: {best_val_acc:.4f}")
print(f"Total time: {time.time() - start_time:.1f}s")
