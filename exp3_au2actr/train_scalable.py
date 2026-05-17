"""
AU2ACTR - Scalable Training
Uses 200k users (good for most systems, avoids OOM)
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
import math
import gc
import time
from collections import defaultdict, Counter

# ============= CONFIGURATION =============
DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Parameters - adjust these
MAX_USERS = 200000        # Use 200k users (good balance)
MIN_SESSIONS = 5         # Minimum sessions per user
SEQ_LEN = 10             # Sequence length
EMB_DIM = 128            # Embedding dimension  
HID_DIM = 256            # Hidden dimension
BATCH = 512              # Batch size
EPOCHS = 50              # Number of epochs
LR = 0.001               # Learning rate
LAMBDA_AU = 0.6          # Audio weight

print("="*60)
print("AU2ACTR - Scalable Training")
print("="*60)
print(f"Config: MAX_USERS={MAX_USERS}, MIN_SESSIONS={MIN_SESSIONS}")
print(f"         SEQ_LEN={SEQ_LEN}, EMB_DIM={EMB_DIM}, HID_DIM={HID_DIM}")
print(f"         BATCH={BATCH}, EPOCHS={EPOCHS}")
print("="*60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
print()

# ============= LOAD DATA =============
start_time = time.time()

# Check for cached sessions
session_cache = CACHE_DIR / f"sessions_{MAX_USERS}users_s{MIN_SESSIONS}.pkl"

if session_cache.exists():
    print(f"Loading cached sessions from {session_cache}")
    with open(session_cache, 'rb') as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} users")
else:
    print(f"Loading sessions for up to {MAX_USERS} users...")
    DATA_DIR = DATA_PATH / "user_sessions"
    FILES = sorted(os.listdir(DATA_DIR))
    print(f"Found {len(FILES)} parquet files")
    
    # First pass: count sessions per user
    print("Pass 1: Counting sessions per user...")
    user_counts = Counter()
    
    for i, f in enumerate(FILES):
        if i % 50 == 0:
            print(f"  Processing file {i}/{len(FILES)}...")
        df = pd.read_parquet(DATA_DIR / f, columns=['user_id', 'session_id'])
        for (u, s), _ in df.groupby(['user_id', 'session_id']):
            user_counts[u] += 1
    
    print(f"Total unique users: {len(user_counts)}")
    
    # Get users with enough sessions, sorted by session count
    eligible = [(u, c) for u, c in user_counts.items() if c >= MIN_SESSIONS]
    eligible.sort(key=lambda x: x[1], reverse=True)  # Sort by session count (desc)
    eligible = eligible[:MAX_USERS]  # Take top MAX_USERS
    eligible_users = {u for u, c in eligible}
    
    print(f"Selected top {len(eligible_users)} users with most sessions")
    
    del user_counts, eligible
    gc.collect()
    
    # Second pass: load sessions for selected users
    print("Pass 2: Loading sessions...")
    sessions = defaultdict(list)
    
    for i, f in enumerate(FILES):
        if i % 50 == 0:
            print(f"  Processing file {i}/{len(FILES)}...")
        
        df = pd.read_parquet(DATA_DIR / f)
        df = df[df['user_id'].isin(eligible_users)]
        
        if len(df) == 0:
            continue
        
        df = df.sort_values('ts')
        
        for (u, s), g in df.groupby(['user_id', 'session_id']):
            tracks = sorted(set(g['track_id'].tolist()))
            sessions[u].append({'tracks': tracks})
    
    # Sort sessions
    for u in sessions:
        sessions[u] = sorted(sessions[u], key=lambda x: x['tracks'][0] if x['tracks'] else 0)
    
    # Final filter
    sessions = {u: s for u, s in sessions.items() if len(s) >= MIN_SESSIONS}
    print(f"Final: {len(sessions)} users")
    
    # Save cache
    print(f"Saving to cache: {session_cache}")
    with open(session_cache, 'wb') as f:
        pickle.dump(dict(sessions), f)
    print(f"Cache saved ({session_cache.stat().st_size / 1024 / 1024:.1f} MB)")

print(f"Data loading time: {time.time() - start_time:.1f}s")

# ============= BUILD VOCABULARY =============
print("\nBuilding vocabulary...")
all_tracks = set()
for s in sessions.values():
    for sess in s:
        all_tracks.update(sess['tracks'])

track_list = sorted(all_tracks)
track2id = {t: i+1 for i, t in enumerate(track_list)}
num_tracks = len(track_list) + 1
print(f"Vocabulary: {num_tracks} tracks")

del all_tracks
gc.collect()

# ============= LOAD EMBEDDINGS =============
print("\nLoading pretrained embeddings...")
emb_files = list((DATA_PATH / "track_embeddings").iterdir())

svd_emb = {}
audio_emb = {}

for f in emb_files:
    print(f"  Loading {f.name}...")
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        tid = r['track_id']
        if tid not in track2id:
            continue
        
        svd = r['svd']
        if isinstance(svd, dict) and 'list' in svd:
            arr = np.array([x['item'] for x in svd['list'][:EMB_DIM]], dtype=np.float32)
            svd_emb[tid] = arr
        
        audio = r['audio']
        if isinstance(audio, np.ndarray):
            audio_emb[tid] = audio[:128].astype(np.float32)

print(f"SVD embeddings: {len(svd_emb)}, Audio embeddings: {len(audio_emb)}")

# Create matrices
embedding_matrix = np.random.randn(num_tracks, EMB_DIM).astype(np.float32) * 0.01
audio_matrix = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01

for tid, rid in track2id.items():
    if tid in svd_emb:
        embedding_matrix[rid] = svd_emb[tid]
    if tid in audio_emb:
        audio_matrix[rid] = audio_emb[tid]

print(f"Embedding matrix: {embedding_matrix.shape}")

del svd_emb, audio_emb
gc.collect()

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


class ACTRCognitive(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pm = nn.Parameter(torch.ones(dim) * 0.1)
    
    def forward(self, x):
        f = torch.mean(x * self.pm, dim=-1, keepdim=True)
        return x * torch.sigmoid(f)


class AudioEncoder(nn.Module):
    def __init__(self, audio_dim=128, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class AU2ACTR(nn.Module):
    def __init__(self, n_tracks, dim=128, hidden=256):
        super().__init__()
        
        self.emb = nn.Embedding(n_tracks, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim, max_len=100)
        
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=2, dim_feedforward=hidden,
            dropout=0, activation='gelu', batch_first=True, norm_first=True
        )
        self.trans = nn.TransformerEncoder(enc, num_layers=2, enable_nested_tensor=False)
        
        self.actr = ACTRCognitive(dim)
        
        self.audio_enc = AudioEncoder(128, 256, dim)
        self.audio_emb = nn.Embedding.from_pretrained(
            torch.from_numpy(audio_matrix), freeze=False
        )
        
        self.out = nn.Linear(dim, n_tracks)
        
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(embedding_matrix))
    
    def forward(self, x, return_emb=False):
        e = self.emb(x)
        e = self.pos(e)
        o = self.trans(e)
        h = o[:, -1, :]
        h = self.actr(h)
        logits = self.out(h)
        
        if return_emb:
            return logits, h
        return logits
    
    def predict(self, x):
        logits, h = self.forward(x, return_emb=True)
        audio_all = self.audio_enc(self.audio_emb.weight)
        scores = torch.matmul(h, audio_all.T)
        return logits + LAMBDA_AU * scores


# ============= DATASET =============

class SessionDataset(Dataset):
    def __init__(self, sess, t2id, seq=10, min_s=5):
        self.samples = []
        
        for u, s in sess.items():
            if len(s) < min_s + 3:
                continue
            
            train = s[:-3]
            
            for i in range(len(train) - seq):
                inp = []
                for ss in train[i:i+seq]:
                    inp.extend([t2id.get(t, 0) for t in ss['tracks'][:5]])
                
                tgt = t2id.get(train[i+seq]['tracks'][0], 0)
                if tgt == 0:
                    continue
                
                fl = seq * 5
                inp = (inp + [0] * fl)[:fl]
                self.samples.append((inp, tgt))
        
        print(f"  Created {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============= TRAINING =============

print("\nCreating dataset...")
ds = SessionDataset(sessions, track2id, SEQ_LEN, MIN_SESSIONS)
print(f"Total samples: {len(ds)}")

del sessions
gc.collect()

train_size = int(0.9 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=0)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

print("\nCreating model...")
model = AU2ACTR(num_tracks, EMB_DIM, HID_DIM).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

print("\n" + "="*60)
print("TRAINING STARTED")
print("="*60)

best_acc = 0
training_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model.predict(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    correct, total = 0, 0
    recall_at_10 = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model.predict(x)
            
            preds = out.argmax(dim=1)
            correct += (preds == y.to(device)).sum().item()
            
            top10 = out.topk(10, dim=1).indices
            for i in range(len(y)):
                if y[i].item() in top10[i]:
                    recall_at_10 += 1
            
            total += y.size(0)
    
    acc = correct / total if total > 0 else 0
    recall10 = recall_at_10 / total if total > 0 else 0
    
    if acc > best_acc:
        best_acc = acc
        checkpoint_path = CACHE_DIR / f"best_{MAX_USERS}users.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  *** NEW BEST! Accuracy: {acc:.4f}, Recall@10: {recall10:.4f} ***")
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - training_start
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={total_loss/len(train_loader):.4f}, "
          f"Acc={acc:.4f}, Recall@10={recall10:.4f}, "
          f"Best={best_acc:.4f}, Time={epoch_time:.1f}s, Total={total_time/3600:.2f}h")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best Accuracy: {best_acc:.4f}")
print(f"Total Time: {(time.time() - training_start) / 3600:.2f} hours")
