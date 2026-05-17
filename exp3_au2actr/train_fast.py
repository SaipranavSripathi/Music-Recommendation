"""
AU2ACTR - Fast Training with Sampled Users
Optimized for quick iteration
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
import time
import datetime

print(f"PyTorch: {torch.__version__}")

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using: {device}")

# ============= CONFIG =============
DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")
MAX_USERS = 20000  # Quick test
MIN_SESSIONS = 10
SEQ_LEN = 10
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_BLOCKS = 2
NUM_HEADS = 2
LAMBDA_AUENC = 0.6

print("\n[1] Loading sessions from cache or creating new...")
start_time = time.time()

session_cache_path = CACHE_DIR / "deezer" / f"min{MIN_SESSIONS}sess_{MAX_USERS}users"
os.makedirs(session_cache_path, exist_ok=True)

cached_file = session_cache_path / "user_sessions.pkl"

if cached_file.exists():
    print("Loading from cache...")
    with open(cached_file, 'rb') as f:
        filtered_sessions = pickle.load(f)
    print(f"Loaded {len(filtered_sessions)} users")
else:
    # FAST: Only process first 20 files
    DATA_DIR = DATA_PATH / "user_sessions"
    FILES = sorted(os.listdir(DATA_DIR))[:20]
    
    print(f"Processing {len(FILES)} files...")
    user_counts = defaultdict(int)
    
    for fname in FILES:
        df = pd.read_parquet(DATA_DIR / fname, columns=['user_id', 'session_id'])
        for (uid, sid), _ in df.groupby(['user_id', 'session_id']):
            user_counts[uid] += 1
    
    # Select users with enough sessions
    eligible = {u for u, c in user_counts.items() if c >= MIN_SESSIONS}
    eligible = list(eligible)[:MAX_USERS]
    eligible_set = set(eligible)
    print(f"Selected {len(eligible)} users")
    
    del user_counts
    gc.collect()
    
    # Load sessions
    user_sessions = defaultdict(list)
    for fname in FILES:
        df = pd.read_parquet(DATA_DIR / fname)
        df = df[df['user_id'].isin(eligible_set)]
        df = df.sort_values('ts')
        
        for (uid, sid), grp in df.groupby(['user_id', 'session_id']):
            tracks = sorted(set(grp['track_id'].tolist()))
            ts = grp['ts'].iloc[0]
            dt = datetime.datetime.fromtimestamp(ts)
            user_sessions[uid].append({
                'session_id': sid,
                'context': {'ts': ts, 'day_of_week': dt.weekday(), 'hour_of_day': dt.hour},
                'track_ids': tracks
            })
    
    # Sort and filter
    for uid in user_sessions:
        user_sessions[uid] = sorted(user_sessions[uid], key=lambda x: x['context']['ts'])
    
    filtered_sessions = {u: s for u, s in user_sessions.items() if len(s) >= MIN_SESSIONS}
    print(f"Final: {len(filtered_sessions)} users")
    
    del user_sessions
    gc.collect()
    
    # Save
    with open(cached_file, 'wb') as f:
        pickle.dump(filtered_sessions, f)
    print("Saved to cache")

# Build vocabulary
print("\n[2] Building vocabulary...")
all_tracks = set()
for sessions in filtered_sessions.values():
    for s in sessions:
        all_tracks.update(s['track_ids'])

track_list = sorted(all_tracks)
track2id = {t: i+1 for i, t in enumerate(track_list)}
num_tracks = len(track_list) + 1
print(f"Tracks: {num_tracks}")

del all_tracks
gc.collect()

# Load embeddings
print("\n[3] Loading embeddings...")
emb_files = sorted((DATA_PATH / "track_embeddings").glob("*.parquet"))

svd_emb = {}
audio_emb = {}

for f in emb_files:
    print(f"  {f.name}")
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        tid = r['track_id']
        if tid not in track2id:
            continue
        svd = r['svd']
        audio = r['audio']
        if isinstance(svd, list):
            svd_emb[tid] = np.array(svd[:128], dtype=np.float32)
        if isinstance(audio, list):
            audio_emb[tid] = np.array(audio[:128], dtype=np.float32)

print(f"Loaded {len(svd_emb)} SVD, {len(audio_emb)} audio")

# Embedding matrices
embedding_matrix = np.random.randn(num_tracks, EMBEDDING_DIM).astype(np.float32) * 0.01
audio_matrix = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01

for tid, rid in track2id.items():
    if tid in svd_emb:
        embedding_matrix[rid] = svd_emb[tid]
    if tid in audio_emb:
        audio_matrix[rid] = audio_emb[tid]

print(f"Matrices: {embedding_matrix.shape}, {audio_matrix.shape}")


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
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ACTRCognitive(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pm = nn.Parameter(torch.ones(dim) * 0.1)
    
    def forward(self, x):
        f = torch.mean(x * self.pm, dim=-1, keepdim=True)
        return x * torch.sigmoid(f)


class AU2ACTR(nn.Module):
    def __init__(self, num_tracks, dim=128, hidden=256, heads=2, blocks=2):
        super().__init__()
        
        self.track_emb = nn.Embedding(num_tracks, dim, padding_idx=0)
        self.pos = PositionalEncoding(dim, max_len=100)
        
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=hidden,
                                         dropout=0, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=blocks, enable_nested_tensor=False)
        
        self.actr = ACTRCognitive(dim)
        
        self.audio_enc = AudioEncoder(128, 256, dim)
        self.audio_emb = nn.Embedding.from_pretrained(torch.from_numpy(audio_matrix), freeze=False)
        
        self.out = nn.Linear(dim, num_tracks)
        
        with torch.no_grad():
            self.track_emb.weight.copy_(torch.from_numpy(embedding_matrix))
            self.audio_emb.weight.copy_(torch.from_numpy(audio_matrix))
    
    def forward(self, x):
        e = self.track_emb(x)
        e = self.pos(e)
        o = self.transformer(e)
        h = o[:, -1, :]
        h = self.actr(h)
        return self.out(h)
    
    def predict(self, x):
        logits, hidden, _ = self.forward(x, return_embeddings=True)
        audio_all = self.audio_enc(self.audio_emb.weight)
        scores = torch.matmul(hidden, audio_all.T)
        return logits + LAMBDA_AUENC * scores
    
    def forward(self, x, return_embeddings=False):
        e = self.track_emb(x)
        e = self.pos(e)
        o = self.transformer(e)
        h = o[:, -1, :]
        h = self.actr(h)
        logits = self.out(h)
        if return_embeddings:
            return logits, h, None
        return logits


# ============= DATASET =============

class DS(Dataset):
    def __init__(self, sessions, t2id, seq_len=10, min_sess=10):
        self.samples = []
        
        for uid, sess in sessions.items():
            if len(sess) < min_sess + 5:
                continue
            
            train = sess[:-5]
            for i in range(len(train) - seq_len):
                inp = []
                for s in train[i:i+seq_len]:
                    inp.extend([t2id.get(t, 0) for t in s['track_ids'][:5]])
                
                target = t2id.get(train[i+seq_len]['track_ids'][0], 0)
                if target == 0:
                    continue
                
                flat_len = seq_len * 5
                if len(inp) > flat_len:
                    inp = inp[:flat_len]
                else:
                    inp = inp + [0] * (flat_len - len(inp))
                
                self.samples.append((inp, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============= TRAINING =============

print("\n[4] Creating dataset...")
ds = DS(filtered_sessions, track2id, SEQ_LEN, MIN_SESSIONS)
print(f"Samples: {len(ds)}")

del filtered_sessions
gc.collect()

train_sz = int(0.9 * len(ds))
val_sz = len(ds) - train_sz
tr_ds, val_ds = torch.utils.data.random_split(ds, [train_sz, val_sz])

tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"Train: {len(tr_ds)}, Val: {len(val_ds)}")

print("\n[5] Creating model...")
model = AU2ACTR(num_tracks, EMBEDDING_DIM, HIDDEN_DIM, NUM_HEADS, NUM_BLOCKS).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

print("\n[6] Training...")
best = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for i, (x, y) in enumerate(tr_ld):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model.predict(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"  E{epoch} B{i} L{loss.item():.4f}")
    
    sched.step()
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_ld:
            x = x.to(device)
            out = model.predict(x)
            preds = out.argmax(1)
            correct += (preds == y.to(device)).sum().item()
            total += y.size(0)
    
    acc = correct / total if total > 0 else 0
    
    if acc > best:
        best = acc
        torch.save(model.state_dict(), CACHE_DIR / "best.pt")
        print(f"  BEST! {acc:.4f}")
    
    print(f"E{epoch+1} L{total_loss/len(tr_ld):.4f} A{acc:.4f} B{best:.4f}")

print(f"\n{'='*50}")
print(f"Done! Best: {best:.4f}")
print(f"Time: {time.time() - start_time:.1f}s")
