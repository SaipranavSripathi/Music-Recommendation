"""
AU2ACTR - Memory-Efficient Training
Uses streaming data loading to avoid OOM
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

# Use 50k users - safe for most Macs
MAX_USERS = 50000
MIN_SESSIONS = 5
SEQ_LEN = 10
EMB_DIM = 128
HID_DIM = 256
BATCH = 512
EPOCHS = 50
LR = 0.001
LAMBDA_AU = 0.6

print("="*60)
print("AU2ACTR - Memory-Efficient Training")
print("="*60)
print(f"Config: MAX_USERS={MAX_USERS}, MIN_SESSIONS={MIN_SESSIONS}")
print("="*60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ============= STEP 1: Find eligible users =============
start_time = time.time()

DATA_DIR = DATA_PATH / "user_sessions"
FILES = sorted(os.listdir(DATA_DIR))

print("\nStep 1: Counting sessions per user...")
user_counts = Counter()

for i, f in enumerate(FILES):
    if i % 100 == 0:
        print(f"  {i}/{len(FILES)}...")
    df = pd.read_parquet(DATA_DIR / f, columns=['user_id', 'session_id'])
    for (u, s), _ in df.groupby(['user_id', 'session_id']):
        user_counts[u] += 1

print(f"Total users: {len(user_counts)}")

# Get top users by session count
eligible = [(u, c) for u, c in user_counts.items() if c >= MIN_SESSIONS]
eligible.sort(key=lambda x: x[1], reverse=True)
eligible = eligible[:MAX_USERS]
eligible_users = {u for u, c in eligible}
eligible_list = [u for u, c in eligible]

print(f"Selected top {len(eligible_users)} users")

del user_counts, eligible
gc.collect()

# ============= STEP 2: Load sessions file by file =============
print("\nStep 2: Loading sessions incrementally...")

user_sessions = defaultdict(list)
files_needed = []

for i, f in enumerate(FILES):
    if i % 100 == 0:
        print(f"  {i}/{len(FILES)}... ({len(user_sessions)} users loaded)")
    
    df = pd.read_parquet(DATA_DIR / f)
    df = df[df['user_id'].isin(eligible_users)]
    
    if len(df) == 0:
        continue
    
    df = df.sort_values('ts')
    
    for (u, s), g in df.groupby(['user_id', 'session_id']):
        tracks = sorted(set(g['track_id'].tolist()))
        user_sessions[u].append(tracks)
    
    # Check if we have enough users with enough sessions
    ready_users = [u for u in eligible_list if len(user_sessions.get(u, [])) >= MIN_SESSIONS]
    if len(ready_users) >= MAX_USERS:
        print(f"  Have {len(ready_users)} ready users, stopping early")
        break

print(f"Loaded {len(user_sessions)} users")

# Filter and sort
final_sessions = {}
for u in eligible_list:
    if u in user_sessions and len(user_sessions[u]) >= MIN_SESSIONS:
        final_sessions[u] = sorted(user_sessions[u], key=lambda x: x[0] if x else 0)

print(f"Final: {len(final_sessions)} users with {MIN_SESSIONS}+ sessions")

del user_sessions
gc.collect()

# Save for potential reuse
cache_file = CACHE_DIR / f"sessions_{MAX_USERS}users_s{MIN_SESSIONS}.pkl"
print(f"Saving to {cache_file}")
with open(cache_file, 'wb') as f:
    pickle.dump(final_sessions, f)
print(f"Saved ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")

print(f"Data loading: {time.time() - start_time:.1f}s")

# ============= Build vocabulary =============
print("\nBuilding vocabulary...")
all_tracks = set()
for tracks in final_sessions.values():
    for t in tracks:
        all_tracks.update(t)

track_list = sorted(all_tracks)
track2id = {t: i+1 for i, t in enumerate(track_list)}
num_tracks = len(track_list) + 1
print(f"Tracks: {num_tracks}")

del all_tracks
gc.collect()

# ============= Load embeddings =============
print("\nLoading embeddings...")
emb_files = list((DATA_PATH / "track_embeddings").iterdir())

svd_emb = {}
audio_emb = {}

for f in emb_files:
    print(f"  {f.name}...")
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        tid = r['track_id']
        if tid not in track2id:
            continue
        svd = r['svd']
        if isinstance(svd, dict) and 'list' in svd:
            svd_emb[tid] = np.array([x['item'] for x in svd['list'][:EMB_DIM]], dtype=np.float32)
        audio = r['audio']
        if isinstance(audio, np.ndarray):
            audio_emb[tid] = audio[:128].astype(np.float32)

print(f"SVD: {len(svd_emb)}, Audio: {len(audio_emb)}")

embedding_matrix = np.random.randn(num_tracks, EMB_DIM).astype(np.float32) * 0.01
audio_matrix = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01

for tid, rid in track2id.items():
    if tid in svd_emb:
        embedding_matrix[rid] = svd_emb[tid]
    if tid in audio_emb:
        audio_matrix[rid] = audio_emb[tid]

del svd_emb, audio_emb
gc.collect()

# ============= Model =============

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
            nn.Linear(audio_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
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
        self.audio_emb = nn.Embedding.from_pretrained(torch.from_numpy(audio_matrix), freeze=False)
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


# ============= Dataset =============

class SessionDataset(Dataset):
    def __init__(self, sess, t2id, seq=10, min_s=5):
        self.samples = []
        for u, tracks in sess.items():
            if len(tracks) < min_s + 3:
                continue
            train = tracks[:-3]
            for i in range(len(train) - seq):
                inp = []
                for t_list in train[i:i+seq]:
                    inp.extend([t2id.get(t, 0) for t in t_list[:5]])
                tgt = t2id.get(train[i+seq][0], 0)
                if tgt == 0:
                    continue
                fl = seq * 5
                inp = (inp + [0] * fl)[:fl]
                self.samples.append((inp, tgt))
        print(f"  Samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============= Training =============

print("\nCreating dataset...")
ds = SessionDataset(final_sessions, track2id, SEQ_LEN, MIN_SESSIONS)
print(f"Total: {len(ds)} samples")

del final_sessions
gc.collect()

train_sz = int(0.9 * len(ds))
train_ds, val_ds = torch.utils.data.random_split(ds, [train_sz, len(ds) - train_sz])
train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_ld = DataLoader(val_ds, batch_size=BATCH)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

print("\nCreating model...")
model = AU2ACTR(num_tracks, EMB_DIM, HID_DIM).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

print("\n" + "="*60)
print("TRAINING")
print("="*60)

best_acc = 0
training_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for i, (x, y) in enumerate(train_ld):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model.predict(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
        if i % 100 == 0:
            print(f"  E{epoch+1} B{i}/{len(train_ld)} L={loss.item():.4f}")
    
    # Validation
    model.eval()
    correct, total, r10 = 0, 0, 0
    with torch.no_grad():
        for x, y in val_ld:
            x = x.to(device)
            out = model.predict(x)
            preds = out.argmax(1)
            correct += (preds == y.to(device)).sum().item()
            top10 = out.topk(10, dim=1).indices
            for j in range(len(y)):
                if y[j].item() in top10[j]:
                    r10 += 1
            total += y.size(0)
    
    acc = correct / total if total > 0 else 0
    recall10 = r10 / total if total > 0 else 0
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), CACHE_DIR / f"best_{MAX_USERS}users.pt")
        print(f"  *** BEST! Acc={acc:.4f}, R@10={recall10:.4f} ***")
    
    elapsed = time.time() - training_start
    print(f"E{epoch+1}: L={total_loss/len(train_ld):.4f} A={acc:.4f} R@10={recall10:.4f} Best={best_acc:.4f} Time={elapsed/60:.1f}min")

print(f"\nDONE! Best={best_acc:.4f}")
