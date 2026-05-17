"""
AU2ACTR - Conservative Version (20k users)
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

DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_USERS = 20000  # Conservative
MIN_SESSIONS = 5
SEQ_LEN = 10
EMB_DIM = 128
HID_DIM = 256
BATCH = 256  # Smaller batch
EPOCHS = 50
LR = 0.001
LAMBDA_AU = 0.6

print("="*60)
print("AU2ACTR - Conservative (20k users)")
print("="*60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

start_time = time.time()

# Count sessions
DATA_DIR = DATA_PATH / "user_sessions"
FILES = sorted(os.listdir(DATA_DIR))

print("\nCounting...")
user_counts = Counter()
for i, f in enumerate(FILES):
    if i % 100 == 0:
        print(f"  {i}/{len(FILES)}")
    df = pd.read_parquet(DATA_DIR / f, columns=['user_id', 'session_id'])
    for (u, s), _ in df.groupby(['user_id', 'session_id']):
        user_counts[u] += 1

eligible = [(u, c) for u, c in user_counts.items() if c >= MIN_SESSIONS]
eligible.sort(key=lambda x: x[1], reverse=True)
eligible = eligible[:MAX_USERS]
eligible_users = {u for u, c in eligible}
print(f"Selected {len(eligible_users)} users")

del user_counts, eligible
gc.collect()

# Load sessions
print("\nLoading sessions...")
sessions = defaultdict(list)
for i, f in enumerate(FILES):
    if i % 100 == 0:
        print(f"  {i}/{len(FILES)}...")
    
    df = pd.read_parquet(DATA_DIR / f)
    df = df[df['user_id'].isin(eligible_users)]
    
    if len(df) == 0:
        continue
    
    for (u, s), g in df.groupby(['user_id', 'session_id']):
        tracks = sorted(set(g['track_id'].tolist()))
        sessions[u].append(tracks)

# Sort and filter
for u in sessions:
    sessions[u] = sorted(sessions[u])
sessions = {u: s for u, s in sessions.items() if len(s) >= MIN_SESSIONS}
print(f"Final: {len(sessions)} users")

# Save
cache_file = CACHE_DIR / "sessions_20k.pkl"
with open(cache_file, 'wb') as f:
    pickle.dump(dict(sessions), f)
print(f"Saved: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

# Vocabulary
print("\nBuilding vocab...")
all_tracks = set()
for t_list in sessions.values():
    for t in t_list:
        all_tracks.update(t)

track2id = {t: i+1 for i, t in enumerate(sorted(all_tracks))}
num_tracks = len(track2id) + 1
print(f"Tracks: {num_tracks}")

del all_tracks
gc.collect()

# Embeddings
print("\nLoading embeddings...")
emb_files = list((DATA_PATH / "track_embeddings").iterdir())
svd_emb, audio_emb = {}, {}

for f in emb_files:
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

emb_mat = np.random.randn(num_tracks, EMB_DIM).astype(np.float32) * 0.01
audio_mat = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01

for tid, rid in track2id.items():
    if tid in svd_emb:
        emb_mat[rid] = svd_emb[tid]
    if tid in audio_emb:
        audio_mat[rid] = audio_emb[tid]

del svd_emb, audio_emb
gc.collect()

# Model classes
class PosEnc(nn.Module):
    def __init__(self, d, m=100):
        super().__init__()
        p = torch.zeros(m, d)
        pos = torch.arange(0, m, dtype=torch.float).unsqueeze(1)
        dt = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        p[:, 0::2] = torch.sin(pos * dt)
        p[:, 1::2] = torch.cos(pos * dt)
        self.register_buffer('p', p)
    
    def forward(self, x):
        return x + self.p[:x.size(1)].unsqueeze(0)

class ACTR(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.pm = nn.Parameter(torch.ones(d) * 0.1)
    
    def forward(self, x):
        f = torch.mean(x * self.pm, dim=-1, keepdim=True)
        return x * torch.sigmoid(f)

class AudioEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        return self.net(x)

class AU2ACTR(nn.Module):
    def __init__(self, n_tracks):
        super().__init__()
        self.emb = nn.Embedding(n_tracks, 128, padding_idx=0)
        self.pos = PosEnc(128)
        enc = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=256, dropout=0, activation='gelu', batch_first=True, norm_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=2, enable_nested_tensor=False)
        self.actr = ACTR(128)
        self.audio_enc = AudioEnc()
        self.audio_emb = nn.Embedding.from_pretrained(torch.from_numpy(audio_mat), freeze=False)
        self.out = nn.Linear(128, n_tracks)
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(emb_mat))
    
    def forward(self, x, ret=False):
        e = self.pos(self.emb(x))
        h = self.trans(e)[:, -1, :]
        h = self.actr(h)
        logits = self.out(h)
        return (logits, h) if ret else logits
    
    def predict(self, x):
        logits, h = self.forward(x, ret=True)
        return logits + LAMBDA_AU * torch.matmul(h, self.audio_enc(self.audio_emb.weight).T)

# Dataset
class DS(Dataset):
    def __init__(self, sess, t2id):
        self.samples = []
        for u, tracks in sess.items():
            if len(tracks) < MIN_SESSIONS + 3:
                continue
            train = tracks[:-3]
            for i in range(len(train) - SEQ_LEN):
                inp = []
                for t_list in train[i:i+SEQ_LEN]:
                    inp.extend([t2id.get(t, 0) for t in t_list[:5]])
                tgt = t2id.get(train[i+SEQ_LEN][0], 0)
                if tgt == 0:
                    continue
                inp = (inp + [0] * (SEQ_LEN * 5))[:SEQ_LEN * 5]
                self.samples.append((inp, tgt))
        print(f"Samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

print("\nCreating dataset...")
ds = DS(sessions, track2id)
del sessions
gc.collect()

tr_sz = int(0.9 * len(ds))
train_ds, val_ds = torch.utils.data.random_split(ds, [tr_sz, len(ds) - tr_sz])
train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_ld = DataLoader(val_ds, batch_size=BATCH)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

print("\nCreating model...")
model = AU2ACTR(num_tracks).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

print("\n" + "="*60)
print("TRAINING")
print("="*60)

best = 0
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for i, (x, y) in enumerate(train_ld):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model.predict(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"  E{epoch+1} B{i} L={loss.item():.4f}")
    
    # Val
    model.eval()
    correct, total, r10 = 0, 0, 0
    with torch.no_grad():
        for x, y in val_ld:
            out = model.predict(x.to(device))
            preds = out.argmax(1)
            correct += (preds == y.to(device)).sum().item()
            top10 = out.topk(10, dim=1).indices
            for j in range(len(y)):
                if y[j].item() in top10[j]:
                    r10 += 1
            total += y.size(0)
    
    acc = correct / total
    recall10 = r10 / total
    if acc > best:
        best = acc
        torch.save(model.state_dict(), CACHE_DIR / "best_20k.pt")
        print(f"  *** BEST! Acc={acc:.4f} R@10={recall10:.4f} ***")
    
    print(f"E{epoch+1}: L={total_loss/len(train_ld):.4f} A={acc:.4f} R@10={recall10:.4f} Best={best:.4f} Time={(time.time()-t0)/60:.1f}min")

print(f"\nDONE! Best={best:.4f}")
