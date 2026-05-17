"""
AU2ACTR - CPU Fast Version
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
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
device = torch.device("cpu")  # Force CPU
print(f"Device: {device}")

DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")

MAX_USERS = 5000
MIN_SESSIONS = 5
SEQ_LEN = 5
EMB_DIM = 32
HID_DIM = 64
BATCH = 64
EPOCHS = 5
LR = 0.001

start = time.time()

# Check cache
cache_path = CACHE_DIR / f"cpu_u{MAX_USERS}_s{MIN_SESSIONS}.pkl"
if cache_path.exists():
    print("Loading cache...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    filtered_sessions = data['sessions']
    track2id = data['track2id']
    num_tracks = data['num_tracks']
    embedding_matrix = data['emb']
else:
    print("Creating data...")
    DATA_DIR = DATA_PATH / "user_sessions"
    FILES = sorted(os.listdir(DATA_DIR))[:10]
    
    user_counts = defaultdict(int)
    for f in FILES:
        df = pd.read_parquet(DATA_DIR / f, columns=['user_id', 'session_id'])
        for (u, s), _ in df.groupby(['user_id', 'session_id']):
            user_counts[u] += 1
    
    eligible = [u for u, c in user_counts.items() if c >= MIN_SESSIONS][:MAX_USERS]
    eligible_set = set(eligible)
    print(f"Users: {len(eligible)}")
    
    del user_counts
    gc.collect()
    
    sessions = defaultdict(list)
    for f in FILES:
        df = pd.read_parquet(DATA_DIR / f)
        df = df[df['user_id'].isin(eligible_set)]
        df = df.sort_values('ts')
        for (u, s), g in df.groupby(['user_id', 'session_id']):
            t = sorted(set(g['track_id'].tolist()))
            sessions[u].append({'tracks': t})
    
    for u in sessions:
        sessions[u] = sorted(sessions[u], key=lambda x: x['tracks'][0] if x['tracks'] else 0)
    
    filtered_sessions = {u: s for u, s in sessions.items() if len(s) >= MIN_SESSIONS}
    print(f"Final: {len(filtered_sessions)}")
    
    del sessions
    gc.collect()
    
    all_tracks = set()
    for s in filtered_sessions.values():
        for sess in s:
            all_tracks.update(sess['tracks'])
    
    track_list = sorted(all_tracks)
    track2id = {t: i+1 for i, t in enumerate(track_list)}
    num_tracks = len(track_list) + 1
    print(f"Tracks: {num_tracks}")
    
    del all_tracks
    gc.collect()
    
    embedding_matrix = np.random.randn(num_tracks, EMB_DIM).astype(np.float32) * 0.01
    
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'sessions': filtered_sessions,
            'track2id': track2id,
            'num_tracks': num_tracks,
            'emb': embedding_matrix
        }, f)
    print("Saved cache")


# Simple Model
class SimpleModel(nn.Module):
    def __init__(self, n_tracks, dim=32):
        super().__init__()
        self.emb = nn.Embedding(n_tracks, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.out = nn.Linear(dim, n_tracks)
    
    def forward(self, x):
        e = self.emb(x)
        _, (h, _) = self.lstm(e)
        h = h.squeeze(0)
        return self.out(h)


class DS(Dataset):
    def __init__(self, sess, t2id, seq=5, min_s=5):
        self.samples = []
        for u, s in sess.items():
            if len(s) < min_s + 1:
                continue
            for i in range(len(s) - seq):
                inp = []
                for ss in s[i:i+seq]:
                    inp.extend([t2id.get(t, 0) for t in ss['tracks'][:2]])
                tgt = t2id.get(s[i+seq]['tracks'][0], 0)
                if tgt == 0:
                    continue
                fl = seq * 2
                inp = (inp + [0] * fl)[:fl]
                self.samples.append((inp, tgt))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


print("\nCreating dataset...")
ds = DS(filtered_sessions, track2id, SEQ_LEN, MIN_SESSIONS)
print(f"Samples: {len(ds)}")

del filtered_sessions
gc.collect()

tr_sz = int(0.9 * len(ds))
tr, val = torch.utils.data.random_split(ds, [tr_sz, len(ds) - tr_sz])
tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True)
val_ld = DataLoader(val, batch_size=BATCH)

print(f"Train: {len(tr)}, Val: {len(val)}")

print("\nCreating model...")
model = SimpleModel(num_tracks, EMB_DIM).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("\nTraining...")
best = 0

for epoch in range(EPOCHS):
    model.train()
    total = 0
    for i, (x, y) in enumerate(tr_ld):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        total += loss.item()
        
        if i % 50 == 0:
            print(f"  E{epoch} B{i} L{loss.item():.4f}")
    
    # Val
    model.eval()
    correct, total_v = 0, 0
    with torch.no_grad():
        for x, y in val_ld:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y.to(device)).sum().item()
            total_v += y.size(0)
    
    acc = correct / total_v if total_v > 0 else 0
    if acc > best:
        best = acc
        torch.save(model.state_dict(), CACHE_DIR / "best_cpu.pt")
        print(f"  BEST! {acc:.4f}")
    
    print(f"E{epoch+1} L{total/len(tr_ld):.4f} A{acc:.4f} B{best:.4f}")

print(f"\n{'='*50}")
print(f"Best: {best:.4f}")
print(f"Time: {time.time()-start:.1f}s")
