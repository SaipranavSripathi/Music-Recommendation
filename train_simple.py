"""
AU2ACTR - Simple Fast Training
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
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")

MAX_USERS = 10000
MIN_SESSIONS = 5
SEQ_LEN = 10
EMB_DIM = 64
HID_DIM = 128
BATCH = 128
EPOCHS = 10
LR = 0.001
LAMBDA_AU = 0.5

start = time.time()

# Check cache
cache_path = CACHE_DIR / f"u{MAX_USERS}_s{MIN_SESSIONS}.pkl"
if cache_path.exists():
    print("Loading cache...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    filtered_sessions = data['sessions']
    track2id = data['track2id']
    num_tracks = data['num_tracks']
    embedding_matrix = data['emb']
    audio_matrix = data['audio']
else:
    print("Creating data...")
    DATA_DIR = DATA_PATH / "user_sessions"
    FILES = sorted(os.listdir(DATA_DIR))[:15]
    
    # Count
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
    
    # Load sessions
    sessions = defaultdict(list)
    for f in FILES:
        df = pd.read_parquet(DATA_DIR / f)
        df = df[df['user_id'].isin(eligible_set)]
        df = df.sort_values('ts')
        for (u, s), g in df.groupby(['user_id', 'session_id']):
            t = sorted(set(g['track_id'].tolist()))
            ts = g['ts'].iloc[0]
            dt = datetime.datetime.fromtimestamp(ts)
            sessions[u].append({
                'session_id': s,
                'context': {'ts': ts, 'dow': dt.weekday(), 'hod': dt.hour},
                'tracks': t
            })
    
    for u in sessions:
        sessions[u] = sorted(sessions[u], key=lambda x: x['context']['ts'])
    
    filtered_sessions = {u: s for u, s in sessions.items() if len(s) >= MIN_SESSIONS}
    print(f"Final: {len(filtered_sessions)}")
    
    del sessions
    gc.collect()
    
    # Vocab
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
    
    # Embeddings
    print("Loading embeddings...")
    emb_files = sorted((DATA_PATH / "track_embeddings").glob("*.parquet"))
    
    svd_e = {}
    aud_e = {}
    for f in emb_files:
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            tid = r['track_id']
            if tid not in track2id:
                continue
            if isinstance(r['svd'], list):
                svd_e[tid] = np.array(r['svd'][:EMB_DIM], dtype=np.float32)
            if isinstance(r['audio'], list):
                aud_e[tid] = np.array(r['audio'][:128], dtype=np.float32)
    
    embedding_matrix = np.random.randn(num_tracks, EMB_DIM).astype(np.float32) * 0.01
    audio_matrix = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01
    
    for tid, rid in track2id.items():
        if tid in svd_e:
            embedding_matrix[rid] = svd_e[tid]
        if tid in aud_e:
            audio_matrix[rid] = aud_e[tid]
    
    print(f"SVD: {len(svd_e)}, Audio: {len(aud_e)}")
    
    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'sessions': filtered_sessions,
            'track2id': track2id,
            'num_tracks': num_tracks,
            'emb': embedding_matrix,
            'audio': audio_matrix
        }, f)
    print("Saved cache")


# Model
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


class AU2ACTR(nn.Module):
    def __init__(self, n_tracks, dim=64, hdim=128):
        super().__init__()
        
        self.emb = nn.Embedding(n_tracks, dim, padding_idx=0)
        self.pos = PosEnc(dim)
        
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=2, dim_feedforward=hdim,
                                        dropout=0, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=2)
        
        self.pm = nn.Parameter(torch.ones(dim) * 0.1)
        
        self.aud_enc = nn.Sequential(
            nn.Linear(128, hdim), nn.ReLU(), nn.Linear(hdim, dim)
        )
        self.aud_emb = nn.Embedding.from_pretrained(torch.from_numpy(audio_matrix), freeze=False)
        
        self.out = nn.Linear(dim, n_tracks)
        
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(embedding_matrix))
    
    def forward(self, x):
        e = self.emb(x)
        e = self.pos(e)
        o = self.trans(e)
        h = o[:, -1, :]
        
        f = torch.mean(h * self.pm, dim=-1, keepdim=True)
        h = h * torch.sigmoid(f)
        
        return self.out(h)
    
    def predict(self, x):
        logits = self.forward(x)
        h = self.emb(x)
        h = self.pos(h)
        h = self.trans(h)[:, -1, :]
        f = torch.mean(h * self.pm, dim=-1, keepdim=True)
        h = h * torch.sigmoid(f)
        
        aud = self.aud_enc(self.aud_emb.weight)
        scores = torch.matmul(h, aud.T)
        return logits + LAMBDA_AU * scores


# Dataset
class DS(Dataset):
    def __init__(self, sess, t2id, seq=10, min_s=5):
        self.samples = []
        for u, s in sess.items():
            if len(s) < min_s + 3:
                continue
            train = s[:-3]
            for i in range(len(train) - seq):
                inp = []
                for ss in train[i:i+seq]:
                    inp.extend([t2id.get(t, 0) for t in ss['tracks'][:3]])
                
                tgt = t2id.get(train[i+seq]['tracks'][0], 0)
                if tgt == 0:
                    continue
                
                fl = seq * 3
                if len(inp) > fl:
                    inp = inp[:fl]
                else:
                    inp = inp + [0] * (fl - len(inp))
                
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
model = AU2ACTR(num_tracks, EMB_DIM, HID_DIM).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

print("\nTraining...")
best = 0

for epoch in range(EPOCHS):
    model.train()
    total = 0
    for i, (x, y) in enumerate(tr_ld):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model.predict(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        total += loss.item()
        
        if i % 100 == 0:
            print(f"  E{epoch} B{i} L{loss.item():.4f}")
    
    # Val
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
        torch.save(model.state_dict(), CACHE_DIR / "best_fast.pt")
        print(f"  BEST! {acc:.4f}")
    
    print(f"E{epoch+1} L{total/len(tr_ld):.4f} A{acc:.4f} B{best:.4f}")

print(f"\n{'='*50}")
print(f"Best: {best:.4f}")
print(f"Time: {time.time()-start:.1f}s")
