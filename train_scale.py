"""
Quick scaling test - uses cached sessions
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
import math
import gc
import time
import psutil

print(f"PyTorch: {torch.__version__}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")

SEQ_LEN = 10
BATCH = 256
EPOCHS = 3  # Quick test
LR = 0.001
LAMBDA_AU = 0.6
EMB_DIM = 128
HID_DIM = 256
MIN_SESSIONS = 10


def get_memory_mb():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / 1024 / 1024


# Try to load progressively larger sessions
user_counts = [10000, 20000, 40000, 80000, 160000, 320000]

results = {}

for max_users in user_counts:
    cache_file = CACHE_DIR / f"sessions_u{max_users}_s{MIN_SESSIONS}.pkl"
    
    if not cache_file.exists():
        print(f"\n=== {max_users} users: Cache not found, need to generate it first ===")
        print(f"File: {cache_file}")
        print("Let me generate the cache first...")
        
        # Generate this cache
        print(f"Generating sessions for {max_users} users...")
        
        import pandas as pd
        from collections import defaultdict
        
        DATA_DIR = DATA_PATH / "user_sessions"
        FILES = sorted(os.listdir(DATA_DIR))
        
        # First pass
        user_counts_dict = {}
        for f in FILES:
            df = pd.read_parquet(DATA_DIR / f, columns=['user_id', 'session_id'])
            for (u, s), _ in df.groupby(['user_id', 'session_id']):
                if u not in user_counts_dict:
                    user_counts_dict[u] = 0
                user_counts_dict[u] += 1
        
        eligible = [u for u, c in user_counts_dict.items() if c >= MIN_SESSIONS]
        eligible = sorted(eligible, key=lambda x: user_counts_dict[x], reverse=True)[:max_users]
        eligible_set = set(eligible)
        
        print(f"Selected {len(eligible)} users")
        
        del user_counts_dict
        gc.collect()
        
        # Second pass
        sessions = defaultdict(list)
        for f in FILES:
            df = pd.read_parquet(DATA_DIR / f)
            df = df[df['user_id'].isin(eligible_set)]
            if len(df) == 0:
                continue
            df = df.sort_values('ts')
            for (u, s), g in df.groupby(['user_id', 'session_id']):
                t = sorted(set(g['track_id'].tolist()))
                sessions[u].append({'tracks': t})
        
        for u in sessions:
            sessions[u] = sorted(sessions[u], key=lambda x: x['tracks'][0] if x['tracks'] else 0)
        
        filtered = {u: s for u, s in sessions.items() if len(s) >= MIN_SESSIONS}
        
        with open(cache_file, 'wb') as f:
            pickle.dump(filtered, f)
        
        print(f"Saved to {cache_file}")
        
        del sessions, filtered
        gc.collect()
    
    # Load and train
    print(f"\n{'='*50}")
    print(f"STEP: {max_users} users")
    print(f"{'='*50}")
    
    with open(cache_file, 'rb') as f:
        sessions = pickle.load(f)
    
    print(f"Loaded {len(sessions)} users")
    
    # Check memory
    mem_before = get_memory_mb()
    print(f"Memory: {mem_before:.1f} MB")
    
    if mem_before > 15000:
        print("WARNING: High memory usage!")
    
    # Build vocab
    all_tracks = set()
    for s in sessions.values():
        for sess in s:
            all_tracks.update(sess['tracks'])
    
    track_list = sorted(all_tracks)
    track2id = {t: i+1 for i, t in enumerate(track_list)}
    num_tracks = len(track_list) + 1
    
    del all_tracks
    gc.collect()
    
    print(f"Tracks: {num_tracks}")
    
    # Load embeddings
    print("Loading embeddings...")
    emb_files = list((DATA_PATH / "track_embeddings").iterdir())
    
    svd_emb = {}
    audio_emb = {}
    
    for f in emb_files:
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
    
    print(f"SVD: {len(svd_emb)}, Audio: {len(audio_emb)}")
    
    emb_matrix = np.random.randn(num_tracks, EMB_DIM).astype(np.float32) * 0.01
    audio_matrix = np.random.randn(num_tracks, 128).astype(np.float32) * 0.01
    
    for tid, rid in track2id.items():
        if tid in svd_emb:
            emb_matrix[rid] = svd_emb[tid]
        if tid in audio_emb:
            audio_matrix[rid] = audio_emb[tid]
    
    del svd_emb, audio_emb
    gc.collect()
    
    # Dataset
    class DS(Dataset):
        def __init__(self, sess, t2id, seq=10, min_s=10):
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
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, i):
            x, y = self.samples[i]
            return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    ds = DS(sessions, track2id, SEQ_LEN, MIN_SESSIONS)
    print(f"Samples: {len(ds)}")
    
    del sessions
    gc.collect()
    
    if len(ds) < 100:
        print("Not enough samples, skipping...")
        results[max_users] = "SKIP"
        continue
    
    tr_sz = int(0.9 * len(ds))
    val_sz = len(ds) - tr_sz
    tr, val = torch.utils.data.random_split(ds, [tr_sz, val_sz])
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True)
    val_ld = DataLoader(val, batch_size=BATCH)
    
    print(f"Train: {len(tr)}, Val: {len(val)}")
    
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
        def __init__(self, n_tracks, emb_matrix, audio_matrix, dim=128, hidden=256):
            super().__init__()
            
            self.emb = nn.Embedding(n_tracks, dim, padding_idx=0)
            self.pos = PosEnc(dim, m=100)
            
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
                self.emb.weight.copy_(torch.from_numpy(emb_matrix))
        
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
    
    print("Creating model...")
    model = AU2ACTR(num_tracks, emb_matrix, audio_matrix, EMB_DIM, HID_DIM).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    mem_after_model = get_memory_mb()
    print(f"Memory after model: {mem_after_model:.1f} MB")
    
    if mem_after_model > 18000:
        print("WARNING: Memory is getting too high! Stopping before OOM.")
        results[max_users] = "NEAR_OOM"
        del model
        gc.collect()
        break
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("Training...")
    best = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for i, (x, y) in enumerate(tr_ld):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model.predict(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            if i % 200 == 0:
                print(f"  E{epoch} B{i} L={loss.item():.4f}")
        
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
            torch.save(model.state_dict(), CACHE_DIR / f"best_u{max_users}.pt")
        
        print(f"  E{epoch+1} Acc={acc:.4f} Best={best:.4f}")
    
    results[max_users] = best
    
    # Cleanup
    del model, tr_ld, val_ld, ds
    gc.collect()
    
    mem_final = get_memory_mb()
    print(f"Memory after cleanup: {mem_final:.1f} MB")
    
    print(f"\n=== {max_users} users: {best:.4f} accuracy ===")

# Summary
print(f"\n{'='*60}")
print("SCALING RESULTS")
print(f"{'='*60}")
for users, acc in results.items():
    print(f"  {users:>7} users: {acc}")
print(f"{'='*60}")
