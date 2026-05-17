"""
AU2ACTR - Progressive Scaling
Starts small and increases users/sessions to find OOM threshold
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
import time
import psutil
from collections import defaultdict

print(f"PyTorch: {torch.__version__}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

DATA_PATH = Path("/Users/spartan/Desktop/CS274/deezer-recsys25")
CACHE_DIR = Path("/Users/spartan/Desktop/CS274/Music-Recommendation/cache")

# Scaling config - start small and increase
USER_STEPS = [5000, 10000, 20000, 40000, 80000, 160000]  # Increase gradually
MIN_SESSIONS = 10
SEQ_LEN = 10
EMB_DIM = 128
HID_DIM = 256
BATCH = 256  # Larger batch for efficiency
EPOCHS = 5   # Fewer epochs per step
LR = 0.001
LAMBDA_AU = 0.6


def get_memory_mb():
    """Get current memory usage in MB"""
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / 1024 / 1024


def load_sessions(max_users, min_sessions):
    """Load user sessions with given constraints"""
    cache_file = CACHE_DIR / f"sessions_u{max_users}_s{min_sessions}.pkl"
    
    if cache_file.exists():
        print(f"Loading cached sessions from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Loading sessions for {max_users} users with {min_sessions}+ sessions...")
    
    DATA_DIR = DATA_PATH / "user_sessions"
    FILES = sorted(os.listdir(DATA_DIR))
    
    # First pass: count sessions per user
    print("  First pass: counting sessions...")
    user_counts = {}
    
    # Process files incrementally until we have enough users
    for f in FILES:
        df = pd.read_parquet(DATA_DIR / f, columns=['user_id', 'session_id'])
        for (u, s), _ in df.groupby(['user_id', 'session_id']):
            if u not in user_counts:
                user_counts[u] = 0
            user_counts[u] += 1
        
        # Check if we have enough users with min_sessions
        eligible = [u for u, c in user_counts.items() if c >= min_sessions]
        if len(eligible) >= max_users:
            break
    
    # Select top max_users by session count
    eligible = sorted(eligible, key=lambda x: user_counts[x], reverse=True)[:max_users]
    eligible_set = set(eligible)
    print(f"  Selected {len(eligible)} users")
    
    # Second pass: load sessions
    print("  Second pass: loading sessions...")
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
    
    # Sort sessions
    for u in sessions:
        sessions[u] = sorted(sessions[u], key=lambda x: x['tracks'][0] if x['tracks'] else 0)
    
    # Filter
    filtered = {u: s for u, s in sessions.items() if len(s) >= min_sessions}
    print(f"  Final: {len(filtered)} users")
    
    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(filtered, f)
    print(f"  Saved to cache")
    
    return filtered


def load_embeddings(track2id, dim=128):
    """Load embeddings"""
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
                arr = np.array([x['item'] for x in svd['list'][:dim]], dtype=np.float32)
                svd_emb[tid] = arr
            
            audio = r['audio']
            if isinstance(audio, np.ndarray):
                audio_emb[tid] = audio[:128].astype(np.float32)
    
    return svd_emb, audio_emb


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


def train_step(sessions, num_tracks, emb_matrix, audio_matrix, step_name):
    """Train one step and return accuracy"""
    print(f"\n{'='*50}")
    print(f"Training: {step_name}")
    print(f"{'='*50}")
    
    start_mem = get_memory_mb()
    print(f"Memory before dataset: {start_mem:.1f} MB")
    
    # Create dataset
    ds = DS(sessions, track2id, SEQ_LEN, MIN_SESSIONS)
    print(f"Samples: {len(ds)}")
    
    after_ds_mem = get_memory_mb()
    print(f"Memory after dataset: {after_ds_mem:.1f} MB (+{after_ds_mem - start_mem:.1f} MB)")
    
    if len(ds) < 100:
        print("Not enough samples, skipping...")
        return None
    
    tr_sz = int(0.9 * len(ds))
    val_sz = len(ds) - tr_sz
    tr, val = torch.utils.data.random_split(ds, [tr_sz, val_sz])
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True)
    val_ld = DataLoader(val, batch_size=BATCH)
    
    print(f"Train: {len(tr)}, Val: {len(val)}")
    
    # Create model
    model = AU2ACTR(num_tracks, emb_matrix, audio_matrix, EMB_DIM, HID_DIM).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    after_model_mem = get_memory_mb()
    print(f"Memory after model: {after_model_mem:.1f} MB (+{after_model_mem - after_ds_mem:.1f} MB)")
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("\nTraining...")
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
            
            if i % 100 == 0:
                print(f"  E{epoch} B{i} L={loss.item():.4f}")
        
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
            torch.save(model.state_dict(), CACHE_DIR / f"best_{step_name}.pt")
        
        print(f"  E{epoch+1} Acc={acc:.4f} Best={best:.4f}")
    
    # Cleanup
    del model, tr_ld, val_ld, ds
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_mem = get_memory_mb()
    print(f"\nMemory after cleanup: {final_mem:.1f} MB")
    
    return best


# Main scaling loop
print(f"\n=== PROGRESSIVE SCALING ===")
print(f"Starting with {USER_STEPS[0]} users, scaling up...")
print(f"Initial memory: {get_memory_mb():.1f} MB")

results = {}

for i, max_users in enumerate(USER_STEPS):
    step_name = f"u{max_users}_s{MIN_SESSIONS}"
    
    print(f"\n{'#'*60}")
    print(f"# STEP {i+1}/{len(USER_STEPS)}: {max_users} users")
    print(f"# {'#'*60}")
    
    try:
        # Load sessions
        sessions = load_sessions(max_users, MIN_SESSIONS)
        
        if len(sessions) < 100:
            print(f"Not enough users ({len(sessions)}), stopping...")
            break
        
        # Build vocabulary
        all_tracks = set()
        for s in sessions.values():
            for sess in s:
                all_tracks.update(sess['tracks'])
        
        track_list = sorted(all_tracks)
        track2id = {t: idx+1 for idx, t in enumerate(track_list)}
        num_tracks = len(track_list) + 1
        print(f"Tracks: {num_tracks}")
        
        del all_tracks
        gc.collect()
        
        # Load embeddings
        print("Loading embeddings...")
        svd_emb, audio_emb = load_embeddings(track2id, EMB_DIM)
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
        
        # Train
        acc = train_step(sessions, num_tracks, emb_matrix, audio_matrix, step_name)
        results[max_users] = acc
        
        print(f"\n=== RESULT: {max_users} users -> {acc:.4f} accuracy ===")
        
        # Check memory
        current_mem = get_memory_mb()
        print(f"Current memory usage: {current_mem:.1f} MB")
        
        # If memory is getting high, warn
        if current_mem > 10000:  # > 10GB
            print("WARNING: Memory usage is getting high!")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print(f"\n!!! OOM at {max_users} users !!!")
            print(f"Error: {e}")
            results[max_users] = "OOM"
            break
        else:
            raise

# Summary
print(f"\n{'='*60}")
print("SCALING RESULTS SUMMARY")
print(f"{'='*60}")
for users, acc in results.items():
    print(f"  {users:>6} users: {acc}")
print(f"{'='*60}")
