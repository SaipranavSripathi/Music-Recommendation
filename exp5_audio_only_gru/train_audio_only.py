"""
Experiment 5: Audio-Only GRU (Controlled Baseline)
GRU4Rec + Linear Attention using only 128-dim SVD embeddings.
Proper session continuity across files. Top 30k users, 30 files.
Result: R@10 = 4.48%, NDCG@10 = 4.21%
Evaluated on sliding-window protocol: 5-session context → predict next session's first track.
"""

import os, math, time, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "/Users/spartan/Desktop/Music Recommendation/deezer-recsys25"
N_SESSION_FILES = 30
MAX_USERS       = 30000
MIN_SESSIONS    = 6
SEQ_LEN         = 5
MAX_SEQ_TRACKS  = 50

HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 256
EPOCHS      = 15
LR          = 0.001
SVD_DIM     = 128

# ── Load SVD embeddings ───────────────────────────────────────────────────────
print("Loading SVD embeddings...")
emb_files  = sorted(Path(f"{DATA_PATH}/track_embeddings").glob("svd_audio_*"))
raw        = pd.concat([pd.read_parquet(f) for f in emb_files], ignore_index=True)

def extract_svd(x):
    if isinstance(x, dict) and 'list' in x:
        return np.array([item['item'] for item in x['list'][:SVD_DIM]], dtype=np.float32)
    return np.array(x, dtype=np.float32)[:SVD_DIM]

svd_lookup = {row['track_id']: extract_svd(row['svd']) for _, row in raw.iterrows()}
print(f"SVD lookup: {len(svd_lookup)} tracks")
del raw; gc.collect()

# ── Two-pass session loading (proper continuity across files) ─────────────────
print(f"Pass 1: counting sessions across {N_SESSION_FILES} files...")
sess_files = sorted(Path(f"{DATA_PATH}/user_sessions").glob("sessions_*"))[:N_SESSION_FILES]

user_sess_count = defaultdict(set)
for f in sess_files:
    df = pd.read_parquet(f, columns=['user_id', 'session_id'])
    for uid, sid in zip(df['user_id'], df['session_id']):
        user_sess_count[uid].add(sid)
    del df; gc.collect()

qualified = {u for u, s in user_sess_count.items() if len(s) >= MIN_SESSIONS}
top_users = set(sorted(qualified, key=lambda u: len(user_sess_count[u]), reverse=True)[:MAX_USERS])
print(f"Qualified users: {len(top_users)}")
del user_sess_count; gc.collect()

print("Pass 2: loading full event data for selected users...")
dfs = []
for f in sess_files:
    df = pd.read_parquet(f)
    df = df[df['user_id'].isin(top_users)]
    if len(df): dfs.append(df)
    del df; gc.collect()

full_df    = pd.concat(dfs, ignore_index=True)
full_df.sort_values('ts', inplace=True)
sess_order = full_df.groupby(['user_id','session_id'])['ts'].min().rename('min_ts').reset_index()
sess_tracks= full_df.groupby(['user_id','session_id'])['track_id'].apply(list).reset_index()
merged     = sess_tracks.merge(sess_order, on=['user_id','session_id'])
merged.sort_values(['user_id','min_ts'], inplace=True)
del full_df, dfs; gc.collect()

user_sessions = defaultdict(list)
for _, row in merged.iterrows():
    user_sessions[row['user_id']].append(row['track_id'])
del merged; gc.collect()

# ── Build samples ─────────────────────────────────────────────────────────────
samples, all_tracks = [], set()
for u, sessions in user_sessions.items():
    if len(sessions) < SEQ_LEN + 1: continue
    for i in range(len(sessions) - SEQ_LEN):
        flat_seq  = [t for sess in sessions[i:i+SEQ_LEN] for t in sess]
        tgt_track = sessions[i+SEQ_LEN][0]
        all_tracks.update(flat_seq); all_tracks.add(tgt_track)
        samples.append((flat_seq, tgt_track))

track_list = sorted(all_tracks)
track2idx  = {t: i+1 for i, t in enumerate(track_list)}
NUM_TRACKS = len(track_list) + 1
print(f"Samples: {len(samples):,}  |  Vocab: {NUM_TRACKS}")

svd_matrix = np.zeros((NUM_TRACKS, SVD_DIM), dtype=np.float32)
for tid, idx in track2idx.items():
    if tid in svd_lookup:
        svd_matrix[idx] = svd_lookup[tid]

conv_samples = [([track2idx.get(t, 0) for t in seq], track2idx.get(tgt, 0))
                for seq, tgt in samples]
del user_sessions; gc.collect()

# ── Dataset ───────────────────────────────────────────────────────────────────
class SessionDataset(Dataset):
    def __init__(self, samples, max_len):
        self.samples = samples; self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        seq, tgt = self.samples[idx]
        seq = seq[-self.max_len:] if len(seq) > self.max_len else seq + [0]*(self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

n       = len(conv_samples)
n_train = int(0.8*n); n_val = int(0.1*n); n_test = n - n_train - n_val
train_ds, val_ds, test_ds = torch.utils.data.random_split(
    SessionDataset(conv_samples, MAX_SEQ_TRACKS), [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)

    def forward(self, x):
        w = torch.softmax(self.W(x), dim=1)
        return (w * x).sum(dim=1)


class AudioOnlyGRU(nn.Module):
    """GRU4Rec + Linear Attention, SVD embeddings only."""
    def __init__(self, svd_matrix, hidden_dim, num_layers, dropout):
        super().__init__()
        n, d = svd_matrix.shape
        self.embedding = nn.Embedding(n, d, padding_idx=0)
        self.embedding.weight.data.copy_(torch.FloatTensor(svd_matrix))

        self.pos_enc = PositionalEncoding(d)
        self.gru     = nn.GRU(d, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn    = LinearAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, n)

    def forward(self, x):
        emb     = self.pos_enc(self.embedding(x))
        out, _  = self.gru(emb)
        context = self.attn(out)
        return self.fc(self.dropout(context))

model = AudioOnlyGRU(svd_matrix, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, ks=(10, 20)):
    model.eval()
    hits = {k: 0 for k in ks}; dcg = {k: 0.0 for k in ks}; total = 0
    with torch.no_grad():
        for seq, tgt in loader:
            seq, tgt = seq.to(device), tgt.to(device)
            logits = model(seq)
            for k in ks:
                topk = logits.topk(k, dim=1).indices
                for j in range(len(tgt)):
                    t = tgt[j].item(); row = topk[j].tolist()
                    if t in row:
                        hits[k] += 1
                        dcg[k]  += 1.0 / math.log2(row.index(t) + 2)
            total += len(tgt)
    r  = {f"R@{k}":    hits[k]/total*100 for k in ks}
    nd = {f"NDCG@{k}": dcg[k]/total*100  for k in ks}
    return {**r, **nd}

# ── Train ─────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_val = 0.0
for epoch in range(1, EPOCHS+1):
    model.train(); total_loss = 0
    for seq, tgt in train_loader:
        seq, tgt = seq.to(device), tgt.to(device)
        optimizer.zero_grad()
        loss = criterion(model(seq), tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    metrics = evaluate(model, val_loader)
    val_r10 = metrics['R@10']
    if val_r10 > best_val:
        best_val = val_r10
        torch.save(model.state_dict(), "best_audio_only.pt")
    print(f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | val R@10={val_r10:.2f}%")

model.load_state_dict(torch.load("best_audio_only.pt"))
test_metrics = evaluate(model, test_loader)
print(f"\nTest  R@10={test_metrics['R@10']:.2f}%  NDCG@10={test_metrics['NDCG@10']:.2f}%")
