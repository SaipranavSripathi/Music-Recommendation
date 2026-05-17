"""
Experiment 6: Multimodal GRU with Learned Per-Dimension Fusion Gate
Extends Exp 5 with a second branch for 1024-dim raw audio features.
Gate α ∈ R^128 (per-dimension sigmoid) fuses SVD and audio.
Audio embedding table (27M params) is frozen during training.
Result: R@10 = 4.34%, NDCG@10 = 4.08%  |  gate ᾱ ≈ 0.74 (SVD dominant)
"""

import os, math, time, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
AUDIO_DIM   = 1024
PROJ_DIM    = 128   # project audio down to match SVD dim

# ── Load embeddings ───────────────────────────────────────────────────────────
print("Loading track embeddings (SVD + audio)...")
emb_files  = sorted(Path(f"{DATA_PATH}/track_embeddings").glob("svd_audio_*"))
raw        = pd.concat([pd.read_parquet(f) for f in emb_files], ignore_index=True)

def extract_svd(x):
    if isinstance(x, dict) and 'list' in x:
        return np.array([item['item'] for item in x['list'][:SVD_DIM]], dtype=np.float32)
    return np.array(x, dtype=np.float32)[:SVD_DIM]

svd_lookup   = {row['track_id']: extract_svd(row['svd']) for _, row in raw.iterrows()}
audio_lookup = {row['track_id']: np.array(row['audio'], dtype=np.float32)[:AUDIO_DIM]
                for _, row in raw.iterrows()}
print(f"SVD: {len(svd_lookup)}  Audio: {len(audio_lookup)}")
del raw; gc.collect()

# ── Two-pass session loading ──────────────────────────────────────────────────
print(f"Pass 1: scanning {N_SESSION_FILES} files...")
sess_files = sorted(Path(f"{DATA_PATH}/user_sessions").glob("sessions_*"))[:N_SESSION_FILES]

user_sess_count = defaultdict(set)
for f in sess_files:
    df = pd.read_parquet(f, columns=['user_id', 'session_id'])
    for uid, sid in zip(df['user_id'], df['session_id']):
        user_sess_count[uid].add(sid)
    del df; gc.collect()

qualified = {u for u, s in user_sess_count.items() if len(s) >= MIN_SESSIONS}
top_users = set(sorted(qualified, key=lambda u: len(user_sess_count[u]), reverse=True)[:MAX_USERS])
print(f"Selected {len(top_users)} users")
del user_sess_count; gc.collect()

print("Pass 2: loading events...")
dfs = []
for f in sess_files:
    df = pd.read_parquet(f)
    df = df[df['user_id'].isin(top_users)]
    if len(df): dfs.append(df)
    del df; gc.collect()

full_df     = pd.concat(dfs, ignore_index=True)
full_df.sort_values('ts', inplace=True)
sess_order  = full_df.groupby(['user_id','session_id'])['ts'].min().rename('min_ts').reset_index()
sess_tracks = full_df.groupby(['user_id','session_id'])['track_id'].apply(list).reset_index()
merged      = sess_tracks.merge(sess_order, on=['user_id','session_id'])
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

svd_matrix   = np.zeros((NUM_TRACKS, SVD_DIM),   dtype=np.float32)
audio_matrix = np.zeros((NUM_TRACKS, AUDIO_DIM),  dtype=np.float32)
for tid, idx in track2idx.items():
    if tid in svd_lookup:   svd_matrix[idx]   = svd_lookup[tid]
    if tid in audio_lookup: audio_matrix[idx] = audio_lookup[tid]

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


class MultimodalGRU(nn.Module):
    """
    GRU4Rec + Linear Attention with per-dimension learned fusion gate.
    h = α ⊙ e_SVD + (1 - α) ⊙ e_audio
    Audio embedding table is frozen; MLP projects 1024 → 128.
    """
    def __init__(self, svd_matrix, audio_matrix, hidden_dim, num_layers, dropout, proj_dim):
        super().__init__()
        n, svd_d   = svd_matrix.shape
        _, audio_d = audio_matrix.shape

        # SVD branch (trainable)
        self.svd_emb = nn.Embedding(n, svd_d, padding_idx=0)
        self.svd_emb.weight.data.copy_(torch.FloatTensor(svd_matrix))

        # Audio branch (frozen)
        self.audio_emb = nn.Embedding(n, audio_d, padding_idx=0)
        self.audio_emb.weight.data.copy_(torch.FloatTensor(audio_matrix))
        self.audio_emb.weight.requires_grad = False

        # MLP projection: 1024 → 256 → 128 + LayerNorm
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_d, 256), nn.ReLU(),
            nn.Linear(256, proj_dim), nn.LayerNorm(proj_dim)
        )

        # Per-dimension gate: α ∈ R^proj_dim, initialized at 0.5
        self.gate = nn.Parameter(torch.full((proj_dim,), 0.0))  # sigmoid(0)=0.5

        self.pos_enc = PositionalEncoding(proj_dim)
        self.gru     = nn.GRU(proj_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn    = LinearAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, n)

    def forward(self, x):
        # SVD branch
        e_svd   = F.normalize(self.svd_emb(x), dim=-1)            # (B, T, 128)

        # Audio branch
        e_audio = self.audio_proj(self.audio_emb(x))               # (B, T, 128)
        e_audio = F.normalize(e_audio, dim=-1)

        # Per-dimension fusion gate
        alpha   = torch.sigmoid(self.gate)                          # (128,)
        fused   = alpha * e_svd + (1 - alpha) * e_audio            # (B, T, 128)

        out, _  = self.gru(self.pos_enc(fused))
        context = self.attn(out)
        return self.fc(self.dropout(context))

    def gate_value(self):
        return torch.sigmoid(self.gate).mean().item()

model = MultimodalGRU(svd_matrix, audio_matrix, HIDDEN_DIM, NUM_LAYERS, DROPOUT, PROJ_DIM).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Params — trainable: {trainable:,}  /  total: {total:,}")

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
    alpha_mean = model.gate_value()
    if val_r10 > best_val:
        best_val = val_r10
        torch.save(model.state_dict(), "best_multimodal.pt")
    print(f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | "
          f"val R@10={val_r10:.2f}% | gate ᾱ={alpha_mean:.3f}")

model.load_state_dict(torch.load("best_multimodal.pt"))
test_metrics = evaluate(model, test_loader)
print(f"\nTest  R@10={test_metrics['R@10']:.2f}%  NDCG@10={test_metrics['NDCG@10']:.2f}%")
print(f"Learned gate ᾱ = {model.gate_value():.3f}  (SVD weight)")
