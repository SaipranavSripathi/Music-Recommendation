"""
Experiment 1: LSTM Baseline
Sequential next-track recommendation using a simple LSTM over 128-dim SVD embeddings.
Sessions are loaded independently per file (fragmented) — this is the naive baseline.
Result: HR@10 ≈ 4.97%
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
DATA_PATH      = "/Users/spartan/Desktop/Music Recommendation/deezer-recsys25"
N_SESSION_FILES = 20       # subset of 500 files
MAX_USERS       = 10000
MIN_SESSIONS    = 3
SEQ_LEN         = 5        # sessions in context window
MAX_SEQ_TRACKS  = 50       # max tracks fed to LSTM

HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 256
EPOCHS      = 15
LR          = 0.001
SVD_DIM     = 128

# ── Load SVD embeddings ───────────────────────────────────────────────────────
print("Loading track embeddings...")
emb_files  = sorted(Path(f"{DATA_PATH}/track_embeddings").glob("svd_audio_*"))
raw        = pd.concat([pd.read_parquet(f) for f in emb_files], ignore_index=True)

def extract_svd(x):
    if isinstance(x, dict) and 'list' in x:
        return np.array([item['item'] for item in x['list'][:SVD_DIM]], dtype=np.float32)
    return np.array(x, dtype=np.float32)[:SVD_DIM]

svd_lookup = {row['track_id']: extract_svd(row['svd']) for _, row in raw.iterrows()}
print(f"SVD lookup: {len(svd_lookup)} tracks")
del raw; gc.collect()

# ── Load sessions (fragmented — intentionally naive for Exp 1) ────────────────
print(f"Loading {N_SESSION_FILES} session files...")
sess_files   = sorted(Path(f"{DATA_PATH}/user_sessions").glob("sessions_*"))[:N_SESSION_FILES]
user_sessions = defaultdict(list)

for f in sess_files:
    df = pd.read_parquet(f).sort_values('ts')
    for (u, s), g in df.groupby(['user_id', 'session_id'], sort=False):
        tracks = g.sort_values('ts')['track_id'].tolist()
        if tracks:
            user_sessions[u].append(tracks)
    del df; gc.collect()

# Select top users by session count
user_counts  = {u: len(s) for u, s in user_sessions.items() if len(s) >= MIN_SESSIONS}
selected     = sorted(user_counts, key=user_counts.get, reverse=True)[:MAX_USERS]
print(f"Selected {len(selected)} users")

# ── Build samples ─────────────────────────────────────────────────────────────
samples, all_tracks = [], set()
for u in selected:
    sessions = user_sessions[u]
    for i in range(len(sessions) - SEQ_LEN):
        flat_seq  = [t for sess in sessions[i:i+SEQ_LEN] for t in sess]
        tgt_track = sessions[i+SEQ_LEN][0]
        all_tracks.update(flat_seq); all_tracks.add(tgt_track)
        samples.append((flat_seq, tgt_track))

track_list = sorted(all_tracks)
track2idx  = {t: i+1 for i, t in enumerate(track_list)}
NUM_TRACKS = len(track_list) + 1
print(f"Samples: {len(samples):,}  |  Vocab: {NUM_TRACKS}")

# Embedding matrix
svd_matrix = np.zeros((NUM_TRACKS, SVD_DIM), dtype=np.float32)
for tid, idx in track2idx.items():
    if tid in svd_lookup:
        svd_matrix[idx] = svd_lookup[tid]

# Convert track IDs to indices
conv = [(([track2idx.get(t, 0) for t in seq], track2idx.get(tgt, 0))
         for seq, tgt in samples)]
conv_samples = [(([track2idx.get(t, 0) for t in seq], track2idx.get(tgt, 0)))
                for seq, tgt in samples]

del user_sessions; gc.collect()

# ── Dataset ───────────────────────────────────────────────────────────────────
class SessionDataset(Dataset):
    def __init__(self, samples, max_len):
        self.samples = samples
        self.max_len = max_len

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
class LSTMRecommender(nn.Module):
    def __init__(self, svd_matrix, hidden_dim, num_layers, dropout):
        super().__init__()
        n, d = svd_matrix.shape
        self.embedding = nn.Embedding(n, d, padding_idx=0)
        self.embedding.weight.data.copy_(torch.FloatTensor(svd_matrix))

        self.lstm = nn.LSTM(d, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, n)

    def forward(self, x):
        emb = self.embedding(x)                   # (B, T, D)
        out, _ = self.lstm(emb)                   # (B, T, H)
        h   = self.dropout(out[:, -1, :])         # last hidden state
        return self.fc(h)                          # (B, vocab)

model = LSTMRecommender(svd_matrix, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Train & Evaluate ──────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval(); hits10 = 0; total = 0
    with torch.no_grad():
        for seq, tgt in loader:
            seq, tgt = seq.to(device), tgt.to(device)
            top10 = model(seq).topk(10, dim=1).indices
            hits10 += sum(tgt[j].item() in top10[j].tolist() for j in range(len(tgt)))
            total  += len(tgt)
    return hits10 / total * 100

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

    val_hr = evaluate(model, val_loader)
    if val_hr > best_val:
        best_val = val_hr
        torch.save(model.state_dict(), "best_lstm.pt")
    print(f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} | val HR@10={val_hr:.2f}%")

model.load_state_dict(torch.load("best_lstm.pt"))
test_hr = evaluate(model, test_loader)
print(f"\nTest HR@10 = {test_hr:.2f}%")
