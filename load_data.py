"""
Data Loader for AU2ACTR Model

Loads user sessions from Deezer dataset parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import os
import gc
import time

DATA_DIR = Path("D:/CS274/deezer-recsys25/deezer-recsys25/user_sessions")
EMBEDDINGS_DIR = Path("D:/CS274/deezer-recsys25/deezer-recsys25")
OUTPUT_DIR = Path("cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARQUET_FILES = sorted(os.listdir(DATA_DIR))
print(f"Found {len(PARQUET_FILES)} parquet files")


def load_user_session_counts():
    """Load pre-computed session counts"""
    counts_path = EMBEDDINGS_DIR / "user_session_counts.pkl"
    if counts_path.exists():
        with open(counts_path, "rb") as f:
            return pickle.load(f)
    return {}


def load_user_sessions(min_sessions=5, max_sessions=100, max_users=None):
    """
    Load user sessions from parquet files.
    
    Args:
        min_sessions: Minimum number of sessions per user
        max_sessions: Maximum number of sessions per user
        max_users: Maximum number of users to load (None for all)
    
    Returns:
        user_sessions: dict mapping user_id -> list of session dicts
    """
    user_counts = load_user_session_counts()
    print(f"Total users in dataset: {len(user_counts)}")
    
    target_users = []
    for user_id, count in user_counts.items():
        if min_sessions <= count <= max_sessions:
            target_users.append(user_id)
    
    print(f"Users with {min_sessions}-{max_sessions} sessions: {len(target_users)}")
    
    if max_users and len(target_users) > max_users:
        np.random.seed(42)
        indices = np.random.choice(len(target_users), max_users, replace=False)
        target_users = [target_users[i] for i in indices]
        print(f"Sampled to: {len(target_users)} users")
    
    target_set = set(target_users)
    
    user_sessions_raw = defaultdict(list)
    
    print("\nLoading sessions from parquet files...")
    start_time = time.time()
    
    for i, fname in enumerate(PARQUET_FILES):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Processing file {i+1}/{len(PARQUET_FILES)} ({elapsed:.1f}s elapsed)")
        
        try:
            df = pd.read_parquet(DATA_DIR / fname)
            df = df[df['user_id'].isin(target_set)]
            
            if len(df) == 0:
                continue
            
            df = df.sort_values('ts')
            
            for (user_id, session_id), group in df.groupby(['user_id', 'session_id']):
                tracks = sorted(set(group['track_id'].tolist()))
                ts = group['ts'].iloc[0]
                
                existing_sessions = user_sessions_raw[user_id]
                if len(existing_sessions) > 0:
                    last_ts = existing_sessions[-1]['context']['ts']
                    time_since_last = ts - last_ts
                else:
                    time_since_last = 0
                
                session_data = {
                    'session_id': session_id,
                    'context': {
                        'time_since_last_session': time_since_last,
                        'ts': ts,
                        'day_of_week': pd.to_datetime(ts).dayofweek,
                        'hour_of_day': pd.to_datetime(ts).hour
                    },
                    'track_ids': tracks
                }
                user_sessions_raw[user_id].append(session_data)
                
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    
    for user_id, sessions in user_sessions_raw.items():
        sessions.sort(key=lambda x: x['context']['ts'])
    
    print(f"\nLoaded {len(user_sessions_raw)} users")
    
    total_sessions = sum(len(s) for s in user_sessions_raw.values())
    print(f"Total sessions: {total_sessions}")
    
    return dict(user_sessions_raw)


def load_track_embeddings():
    """Load track audio embeddings"""
    emb_path = EMBEDDINGS_DIR / "track_embeddings_small.parquet"
    print(f"Loading embeddings from {emb_path}...")
    df = pd.read_parquet(emb_path)
    print(f"Loaded embeddings for {len(df)} tracks")
    return df


def prepare_track_mappings(user_sessions):
    """Create track to ID mappings"""
    all_tracks = set()
    for sessions in user_sessions.values():
        for session in sessions:
            all_tracks.update(session['track_ids'])
    
    track_list = sorted(all_tracks)
    track2id = {t: i+1 for i, t in enumerate(track_list)}
    num_tracks = len(track_list) + 1
    
    print(f"Unique tracks: {num_tracks}")
    return track2id, num_tracks


def create_embedding_matrix(track2id, num_tracks, embedding_dim=128):
    """Create embedding matrix from track embeddings"""
    emb_df = load_track_embeddings()
    
    embedding_matrix = np.random.randn(num_tracks, embedding_dim).astype(np.float32) * 0.01
    
    for _, row in emb_df.iterrows():
        track_id = row['track_id']
        if track_id in track2id:
            svd_emb = row['svd']
            if isinstance(svd_emb, dict):
                if 'list' in svd_emb:
                    emb_list = svd_emb['list']
                    emb_values = [x['item'] for x in emb_list[:embedding_dim]]
                else:
                    emb_values = list(svd_emb.values())[:embedding_dim]
            elif isinstance(svd_emb, str):
                svd_emb = eval(svd_emb)
                emb_values = list(svd_emb.values())[:embedding_dim] if isinstance(svd_emb, dict) else svd_emb[:embedding_dim]
            else:
                emb_values = svd_emb[:embedding_dim]
            
            if len(emb_values) < embedding_dim:
                emb_values = list(emb_values) + [0.0] * (embedding_dim - len(emb_values))
            else:
                emb_values = emb_values[:embedding_dim]
            
            idx = track2id[track_id]
            embedding_matrix[idx] = np.array(emb_values, dtype=np.float32)
    
    return embedding_matrix


if __name__ == "__main__":
    print("="*60)
    print("Loading data for AU2ACTR")
    print("="*60)
    
    MAX_USERS = 50000
    
    user_sessions = load_user_sessions(min_sessions=5, max_sessions=100, max_users=MAX_USERS)
    
    track2id, num_tracks = prepare_track_mappings(user_sessions)
    
    embedding_matrix = create_embedding_matrix(track2id, num_tracks)
    
    cache_file = OUTPUT_DIR / f"data_{MAX_USERS}k.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump({
            'user_sessions': user_sessions,
            'track2id': track2id,
            'num_tracks': num_tracks,
            'embedding_matrix': embedding_matrix
        }, f)
    
    print(f"\nData cached to {cache_file}")
