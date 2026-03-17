"""
Data Loader for AU2ACTR
Loads user sessions from Deezer dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import os
import time

# Configuration
DATA_PATH = "D:/CS274/deezer-recsys25/deezer-recsys25"  # Change this to your dataset path
MAX_USERS = 50000  # Number of users to load
OUTPUT_DIR = Path("cache")

def high_level_tempo_from_ts(ts):
    """Extract tempo features from timestamp"""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    return dt.weekday(), dt.hour

DATA_DIR = Path(DATA_PATH) / "user_sessions"
OUTPUT_CACHE = OUTPUT_DIR / "deezer" / "min5sess"
PARQUET_FILES = sorted(os.listdir(DATA_DIR))

print("Loading user session counts...")
with open(Path(DATA_PATH) / "user_session_counts.pkl", "rb") as f:
    user_counts = pickle.load(f)

print(f"Total users in dataset: {len(user_counts)}")

# Select users with 5-100 sessions
target_users = []
for user_id, count in user_counts.items():
    if 5 <= count <= 100:
        target_users.append(user_id)
        
print(f"Users with 5-100 sessions: {len(target_users)}")

# Sample
MAX_USERS = min(MAX_USERS, len(target_users))
np.random.seed(42)
indices = np.random.choice(len(target_users), MAX_USERS, replace=False)
target_users = [target_users[i] for i in indices]
print(f"Sampled to: {len(target_users)} users")

target_set = set(target_users)

# Load sessions
print("\nLoading sessions from parquet files...")
start_time = time.time()

user_sessions_raw = defaultdict(list)

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
            
            existing = user_sessions_raw[user_id]
            if len(existing) > 0:
                time_since_last = ts - existing[-1]['context']['ts']
            else:
                time_since_last = 0
            
            day_of_week, hour_of_day = high_level_tempo_from_ts(ts)
            
            session_data = {
                'session_id': session_id,
                'context': {
                    'time_since_last_session': time_since_last,
                    'ts': ts,
                    'day_of_week': day_of_week,
                    'hour_of_day': hour_of_day
                },
                'track_ids': tracks
            }
            user_sessions_raw[user_id].append(session_data)
            
    except Exception as e:
        print(f"Error loading {fname}: {e}")

# Sort sessions by timestamp
for user_id, sessions in user_sessions_raw.items():
    sorted_sessions = sorted(sessions, key=lambda x: x['context']['ts'])
    user_sessions_raw[user_id] = sorted_sessions

# Filter
filtered_sessions = {uid: sess for uid, sess in user_sessions_raw.items() if len(sess) >= 5}
print(f"Users with 5+ sessions: {len(filtered_sessions)}")

# Save
os.makedirs(OUTPUT_CACHE, exist_ok=True)
with open(OUTPUT_CACHE / "user_sessions.pkl", "wb") as f:
    pickle.dump(dict(filtered_sessions), f)

print(f"\nSaved to: {OUTPUT_CACHE / 'user_sessions.pkl'}")
print(f"Total time: {time.time() - start_time:.1f}s")
