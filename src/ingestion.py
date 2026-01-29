import pandas as pd
import os
from spotify_client import SpotifyClient
import argparse

def ingest_data(limit=50, output_path="data/raw/spotifyDataset.csv"):
    client = SpotifyClient()
    new_df = client.fetch_new_releases(limit=limit)
    
    if new_df.empty:
        print("No new data fetched.")
        return

    if os.path.exists(output_path):
        print(f"Appending to {output_path}...")
        # Check for duplicates? For now, just append.
        # Ideally we load existing, append, drop dupes, save.
        existing_df = pd.read_csv(output_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Drop duplicates based on track_id
        combined = combined.drop_duplicates(subset=['track_id'])
        combined.to_csv(output_path, index=False)
        print(f"Total tracks: {len(combined)}")
    else:
        new_df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {len(new_df)} tracks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    
    ingest_data(limit=args.limit)
