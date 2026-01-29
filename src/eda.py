import pandas as pd
import os

def run_eda(filepath):
    print(f"--- EDA for {filepath} ---")
    if not os.path.exists(filepath):
        print("File not found.")
        return

    df = pd.read_csv(filepath)
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for duplicates on track_id
    dup_ids = df.duplicated(subset=['track_id']).sum()
    print(f"Duplicate track_ids: {dup_ids}")
    
    # Check for duplicates on artist + track_name (same song, different version/album?)
    # Normalize strings for comparison
    df['name_norm'] = df['track_name'].str.lower().str.strip()
    df['artist_norm'] = df['artists'].str.lower().str.strip()
    
    dup_names = df.duplicated(subset=['name_norm', 'artist_norm']).sum()
    print(f"Duplicate Name+Artist pairs: {dup_names}")
    
    # Check for nulls
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Show example duplicates if any
    if dup_names > 0:
        print("\nExample Duplicates (Name+Artist):")
        dups = df[df.duplicated(subset=['name_norm', 'artist_norm'], keep=False)].sort_values(by=['name_norm'])
        print(dups[['track_id', 'track_name', 'artists', 'album_name']].head(10))

if __name__ == "__main__":
    run_eda("data/raw/spotifyDataset.csv")
    print("\n" + "="*30 + "\n")
    if os.path.exists("data/processed/processed_data.csv"):
        run_eda("data/processed/processed_data.csv")
