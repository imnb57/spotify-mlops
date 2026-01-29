from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="Spotify Recommender API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and data
model = None
data = None
feature_columns = None
scaler = None

class Song(BaseModel):
    track_id: str
    track_name: str
    artists: str
    album_name: str
    popularity: int
    duration_ms: int
    explicit: bool
    album_cover: Optional[str] = None # Placeholder if we had it

@app.on_event("startup")
def load_artifacts():
    global model, data, feature_columns, scaler
    try:
        print("Loading artifacts...")
        # Load data (use raw or processed? Raw has metadata, processed has features)
        # We need raw for display info, and processed features for inference if new data comes in.
        # But for this simple recommender, we might just need the dataframe and the lookups.
        
        # Let's load the raw data for metadata
        raw_data_path = "data/raw/spotifyDataset.csv"
        if os.path.exists(raw_data_path):
            data = pd.read_csv(raw_data_path)
            # Drop duplicates to match training
            data = data.drop_duplicates(subset=['track_id']).reset_index(drop=True)
            print(f"Loaded data with {len(data)} records")
        
        # Load model
        model_path = "models/recommender_model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded")
        
        # Load feature cols
        features_path = "models/feature_columns.joblib"
        if os.path.exists(features_path):
            feature_columns = joblib.load(features_path)
            
        processed_data_path = "data/processed/processed_data.csv"
        if os.path.exists(processed_data_path):
            processed_data = pd.read_csv(processed_data_path)
            # Create a matrix for lookup
            global features_matrix
            if feature_columns:
                existing_cols = [c for c in feature_columns if c in processed_data.columns]
                features_matrix = processed_data[existing_cols].values
                print("Features matrix loaded")
                
    except Exception as e:
        print(f"Error loading artifacts: {e}")

@app.get("/")
def root():
    return {"message": "Spotify Recommender API is running"}

@app.get("/search", response_model=List[Song])
def search_songs(q: str = Query(..., min_length=1), limit: int = 10):
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Simple substring search
    results = data[
        data['track_name'].str.contains(q, case=False, na=False) | 
        data['artists'].str.contains(q, case=False, na=False)
    ].head(limit)
    
    songs = []
    for _, row in results.iterrows():
        songs.append(Song(
            track_id=row['track_id'],
            track_name=row['track_name'],
            artists=row['artists'],
            album_name=row['album_name'],
            popularity=row['popularity'],
            duration_ms=row['duration_ms'],
            explicit=bool(row['explicit'])
        ))
    return songs

@app.get("/recommend/{track_id}", response_model=List[Song])
def recommend(track_id: str, limit: int = 5):
    if model is None or data is None or 'features_matrix' not in globals():
        raise HTTPException(status_code=503, detail="System not ready")
        
    # Find index of track
    idx_list = data.index[data['track_id'] == track_id].tolist()
    if not idx_list:
        raise HTTPException(status_code=404, detail="Track not found")
    
    idx = idx_list[0]
    
    # Get features
    # features = features_matrix[idx].reshape(1, -1)
    
    # Find neighbors
    # distances, indices = model.kneighbors(features, n_neighbors=limit+1)
    
    # Optimize: Pre-computed model allows finding neighbors
    distances, indices = model.kneighbors(features_matrix[idx].reshape(1, -1), n_neighbors=limit+1)
    
    recommended_songs = []
    for i in range(1, len(indices[0])): # Skip self
        neighbor_idx = indices[0][i]
        row = data.iloc[neighbor_idx]
        recommended_songs.append(Song(
            track_id=row['track_id'],
            track_name=row['track_name'],
            artists=row['artists'],
            album_name=row['album_name'],
            popularity=row['popularity'],
            duration_ms=row['duration_ms'],
            explicit=bool(row['explicit'])
        ))
        
    return recommended_songs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
