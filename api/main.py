from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from typing import List, Optional
from src.model_manager import ModelManager

app = FastAPI(title="Spotify Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Manager ---
manager = ModelManager()

class Song(BaseModel):
    track_id: str
    track_name: str
    artists: str
    album_name: str
    album_cover: Optional[str] = None
    url: Optional[str] = None
    reason: Optional[str] = None # New field for explainability

@app.get("/")
def home():
    return {"message": "Spotify Recommender API is running (V4.0 Search & Discovery)"}

@app.get("/search", response_model=List[Song])
def search(q: str, limit: int = 5):
    _, _, data = manager.get_active_model()
    if data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    results = data[data['track_name'].str.contains(q, case=False, na=False) | 
                   data['artists'].str.contains(q, case=False, na=False)]
    results = results.head(limit)
    
    songs = []
    for _, row in results.iterrows():
        try:
            songs.append(Song(
                track_id=row['track_id'],
                track_name=row['track_name'],
                artists=row['artists'],
                album_name=row['album_name'],
                album_cover=row.get('album_cover', None)
            ))
        except Exception as e:
            continue
            
    return songs

class Feedback(BaseModel):
    track_id: str
    liked: bool

@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    print(f"Received feedback: {feedback}")
    
    # Save Feedback Only - No Retraining Here
    feedback_file = "data/feedback_data.csv"
    new_data = pd.DataFrame([feedback.dict()])
    
    if os.path.exists(feedback_file):
        new_data.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        new_data.to_csv(feedback_file, index=False)
        
    return {"status": "success", "message": "Feedback recorded."}

@app.post("/control/retrain")
def retrain_model():
    """Manually triggers retraining."""
    if manager.is_training:
        return {"status": "busy", "message": "Training already in progress."}
    
    success = manager.trigger_retraining()
    if success:
        return {"status": "success", "message": "Training started."}
    else:
         raise HTTPException(status_code=500, detail="Failed to start training.")

@app.post("/reset")
def reset_profile():
    success = manager.reset_baseline()
    if success:
        return {"status": "success", "message": "Profile reset to baseline."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reset profile.")

@app.get("/model-info")
def get_model_info():
    model, features, data = manager.get_active_model()
    return {
        "version": manager.model_version,
        "features": features,
        "is_training": manager.is_training,
        "data_size": len(data) if data is not None else 0
    }

@app.get("/recommend/home", response_model=List[Song])
def recommend_home(
    limit: int = 10, 
    target_energy: Optional[float] = None,
    target_danceability: Optional[float] = None,
    target_valence: Optional[float] = None,
    target_acousticness: Optional[float] = None
):
    """Smart Home Feed with optional feature tuning."""
    
    overrides = {}
    if target_energy is not None: overrides['energy'] = target_energy
    if target_danceability is not None: overrides['danceability'] = target_danceability
    if target_valence is not None: overrides['valence'] = target_valence
    if target_acousticness is not None: overrides['acousticness'] = target_acousticness
    
    recs = manager.recommend_home(limit=limit, feature_overrides=overrides)
    
    songs = []
    for item in recs:
        try:
            songs.append(Song(
                track_id=item['track_id'],
                track_name=item['track_name'],
                artists=item['artists'],
                album_name=item['album_name'],
                album_cover=item.get('album_cover', None),
                reason=item.get('reason', None)
            ))
        except Exception:
            continue
    return songs

@app.get("/recommend/radio/{track_id}", response_model=List[Song])
def recommend_radio(track_id: str, limit: int = 10):
    """Item-to-Item recommendation (Radio)."""
    recs = manager.recommend_radio(seed_track_id=track_id, limit=limit)
    
    songs = []
    for item in recs:
        try:
            songs.append(Song(
                track_id=item['track_id'],
                track_name=item['track_name'],
                artists=item['artists'],
                album_name=item['album_name'],
                album_cover=item.get('album_cover', None),
                reason=item.get('reason', "Similar Vibe")
            ))
        except Exception:
            continue
    return songs

@app.get("/recommend/{track_id}", response_model=List[Song])
def recommend(track_id: str, limit: int = 5):
    # Standard item-to-item (legacy endpoint, maybe keep for PDP)
    # Re-using the logic from manager for consistency could be better, 
    # but sticking to original impl for now to avoid breaking changes if any.
    return recommend_radio(track_id, limit)
