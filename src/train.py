import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
import os

class SpotifyRecommender:
    def __init__(self, data_path, model_dir):
        self.data_path = data_path
        self.model_dir = model_dir
        self.feature_columns = [
            'popularity', 'duration_ms', 'danceability', 'energy', 
            'key', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo', 'explicit'
        ]
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        
    def load_data(self):
        print("Loading processed data...")
        df = pd.read_csv(self.data_path)
        return df

    def train(self):
        self.train_memory()

    def train_memory(self):
        """Trains and returns the model object without necessarily saving (though we do save)."""
        df = self.load_data()
        
        # --- Feedback Integration ---
        feedback_path = "data/feedback_data.csv"
        if os.path.exists(feedback_path):
            print("Incorporating user feedback...")
            try:
                feedback_df = pd.read_csv(feedback_path, names=['track_id', 'liked'])
                liked_tracks = feedback_df[feedback_df['liked'] == True]['track_id'].unique()
                
                if len(liked_tracks) > 0:
                    print(f"Boosting popularity for {len(liked_tracks)} liked tracks...")
                    if 'track_id' in df.columns and 'popularity' in df.columns:
                         mask = df['track_id'].isin(liked_tracks)
                         df.loc[mask, 'popularity'] = df.loc[mask, 'popularity'] + 0.2
                         df.loc[df['popularity'] > 1.0, 'popularity'] = 1.0
            except Exception as e:
                print(f"Error loading feedback: {e}")
        # ----------------------------

        train_cols = [c for c in self.feature_columns if c in df.columns]
        
        print(f"Training model with features: {train_cols}")
        X = df[train_cols].values
        
        print("Fitting NearestNeighbours model...")
        self.model.fit(X)
        
        # Save artifacts (still good for persistence)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        model_path = os.path.join(self.model_dir, 'recommender_model.joblib')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        joblib.dump(train_cols, os.path.join(self.model_dir, 'feature_columns.joblib'))
        
        with open(os.path.join(self.model_dir, 'version.txt'), 'w') as f:
            f.write("v1.1.0")
            
        return self.model, train_cols

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "processed_data.csv")
    model_dir = os.path.join("models")
    
    trainer = SpotifyRecommender(data_path, model_dir)
    trainer.train()
