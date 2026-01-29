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
        df = self.load_data()
        
        # Ensure only numeric columns are used for training
        # We assume they are already scaled from preprocessing
        # Filter for existing columns
        train_cols = [c for c in self.feature_columns if c in df.columns]
        
        print(f"Training model with features: {train_cols}")
        X = df[train_cols].values
        
        print("Fitting NearestNeighbours model...")
        self.model.fit(X)
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        model_path = os.path.join(self.model_dir, 'recommender_model.joblib')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save feature columns used
        joblib.dump(train_cols, os.path.join(self.model_dir, 'feature_columns.joblib'))
        
        # Save track_ids reference if needed, but the API will just load the main dataset
        # to map index to metadata. We assume the index corresponds to the processed dataframe.

if __name__ == "__main__":
    data_path = os.path.join("data", "processed", "processed_data.csv")
    model_dir = os.path.join("models")
    
    trainer = SpotifyRecommender(data_path, model_dir)
    trainer.train()
