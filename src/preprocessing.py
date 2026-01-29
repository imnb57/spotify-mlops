import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
import os

class SpotifyPreprocessor:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.feature_columns = [
            'popularity', 'duration_ms', 'danceability', 'energy', 
            'key', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Loads data from CSV."""
        print(f"Loading data from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        return df

    def clean_data(self, df):
        """Removes duplicates and handles missing values."""
        print("Cleaning data...")
        # Drop duplicates based on track_id
        df = df.drop_duplicates(subset=['track_id'])
        
        # Drop rows with missing values in critical columns
        df = df.dropna(subset=self.feature_columns)
        
        # Handle explicitly if needed (convert to int)
        if 'explicit' in df.columns:
            df['explicit'] = df['explicit'].astype(int)
            if 'explicit' not in self.feature_columns:
                self.feature_columns.append('explicit')
                
        return df.reset_index(drop=True)

    def process(self):
        """Executes the full preprocessing pipeline."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        df = self.load_data()
        df = self.clean_data(df)
        
        print("Scaling features...")
        # Scale numerical features
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        
        # Save processed data
        output_path = os.path.join(self.output_dir, 'processed_data.csv')
        df.to_csv(output_path, index=False)
        
        # Save scaler
        scaler_path = os.path.join(self.output_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Preprocessing complete. Data saved to {output_path}")
        print(f"Scaler saved to {scaler_path}")
        return df

if __name__ == "__main__":
    input_csv = os.path.join("data", "raw", "spotifyDataset.csv")
    output_folder = os.path.join("data", "processed")
    
    preprocessor = SpotifyPreprocessor(input_csv, output_folder)
    preprocessor.process()
