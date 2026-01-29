import threading
import time
import pandas as pd
import numpy as np
import joblib
import os
from src.train import SpotifyRecommender
from src.preprocessing import SpotifyPreprocessor

class ModelManager:
    def __init__(self, model_dir="models", data_dir="data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.active_model = None
        self.feature_columns = None
        self.data = None # Store loaded data for recommendations
        self.model_version = "v1.0.0"
        self.is_training = False
        self.lock = threading.Lock()
        
        # Load initial resources
        self.load_resources()

    def load_resources(self):
        """Loads model and data from disk."""
        print("Loading resources...")
        try:
            # Load Data
            data_path = os.path.join(self.data_dir, "processed", "processed_data.csv")
            if os.path.exists(data_path):
                self.data = pd.read_csv(data_path)
            
            # Load Model
            model_path = os.path.join(self.model_dir, 'recommender_model.joblib')
            features_path = os.path.join(self.model_dir, 'feature_columns.joblib')
            version_path = os.path.join(self.model_dir, 'version.txt')

            if os.path.exists(model_path):
                self.active_model = joblib.load(model_path)
                
            if os.path.exists(features_path):
                self.feature_columns = joblib.load(features_path)
                
            if os.path.exists(version_path):
                with open(version_path, 'r') as f:
                    self.model_version = f.read().strip()
            
            print(f"Resources loaded. Version: {self.model_version}")
        except Exception as e:
            print(f"Error loading resources: {e}")

    def get_active_model(self):
        return self.active_model, self.feature_columns, self.data

    def recommend_home(self, limit=10, feature_overrides=None):
        """
        Smart Home Feed Logic:
        1. If no feedback -> Top Popularity (Cold Start).
        2. If feedback -> Mean Vector of Liked Songs -> KNN.
        3. Explainability -> Add tags based on features.
        """
        if self.data is None or self.active_model is None:
            return []

        # 1. Load Feedback
        feedback_path = os.path.join(self.data_dir, "feedback_data.csv")
        liked_track_ids = []
        if os.path.exists(feedback_path):
            try:
                fb_df = pd.read_csv(feedback_path, names=['track_id', 'liked'])
                # Robust check for True/False strings or booleans
                liked_track_ids = fb_df[fb_df['liked'].astype(str) == 'True']['track_id'].unique()
            except Exception:
                pass
        
        recs = []
        
        # Cold Start or No Likes
        if len(liked_track_ids) == 0:
            print("Home: Cold Start")
            if 'popularity' in self.data.columns:
                # Top Popular not already returned? Logic is simple for now.
                top_pop = self.data.sort_values(by='popularity', ascending=False).head(limit)
                recs = self._annotate_recs(top_pop, "Trending Now")
            else:
                recs = self._annotate_recs(self.data.head(limit), "Discover")
        else:
            print(f"Home: Personalized ({len(liked_track_ids)} likes)")
            # Get features of liked songs
            liked_features = self.data[self.data['track_id'].isin(liked_track_ids)][self.feature_columns]
            
            if len(liked_features) > 0:
                # Create Mean User Profile
                user_profile = liked_features.mean(axis=0).values.reshape(1, -1)
                
                # Apply Feature Overrides (Sonic Sliders)
                if feature_overrides:
                    print(f"Applying overrides: {feature_overrides}")
                    # Map overrides to vector indices
                    # feature_columns tells us the index of each feature
                    for feature_name, offset in feature_overrides.items():
                        if feature_name in self.feature_columns:
                            idx = self.feature_columns.index(feature_name)
                            # We add the offset to the mean profile
                            # Example: high energy -> add 1.0 to energy
                            user_profile[0, idx] += offset

                # KNN Search
                # Fetch more candidates (e.g. 50) to allow for randomization
                distances, indices = self.active_model.kneighbors(user_profile, n_neighbors=50 + len(liked_track_ids))
                
                # Filter out songs user already liked
                candidate_indices = indices[0]
                candidates = self.data.iloc[candidate_indices]
                
                # Exclude already liked
                candidates = candidates[~candidates['track_id'].isin(liked_track_ids)]
                
                print(f"DEBUG: Found {len(candidates)} candidates for home feed.")

                # Randomize the selection from the top candidates to keep feed fresh
                if len(candidates) > limit:
                     print("DEBUG: Random sampling applied.")
                     candidates = candidates.sample(n=limit)
                else:
                     print("DEBUG: Not enough candidates to sample, returning top.")
                     candidates = candidates.head(limit)
                
                recs = self._annotate_recs(candidates, "Based on your taste", user_profile=user_profile)
            else:
                # Fallback if IDs dont match data
                top_pop = self.data.sort_values(by='popularity', ascending=False).head(limit)
                recs = self._annotate_recs(top_pop, "Trending Now")

        return recs

    def recommend_radio(self, seed_track_id, limit=10):
        """Item-to-Item recommendation starting from a seed song."""
        if self.data is None or self.active_model is None:
            return []
            
        # Find the seed song features
        seed_row = self.data[self.data['track_id'] == seed_track_id]
        if seed_row.empty:
            return []
            
        seed_features = seed_row[self.feature_columns].values.reshape(1, -1)
        
        # Similar items
        distances, indices = self.active_model.kneighbors(seed_features, n_neighbors=limit + 1)
        
        candidate_indices = indices[0]
        candidates = self.data.iloc[candidate_indices]
        
        # Exclude the seed song itself
        candidates = candidates[candidates['track_id'] != seed_track_id].head(limit)
        
        return self._annotate_recs(candidates, "Similar Vibe")

    def _annotate_recs(self, df, default_reason, user_profile=None):
        """Adds explainability tags to recommendations."""
        annotated = []
        
        # Define some indices for the user_profile vector (based on self.feature_columns)
        # ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 
        #  'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        # Map column name to index for faster lookup if needed, but we can just use dicts
        
        for _, row in df.iterrows():
            reason = default_reason
            context_reasons = []
            
            # --- 1. Genre Based (If available) ---
            if 'track_genre' in row and pd.notna(row['track_genre']):
                genre = str(row['track_genre']).title()
                context_reasons.append(genre)

            # --- 2. Audio Feature Heuristics ---
            # RobustScaler values are centered around 0. 
            # > 1 is high, < -1 is low (roughly).
            
            dance = row.get('danceability', 0)
            energy = row.get('energy', 0)
            valence = row.get('valence', 0)
            acoustic = row.get('acousticness', 0)
            instrumental = row.get('instrumentalness', 0)
            
            # Combinations
            if dance > 0.8 and energy > 0.8:
                context_reasons.append("Party Vibes")
            elif acoustic > 0.8 and energy < -0.5:
                context_reasons.append("Chill Acoustic")
            elif instrumental > 1.0:
                context_reasons.append("Focus Flow")
            elif valence > 1.0:
                context_reasons.append("Feel Good")
            elif valence < -1.0:
                 context_reasons.append("Melancholic")
            elif energy > 1.2:
                 context_reasons.append("High Intensity")
                 
            # --- 3. User Profile Context (Why *You* might like it) ---
            if user_profile is not None:
                # user_profile is a shape (1, n_features) numpy array
                # We need to map it back to features
                # Simple check: Does this song match the user's extreme comparisons?
                pass 
                # (For now, heuristic tags are improving the UX significantly enough)

            # Select Best Tag
            if context_reasons:
                # Prioritize: "Mood" (Heuristic) > "Genre"
                # Heuristics are usually added last in the checks above, so reverse?
                # Actually, let's pick the most descriptive non-genre one first if exists.
                
                # Filter out genre if we have a mood
                moods = [r for r in context_reasons if r not in [str(row.get('track_genre', '')).title()]]
                
                if moods:
                    reason = moods[-1] # Pick last added (most specific?)
                else:
                    reason = context_reasons[0]
            
            item = row.to_dict()
            item['reason'] = reason
            annotated.append(item)
            
        return annotated

    def trigger_retraining(self):
        """Starts background training (Green Model)."""
        if self.is_training:
            print("Training already in progress. Skipping.")
            return False
            
        print("Triggering background retraining...")
        thread = threading.Thread(target=self._train_and_swap)
        thread.start()
        return True

    def _train_and_swap(self):
        """Private method to train new model and swap it in."""
        with self.lock:
            self.is_training = True
        
        try:
            print("--- [Green Model] Training Start ---")
            # 1. Preprocess Data 
            preprocessor = SpotifyPreprocessor(
                input_path=os.path.join(self.data_dir, "raw", "spotifyDataset.csv"),
                output_dir=os.path.join(self.data_dir, "processed")
            )
            df = preprocessor.process()
            
            # Update internal data
            self.data = df

            # 2. Train Model
            trainer = SpotifyRecommender(
                data_path=os.path.join(self.data_dir, "processed", "processed_data.csv"),
                model_dir=self.model_dir
            )
            new_model, used_features = trainer.train_memory() 
            
            # 3. Swap 
            self.active_model = new_model
            self.feature_columns = used_features
            
            new_version = f"v1.1.{int(time.time())}"
            self.model_version = new_version
            
            with open(os.path.join(self.model_dir, 'version.txt'), 'w') as f:
                f.write(new_version)
                
            print(f"--- [swapped] Active Model updated to {new_version} ---")
            
        except Exception as e:
            print(f"Training failed: {e}")
        finally:
            with self.lock:
                self.is_training = False

    def reset_baseline(self):
        """Resets to baseline model (clears feedback influence)."""
        print("Resetting to baseline model...")
        with self.lock:
            self.is_training = True
            
        try:
            feedback_file = os.path.join(self.data_dir, "feedback_data.csv")
            if os.path.exists(feedback_file):
                os.remove(feedback_file)
                
            trainer = SpotifyRecommender(
                data_path=os.path.join(self.data_dir, "processed", "processed_data.csv"),
                model_dir=self.model_dir
            )
            new_model, used_features = trainer.train_memory()
            
            self.active_model = new_model
            self.feature_columns = used_features
            self.model_version = "v1.0.0-baseline"
            
            print("--- [Reset] Model reset to baseline ---")
            return True
        except Exception as e:
            print(f"Reset failed: {e}")
            return False
        finally:
            with self.lock:
                self.is_training = False
