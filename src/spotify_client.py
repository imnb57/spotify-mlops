import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
import time

class SpotifyClient:
    def __init__(self, client_id=None, client_secret=None):
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set.")
            
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        ))

    def fetch_new_releases(self, limit=50):
        """Fetches new releases and extracts features."""
        print("Fetching new releases...")
        results = self.sp.new_releases(limit=limit)
        albums = results['albums']['items']
        
        tracks_data = []
        
        for album in albums:
            album_name = album['name']
            album_id = album['id']
            # Get tracks for album
            tracks = self.sp.album_tracks(album_id)['items']
            
            for track in tracks:
                track_id = track['id']
                track_name = track['name']
                artists = ";".join([artist['name'] for artist in track['artists']])
                
                # specific features
                # We need audio features
                try:
                    features = self.sp.audio_features([track_id])[0]
                    if features:
                        track_info = {
                            'track_id': track_id,
                            'artists': artists,
                            'album_name': album_name,
                            'track_name': track_name,
                            'popularity': 50, # New releases default? Or fetch track details
                            'duration_ms': features['duration_ms'],
                            'explicit': False, # Need track details for this
                            'danceability': features['danceability'],
                            'energy': features['energy'],
                            'key': features['key'],
                            'loudness': features['loudness'],
                            'mode': features['mode'],
                            'speechiness': features['speechiness'],
                            'acousticness': features['acousticness'],
                            'instrumentalness': features['instrumentalness'],
                            'liveness': features['liveness'],
                            'valence': features['valence'],
                            'tempo': features['tempo'],
                            'time_signature': features['time_signature'],
                            'track_genre': 'pop' # Placeholder
                        }
                        tracks_data.append(track_info)
                except Exception as e:
                    print(f"Error fetching features for {track_name}: {e}")
                
                time.sleep(0.1) # Rate limit
                
        return pd.DataFrame(tracks_data)

    def save_data(self, df, filepath="data/raw/new_data.csv"):
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} new tracks to {filepath}")

if __name__ == "__main__":
    # Test execution
    try:
        client = SpotifyClient()
        df = client.fetch_new_releases(limit=5)
        print(df.head())
    except ValueError as e:
        print(e)
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
