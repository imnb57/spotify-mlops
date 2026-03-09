# ML Architecture: Spotify KNN Recommender

This document details the Machine Learning pipeline used in the Spotify MLOps project.

## 1. Feature Engineering
We use audio features extracted from the Spotify API to represent each song as a vector in 13-dimensional space.

### Features
| Feature | Description | Range (Normalized) |
| :--- | :--- | :--- |
| `danceability` | Suitability for dancing | -1.0 to 1.0 |
| `energy` | Intensity and activity | -1.0 to 1.0 |
| `valence` | Musical positiveness (Mood) | -1.0 to 1.0 |
| `acousticness` | Confidence measure of acoustic | -1.0 to 1.0 |
| `instrumentalness` | Prediction of no vocals | -1.0 to 1.0 |
| `tempo` | BPM (Beats Per Minute) | Scaled |
| `loudness` | Overall loudness in dB | Scaled |
| `speechiness` | Presence of spoken words | Scaled |
| `liveness` | Presence of audience | Scaled |
| `key` | Musical key | Scaled |
| `duration_ms` | Track length | Scaled |
| `popularity` | Spotify Popularity 0-100 | Scaled |
| `explicit` | 1 if explicit, 0 if clean | Binary |

### Preprocessing (`src/preprocessing.py`)
-   **Scaler**: `RobustScaler` (Scikit-Learn).
    -   *Why?* Audio features often have outliers (e.g., extremely long songs). RobustScaler removes the median and scales according to the quantile range, making it robust to outliers.
-   **Deduplication**: Logic handles exact duplicate IDs and semantic duplicates (same song name + artist).

## 2. Model Algorithm (`src/train.py`)

### Algorithm: K-Nearest Neighbors (KNN)
We treat recommendation as a "vector search" problem.

-   **Library**: `sklearn.neighbors.NearestNeighbors`
-   **Metric**: `cosine` similarity.
    -   We measure the *angle* between song vectors, not just the Euclidean distance. This helps when magnitude (like popularity) varies but the "vibe" (direction) is similar.
-   **Algorithm**: `brute` force search.
    -   For datasets < 100k rows, brute force is fast enough and exact. For larger scale, we would switch to FAISS or Annoy (Approximate Nearest Neighbors).

## 3. Inference Logic (`src/model_manager.py`)

### A. Home Feed (User-to-Item)
1.  **Input**: List of user's liked Song IDs.
2.  **Profile Vector**: We calculate the **Mean Point** (Centroid) of all liked songs in the 13D space.
3.  **Sonic Sliders (V5)**: We add an **Offset Vector** to this centroid based on user input (e.g., `Profile_Vector + [0, 0, 0.5, ...]` for extra energy).
4.  **Query**: `KNN(User_Centroid)` returns the closest tracks.

### B. Song Radio (Item-to-Item)
1.  **Input**: A single "Seed" Song ID.
2.  **Query**: We lookup the feature vector of that specific song.
3.  **Search**: `KNN(Seed_Vector)` returns the closest tracks.

## 4. MLOps: Blue/Green Deployment
-   **Training**: Occurs in a background thread.
-   **Swap**: The system maintains a reference `self.active_model`. When training finishes, this reference is atomically updated to the new model object.
-   **Effect**: Zero downtime. Users getting recommendations during training use the old model; the next request uses the new one.
