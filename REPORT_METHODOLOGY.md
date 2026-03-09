# Methodology

## 3.1 Research Design Overview

This project follows an applied research design grounded in the **Design Science Research Methodology (DSRM)**, wherein a functional software artifact — an interactive music recommendation system — is iteratively designed, developed, and evaluated. The system integrates content-based filtering, real-time user interaction, and modern MLOps deployment practices into a unified full-stack application. The research is structured across five sequential phases: (i) data collection and exploratory analysis, (ii) data preprocessing and feature engineering, (iii) model development and inference design, (iv) system architecture and deployment, and (v) evaluation and validation. Each phase is detailed in the subsequent subsections.

---

## 3.2 Dataset Description

### 3.2.1 Data Source

The primary dataset is a publicly available collection of Spotify track metadata and audio features, comprising over **114,000 tracks** spanning a diverse range of genres. The dataset was sourced in CSV format (`spotifyDataset.csv`, approximately 20 MB) and stored under version control using **DVC (Data Version Control)** to ensure reproducibility and traceability across experiments.

To support incremental data growth, a supplementary data ingestion pipeline (`src/ingestion.py`) was implemented using the **Spotipy** library — a Python wrapper for the **Spotify Web API**. This pipeline authenticates via OAuth 2.0 Client Credentials flow and programmatically fetches new releases, extracting audio features for each track using the `/audio-features` endpoint. Fetched records are appended to the raw dataset with deduplication on `track_id` to prevent data inflation.

### 3.2.2 Feature Set

Each track in the dataset is described by a set of **13 numerical audio features** provided by Spotify's audio analysis engine. These features collectively characterize the sonic properties of each song and serve as the input vector for the recommendation model. The feature set is enumerated in Table 1.

**Table 1: Audio Feature Descriptions and Value Ranges**

| Feature | Description | Original Range |
|:---|:---|:---|
| `popularity` | Spotify-assigned track popularity score | 0–100 |
| `duration_ms` | Track duration in milliseconds | Continuous |
| `danceability` | Suitability for dancing based on tempo, rhythm stability, and beat strength | 0.0–1.0 |
| `energy` | Perceptual measure of intensity and activity | 0.0–1.0 |
| `key` | Estimated overall musical key (pitch class notation) | 0–11 |
| `loudness` | Overall average loudness in decibels (dB) | –60 to 0 dB |
| `speechiness` | Presence of spoken words in a track | 0.0–1.0 |
| `acousticness` | Confidence measure of whether the track is acoustic | 0.0–1.0 |
| `instrumentalness` | Prediction of whether a track contains no vocals | 0.0–1.0 |
| `liveness` | Detection of an audience in the recording | 0.0–1.0 |
| `valence` | Musical positiveness conveyed by a track (mood) | 0.0–1.0 |
| `tempo` | Estimated tempo in beats per minute (BPM) | Continuous |
| `explicit` | Binary flag indicating explicit lyrical content | 0 or 1 |

### 3.2.3 Exploratory Data Analysis (EDA)

Prior to model development, an exploratory data analysis phase was conducted (`src/eda.py`) to understand the structure, quality, and statistical properties of the dataset. The key findings that informed subsequent design decisions are summarized below:

1. **Genre Distribution**: The dataset exhibits a balanced distribution across multiple genres (e.g., pop, rock, hip-hop, electronic, classical, jazz), ensuring that the model does not develop systematic bias toward any single genre cluster.

2. **Feature Correlation Analysis**: A Pearson correlation matrix was computed to identify linear dependencies among audio features. Strong positive correlations were observed between `energy` and `loudness` (r ≈ 0.77), while a strong negative correlation was found between `energy` and `acousticness` (r ≈ –0.73). This insight guided the selection of `energy` as a primary user-facing control parameter (Sonic Slider), as manipulating it implicitly influences multiple correlated features, thereby amplifying the perceptual impact of a single slider adjustment.

3. **Feature Density Mapping**: A kernel density estimation (KDE) of `energy` versus `danceability` revealed natural clustering patterns in the feature space — for example, high-energy/high-danceability tracks (party music) and low-energy/high-acousticness tracks (acoustic ballads). This validated the use of a distance-based algorithm that can exploit such spatial structure.

4. **Duplicate Detection**: The dataset contained both exact duplicates (identical `track_id`) and semantic duplicates (identical `track_name` + `artist` combinations across different albums or releases). Both categories were flagged for removal during preprocessing.

---

## 3.3 Data Preprocessing Pipeline

The preprocessing pipeline is encapsulated in the `SpotifyPreprocessor` class (`src/preprocessing.py`) and executes a deterministic, reproducible sequence of transformations. The pipeline is designed as an idempotent operation: re-running it on the same raw input always produces identical output.

### 3.3.1 Data Cleaning

The cleaning phase addresses three categories of data quality issues:

1. **Exact Deduplication**: Records with duplicate `track_id` values are removed, retaining the first occurrence.

2. **Semantic Deduplication**: Track names and artist names are normalized to lowercase with whitespace trimming. Records are then sorted by `popularity` in descending order, and duplicates on the `(track_name, artist)` pair are dropped — retaining the most popular version. This prevents the model from being biased by multiple versions of the same song (e.g., album version, deluxe edition, remaster).

3. **Missing Value Handling**: Rows with null values in any of the 13 feature columns are dropped entirely, as imputation of audio features would introduce artificial signals into the distance-based model. The `explicit` field is additionally cast from boolean to integer (0 or 1) to ensure numerical consistency.

### 3.3.2 Feature Scaling: RobustScaler

Feature scaling is a critical requirement for distance-based algorithms such as K-Nearest Neighbors, where features with larger absolute ranges (e.g., `duration_ms` in the range of 100,000–500,000 ms) would disproportionately dominate the distance computation relative to features like `danceability` (range 0.0–1.0).

We employ **Scikit-Learn's `RobustScaler`** rather than the more common `StandardScaler` (Z-score normalization) or `MinMaxScaler`. The rationale is as follows:

- **`StandardScaler`** centers data by mean and scales by standard deviation. It is sensitive to outliers, which shift the mean and inflate the standard deviation, compressing the useful range of the majority of data points.
- **`MinMaxScaler`** maps values to a fixed [0, 1] range but is even more sensitive to outliers, as a single extreme value can compress all other values into a narrow band.
- **`RobustScaler`** removes the **median** (rather than the mean) and scales data according to the **Interquartile Range (IQR = Q3 – Q1)**. This renders the transformation robust to outliers, which is essential for music data where valid but extreme values exist (e.g., exceptionally long tracks, unusually loud recordings, or spoken-word content with high speechiness).

Mathematically, for each feature x:

> x_scaled = (x − median(x)) / IQR(x)

The fitted scaler is serialized using **Joblib** (`scaler.joblib`) to ensure that the exact same transformation can be applied to new data during inference or retraining.

### 3.3.3 Feature Vector Representation

After preprocessing, each song is represented as a **13-dimensional numerical vector**:

```
v = [popularity, duration_ms, danceability, energy, key, loudness,
     speechiness, acousticness, instrumentalness, liveness, valence,
     tempo, explicit]
```

This vector representation enables the application of distance-based similarity metrics in a shared feature space, where proximity between vectors indicates sonic similarity between tracks.

---

## 3.4 Recommendation Engine

### 3.4.1 Algorithm Selection: K-Nearest Neighbors (KNN)

The recommendation problem is formulated as a **vector similarity search** in the 13-dimensional audio feature space. The system employs `sklearn.neighbors.NearestNeighbors` configured with the following parameters:

- **Distance Metric**: Cosine Similarity
- **Search Algorithm**: Brute-force exhaustive search

**Cosine Similarity vs. Euclidean Distance**: The choice of cosine similarity over Euclidean distance is motivated by the nature of music perception. Two songs may have similar *ratios* of features (e.g., both are relatively high-energy and high-danceability) but differ in absolute magnitude (e.g., one is louder overall). Cosine similarity measures the *angle* between two vectors in the feature space, thereby capturing the "direction" or "profile shape" of a song's audio fingerprint, irrespective of magnitude. This makes it more suitable for capturing subjective notions of musical "vibe" similarity.

**Brute-Force Search**: For a catalog of approximately 114,000 tracks, brute-force exact search is computationally feasible (sub-second query times on modern hardware) and avoids the approximation errors introduced by Approximate Nearest Neighbor (ANN) methods such as FAISS or Annoy. For larger-scale deployments (millions of tracks), the architecture is designed to accommodate a swap to ANN indices without modifying the inference interface.

### 3.4.2 Training Process

The model training process (`SpotifyRecommender` class in `src/train.py`) proceeds as follows:

1. **Data Loading**: The preprocessed CSV (`processed_data.csv`) is loaded into a Pandas DataFrame.

2. **Feedback Integration**: If a user feedback file (`feedback_data.csv`) exists, the system reads it and applies a **popularity boost** of +0.2 (on the scaled feature axis) to tracks that have been explicitly liked by users. This popularity value is capped at 1.0 to prevent runaway amplification. This mechanism creates a lightweight feedback loop wherein user preferences gradually influence the model's behavior upon retraining.

3. **Model Fitting**: The feature matrix X ∈ ℝ^(n×13) is extracted and passed to `NearestNeighbors.fit()`, which indexes the entire dataset for subsequent neighbor queries.

4. **Artifact Serialization**: The trained model, the ordered list of feature column names, and a version string are serialized to disk using Joblib, enabling persistent storage and version tracking.

### 3.4.3 Inference Modes

The system supports two distinct inference paradigms, each addressing a different user intent:

#### A. User-to-Item Recommendation (Home Feed)

This mode generates a personalized feed based on the user's accumulated taste profile. The inference logic (`ModelManager.recommend_home()`) operates as follows:

1. **Cold Start Handling**: If no user feedback exists (i.e., no liked songs), the system falls back to a **popularity-based ranking**, serving the most popular tracks as a "Trending Now" feed. This addresses the cold-start problem inherent in collaborative and content-based filtering systems.

2. **User Profile Construction**: For users with feedback history, the feature vectors of all liked songs are retrieved and their **element-wise arithmetic mean** (centroid) is computed:

   > P_user = (1/N) × Σ v_i, for i = 1 to N

   where N is the number of liked songs and v_i is the 13-dimensional feature vector of the i-th liked song. This centroid represents the user's average musical taste in the feature space.

3. **Sonic Sliders (Interactive Feature Override)**: The system's key innovation lies in its **Sonic Sliders** mechanism. Users can apply real-time offsets to specific dimensions of their profile vector through UI sliders. For example, if the user drags the "Energy" slider to +0.5, the system modifies the query vector as:

   > P_query = P_user + [0, 0, 0, +0.5, 0, ..., 0]

   This offset is applied to the corresponding index of the centroid vector (determined by the feature column ordering). Four sliders are exposed: **Energy**, **Danceability**, **Valence** (Mood), and **Acousticness**. This mechanism transforms the recommendation algorithm from a "black box" into a transparent, user-controllable exploration tool.

4. **KNN Query and Filtering**: The modified query vector is passed to `kneighbors()` with an expanded neighbor count (50 + N_liked) to provide a sufficiently large candidate pool. Already-liked songs are filtered out to avoid redundancy. From the remaining candidates, a **random sample** of the requested limit is drawn to introduce serendipity and prevent the home feed from becoming stale across repeated requests.

#### B. Item-to-Item Recommendation (Song Radio)

This mode generates recommendations based on a single "seed" song, independent of the user's taste profile. The inference logic (`ModelManager.recommend_radio()`) operates as follows:

1. **Seed Vector Retrieval**: The 13-dimensional feature vector of the specified seed track is looked up from the dataset.

2. **KNN Query**: `kneighbors()` is called with the seed vector, returning the (limit + 1) nearest neighbors. The seed song itself is excluded from the results.

3. **Use Case**: Song Radio allows users to explore sonic neighborhoods outside their established preferences, effectively creating an "infinite playlist" of songs that sound similar to any individual track in the catalog.

### 3.4.4 Explainability Layer

Each recommendation is annotated with a human-readable explanation tag generated by the `_annotate_recs()` method. The annotation system applies heuristic rules based on the RobustScaler-normalized feature values:

| Condition | Tag |
|:---|:---|
| danceability > 0.8 AND energy > 0.8 | "Party Vibes" |
| acousticness > 0.8 AND energy < –0.5 | "Chill Acoustic" |
| instrumentalness > 1.0 | "Focus Flow" |
| valence > 1.0 | "Feel Good" |
| valence < –1.0 | "Melancholic" |
| energy > 1.2 | "High Intensity" |

If a track has an associated genre label, this is included as a secondary tag. Mood-based tags (derived from audio features) are prioritized over genre tags when both are present. This explainability layer addresses a common criticism of recommendation systems — opacity — by providing users with an intuitive rationale for each suggestion.

---

## 3.5 System Architecture and MLOps

### 3.5.1 Technology Stack

The system is implemented as a decoupled, containerized full-stack application with the following components:

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Backend API** | FastAPI (Python) | Asynchronous REST API serving recommendations, feedback, and control endpoints |
| **Frontend UI** | React (Vite), TailwindCSS, Framer Motion | Interactive single-page application with responsive design and micro-animations |
| **ML Core** | Scikit-Learn (NearestNeighbors), Pandas, NumPy | Data preprocessing, model training, and inference |
| **Serialization** | Joblib | Model and scaler persistence |
| **Data Versioning** | DVC (Data Version Control) | Tracking dataset versions alongside code changes |
| **Containerization** | Docker, Docker Compose | Environment isolation and reproducible deployments |
| **Cloud Deployment** | AWS ECR + ECS (Fargate) | Container registry and serverless container orchestration |

### 3.5.2 API Design

The FastAPI backend exposes the following RESTful endpoints:

| Endpoint | Method | Description |
|:---|:---|:---|
| `/recommend/home` | GET | Personalized home feed with optional Sonic Slider parameters (`target_energy`, `target_danceability`, `target_valence`, `target_acousticness`) |
| `/recommend/radio/{track_id}` | GET | Item-to-item recommendations from a seed song |
| `/search` | GET | Full-text search over track names and artist names |
| `/feedback` | POST | Records user like/dislike interactions |
| `/control/retrain` | POST | Triggers background model retraining |
| `/reset` | POST | Resets user profile to baseline (clears feedback) |
| `/model-info` | GET | Returns current model version, feature list, and training status |

Cross-Origin Resource Sharing (CORS) middleware is configured to allow requests from the frontend origin, enabling the decoupled deployment architecture.

### 3.5.3 Blue/Green Model Deployment

A critical MLOps requirement is the ability to update the recommendation model without service downtime. The system implements a **Blue/Green deployment strategy** at the model level, managed by the `ModelManager` class:

1. **Blue (Active) Model**: The currently serving model instance (`self.active_model`) handles all incoming recommendation requests.

2. **Green (Staging) Model**: When retraining is triggered (via the `/control/retrain` endpoint), a new model is trained in a **background thread** (`threading.Thread`). During this process:
   - The raw dataset is re-preprocessed from scratch.
   - User feedback is incorporated (popularity boosting).
   - A new `NearestNeighbors` model is fitted.

3. **Atomic Swap**: Upon successful training completion, the `ModelManager` atomically updates its internal reference:
   ```
   self.active_model = new_model
   self.feature_columns = used_features
   self.model_version = new_version
   ```
   The Python GIL (Global Interpreter Lock) ensures that reference assignment is atomic, meaning that any request arriving during the swap will see either the old or the new model — never a partially updated state.

4. **Thread Safety**: A `threading.Lock` protects the `is_training` flag to prevent concurrent retraining operations. A version string incorporating the Unix timestamp (`v1.1.<timestamp>`) provides a unique identifier for each trained model.

This design enables **zero-downtime model updates**: users experience no interruption, and the transition from the old model to the new model is seamless.

### 3.5.4 Feedback Loop Architecture

The system implements a closed-loop learning cycle:

1. **Feedback Collection**: User interactions (likes/dislikes) are submitted via the `/feedback` POST endpoint and appended to `data/feedback_data.csv`.

2. **Feedback Incorporation**: During retraining, liked tracks receive a popularity boost (+0.2 on the scaled axis), subtly shifting them closer to future query centroids.

3. **Profile Reset**: The `/reset` endpoint allows users to clear their feedback history and revert to the baseline model, supporting experimentation and fresh starts.

This mechanism enables the system to adapt to user preferences over time without requiring explicit collaborative filtering infrastructure.

### 3.5.5 Containerization and Deployment

The application is containerized using separate Dockerfiles for the backend (`Dockerfile.backend`) and frontend (`Dockerfile.frontend`), orchestrated via **Docker Compose**. The composition defines:

- **Backend service**: Exposes port 8000, mounts `data/` and `models/` as persistent volumes, and receives Spotify API credentials via environment variables.
- **Frontend service**: Exposes port 5173, depends on the backend service, and receives the API URL via build-time environment variable (`VITE_API_URL`).

For production deployment, the containers are pushed to **AWS Elastic Container Registry (ECR)** and deployed on **AWS Elastic Container Service (ECS)** using the **Fargate** launch type (serverless). An Application Load Balancer (ALB) routes traffic to the appropriate container based on port. **AWS Elastic File System (EFS)** is recommended for mounting persistent storage to the backend container, ensuring that feedback data and model artifacts survive container restarts.

---

## 3.6 Evaluation Methodology

The evaluation framework (`src/evaluate.py`) implements a suite of **six quantitative metrics** designed to assess the recommendation system across multiple quality dimensions. All metrics are computed programmatically against the live model instance, ensuring that evaluation reflects actual system behavior.

### 3.6.1 Relevance: Average Cosine Similarity

**Purpose**: Measures how semantically close the recommended tracks are to the user's taste profile.

**Method**: The user profile centroid is computed from liked song vectors. The top-K nearest neighbors are retrieved, and the average cosine similarity between the centroid and each recommended track's feature vector is calculated.

**Interpretation**: Values range from –1 to 1. A score above 0.7 indicates relevant recommendations; above 0.9 indicates highly relevant recommendations.

### 3.6.2 Diversity: Intra-List Diversity (ILD)

**Purpose**: Ensures that the recommendation list is not composed of near-identical songs.

**Method**: For each pair of recommended tracks, the pairwise cosine distance (1 – cosine similarity) is computed. ILD is the mean of all pairwise distances within the recommendation list.

**Interpretation**: Values range from 0 to 2. A score of 0 means all recommendations are identical; 0.2–0.5 indicates healthy diversity; above 0.8 may indicate overly random results.

### 3.6.3 Catalog Coverage

**Purpose**: Quantifies the fraction of the total catalog that the system is capable of recommending, detecting potential popularity bias.

**Method**: The home feed recommendation endpoint is invoked 50 times, and the union of all unique recommended `track_id` values is computed. Coverage is reported as the ratio of unique recommendations to total catalog size.

**Interpretation**: For a 114,000-track catalog, coverage above 5% is considered good; above 1% is acceptable.

### 3.6.4 Novelty Score

**Purpose**: Measures whether the system recommends obscure or "hidden gem" tracks rather than only popular mainstream content.

**Method**: Novelty is computed using **self-information theory**. Each recommended track's popularity is transformed via a sigmoid function to produce a probability-like value p, and the novelty of the track is defined as –log₂(p). The system's average novelty is compared against a **popularity baseline** (the novelty score of the top-20 most popular tracks in the catalog).

**Interpretation**: Higher novelty scores indicate that the system surfaces more obscure content. A positive novelty gain over the popularity baseline demonstrates that the system avoids pure popularity-based recommendations.

### 3.6.5 Sonic Slider Sensitivity

**Purpose**: Validates that the interactive Sonic Slider mechanism produces a measurable, monotonic effect on recommendation output.

**Method**: For each of four slider features (`energy`, `danceability`, `valence`, `acousticness`), the home feed is queried at five offset levels: –1.0, –0.5, 0.0, +0.5, +1.0. The average value of the corresponding feature across the returned recommendations is recorded at each offset level. A monotonicity check verifies that at least 60% of consecutive (offset, average value) pairs are non-decreasing.

**Interpretation**: A PASS result indicates that increasing the slider value reliably shifts the recommendations in the expected direction, confirming that user controls have a genuine effect on the algorithm's output.

### 3.6.6 Hit Rate @ K (Leave-One-Out Evaluation)

**Purpose**: Measures the model's predictive accuracy — its ability to recover a known-liked song from a user's profile.

**Method**: A leave-one-out cross-validation protocol is applied to the user's liked songs. For each liked song:
1. The song is held out from the profile.
2. A new centroid is computed from the remaining liked songs.
3. The KNN model is queried for the top-K neighbors of this centroid.
4. A "hit" is recorded if the held-out song appears in the top-K results.

Hit Rate @ K is reported for K ∈ {10, 20, 50}.

**Interpretation**: For a catalog of 114,000 tracks, a hit rate above 5% at K=10 is considered strong, as the random baseline would be approximately 0.009% (10/114,000).

---

## 3.7 Tools and Technologies

Table 2 summarizes the complete set of tools, libraries, and services used in this project.

**Table 2: Tools and Technologies**

| Category | Tool / Library | Version | Purpose |
|:---|:---|:---|:---|
| Programming Language | Python | 3.x | Backend, ML pipeline, data processing |
| Programming Language | JavaScript (ES6+) | — | Frontend application |
| Web Framework | FastAPI | Latest | Asynchronous REST API |
| Frontend Framework | React (via Vite) | Latest | Single-page application UI |
| CSS Framework | TailwindCSS | Latest | Utility-first responsive styling |
| Animation Library | Framer Motion | Latest | UI micro-animations and transitions |
| ML Library | Scikit-Learn | Latest | KNN model (`NearestNeighbors`) and `RobustScaler` |
| Data Processing | Pandas, NumPy | Latest | DataFrame operations, numerical computation |
| Serialization | Joblib | Latest | Model and scaler persistence |
| API Client | Spotipy | Latest | Spotify Web API integration |
| Containerization | Docker, Docker Compose | Latest | Environment isolation, multi-service orchestration |
| Data Versioning | DVC | Latest | Dataset version tracking |
| Cloud Platform | AWS (ECR, ECS Fargate, EFS, ALB) | — | Production deployment infrastructure |
| Version Control | Git, GitHub | — | Source code management and CI/CD |

---

## 3.8 Summary of Methodology

The methodology follows a structured pipeline from data acquisition to production deployment. Raw Spotify audio features are cleaned, deduplicated, and scaled using RobustScaler to produce 13-dimensional track representations. A KNN model with cosine similarity performs vector search across the feature space, supporting two inference modes: user-to-item (centroid-based with interactive Sonic Slider offsets) and item-to-item (seed-track-based Song Radio). The system is deployed as a containerized full-stack application with Blue/Green model swapping for zero-downtime updates and a closed-loop feedback mechanism for iterative model improvement. Evaluation is conducted across six dimensions — relevance, diversity, coverage, novelty, interactivity, and predictive power — providing comprehensive quantitative evidence of system effectiveness.
