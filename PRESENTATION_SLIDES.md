# Spotify MLOps: Interactive Music Recommendation System
## Presentation Slide Content

> **Instructions for LLM Slide Generators**: Use each `---` separated section as one slide. The `# Heading` is the slide title, bullet points are the content, and the `> 🎤 Speaker Notes` block contains what to say during the presentation. Generate a modern, dark-themed design with green (#1DB954 Spotify green) accents.

---

# Slide 1 — Title Slide

## Spotify MLOps
### An Interactive Music Recommendation System with Real-Time Personalization & Blue/Green Deployment

**Presented by**: [Your Name]
**Date**: February 2026

> 🎤 **Speaker Notes**: Good morning/afternoon everyone. Today I'll be presenting my project — Spotify MLOps — an end-to-end music recommendation system that goes beyond typical "black box" recommenders by letting users actively control how recommendations are generated, all while following production-grade MLOps practices.

---

# Slide 2 — Problem Statement

## The Problem with Music Recommendations

- 🔒 Most recommendation systems are **opaque black boxes** — users have no visibility into *why* a song was suggested
- 🔄 Systems are **static** — users can't adjust or steer recommendations in real-time
- ❄️ **Cold start problem** — new users with no history receive poor or generic suggestions
- 🔧 Models are difficult to **update without downtime** in production environments

### Our Goal
Build a **transparent, interactive, and production-ready** recommendation system that puts users in control.

> 🎤 **Speaker Notes**: The core problem we're addressing is this — when you use any music platform, the algorithm decides what you hear, but you can't tell it "I want more energy" or "make it more acoustic." Our system solves this by making the recommendation algorithm interactive and transparent.

---

# Slide 3 — Project Overview

## What We Built

| Component | Description |
|:---|:---|
| 🧠 **ML Engine** | KNN-based content filtering on 13 audio features |
| 🎚️ **Sonic Sliders** | Real-time UI controls that modify the recommendation vector |
| 📻 **Song Radio** | Item-to-item "play more like this" infinite playlist |
| 🟢 **Blue/Green Deploy** | Zero-downtime model swapping in production |
| 🔄 **Feedback Loop** | User likes retrain and improve the model |
| 🏷️ **Explainability** | Tags like "Party Vibes" and "Chill Acoustic" on every recommendation |

> 🎤 **Speaker Notes**: At a high level, we built six key capabilities. The ML engine uses K-Nearest Neighbors on audio features. What makes it unique is the Sonic Sliders — users can drag sliders to bias recommendations toward more energy, more danceability, and so on. We also have Song Radio for item-to-item discovery, Blue/Green deployment for zero-downtime updates, a feedback loop, and explainability tags on every recommendation.

---

# Slide 4 — Tech Stack

## Technology Stack

### Backend
- **FastAPI** — Async REST API
- **Scikit-Learn** — NearestNeighbors (KNN)
- **Pandas / NumPy** — Data processing
- **Joblib** — Model serialization

### Frontend
- **React** (Vite) — SPA framework
- **TailwindCSS** — Styling
- **Framer Motion** — Animations

### DevOps / MLOps
- **Docker & Docker Compose** — Containerization
- **DVC** — Data version control
- **AWS ECR / ECS Fargate** — Cloud deployment
- **GitHub Actions** — CI/CD automation

> 🎤 **Speaker Notes**: Here's our complete tech stack. The backend is Python with FastAPI for high-performance async endpoints. The ML core uses Scikit-Learn. The frontend is React with Vite for fast development and TailwindCSS for styling. For MLOps, we use Docker for containerization, DVC for data versioning, and AWS for cloud deployment.

---

# Slide 5 — Dataset

## Dataset Overview

- **Source**: Spotify audio features dataset + Spotify Web API (via Spotipy)
- **Size**: **114,000+ tracks** across diverse genres
- **Version controlled** with **DVC** (Data Version Control)

### 13 Audio Features Per Track

| Sonic Features | Metadata |
|:---|:---|
| danceability, energy, valence | popularity |
| acousticness, instrumentalness | duration_ms |
| speechiness, liveness, tempo | key, loudness |
| | explicit |

Each song → **13-dimensional numerical vector** in feature space

> 🎤 **Speaker Notes**: Our dataset contains over 114,000 tracks with 13 audio features each. These features are extracted by Spotify's own audio analysis engine and capture things like how danceable, energetic, or acoustic a song is. Each song becomes a point in a 13-dimensional space, and that's what our algorithm searches through.

---

# Slide 6 — Exploratory Data Analysis

## Key EDA Insights

### 1. Feature Correlations
- **Energy ↔ Loudness**: Strong positive (r ≈ 0.77)
- **Energy ↔ Acousticness**: Strong negative (r ≈ –0.73)
- → Energy chosen as primary slider: manipulating it implicitly affects multiple features

### 2. Feature Space Clustering
- KDE of Energy vs. Danceability reveals natural clusters:
  - 🎉 High-energy + High-dance = Party tracks
  - 🎸 Low-energy + High-acoustic = Acoustic ballads
- → Validates use of distance-based algorithms

### 3. Data Quality
- Found exact duplicates (same `track_id`) and semantic duplicates (same song name + artist, different albums)
- Both addressed in preprocessing

> 🎤 **Speaker Notes**: Our EDA revealed three critical insights. First, energy strongly correlates with loudness and inversely with acousticness — so we chose energy as a primary slider since moving it implicitly affects other features. Second, plotting energy vs danceability shows natural clusters of song types, which validates our distance-based approach. Third, we found duplicate issues that needed cleaning.

---

# Slide 7 — Data Preprocessing

## Preprocessing Pipeline

```
Raw CSV (114K tracks)
    │
    ├── 1. Remove exact duplicates (track_id)
    ├── 2. Remove semantic duplicates (song name + artist)
    │       └── Keep most popular version
    ├── 3. Drop rows with missing features
    ├── 4. Cast explicit → integer (0/1)
    │
    └── 5. Scale with RobustScaler
            └── Output: processed_data.csv + scaler.joblib
```

### Why RobustScaler?

| Scaler | Problem |
|:---|:---|
| StandardScaler | Sensitive to outliers (shifts mean) |
| MinMaxScaler | Single outlier compresses all values |
| **RobustScaler** ✅ | Uses **median & IQR** — robust to extreme values |

**Formula**: x_scaled = (x − median) / IQR

> 🎤 **Speaker Notes**: Our preprocessing pipeline cleans and transforms raw data in five steps. The key decision here is using RobustScaler instead of StandardScaler. Music data has valid but extreme values — very long songs, very loud tracks — and RobustScaler handles these by using the median and interquartile range instead of mean and standard deviation. This prevents outliers from distorting the feature space.

---

# Slide 8 — The KNN Algorithm

## Recommendation Algorithm: K-Nearest Neighbors

### Why KNN?
- Recommendation = **finding similar songs in feature space**
- No training in the traditional sense — it indexes all data for fast lookup

### Configuration
- **Library**: `sklearn.neighbors.NearestNeighbors`
- **Metric**: Cosine Similarity
- **Algorithm**: Brute-force (exact search)

### Why Cosine over Euclidean?

```
Cosine Similarity = cos(θ) = (A · B) / (|A| × |B|)
```

- Measures the **angle** (direction) between vectors, not magnitude
- Two songs with similar feature **ratios** (same "vibe") but different loudness → Cosine says "similar", Euclidean says "different"
- Better captures subjective **"sounds like"** similarity

> 🎤 **Speaker Notes**: We use K-Nearest Neighbors with cosine similarity. The key insight is: cosine measures the angle between vectors — the direction, not the magnitude. Two songs might both be high-energy and danceable but differ in loudness. Cosine recognizes they have the same "vibe profile" even if their absolute values differ. Brute force search is used because for 114K tracks, it's fast enough and gives exact results.

---

# Slide 9 — Inference Mode 1: Home Feed

## User-to-Item: Personalized Home Feed

```
Step 1: Collect user's liked songs
Step 2: Compute CENTROID (mean vector) of liked songs
Step 3: Apply Sonic Slider offsets to centroid
Step 4: KNN search → top-50 candidates
Step 5: Filter out already-liked songs
Step 6: Random sample for freshness
```

### Cold Start Solution
- No likes yet? → Serve **top popular tracks** ("Trending Now")
- As user interacts → personalized recommendations gradually take over

### The Centroid Approach
```
User Profile = (1/N) × Σ liked_song_vectors
```
- Averages all liked songs into a single point representing the user's "taste center"

> 🎤 **Speaker Notes**: The home feed works by computing a centroid — the geometric average — of all songs the user has liked. This single point in 13D space represents their taste. We then find the nearest neighbors to this point. For cold start, when there are no likes, we simply show trending tracks. As the user likes more songs, recommendations become increasingly personalized.

---

# Slide 10 — Sonic Sliders (Key Innovation)

## 🎚️ Sonic Sliders: Interactive Recommendation Control

### How It Works
```
Query Vector = User Centroid + Slider Offsets
```

**Example**: User wants "More Energy"
```
Original:  [0.2, -0.1, 0.3, [0.1], ...]
                              ↑ energy
Offset:    [0.0,  0.0, 0.0, [+0.5], ...]

Modified:  [0.2, -0.1, 0.3, [0.6], ...]
                              ↑ shifted!
```

### Available Sliders
| Slider | Controls |
|:---|:---|
| 🔥 Energy | Intensity & activity |
| 💃 Danceability | Rhythm & groove |
| 😊 Valence | Mood (happy ↔ sad) |
| 🎸 Acousticness | Acoustic vs. electronic |

### Why This Matters
- Transforms a black-box algorithm into a **user-controllable tool**
- Users **explore** the feature space, not just consume it

> 🎤 **Speaker Notes**: This is our key innovation — Sonic Sliders. The idea is simple but powerful: we let users add offsets to specific dimensions of their profile vector. Drag the Energy slider up, and we literally add +0.5 to the energy component of the query vector before searching. The result? The recommendations shift toward higher-energy songs. This turns a black box into an exploration tool.

---

# Slide 11 — Inference Mode 2: Song Radio

## Item-to-Item: Song Radio 📻

### Concept
> "Play me more songs that sound like *this specific song*"

### How It Works
```
1. User clicks Radio icon on any track
2. System retrieves that song's 13D feature vector
3. KNN finds nearest neighbors to THAT vector
4. Ignores user profile entirely
```

### Use Case
- Explore music **outside your usual bubble**
- Discover songs similar to a single track, not your overall taste
- Creates an **infinite playlist** of similar-sounding music

> 🎤 **Speaker Notes**: Song Radio is our item-to-item mode. Instead of using the user's overall profile, we take one specific song's feature vector and find its nearest neighbors. This is powerful for exploration — if you hear one great song outside your usual taste, you can instantly explore that neighborhood of the feature space.

---

# Slide 12 — Explainability

## Why Was This Song Recommended?

Every recommendation includes a human-readable **explanation tag**:

| Audio Signal | Tag |
|:---|:---|
| High dance + High energy | 🎉 "Party Vibes" |
| High acoustic + Low energy | 🎵 "Chill Acoustic" |
| High instrumentalness | 🎧 "Focus Flow" |
| High valence | 😊 "Feel Good" |
| Low valence | 🌧️ "Melancholic" |
| Very high energy | ⚡ "High Intensity" |

### Implementation
- Heuristic rules on RobustScaler-normalized values
- Mood tags prioritized over genre tags
- Addresses the #1 criticism of recommenders: **opacity**

> 🎤 **Speaker Notes**: We solve the "why was this recommended?" problem with explainability tags. Based on the normalized feature values of each song, we apply heuristic rules — if a song has high danceability AND high energy, it gets tagged "Party Vibes." These tags help users understand and trust the recommendations.

---

# Slide 13 — System Architecture

## Full-Stack Architecture

```
┌──────────────────────────────────────────────┐
│                  FRONTEND                     │
│         React (Vite) + TailwindCSS           │
│      Sonic Sliders | Song Cards | Radio      │
└──────────────────┬───────────────────────────┘
                   │ HTTP REST
┌──────────────────▼───────────────────────────┐
│               FastAPI BACKEND                 │
│                                               │
│  /recommend/home    (Sonic Sliders support)   │
│  /recommend/radio   (Item-to-Item)            │
│  /feedback          (Like/Dislike)            │
│  /control/retrain   (Trigger retraining)      │
│  /search            (Full-text search)        │
│  /model-info        (Version & status)        │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│             MODEL MANAGER                     │
│                                               │
│  ┌─────────┐    ┌─────────┐                  │
│  │  BLUE   │◄──►│  GREEN  │   Blue/Green     │
│  │ (Active)│    │(Staging)│   Model Swap     │
│  └─────────┘    └─────────┘                  │
│                                               │
│  Data: processed_data.csv                     │
│  Model: recommender_model.joblib              │
└───────────────────────────────────────────────┘
```

> 🎤 **Speaker Notes**: Here's our system architecture. The React frontend communicates with the FastAPI backend via REST APIs. The backend is managed by a ModelManager class that holds the active KNN model and data in memory. The key MLOps feature is the Blue/Green deployment — we maintain an active model serving live traffic while a new model can be trained in the background.

---

# Slide 14 — Blue/Green Deployment

## MLOps: Zero-Downtime Model Updates

### The Problem
- Retraining a model takes time
- Can't stop serving users during training

### Our Solution: Blue/Green at the Model Level

```
Time ─────────────────────────────────────────►

Blue (v1.0) ████████████████████████████
                                        │ SWAP
Green (v1.1)        [===TRAINING===]────▼
                                        │
Blue (v1.1) ─────────────────────────────████████
```

### How It Works
1. **Blue** (Active): Serving all live traffic
2. `/control/retrain` triggered → **Green** model trains in **background thread**
3. Training completes → **atomic reference swap**
4. Next request automatically uses the new model
5. **Thread lock** prevents concurrent retraining

> 🎤 **Speaker Notes**: Our Blue/Green deployment works at the model level. When retraining is triggered, a background thread trains a brand-new model from scratch — re-preprocessing data, incorporating feedback, and fitting a new KNN. Once done, the Python reference to the active model is atomically swapped. Python's GIL ensures this is thread-safe. Users experience zero downtime.

---

# Slide 15 — Feedback Loop

## Closed-Loop Learning Cycle

```
          ┌────────────┐
          │   USER     │
          │ likes song │
          └─────┬──────┘
                │
                ▼
       ┌────────────────┐
       │ POST /feedback │
       └────────┬───────┘
                │
                ▼
      ┌─────────────────┐
      │ feedback_data.csv│
      └────────┬────────┘
                │
                ▼
     ┌──────────────────┐
     │ Retrain triggered │
     │ → popularity +0.2 │
     │ for liked tracks  │
     └────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │ New Model Deployed │
    │ (Blue/Green swap)  │
    └────────┬──────────┘
             │
             ▼
     Better recommendations
             │
             └──────► back to USER
```

> 🎤 **Speaker Notes**: We implement a complete feedback loop. When users like songs, that feedback is stored in a CSV. During retraining, liked tracks receive a popularity boost of 0.2 on the scaled axis, subtly pulling future recommendations toward content the user has endorsed. The system can also be fully reset to baseline for experimentation.

---

# Slide 16 — Containerization & Cloud Deployment

## Production Deployment

### Local (Docker Compose)
```yaml
services:
  backend:   # Port 8000 — FastAPI
  frontend:  # Port 5173 — React
```
- Volumes persist `data/` and `models/`
- Environment variables for Spotify API keys

### Cloud (AWS)
| AWS Service | Role |
|:---|:---|
| **ECR** | Container image registry |
| **ECS Fargate** | Serverless container orchestration |
| **ALB** | Application Load Balancer (routing) |
| **EFS** | Persistent file storage for feedback & models |

- **Fargate** = no server management, auto-scaling
- **EFS mount** solves ephemeral filesystem problem

> 🎤 **Speaker Notes**: For deployment, Docker Compose handles local development with separate containers for frontend and backend. For production, we push images to AWS ECR and deploy on ECS Fargate — which is serverless, meaning no servers to manage. AWS EFS provides persistent storage so feedback data survives container restarts.

---

# Slide 17 — Evaluation Metrics

## 6-Metric Evaluation Framework

| # | Metric | What It Measures | Target |
|:---|:---|:---|:---|
| 1 | **Avg Cosine Similarity** | Are recs relevant to user taste? | > 0.7 |
| 2 | **Intra-List Diversity** | Are recs varied (not all the same)? | 0.2–0.5 |
| 3 | **Catalog Coverage** | Does the system use the full catalog? | > 1% |
| 4 | **Novelty Score** | Does it surface hidden gems? | > baseline |
| 5 | **Slider Sensitivity** | Do sliders actually change output? | Monotonic |
| 6 | **Hit Rate @ K** | Can it recover a known-liked song? | > 5% @ K=10 |

### Why These 6?
- Cover **all dimensions** of recommendation quality
- Not just accuracy — also diversity, novelty, interactivity
- Computed **programmatically** against the live model

> 🎤 **Speaker Notes**: We evaluate the system across six dimensions. Relevance checks if recommendations match user taste. Diversity ensures we don't just give the same song 10 times. Coverage checks if we use the whole catalog or just popular songs. Novelty measures if we surface hidden gems. Slider sensitivity proves our interactive controls actually work. And hit rate at K measures predictive power through leave-one-out evaluation.

---

# Slide 18 — Evaluation Results Detail

## Key Evaluation Metrics Explained

### 1. Relevance (Cosine Similarity)
- Build user centroid → query KNN → measure similarity of results to centroid
- Score > 0.9 = highly relevant

### 2. Slider Sensitivity Test
```
Energy Slider:  -1.0 → -0.5 → 0.0 → +0.5 → +1.0
Avg Energy:      0.12   0.28   0.45   0.63   0.81  ← MONOTONIC ✓
```
- Each slider tested at 5 offset levels
- Pass = ≥ 60% of consecutive pairs are non-decreasing

### 3. Hit Rate @ K (Leave-One-Out)
```
For each liked song:
  1. Remove it from profile
  2. Build centroid from remaining likes
  3. Query KNN for top-K
  4. Check if removed song appears → HIT
```
- Random baseline: 10/114,000 = 0.009%
- Our system: significantly higher → model has learned real patterns

> 🎤 **Speaker Notes**: Let me highlight three key metrics. The slider sensitivity test proves our sliders work — as we increase the energy offset from -1 to +1, the average energy in recommendations monotonically increases. The hit rate test uses leave-one-out cross-validation: we remove each liked song, rebuild the profile, and check if KNN can recover it. Our system vastly outperforms the random baseline, proving the model captures real musical preference patterns.

---

# Slide 19 — Live Demo / Screenshots

## System in Action

### Home Feed with Sonic Sliders
- Personalized recommendations with adjustable Energy, Danceability, Valence, Acousticness sliders
- Each song card shows artist, album, and explainability tag

### Song Radio
- Click the radio icon on any track → instant "more like this" playlist
- Switches from user-to-item to item-to-item inference

### Control Panel
- Model version display
- Retrain button → triggers Blue/Green background training
- Reset button → clears feedback and reverts to baseline

> 🎤 **Speaker Notes**: [DEMO TIME] Let me show you the system in action. On the home feed, you can see personalized recommendations with explainability tags. Watch what happens when I drag the Energy slider up — the recommendations shift toward high-intensity tracks in real-time. Now let me click the Radio icon on this song — it switches to item-to-item mode, showing songs with a similar sonic fingerprint.

---

# Slide 20 — Challenges & Limitations

## Challenges Faced

| Challenge | Solution |
|:---|:---|
| Cold start (no user history) | Popularity-based fallback ("Trending Now") |
| Feature scale imbalance | RobustScaler (median/IQR) |
| Same song, multiple albums | Semantic deduplication (name + artist) |
| Model update downtime | Blue/Green background thread swap |
| Ephemeral container storage | AWS EFS persistent mount |

## Known Limitations
- Content-based only — no collaborative signals (what similar users like)
- Brute-force KNN — would need ANN (FAISS/Annoy) for millions of tracks
- Audio features lack lyrical/cultural context
- Feedback loop is lightweight (popularity boost only)

> 🎤 **Speaker Notes**: We encountered several challenges. The cold start problem is solved with a popularity fallback. Feature scale imbalance is handled by RobustScaler. Model update downtime is eliminated with Blue/Green deployment. Key limitations include the lack of collaborative filtering and the brute-force search that would need upgrading for much larger catalogs.

---

# Slide 21 — Future Work

## Roadmap & Extensions

### Short-Term
- 📊 Add **A/B testing** framework between model versions
- 🎵 Integrate **Spotify Playback SDK** for in-app listening
- 📈 Dashboard for model performance monitoring

### Medium-Term
- 🤝 Hybrid approach: **content-based + collaborative filtering**
- 🚀 Replace brute-force with **FAISS** (Approximate Nearest Neighbors) for scale
- 🔀 **Multi-Armed Bandit** for explore-exploit slider defaults

### Long-Term
- 🧠 Embed tracks using **deep learning** (e.g., music audio embeddings)
- 🌍 Real-time streaming data pipeline with **Apache Kafka**
- 📱 Mobile app (React Native)

> 🎤 **Speaker Notes**: For future work, short-term we'd add A/B testing between model versions and Spotify playback integration. Medium-term, a hybrid approach combining our content-based filtering with collaborative filtering would significantly improve recommendations. Long-term, we'd move to deep learning embeddings for richer representations and a streaming pipeline for real-time data processing.

---

# Slide 22 — Conclusion

## Key Takeaways

1. ✅ **Interactive Recommendations**: Sonic Sliders transform a black-box algorithm into a user-controlled exploration tool

2. ✅ **Production-Grade MLOps**: Blue/Green deployment enables zero-downtime model updates with a closed-loop feedback cycle

3. ✅ **Comprehensive Evaluation**: 6-metric framework validates relevance, diversity, coverage, novelty, interactivity, and predictive power

4. ✅ **Full-Stack Delivery**: End-to-end system from data ingestion → preprocessing → training → serving → deployment → monitoring

### One-Line Summary
> We demonstrated that recommendation systems can be **transparent, interactive, and production-ready** — putting users in the driver's seat of music discovery.

> 🎤 **Speaker Notes**: To conclude — our project proves that recommendation systems don't have to be opaque. By exposing the feature space through Sonic Sliders, implementing proper MLOps with Blue/Green deployment, and validating across six evaluation metrics, we've built a system that is transparent, interactive, and production-ready. Thank you for your attention. I'm happy to take questions.

---

# Slide 23 — Q&A

## Thank You! 🎵

### Questions?

**Project Repository**: [GitHub Link]
**Tech Stack**: FastAPI + React + Scikit-Learn + Docker + AWS
**Dataset**: 114,000+ Spotify tracks, 13 audio features

> 🎤 **Speaker Notes**: Thank you! I'm happy to answer any questions about the methodology, implementation, or evaluation results.
