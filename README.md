# 🎵 Spotify MLOps: Interactive Recommender System

A production-grade Recommendation System with **Blue/Green Deployment**, **Real-Time Personalization**, and a **"Spotify-like" UI**.

![App Screenshot](https://via.placeholder.com/800x400?text=Spotify+MLOps+Dashboard)

## 🚀 Key Features

### 1. Interactive Personalization (V5)
-   **Sonic Sliders**: Manually tune your feed in real-time. Want more energy? Drag the slider. The backend dynamically adjusts the recommendation vector.
-   **Song Radio**: Click the Radio icon on any track to switch from "User-to-Item" filtering to "Item-to-Item" filtering (infinite similar songs).

### 2. MLOps Logic
-   **Blue/Green Model Swapping**: Retrain the model in the background without downtime. The system hot-swaps the "Active Model" in memory.
-   **Feedback Loop**: User "Likes" are saved to `data/feedback_data.csv` and used for future retraining.
-   **Explainability**: Recommendations come with tags like "HIGH INTENSITY" or "CHILL ACOUSTIC" based on feature analysis.

## 🛠️ Tech Stack
-   **Frontend**: React (Vite), TailwindCSS, Framer Motion (Animations), Lucide (Icons).
-   **Backend**: FastAPI, Pandas, Scikit-Learn (Nearest Neighbors).
-   **DevOps**: Docker, Docker Compose, AWS ECR/ECS support.

## 🏃‍♂️ Quick Start (Docker)

The easiest way to run the full stack:

```bash
docker-compose up --build
```
Access the app at `http://localhost:5173`.

## 💻 Manual Setup (Local Dev)

**1. Backend**
```bash
cd spotify-mlops
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Architecture

1.  **Cold Start**: If no history exists, serves "Trending" (Popularity-based).
2.  **Profile Building**: As you like songs, a "Mean Vector" of your taste is built in 12-dimensional audio space.
3.  **Inference**:
    *   *Home Feed*: `KNN(User_Vector + Slider_Offsets, All_Songs)`
    *   *Radio*: `KNN(Seed_Song_Vector, All_Songs)`
4.  **Retraining**: Triggered via API `/control/retrain`. Re-reads CSVs, trains a new KNN, and replaces the singleton `active_model`.

## ☁️ Deployment
See [DEPLOYMENT.md](./DEPLOYMENT.md) for AWS ECR & ECS instructions.
See [NGROK_GUIDE.md](./NGROK_GUIDE.md) for sharing with friends via tunnels.

## 📂 Project Structure
```
├── api/             # FastAPI Endpoints
├── data/            # Dataset & Feedback CSVs
├── frontend/        # React App
├── models/          # Serialized Models (.joblib)
├── src/             # ML Core (Training, Preprocessing)
└── workflows/       # GitHub Actions / Automation
```
