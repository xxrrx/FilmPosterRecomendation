"""
app.py — FastAPI entry point
Chay: uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import numpy as np
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db
from recommender import Recommender
from routers import movies, recommend, genres, clusters

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tat ca models vao RAM khi server khoi dong."""
    print("[Startup] Dang load models...")

    # Load numpy arrays (dung cho cosine similarity)
    combined   = np.load(os.path.join(MODELS_DIR, "combined_features.npy"))
    cnn        = np.load(os.path.join(MODELS_DIR, "cnn_features.npy"))
    tfidf_mat  = np.load(os.path.join(MODELS_DIR, "tfidf_matrix.npy"))
    movie_ids  = np.load(os.path.join(MODELS_DIR, "movie_ids.npy"))
    clusters   = np.load(os.path.join(MODELS_DIR, "cluster_labels.npy"))

    # Load ML models
    nb_model   = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
    mlb        = joblib.load(os.path.join(MODELS_DIR, "mlb_encoder.pkl"))

    print(f"[Startup] combined_features: {combined.shape}")
    print(f"[Startup] movie_ids: {movie_ids.shape}")

    # Khoi tao Recommender
    app.state.recommender = Recommender(
        combined_features = combined,
        movie_ids         = movie_ids,
        cnn_features      = cnn,
        tfidf_matrix      = tfidf_mat,
        cluster_labels    = clusters,
    )

    # Khoi tao DB (import CSV neu lan dau)
    init_db(nb_model=nb_model, mlb=mlb, cnn_features=cnn)

    print("[Startup] San sang! Truy cap http://localhost:8000/docs")
    yield
    print("[Shutdown] Server dung.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Movie Recommendation API",
    description = "Goi y phim da phuong tien: CNN poster + TF-IDF text + K-Means + Naive Bayes + Association Rules",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — cho phep frontend React (localhost:5173) goi API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://localhost:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Dang ky routers
app.include_router(movies.router)
app.include_router(recommend.router)
app.include_router(genres.router)
app.include_router(clusters.router)


# ─── Root endpoint ────────────────────────────────────────────────────────────

@app.get("/", tags=["root"])
def root():
    return {
        "message": "Movie Recommendation API",
        "docs":    "http://localhost:8000/docs",
        "endpoints": [
            "GET /api/movies",
            "GET /api/movies/{id}",
            "GET /api/movies/{id}/recommend",
            "GET /api/movies/search?q=",
            "GET /api/genres/rules",
            "GET /api/clusters",
            "GET /api/clusters/{cluster_id}/movies",
        ]
    }
