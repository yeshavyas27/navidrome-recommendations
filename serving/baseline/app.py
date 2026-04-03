"""
Navidrome Recommendation API - Baseline Serving Option
Adapted from lab's Food Classification FastAPI endpoint.
Instead of classifying food images, we recommend songs using
embedding dot products.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import time

#App setup
app = FastAPI(
    title="Navidrome Recommendation API",
    description="API for generating song recommendations using embedding dot products",
    version="1.0.0"
)


#Request/Response models

class RecommendRequest(BaseModel):
    user_id: str                    # which user wants recommendations
    n_recommendations: int = 10     # how many songs to return, default 10

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list
    model_version: str
    generated_at: str
    inference_latency_ms: float


# Load model and embeddings
# Dummy embeddings for now
#   with open("model.pkl", "rb") as f:
#       model = pickle.load(f)
#       user_embeddings = model["user_embeddings"]
#       song_embeddings = model["song_embeddings"]

NUM_USERS = 500
NUM_SONGS = 10000
EMBEDDING_DIM = 64

np.random.seed(42)
user_embeddings = np.random.randn(NUM_USERS, EMBEDDING_DIM).astype(np.float32)
song_embeddings = np.random.randn(NUM_SONGS, EMBEDDING_DIM).astype(np.float32)

# Dummy song metadata (in production this comes from Navidrome's database)
song_metadata = [
    {"song_id": f"song_{i:05d}", "title": f"Song {i}", "artist": f"Artist {i % 50}",
     "album": f"Album {i % 200}", "genre": "rock"}
    for i in range(NUM_SONGS)
]


#Health check
@app.get("/health")
def health():
    return {"status": "ok"}


#Recommendation endpoint (lab: @app.post("/predict"))
@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):

    start_time = time.time()

    # Step 1: Map user_id to an index (lab: decode + preprocess image)
    user_idx = hash(request.user_id) % NUM_USERS

    # Step 2: Model inference — dot product (lab: output = model(image))
    # Multiplies user's 64-dim vector against all 10,000 song vectors
    # Result: 10,000 scores, one per song
    scores = user_embeddings[user_idx] @ song_embeddings.T

    # Step 3: Pick top N songs (lab: torch.argmax for top 1 class)
    top_k_indices = np.argsort(scores)[-request.n_recommendations:][::-1]

    # Step 4: Build response (lab: PredictionResponse(prediction=..., probability=...))
    recommendations = []
    for rank, idx in enumerate(top_k_indices, 1):
        meta = song_metadata[idx]
        recommendations.append({
            "rank": rank,
            "song_id": meta["song_id"],
            "title": meta["title"],
            "artist": meta["artist"],
            "album": meta["album"],
            "genre": meta["genre"],
            "score": float(scores[idx]),
            "method": "collaborative_filtering",
        })

    inference_latency_ms = (time.time() - start_time) * 1000

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_version="v0.1.0-dummy",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        inference_latency_ms=round(inference_latency_ms, 2),
    )
