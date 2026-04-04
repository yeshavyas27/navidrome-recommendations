"""
Navidrome Recommendation API - FAISS CPU Serving Option
Copied from baseline, with numpy dot product replaced by FAISS index search.
This is a MODEL-LEVEL optimization: same hardware, smarter search.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss  # <-- NEW: Facebook AI Similarity Search
import time

#App setup
app = FastAPI(
    title="Navidrome Recommendation API",
    description="API for generating song recommendations using FAISS index search",
    version="1.0.0"
)


#Request/Response models (same as baseline)

class RecommendRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list
    model_version: str
    generated_at: str
    inference_latency_ms: float


# Load model and embeddings (same as baseline)
NUM_USERS = 500
NUM_SONGS = 10000
EMBEDDING_DIM = 64

np.random.seed(42)
user_embeddings = np.random.randn(NUM_USERS, EMBEDDING_DIM).astype(np.float32)
song_embeddings = np.random.randn(NUM_SONGS, EMBEDDING_DIM).astype(np.float32)

# NEW: Build FAISS index from song embeddings
# This organizes songs into a searchable structure so we don't
# have to compare against all 10,000 every time
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)  # IP = Inner Product (dot product)
faiss_index.add(song_embeddings)                 # Add all songs to the index

# Dummy song metadata (same as baseline)
song_metadata = [
    {"song_id": f"song_{i:05d}", "title": f"Song {i}", "artist": f"Artist {i % 50}",
     "album": f"Album {i % 200}", "genre": "rock"}
    for i in range(NUM_SONGS)
]


#Health check
@app.get("/health")
def health():
    return {"status": "ok"}


#Recommendation endpoint
@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):

    start_time = time.time()

    # Step 1: Map user_id to an index (same as baseline)
    user_idx = hash(request.user_id) % NUM_USERS

    # Step 2: FAISS search (CHANGED from baseline)
    # Baseline did: scores = user_embeddings[user_idx] @ song_embeddings.T
    #               top_k = np.argsort(scores)[-n:][::-1]
    # FAISS does both in one optimized call:
    user_vector = user_embeddings[user_idx].reshape(1, -1)  # FAISS expects 2D input
    scores, top_k_indices = faiss_index.search(user_vector, request.n_recommendations)
    scores = scores[0]           # Remove extra dimension
    top_k_indices = top_k_indices[0]

    # Step 3: Build response (same as baseline)
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
            "score": float(scores[rank - 1]),
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
