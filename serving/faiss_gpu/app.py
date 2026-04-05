"""
Navidrome Recommendation API - FAISS GPU Serving Option
Copied from faiss_cpu, with FAISS GPU index instead of CPU index.
This is an INFRASTRUCTURE-LEVEL optimization: same model, same code logic,
but FAISS uses GPU for parallel similarity search.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import time
import threading

# FAISS GPU is not thread-safe — serialize concurrent searches
faiss_lock = threading.Lock()

#App setup
app = FastAPI(
    title="Navidrome Recommendation API",
    description="API for song recommendations using FAISS GPU index search",
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


# Load model and embeddings (same as faiss_cpu)
NUM_USERS = 500
NUM_SONGS = 10000
EMBEDDING_DIM = 64

np.random.seed(42)
user_embeddings = np.random.randn(NUM_USERS, EMBEDDING_DIM).astype(np.float32)
song_embeddings = np.random.randn(NUM_SONGS, EMBEDDING_DIM).astype(np.float32)

# CHANGED from faiss_cpu: Build FAISS index on GPU instead of CPU
# Step 1: Create a CPU index first (same as faiss_cpu)
cpu_index = faiss.IndexFlatIP(EMBEDDING_DIM)
# Step 2: Move it to GPU
gpu_resource = faiss.StandardGpuResources()
faiss_index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)  # 0 = GPU device 0
# Step 3: Add song embeddings to GPU index
faiss_index.add(song_embeddings)

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


#Recommendation endpoint (same as faiss_cpu, index is on GPU now)
@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):

    start_time = time.time()

    # Step 1: Map user_id to an index (same as baseline)
    user_idx = hash(request.user_id) % NUM_USERS

    # Step 2: FAISS GPU search (lock required — not thread-safe)
    user_vector = user_embeddings[user_idx].reshape(1, -1)
    with faiss_lock:
        scores, top_k_indices = faiss_index.search(user_vector, request.n_recommendations)
    scores = scores[0]
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
