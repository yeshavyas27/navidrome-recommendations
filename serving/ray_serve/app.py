"""
Navidrome Recommendation API - Ray Serve Option (Bonus)
Same FAISS logic as faiss_cpu, but deployed via Ray Serve instead of FastAPI.

Ray Serve adds:
- Auto-scaling (replicas scale up/down based on load)
- Built-in batching (groups requests for efficiency)
- Multi-model composition (can route between BPR-MF and BPR-kNN)
"""

import numpy as np
import faiss
import time

from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request
import json


@serve.deployment(
    num_replicas=2,           # Start with 2 replicas (auto-scales)
    ray_actor_options={"num_cpus": 0.5},  # Each replica uses 0.5 CPU
)
class SongRecommender:
    def __init__(self):
        # Same model setup as faiss_cpu
        NUM_USERS = 500
        NUM_SONGS = 10000
        EMBEDDING_DIM = 64

        np.random.seed(42)
        self.user_embeddings = np.random.randn(NUM_USERS, EMBEDDING_DIM).astype(np.float32)
        song_embeddings = np.random.randn(NUM_SONGS, EMBEDDING_DIM).astype(np.float32)

        # Build FAISS index
        self.faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.faiss_index.add(song_embeddings)
        self.num_users = NUM_USERS

        # Dummy metadata
        self.song_metadata = [
            {"song_id": f"song_{i:05d}", "title": f"Song {i}", "artist": f"Artist {i % 50}",
             "album": f"Album {i % 200}", "genre": "rock"}
            for i in range(NUM_SONGS)
        ]

    async def __call__(self, request: Request):
        start_time = time.time()

        body = await request.json()
        user_id = body.get("user_id", "user_0")
        n = body.get("n_recommendations", 10)

        user_idx = hash(user_id) % self.num_users

        # FAISS search (same as faiss_cpu)
        user_vector = self.user_embeddings[user_idx].reshape(1, -1)
        scores, top_k_indices = self.faiss_index.search(user_vector, n)
        scores = scores[0]
        top_k_indices = top_k_indices[0]

        recommendations = []
        for rank, idx in enumerate(top_k_indices, 1):
            meta = self.song_metadata[idx]
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

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "model_version": "v0.1.0-dummy",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "inference_latency_ms": round(inference_latency_ms, 2),
        }


# Bind and create the application
app = SongRecommender.bind()
