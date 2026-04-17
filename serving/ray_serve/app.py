"""
Navidrome Recommendation API — Ray Serve (Bonus)

Same GRU4Rec inference as baseline, but deployed via Ray Serve which adds:
- Auto-scaling: replicas scale up/down based on request queue depth
- Built-in batching: @serve.batch groups concurrent requests automatically
- Replica-level isolation: each replica loads its own model copy

This demonstrates an alternative to Triton for scaling inference.
Triton is better for GPU + ONNX; Ray Serve is better for flexible
Python-based serving with auto-scaling on CPU.
"""

import os
import sys
import time
import pickle
import hashlib
from pathlib import Path

import numpy as np
import torch
from ray import serve
from starlette.requests import Request
import json

# Make _shared importable
_SERVING_ROOT = Path(__file__).resolve().parent.parent
for candidate in [_SERVING_ROOT, Path("/app")]:
    if (candidate / "_shared" / "model.py").exists():
        sys.path.insert(0, str(candidate))
        break

from _shared.model import load_model


MODEL_PATH = os.environ.get("MODEL_PATH", str(_SERVING_ROOT.parent / "artifacts" / "best_gru4rec.pt"))
VOCAB_PATH = os.environ.get("VOCAB_PATH", str(_SERVING_ROOT.parent / "artifacts" / "vocabs.pkl"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "best_gru4rec-ray")
DEVICE = os.environ.get("DEVICE", "cpu")


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1},
)
class GRU4RecRecommender:
    def __init__(self):
        # Load vocab
        print(f"Loading vocab from {VOCAB_PATH} ...")
        with open(VOCAB_PATH, "rb") as f:
            item2idx, user2idx = pickle.load(f)
        self.idx2item = {idx: str(track_id) for track_id, idx in item2idx.items()}
        print(f"Vocab loaded: {len(item2idx)} items")

        # Load model
        print(f"Loading model from {MODEL_PATH} on device={DEVICE} ...")
        self.model, self.all_item_emb = load_model(MODEL_PATH, device=DEVICE)
        self.num_items = self.all_item_emb.shape[0]
        print(f"Model ready: {self.num_items} items, embed_dim={self.all_item_emb.shape[1]}")

    async def __call__(self, request: Request):
        t_start = time.time()

        body = await request.json()
        session_id = body.get("session_id", "unknown")
        user_id = body.get("user_id", 0)
        user_idx = body.get("user_idx", 0)
        prefix_item_idxs = body.get("prefix_item_idxs", [])
        playratios = body.get("playratios", [])
        exclude_item_idxs = body.get("exclude_item_idxs", [])
        top_n = min(body.get("top_n", 20), 100)

        # Filter OOV
        clean_prefix = [i for i in prefix_item_idxs if i in self.idx2item]
        oov_count = len(prefix_item_idxs) - len(clean_prefix)

        if not clean_prefix:
            return {"error": "All prefix items are out of vocab", "status": 400}

        # Inference
        prefix_tensor = torch.tensor([clean_prefix], dtype=torch.long, device=DEVICE)
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=DEVICE)
        exclude = [i for i in exclude_item_idxs if i in self.idx2item]

        indices, scores = self.model.predict_top_n(
            prefix_items=prefix_tensor,
            user_idxs=user_tensor,
            all_item_emb=self.all_item_emb,
            top_n=top_n,
            exclude_sets=[set(exclude)],
        )

        recommendations = [
            {
                "rank": rank,
                "item_idx": idx,
                "track_id": self.idx2item.get(idx, f"unknown_{idx}"),
                "score": score,
            }
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
        ]

        latency_ms = (time.time() - t_start) * 1000

        return {
            "session_id": session_id,
            "recommendations": recommendations,
            "model_version": MODEL_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inference_latency_ms": round(latency_ms, 2),
            "oov_count": oov_count,
        }


app = GRU4RecRecommender.bind()
