"""
Export the recommendation dot product as an ONNX model for Triton.
Same as triton_gpu/export_onnx.py — the exported model is the same,
only the Triton config changes (KIND_CPU vs KIND_GPU).
"""

import torch
import torch.nn as nn
import numpy as np

NUM_SONGS = 10000
EMBEDDING_DIM = 64

np.random.seed(42)
song_embeddings_np = np.random.randn(NUM_SONGS, EMBEDDING_DIM).astype(np.float32)


class RecommenderModel(nn.Module):
    def __init__(self, song_embeddings):
        super().__init__()
        self.song_embeddings = nn.Parameter(
            torch.tensor(song_embeddings), requires_grad=False
        )

    def forward(self, user_embedding):
        scores = torch.matmul(user_embedding, self.song_embeddings.T)
        return scores


model = RecommenderModel(song_embeddings_np)
model.eval()

dummy_input = torch.randn(1, EMBEDDING_DIM)

onnx_path = "model_repository/song_recommender/1/model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    input_names=["user_embedding"],
    output_names=["scores"],
    dynamic_axes={
        "user_embedding": {0: "batch_size"},
        "scores": {0: "batch_size"},
    },
)

print(f"ONNX model exported to {onnx_path}")
