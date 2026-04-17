"""
GRU4Rec serving class — matches Yesha's current training code.

Architecture (from .pt tensor shapes):
    num_items        = 745352
    num_users        = 0        (use_user_context is False)
    embedding_dim    = 64
    hidden_dim       = 128
    num_layers       = 1
    output_proj      = Linear(128 → 64, bias=False)
    layer_norm       = LayerNorm(64)

This class is a direct copy of train/gru4rec.py:GRU4Rec with only
the training-specific code removed. The new .pt (Apr 17) matches
this class exactly — no adapter needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


DEFAULT_CFG = {
    "embedding_dim":     64,
    "hidden_dim":        128,
    "num_layers":        1,
    "dropout":           0.0,   # irrelevant at inference — eval() disables it
    "embedding_dropout": 0.0,
    "use_user_context":  False,
}

DEFAULT_NUM_ITEMS = 745352
DEFAULT_NUM_USERS = 0


class GRU4Rec(nn.Module):
    """Session-based recommender. Encodes a sequence of item indices
    into a session representation, then scores all items via dot product."""

    def __init__(self, num_items: int, num_users: int, cfg: dict):
        super().__init__()
        self.use_user_context = cfg["use_user_context"]
        embed_dim  = cfg["embedding_dim"]
        hidden_dim = cfg["hidden_dim"]

        self.item_emb    = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(cfg.get("embedding_dropout", 0.0))

        gru_input_dim = embed_dim
        if self.use_user_context:
            self.user_emb = nn.Embedding(num_users + 1, embed_dim, padding_idx=0)
            gru_input_dim += embed_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=cfg["num_layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg["dropout"])
        self.output_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.layer_norm  = nn.LayerNorm(embed_dim)

    def encode_session(self, prefix_items: torch.Tensor, user_idxs: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(prefix_items)
        x = self.emb_dropout(x)

        if self.use_user_context:
            u = self.user_emb(user_idxs).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, u], dim=-1)

        lengths = (prefix_items != 0).sum(dim=1).cpu().clamp(min=1)

        packed        = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n        = self.gru(packed)
        h_last        = self.dropout(h_n[-1])
        session_repr  = self.layer_norm(self.output_proj(h_last))
        return session_repr

    @torch.no_grad()
    def predict_top_n(
        self,
        prefix_items: torch.Tensor,
        user_idxs: torch.Tensor,
        all_item_emb: torch.Tensor,
        top_n: int,
        exclude_sets: list,
    ) -> tuple[list, list]:
        """Batched top-N prediction.

        Returns (indices, scores):
            indices: list[list[int]] — 1-based item indices per row
            scores:  list[list[float]] — raw dot-product scores per row
        """
        session_repr = self.encode_session(prefix_items, user_idxs)
        logits       = session_repr @ all_item_emb.T

        for b, excl in enumerate(exclude_sets):
            for item_idx in excl:
                logits[b, item_idx - 1] = float("-inf")

        top = torch.topk(logits, top_n, dim=-1)
        indices = [[int(i) + 1 for i in row] for row in top.indices.cpu().numpy()]
        scores  = [[float(s) for s in row] for row in top.values.cpu().numpy()]
        return indices, scores


def load_model(
    ckpt_path: str,
    num_items: int = DEFAULT_NUM_ITEMS,
    num_users: int = DEFAULT_NUM_USERS,
    cfg: dict | None = None,
    device: str = "cpu",
) -> tuple[GRU4Rec, torch.Tensor]:
    """Load weights and return (model, all_item_emb).

    all_item_emb is the full item embedding matrix with the padding row
    (index 0) stripped — shape (num_items, embed_dim).
    """
    cfg = cfg or DEFAULT_CFG
    model = GRU4Rec(num_items=num_items, num_users=num_users, cfg=cfg)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    all_item_emb = model.item_emb.weight[1:].detach()

    return model, all_item_emb


if __name__ == "__main__":
    import sys, pathlib

    ckpt = pathlib.Path(__file__).resolve().parents[2] / "artifacts" / "best_gru4rec.pt"
    if not ckpt.exists():
        print(f"ERROR: {ckpt} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {ckpt} ...")
    model, all_item_emb = load_model(str(ckpt))
    print(f"OK. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
    print(f"Item embedding matrix: {tuple(all_item_emb.shape)} (num_items × embed_dim)")

    prefix = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    users  = torch.tensor([0], dtype=torch.long)
    idxs, scores = model.predict_top_n(prefix, users, all_item_emb, top_n=5, exclude_sets=[set()])
    print(f"Dummy inference OK.")
    print(f"  Top-5 indices: {idxs[0]}")
    print(f"  Top-5 scores:  {[round(s, 3) for s in scores[0]]}")
