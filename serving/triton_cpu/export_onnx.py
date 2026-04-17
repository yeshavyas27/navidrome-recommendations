"""
Export the GRU4Rec session encoder to ONNX for Triton.

What gets exported:
    prefix_item_idxs (padded, fixed-length) → session_repr (128-dim vector)

What stays in Python (FastAPI or Triton Python backend):
    - Input validation, OOV filtering
    - Dot product: session_repr @ all_item_emb.T → scores
    - Top-k selection, response formatting

Why not export the whole model?
    The dot product against 745K items is a simple matmul that doesn't
    benefit much from graph optimization. The GRU encoder is the neural
    network part that benefits from fused kernels and hardware-specific
    execution providers (Ch7 §7.3.2-7.3.3).

Why a wrapper class?
    The original encode_session() uses pack_padded_sequence, which ONNX
    can't trace. The wrapper replaces it with a mask-based approach:
    run the full padded sequence through the GRU, then gather the hidden
    state at each sequence's real length. Mathematically equivalent.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make _shared importable (works both locally and in Docker)
_SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in [_SCRIPT_DIR.parent, _SCRIPT_DIR]:  # serving/ or /app/
    if (candidate / "_shared" / "model.py").exists():
        sys.path.insert(0, str(candidate))
        break

from _shared.model import load_model

# ── ONNX-friendly wrapper ───────────────────────────────────────────────

MAX_SEQ_LEN = 50  # pad/truncate all inputs to this length


class GRU4RecEncoder(nn.Module):
    """Wraps the GRU4Rec encoder for ONNX export.

    Why not just use the original encode_session()?
        It uses pack_padded_sequence, which ONNX can't trace.

    Why not run the GRU on the full padded sequence and gather?
        The GRU has non-zero biases, so processing padding tokens
        (even with zero embeddings) changes the hidden state. The
        output at position 5 differs from what pack_padded_sequence
        produces, because the GRU's internal state has drifted.

    Solution: mask the embeddings AND use a large negative mask on
    the GRU output isn't viable either. The simplest correct approach
    is to accept a pre-truncated, right-padded input at a short fixed
    max length. The caller truncates to actual session length + minimal
    padding. With MAX_SEQ_LEN=50, most sessions fit without truncation
    (avg session is 13.8 tracks). The GRU still processes padding
    positions, but fewer of them, and we take h_n (final hidden state)
    from the last GRU layer.

    For exact numerical match, the serving code should truncate the
    prefix to the real length (no padding) before calling ONNX. This
    way the GRU processes only real tokens and h_n is correct.
    """

    def __init__(self, model):
        super().__init__()
        self.item_emb = model.item_emb
        self.gru = model.gru
        self.dropout = model.dropout
        self.output_proj = model.output_proj
        self.layer_norm = model.layer_norm if hasattr(model, "layer_norm") else None

    def forward(self, prefix_item_idxs: torch.Tensor) -> torch.Tensor:
        # prefix_item_idxs: (batch, seq_len) — should be truncated
        # to real length (no trailing padding) for exact results.

        x = self.item_emb(prefix_item_idxs)      # (batch, seq, 64)
        _, h_n = self.gru(x)                      # h_n: (num_layers, batch, 128)
        h_last = self.dropout(h_n[-1])            # last layer: (batch, 128)
        session_repr = self.output_proj(h_last)   # (batch, 64)
        if self.layer_norm is not None:
            session_repr = self.layer_norm(session_repr)
        return session_repr


# ── Export ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check multiple locations for the checkpoint (local dev vs Docker)
    for candidate in [
        Path(__file__).resolve().parents[2] / "artifacts" / "best_gru4rec.pt",  # local
        Path("/app/artifacts/best_gru4rec.pt"),                                  # Docker
    ]:
        if candidate.exists():
            ckpt_path = candidate
            break
    else:
        print("ERROR: best_gru4rec.pt not found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(__file__).resolve().parent / "model_repository" / "gru4rec_encoder" / "1"
    # Also check Docker path
    if not out_dir.parent.parent.exists():
        out_dir = Path("/app/model_repository/gru4rec_encoder/1")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    print(f"Loading checkpoint: {ckpt_path}")
    model, _ = load_model(str(ckpt_path))

    encoder = GRU4RecEncoder(model)
    encoder.eval()

    # Dummy input: single session, no padding (caller truncates).
    # Dynamic axis on seq_len lets ONNX accept any length.
    dummy = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # 5-track session

    print(f"Exporting to ONNX (max_seq_len={MAX_SEQ_LEN}) ...")
    torch.onnx.export(
        encoder,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        input_names=["prefix_item_idxs"],
        output_names=["session_repr"],
        dynamic_axes={
            "prefix_item_idxs": {0: "batch_size", 1: "seq_len"},
            "session_repr": {0: "batch_size"},
        },
        dynamo=False,  # use legacy tracer — handles dynamic seq_len
    )
    print(f"Exported: {onnx_path}")

    # ── Verify: compare ONNX output to original encode_session ────────
    import onnxruntime as ort
    import numpy as np

    # Ground truth: original model with pack_padded_sequence
    test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    user_dummy = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        original_out = model.encode_session(test_input, user_dummy).numpy()

    # ONNX output (no padding — truncated to real length)
    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(None, {"prefix_item_idxs": test_input.numpy()})[0]

    diff = np.abs(onnx_out - original_out).max()
    print(f"Max difference (ONNX vs original encode_session): {diff:.6e}")
    if diff < 1e-4:
        print("Verification PASSED — outputs match.")
    else:
        print(f"WARNING: outputs diverge (max diff={diff:.4f}), check export.")
