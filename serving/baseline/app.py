"""
Navidrome Recommendation API — baseline serving.

Wraps Yesha's GRU4Rec model (loaded via the adapter class in
`serving/_shared/model.py`) behind a FastAPI endpoint that speaks the
team's agreed-upon input schema (see samples/input_sample.json).

Responsibilities of this layer:
    1. Accept the session-history JSON from Navidrome.
    2. Translate between string track IDs ↔ integer vocab indices.
       (Stubbed today — real item2idx / idx2item pending from training team.)
    3. Call model.predict_top_n() for inference.
    4. Return a JSON response with recommendations.
    5. Emit Prometheus metrics (latency p50/p95, throughput, error rate,
       OOV rate) for the monitoring team.
    6. Apply safeguards: input validation, PII-safe logging,
       graceful OOV handling.

Model and vocab are loaded once at startup via a FastAPI lifespan hook.
When the real vocab lands, swap `_stub_idx2item` / `_stub_vocab_contains`
for a loaded JSON — nothing else needs to change.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pickle

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


# Make `serving/_shared` importable whether we run locally or via Docker.
_SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(_SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVING_ROOT))

from _shared.model import DEFAULT_CFG, DEFAULT_NUM_ITEMS, load_model  # noqa: E402


# ─── configuration ───────────────────────────────────────────────────────
MODEL_PATH     = os.environ.get(
    "MODEL_PATH",
    str(_SERVING_ROOT.parent / "artifacts" / "best_gru4rec.pt"),
)
VOCAB_PATH     = os.environ.get(
    "VOCAB_PATH",
    str(_SERVING_ROOT.parent / "artifacts" / "vocabs.pkl"),
)
MODEL_VERSION  = os.environ.get("MODEL_VERSION", "best_gru4rec")

# MLflow: if set, pull model artifact from MLflow instead of local file.
# Set MLFLOW_TRACKING_URI to enable (e.g. http://129.114.25.168:8000).
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT", "30music-session-recommendation")
MLFLOW_ARTIFACT     = os.environ.get("MLFLOW_ARTIFACT", "best_gru4rec.pt")
DEVICE         = os.environ.get("DEVICE", "cpu")
MAX_PREFIX_LEN = int(os.environ.get("MAX_PREFIX_LEN", "200"))
MAX_TOP_N      = int(os.environ.get("MAX_TOP_N", "100"))


# ─── logging (PII-safe) ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("serving.baseline")


def _hash_user(user_id: int | str) -> str:
    """Hash a user_id so we can correlate logs without storing raw PII."""
    return hashlib.sha256(str(user_id).encode()).hexdigest()[:12]


# ─── Prometheus metrics ──────────────────────────────────────────────────
REQUESTS = Counter(
    "recommend_requests_total",
    "Total /recommend requests, labelled by HTTP-style status.",
    ["status"],
)
LATENCY = Histogram(
    "recommend_latency_seconds",
    "End-to-end /recommend latency (seconds), for p50/p95/p99 derivation.",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
OOV_ITEMS = Counter(
    "recommend_oov_items_total",
    "Item indices in incoming requests that fall outside the vocab range.",
)
TOP_N_HIST = Histogram(
    "recommend_top_n",
    "Requested top_n values (for capacity planning).",
    buckets=(1, 5, 10, 20, 50, 100),
)
MODEL_INFO = Gauge(
    "recommend_model_info",
    "Metadata about the loaded model. Value is always 1; labels carry the info.",
    ["model_version", "num_items", "embedding_dim"],
)


# ─── request / response schema (matches samples/input_sample.json) ───────
class RecommendRequest(BaseModel):
    """Serving input schema — see samples/input_sample.json (v1.0)."""

    session_id:         str
    user_id:            int | str
    user_idx:           int = 0
    request_timestamp:  str | None = None

    prefix_track_ids:   list[int | str] = Field(default_factory=list)
    prefix_item_idxs:   Annotated[list[int], Field(min_length=1, max_length=MAX_PREFIX_LEN)]
    playratios:         list[float] = Field(default_factory=list)
    exclude_item_idxs:  list[int]   = Field(default_factory=list)

    top_n:              int = Field(default=20, ge=1, le=MAX_TOP_N)


class Recommendation(BaseModel):
    rank:     int
    item_idx: int
    track_id: str
    score:    float


class RecommendResponse(BaseModel):
    session_id:            str
    request_id:            str
    recommendations:       list[Recommendation]
    model_version:         str
    generated_at:          str
    inference_latency_ms:  float
    oov_count:             int


# ─── app state (populated on startup) ────────────────────────────────────
state: dict = {}


def _fetch_model_from_mlflow() -> str:
    """Download the latest model artifact from MLflow and return local path.

    Queries the MLflow experiment for the most recent run, then downloads
    the model .pt file to a local cache directory. This is what enables
    the retrain → redeploy loop: Yesha trains a new model → logs it to
    MLflow → this service pulls it on next startup.
    """
    import requests as http_requests

    log.info(f"Fetching model from MLflow at {MLFLOW_TRACKING_URI} ...")

    # Find the latest run in the experiment
    exp_resp = http_requests.get(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/get-by-name",
        params={"experiment_name": MLFLOW_EXPERIMENT},
    )
    exp_resp.raise_for_status()
    experiment_id = exp_resp.json()["experiment"]["experiment_id"]

    runs_resp = http_requests.post(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search",
        json={
            "experiment_ids": [experiment_id],
            "max_results": 1,
            "order_by": ["start_time DESC"],
        },
    )
    runs_resp.raise_for_status()
    runs = runs_resp.json().get("runs", [])
    if not runs:
        raise RuntimeError(f"No runs found in MLflow experiment '{MLFLOW_EXPERIMENT}'")

    run_id = runs[0]["info"]["run_id"]
    run_name = runs[0]["info"].get("run_name", run_id)
    log.info(f"Found MLflow run: {run_name} (id={run_id})")

    # Download the artifact
    cache_dir = Path("/tmp/mlflow_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{run_id}_{MLFLOW_ARTIFACT}"

    if local_path.exists():
        log.info(f"Using cached artifact: {local_path}")
        return str(local_path)

    artifact_url = (
        f"{MLFLOW_TRACKING_URI}/get-artifact"
        f"?run_id={run_id}&path={MLFLOW_ARTIFACT}"
    )
    log.info(f"Downloading {MLFLOW_ARTIFACT} from run {run_name} ...")
    resp = http_requests.get(artifact_url, stream=True)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info(f"Downloaded to {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")

    return str(local_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load vocab (item2idx, user2idx) from Yesha's training cache.
    log.info(f"Loading vocab from {VOCAB_PATH} ...")
    with open(VOCAB_PATH, "rb") as f:
        item2idx, user2idx = pickle.load(f)
    idx2item = {idx: str(track_id) for track_id, idx in item2idx.items()}
    state["item2idx"] = item2idx
    state["idx2item"] = idx2item
    log.info(f"Vocab loaded: {len(item2idx)} items, {len(user2idx)} users")

    # Load model: from MLflow if configured, otherwise from local file.
    model_path = MODEL_PATH
    if MLFLOW_TRACKING_URI:
        model_path = _fetch_model_from_mlflow()

    log.info(f"Loading model from {model_path} on device={DEVICE} ...")
    t0 = time.time()
    model, all_item_emb = load_model(model_path, device=DEVICE)
    elapsed = time.time() - t0

    state["model"]        = model
    state["all_item_emb"] = all_item_emb
    state["num_items"]    = all_item_emb.shape[0]
    state["embed_dim"]    = all_item_emb.shape[1]

    MODEL_INFO.labels(
        model_version=MODEL_VERSION,
        num_items=str(state["num_items"]),
        embedding_dim=str(state["embed_dim"]),
    ).set(1)

    log.info(
        f"Model ready in {elapsed:.2f}s | version={MODEL_VERSION} "
        f"num_items={state['num_items']} embed_dim={state['embed_dim']}"
    )
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Navidrome Recommendation API",
    description="Session-based next-track recommendations via GRU4Rec.",
    version="1.0.0",
    lifespan=lifespan,
)
# /metrics endpoint for Prometheus scrape
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest, http_request: Request):
    # 1) request-id for log/metric correlation (do this before anything
    #    that can fail so we have it in error logs).
    request_id = http_request.headers.get("x-request-id") or str(uuid.uuid4())

    t_start = time.time()
    TOP_N_HIST.observe(request.top_n)

    # 2) PII-safe log: hashed user, session id, and shapes only.
    user_hash = _hash_user(request.user_id)
    log.info(
        f"request_id={request_id} session_id={request.session_id} "
        f"user_hash={user_hash} prefix_len={len(request.prefix_item_idxs)} "
        f"top_n={request.top_n}"
    )

    try:
        with LATENCY.time():
            model        = state["model"]
            all_item_emb = state["all_item_emb"]

            # 3) safeguard: filter OOV indices instead of 500-ing.
            clean_prefix = [i for i in request.prefix_item_idxs if i in state["idx2item"]]
            oov_count    = len(request.prefix_item_idxs) - len(clean_prefix)
            if oov_count:
                OOV_ITEMS.inc(oov_count)
                log.warning(
                    f"request_id={request_id} dropped_oov_items={oov_count} "
                    f"remaining_prefix_len={len(clean_prefix)}"
                )
            if not clean_prefix:
                REQUESTS.labels(status="400").inc()
                raise HTTPException(
                    status_code=400,
                    detail="All prefix_item_idxs are out of vocab range.",
                )

            exclude = [i for i in request.exclude_item_idxs if i in state["idx2item"]]

            # 4) shape for the model: (batch=1, seq_len=len(clean_prefix))
            prefix_tensor = torch.tensor([clean_prefix], dtype=torch.long, device=DEVICE)
            user_tensor   = torch.tensor([request.user_idx], dtype=torch.long, device=DEVICE)

            indices, scores = model.predict_top_n(
                prefix_items=prefix_tensor,
                user_idxs=user_tensor,
                all_item_emb=all_item_emb,
                top_n=request.top_n,
                exclude_sets=[set(exclude)],
            )

        # 5) translate predicted integer indices → track id strings.
        recs = [
            Recommendation(
                rank=rank,
                item_idx=idx,
                track_id=state["idx2item"].get(idx, f"unknown_{idx}"),
                score=score,
            )
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
        ]

        REQUESTS.labels(status="200").inc()
        latency_ms = (time.time() - t_start) * 1000
        return RecommendResponse(
            session_id=request.session_id,
            request_id=request_id,
            recommendations=recs,
            model_version=MODEL_VERSION,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            inference_latency_ms=round(latency_ms, 2),
            oov_count=oov_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        REQUESTS.labels(status="500").inc()
        log.exception(f"request_id={request_id} inference_failed: {e}")
        raise HTTPException(status_code=500, detail="internal error")
