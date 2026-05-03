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
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pickle

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


# Make `serving/_shared` importable whether we run locally or via Docker.
_SERVING_ROOT = Path(__file__).resolve().parent.parent
if str(_SERVING_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVING_ROOT))

from _shared.model import DEFAULT_CFG, DEFAULT_NUM_ITEMS, load_model  # noqa: E402
from _shared.cold_start import ColdStartBlender  # noqa: E402


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

# Track metadata: bucket holding track_id → title, artist parquet.
# Per-resource S3 endpoint (defaults to MINIO_*) — set TRACK_META_ENDPOINT_URL
# to point at Chameleon RGW (https://chi.uc.chameleoncloud.org:7480) using
# EC2-style credentials so the parquet is read straight from Swift.
TRACK_META_BUCKET         = os.environ.get("TRACK_META_BUCKET", "navidrome-metadata")
TRACK_META_KEY            = os.environ.get("TRACK_META_KEY", "track_dict.parquet")
TRACK_META_ENDPOINT_URL   = os.environ.get("TRACK_META_ENDPOINT_URL", "")
TRACK_META_ACCESS_KEY     = os.environ.get("TRACK_META_ACCESS_KEY", "")
TRACK_META_SECRET_KEY     = os.environ.get("TRACK_META_SECRET_KEY", "")

# Audio: bucket holding track_id → mp3. Same per-resource override pattern.
# Set AUDIO_ENDPOINT_URL to Chameleon RGW + AUDIO_KEY_PREFIX="audio_complete/"
# + AUDIO_BUCKET="navidrome-bucket-proj05" to stream straight from Swift.
AUDIO_BUCKET            = os.environ.get("AUDIO_BUCKET", "audio-cache")
AUDIO_KEY_PREFIX        = os.environ.get("AUDIO_KEY_PREFIX", "audio/")
AUDIO_ENDPOINT_URL      = os.environ.get("AUDIO_ENDPOINT_URL", "")
AUDIO_ACCESS_KEY        = os.environ.get("AUDIO_ACCESS_KEY", "")
AUDIO_SECRET_KEY        = os.environ.get("AUDIO_SECRET_KEY", "")
AUDIO_PRESIGN_TTL   = int(os.environ.get("AUDIO_PRESIGN_TTL", "3600"))
AUDIO_DOWNLOAD_TIMEOUT = int(os.environ.get("AUDIO_DOWNLOAD_TIMEOUT", "120"))
# Demo fallback: when a recommended track_id has no audio uploaded, play this
# one instead. The UI still shows the correct title/artist for the clicked row.
AUDIO_FALLBACK_TRACK_ID = os.environ.get("AUDIO_FALLBACK_TRACK_ID", "1040741")

# MLflow: if set, pull model artifact from MLflow instead of local file.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT", "30music-session-recommendation")
MLFLOW_ARTIFACT     = os.environ.get("MLFLOW_ARTIFACT", "best_gru4rec.pt")

# MinIO: if MINIO_URL is set, pull latest model from MinIO at startup.
# MINIO_MODEL_KEY overrides auto-discovery (e.g. "finetune/2026-04-19/abc/model.pt").
# MINIO_MODEL_RUN_TYPE controls which run type to search: "finetune" (default) or "pretrain".
MINIO_URL           = os.environ.get("MINIO_URL", "")
MINIO_USER          = os.environ.get("MINIO_USER", "")
MINIO_PASSWORD      = os.environ.get("MINIO_PASSWORD", "")
MINIO_MODEL_KEY     = os.environ.get("MINIO_MODEL_KEY", "")
MINIO_MODEL_RUN_TYPE = os.environ.get("MINIO_MODEL_RUN_TYPE", "finetune")
DEVICE              = os.environ.get("DEVICE", "cpu")
MAX_PREFIX_LEN      = int(os.environ.get("MAX_PREFIX_LEN", "200"))
MAX_TOP_N           = int(os.environ.get("MAX_TOP_N", "100"))
POPULARITY_PATH         = os.environ.get(
    "POPULARITY_PATH",
    str(_SERVING_ROOT.parent / "artifacts" / "popularity.npy"),
)
COLD_START_RAMP         = int(os.environ.get("COLD_START_RAMP", "3"))
# MinIO coords for popularity — set MINIO_URL/USER/PASSWORD + this key to pull at startup
MINIO_POPULARITY_KEY    = os.environ.get("MINIO_POPULARITY_KEY", "")
MINIO_BUCKET            = os.environ.get("MINIO_BUCKET", "artifacts")


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
COLD_START_ACTIVATIONS = Counter(
    "recommend_cold_start_activations_total",
    "Requests where cold-start blending was applied (alpha < 1.0).",
)
PLAY_CACHE_HITS = Counter(
    "play_cache_hits_total",
    "/play requests served from Swift cache (no yt-dlp fetch).",
)
PLAY_CACHE_MISSES = Counter(
    "play_cache_misses_total",
    "/play requests that triggered a live audio fetch.",
)
PLAY_DOWNLOAD_DURATION = Histogram(
    "play_download_seconds",
    "yt-dlp fetch + Swift upload time on cache miss.",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120),
)
PLAY_DOWNLOAD_ERRORS = Counter(
    "play_download_errors_total",
    "/play cache-miss fetches that failed.",
    ["reason"],
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
    title:    str = ""
    artist:   str = ""
    score:    float


class RecommendResponse(BaseModel):
    session_id:            str
    request_id:            str
    recommendations:       list[Recommendation]
    model_version:         str
    generated_at:          str
    inference_latency_ms:  float
    oov_count:             int
    cold_start_alpha:      float   # 0.0 = pure popularity, 1.0 = pure GRU4Rec


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


def _fetch_model_from_minio() -> str:
    """Download latest finetune (or pretrain) model.pt from MinIO.

    Uses MINIO_MODEL_KEY if set explicitly, otherwise auto-discovers the
    most recently uploaded model under MINIO_MODEL_RUN_TYPE/.
    Returns local path to the downloaded file.
    """
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASSWORD,
        region_name="us-east-1",
    )
    bucket = MINIO_BUCKET

    key = MINIO_MODEL_KEY
    if not key:
        # auto-discover latest model
        paginator = s3.get_paginator("list_objects_v2")
        best_key, best_ts = None, None

        for run_type in [MINIO_MODEL_RUN_TYPE, "pretrain"]:
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{run_type}/"):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith("/model.pt"):
                        if best_ts is None or obj["LastModified"] > best_ts:
                            best_key = obj["Key"]
                            best_ts  = obj["LastModified"]
            if best_key:
                break

        if not best_key:
            raise RuntimeError(f"No model.pt found in MinIO bucket '{bucket}'")
        key = best_key

    log.info(f"Downloading model from MinIO: s3://{bucket}/{key}")
    cache_dir = Path("/tmp/minio_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / key.replace("/", "_")

    # Record the resolved key so /version can report what's actually loaded.
    # The cron-finetune integration test greps the /version response for the
    # key it just deployed via `kubectl set env MINIO_MODEL_KEY=...`.
    state["loaded_model_key"] = key

    if local_path.exists():
        log.info(f"Using cached model: {local_path}")
        return str(local_path)

    with open(local_path, "wb") as fh:
        s3.download_fileobj(bucket, key, fh)
    log.info(f"Downloaded model ({local_path.stat().st_size / 1e6:.1f} MB) → {local_path}")
    return str(local_path)


def _resolve_s3(prefix_url, prefix_access, prefix_secret):
    """Pick a per-resource S3 endpoint, falling back to the global MINIO_*.

    The 'prefix_*' args are env-var-resolved values for one resource (track
    meta or audio). If any is empty, the corresponding MINIO_* value is used.
    Falls back further to AWS_* env vars for parity with boto3 conventions.
    """
    endpoint = prefix_url or MINIO_URL or os.environ.get("S3_ENDPOINT_URL", "")
    access = prefix_access or MINIO_USER or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret = prefix_secret or MINIO_PASSWORD or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    return endpoint, access, secret


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

    # Load track metadata (title, artist) from any S3-compatible endpoint.
    # Set TRACK_META_ENDPOINT_URL/_ACCESS_KEY/_SECRET_KEY to point at Swift's
    # RGW (https://chi.uc.chameleoncloud.org:7480) — Swift is the durable
    # source of truth and the bucket+key default to the Chameleon layout.
    # Falls back to MINIO_* if the per-resource override isn't set.
    # Keys are forced to str to match idx2item downstream — parquet track_id
    # column is int64, idx2item returns strings, so without normalization
    # every lookup would miss and the UI would show empty names.
    state["track_meta"] = {}
    _parquet_bytes = None
    _meta_endpoint, _meta_access, _meta_secret = _resolve_s3(
        TRACK_META_ENDPOINT_URL, TRACK_META_ACCESS_KEY, TRACK_META_SECRET_KEY,
    )
    if _meta_endpoint:
        try:
            import boto3
            log.info(
                f"Loading track metadata: s3://{TRACK_META_BUCKET}/{TRACK_META_KEY} "
                f"from {_meta_endpoint}"
            )
            _s3 = boto3.client("s3",
                endpoint_url=_meta_endpoint,
                aws_access_key_id=_meta_access,
                aws_secret_access_key=_meta_secret,
                region_name=os.environ.get("AWS_DEFAULT_REGION", "default"),
            )
            _obj = _s3.get_object(Bucket=TRACK_META_BUCKET, Key=TRACK_META_KEY)
            _parquet_bytes = _obj["Body"].read()
        except Exception as e:
            log.error(
                f"track_dict load failed: {type(e).__name__}: {e}. "
                f"Bucket={TRACK_META_BUCKET} Key={TRACK_META_KEY} "
                f"Endpoint={_meta_endpoint}"
            )

    if _parquet_bytes is None:
        log.error(
            "Track metadata NOT loaded — no S3 endpoint reachable. "
            "Recommendations will have no title/artist."
        )
    else:
        try:
            import pyarrow.parquet as pq
            import io as _io
            _table = pq.read_table(_io.BytesIO(_parquet_bytes))
            _cols = set(_table.column_names)
            for _req in ("track_id", "title", "artist"):
                if _req not in _cols:
                    raise RuntimeError(
                        f"track_dict.parquet missing required column '{_req}'. "
                        f"Found columns: {sorted(_cols)}"
                    )
            _tids = _table.column("track_id")
            _titles = _table.column("title")
            _artists = _table.column("artist")
            track_meta = {}
            for i in range(len(_tids)):
                track_meta[str(_tids[i].as_py())] = {
                    "title": _titles[i].as_py() or "",
                    "artist": _artists[i].as_py() or "",
                }
            state["track_meta"] = track_meta
            log.info(
                f"Track metadata loaded: {len(track_meta)} tracks "
                f"(keys normalized to str)"
            )
        except Exception as e:
            log.error(
                f"Track metadata parse FAILED: {type(e).__name__}: {e}. "
                f"Recommendations will have no names."
            )

    # Load model: MinIO > MLflow > local file.
    model_path = MODEL_PATH
    if MINIO_URL:
        model_path = _fetch_model_from_minio()
    elif MLFLOW_TRACKING_URI:
        model_path = _fetch_model_from_mlflow()

    log.info(f"Loading model from {model_path} on device={DEVICE} ...")
    t0 = time.time()
    model, all_item_emb = load_model(model_path, device=DEVICE)
    elapsed = time.time() - t0

    state["model"]        = model
    state["all_item_emb"] = all_item_emb
    state["num_items"]    = all_item_emb.shape[0]
    state["embed_dim"]    = all_item_emb.shape[1]

    # Cold-start blender — fetch popularity.npy from MinIO if not already local
    if not Path(POPULARITY_PATH).exists() and MINIO_POPULARITY_KEY:
        log.info(f"Fetching popularity.npy from MinIO (key={MINIO_POPULARITY_KEY}) ...")
        try:
            import boto3
            s3 = boto3.client(
                "s3",
                endpoint_url=os.environ["MINIO_URL"],
                aws_access_key_id=os.environ["MINIO_USER"],
                aws_secret_access_key=os.environ["MINIO_PASSWORD"],
                region_name="us-east-1",
            )
            Path(POPULARITY_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(POPULARITY_PATH, "wb") as fh:
                s3.download_fileobj(MINIO_BUCKET, MINIO_POPULARITY_KEY, fh)
            log.info(f"Downloaded popularity.npy → {POPULARITY_PATH}")
        except Exception as e:
            log.warning(f"MinIO popularity download failed: {e} — cold-start disabled.")

    if Path(POPULARITY_PATH).exists():
        log.info(f"Loading popularity scores from {POPULARITY_PATH} (ramp={COLD_START_RAMP}) ...")
        state["cold_start"] = ColdStartBlender.from_file(POPULARITY_PATH, ramp_sessions=COLD_START_RAMP)
        log.info("Cold-start blender ready.")
    else:
        state["cold_start"] = None
        log.warning(f"popularity.npy not found at {POPULARITY_PATH} — cold-start disabled.")

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


@app.get("/version")
def version():
    """Report the model version + the actual MinIO/Swift key currently loaded.

    Used by the cron-finetune integration test to verify the serving pod
    picked up the model that was just deployed via
    `kubectl set env MINIO_MODEL_KEY=...`. The test greps the response
    body for the expected key string, so plain JSON works.
    """
    return {
        "model_version":     MODEL_VERSION,
        "loaded_model_key":  state.get("loaded_model_key", ""),
        "model_loaded":      "model" in state,
    }


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

            blender = state["cold_start"]
            if blender is not None:
                indices, scores, alphas = blender.predict(
                    model=model,
                    prefix_items=prefix_tensor,
                    user_idxs=user_tensor,
                    all_item_emb=all_item_emb,
                    top_n=request.top_n,
                    exclude_sets=[set(exclude)],
                )
                cs_alpha = alphas[0]
            else:
                indices, scores = model.predict_top_n(
                    prefix_items=prefix_tensor,
                    user_idxs=user_tensor,
                    all_item_emb=all_item_emb,
                    top_n=request.top_n,
                    exclude_sets=[set(exclude)],
                )
                cs_alpha = 1.0

        if cs_alpha < 1.0:
            COLD_START_ACTIVATIONS.inc()
        log.info(f"request_id={request_id} cold_start_alpha={cs_alpha}")

        # 5) translate predicted integer indices → track id strings + metadata.
        track_meta = state["track_meta"]
        recs = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            tid = state["idx2item"].get(idx, f"unknown_{idx}")
            meta = track_meta.get(tid, {})
            recs.append(Recommendation(
                rank=rank,
                item_idx=idx,
                track_id=tid,
                title=meta.get("title", ""),
                artist=meta.get("artist", ""),
                score=score,
            ))

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
            cold_start_alpha=cs_alpha,
        )

    except HTTPException:
        raise
    except Exception as e:
        REQUESTS.labels(status="500").inc()
        log.exception(f"request_id={request_id} inference_failed: {e}")
        raise HTTPException(status_code=500, detail="internal error")


# MMR diversity re-ranking. Model produces a popularity-biased ranking
# (popular tracks have dominant embeddings → keep showing up regardless of
# user). MMR fixes this at inference: greedily pick the next track that
# maximises (lambda * relevance) - (1 - lambda) * max-similarity-to-already-picked,
# which forces diversity in the final top-N. Deterministic, ~10ms, no retrain.
MMR_LAMBDA      = float(os.environ.get("MMR_LAMBDA", "0.5"))   # 1.0 = no diversity, 0.0 = pure diversity
MMR_CANDIDATES  = int(os.environ.get("MMR_CANDIDATES", "50"))  # how many to pull from model before re-rank


def _mmr_rerank(indices, scores, all_item_emb, top_n: int, lambda_: float = MMR_LAMBDA):
    """Greedy MMR over the model's candidate ranking. Returns reordered
    (indices, scores) trimmed to top_n. Original model scores are preserved
    in the output — only the ORDER changes — so logged scores stay meaningful.
    """
    K = len(indices)
    if K <= top_n:
        return indices, scores

    # Cosine similarity matrix over candidates (K, K).
    cand_emb = all_item_emb[list(indices)]
    cand_norm = cand_emb / (cand_emb.norm(dim=-1, keepdim=True) + 1e-9)
    sim = (cand_norm @ cand_norm.T).cpu().numpy()

    rel = list(scores)
    selected = [int(max(range(K), key=lambda i: rel[i]))]  # start with most relevant
    remaining = set(range(K)) - set(selected)

    while len(selected) < top_n and remaining:
        best_i, best_score = None, -float("inf")
        for i in remaining:
            div_penalty = max(sim[i, j] for j in selected)
            mmr = lambda_ * rel[i] - (1.0 - lambda_) * float(div_penalty)
            if mmr > best_score:
                best_score, best_i = mmr, i
        selected.append(best_i)
        remaining.discard(best_i)

    out_indices = [indices[i] for i in selected]
    out_scores  = [scores[i]  for i in selected]
    return out_indices, out_scores


# ─── track-ID-based endpoint (for Navidrome integration) ────────────────
class TrackRecommendRequest(BaseModel):
    """Simpler input schema for callers that only know track IDs.
    The vocab translation (track_id → item_idx) happens server-side."""

    session_id:         str = "unknown"
    user_id:            int | str = 0
    # min_length=0 lets brand-new users (no plays yet) reach the handler;
    # the cold-start seed fallback below samples from popularity so the
    # response is "popular tracks" instead of "no recommendations yet".
    track_ids:          list[str] = Field(default_factory=list, max_length=MAX_PREFIX_LEN)
    exclude_track_ids:  list[str] = Field(default_factory=list)
    top_n:              int = Field(default=20, ge=1, le=MAX_TOP_N)


@app.post("/recommend-by-tracks", response_model=RecommendResponse)
def recommend_by_tracks(request: TrackRecommendRequest, http_request: Request):
    """Accept raw 30Music track IDs, translate to vocab indices internally,
    run inference, and return recommendations as track IDs.

    This is the endpoint Navidrome (or any frontend) should call —
    no need to know about vocab indices.
    """
    request_id = http_request.headers.get("x-request-id") or str(uuid.uuid4())
    t_start = time.time()

    log.info(
        f"request_id={request_id} session_id={request.session_id} "
        f"user_hash={_hash_user(request.user_id)} "
        f"num_tracks={len(request.track_ids)} top_n={request.top_n}"
    )

    try:
        with LATENCY.time():
            model        = state["model"]
            all_item_emb = state["all_item_emb"]
            item2idx     = state["item2idx"]
            idx2item     = state["idx2item"]

            # Translate track IDs → vocab indices, skip unknown tracks
            clean_prefix = []
            oov_count = 0
            for tid in request.track_ids:
                # item2idx keys are numpy int64, so try int conversion
                try:
                    key = int(tid)
                except (ValueError, TypeError):
                    key = tid
                if key in item2idx:
                    clean_prefix.append(item2idx[key])
                else:
                    oov_count += 1

            if oov_count:
                OOV_ITEMS.inc(oov_count)
                log.warning(f"request_id={request_id} unknown_tracks={oov_count}")

            # Cold-start seed fallback: if no usable seeds (empty input, or all
            # OOV — e.g. Navidrome library has no 30Music-tagged tracks yet),
            # inject a random sample from the top-100 most popular items so the
            # model has something to condition on. We track this so we can
            # honestly report cs_alpha=0.0 in the response — otherwise the
            # blender sees a 3-track prefix and ramps alpha to 1.0, which
            # would lie about whether the result is personalized.
            used_seed_fallback = False
            if not clean_prefix:
                used_seed_fallback = True
                blender = state.get("cold_start")
                if blender is not None and getattr(blender, "_pop", None) is not None:
                    # _pop is 0-indexed; vocab item_idx is 1-based (idx = position + 1).
                    import random as _random
                    top_k = min(100, blender._pop.shape[0])
                    top_positions = torch.topk(blender._pop, k=top_k).indices.tolist()
                    sampled = _random.sample(top_positions, k=min(3, top_k))
                    clean_prefix = [p + 1 for p in sampled if (p + 1) in idx2item]
                    log.info(
                        f"request_id={request_id} cold_start_seed=popularity "
                        f"sampled={len(clean_prefix)}"
                    )
                if not clean_prefix:
                    REQUESTS.labels(status="400").inc()
                    raise HTTPException(
                        status_code=400,
                        detail="No usable seeds and no popularity fallback available.",
                    )

            # Translate exclude track IDs too
            exclude = set()
            for tid in request.exclude_track_ids:
                try:
                    key = int(tid)
                except (ValueError, TypeError):
                    key = tid
                if key in item2idx:
                    exclude.add(item2idx[key])

            prefix_tensor = torch.tensor([clean_prefix], dtype=torch.long, device=DEVICE)
            user_tensor   = torch.tensor([0], dtype=torch.long, device=DEVICE)

            # Pull a wider candidate set than the user asked for, then
            # MMR-rerank to the requested top_n. Without this, popular
            # embeddings dominate every response → "same 10 songs" across
            # users (Yesha + Salauat reported this).
            internal_top_n = max(MMR_CANDIDATES, request.top_n)

            blender = state["cold_start"]
            if blender is not None:
                indices, scores, alphas = blender.predict(
                    model=model,
                    prefix_items=prefix_tensor,
                    user_idxs=user_tensor,
                    all_item_emb=all_item_emb,
                    top_n=internal_top_n,
                    exclude_sets=[exclude],
                )
                cs_alpha = alphas[0]
            else:
                indices, scores = model.predict_top_n(
                    prefix_items=prefix_tensor,
                    user_idxs=user_tensor,
                    all_item_emb=all_item_emb,
                    top_n=internal_top_n,
                    exclude_sets=[exclude],
                )
                cs_alpha = 1.0

            # If the prefix was synthesised from popularity (this user had
            # no real plays), the blender's alpha is meaningless — it just
            # reflects the synthetic prefix length. Force alpha=0.0 so the
            # response honestly reports "popularity, not personalisation."
            # This is computed per-request: a different user with real plays
            # in the same instant gets their own alpha unmolested.
            if used_seed_fallback:
                cs_alpha = 0.0

            # Diversity rerank if we have headroom.
            cand_indices = list(indices[0])
            cand_scores  = list(scores[0])
            if len(cand_indices) > request.top_n:
                cand_indices, cand_scores = _mmr_rerank(
                    cand_indices, cand_scores, all_item_emb,
                    top_n=request.top_n, lambda_=MMR_LAMBDA,
                )

        if cs_alpha < 1.0:
            COLD_START_ACTIVATIONS.inc()
        log.info(
            f"request_id={request_id} cold_start_alpha={cs_alpha} "
            f"mmr_lambda={MMR_LAMBDA} candidate_pool={internal_top_n}"
        )

        track_meta = state["track_meta"]
        recs = []
        for rank, (idx, score) in enumerate(zip(cand_indices, cand_scores), start=1):
            tid = idx2item.get(idx, f"unknown_{idx}")
            meta = track_meta.get(tid, {})
            recs.append(Recommendation(
                rank=rank,
                item_idx=idx,
                track_id=tid,
                title=meta.get("title", ""),
                artist=meta.get("artist", ""),
                score=score,
            ))

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
            cold_start_alpha=cs_alpha,
        )

    except HTTPException:
        raise
    except Exception as e:
        REQUESTS.labels(status="500").inc()
        log.exception(f"request_id={request_id} inference_failed: {e}")
        raise HTTPException(status_code=500, detail="internal error")


# ─── audio cache (/play) ─────────────────────────────────────────────────
def _audio_s3_client():
    """boto3 client for the audio bucket — Swift's RGW S3 in production.

    Uses the per-resource AUDIO_ENDPOINT_URL / AUDIO_ACCESS_KEY / AUDIO_SECRET_KEY
    when set (typical: Chameleon RGW with EC2 creds, container=navidrome-bucket-proj05,
    prefix=audio_complete/). Falls back to MINIO_* for local/legacy setups.
    Cached in state.
    """
    if "audio_s3" in state:
        return state["audio_s3"]
    endpoint, access, secret = _resolve_s3(
        AUDIO_ENDPOINT_URL, AUDIO_ACCESS_KEY, AUDIO_SECRET_KEY,
    )
    if not endpoint:
        return None
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "default"),
    )
    state["audio_s3"] = client
    return client


def _lookup_track_meta(track_id: str) -> dict:
    """track_meta keys may be int or str depending on parquet dtype."""
    meta = state.get("track_meta", {})
    if track_id in meta:
        return meta[track_id]
    try:
        return meta.get(int(track_id), {})
    except (ValueError, TypeError):
        return {}


def _audio_cache_key(track_id: str) -> str:
    return f"{AUDIO_KEY_PREFIX}{track_id}.mp3"


def _audio_cached(s3, track_id: str) -> bool:
    try:
        s3.head_object(Bucket=AUDIO_BUCKET, Key=_audio_cache_key(track_id))
        return True
    except Exception:
        return False


def _list_audio_keys(s3, max_keys: int = 1000) -> list[str]:
    """List up to max_keys audio object keys under the AUDIO_KEY_PREFIX."""
    try:
        resp = s3.list_objects_v2(
            Bucket=AUDIO_BUCKET, Prefix=AUDIO_KEY_PREFIX, MaxKeys=max_keys
        )
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as e:
        log.warning(f"audio bucket list failed: {e}")
        return []


@app.get("/play/{track_id}")
def play(track_id: str):
    """Redirect the browser to a presigned Swift URL for this track's audio.

    Cache hit  → presigned URL for the track's own mp3.
    Cache miss → presigned URL for a random mp3 in the bucket (demo
                 fallback — we only have ~2k audio files from the catalog
                 but the UI still shows the correct title/artist from
                 track_dict for whatever row was clicked).
    """
    s3 = _audio_s3_client()
    if s3 is None:
        raise HTTPException(status_code=503, detail="Audio storage not configured")

    key = _audio_cache_key(track_id)
    if _audio_cached(s3, track_id):
        PLAY_CACHE_HITS.inc()
    else:
        PLAY_CACHE_MISSES.inc()
        keys = _list_audio_keys(s3)
        if not keys:
            raise HTTPException(
                status_code=404,
                detail="No audio available in cache bucket",
            )
        import random as _random
        key = _random.choice(keys)
        log.info(f"play fallback track_id={track_id} -> {key}")

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": AUDIO_BUCKET, "Key": key},
        ExpiresIn=AUDIO_PRESIGN_TTL,
    )
    return RedirectResponse(url=url, status_code=302)
