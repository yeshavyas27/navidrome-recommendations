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

# Track metadata: MinIO bucket with track_id → title, artist mapping.
TRACK_META_BUCKET = os.environ.get("TRACK_META_BUCKET", "navidrome-metadata")
TRACK_META_KEY    = os.environ.get("TRACK_META_KEY", "track_dict.parquet")

# Audio cache: Swift bucket holding track_id → mp3 bytes for /play streaming.
# Populated lazily on cache miss and by scripts/warmup_cache.py.
AUDIO_BUCKET        = os.environ.get("AUDIO_BUCKET", "audio-cache")
AUDIO_KEY_PREFIX    = os.environ.get("AUDIO_KEY_PREFIX", "audio/")
AUDIO_PRESIGN_TTL   = int(os.environ.get("AUDIO_PRESIGN_TTL", "3600"))
AUDIO_DOWNLOAD_TIMEOUT = int(os.environ.get("AUDIO_DOWNLOAD_TIMEOUT", "120"))

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

    if local_path.exists():
        log.info(f"Using cached model: {local_path}")
        return str(local_path)

    with open(local_path, "wb") as fh:
        s3.download_fileobj(bucket, key, fh)
    log.info(f"Downloaded model ({local_path.stat().st_size / 1e6:.1f} MB) → {local_path}")
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

    # Load track metadata (title, artist) from MinIO if available.
    # Uses MINIO_URL/USER/PASSWORD (same creds as model fetch).
    _meta_minio_url = MINIO_URL or os.environ.get("S3_ENDPOINT_URL", "")
    _meta_minio_user = MINIO_USER or os.environ.get("AWS_ACCESS_KEY_ID", "")
    _meta_minio_pass = MINIO_PASSWORD or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    state["track_meta"] = {}
    if _meta_minio_url:
        try:
            import boto3
            import pyarrow.parquet as pq
            import io as _io
            log.info(f"Loading track metadata from MinIO ({TRACK_META_BUCKET}/{TRACK_META_KEY}) ...")
            _s3 = boto3.client("s3",
                endpoint_url=_meta_minio_url,
                aws_access_key_id=_meta_minio_user,
                aws_secret_access_key=_meta_minio_pass,
                region_name="us-east-1",
            )
            _obj = _s3.get_object(Bucket=TRACK_META_BUCKET, Key=TRACK_META_KEY)
            _table = pq.read_table(_io.BytesIO(_obj["Body"].read()))
            _tids = _table.column("track_id")
            _titles = _table.column("title")
            _artists = _table.column("artist")
            track_meta = {}
            for i in range(len(_tids)):
                track_meta[_tids[i].as_py()] = {
                    "title": _titles[i].as_py() or "",
                    "artist": _artists[i].as_py() or "",
                }
            state["track_meta"] = track_meta
            log.info(f"Track metadata loaded: {len(track_meta)} tracks")
        except Exception as e:
            log.warning(f"Failed to load track metadata: {e}")

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


# ─── track-ID-based endpoint (for Navidrome integration) ────────────────
class TrackRecommendRequest(BaseModel):
    """Simpler input schema for callers that only know track IDs.
    The vocab translation (track_id → item_idx) happens server-side."""

    session_id:         str = "unknown"
    user_id:            int | str = 0
    track_ids:          list[str] = Field(min_length=1, max_length=MAX_PREFIX_LEN)
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

            if not clean_prefix:
                REQUESTS.labels(status="400").inc()
                raise HTTPException(
                    status_code=400,
                    detail="None of the provided track_ids exist in the model vocabulary.",
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

            blender = state["cold_start"]
            if blender is not None:
                indices, scores, alphas = blender.predict(
                    model=model,
                    prefix_items=prefix_tensor,
                    user_idxs=user_tensor,
                    all_item_emb=all_item_emb,
                    top_n=request.top_n,
                    exclude_sets=[exclude],
                )
                cs_alpha = alphas[0]
            else:
                indices, scores = model.predict_top_n(
                    prefix_items=prefix_tensor,
                    user_idxs=user_tensor,
                    all_item_emb=all_item_emb,
                    top_n=request.top_n,
                    exclude_sets=[exclude],
                )
                cs_alpha = 1.0

        if cs_alpha < 1.0:
            COLD_START_ACTIVATIONS.inc()
        log.info(f"request_id={request_id} cold_start_alpha={cs_alpha}")

        track_meta = state["track_meta"]
        recs = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
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
    """boto3 client against the Swift/MinIO endpoint. Cached in state."""
    if "audio_s3" in state:
        return state["audio_s3"]
    endpoint = MINIO_URL or os.environ.get("S3_ENDPOINT_URL", "")
    access   = MINIO_USER or os.environ.get("AWS_ACCESS_KEY_ID", "")
    secret   = MINIO_PASSWORD or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not endpoint:
        return None
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
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


def _fetch_audio_from_youtube(title: str, artist: str) -> bytes:
    """Run yt-dlp, return mp3 bytes. Raises on failure."""
    query = f"ytsearch1:{title} {artist}".strip()
    with tempfile.TemporaryDirectory() as td:
        out_template = str(Path(td) / "audio.%(ext)s")
        subprocess.run(
            [
                "yt-dlp", "--no-playlist", "--quiet", "--no-warnings",
                "-x", "--audio-format", "mp3", "--audio-quality", "5",
                "-o", out_template,
                query,
            ],
            check=True,
            capture_output=True,
            timeout=AUDIO_DOWNLOAD_TIMEOUT,
        )
        mp3_files = list(Path(td).glob("*.mp3"))
        if not mp3_files:
            raise RuntimeError("yt-dlp produced no mp3")
        return mp3_files[0].read_bytes()


@app.get("/play/{track_id}")
def play(track_id: str):
    """Redirect the browser to a presigned Swift URL for this track's audio.

    Cache miss → yt-dlp ingest from public sources → upload to Swift → redirect.
    Cache hit  → presign immediately.
    """
    s3 = _audio_s3_client()
    if s3 is None:
        raise HTTPException(status_code=503, detail="Audio storage not configured")

    if _audio_cached(s3, track_id):
        PLAY_CACHE_HITS.inc()
    else:
        PLAY_CACHE_MISSES.inc()
        meta = _lookup_track_meta(track_id)
        title  = meta.get("title", "").strip()
        artist = meta.get("artist", "").strip()
        if not title:
            raise HTTPException(status_code=404, detail=f"No metadata for track_id={track_id}")

        log.info(f"play cache-miss track_id={track_id} query='{title} — {artist}'")
        t0 = time.time()
        try:
            audio_bytes = _fetch_audio_from_youtube(title, artist)
        except subprocess.TimeoutExpired:
            PLAY_DOWNLOAD_ERRORS.labels(reason="timeout").inc()
            raise HTTPException(status_code=504, detail="Audio fetch timed out")
        except subprocess.CalledProcessError as e:
            PLAY_DOWNLOAD_ERRORS.labels(reason="yt_dlp_failed").inc()
            stderr = e.stderr.decode(errors="ignore")[:200] if e.stderr else str(e)
            log.warning(f"yt-dlp failed track_id={track_id}: {stderr}")
            raise HTTPException(status_code=502, detail="Audio fetch failed")
        except Exception as e:
            PLAY_DOWNLOAD_ERRORS.labels(reason="other").inc()
            log.exception(f"audio fetch error track_id={track_id}: {e}")
            raise HTTPException(status_code=500, detail="Audio fetch error")
        PLAY_DOWNLOAD_DURATION.observe(time.time() - t0)

        s3.put_object(
            Bucket=AUDIO_BUCKET,
            Key=_audio_cache_key(track_id),
            Body=audio_bytes,
            ContentType="audio/mpeg",
        )
        log.info(f"play cached track_id={track_id} size={len(audio_bytes)/1e6:.1f}MB")

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": AUDIO_BUCKET, "Key": _audio_cache_key(track_id)},
        ExpiresIn=AUDIO_PRESIGN_TTL,
    )
    return RedirectResponse(url=url, status_code=302)
