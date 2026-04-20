"""
Parallel audio cache warmup for /play endpoint.

Pre-fetches audio for the top-N most popular 30Music tracks into the Swift
audio-cache bucket, so that demo-day clicks on recommendations are cache
hits (instant) rather than misses (10s yt-dlp fetch).

Design notes
------------
* Runs OUTSIDE the serving container — on any machine with yt-dlp, ffmpeg,
  Swift creds, and network. Designed to run overnight on a teammate's box.
* Concurrency ceiling is network-bound, not CPU-bound. YouTube will throttle
  a single IP around 8–10 concurrent. Default is 8; raise at your own risk
  (past ~15 you'll start getting 429s). For real scale you need proxy
  rotation, which isn't wired in here.
* Resumable: does a HEAD on each object before downloading. Safe to kill
  and restart — already-cached tracks get skipped.
* Never raises per-track errors; logs and moves on. The goal is "warm as
  many tracks as we can overnight," not perfection.

Usage
-----
    export S3_ENDPOINT_URL=...     # Swift / MinIO endpoint
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...

    python scripts/warmup_cache.py \
        --popularity artifacts/popularity.npy \
        --vocab      artifacts/vocabs.pkl \
        --top-n      10000 \
        --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import numpy as np
import pyarrow.parquet as pq


log = logging.getLogger("warmup")


# ─── config ──────────────────────────────────────────────────────────────
@dataclass
class Config:
    popularity_path:    Path
    vocab_path:         Path
    top_n:              int
    concurrency:        int
    audio_bucket:       str
    audio_prefix:       str
    meta_bucket:        str
    meta_key:           str
    artifacts_bucket:   str
    popularity_key:     str
    vocab_key:          str
    checkpoint_key:     str
    per_track_timeout:  int
    progress_every:     int
    endpoint:           str = field(init=False)
    access_key:         str = field(init=False)
    secret_key:         str = field(init=False)

    def __post_init__(self):
        self.endpoint   = os.environ.get("S3_ENDPOINT_URL", "")
        self.access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        self.secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        if not self.endpoint:
            raise SystemExit("S3_ENDPOINT_URL must be set (Swift/MinIO endpoint).")


# ─── swift helpers ───────────────────────────────────────────────────────
def make_s3(cfg: Config):
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name="us-east-1",
    )


def audio_key(cfg: Config, track_id: str) -> str:
    return f"{cfg.audio_prefix}{track_id}.mp3"


def is_cached(s3, cfg: Config, track_id: str) -> bool:
    try:
        s3.head_object(Bucket=cfg.audio_bucket, Key=audio_key(cfg, track_id))
        return True
    except Exception:
        return False


# ─── input loading ───────────────────────────────────────────────────────
def _fetch_from_swift_if_missing(s3, cfg: Config, local: Path, bucket: str, key: str):
    if local.exists():
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Fetching s3://{bucket}/{key} → {local}")
    with open(local, "wb") as fh:
        s3.download_fileobj(bucket, key, fh)


def load_top_tracks(s3, cfg: Config) -> list[str]:
    """Return list of track_ids ordered by popularity (highest first)."""
    _fetch_from_swift_if_missing(s3, cfg, cfg.popularity_path,
                                 cfg.artifacts_bucket, cfg.popularity_key)
    _fetch_from_swift_if_missing(s3, cfg, cfg.vocab_path,
                                 cfg.artifacts_bucket, cfg.vocab_key)

    log.info(f"Loading popularity from {cfg.popularity_path}")
    pop = np.load(cfg.popularity_path)  # shape (num_items,), indexed by vocab idx

    log.info(f"Loading vocab from {cfg.vocab_path}")
    with open(cfg.vocab_path, "rb") as f:
        item2idx, _user2idx = pickle.load(f)
    idx2item = {idx: tid for tid, idx in item2idx.items()}

    order = np.argsort(pop)[::-1]
    top = []
    for idx in order:
        tid = idx2item.get(int(idx))
        if tid is not None:
            top.append(str(tid))
            if len(top) >= cfg.top_n:
                break
    log.info(f"Selected top {len(top)} tracks by popularity")
    return top


def load_track_meta(s3, cfg: Config) -> dict[str, dict]:
    log.info(f"Loading track metadata from s3://{cfg.meta_bucket}/{cfg.meta_key}")
    obj = s3.get_object(Bucket=cfg.meta_bucket, Key=cfg.meta_key)
    import io as _io
    table = pq.read_table(_io.BytesIO(obj["Body"].read()))
    tids    = table.column("track_id")
    titles  = table.column("title")
    artists = table.column("artist")
    meta: dict[str, dict] = {}
    for i in range(len(tids)):
        meta[str(tids[i].as_py())] = {
            "title":  (titles[i].as_py()  or "").strip(),
            "artist": (artists[i].as_py() or "").strip(),
        }
    log.info(f"Loaded metadata for {len(meta)} tracks")
    return meta


# ─── yt-dlp worker ───────────────────────────────────────────────────────
def fetch_audio(title: str, artist: str, timeout: int) -> bytes:
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
            timeout=timeout,
        )
        mp3s = list(Path(td).glob("*.mp3"))
        if not mp3s:
            raise RuntimeError("yt-dlp produced no mp3")
        return mp3s[0].read_bytes()


# ─── checkpointing ───────────────────────────────────────────────────────
def save_checkpoint(s3, cfg: Config, stats: dict):
    try:
        s3.put_object(
            Bucket=cfg.audio_bucket,
            Key=cfg.checkpoint_key,
            Body=json.dumps(stats).encode(),
            ContentType="application/json",
        )
    except Exception as e:
        log.warning(f"Checkpoint upload failed: {e}")


# ─── main worker loop ────────────────────────────────────────────────────
async def process_one(sem, s3, cfg, track_id, meta, stats):
    async with sem:
        if is_cached(s3, cfg, track_id):
            stats["already_cached"] += 1
            return

        m = meta.get(track_id, {})
        title  = m.get("title", "")
        artist = m.get("artist", "")
        if not title:
            stats["no_metadata"] += 1
            return

        t0 = time.time()
        try:
            audio_bytes = await asyncio.to_thread(fetch_audio, title, artist, cfg.per_track_timeout)
        except subprocess.TimeoutExpired:
            stats["timeout"] += 1
            return
        except subprocess.CalledProcessError as e:
            stats["yt_dlp_failed"] += 1
            err = (e.stderr.decode(errors="ignore") if e.stderr else "")[:120]
            log.debug(f"yt-dlp failed track_id={track_id} ({title!r}): {err}")
            return
        except Exception as e:
            stats["other_error"] += 1
            log.debug(f"error track_id={track_id}: {e}")
            return

        try:
            await asyncio.to_thread(
                s3.put_object,
                Bucket=cfg.audio_bucket,
                Key=audio_key(cfg, track_id),
                Body=audio_bytes,
                ContentType="audio/mpeg",
            )
        except Exception as e:
            stats["upload_failed"] += 1
            log.warning(f"upload failed track_id={track_id}: {e}")
            return

        stats["downloaded"] += 1
        stats["bytes"]      += len(audio_bytes)
        log.info(
            f"[{stats['downloaded']}/{cfg.top_n}] track_id={track_id} "
            f"size={len(audio_bytes)/1e6:.1f}MB dur={time.time()-t0:.1f}s"
        )


async def run(cfg: Config):
    s3 = make_s3(cfg)
    top   = load_top_tracks(s3, cfg)
    meta  = load_track_meta(s3, cfg)

    stats = {
        "started_at":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "top_n":           cfg.top_n,
        "concurrency":     cfg.concurrency,
        "downloaded":      0,
        "already_cached":  0,
        "no_metadata":     0,
        "timeout":         0,
        "yt_dlp_failed":   0,
        "upload_failed":   0,
        "other_error":     0,
        "bytes":           0,
    }

    stop = asyncio.Event()
    def on_signal(*_):
        log.warning("Signal received — finishing in-flight tasks then exiting.")
        stop.set()
    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    sem = asyncio.Semaphore(cfg.concurrency)
    pending: set[asyncio.Task] = set()

    for i, tid in enumerate(top):
        if stop.is_set():
            break
        task = asyncio.create_task(process_one(sem, s3, cfg, tid, meta, stats))
        pending.add(task)
        task.add_done_callback(pending.discard)

        if (i + 1) % cfg.progress_every == 0:
            save_checkpoint(s3, cfg, stats)
            log.info(f"checkpoint: {stats}")

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    stats["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_checkpoint(s3, cfg, stats)
    log.info(f"DONE: {stats}")


# ─── cli ─────────────────────────────────────────────────────────────────
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Parallel audio cache warmup.")
    p.add_argument("--popularity", type=Path, default=Path("artifacts/popularity.npy"))
    p.add_argument("--vocab",      type=Path, default=Path("artifacts/vocabs.pkl"))
    p.add_argument("--top-n",      type=int,  default=10000)
    p.add_argument("--concurrency", type=int, default=8,
                   help="Concurrent yt-dlp downloads (single-IP ceiling ~8–10).")
    p.add_argument("--audio-bucket", default=os.environ.get("AUDIO_BUCKET", "audio-cache"))
    p.add_argument("--audio-prefix", default="audio/")
    p.add_argument("--meta-bucket",  default=os.environ.get("TRACK_META_BUCKET", "navidrome-metadata"))
    p.add_argument("--meta-key",     default=os.environ.get("TRACK_META_KEY", "track_dict.parquet"))
    p.add_argument("--artifacts-bucket", default=os.environ.get("MINIO_BUCKET", "artifacts"),
                   help="Bucket holding popularity.npy + vocabs.pkl. Auto-downloaded if local paths missing.")
    p.add_argument("--popularity-key", default="popularity.npy")
    p.add_argument("--vocab-key",      default="vocabs.pkl")
    p.add_argument("--checkpoint-key", default="warmup/progress.json")
    p.add_argument("--per-track-timeout", type=int, default=120)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, a.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    return Config(
        popularity_path=a.popularity,
        vocab_path=a.vocab,
        top_n=a.top_n,
        concurrency=a.concurrency,
        audio_bucket=a.audio_bucket,
        audio_prefix=a.audio_prefix,
        meta_bucket=a.meta_bucket,
        meta_key=a.meta_key,
        artifacts_bucket=a.artifacts_bucket,
        popularity_key=a.popularity_key,
        vocab_key=a.vocab_key,
        checkpoint_key=a.checkpoint_key,
        per_track_timeout=a.per_track_timeout,
        progress_every=a.progress_every,
    )


if __name__ == "__main__":
    cfg = parse_args()
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        sys.exit(130)
