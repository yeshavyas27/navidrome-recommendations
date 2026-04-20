"""
Enrich the audio bucket with ID3 metadata pulled from track_dict.parquet.

For each audio object under audio/<track_id>.mp3 in the Swift/MinIO bucket:
  1. Look up the track_id in track_dict.parquet on MinIO.
  2. Embed title + artist as ID3 tags into the mp3 bytes (via mutagen).
  3. Upload the enriched mp3 to audio_complete/<track_id>.mp3 (same bucket).

Rationale: the audio files Hashir has on Swift are identified only by
filename (track_id.mp3). When Salauat imports them into Navidrome or
anyone downloads them, there's no title/artist embedded. This script
produces a parallel folder where each mp3 IS self-describing.

Usage
-----
    export S3_ENDPOINT_URL=...       # Swift / MinIO endpoint
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...

    pip install boto3 pyarrow mutagen

    python scripts/enrich_audio.py \
        --audio-bucket audio-cache \
        --meta-bucket  navidrome-metadata \
        --concurrency  8
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import pickle  # noqa: F401
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import boto3
import pyarrow.parquet as pq
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp3 import MP3


log = logging.getLogger("enrich")


@dataclass
class Config:
    audio_bucket:      str
    audio_prefix:      str
    complete_prefix:   str
    meta_bucket:       str
    meta_key:          str
    concurrency:       int
    overwrite:         bool
    endpoint:          str
    access_key:        str
    secret_key:        str


def make_s3(cfg: Config):
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name="us-east-1",
    )


def load_track_meta(s3, cfg: Config) -> dict[str, dict]:
    log.info(f"Loading track_dict from s3://{cfg.meta_bucket}/{cfg.meta_key}")
    obj = s3.get_object(Bucket=cfg.meta_bucket, Key=cfg.meta_key)
    table = pq.read_table(io.BytesIO(obj["Body"].read()))
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


_TRACK_ID_RE = re.compile(r"([^/]+)\.mp3$", re.IGNORECASE)


def _track_id_from_key(key: str) -> str | None:
    m = _TRACK_ID_RE.search(key)
    return m.group(1) if m else None


def list_audio_objects(s3, cfg: Config) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cfg.audio_bucket, Prefix=cfg.audio_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".mp3"):
                keys.append(obj["Key"])
    log.info(f"Found {len(keys)} mp3s under s3://{cfg.audio_bucket}/{cfg.audio_prefix}")
    return keys


def _dest_key(cfg: Config, track_id: str) -> str:
    return f"{cfg.complete_prefix}{track_id}.mp3"


def _already_enriched(s3, cfg: Config, track_id: str) -> bool:
    try:
        s3.head_object(Bucket=cfg.audio_bucket, Key=_dest_key(cfg, track_id))
        return True
    except Exception:
        return False


def _embed_id3(audio_bytes: bytes, track_id: str, title: str, artist: str) -> bytes:
    """Write title/artist + MusicBrainz track id ID3 tags into mp3 bytes.

    musicbrainz_trackid matters: when Navidrome scans these files, it reads
    that tag into MediaFile.MbzRecordingID — which is what the recommendation
    seed loop + feedback scrobbler look at. Without this tag, the 30Music id
    never makes it into Navidrome's DB and the ML loop is broken on its
    upstream end even though the audio plays fine.
    """
    import os as _os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        tf.write(audio_bytes)
        tmp_path = tf.name
    try:
        try:
            mp3 = MP3(tmp_path, ID3=EasyID3)
        except ID3NoHeaderError:
            mp3 = MP3(tmp_path)
            mp3.add_tags(ID3=EasyID3)
            mp3 = MP3(tmp_path, ID3=EasyID3)
        if title:
            mp3["title"] = title
        if artist:
            mp3["artist"] = artist
        if track_id:
            # EasyID3 "musicbrainz_trackid" maps to TXXX:MusicBrainz Track Id,
            # which is what Navidrome reads into MbzRecordingID at scan time.
            mp3["musicbrainz_trackid"] = str(track_id)
        mp3.save()
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        _os.unlink(tmp_path)


def process_one(s3, cfg: Config, meta: dict, src_key: str, stats: dict) -> None:
    track_id = _track_id_from_key(src_key)
    if not track_id:
        stats["bad_name"] += 1
        return

    if not cfg.overwrite and _already_enriched(s3, cfg, track_id):
        stats["skipped_exists"] += 1
        return

    m = meta.get(str(track_id), {})
    title  = m.get("title", "")
    artist = m.get("artist", "")
    if not title and not artist:
        stats["no_metadata"] += 1
        # still copy, but without tags
    try:
        obj = s3.get_object(Bucket=cfg.audio_bucket, Key=src_key)
        body = obj["Body"].read()
    except Exception as e:
        log.warning(f"download failed {src_key}: {e}")
        stats["download_failed"] += 1
        return

    try:
        new_bytes = _embed_id3(body, track_id, title, artist)
    except Exception as e:
        log.warning(f"tag-write failed {src_key}: {e}")
        stats["tag_failed"] += 1
        # fall back to uploading the raw bytes unchanged
        new_bytes = body

    try:
        s3.put_object(
            Bucket=cfg.audio_bucket,
            Key=_dest_key(cfg, track_id),
            Body=new_bytes,
            ContentType="audio/mpeg",
            Metadata={"track_id": track_id, "title": title, "artist": artist},
        )
    except Exception as e:
        log.warning(f"upload failed {src_key}: {e}")
        stats["upload_failed"] += 1
        return

    stats["enriched"] += 1
    if stats["enriched"] % 100 == 0:
        log.info(f"progress: {stats}")


def run(cfg: Config) -> None:
    s3   = make_s3(cfg)
    meta = load_track_meta(s3, cfg)
    keys = list_audio_objects(s3, cfg)

    stats = {
        "enriched":        0,
        "skipped_exists":  0,
        "no_metadata":     0,
        "bad_name":        0,
        "download_failed": 0,
        "tag_failed":      0,
        "upload_failed":   0,
    }

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = [pool.submit(process_one, s3, cfg, meta, k, stats) for k in keys]
        for _ in as_completed(futures):
            pass

    log.info(f"DONE: {stats}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Enrich audio bucket with ID3 tags.")
    p.add_argument("--audio-bucket", default=os.environ.get("AUDIO_BUCKET", "audio-cache"))
    p.add_argument("--audio-prefix",    default="audio/")
    p.add_argument("--complete-prefix", default="audio_complete/")
    p.add_argument("--meta-bucket", default=os.environ.get("TRACK_META_BUCKET", "navidrome-metadata"))
    p.add_argument("--meta-key",    default=os.environ.get("TRACK_META_KEY", "track_dict.parquet"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-enrich tracks that already exist under audio_complete/.")
    p.add_argument("--log-level",   default="INFO")
    a = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, a.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    endpoint = os.environ.get("S3_ENDPOINT_URL", "")
    if not endpoint:
        raise SystemExit("S3_ENDPOINT_URL must be set (Swift/MinIO endpoint).")

    return Config(
        audio_bucket=a.audio_bucket,
        audio_prefix=a.audio_prefix,
        complete_prefix=a.complete_prefix,
        meta_bucket=a.meta_bucket,
        meta_key=a.meta_key,
        concurrency=a.concurrency,
        overwrite=a.overwrite,
        endpoint=endpoint,
        access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    )


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
