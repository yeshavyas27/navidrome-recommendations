"""
Enrich Chameleon Swift audio files with ID3 tags for Navidrome.

For each audio/<track_id>.mp3 in the Swift container: look up the
track_id in track_dict.parquet (on MinIO), embed title + artist +
musicbrainz_trackid as ID3 tags in the mp3 bytes, and upload the
enriched file to audio_complete/<track_id>.mp3 in the same Swift
container. Also writes audio_complete/manifest.json listing every
track that was processed.

Navidrome's scanner (TagLib) reads ID3 tags from the file bytes — so
the embedding in step 2 is what makes "Track 1559341" become
"Real Song Title — Real Artist" in the UI after Navidrome scans.
The manifest.json is for human inspection.

Setup
-----
    pip install python-swiftclient python-keystoneclient boto3 pyarrow mutagen

    # Chameleon Swift (audio lives here)
    export OS_AUTH_URL=https://chi.uc.chameleoncloud.org:5000/v3
    export OS_AUTH_TYPE=v3applicationcredential
    export OS_APPLICATION_CREDENTIAL_ID=...
    export OS_APPLICATION_CREDENTIAL_SECRET=...

    # MinIO (track_dict.parquet lives here)
    export MINIO_URL=http://minio.navidrome-platform.svc.cluster.local:9000
    export MINIO_USER=...
    export MINIO_PASSWORD=...

Usage
-----
    python scripts/enrich_audio_swift.py --concurrency 8
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import boto3
import pyarrow.parquet as pq
from keystoneauth1.identity.v3 import ApplicationCredential
from keystoneauth1.session import Session
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp3 import MP3
from swiftclient.client import Connection as SwiftConnection
from swiftclient.exceptions import ClientException


log = logging.getLogger("enrich-swift")


@dataclass
class Config:
    swift_container: str
    audio_prefix:    str
    complete_prefix: str
    manifest_key:    str
    minio_meta_bucket: str
    minio_meta_key:    str
    concurrency:      int
    overwrite:        bool


_TRACK_ID_RE = re.compile(r"([^/]+)\.mp3$", re.IGNORECASE)


# ─── auth / clients ─────────────────────────────────────────────────────
def make_swift() -> SwiftConnection:
    auth = ApplicationCredential(
        auth_url=os.environ["OS_AUTH_URL"],
        application_credential_id=os.environ["OS_APPLICATION_CREDENTIAL_ID"],
        application_credential_secret=os.environ["OS_APPLICATION_CREDENTIAL_SECRET"],
    )
    return SwiftConnection(session=Session(auth=auth))


def make_minio_s3():
    url  = os.environ.get("MINIO_URL", "") or os.environ.get("S3_ENDPOINT_URL", "")
    user = os.environ.get("MINIO_USER", "") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    pw   = os.environ.get("MINIO_PASSWORD", "") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not url:
        raise SystemExit("MINIO_URL (or S3_ENDPOINT_URL) required to read track_dict.parquet.")
    return boto3.client(
        "s3",
        endpoint_url=url,
        aws_access_key_id=user,
        aws_secret_access_key=pw,
        region_name="us-east-1",
    )


# ─── data loading ───────────────────────────────────────────────────────
def load_track_dict(s3, cfg: Config) -> dict[str, dict]:
    log.info(f"Loading track_dict from s3://{cfg.minio_meta_bucket}/{cfg.minio_meta_key}")
    obj = s3.get_object(Bucket=cfg.minio_meta_bucket, Key=cfg.minio_meta_key)
    table = pq.read_table(io.BytesIO(obj["Body"].read()))
    tids    = table.column("track_id")
    titles  = table.column("title")
    artists = table.column("artist")
    out: dict[str, dict] = {}
    for i in range(len(tids)):
        out[str(tids[i].as_py())] = {
            "title":  (titles[i].as_py()  or "").strip(),
            "artist": (artists[i].as_py() or "").strip(),
        }
    log.info(f"Loaded metadata for {len(out)} tracks")
    return out


def list_swift_audio(swift: SwiftConnection, cfg: Config) -> list[str]:
    log.info(f"Listing swift://{cfg.swift_container}/{cfg.audio_prefix}*")
    keys: list[str] = []
    marker = ""
    while True:
        _h, page = swift.get_container(
            cfg.swift_container, prefix=cfg.audio_prefix, marker=marker, limit=1000,
        )
        if not page:
            break
        for obj in page:
            if obj["name"].lower().endswith(".mp3"):
                keys.append(obj["name"])
        marker = page[-1]["name"]
    log.info(f"Found {len(keys)} mp3s to enrich")
    return keys


def _track_id_from_key(key: str) -> str | None:
    m = _TRACK_ID_RE.search(key)
    return m.group(1) if m else None


def _dest_key(cfg: Config, track_id: str) -> str:
    return f"{cfg.complete_prefix}{track_id}.mp3"


def _already_enriched(swift: SwiftConnection, cfg: Config, track_id: str) -> bool:
    try:
        swift.head_object(cfg.swift_container, _dest_key(cfg, track_id))
        return True
    except ClientException:
        return False


# ─── ID3 embedding (identical to enrich_audio.py) ───────────────────────
def embed_id3(audio_bytes: bytes, track_id: str, title: str, artist: str) -> bytes:
    """Write title/artist + musicbrainz_trackid ID3 tags; return new bytes.

    musicbrainz_trackid matters: Navidrome's scanner reads it into
    MediaFile.MbzRecordingID, which the recommendation seed loop and
    feedback scrobbler both depend on.
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
            mp3["musicbrainz_trackid"] = str(track_id)
        mp3.save()
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        _os.unlink(tmp_path)


# ─── per-object worker ──────────────────────────────────────────────────
def process_one(swift: SwiftConnection, cfg: Config, meta: dict,
                src_key: str, stats: dict, manifest: dict,
                lock: threading.Lock) -> None:
    track_id = _track_id_from_key(src_key)
    if not track_id:
        stats["bad_name"] += 1
        return

    if not cfg.overwrite and _already_enriched(swift, cfg, track_id):
        stats["skipped_exists"] += 1
        m = meta.get(track_id, {})
        with lock:
            manifest[track_id] = {
                "track_id":     track_id,
                "title":        m.get("title", ""),
                "artist":       m.get("artist", ""),
                "source_key":   src_key,
                "enriched_key": _dest_key(cfg, track_id),
                "status":       "already_enriched",
            }
        return

    m = meta.get(track_id, {})
    title  = m.get("title", "")
    artist = m.get("artist", "")
    if not title and not artist:
        stats["no_metadata"] += 1

    try:
        _h, body = swift.get_object(cfg.swift_container, src_key)
    except ClientException as e:
        stats["download_failed"] += 1
        log.warning(f"download failed {src_key}: {e}")
        return

    try:
        new_bytes = embed_id3(body, track_id, title, artist)
        tags_written = True
    except Exception as e:
        stats["tag_failed"] += 1
        log.warning(f"tag-write failed {src_key}: {e}")
        new_bytes = body
        tags_written = False

    try:
        swift.put_object(
            container=cfg.swift_container,
            obj=_dest_key(cfg, track_id),
            contents=new_bytes,
            content_type="audio/mpeg",
            headers={
                "X-Object-Meta-Track-Id": str(track_id),
            },
        )
    except ClientException as e:
        stats["upload_failed"] += 1
        log.warning(f"upload failed {src_key}: {e}")
        return

    with lock:
        manifest[track_id] = {
            "track_id":             track_id,
            "title":                title,
            "artist":               artist,
            "musicbrainz_trackid":  track_id,
            "source_key":           src_key,
            "enriched_key":         _dest_key(cfg, track_id),
            "id3_tags_written":     tags_written,
            "status":               "enriched",
        }

    stats["enriched"] += 1
    if stats["enriched"] % 50 == 0:
        log.info(f"progress: {stats}")


def upload_manifest(swift: SwiftConnection, cfg: Config, manifest: dict,
                    stats: dict) -> None:
    doc = {
        "version":      1,
        "container":    cfg.swift_container,
        "prefix":       cfg.complete_prefix,
        "track_count":  len(manifest),
        "stats":        stats,
        "tracks":       manifest,
    }
    body = json.dumps(doc, indent=2, default=str).encode()
    log.info(
        f"Uploading manifest → swift://{cfg.swift_container}/{cfg.manifest_key} "
        f"({len(body)/1024:.1f} KB)"
    )
    swift.put_object(
        container=cfg.swift_container,
        obj=cfg.manifest_key,
        contents=body,
        content_type="application/json",
    )


# ─── cli ────────────────────────────────────────────────────────────────
def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--swift-container", default="navidrome-bucket-proj05")
    p.add_argument("--audio-prefix",    default="audio/")
    p.add_argument("--complete-prefix", default="audio_complete/")
    p.add_argument("--manifest-key",    default="audio_complete/manifest.json")
    p.add_argument("--minio-meta-bucket", default=os.environ.get("TRACK_META_BUCKET", "navidrome-metadata"))
    p.add_argument("--minio-meta-key",    default=os.environ.get("TRACK_META_KEY", "track_dict.parquet"))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-enrich tracks already under audio_complete/.")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, a.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    return Config(
        swift_container=a.swift_container,
        audio_prefix=a.audio_prefix,
        complete_prefix=a.complete_prefix,
        manifest_key=a.manifest_key,
        minio_meta_bucket=a.minio_meta_bucket,
        minio_meta_key=a.minio_meta_key,
        concurrency=a.concurrency,
        overwrite=a.overwrite,
    )


def main() -> None:
    cfg = parse_args()
    swift = make_swift()
    s3    = make_minio_s3()

    meta = load_track_dict(s3, cfg)
    keys = list_swift_audio(swift, cfg)

    stats = {
        "enriched":        0,
        "skipped_exists":  0,
        "no_metadata":     0,
        "bad_name":        0,
        "download_failed": 0,
        "tag_failed":      0,
        "upload_failed":   0,
    }
    manifest: dict[str, dict] = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futs = [
            pool.submit(process_one, swift, cfg, meta, k, stats, manifest, lock)
            for k in keys
        ]
        for _ in as_completed(futs):
            pass

    log.info(f"DONE: {stats}")
    upload_manifest(swift, cfg, manifest, stats)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
