# Audio cache warmup — for Hashir

Pre-fetches audio for the top-N most popular 30Music tracks into the Swift
`audio-cache` bucket. Runs once, overnight, on any machine with network.
Demo-day clicks on those tracks become cache hits → instant playback.

## Setup

```bash
# Python deps
pip install yt-dlp boto3 numpy pyarrow

# ffmpeg is required by yt-dlp for mp3 extraction
brew install ffmpeg               # macOS
# sudo apt-get install ffmpeg     # linux

# Swift / MinIO credentials (from navidrome-platform secret)
export S3_ENDPOINT_URL=http://129.114.27.204
export AWS_ACCESS_KEY_ID=<...>
export AWS_SECRET_ACCESS_KEY=<...>
```

## Run

```bash
python scripts/warmup_cache.py --top-n 10000 --concurrency 8
```

On first run it auto-downloads `popularity.npy` and `vocabs.pkl` from the
`artifacts` bucket and `track_dict.parquet` from `navidrome-metadata`. No
file juggling needed.

## What it writes

- `audio-cache/audio/<track_id>.mp3` — one object per successfully fetched track
- `audio-cache/warmup/progress.json` — checkpoint, updated every 100 tracks

## Expected behavior

- ~8–15 hours for 10k tracks, ~50 GB Swift storage
- Safe to Ctrl-C and restart — already-cached tracks get skipped via HEAD
- `--concurrency` ceiling is ~10 per IP before YouTube 429s. Higher needs
  proxy rotation, which isn't wired in here.

## If something looks wrong

- `yt_dlp_failed` stats climbing → likely 429 rate-limits; lower concurrency
- `no_metadata` climbing → track_dict.parquet missing titles for those ids
- `upload_failed` → Swift credential / bucket issue
