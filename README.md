# Navidrome Recommendations — Serving Layer

Session-based music recommendation serving for a forked Navidrome deployment,
powered by a GRU4Rec model trained on the 30Music dataset. FastAPI in front,
Kubernetes underneath, MLflow + MinIO + Prometheus keeping it honest.

This repo is the serving half of a group project for **ECE-GY 9183: ML Systems
Design & Operations** at NYU. Three repos, four people, one Kubernetes cluster
on Chameleon Cloud.

---

## Team

| Member | Role | Primary surface |
|---|---|---|
| **Vanshika Bagaria** | Serving + MLOps | This repo + `navidrome_mlops` Go/React integration |
| **Yesha Vyas** | Training | `train/` in `navidrome_mlops` (GRU4Rec, finetune pipeline, safeguard metrics) |
| **Hashir Muzaffar** | Data | `data/` in `navidrome_mlops` (30Music parser, feedback API, datasets) |
| **Salauat Kakimzhanov** | Platform / DevOps | `salawhaaat/navidrome-iac` (K8s, Argo workflows, monitoring) |

Write access is cross-shared where it makes sense. Each of us owns a primary
surface but the repos aren't walled off.

---

## The three repos

| Repo | What it is | Branch to track |
|---|---|---|
| **[yeshavyas27/navidrome-recommendations](https://github.com/yeshavyas27/navidrome-recommendations)** *(this one)* | FastAPI serving code, scripts for audio enrichment + cache warmup, model inference layer | `main` |
| **[yeshavyas27/navidrome_mlops](https://github.com/yeshavyas27/navidrome_mlops)** | Forked Navidrome (Go + React) with the recommendation UI integration, plus `train/` and `data/` pipelines | `navidrome-custom` |
| **[salawhaaat/navidrome-iac](https://github.com/salawhaaat/navidrome-iac)** | Kubernetes manifests, Argo workflows, Prometheus/Grafana/Alertmanager config, CI/CD | `main` |

The Navidrome fork pulls this serving repo in as a git submodule, so when the
platform CI builds the serving image it uses the SHA pinned on `navidrome-custom`.
That's how a change here lands in the cluster without a separate image tag.

---

## Architecture in one picture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Navidrome web UI (React)                        │
│  ┌─────────────┐                                                         │
│  │ Recs page   │───GET /api/recommendation──┐                           │
│  │ (play btn)  │                            │                           │
│  └─────────────┘                            ▼                           │
└──────────────────────────────────────────┬───────────────────────────────┘
                                           │ (inside Navidrome Go binary)
                                           │
                             ┌─────────────▼─────────────┐
                             │ recommendations.go handler│
                             │  - random-sample MediaFiles│
                             │  - extract 30Music track_id│
                             │    from filename           │
                             └─────────────┬─────────────┘
                                           │
                                           ▼ POST /recommend-by-tracks
                             ┌───────────────────────────────────┐
                             │   FastAPI serving (this repo)     │
                             │   port 8080 inside k8s            │
                             │   - vocab lookup                  │
                             │   - GRU4Rec inference (CPU)       │
                             │   - cold-start blender            │
                             │   - Prometheus /metrics           │
                             └─┬──────┬────────────────────────┬──┘
                               │      │                        │
                 MinIO (S3) ◄──┘      └── track_dict ──►       └── /play
                 - models                parquet lookup            - Swift audio
                 - vocabs                (title, artist)           - random fallback
                 - popularity.npy

 MLflow tracks all training runs │ Pushgateway receives safeguard metrics
 Prometheus scrapes /metrics     │ Grafana renders dashboards
 Alertmanager fires on thresholds│ TRIGGERS.md is the policy doc
```

---

## What this repo does

### `serving/baseline/app.py`

A FastAPI app. Loads GRU4Rec weights + vocab + track metadata at startup.
Serves four endpoints:

- `POST /recommend` — full session-aware recommendation. Expects a prefix of
  vocab indices; returns top-N ranked items with scores.
- `POST /recommend-by-tracks` — **the one Navidrome calls.** Accepts raw 30Music
  track IDs, translates them to vocab indices internally, runs inference, and
  returns ranked recommendations enriched with title/artist from
  `track_dict.parquet`.
- `GET /play/{track_id}` — audio proxy. Checks the configured Swift/MinIO
  bucket for `audio_complete/<id>.mp3`; if present, redirects the browser to
  a presigned URL. On cache miss, redirects to a random mp3 in the bucket
  (demo fallback — the UI still shows the correct title because the metadata
  came from the recommendation response).
- `GET /metrics` — Prometheus exposition.

The startup flow is worth knowing:

1. Load vocab from `/app/artifacts/vocabs.pkl` (local, baked in).
2. Load `track_dict.parquet` from MinIO (keys normalized to `str` so lookups
   actually hit — that was a silent bug once).
3. If `MINIO_URL` is set, pull the latest `finetune/<date>/<run_id>/model.pt`
   from MinIO. **This is the hook the retrain-redeploy loop uses.** If MinIO
   isn't reachable, fall back to the local baked-in model so the pod still
   starts.
4. Load `popularity.npy` for the cold-start blender. If empty session, blend
   GRU4Rec output with popularity. Alpha ramps up as the session gets longer.

### `scripts/`

- **`enrich_audio_swift.py`** — the one Hashir runs on his VM. For every mp3
  already in `audio/` on Chameleon Swift, it pulls title/artist/track_id from
  `track_dict.parquet` (on MinIO), embeds them as ID3 tags via mutagen, and
  uploads to `audio_complete/`. Also writes a `manifest.json` listing every
  track. **This is how Navidrome's TagLib scanner gets real names into the
  library** — without it, everything shows as "Track 1234567 / Unknown."
- **`enrich_audio.py`** — same logic but using boto3/S3 for environments
  where audio is already on MinIO instead of Chameleon Swift.
- **`make_swift_manifest.py`** — lightweight variant that only writes
  manifest.json + sets Swift object metadata (doesn't rewrite audio bytes).
- **`warmup_cache.py`** — parallel yt-dlp worker for pre-populating the audio
  cache. Abandoned mid-project when YouTube rate-limited us after ~2k tracks.
  Kept in the repo because the code's still useful if we ever get a different
  audio source.
- **`rollback.sh`** — manual runbook for rolling the serving pod back to a
  previous MinIO model key. See `TRIGGERS.md` for when to use it.

### `TRIGGERS.md`

The policy document for model promotion and rollback. Defines offline gates
(Recall@10 must beat baseline, p95 latency must stay under 200ms) and online
rollback triggers (error rate, latency p95, cold-start surge, OOV rate, play
cache hit rate) with numeric justifications for every threshold. The alert
rules in `salawhaaat/navidrome-iac/k8s/monitoring/templates/prometheus-rules.yaml`
enforce these.

### `serving/baseline/test_app.py`

Pytest tests pinning down two contracts that broke on us:

1. Track metadata lookup after loading `track_dict.parquet` (the str/int key
   mismatch that silently dropped every title/artist).
2. `/play` cache-hit vs random-fallback behavior.

Run with `pytest serving/baseline/test_app.py -v` from the repo root.
Eleven tests, ~1 second, all green.

---

## Running it locally (no cluster needed)

You need Python 3.11+ and a local venv.

```bash
# 1. deps
pip install -r serving/baseline/requirements.txt

# 2. get the artifacts someone's already trained
#    - best_gru4rec.pt  (GRU4Rec weights)
#    - vocabs.pkl       (item2idx / user2idx pickle)
#    both live in MLflow or can be scp'd from a teammate

# 3. run serving
cd serving/baseline
uvicorn app:app --host 0.0.0.0 --port 8080

# 4. smoke test
curl -s http://localhost:8080/health
# {"status":"ok","model_loaded":true}

curl -s -X POST http://localhost:8080/recommend-by-tracks \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","user_id":"0","track_ids":[],"top_n":3}' | jq .
```

The empty `track_ids` triggers the cold-start fallback — you'll get three
popular tracks back instead of a 400.

---

## The retrain → redeploy loop

This is the part graders tend to ask about. It spans all three repos.

1. **Scheduled trigger.** An Argo `WorkflowTemplate` called `cron-finetune`
   (in `salawhaaat/navidrome-iac`) fires on a schedule. It SSHes into an
   MI100 GPU server and runs `train/finetune_gru4rec.py` from the Navidrome
   fork.

2. **Training happens.** Yesha's script trains for N epochs, logs loss + val
   HR@10 / NDCG@10 to MLflow. Before uploading the model, it runs two
   safeguard checks:
   - **Data quality** — flags suspiciously long sessions or single-item
     frequency spikes (possible data poisoning).
   - **Regression detection** — compares new val HR@10 to the previous best;
     logs a warning if it dropped by more than 5 percentage points.

3. **Safeguard metrics published.** The script pushes three gauges to the
   Prometheus pushgateway: `gru4rec_safeguard_hr_delta`,
   `gru4rec_safeguard_popularity_share`, `gru4rec_safeguard_data_quality_passed`.
   Prometheus scrapes these into the TSDB. Alert rules act on them.

4. **Model.pt uploaded to MinIO** at
   `s3://artifacts/finetune/<date>/<run_id>/model.pt`.

5. **Deploy gates.** The Argo workflow now:
   - Backs up the current `MINIO_MODEL_KEY` env var (enables rollback).
   - Sets the new key and runs `kubectl rollout restart navidrome-serve`.
   - Runs an **integration test** (hits `/health` and a real
     `/recommend-by-tracks` call).
   - Runs a **load test** (10 concurrent workers for 30 seconds; requires
     p95 < 500ms — mirrors the system-optimization lab's `hey` setup).

6. **Promote or roll back.** If both tests pass, the workflow tags the MLflow
   run as `@live`. If either fails, it restores the backed-up `MINIO_MODEL_KEY`
   and restarts — automatic rollback.

7. **Live monitoring.** Prometheus watches the running pod's `/metrics`. If
   any of the six serving-layer rules go red, Alertmanager fires. Manual
   rollback is `scripts/rollback.sh <previous-key>`.

The piece many teams skip: **rollback has to be operable when things go
wrong at 3am.** That's why we have a simple bash script, a policy doc with
explicit thresholds, and a workflow that keeps the previous key around.

---

## Monitoring & observability

### Prometheus

Twenty alert rules across three groups
(`salawhaaat/navidrome-iac/k8s/monitoring/templates/prometheus-rules.yaml`):

- **`infrastructure.rules`** (11) — cluster health: nodes, pods, GPUs, PVCs.
  Standard stuff Salauat shipped early.
- **`serving.rules`** (6) — live SLIs:
  - `ServingErrorRateCritical` (> 5% 500s for 5m)
  - `ServingLatencyP95Critical` (> 1s p95 for 5m)
  - `ColdStartSurge` (> 30% cold-start for 10m)
  - `OOVRateHigh` (> 5 OOV items/request for 10m)
  - `PlayCacheHitRateLow` (< 30% hit rate for 30m)
  - `NavidromeServeDown` (scrape endpoint unreachable for 2m)
- **`training.rules`** (3) — safeguards:
  - `FinetuneRegression` (`hr_delta < -0.05`)
  - `FinetuneDataQualityFailed` (`data_quality_passed == 0`)
  - `PopularityConcentrationHigh` (> 70% top-10% items for 10m)

Every threshold is explained in `TRIGGERS.md §4`. They're not pulled from a
hat — each one has a numeric rationale tied to the SLO.

### Grafana dashboards

- **Navidrome MLOps Monitoring** — main operational SLI panel (request rate,
  p95 latency, OOV rate, Redis infra, total recs served, model info).
  Salauat's build, this is the one to show a grader first.
- **Navidrome Data Drift Monitor** — model output distribution checks.
- **Navidrome Live User Activity** — scrobble / play events from Postgres
  (user-feedback signal).
- **Navidrome Serving — SLIs & Safeguards** — our nine-panel dashboard that
  specifically surfaces the `gru4rec_safeguard_*` metrics alongside the
  live SLIs. Import JSON lives in
  `salawhaaat/navidrome-iac/k8s/monitoring/templates/grafana-deployment.yaml`.

### Metrics this service emits

- `recommend_requests_total{status}` — request counter by HTTP status
- `recommend_latency_seconds` — latency histogram with p50/p95/p99 buckets
- `recommend_oov_items_total` — vocab lookups that missed
- `recommend_cold_start_activations_total` — requests where the blender was
  applied (short sessions → popularity blend)
- `recommend_top_n` — requested top_n values (capacity planning)
- `recommend_model_info` — gauge tagged with `model_version`, `num_items`,
  `embedding_dim` — lets you correlate metric changes to specific model
  versions
- `play_cache_hits_total` / `play_cache_misses_total` — /play audio cache
  behavior
- `play_download_seconds` — histogram for fallback path (unused in current
  random-fallback mode but instrumented)
- `play_download_errors_total{reason}` — counter for errored fallbacks

---

## Integration points worth knowing about

| Concern | Owner | Touchpoint |
|---|---|---|
| Getting user plays into training data | Hashir + Yesha | Navidrome scrobbler (`adapters/navidrome_feedback/navidrome_feedback.go`) → feedback API → Postgres → `build_dataset_live.py` |
| Translating Navidrome UUIDs ↔ 30Music IDs | Vanshika + Salauat | `mf.MbzRecordingID` populated at ingest time via ID3 tags (`musicbrainz_trackid`) written by `scripts/enrich_audio_swift.py` |
| Loading the right model at startup | Vanshika | `_fetch_model_from_minio()` in `app.py` — auto-discovers latest `finetune/.../model.pt` |
| Gating deploys | Salauat + Vanshika | `cron-finetune` workflow in iac — integration + load test gates, rollback on failure |
| Rollback policy | Vanshika | `TRIGGERS.md` — explicit thresholds and what they mean |

---

## Running the full stack

You don't. That's what the Kubernetes cluster is for. Once a teammate has
the iac applied:

- **Navidrome (UI + backend):** http://129.114.27.204:4533
- **MLflow:** http://129.114.27.204:8000
- **MinIO Console:** http://129.114.27.204:9001 (minioadmin / navidrome2026)
- **Prometheus:** http://129.114.27.204:9090
- **Grafana:** http://129.114.27.204:3000 (admin / admin)
- **Alertmanager:** http://129.114.27.204:9093

Inside the cluster, services resolve via
`{name}.navidrome-platform.svc.cluster.local`.

---

## Known gaps and what we'd do next

Being honest because a grader will spot these anyway:

- **Automated webhook rollback.** The policy doc says a human runs
  `scripts/rollback.sh` when a soft-alert fires. The next step is an
  Alertmanager receiver that POSTs to a handler which invokes the script.
  Not built — traffic is too low to justify auto-action without a human in
  the loop.

- **Canary deployments.** New models go straight to 100% traffic after
  passing integration + load tests. A canary would route 10% to the new
  version and compare live SLIs before ramping. Right pattern for a real
  product; overkill for a demo with < 10 QPS.

- **YouTube audio scraping was abandoned.** `warmup_cache.py` still works
  and produced ~2k tracks before YouTube started rate-limiting us. For the
  remainder, `/play` returns a random available track — the UI shows the
  correct title/artist regardless (those come from the recommendation
  response, not the audio file).

- **One serve replica.** RWO PVC and single-replica deployment mean rolling
  updates need the `Recreate` strategy (patched in). Fine for now, not
  horizontally scalable.

- **No A/B experimentation.** MLflow tracks runs, but there's no framework
  for running two model versions simultaneously and comparing them on live
  traffic. Would need traffic splitting (Istio or similar).

---

## AI-use disclosure

Per the course AI-use policy, this repo was developed with Claude's
assistance. The architecture, design choices, threshold selection, and all
written content (including this README) are mine. Commits with AI involvement
are tagged `[AI-assisted]` in the subject line. Individual LLM usage is
documented inline where it mattered.

---

## Quick links

- Policy: [`TRIGGERS.md`](TRIGGERS.md)
- Tests: [`serving/baseline/test_app.py`](serving/baseline/test_app.py)
- Rollback: [`scripts/rollback.sh`](scripts/rollback.sh)
- Audio enrichment: [`scripts/enrich_audio_swift.py`](scripts/enrich_audio_swift.py)
- IaC: [salawhaaat/navidrome-iac](https://github.com/salawhaaat/navidrome-iac)
- Navidrome fork: [yeshavyas27/navidrome_mlops](https://github.com/yeshavyas27/navidrome_mlops) (branch `navidrome-custom`)
