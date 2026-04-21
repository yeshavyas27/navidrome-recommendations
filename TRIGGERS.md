# Model promotion & rollback triggers

This document defines **when** a new GRU4Rec model is promoted into the
serving container, and **when** we roll back to a previous version. The
triggers are enforced at two layers:

1. **Offline gates** (in the training + CI pipeline) — a new model must
   pass these before it's eligible for promotion.
2. **Online watchdog** (in Prometheus + manual runbook) — live metrics
   that fire an alert and inform a rollback decision.

The goal of separating these is that offline gates catch regressions we
can see before shipping; online watchdogs catch things only production
load reveals (user-behavior drift, unexpected OOV distributions, etc.).

## 1. Promotion

A finetune run produces a new `model.pt` uploaded to
`s3://artifacts/finetune/<YYYYMMDD>/<run_id>/model.pt` on MinIO. The
serving container's startup code (`_fetch_model_from_minio` in
[app.py](serving/baseline/app.py)) auto-discovers the newest
`finetune/.../model.pt` on pod restart. A run becomes eligible only when
the training pipeline verifies:

| Gate | Threshold | Why |
|---|---|---|
| Val Recall@10 | ≥ baseline + 0.5pp | baseline here is the currently-deployed model's val Recall@10 at its own training time. +0.5pp is tight enough to reject noise-only wins but permissive enough to allow incremental improvements. |
| Val NDCG@10 | ≥ baseline − 0.2pp | allows a small NDCG drop if Recall@10 improves significantly — recall matters more for seeding a session-based model. |
| CPU p95 inference latency | ≤ 200 ms | serving runs on CPU at `DEVICE=cpu`. 200 ms matches our p95 SLO with headroom. |
| Integration tests | pass | `test_app.py` + end-to-end smoke test in CI; prevents shape/vocab-mismatch regressions. |

Runs that fail any gate stay in MinIO for forensics but the training
pipeline doesn't mark them as promotable. The serving auto-discover
logic picks the latest *promotable* run — not just the latest upload.

## 2. Rollback triggers

The watchdog reads Prometheus metrics exposed by the serving `/metrics`
endpoint. An alert firing does **not** automatically roll back — a
human on-call reviews the alert context first. This is deliberate:
auto-rollback on a false positive is worse than a brief degraded
experience, and our traffic isn't high enough to justify the infra.

### Hard rollback — alert fires, execute runbook immediately

| Alert | Expression | Reason |
|---|---|---|
| `ServingErrorRateCritical` | `rate(recommend_requests_total{status="500"}[5m]) > 0.05` for 5m | > 5% 500s indicates systemic issue (bad model, OOM, etc.). UX badly broken. |
| `ServingLatencyP95Critical` | `histogram_quantile(0.95, rate(recommend_latency_seconds_bucket[5m])) > 1.0` for 5m | > 1s p95 is 4× the SLO. Something pathological in inference. |

### Soft rollback — investigate first, roll back if confirmed regression

| Alert | Expression | Reason |
|---|---|---|
| `ColdStartSurge` | `rate(recommend_cold_start_activations_total[10m]) / rate(recommend_requests_total[10m]) > 0.30` for 10m | Cold-start fallback fires when the model has no usable seeds. Steady-state expectation is ~10% (new sessions). 30%+ suggests the model vocab no longer aligns with the library (e.g. post-refresh drift) OR the track-id mapping pipeline broke. Rollback recovers quickly while data team investigates. |
| `OOVRateHigh` | `rate(recommend_oov_items_total[10m]) / rate(recommend_requests_total[10m]) > 5` for 10m | Average > 5 OOV items per request means either the Navidrome library drifted away from the model's vocab OR the request pipeline is sending UUIDs again. Not directly model quality, but the symptom of a silent pipeline break. |
| `PlayCacheHitRateDrop` | `rate(play_cache_hits_total[30m]) / (rate(play_cache_hits_total[30m]) + rate(play_cache_misses_total[30m])) < 0.3` for 30m | Warm cache should give 50%+ hit rate during demos. Sustained < 30% means warmup didn't populate or the bucket got wiped. Not a model-rollback trigger per se — fixes the cache, not the model. Included here because it affects user-visible latency. |

## 3. Rollback mechanism

Manual rollback runbook — see [`scripts/rollback.sh`](scripts/rollback.sh):

```bash
# find the previous known-good model key
kubectl logs -n navidrome-platform deploy/navidrome-serve | grep "Loading model from"
# => "Loading model from MinIO: s3://artifacts/finetune/20260420/abc123/model.pt"

# roll back to a specific earlier run
scripts/rollback.sh finetune/20260419/def456/model.pt
```

The script:
1. Sets `MINIO_MODEL_KEY=<previous_key>` on the serving deployment (overrides the auto-discover logic).
2. `kubectl rollout restart deploy/navidrome-serve` so the new env var takes effect.
3. Waits for rollout to complete and prints pod status.

To undo a rollback (return to auto-latest):
```bash
kubectl set env deploy/navidrome-serve MINIO_MODEL_KEY- -n navidrome-platform
kubectl rollout restart deploy/navidrome-serve -n navidrome-platform
```

## 4. Why these thresholds, not stricter ones

- **Offline gate tight enough to matter, loose enough to advance.** +0.5pp Recall@10 is a real signal at our val-set size (session-level split). Tighter than that, we reject real improvements as noise. Looser, we ship flat models.
- **5% error rate as critical, not 1%.** Our 1% SLO is a target; alerting at 1% creates noise on transient blips (pod restarts, momentary MinIO timeouts). 5% sustained for 5 minutes is past "transient."
- **1s p95 as critical, not 500ms.** Same reasoning — 500ms is a warning threshold (observable but tolerable); 1s is UX-breaking.
- **30% cold-start surge, not 15%.** Cold-start naturally fires on short sessions. 30% means the distribution shifted enough that retraining or rollback is needed; 15% is within normal day-to-day variance.

## 5. What's NOT automated (yet)

- Automated webhook rollback on alert fire. Adding a tiny handler that
  Alertmanager POSTs to, which in turn runs `rollback.sh`, is the next
  step. Deferred because: (a) our traffic doesn't justify the
  complexity, (b) a human in the loop catches false positives that
  threshold-based auto-rollback would act on, (c) rolling back the
  wrong model is hard to undo quickly in a live demo.

- Canary deployments. A canary would route 10% of traffic to the new
  model, compare against control, and auto-roll-forward only if
  metrics hold. Not built — current traffic is too low for statistical
  significance on a canary window.

Both are called out as post-May-4 roadmap items.
