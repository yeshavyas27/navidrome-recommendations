# Serving Safeguarding Plan

## Overview

This document describes the safeguards built into the serving layer to ensure reliable, safe, and observable operation. Each safeguard follows the pattern: **what can go wrong → how we detect it → how we handle it**.

---

## 1. Input Validation

**Risk:** Malformed or adversarial requests crash the server or produce garbage recommendations.

**Detection:** Pydantic schema validation on every request (`RecommendRequest` model). Enforces:
- `prefix_item_idxs` must have 1–200 items
- `top_n` must be 1–100
- All fields are typed (string, int, list)

**Handling:** FastAPI returns a `422 Unprocessable Entity` with a clear error body. The server never crashes — invalid requests are rejected before reaching the model.

---

## 2. Out-of-Vocabulary (OOV) Items

**Risk:** A request contains track indices that don't exist in the model's vocabulary (e.g., a new track added after training, or a data pipeline bug sending wrong IDs).

**Detection:** Every index in `prefix_item_idxs` is checked against the loaded `idx2item` dictionary. OOV items are counted via the `recommend_oov_items_total` Prometheus counter.

**Handling:**
- OOV items are **silently filtered** from the prefix — the model runs on whatever valid items remain.
- If ALL items are OOV, the request returns `400 Bad Request` (we can't recommend anything with an empty session).
- A warning is logged with the count of dropped items for debugging.
- If OOV rate spikes in Prometheus, it signals a vocab mismatch between the data pipeline and the serving layer (e.g., retraining happened with a new vocab but serving wasn't updated).

---

## 3. PII-Safe Logging

**Risk:** User IDs or other personally identifiable information leak into server logs, violating privacy.

**Detection:** N/A — this is a preventive safeguard.

**Handling:** All user IDs are hashed with SHA-256 before logging. Logs only show `user_hash=a1b2c3d4e5f6` (first 12 hex chars), which is enough for log correlation but cannot be reversed to the original user ID. Raw user IDs are never written to disk.

---

## 4. Latency Monitoring

**Risk:** Inference latency degrades (due to model size increase, resource contention, or infrastructure issues), causing poor user experience.

**Detection:** The `recommend_latency_seconds` Prometheus histogram tracks every request with buckets from 5ms to 5s. Deriving p50, p95, and p99 latencies from this histogram reveals both typical and tail latency behavior.

**Handling:**
- p50 tells us what most users experience.
- p95/p99 catches tail latency spikes that averages would hide.
- If p95 exceeds the target (e.g., 100ms on GPU), investigate: is it queuing delay (scale replicas) or inference time (check model size, batch settings)?
- The `recommend_requests_total` counter by status (200/400/500) tracks error rate alongside latency.

---

## 5. Model Staleness

**Risk:** The model becomes stale — trained on old data that no longer reflects user behavior, leading to irrelevant recommendations.

**Detection:** The `recommend_model_info` Prometheus gauge label includes `model_version`, which is logged at startup. Comparing this against the latest MLflow run shows whether serving is running the most recent model.

**Handling:**
- The Argo CronWorkflow triggers weekly retraining. New models are logged to MLflow with eval metrics.
- On restart, the serving layer fetches the latest model from MLflow automatically (`MLFLOW_TRACKING_URI` env var).
- Model version is included in every API response (`model_version` field), so downstream services can verify they're getting recommendations from the expected model.

---

## 6. Model Quality Regression

**Risk:** A newly retrained model is worse than the previous one (e.g., bad training data, hyperparameter regression), and serving it degrades recommendation quality.

**Detection:** Every MLflow run logs evaluation metrics: `strict_HR20`, `session_HR20`, `session_MRR20`. These can be compared against previous runs before deployment.

**Handling:**
- The serving layer currently fetches the **latest** run from MLflow. A stronger safeguard would be to only fetch runs where `session_HR20` exceeds a quality floor (e.g., 0.40).
- Until that check is automated, the team manually reviews MLflow metrics before triggering a restart.
- If a bad model is deployed: restart the serving container without `MLFLOW_TRACKING_URI` to fall back to the local `.pt` file (last known good).

---

## 7. Graceful Error Handling

**Risk:** Unexpected errors (CUDA out of memory, tensor shape mismatch, network issues with MLflow) crash the server or return unhelpful errors.

**Detection:** All exceptions in the `/recommend` handler are caught. The `recommend_requests_total{status="500"}` counter tracks internal errors.

**Handling:**
- Known errors (empty prefix after OOV filtering) return `400` with a clear message.
- Unknown errors return `500` with `"internal error"` (no stack trace leaked to client).
- Full stack traces are logged server-side with the `request_id` for debugging.
- The `request_id` (from `x-request-id` header or generated UUID) enables end-to-end tracing across the app tier, serving tier, and logs.

---

## 8. Resource Safeguards

**Risk:** A single request consumes excessive resources (very long session history, very large top_n), starving other requests.

**Detection:** Input bounds enforced by Pydantic:
- `prefix_item_idxs`: max 200 items (`MAX_PREFIX_LEN`)
- `top_n`: max 100 (`MAX_TOP_N`)

**Handling:** Requests exceeding these limits are rejected at validation time (422), before any model computation happens. The `recommend_top_n` histogram tracks requested top_n values for capacity planning.

---

## Summary

| # | Safeguard | Detection Method | Response |
|---|-----------|-----------------|----------|
| 1 | Input validation | Pydantic schemas | 422 rejection |
| 2 | OOV handling | `idx2item` lookup + Prometheus counter | Filter + warn, 400 if all OOV |
| 3 | PII protection | SHA-256 hash at log time | Never log raw user IDs |
| 4 | Latency monitoring | Prometheus histogram (p50/p95/p99) | Scale replicas or investigate model |
| 5 | Model staleness | Model version tracking + MLflow comparison | Weekly retrain + auto-fetch on restart |
| 6 | Quality regression | MLflow eval metrics (HR@20, MRR@20) | Quality floor check before deploy |
| 7 | Error handling | Exception catch + 500 counter + request_id | Structured errors, server-side logging |
| 8 | Resource limits | Pydantic field constraints | 422 rejection at validation |
