# Navidrome Recommendations — Serving Layer

Session-based music recommendation serving for Navidrome, using GRU4Rec trained on the 30Music dataset. Part of the ECE-GY 9183 ML Systems Design & Operations group project at NYU.

## Team

| Member | Role | Repo |
|--------|------|------|
| Vanshika Bagaria | **Serving** (this repo) | [vanshika2022/navidrome-recommendations](https://github.com/vanshika2022/navidrome-recommendations) |
| Yesha Vyas | Training (GRU4Rec, SessionKNN) | navidrome-iac/navidrome-train |
| Hashir Muzaffar | Data pipeline | [hashirmuzaffar/navidrome-mlops-data-proj05](https://github.com/hashirmuzaffar/navidrome-mlops-data-proj05) |
| Salauat Kakimzhanov | DevOps (K8s, ArgoCD, MLflow) | navidrome-iac |

## Architecture

```
Client → FastAPI wrapper (port 8080)     → Triton Inference Server (port 8000)
         - Input validation                 - GRU encoder (ONNX, GPU)
         - OOV filtering                    - Dynamic batching
         - Vocab lookup (track ID ↔ idx)    - CUDA execution provider
         - Dot product (session × 745K items)
         - Top-k selection
         - Prometheus metrics
```

On startup, the serving layer fetches the latest model from MLflow (`MLFLOW_TRACKING_URI`), enabling automatic retrain → redeploy.

## Repository Structure

```
navidrome-recommendations/
├── serving/
│   ├── _shared/                # GRU4Rec model class (shared across variants)
│   │   └── model.py
│   ├── baseline/               # FastAPI + PyTorch (CPU baseline)
│   │   ├── app.py              # /recommend endpoint, Prometheus metrics, MLflow fetch
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── triton_cpu/             # Triton Inference Server + FastAPI wrapper
│   │   ├── docker-compose.yaml # Triton + wrapper + Jupyter (one command)
│   │   ├── export_onnx.py      # PyTorch → ONNX export
│   │   ├── Dockerfile.export   # Docker container for ONNX export
│   │   ├── Dockerfile.wrapper  # Docker container for FastAPI wrapper
│   │   ├── triton_wrapper.py   # Wrapper: Triton → dot product → top-k
│   │   ├── benchmark_triton.py # Benchmark script (direct + e2e modes)
│   │   └── model_repository/
│   │       └── gru4rec_encoder/
│   │           ├── config.pbtxt    # Triton model config
│   │           └── 1/model.onnx    # ONNX model (gitignored)
│   ├── faiss_cpu/              # FAISS index search (proof-of-concept)
│   ├── faiss_cached/           # FAISS + Redis caching (proof-of-concept)
│   ├── faiss_gpu/              # FAISS on GPU (proof-of-concept)
│   ├── ray_serve/              # Ray Serve deployment (bonus)
│   └── benchmark.py            # CPU baseline benchmark script
├── samples/
│   ├── input_sample.json       # GRU4Rec request schema (v1.0)
│   └── output_sample.json      # GRU4Rec response schema (v1.0)
├── artifacts/                  # Model files (gitignored)
│   ├── best_gru4rec.pt         # Trained model weights (from MLflow)
│   └── vocabs.pkl              # item2idx / idx2item mappings
├── docs/
│   ├── safeguarding_plan.md    # Serving safeguards (8 risk areas)
│   └── benchmark_results.md    # CPU vs GPU benchmark comparison
└── README.md
```

## Quickstart

### Prerequisites

- Docker with NVIDIA container toolkit (for GPU)
- Access to Chameleon Cloud CHI@TACC (gpu_p100 node)
- Model artifacts: `best_gru4rec.pt` and `vocabs.pkl`

### 1. Get artifacts

From MLflow:
```bash
mkdir -p artifacts
curl -o artifacts/best_gru4rec.pt \
  'http://129.114.25.168:8000/get-artifact?run_id=50d806ff55c74f1ca1f3eaa81fea177e&path=best_gru4rec.pt'
```

Vocab file from training team (Yesha's cached vocabs):
```bash
# scp from team member or download from shared storage
```

### 2. Export ONNX model

```bash
docker build -f serving/triton_cpu/Dockerfile.export -t gru4rec-export .
docker run \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/serving/triton_cpu/model_repository:/app/model_repository \
  gru4rec-export
```

### 3. Start Triton + wrapper

```bash
cd serving/triton_cpu
docker compose up -d
```

Wait for Triton to show `READY`:
```bash
docker logs triton_server -f
```

### 4. Test

```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "user_id": 1,
    "user_idx": 0,
    "prefix_item_idxs": [1, 2, 3, 4, 5],
    "playratios": [1, 1, 1, 1, 1],
    "exclude_item_idxs": [1, 2, 3, 4, 5],
    "top_n": 20
  }'
```

### 5. Benchmark

```bash
# Direct Triton (GRU encoder only)
docker exec jupyter python /work/benchmark_triton.py --mode triton --url triton_server:8000

# End-to-end (wrapper → Triton → dot product → response)
docker exec jupyter python /work/benchmark_triton.py --mode e2e --url fastapi_wrapper:8080
```

### Running baseline only (no GPU needed)

```bash
cd serving/baseline
docker build -t baseline .
docker run -d -p 8000:8000 \
  -v $(pwd)/../../artifacts:/app/artifacts \
  -e DEVICE=cpu \
  baseline
```

With MLflow auto-fetch:
```bash
docker run -d -p 8000:8000 \
  -v $(pwd)/../../artifacts:/app/artifacts \
  -e MLFLOW_TRACKING_URI=http://129.114.25.168:8000 \
  baseline
```

## Benchmark Results

| Configuration | Concurrency | Median Latency | Throughput |
|--------------|-------------|---------------|------------|
| CPU baseline (PyTorch) | 1 | 14ms | 68 req/s |
| CPU baseline (PyTorch) | 16 | 110ms | 176 req/s |
| **GPU Triton encoder** | **1** | **0.76ms** | **1,292 req/s** |
| **GPU Triton encoder** | **16** | **0.88ms** | **18,941 req/s** |
| E2E (Triton + dot product) | 1 | 30ms | 33 req/s |

GPU Triton achieves **18x latency improvement** and **107x throughput improvement** over CPU baseline for the GRU encoder. See [docs/benchmark_results.md](docs/benchmark_results.md) for full analysis.

## API Schema

See [samples/input_sample.json](samples/input_sample.json) and [samples/output_sample.json](samples/output_sample.json).

## Monitoring

Prometheus metrics exposed at `/metrics`:
- `recommend_requests_total{status}` — request count by HTTP status
- `recommend_latency_seconds` — latency histogram (p50/p95/p99)
- `recommend_oov_items_total` — out-of-vocabulary items dropped
- `recommend_model_info` — loaded model metadata

## Safeguarding

See [docs/safeguarding_plan.md](docs/safeguarding_plan.md) for the full plan covering input validation, OOV handling, PII-safe logging, latency monitoring, model staleness, quality regression, error handling, and resource limits.

## Disclosure

Code in this repository was developed with AI assistance (Claude) as permitted by the course AI use policy. Design decisions, architecture choices, and all written documentation are my own.
