# navidrome-recommendations

<!-- TODO: 1-2 sentences describing what this repo is -->

## Project Context
<!-- TODO: describe the ML feature you're adding to Navidrome -->

## Repository Structure

```
navidrome-recommendations/
├── samples/                    # JSON input/output samples (joint deliverable)
├── serving/
│   ├── baseline/               # Row 1: FastAPI + numpy dot product
│   ├── faiss_cpu/              # Row 2: FastAPI + FAISS index
│   ├── faiss_cached/           # Row 3: FastAPI + FAISS + Redis cache
│   ├── faiss_gpu/              # Row 4: FastAPI + FAISS on GPU
│   ├── triton_gpu/             # Row 5: Triton Inference Server on GPU
│   └── benchmark.py            # Load test script (concurrent requests)
└── README.md
```

## Serving Options

| Row | Option | Optimization | Status |
|---|---|---|---|
| 1 | baseline | None (reference) | Tested on CPU |
| 2 | faiss_cpu | Model-level | Tested on CPU |
| 3 | faiss_cached | Model + System | Tested on CPU |
| 4 | faiss_gpu | Infrastructure | Code ready |
| 5 | triton_gpu | Model + System + Infra | Code ready |

## Running a Serving Option

Each folder is self-contained. To run any option:

```bash
cd serving/<option_name>
docker build -t <option_name> .
docker run -d -p 8000:8000 --name <option_name> <option_name>
```

For options with Docker Compose (faiss_cached, triton_gpu):
```bash
cd serving/<option_name>
docker compose up -d
```

## Testing the Endpoint

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_42", "n_recommendations": 5}'
```

## Running the Benchmark

The benchmark script tests the current running option at concurrency levels 1, 5, 10, 20:

```bash
docker run --rm --network host \
  -v $(pwd)/serving:/app \
  python:3.11-slim bash -c "pip install requests numpy -q && python /app/benchmark.py"
```

## Infrastructure

All experiments run on Chameleon Cloud.

- CPU instance: m1.medium (2 vCPU, 4GB RAM) at KVM@TACC
- GPU instance: gpu_mi100 at CHI@TACC (for rows 4-5)

## Disclosure

Code in this repository was developed with AI assistance (Claude) as permitted by the course AI use policy. Design decisions, architecture choices, and all written documentation are my own.
