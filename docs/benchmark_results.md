# Benchmark Results

## Hardware

- **CPU baseline**: MacBook local (Apple Silicon)
- **GPU Triton**: Chameleon CHI@TACC, 2x Tesla P100-PCIE-16GB, Ubuntu 24.04

## Model

- GRU4Rec: 745,352 items, embedding_dim=64, hidden_dim=128, 1 GRU layer
- 48M parameters, 182MB checkpoint

## Results

### CPU Baseline (FastAPI + PyTorch, no GPU)

| Concurrency | Median Latency | p95 Latency | Throughput |
|-------------|---------------|-------------|------------|
| 1 | 14ms | 18ms | 68 req/s |
| 5 | 31ms | 55ms | 152 req/s |
| 10 | 59ms | 80ms | 172 req/s |
| 20 | 110ms | 164ms | 176 req/s |

Latency grows linearly with concurrency — pure queuing delay.
Throughput plateaus ~176 req/s regardless of concurrency.

### GPU Triton Encoder (ONNX backend, P100, dynamic batching)

| Concurrency | Median Latency | p95 Latency | Throughput |
|-------------|---------------|-------------|------------|
| 1 | 0.76ms | 0.93ms | 1,292 req/s |
| 4 | 0.69ms | 0.77ms | 5,669 req/s |
| 8 | 0.77ms | 1.01ms | 9,428 req/s |
| 16 | 0.88ms | 1.02ms | 18,941 req/s |

Latency stays flat under load — dynamic batching absorbs concurrency.
Throughput scales linearly with concurrency.

### End-to-End (FastAPI wrapper → Triton → dot product → top-k)

| Concurrency | Median Latency | p95 Latency | Throughput |
|-------------|---------------|-------------|------------|
| 1 | 30ms | 31ms | 33 req/s |

Bottleneck shifts to CPU dot product (64-dim × 745,352 items).
Triton encoder is <1ms; remaining ~29ms is the dot product + HTTP overhead.

## Key Takeaways

1. **Model optimization (Ch7 §7.3.1)**: Switching from 128-dim/2-layer to 64-dim/1-layer
   cut parameters in half (96M → 48M), improved CPU throughput 2.5x.

2. **Graph optimization (Ch7 §7.3.2)**: ONNX export enables fused kernels and
   hardware-specific execution via CUDA execution provider on P100.

3. **System optimization (Ch7 §7.4.3)**: Dynamic batching in Triton keeps latency
   flat at <1ms even at concurrency 16. Without batching, queuing delay grows linearly.

4. **Infrastructure optimization (Ch7 §7.4.4)**: Moving GRU inference from CPU to GPU
   gives 18x latency improvement (14ms → 0.76ms) and 107x throughput improvement
   (176 → 18,941 req/s) for the encoder.

5. **Bottleneck shift**: After GPU optimization, the bottleneck moves from the neural
   network to the dot product search. Next optimization: GPU-accelerated similarity
   search (e.g., FAISS GPU) or approximate nearest neighbor methods.
