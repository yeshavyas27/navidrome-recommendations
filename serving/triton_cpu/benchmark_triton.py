"""
Benchmark script for Triton serving.
Triton uses a different request format than FastAPI, so we send
requests to /v2/models/song_recommender/infer with input tensors.

Adapted from lab's Triton testing pattern.
"""

import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

TRITON_URL = "http://localhost:8000/v2/models/song_recommender/infer"

# Build a Triton-format request with a dummy user embedding
np.random.seed(42)
user_embedding = np.random.randn(64).astype(np.float32).tolist()

payload = {
    "inputs": [{
        "name": "user_embedding",
        "shape": [1, 64],
        "datatype": "FP32",
        "data": [user_embedding]
    }]
}


def send_request():
    start_time = time.time()
    response = requests.post(TRITON_URL, json=payload)
    elapsed = time.time() - start_time
    return elapsed, response.status_code


def run_benchmark(num_requests, concurrency):
    inference_times = []
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        for future in as_completed(futures):
            elapsed, status = future.result()
            if status == 200:
                inference_times.append(elapsed)
            else:
                errors += 1

    inference_times = np.array(inference_times)
    median_time = np.median(inference_times)
    percentile_95 = np.percentile(inference_times, 95)
    percentile_99 = np.percentile(inference_times, 99)
    throughput = num_requests / inference_times.sum() * concurrency

    print(f"\n--- Concurrency: {concurrency} ({num_requests} requests) ---")
    print(f"Median inference time:  {1000 * median_time:.4f} ms")
    print(f"95th percentile:        {1000 * percentile_95:.4f} ms")
    print(f"99th percentile:        {1000 * percentile_99:.4f} ms")
    print(f"Throughput:             {throughput:.2f} requests/sec")
    print(f"Error rate:             {errors / num_requests * 100:.1f}%")


print(f"Benchmarking {TRITON_URL}")
for c in [1, 5, 10, 20]:
    run_benchmark(num_requests=100, concurrency=c)
