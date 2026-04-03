"""
Benchmark script for recommendation serving options.
This script sends a series of requests to the FastAPI 
recommendation endpoint and measures latency and throughput.
"""

import requests
import time
import numpy as np

#Configuration for benchmarking the FastAPI recommendation endpoint that serves song recommendations based on embedding dot products.
FASTAPI_URL = "http://localhost:8000/recommend"
payload = {"user_id": "user_42", "n_recommendations": 10}

#Single-request latency test (100 requests one at a time)
num_requests = 100
inference_times = []

print(f"Sending {num_requests} requests to {FASTAPI_URL}...")

for i in range(num_requests):
    start_time = time.time()
    response = requests.post(FASTAPI_URL, json=payload)
    end_time = time.time()

    if response.status_code == 200:
        inference_times.append(end_time - start_time)
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

#Calculate stats on inference times
inference_times = np.array(inference_times)
median_time = np.median(inference_times)
percentile_95 = np.percentile(inference_times, 95)
percentile_99 = np.percentile(inference_times, 99)
throughput = num_requests / inference_times.sum()

#Print results 
print(f"\n--- Results ({num_requests} requests) ---")
print(f"Median inference time:  {1000 * median_time:.4f} ms")
print(f"95th percentile:        {1000 * percentile_95:.4f} ms")
print(f"99th percentile:        {1000 * percentile_99:.4f} ms")
print(f"Throughput:             {throughput:.2f} requests/sec")
print(f"Error rate:             {(num_requests - len(inference_times)) / num_requests * 100:.1f}%")
