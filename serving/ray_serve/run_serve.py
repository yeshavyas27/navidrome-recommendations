"""
Start Ray Serve with explicit host binding to 0.0.0.0
so it's accessible from outside the Docker container.
"""
import sys
import os
from pathlib import Path

# Ensure _shared is importable by Ray workers
_SERVING_ROOT = Path(__file__).resolve().parent.parent
for candidate in [_SERVING_ROOT, Path("/app")]:
    if (candidate / "_shared" / "model.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        # Also set PYTHONPATH so Ray worker processes inherit it
        os.environ["PYTHONPATH"] = str(candidate) + ":" + os.environ.get("PYTHONPATH", "")
        break

import ray
import time
from ray import serve

ray.init(runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})

serve.start(http_options={"host": "0.0.0.0", "port": 8000})

from app import app
serve.run(app)

print("Ray Serve is running on http://0.0.0.0:8000")

while True:
    time.sleep(60)
