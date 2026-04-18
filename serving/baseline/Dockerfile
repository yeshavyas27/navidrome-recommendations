# GRU4Rec serving container for K8s deployment
#
# Build:  docker build -f serving/baseline/Dockerfile -t navidrome-serve .
# Run:    docker run -d -p 8080:8080 -v $(pwd)/artifacts:/app/artifacts navidrome-serve
#
# Required env vars (Salauat injects these in K8s):
#   MLFLOW_TRACKING_URI  — e.g. http://mlflow.navidrome-platform.svc.cluster.local:8000
#   VOCAB_PATH           — path to vocabs.pkl (default: /app/artifacts/vocabs.pkl)

FROM python:3.11-slim

WORKDIR /app

COPY serving/baseline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/_shared/ _shared/
COPY serving/baseline/app.py .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
