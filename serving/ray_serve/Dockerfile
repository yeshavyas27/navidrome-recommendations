# Ray Serve container for GRU4Rec (bonus)
#
# Build:  docker build -f serving/ray_serve/Dockerfile -t gru4rec-ray .
# Run:    docker run -d -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts gru4rec-ray

FROM python:3.11-slim

WORKDIR /app

COPY serving/ray_serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/_shared/ _shared/
COPY serving/ray_serve/app.py .
COPY serving/ray_serve/run_serve.py .

EXPOSE 8000

CMD ["python", "run_serve.py"]
