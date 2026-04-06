# Ray Serve container for bonus extra credit
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# Ray Serve starts its own HTTP server on port 8000
CMD ["serve", "run", "app:app", "--host", "0.0.0.0", "--port", "8000"]
