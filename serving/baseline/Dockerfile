# Base image with Python (like lab's Dockerfile)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies first (cached if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the serving code
COPY app.py .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the FastAPI server (like lab used uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
