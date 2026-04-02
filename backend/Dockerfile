FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads and FAISS index
RUN mkdir -p /app/uploads /app/faiss_index

# Copy frontend build (mounted or copied at compose level)
# COPY ../frontend /app/frontend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
