# Semantic Omni Search Engine

A production-ready **AI-powered Multi-Modal Search Engine** that supports semantic search across text, images, PDFs, audio, and video files.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Browser                           │
│              (HTML/CSS/JS SPA – search, upload, history)        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / REST
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Nginx (Frontend)                           │
│  Serves static SPA  │  Proxies /upload /search /files → backend│
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│                                                                 │
│   POST /upload    ──► Ingestion Pipeline                        │
│   POST /search    ──► Search Pipeline                           │
│   GET  /files     ──► File Registry                             │
│   DELETE /file/:id──► Soft-delete                               │
│   GET  /search/history ──► Search log                           │
│   GET  /metrics   ──► Prometheus scrape endpoint                │
│   GET  /health    ──► Liveness probe                            │
└──────┬────────────────────────────────────────────┬────────────-┘
       │                                            │
       ▼                                            ▼
┌─────────────┐                           ┌─────────────────────┐
│   FAISS     │                           │    PostgreSQL       │
│ Vector Store│                           │  (file metadata,   │
│(per-modality│                           │  search history)   │
│  indexes)   │                           └─────────────────────┘
└─────────────┘
       ▲
       │ embeddings
┌─────────────────────────────────────────────────────────────────┐
│                      Embedding Engine                           │
│                                                                 │
│  Text  ──► Sentence Transformers (all-MiniLM-L6-v2, dim=384)   │
│  Image ──► CLIP ViT-B/32 (openai/clip-vit-base-patch32, d=512) │
│  Audio ──► Whisper transcription ──► text embedding             │
│  Video ──► Frame sampling (8 frames) ──► CLIP ──► averaged      │
│  PDF   ──► pdfplumber text extract ──► text embedding           │
└─────────────────────────────────────────────────────────────────┘
       ▲
       │
┌─────────────────────────────────────────────────────────────────┐
│                   Observability Stack                           │
│                                                                 │
│  Prometheus ──► scrapes /metrics every 10s                      │
│  Grafana    ──► pre-provisioned dashboard (latency, RPS, errors)│
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **Multi-modal search** | Text, Image, Audio, Video, PDF |
| **Semantic embeddings** | Sentence Transformers + CLIP + Whisper |
| **Vector database** | FAISS (IndexFlatIP, cosine similarity) |
| **Metadata storage** | PostgreSQL via SQLAlchemy |
| **Re-ranking** | Score + keyword overlap |
| **Explainability** | Per-result: similarity score, distance, matching keywords, explanation |
| **Observability** | Prometheus metrics + Grafana dashboard + structured JSON logs |
| **API** | FastAPI with automatic OpenAPI docs |
| **Frontend** | Vanilla SPA: search, upload, file manager, history |
| **Containers** | Docker Compose (dev) + Kubernetes manifests (prod) |
| **CI/CD** | GitHub Actions: test → build → push → deploy |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- At least **8 GB RAM** (ML models are loaded on first request)

### 1. Clone and configure

```bash
git clone https://github.com/yadavanujkumar/Semantic-Omni-Search-Engine.git
cd Semantic-Omni-Search-Engine
cp .env.example .env
```

### 2. Start with Docker Compose

```bash
docker compose up --build -d
```

This starts:

| Service | URL | Purpose |
|---|---|---|
| Frontend (Nginx) | http://localhost | SPA + API proxy |
| Backend (FastAPI) | http://localhost:8000 | REST API |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| PostgreSQL | localhost:5432 | Metadata storage |

### 3. Open the app

Navigate to **http://localhost** in your browser.

---

## API Reference

### Upload a file
```http
POST /upload
Content-Type: multipart/form-data

file: <binary>
```

**Response:**
```json
{
  "file_id": "uuid",
  "filename": "document.pdf",
  "file_type": "pdf",
  "file_size": 102400,
  "content_preview": "First 300 chars of extracted content...",
  "latency_ms": 1234.5
}
```

---

### Search
```http
POST /search
Content-Type: application/json

{
  "query": "machine learning papers",
  "modality": "pdf",
  "top_k": 10,
  "min_score": 0.0
}
```

**Response:**
```json
{
  "query": "machine learning papers",
  "results": [
    {
      "file_id": "uuid",
      "filename": "ml-survey.pdf",
      "file_type": "pdf",
      "content_preview": "...",
      "similarity_score": 0.8743,
      "embedding_distance": 0.1257,
      "matching_modality": "pdf",
      "matching_keywords": ["machine", "learning"],
      "explanation": "Semantic similarity score of 87.43% (Strong match). Matched via pdf embedding. Shared keywords: learning, machine.",
      "rank": 1,
      "created_at": "2024-01-01T00:00:00",
      "file_size": 102400,
      "mime_type": "application/pdf"
    }
  ],
  "total": 1,
  "latency_ms": 45.2,
  "embedding_time_ms": 12.3,
  "retrieval_time_ms": 0.8
}
```

---

### List files
```http
GET /files?modality=pdf&limit=50&offset=0
```

### Delete a file
```http
DELETE /file/{file_id}
```

### Search history
```http
GET /search/history?limit=20
```

---

## Supported File Types

| Modality | Extensions | Embedding Model |
|---|---|---|
| **Text** | .txt, .md, .csv, .html | Sentence Transformers (all-MiniLM-L6-v2) |
| **PDF** | .pdf | pdfplumber → Sentence Transformers |
| **Image** | .jpg, .png, .gif, .webp, .bmp | CLIP ViT-B/32 |
| **Audio** | .mp3, .wav, .ogg, .flac, .m4a | Whisper → Sentence Transformers |
| **Video** | .mp4, .avi, .mov, .webm, .mkv | Frame sampling → CLIP ViT-B/32 |

---

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Pydantic settings
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── api/
│   │   ├── upload.py        # POST /upload
│   │   ├── search.py        # POST /search, GET /search/history
│   │   └── files.py         # GET /files, DELETE /file/{id}
│   ├── core/
│   │   ├── embeddings.py    # Multi-modal embedding engine
│   │   ├── ingestion.py     # End-to-end ingestion pipeline
│   │   ├── vector_store.py  # FAISS wrapper
│   │   └── reranker.py      # Re-ranking + explainability
│   ├── db/
│   │   └── database.py      # SQLAlchemy engine + session
│   ├── models/
│   │   └── models.py        # ORM: FileRecord, SearchHistory
│   └── tests/
│       └── test_backend.py  # pytest test suite
├── frontend/
│   ├── index.html           # SPA shell
│   ├── static/
│   │   ├── css/style.css    # Dark-theme UI
│   │   └── js/app.js        # Vanilla JS application
│   ├── nginx.conf           # Nginx config with API proxy
│   └── Dockerfile
├── monitoring/
│   ├── prometheus.yml       # Prometheus scrape config
│   └── grafana/
│       ├── datasources.yml  # Auto-provision Prometheus datasource
│       ├── dashboards.yml   # Auto-provision dashboards
│       └── dashboards/
│           └── search-engine.json  # Grafana dashboard JSON
├── k8s/
│   └── deployment.yaml      # K8s: Namespace, Deployments, Services, Ingress, HPA
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # GitHub Actions CI/CD pipeline
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Development

### Run backend locally (without Docker)

```bash
cd backend
pip install -r requirements.txt
export DATABASE_URL="postgresql://search_user:search_pass@localhost:5432/search_db"
export UPLOAD_DIR="/tmp/uploads"
export FAISS_INDEX_DIR="/tmp/faiss"
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### Run tests

```bash
cd backend
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## Kubernetes Deployment

```bash
# Create namespace and apply all manifests
kubectl apply -f k8s/deployment.yaml

# Check rollout status
kubectl rollout status deployment/backend -n search-engine
kubectl rollout status deployment/frontend -n search-engine
```

Update `k8s/deployment.yaml`:
- Replace `search.example.com` with your actual domain
- Update the `db-secret` with real credentials

---

## Observability

### Prometheus Metrics

The backend exposes metrics at `/metrics` (scraped by Prometheus every 10s):
- `http_requests_total` – total requests by method/handler/status
- `http_request_duration_seconds` – request latency histogram
- `http_request_size_bytes` – request size
- `http_response_size_bytes` – response size

### Grafana Dashboard

Access Grafana at **http://localhost:3000** (default: admin/admin).

The pre-provisioned **"Semantic Omni Search Engine"** dashboard shows:
- Search latency percentiles (p50, p95, p99)
- Request rate (search and upload RPS)
- Error rate
- Total searches and uploads

### Structured Logging

All logs are emitted as structured JSON:
```json
{"event": "POST /search → 200 (45.1 ms)", "level": "info", "timestamp": "2024-01-01T00:00:00Z"}
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push:

1. **Test** – install deps, run pytest, run ruff linter
2. **Build** – build and push Docker images to GHCR
3. **Deploy** – apply K8s manifests and wait for rollout (requires `KUBECONFIG` secret)

---

## Performance Characteristics

| Operation | Typical Latency |
|---|---|
| Text search | ~15–50 ms |
| Image search | ~20–60 ms |
| PDF ingestion | ~500 ms – 2s |
| Image ingestion | ~200–500 ms |
| Audio ingestion | ~2–30s (depends on duration) |
| Video ingestion | ~5–60s (depends on duration) |

> **Note**: First request after startup will be slower due to model loading (30–120s depending on hardware).

---

## License

MIT License – see [LICENSE](LICENSE).