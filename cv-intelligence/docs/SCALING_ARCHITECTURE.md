# Scaling CV Intelligence to 90GB+ Data

This document outlines the architectural changes required to scale the CV Intelligence system from its current design (~58 files) to handle 90GB+ of CV documents.

## Table of Contents

1. [Qdrant Storage Fundamentals](#qdrant-storage-fundamentals)
2. [Local Vector Store Deployment](#local-vector-store-deployment)
3. [Current Architecture Limitations](#current-architecture-limitations)
4. [Target Architecture](#target-architecture)
5. [Infrastructure Changes](#infrastructure-changes)
6. [Code Changes](#code-changes)
7. [Migration Strategy](#migration-strategy)
8. [Cost Estimates](#cost-estimates)
9. [Implementation Checklist](#implementation-checklist)

---

## Qdrant Storage Fundamentals

Before scaling, it's essential to understand how Qdrant stores data and the implications for your architecture.

### Storage Modes Overview

Qdrant supports three storage modes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       QDRANT STORAGE MODES                              │
├───────────────────┬───────────────────┬─────────────────────────────────┤
│    IN-MEMORY      │    FILE-BASED     │    SERVER (Docker/Cloud)        │
│    (:memory:)     │    (path=./data)  │    (url=localhost:6333)         │
├───────────────────┼───────────────────┼─────────────────────────────────┤
│ Stored in RAM     │ Stored on disk    │ Stored on disk                  │
│ Lost on exit      │ Persists          │ Persists                        │
│ No server needed  │ No server needed  │ Requires running server         │
│ Single process    │ Single process    │ Multi-process safe              │
│ Fastest           │ Fast              │ Network overhead                │
└───────────────────┴───────────────────┴─────────────────────────────────┘
```

### What is Persistence?

**Persistence = Data survives after the program stops**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   WITHOUT PERSISTENCE              WITH PERSISTENCE                     │
│   (In-Memory)                      (File-Based or Server)               │
│                                                                         │
│   1. Run program                   1. Run program                       │
│   2. Create 1000 vectors           2. Create 1000 vectors               │
│   3. Stop program                  3. Stop program                      │
│   4. Run again                     4. Run again                         │
│   5. Vectors = 0 ❌                5. Vectors = 1000 ✅                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Simple Analogy:**

| Type | Like | What Happens |
|------|------|--------------|
| No Persistence | Writing on whiteboard | Erased when you leave |
| Persistence | Writing in notebook | Still there tomorrow |

### Storage Mode Details

#### 1. In-Memory Mode

```python
from qdrant_client import QdrantClient

client = QdrantClient(location=":memory:")
```

**How it works:**
- Data stored only in RAM
- Each `QdrantClient()` instance creates a separate database
- Data is lost when the Python process ends

**Use when:**
- Running unit tests
- Quick one-time experiments
- CI/CD pipelines
- Prototyping

**Don't use when:**
- You need data to persist
- Multiple processes need access
- Production deployments

#### 2. File-Based Mode (Embedded)

```python
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_data")
```

**How it works:**
- Creates a folder on disk with all vector data
- Qdrant engine runs embedded in your Python process
- No separate server required

```
your-project/
├── app.py
├── qdrant_data/              ← Qdrant stores everything here
│   ├── collection/
│   │   └── cv_documents/
│   │       ├── segments/
│   │       ├── vectors/
│   │       └── payload/
│   └── meta.json
```

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│              YOUR PYTHON PROCESS                    │
│                                                     │
│   app.py                                            │
│      │                                              │
│      ▼                                              │
│   QdrantClient(path="./qdrant_data")                │
│      │                                              │
│      ▼                                              │
│   ┌───────────────────────────────────┐             │
│   │   Qdrant Engine (embedded)        │             │
│   │   - Runs inside your app          │             │
│   │   - Reads/writes to disk          │             │
│   └───────────────────────────────────┘             │
│                   │                                 │
└───────────────────┼─────────────────────────────────┘
                    ▼
            ./qdrant_data/  (files on disk)
```

**Use when:**
- Single-script batch processing
- Personal CLI tools
- Jupyter notebook analysis
- Small personal projects (single user)

**Don't use when:**
- Multiple processes need simultaneous access (file lock error)
- Web applications with multiple workers
- Production deployments

**The Problem with File-Based:**
```
❌ FAILS - Two processes can't share file-based storage

┌──────────────┐         ┌──────────────┐
│   app.py     │         │  ingest.py   │
│   (web app)  │         │  (ingestion) │
└──────┬───────┘         └──────┬───────┘
       │                        │
       ▼                        ▼
    ┌─────────────────────────────┐
    │      ./qdrant_data/         │  ← LOCKED by first process!
    │      (file lock error!)     │
    └─────────────────────────────┘
```

#### 3. Server Mode (Docker/Cloud)

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
```

**How it works:**
- Qdrant runs as a separate process (Docker container or cloud service)
- Your app connects via HTTP/gRPC
- Multiple apps can connect to the same server

**Architecture:**
```
✅ WORKS - Multiple processes connect to one server

┌──────────────┐         ┌──────────────┐
│   app.py     │         │  ingest.py   │
└──────┬───────┘         └──────┬───────┘
       │                        │
       │   HTTP connection      │
       ▼                        ▼
    ┌─────────────────────────────┐
    │   Qdrant Server (Docker)    │
    │   localhost:6333            │
    │   - Dashboard available     │
    │   - Data persists           │
    └─────────────────────────────┘
```

**Use when:**
- Web applications (Flask, FastAPI, Streamlit)
- APIs serving multiple users
- Separate ingestion + query processes
- Team collaboration
- Production deployments
- You want the Qdrant dashboard

**Don't use when:**
- Quick local experiments (overkill)
- Unit tests (use in-memory instead)

### When to Use What - Decision Guide

```
Do you need data after the program exits?
│
├─ NO  → In-Memory (:memory:)
│
└─ YES → Will multiple processes access it simultaneously?
         │
         ├─ NO  → File-Based (path="./data")
         │
         └─ YES → Server Mode (url="localhost:6333")
```

### Quick Reference Table

| Mode | Code | Persistence | Multi-Process | Dashboard | Use Case |
|------|------|-------------|---------------|-----------|----------|
| In-Memory | `QdrantClient(location=":memory:")` | No | No | No | Unit tests, experiments |
| File-Based | `QdrantClient(path="./data")` | Yes | No | No | CLI tools, notebooks |
| Server | `QdrantClient(url="localhost:6333")` | Yes | Yes | Yes | Web apps, production |

### Configuration in This Project

The project uses the `USE_LOCAL_QDRANT` environment variable:

```python
# config.py
USE_LOCAL_QDRANT = os.getenv("USE_LOCAL_QDRANT", "true").lower() == "true"

# vector_store.py
if use_local:
    self.client = QdrantClient(location=":memory:")  # In-memory
else:
    self.client = QdrantClient(url=qdrant_url)       # Server mode
```

**Recommended Settings:**

| Environment | USE_LOCAL_QDRANT | Why |
|-------------|------------------|-----|
| Unit Tests | `true` | Fast, isolated, no cleanup needed |
| Development | `false` | See data in dashboard, persistence |
| Production | `false` | Multi-process, persistence, scalability |

**For the CV Intelligence project:**
```bash
# .env
USE_LOCAL_QDRANT=false
QDRANT_URL=http://localhost:6333
```

This is correct because:
- `app.py` (Streamlit) and `ingest.py` are separate processes
- Both need to access the same data
- You want to see collections in the dashboard

---

## Local Vector Store Deployment

This section covers deploying Qdrant locally for development and production use cases.

### Local Deployment Options

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOCAL DEPLOYMENT OPTIONS                             │
├─────────────────────┬─────────────────────┬─────────────────────────────┤
│   Single Docker     │   Docker Compose    │   Local Cluster             │
│   (Simplest)        │   (Recommended)     │   (High Availability)       │
├─────────────────────┼─────────────────────┼─────────────────────────────┤
│ One container       │ Multiple services   │ 3+ Qdrant nodes             │
│ Manual start        │ Orchestrated        │ Sharding + replication      │
│ Good for dev        │ Good for prod       │ For large scale             │
└─────────────────────┴─────────────────────┴─────────────────────────────┘
```

### Option 1: Single Docker Container (Development)

**Start Qdrant:**
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**What each flag does:**

| Flag | Purpose |
|------|---------|
| `-d` | Run in background (detached) |
| `--name qdrant` | Container name for easy reference |
| `-p 6333:6333` | HTTP API and dashboard |
| `-p 6334:6334` | gRPC API (faster for large operations) |
| `-v ...:/qdrant/storage` | Persist data to local folder |

**Verify it's running:**
```bash
# Check container status
docker ps

# Check Qdrant health
curl http://localhost:6333/health

# Open dashboard
open http://localhost:6333/dashboard
```

**Stop and restart:**
```bash
docker stop qdrant    # Stop (data preserved)
docker start qdrant   # Start again (data still there)
docker rm qdrant      # Remove container (data still in qdrant_storage/)
```

### Option 2: Docker Compose (Production-Ready Local)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      # Enable API key for security
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY:-}
      # Performance tuning
      - QDRANT__STORAGE__ON_DISK_PAYLOAD=true
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - ./redis_data:/data
    command: redis-server --appendonly yes

  app:
    build: .
    restart: always
    ports:
      - "8501:8501"
    depends_on:
      - qdrant
      - redis
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_API_KEY=${QDRANT_API_KEY:-}
      - USE_LOCAL_QDRANT=false
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data

volumes:
  qdrant_storage:
  redis_data:
```

**Usage:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f qdrant

# Stop all services
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v
```

### Option 3: Local with Persistence and Backup

**Directory structure:**
```
cv-intelligence/
├── docker-compose.yml
├── qdrant_storage/          ← Vector data (persist this!)
│   └── collection/
│       └── cv_documents/
├── redis_data/              ← Job queue data
├── backups/                 ← Snapshots
└── data/
    └── sample_cvs/
```

**Backup script (`scripts/backup_qdrant.sh`):**
```bash
#!/bin/bash
# Create a snapshot of the Qdrant collection

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COLLECTION="cv_documents"

mkdir -p $BACKUP_DIR

# Create snapshot via API
curl -X POST "http://localhost:6333/collections/${COLLECTION}/snapshots"

# Copy snapshot to backup directory
SNAPSHOT=$(curl -s "http://localhost:6333/collections/${COLLECTION}/snapshots" | jq -r '.result[-1].name')

echo "Created snapshot: $SNAPSHOT"
echo "Backup location: $BACKUP_DIR/$SNAPSHOT"
```

**Restore from backup:**
```bash
# List available snapshots
curl http://localhost:6333/collections/cv_documents/snapshots

# Restore a snapshot
curl -X PUT "http://localhost:6333/collections/cv_documents/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///qdrant/storage/snapshots/cv_documents/snapshot_name.snapshot"}'
```

### Local Hardware Requirements

**For different data sizes:**

| Data Size | Documents | Vectors | RAM | Storage | CPU |
|-----------|-----------|---------|-----|---------|-----|
| Small | < 1K | < 10K | 2GB | 10GB | 2 cores |
| Medium | 1K - 10K | 10K - 100K | 4GB | 50GB | 4 cores |
| Large | 10K - 100K | 100K - 1M | 8GB | 100GB | 8 cores |
| Very Large | 100K - 500K | 1M - 5M | 16GB | 250GB | 8 cores |
| Huge | 500K+ | 5M+ | 32GB+ | 500GB+ | 16 cores |

**For 90GB of CV data (~500K documents):**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| Storage | 300GB SSD | 500GB NVMe |
| CPU | 8 cores | 16 cores |
| GPU (for embeddings) | 8GB VRAM | 16GB VRAM |

### Local Environment Configuration

**.env for local development:**
```bash
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=cv_documents
USE_LOCAL_QDRANT=false

# Optional: API key for security (leave empty for no auth)
QDRANT_API_KEY=

# Redis (for job queue)
REDIS_URL=redis://localhost:6379

# Embeddings
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small

# Or use local embeddings (no API cost)
# USE_LOCAL_EMBEDDINGS=true
# EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

**.env for local production:**
```bash
# Qdrant with API key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=cv_documents
USE_LOCAL_QDRANT=false
QDRANT_API_KEY=your-secure-random-key-here

# Redis
REDIS_URL=redis://localhost:6379

# Use local embeddings to avoid API costs
USE_LOCAL_EMBEDDINGS=true
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=100
WORKER_BATCH_SIZE=50
```

### Adding API Key Support to Code

Update `src/config.py`:
```python
# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cv_documents")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Add this
USE_LOCAL_QDRANT = os.getenv("USE_LOCAL_QDRANT", "true").lower() == "true"
```

Update `src/vector_store.py`:
```python
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, USE_LOCAL_QDRANT

class CVVectorStore:
    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        use_local: bool = USE_LOCAL_QDRANT,
        qdrant_url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY  # Add this
    ):
        self.collection_name = collection_name

        if use_local:
            self.client = QdrantClient(location=":memory:")
        else:
            # Connect to server with optional API key
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=api_key if api_key else None
            )
```

### Local vs Cloud Comparison

| Aspect | Local Docker | Qdrant Cloud |
|--------|--------------|--------------|
| **Cost** | Hardware only | $0 (free tier) - $200+/mo |
| **Setup** | DIY | Managed |
| **Maintenance** | You handle updates | Automatic |
| **Backups** | Manual | Automatic |
| **Scaling** | Limited by hardware | Easy scaling |
| **Latency** | Lowest (same machine) | Network latency |
| **Data Privacy** | Full control | Third-party |
| **Dashboard** | http://localhost:6333 | cloud.qdrant.io |

### When to Use Local

**Use Local Docker when:**
- Data privacy is critical (healthcare, finance, legal)
- You have sufficient hardware
- Low latency is required
- You want to avoid recurring cloud costs
- Offline operation is needed
- You're in development/testing phase

**Use Cloud when:**
- You need automatic scaling
- You don't want to manage infrastructure
- You need high availability across regions
- Your team is distributed
- You're starting small and may scale

### Common Local Setup Issues

**Issue 1: Port already in use**
```bash
# Check what's using port 6333
lsof -i :6333

# Use a different port
docker run -d -p 6334:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
# Then set QDRANT_URL=http://localhost:6334
```

**Issue 2: Permission denied on volume**
```bash
# Fix permissions
sudo chown -R $(id -u):$(id -g) ./qdrant_storage
```

**Issue 3: Container keeps restarting**
```bash
# Check logs
docker logs qdrant

# Common fix: increase memory
docker run -d --memory=4g -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Issue 4: Slow performance**
```bash
# Use SSD, not HDD
# Enable on-disk payload for large data
docker run -d \
  -e QDRANT__STORAGE__ON_DISK_PAYLOAD=true \
  -p 6333:6333 \
  -v /ssd/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### Quick Start Commands

```bash
# Start Qdrant locally
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Verify
curl http://localhost:6333/health

# Run ingestion
python ingest.py

# Check collections in dashboard
open http://localhost:6333/dashboard

# Run the app
streamlit run app.py
```

---

## Current Architecture Limitations

### Current Design

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  File System │───►│  Python App  │───►│   Qdrant     │
│  (58 files)  │    │  (single)    │    │  (single)    │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Bottlenecks

| Component | Current Behavior | Problem at 90GB |
|-----------|-----------------|-----------------|
| Document Loading | Loads all files into memory | Out of Memory (OOM) crash |
| Chunking | Sequential, single-threaded | Days to complete |
| Embedding | OpenAI API, sequential | $500-2000 cost, rate limits |
| Vector Store | Single Qdrant container | Performance degradation at 10M+ vectors |
| Metadata Extraction | LLM calls per document | Extreme cost and time |

### Current Code Issues

```python
# document_processor.py - Loads everything into memory
def load_documents(self) -> List[Document]:
    documents = []
    for file_path in self.data_dir.rglob("*"):  # All files at once
        documents.extend(self._load_file(file_path))
    return documents  # OOM with 90GB

# ingest.py - No batching, no resume capability
documents = processor.load_documents()  # Blocks until all loaded
chunks = processor.chunk_documents(documents)  # All in memory
vectorstore.create_collection(chunks)  # Single operation
```

---

## Target Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        File Storage                              │
│                    (S3 / NFS / Local)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Job Queue (Redis)                           │
│              Tracks: pending, processing, completed              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌───────────┐  ┌───────────┐  ┌───────────┐
     │  Worker 1 │  │  Worker 2 │  │  Worker N │
     │           │  │           │  │           │
     │ - Load    │  │ - Load    │  │ - Load    │
     │ - Chunk   │  │ - Chunk   │  │ - Chunk   │
     │ - Embed   │  │ - Embed   │  │ - Embed   │
     └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Qdrant Cluster                                │
│         ┌─────────┐  ┌─────────┐  ┌─────────┐                   │
│         │ Node 1  │  │ Node 2  │  │ Node 3  │                   │
│         │ Shard A │  │ Shard B │  │ Shard C │                   │
│         └─────────┘  └─────────┘  └─────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| File Storage | Durable storage for raw CV files |
| Job Queue | Track processing state, enable resume |
| Workers | Parallel document processing |
| Embedding Service | Local GPU or batched API calls |
| Qdrant Cluster | Distributed vector storage and search |

---

## Infrastructure Changes

### 1. Docker Compose for Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Redis for job queue
  redis:
    image: redis:7-alpine
    restart: always
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # Qdrant Node 1
  qdrant-node-1:
    image: qdrant/qdrant:latest
    restart: always
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data_1:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G

  # Qdrant Node 2
  qdrant-node-2:
    image: qdrant/qdrant:latest
    restart: always
    volumes:
      - qdrant_data_2:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__P2P__BOOTSTRAP=http://qdrant-node-1:6335
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G

  # Qdrant Node 3
  qdrant-node-3:
    image: qdrant/qdrant:latest
    restart: always
    volumes:
      - qdrant_data_3:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__P2P__BOOTSTRAP=http://qdrant-node-1:6335
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G

  # Embedding Service (Local GPU)
  embedding-service:
    build:
      context: .
      dockerfile: Dockerfile.embedding
    restart: always
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Ingestion Workers
  worker:
    build: .
    restart: always
    depends_on:
      - redis
      - qdrant-node-1
      - embedding-service
    environment:
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant-node-1:6333
      - EMBEDDING_SERVICE_URL=http://embedding-service:8001
      - WORKER_BATCH_SIZE=50
    deploy:
      replicas: 4

  # Main Application
  app:
    build: .
    restart: always
    ports:
      - "8000:8000"
    depends_on:
      - qdrant-node-1
      - redis
    environment:
      - QDRANT_URL=http://qdrant-node-1:6333
      - REDIS_URL=redis://redis:6379

volumes:
  redis_data:
  qdrant_data_1:
  qdrant_data_2:
  qdrant_data_3:
```

### 2. Hardware Requirements

| Component | Specification | Quantity |
|-----------|--------------|----------|
| Qdrant Nodes | 32GB RAM, 8 CPU, 500GB SSD | 3 |
| Worker Nodes | 16GB RAM, 8 CPU | 4-8 |
| GPU Node (Embedding) | 16GB VRAM (RTX 4090 / A10) | 1 |
| Redis | 8GB RAM, 4 CPU | 1 |

### 3. Environment Variables

```bash
# .env.production
# Qdrant
QDRANT_URL=http://qdrant-node-1:6333
QDRANT_API_KEY=your-secure-api-key
QDRANT_COLLECTION=cv_documents
USE_LOCAL_QDRANT=false

# Redis
REDIS_URL=redis://redis:6379

# Embedding
EMBEDDING_SERVICE_URL=http://embedding-service:8001
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_BATCH_SIZE=32

# Processing
WORKER_BATCH_SIZE=50
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# OpenAI (fallback or for LLM queries)
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

---

## Code Changes

### 1. New Project Structure

```
cv-intelligence/
├── src/
│   ├── __init__.py
│   ├── config.py              # Updated config
│   ├── document_processor.py  # Updated for streaming
│   ├── vector_store.py        # Updated for cluster
│   ├── metadata_extractor.py
│   ├── query_engine.py
│   ├── job_queue.py           # NEW: Redis job queue
│   ├── worker.py              # NEW: Batch worker
│   └── embedding_service.py   # NEW: Local embeddings
├── scripts/
│   ├── enqueue_files.py       # NEW: Add files to queue
│   └── monitor_progress.py    # NEW: Track ingestion
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.embedding
│   └── docker-compose.prod.yml
└── ...
```

### 2. Job Queue Implementation

```python
# src/job_queue.py
"""Redis-based job queue for document processing."""

import json
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum
import redis


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobQueue:
    """Manages document processing jobs with Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.queue_key = "cv:jobs:pending"
        self.processing_key = "cv:jobs:processing"
        self.completed_key = "cv:jobs:completed"
        self.failed_key = "cv:jobs:failed"
        self.metadata_prefix = "cv:job:"

    def _file_hash(self, file_path: Path) -> str:
        """Generate unique ID for a file."""
        return hashlib.md5(str(file_path).encode()).hexdigest()[:12]

    def enqueue_files(self, file_paths: List[Path]) -> Dict[str, int]:
        """Add files to the processing queue."""
        stats = {"added": 0, "skipped": 0}

        for file_path in file_paths:
            job_id = self._file_hash(file_path)

            # Skip if already processed or in queue
            if self._job_exists(job_id):
                stats["skipped"] += 1
                continue

            job_data = {
                "id": job_id,
                "file_path": str(file_path),
                "status": JobStatus.PENDING.value,
                "created_at": self._now(),
                "attempts": 0
            }

            # Store job metadata
            self.redis.set(
                f"{self.metadata_prefix}{job_id}",
                json.dumps(job_data)
            )

            # Add to pending queue
            self.redis.lpush(self.queue_key, job_id)
            stats["added"] += 1

        return stats

    def get_next_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get next job from queue (blocking)."""
        result = self.redis.brpoplpush(
            self.queue_key,
            self.processing_key,
            timeout=30
        )

        if not result:
            return None

        job_id = result.decode() if isinstance(result, bytes) else result
        job_data = self._get_job(job_id)

        if job_data:
            job_data["status"] = JobStatus.PROCESSING.value
            job_data["worker_id"] = worker_id
            job_data["started_at"] = self._now()
            job_data["attempts"] += 1
            self._save_job(job_data)

        return job_data

    def complete_job(self, job_id: str, vector_ids: List[str]):
        """Mark job as completed."""
        job_data = self._get_job(job_id)
        if job_data:
            job_data["status"] = JobStatus.COMPLETED.value
            job_data["completed_at"] = self._now()
            job_data["vector_ids"] = vector_ids
            self._save_job(job_data)

            self.redis.lrem(self.processing_key, 1, job_id)
            self.redis.sadd(self.completed_key, job_id)

    def fail_job(self, job_id: str, error: str, retry: bool = True):
        """Mark job as failed, optionally retry."""
        job_data = self._get_job(job_id)
        if job_data:
            job_data["status"] = JobStatus.FAILED.value
            job_data["error"] = error
            job_data["failed_at"] = self._now()
            self._save_job(job_data)

            self.redis.lrem(self.processing_key, 1, job_id)

            if retry and job_data["attempts"] < 3:
                self.redis.lpush(self.queue_key, job_id)
            else:
                self.redis.sadd(self.failed_key, job_id)

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        return {
            "pending": self.redis.llen(self.queue_key),
            "processing": self.redis.llen(self.processing_key),
            "completed": self.redis.scard(self.completed_key),
            "failed": self.redis.scard(self.failed_key)
        }

    def _job_exists(self, job_id: str) -> bool:
        return self.redis.exists(f"{self.metadata_prefix}{job_id}")

    def _get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        data = self.redis.get(f"{self.metadata_prefix}{job_id}")
        return json.loads(data) if data else None

    def _save_job(self, job_data: Dict[str, Any]):
        self.redis.set(
            f"{self.metadata_prefix}{job_data['id']}",
            json.dumps(job_data)
        )

    def _now(self) -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat()
```

### 3. Batch Worker Implementation

```python
# src/worker.py
"""Worker process for batch document ingestion."""

import os
import signal
import logging
from pathlib import Path
from typing import List
import uuid

from .config import (
    REDIS_URL, QDRANT_URL, QDRANT_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP, WORKER_BATCH_SIZE
)
from .job_queue import JobQueue
from .document_processor import CVDocumentProcessor
from .vector_store import CVVectorStore
from .embedding_service import LocalEmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionWorker:
    """Worker that processes documents from the job queue."""

    def __init__(self):
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self.running = True
        self.queue = JobQueue(redis_url=REDIS_URL)
        self.vectorstore = CVVectorStore(
            qdrant_url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            use_local=False
        )
        self.embedding_service = LocalEmbeddingService()

        # Handle graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info(f"Worker {self.worker_id} shutting down...")
        self.running = False

    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} started")

        while self.running:
            try:
                job = self.queue.get_next_job(self.worker_id)

                if job is None:
                    continue

                logger.info(f"Processing job {job['id']}: {job['file_path']}")

                try:
                    vector_ids = self._process_file(job['file_path'])
                    self.queue.complete_job(job['id'], vector_ids)
                    logger.info(f"Completed job {job['id']}: {len(vector_ids)} vectors")

                except Exception as e:
                    logger.error(f"Failed job {job['id']}: {e}")
                    self.queue.fail_job(job['id'], str(e))

            except Exception as e:
                logger.error(f"Worker error: {e}")

        logger.info(f"Worker {self.worker_id} stopped")

    def _process_file(self, file_path: str) -> List[str]:
        """Process a single file and return vector IDs."""
        path = Path(file_path)

        # Load and chunk document
        processor = CVDocumentProcessor(data_dir=path.parent)
        documents = processor._load_file(path)
        chunks = processor.chunk_documents(documents)

        # Generate embeddings locally
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        # Add to vector store
        vector_ids = self.vectorstore.add_documents_with_embeddings(
            chunks, embeddings
        )

        return vector_ids


def main():
    worker = IngestionWorker()
    worker.run()


if __name__ == "__main__":
    main()
```

### 4. Local Embedding Service

```python
# src/embedding_service.py
"""Local embedding service using sentence-transformers."""

import os
from typing import List
import numpy as np

# Use sentence-transformers for local embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddingService:
    """Generate embeddings locally using GPU."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 32,
        device: str = None
    ):
        self.batch_size = batch_size
        self.device = device or ("cuda" if self._gpu_available() else "cpu")

        print(f"Loading embedding model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _gpu_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()


# FastAPI service wrapper for distributed deployment
def create_embedding_api():
    """Create FastAPI app for embedding service."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()
    service = LocalEmbeddingService()

    class EmbedRequest(BaseModel):
        texts: List[str]

    class EmbedResponse(BaseModel):
        embeddings: List[List[float]]
        dimension: int

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest):
        embeddings = service.embed_batch(request.texts)
        return EmbedResponse(
            embeddings=embeddings,
            dimension=service.dimension
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "dimension": service.dimension}

    return app
```

### 5. Updated Vector Store

```python
# src/vector_store.py (updated sections)
"""Vector store operations with Qdrant - Updated for scale."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition,
    MatchValue, Range, PointStruct, OptimizersConfigDiff,
    HnswConfigDiff
)

from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION


class CVVectorStore:
    """Vector store for CV documents - Scaled for large datasets."""

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        qdrant_url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        use_local: bool = False
    ):
        self.collection_name = collection_name

        if use_local:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=api_key,
                timeout=60  # Increase timeout for large operations
            )

    def create_collection_optimized(
        self,
        vector_size: int = 1024,  # BGE-large dimension
        shard_number: int = 3,    # Distribute across nodes
        replication_factor: int = 2
    ):
        """Create collection optimized for large scale."""

        # Delete if exists
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                on_disk=True  # Store vectors on disk for large datasets
            ),
            shard_number=shard_number,
            replication_factor=replication_factor,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=50000,  # Build index after 50k points
                memmap_threshold=100000    # Use mmap after 100k points
            ),
            hnsw_config=HnswConfigDiff(
                m=16,              # Connections per node
                ef_construct=100,  # Build quality
                on_disk=True       # Store HNSW index on disk
            )
        )

        print(f"Created optimized collection '{self.collection_name}'")

    def add_documents_with_embeddings(
        self,
        documents: List,
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> List[str]:
        """Add documents with pre-computed embeddings in batches."""

        vector_ids = []
        points = []

        for doc, embedding in zip(documents, embeddings):
            point_id = str(uuid.uuid4())
            vector_ids.append(point_id)

            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": doc.page_content,
                    **doc.metadata
                }
            ))

            # Upload in batches
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=False  # Async for better throughput
                )
                points = []

        # Upload remaining
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

        return vector_ids

    def search_with_embedding(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search using pre-computed query embedding."""

        qdrant_filter = self._build_filter(filter_dict) if filter_dict else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
            }
            for hit in results
        ]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
                "segments_count": len(info.segments) if info.segments else 0,
                "disk_data_size": info.disk_data_size,
                "ram_data_size": info.ram_data_size
            }
        except Exception as e:
            return {"error": str(e)}

    # ... (keep existing _build_filter and other methods)
```

### 6. Streaming Document Processor

```python
# src/document_processor.py (updated sections)
"""Document processor - Updated for streaming/batch processing."""

from typing import Iterator, List, Optional
from pathlib import Path
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class CVDocumentProcessor:
    """Process CV documents with streaming support."""

    def __init__(
        self,
        data_dir: Path,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.data_dir = Path(data_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def iter_files(self) -> Iterator[Path]:
        """Iterate over files without loading into memory."""
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in self.data_dir.rglob(f"*{ext}"):
                yield file_path

    def count_files(self) -> int:
        """Count total files to process."""
        return sum(1 for _ in self.iter_files())

    def process_file(self, file_path: Path) -> List[Document]:
        """Process a single file and return chunks."""
        try:
            documents = self._load_file(file_path)
            chunks = self.chunk_documents(documents)
            return chunks
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

    def iter_batches(
        self,
        batch_size: int = 50
    ) -> Iterator[List[Path]]:
        """Yield batches of file paths."""
        batch = []
        for file_path in self.iter_files():
            batch.append(file_path)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ... (keep existing _load_file, chunk_documents methods)
```

### 7. Enqueue Script

```python
# scripts/enqueue_files.py
"""Script to enqueue files for processing."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_queue import JobQueue
from src.document_processor import CVDocumentProcessor
from src.config import DATA_DIR, REDIS_URL


def main():
    parser = argparse.ArgumentParser(description="Enqueue CV files for processing")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing CV files"
    )
    parser.add_argument(
        "--redis-url",
        default=REDIS_URL,
        help="Redis URL"
    )
    args = parser.parse_args()

    # Initialize
    queue = JobQueue(redis_url=args.redis_url)
    processor = CVDocumentProcessor(data_dir=args.data_dir)

    # Count files
    total_files = processor.count_files()
    print(f"Found {total_files} files in {args.data_dir}")

    # Enqueue all files
    file_paths = list(processor.iter_files())
    stats = queue.enqueue_files(file_paths)

    print(f"Enqueued: {stats['added']}, Skipped (already processed): {stats['skipped']}")
    print(f"\nQueue stats: {queue.get_stats()}")


if __name__ == "__main__":
    main()
```

### 8. Monitor Script

```python
# scripts/monitor_progress.py
"""Monitor ingestion progress."""

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.job_queue import JobQueue
from src.vector_store import CVVectorStore
from src.config import REDIS_URL, QDRANT_URL, QDRANT_API_KEY


def main():
    parser = argparse.ArgumentParser(description="Monitor ingestion progress")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval in seconds")
    args = parser.parse_args()

    queue = JobQueue(redis_url=REDIS_URL)
    vectorstore = CVVectorStore(
        qdrant_url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        use_local=False
    )

    def print_stats():
        queue_stats = queue.get_stats()
        vector_stats = vectorstore.get_collection_stats()

        total = sum(queue_stats.values())
        completed = queue_stats['completed']
        progress = (completed / total * 100) if total > 0 else 0

        print("\033[2J\033[H")  # Clear screen
        print("=" * 50)
        print("CV Intelligence Ingestion Monitor")
        print("=" * 50)
        print(f"\nQueue Status:")
        print(f"  Pending:    {queue_stats['pending']:>8}")
        print(f"  Processing: {queue_stats['processing']:>8}")
        print(f"  Completed:  {queue_stats['completed']:>8}")
        print(f"  Failed:     {queue_stats['failed']:>8}")
        print(f"\n  Progress:   {progress:.1f}%")
        print(f"\nVector Store:")
        if 'error' not in vector_stats:
            print(f"  Vectors:    {vector_stats.get('points_count', 0):>8}")
            print(f"  Status:     {vector_stats.get('status', 'unknown')}")
        else:
            print(f"  Error: {vector_stats['error']}")
        print("\n" + "=" * 50)

    if args.watch:
        try:
            while True:
                print_stats()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped monitoring")
    else:
        print_stats()


if __name__ == "__main__":
    main()
```

---

## Migration Strategy

### Phase 1: Preparation (Before Migration)

1. **Audit current data**
   ```bash
   # Count files and estimate size
   find /path/to/cvs -type f \( -name "*.pdf" -o -name "*.docx" \) | wc -l
   du -sh /path/to/cvs
   ```

2. **Set up infrastructure**
   - Deploy Redis
   - Deploy Qdrant cluster (3 nodes)
   - Set up GPU instance for embeddings

3. **Test with subset**
   ```bash
   # Test with 1000 files first
   python scripts/enqueue_files.py --data-dir /path/to/test_subset
   python -m src.worker
   ```

### Phase 2: Migration

1. **Create optimized collection**
   ```python
   from src.vector_store import CVVectorStore
   store = CVVectorStore()
   store.create_collection_optimized(
       vector_size=1024,  # BGE-large
       shard_number=3,
       replication_factor=2
   )
   ```

2. **Enqueue all files**
   ```bash
   python scripts/enqueue_files.py --data-dir /path/to/all_cvs
   ```

3. **Start workers**
   ```bash
   # Start multiple workers
   docker-compose -f docker-compose.prod.yml up -d --scale worker=8
   ```

4. **Monitor progress**
   ```bash
   python scripts/monitor_progress.py --watch
   ```

### Phase 3: Validation

1. **Verify counts**
   ```python
   stats = vectorstore.get_collection_stats()
   assert stats['points_count'] > expected_minimum
   ```

2. **Test queries**
   ```python
   results = query_engine.search("Python developer 5 years experience")
   assert len(results) > 0
   ```

3. **Performance test**
   ```bash
   # Run load test
   locust -f tests/load_test.py --host http://localhost:8000
   ```

---

## Cost Estimates

### One-Time Ingestion Costs

| Item | Specification | Cost |
|------|--------------|------|
| GPU Instance (embedding) | A10 for 24 hours | ~$25-50 |
| Compute (workers) | 8-core x 4 for 24 hours | ~$20-40 |
| OR OpenAI Embeddings | ~11B tokens | ~$220 |

### Monthly Infrastructure Costs

| Component | Cloud (AWS/GCP) | Self-Hosted |
|-----------|-----------------|-------------|
| Qdrant Cluster (3 nodes) | $300-500 | $150-200 (dedicated servers) |
| Redis | $30-50 | $10-20 |
| Application Servers | $100-200 | $50-100 |
| Storage (500GB) | $50-100 | $30-50 |
| **Total** | **$480-850/mo** | **$240-370/mo** |

### Qdrant Cloud Alternative

| Tier | Vectors | Cost |
|------|---------|------|
| Free | 1GB (~500K vectors) | $0 |
| Standard | 10M vectors | ~$200/mo |
| Enterprise | Unlimited | Custom |

---

## Implementation Checklist

### Infrastructure Setup

- [ ] Deploy Redis instance
- [ ] Deploy Qdrant cluster (3 nodes minimum)
- [ ] Set up GPU instance for embeddings
- [ ] Configure networking between services
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure backups for Qdrant volumes

### Code Changes

- [ ] Implement `JobQueue` class
- [ ] Implement `IngestionWorker` class
- [ ] Implement `LocalEmbeddingService`
- [ ] Update `CVVectorStore` for cluster mode
- [ ] Update `CVDocumentProcessor` for streaming
- [ ] Create `enqueue_files.py` script
- [ ] Create `monitor_progress.py` script
- [ ] Update configuration for new environment variables

### Testing

- [ ] Unit tests for new components
- [ ] Integration tests with small dataset
- [ ] Load test with 10K documents
- [ ] Failover test (kill a Qdrant node)
- [ ] Resume test (stop and restart workers)

### Documentation

- [ ] Update README with new architecture
- [ ] Document deployment procedures
- [ ] Create runbook for common operations
- [ ] Document backup/restore procedures

---

## Appendix: Quick Reference Commands

```bash
# Start infrastructure
docker-compose -f docker-compose.prod.yml up -d

# Enqueue files
python scripts/enqueue_files.py --data-dir /data/cvs

# Start workers (4 instances)
docker-compose -f docker-compose.prod.yml up -d --scale worker=4

# Monitor progress
python scripts/monitor_progress.py --watch

# Check Qdrant cluster health
curl http://localhost:6333/cluster

# Check queue stats
redis-cli -h localhost LLEN cv:jobs:pending

# Backup Qdrant
curl -X POST "http://localhost:6333/collections/cv_documents/snapshots"
```
