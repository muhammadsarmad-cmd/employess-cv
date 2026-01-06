# CV Intelligence - Complete Setup Guide

This guide walks you through setting up the CV Intelligence system from laptop testing (20-50 CVs) to production scale (90GB+).

## Table of Contents

1. [Quick Start (Laptop Testing)](#quick-start-laptop-testing)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the System](#running-the-system)
6. [Scaling to Production](#scaling-to-production)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start (Laptop Testing)

**Goal**: Test with 20-50 CVs on your laptop using CPU embeddings.

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- 8GB+ RAM
- 10GB+ free disk space

### 1. Install Dependencies

```bash
cd cv-intelligence

# Install Python packages
pip install -r requirements.txt

# This will install:
# - sentence-transformers (for local embeddings)
# - redis (for job queue)
# - qdrant-client (for vector storage)
# - torch (CPU version - for embeddings)
```

### 2. Start Infrastructure

```bash
# Start Redis and Qdrant
docker-compose up -d
docker compose up -d
# Verify services are running
docker-compose ps

# Check Qdrant dashboard
open http://localhost:6333/dashboard  # or visit in browser
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# The default settings are optimized for laptop testing:
# - USE_LOCAL_EMBEDDINGS=true (CPU-based, no API costs)
# - LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# - EMBEDDING_DEVICE=cpu
```

### 4. Prepare Test Data

```bash
# Create test directory
mkdir -p data/test_cvs

# Add 20-50 CV files (PDF, DOCX) to data/test_cvs/
# You can use sample CVs or your own

# Or use the sample generator
python generate_samples.py --count 20 --output data/test_cvs
```

### 5. Create Collection

```python
# create_collection.py
from src.vector_store import CVVectorStore
from src.embedding_service import LocalEmbeddingService

# Initialize services
embedding_service = LocalEmbeddingService()
vectorstore = CVVectorStore(use_local=False)

# Create optimized collection
vectorstore.create_collection_optimized(
    vector_size=embedding_service.get_dimension(),  # 384 for MiniLM
    shard_number=1,      # Single shard for laptop
    replication_factor=1,# No replication for testing
    on_disk=False        # Keep in memory for speed
)

print("✅ Collection created!")
```

```bash
python create_collection.py
```

### 6. Enqueue Files

```bash
# Enqueue all CV files for processing
python enqueue_files.py data/test_cvs

# Output:
# Found 20 CV files
# Added: 20
# Skipped: 0 (already processed or in queue)
#
# Queue Statistics:
#   Pending:    20
#   Processing: 0
#   Completed:  0
#   Failed:     0
```

### 7. Start Worker

```bash
# In a new terminal
python worker.py

# You'll see:
# [2026-01-06 10:00:00] INFO - Worker worker-abc123 started
# [2026-01-06 10:00:05] INFO - Processing job xyz: data/test_cvs/john_doe.pdf
# [2026-01-06 10:00:12] INFO - Completed job xyz: 15 vectors created
```

### 8. Monitor Progress

```bash
# In another terminal
python monitor_progress.py --watch

# Continuous monitoring display:
# ======================================================================
#                CV INTELLIGENCE - INGESTION MONITOR
# ======================================================================
# Time: 2026-01-06 10:05:30
#
# 📋 JOB QUEUE STATUS
# ----------------------------------------------------------------------
#   Pending:           15  (waiting to be processed)
#   Processing:         1  (currently being worked on)
#   Completed:          4  (successfully processed)
#   Failed:             0  (processing errors)
#
#   Total:             20
#   Progress:       20.0%
```

### 9. Query the System

```bash
# Start the Streamlit app
streamlit run app.py

# Visit http://localhost:8501
# Try queries like:
# - "Python developer with 5 years experience"
# - "Machine learning engineer with NLP skills"
# - "DevOps engineer with Kubernetes"
```

**🎉 Congratulations! You've successfully set up and tested the system.**

---

## Understanding the Architecture

### Current (Prototype) Architecture

```
User → app.py → Qdrant (:memory:)
                   ↑
          OpenAI Embeddings
```

**Limitations:**
- All files loaded into memory
- Sequential processing
- No job queue
- Lost data on restart

### New (Scalable) Architecture

```
┌─────────────────┐
│  CV Files       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Redis Queue    │ ← enqueue_files.py adds jobs
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌────────┐ ┌────────┐ ... (N workers)
│Worker 1│ │Worker 2│
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
         ▼
┌─────────────────┐
│  Qdrant Vector  │ ← Persistent storage
│  Store (Docker) │
└─────────────────┘
         ▲
         │
    ┌────┴────┐
    │ app.py  │ ← Query interface
    └─────────┘
```

**Benefits:**
- Memory-efficient (one file at a time)
- Parallel processing (multiple workers)
- Resume capability (queue persists)
- Production-ready

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| CPU | 4 cores | 8 cores+ |
| Disk | 20GB free | 100GB+ SSD |
| Python | 3.9+ | 3.10+ |

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone <your-repo-url>
cd cv-intelligence
```

#### 2. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import sentence_transformers; import redis; import qdrant_client; print('✅ All dependencies installed')"
```

#### 4. Install Docker

If you don't have Docker:

**Mac:**
```bash
# Install Docker Desktop
brew install --cask docker
```

**Linux (Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER  # Add user to docker group
# Log out and back in
```

**Windows:**
- Download Docker Desktop from docker.com
- Install and restart

#### 5. Verify Docker

```bash
docker --version
docker-compose --version

# Should output version numbers
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Key Configuration Options

#### For Laptop Testing (Default)

```bash
# Use CPU embeddings (no API cost)
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

# Local services
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
USE_LOCAL_QDRANT=false  # Use Docker server, not in-memory

# Processing settings
CHUNK_SIZE=500
CHUNK_OVERLAP=100
WORKER_BATCH_SIZE=10
```

#### For Production (GPU Available)

```bash
# Use GPU embeddings
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5  # Better quality
EMBEDDING_DEVICE=cuda  # Use GPU
EMBEDDING_BATCH_SIZE=64

# Same services
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379

# Increase batch size
WORKER_BATCH_SIZE=50
```

#### Alternative: OpenAI Embeddings (No GPU)

```bash
# Use OpenAI API
USE_LOCAL_EMBEDDINGS=false
OPENAI_API_KEY=sk-your-actual-api-key-here
EMBEDDING_MODEL=text-embedding-3-small

# Cost estimate for 90GB:
# ~11 billion tokens × $0.00002 = ~$220 one-time
```

---

## Running the System

### Workflow Overview

```
1. Start services     (docker-compose up)
2. Create collection  (python create_collection.py)
3. Enqueue files      (python enqueue_files.py data/cvs)
4. Start worker(s)    (python worker.py)
5. Monitor progress   (python monitor_progress.py --watch)
6. Query system       (streamlit run app.py)
```

### Detailed Steps

#### Step 1: Start Infrastructure

```bash
# Start Redis + Qdrant
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove data (CAUTION!)
docker-compose down -v
```

#### Step 2: Create Vector Collection

Create `create_collection.py`:

```python
#!/usr/bin/env python3
"""Create optimized Qdrant collection."""

from src.vector_store import CVVectorStore
from src.embedding_service import LocalEmbeddingService
from src.config import USE_LOCAL_EMBEDDINGS

# Initialize embedding service
if USE_LOCAL_EMBEDDINGS:
    embedding_service = LocalEmbeddingService()
    vector_dim = embedding_service.get_dimension()
    print(f"Using local embeddings (dim={vector_dim})")
else:
    # For OpenAI, use known dimensions
    vector_dim = 1536  # text-embedding-3-small
    print(f"Using OpenAI embeddings (dim={vector_dim})")

# Create vector store
vectorstore = CVVectorStore(use_local=False)

# Create collection
vectorstore.create_collection_optimized(
    vector_size=vector_dim,
    shard_number=1,       # 1 for laptop, 3 for production
    replication_factor=1, # 1 for testing, 2 for production
    on_disk=False         # False for laptop, True for 90GB
)

print("✅ Collection created successfully!")
print(f"   Visit http://localhost:6333/dashboard to see it")
```

```bash
python create_collection.py
```

#### Step 3: Enqueue Files

```bash
# Enqueue from default directory
python enqueue_files.py

# Enqueue from custom directory
python enqueue_files.py /path/to/cvs

# Enqueue with clearing previous jobs
python enqueue_files.py --clear data/cvs

# Check queue without enqueueing
python -c "from src.job_queue import JobQueue; q = JobQueue(); print(q.get_stats())"
```

#### Step 4: Start Worker(s)

```bash
# Single worker
python worker.py

# Multiple workers (in separate terminals)
python worker.py  # Terminal 1
python worker.py  # Terminal 2
python worker.py  # Terminal 3

# Or use screen/tmux
screen -S worker1 -dm python worker.py
screen -S worker2 -dm python worker.py
```

#### Step 5: Monitor Progress

```bash
# One-time check
python monitor_progress.py

# Continuous monitoring (refreshes every 5 seconds)
python monitor_progress.py --watch

# Custom refresh interval
python monitor_progress.py --watch --interval 10
```

#### Step 6: Query System

```bash
# Start Streamlit app
streamlit run app.py

# Visit http://localhost:8501
```

### Performance Expectations

| Setup | Files/Hour | Time for 90GB | Cost |
|-------|-----------|---------------|------|
| Laptop CPU (MiniLM) | ~100-200 | Days | $0 |
| GPU (BGE-large) | ~1000-2000 | Hours | $0 |
| OpenAI API | ~500-1000 | Hours | ~$220 |

---

## Scaling to Production

### When to Scale

Scale from laptop testing to production when:
1. ✅ Tested with 20-50 CVs successfully
2. ✅ Verified queries return good results
3. ✅ Ready to process full 90GB dataset

### Scaling Checklist

#### 1. Hardware Upgrade

```bash
# Check current resources
df -h  # Disk space
free -h  # RAM
lscpu  # CPU cores
nvidia-smi  # GPU (if available)
```

**Recommended for 90GB:**
- RAM: 16GB+ (32GB ideal)
- Disk: 500GB SSD
- CPU: 8+ cores
- GPU: 8GB+ VRAM (optional but recommended)

#### 2. Update Configuration

Edit `.env`:

```bash
# Switch to GPU embeddings
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=64

# Increase worker batch size
WORKER_BATCH_SIZE=50

# Enable on-disk storage
# (Update create_collection.py: on_disk=True)
```

#### 3. Create Production Collection

```python
# create_collection.py (production version)
vectorstore.create_collection_optimized(
    vector_size=1024,      # BGE-large dimension
    shard_number=3,        # Distribute across nodes (if using cluster)
    replication_factor=2,  # High availability
    on_disk=True           # Store on disk for large datasets
)
```

#### 4. Run Multiple Workers

```bash
# Start 4 workers
for i in {1..4}; do
    screen -S worker$i -dm python worker.py
done

# Monitor workers
screen -ls
```

#### 5. Monitor Production

```bash
# Watch progress
python monitor_progress.py --watch --interval 30

# Check Qdrant stats
curl http://localhost:6333/collections/cv_documents

# Check Redis stats
redis-cli
> INFO stats
> LLEN cv:jobs:pending
```

### Cluster Deployment (Optional)

For very large scale (500K+ documents), deploy Qdrant cluster:

See `docs/SCALING_ARCHITECTURE.md` for:
- Multi-node Qdrant setup
- Load balancing
- Backup/restore
- High availability

---

## Troubleshooting

### Common Issues

#### 1. "Connection refused" (Redis/Qdrant)

**Problem**: Services not running

**Solution**:
```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart

# Check logs
docker-compose logs redis
docker-compose logs qdrant
```

#### 2. Worker crashes with OOM (Out of Memory)

**Problem**: Not enough RAM

**Solution**:
```bash
# Reduce batch size in .env
WORKER_BATCH_SIZE=5
EMBEDDING_BATCH_SIZE=16

# Use only 1 worker
# Stop other workers
```

#### 3. Slow processing (CPU embeddings)

**Problem**: CPU too slow

**Solutions**:
- Use smaller model: `all-MiniLM-L6-v2` (already default)
- Reduce batch size
- Or switch to OpenAI API (faster, but costs money)

#### 4. "Collection not found"

**Problem**: Collection not created

**Solution**:
```bash
python create_collection.py
```

#### 5. Jobs stuck in "processing"

**Problem**: Worker crashed mid-job

**Solution**:
```python
# Retry failed jobs
from src.job_queue import JobQueue
queue = JobQueue()
print(queue.retry_failed_jobs())
```

### Getting Help

1. Check logs:
   ```bash
   docker-compose logs
   ```

2. Check queue status:
   ```bash
   python monitor_progress.py
   ```

3. View failed jobs:
   ```python
   from src.job_queue import JobQueue
   queue = JobQueue()
   failed = queue.get_failed_jobs()
   for job in failed:
       print(f"{job['file_path']}: {job['error']}")
   ```

---

## Next Steps

1. **Test with sample data** (20-50 CVs)
2. **Validate query quality** (try various searches)
3. **Scale gradually** (100 → 1K → 10K → 90GB)
4. **Monitor performance** (track metrics)
5. **Optimize as needed** (tune batch sizes, add workers)

For architecture details, see:
- `docs/SCALING_ARCHITECTURE.md` - Scaling to 90GB+
- `docs/advise.md` - Testing strategy and philosophy

---

**Questions or issues?** Check the troubleshooting section or review the architecture documentation.
