"""Configuration for CV Intelligence System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_CVS_DIR = DATA_DIR / "sample_cvs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# OpenAI (for LLM queries, optional for embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cv_documents")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional: for production security
USE_LOCAL_QDRANT = os.getenv("USE_LOCAL_QDRANT", "false").lower() == "true"

# Redis Configuration (for job queue)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Local Embeddings Configuration
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # or "cuda" for GPU

# Document Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Worker Configuration
WORKER_BATCH_SIZE = int(os.getenv("WORKER_BATCH_SIZE", "10"))  # Files per batch

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "word",
    ".doc": "word",
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".bmp": "image"
}

# Skills to extract (expandable)
COMMON_SKILLS = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin",
    # Frameworks
    "React", "Angular", "Vue", "Django", "Flask", "FastAPI", "Spring", "Node.js", "Express", ".NET",
    # Cloud & DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "CI/CD", "GitLab",
    # Data & AI
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Scikit-learn",
    "Pandas", "NumPy", "Spark", "Hadoop", "Airflow", "MLOps",
    # Databases
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra", "DynamoDB",
    # Other
    "Git", "Linux", "Agile", "Scrum", "REST API", "GraphQL", "Microservices", "System Design"
]
