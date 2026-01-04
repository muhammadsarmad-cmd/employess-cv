# CV Intelligence System - Technical Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Core Components](#core-components)
   - [config.py](#configpy)
   - [document_processor.py](#document_processorpy)
   - [metadata_extractor.py](#metadata_extractorpy)
   - [vector_store.py](#vector_storepy)
   - [query_engine.py](#query_enginepy)
   - [app.py](#apppy)
5. [Data Flow](#data-flow)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

The CV Intelligence System is an AI-powered recruitment tool that enables:

- **Semantic Search**: Find candidates using natural language queries
- **Metadata Extraction**: Automatically extract skills, experience, education from CVs
- **Candidate Comparison**: Compare multiple candidates side-by-side
- **Best Match Ranking**: Find and rank best candidates for job requirements

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector Database | Qdrant | Store and search CV embeddings |
| Embeddings | OpenAI text-embedding-3-small | Convert text to vectors |
| LLM | GPT-4o | Answer queries, compare candidates |
| Document Loading | LangChain | Process PDF, Word, Image files |
| Web Interface | Streamlit | Recruiter-facing UI |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CV INTELLIGENCE SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────────┐   │
│  │   INGESTION    │     │    STORAGE     │     │      QUERY         │   │
│  │    LAYER       │────▶│     LAYER      │────▶│      LAYER         │   │
│  └────────────────┘     └────────────────┘     └────────────────────┘   │
│         │                       │                       │                │
│  ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐        │
│  │ document_   │         │ vector_     │         │ query_      │        │
│  │ processor.py│         │ store.py    │         │ engine.py   │        │
│  │             │         │             │         │             │        │
│  │ - Load docs │         │ - Qdrant    │         │ - Search    │        │
│  │ - Extract   │         │ - Embeddings│         │ - Compare   │        │
│  │ - Chunk     │         │ - Index     │         │ - LLM calls │        │
│  └─────────────┘         └─────────────┘         └─────────────┘        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         PRESENTATION LAYER                        │   │
│  │                            app.py (Streamlit)                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
cv-intelligence/
│
├── app.py                      # Streamlit web interface
├── ingest.py                   # CLI tool for batch CV ingestion
├── test_system.py              # In-memory test script
├── generate_samples.py         # Generate sample CVs for testing
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── .env.example                # Template for .env file
├── README.md                   # Quick start guide
├── DOCUMENTATION.md            # This file
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── config.py               # Configuration and constants
│   ├── document_processor.py   # Document loading and chunking
│   ├── metadata_extractor.py   # Skill/experience extraction
│   ├── vector_store.py         # Qdrant vector database operations
│   └── query_engine.py         # Search and LLM query logic
│
└── data/                       # Data directory
    ├── sample_cvs/             # Sample CV files for testing
    └── vector_store/           # Qdrant local storage (if used)
```

---

## Core Components

### config.py

**Purpose**: Centralizes all configuration settings.

**Key Variables**:

| Variable | Type | Description |
|----------|------|-------------|
| `OPENAI_API_KEY` | str | API key for OpenAI services |
| `EMBEDDING_MODEL` | str | Model for text embeddings (default: text-embedding-3-small) |
| `LLM_MODEL` | str | Model for chat/analysis (default: gpt-4o) |
| `QDRANT_URL` | str | Qdrant server URL |
| `QDRANT_COLLECTION` | str | Collection name in Qdrant |
| `USE_LOCAL_QDRANT` | bool | Use in-memory (True) or Docker (False) |
| `CHUNK_SIZE` | int | Characters per text chunk (default: 500) |
| `CHUNK_OVERLAP` | int | Overlap between chunks (default: 100) |
| `SUPPORTED_EXTENSIONS` | dict | File types → loader mapping |
| `COMMON_SKILLS` | list | Skills to extract from CVs |

**Usage**:
```python
from src.config import OPENAI_API_KEY, CHUNK_SIZE, COMMON_SKILLS
```

---

### document_processor.py

**Purpose**: Load documents from various formats and split into chunks.

**Classes**:

#### `CVDocumentProcessor`

Main class for processing CV documents.

**Constructor**:
```python
CVDocumentProcessor(
    chunk_size: int = 500,      # Characters per chunk
    chunk_overlap: int = 100    # Overlap between chunks
)
```

**Methods**:

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `load_single_document` | `file_path: str` | `List[Document]` | Load one file based on extension |
| `load_directory` | `directory: str` | `List[Document]` | Load all supported files from folder |
| `process_documents` | `documents: List[Document]` | `List[Document]` | Extract metadata and split into chunks |
| `process_directory` | `directory: str` | `List[Document]` | Load and process entire directory |

**Supported File Types**:

| Extension | Loader | Notes |
|-----------|--------|-------|
| .pdf | PyPDFLoader | Text extraction from PDFs |
| .docx, .doc | Docx2txtLoader | Microsoft Word files |
| .txt | TextLoader | Plain text files |
| .png, .jpg, .jpeg | UnstructuredImageLoader | Requires Tesseract OCR |

**Example**:
```python
from src.document_processor import CVDocumentProcessor

processor = CVDocumentProcessor(chunk_size=500, chunk_overlap=100)
chunks = processor.process_directory("/path/to/cvs")
# Returns: List of Document objects with metadata
```

---

### metadata_extractor.py

**Purpose**: Extract structured information from CV text.

**Functions**:

#### `extract_cv_metadata(text: str) -> Dict[str, Any]`

Main function that extracts all metadata.

**Returns**:
```python
{
    "skills": ["Python", "Docker", "AWS"],  # Matched skills
    "experience_years": 5,                   # Estimated years
    "education_level": "Masters",            # PhD/Masters/Bachelors
    "email": "john@email.com",               # Extracted email
    "has_docker": True,                      # Boolean flags
    "has_kubernetes": False,
    "has_ai_ml": True,
    "has_cloud": True
}
```

#### `extract_skills(text: str) -> List[str]`

Matches text against `COMMON_SKILLS` list using regex word boundaries.

**Algorithm**:
```
For each skill in COMMON_SKILLS:
    If regex \b{skill}\b matches text (case-insensitive):
        Add to found_skills
Return found_skills
```

#### `extract_experience_years(text: str) -> Optional[int]`

Extracts years of experience using regex patterns.

**Patterns Matched**:
- "X years of experience"
- "X+ years experience"
- "experience: X years"
- Date ranges (2019-2023) → calculates duration

#### `extract_education_level(text: str) -> Optional[str]`

Identifies highest education level.

**Detection Order** (highest first):
1. PhD/Doctorate
2. Masters/MBA
3. Bachelors/BS/BTech
4. Associate/Diploma

---

### vector_store.py

**Purpose**: Manage Qdrant vector database operations.

**Classes**:

#### `CVVectorStore`

Handles all vector database operations.

**Constructor**:
```python
CVVectorStore(
    collection_name: str = "cv_documents",
    use_local: bool = True,           # In-memory vs Docker
    qdrant_url: str = "http://localhost:6333"
)
```

**Methods**:

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `create_collection` | `documents: List[Document]` | `QdrantVectorStore` | Create new collection with documents |
| `load_existing` | - | `QdrantVectorStore` | Connect to existing collection |
| `add_documents` | `documents: List[Document]` | - | Add documents to collection |
| `search` | `query: str, k: int, filter_dict: dict` | `List[Document]` | Semantic search |
| `search_with_scores` | `query: str, k: int, filter_dict: dict` | `List[tuple]` | Search with similarity scores |
| `get_collection_info` | - | `dict` | Get collection statistics |

**How Vector Search Works**:

```
1. Query text → OpenAI Embeddings → Vector [0.1, 0.3, ..., 0.8] (1536 dims)
2. Qdrant finds nearest vectors using cosine similarity
3. Returns top-k documents sorted by similarity score
```

**Filter Dictionary Format**:
```python
{
    "experience_years_min": 5,    # >= 5 years
    "experience_years_max": 10,   # <= 10 years
    "has_docker": True,           # Must have Docker
    "has_ai_ml": True             # Must have AI/ML
}
```

**Example**:
```python
from src.vector_store import CVVectorStore

store = CVVectorStore()
store.create_collection(chunks)

# Search with filter
results = store.search(
    query="Python developer",
    k=10,
    filter_dict={"experience_years_min": 3}
)
```

---

### query_engine.py

**Purpose**: High-level query interface combining search and LLM analysis.

**Classes**:

#### `CVQueryEngine`

Main interface for querying CVs.

**Constructor**:
```python
CVQueryEngine(vector_store: CVVectorStore = None)
```

**Methods**:

#### `search_candidates(query, k, filters) -> List[Dict]`

Find candidates matching query.

**Parameters**:
- `query`: Natural language query
- `k`: Number of results (default: 10)
- `filters`: Optional metadata filters

**Returns**:
```python
[
    {
        "source_file": "john_doe.pdf",
        "relevance_score": 0.85,
        "skills": ["Python", "Docker"],
        "experience_years": 5,
        "education_level": "Masters",
        "has_docker": True,
        "has_ai_ml": False,
        "excerpt": "First 300 chars of CV..."
    },
    ...
]
```

#### `find_best_candidates(job_requirement, k, filters) -> Dict`

Find and rank best candidates for a role.

**Parameters**:
- `job_requirement`: Job description text
- `k`: Number of candidates
- `filters`: Optional filters

**Returns**:
```python
{
    "candidates": [...],     # List of candidate dicts
    "analysis": "..."        # LLM-generated ranking and analysis
}
```

**LLM Prompt Used**:
```
You are a recruitment assistant. Based on the job requirement and
candidate profiles, provide a ranking and brief explanation for each.

Job Requirement: {requirement}
Candidates: {candidates}

Provide:
1. Ranked list with justification
2. Key strengths of top candidate
3. Any gaps to consider
```

#### `compare_candidates(candidate_files, comparison_criteria) -> Dict`

Compare specific candidates side-by-side.

**Parameters**:
- `candidate_files`: List of CV filenames
- `comparison_criteria`: Optional focus areas

**Returns**:
```python
{
    "candidates": [...],     # Full candidate data
    "comparison": "..."      # LLM-generated comparison
}
```

**LLM Prompt Used**:
```
You are a recruitment assistant comparing candidates.
{criteria}

Candidates: {candidates}

Provide:
1. Side-by-side comparison table
2. Strengths of each
3. Weaknesses/gaps
4. Recommendation with reasoning
```

#### `answer_query(query) -> Dict`

General query answering.

**Parameters**:
- `query`: Any recruitment query

**Returns**:
```python
{
    "answer": "LLM-generated answer...",
    "candidates": [...]  # Supporting candidates
}
```

---

### app.py

**Purpose**: Streamlit web interface for recruiters.

**Components**:

#### Sidebar
- Database status indicator
- "Index CVs" button
- CV directory input

#### Tab 1: Search
- Query text input
- Filter options (min experience, Docker, AI/ML)
- Results display with AI answer

#### Tab 2: Find Best
- Job requirements text area
- Number of candidates slider
- Ranked results with analysis

#### Tab 3: Compare
- Candidate file input (comma-separated)
- Comparison focus input
- Side-by-side comparison display

**Session State**:
```python
st.session_state.query_engine    # CVQueryEngine instance
st.session_state.db_loaded       # Boolean: database ready
```

**Flow**:
```
1. User clicks "Index CVs"
   → CVDocumentProcessor.process_directory()
   → CVVectorStore.create_collection()
   → CVQueryEngine initialized

2. User enters query in Search tab
   → query_engine.answer_query()
   → Display results

3. User compares candidates
   → query_engine.compare_candidates()
   → Display comparison
```

---

## Data Flow

### Indexing Flow (One-time)

```
CV Files (.pdf, .docx, .txt)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: LOAD DOCUMENTS                               │
│ document_processor.load_single_document()            │
│                                                      │
│ Input:  /cvs/john_doe.pdf                           │
│ Output: Document(                                    │
│           page_content="John Doe\nSoftware...",     │
│           metadata={"source": "john_doe.pdf"}       │
│         )                                            │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: EXTRACT METADATA                             │
│ metadata_extractor.extract_cv_metadata()             │
│                                                      │
│ Input:  "John Doe...5 years Python Docker AWS..."   │
│ Output: {                                            │
│           "skills": ["Python", "Docker", "AWS"],    │
│           "experience_years": 5,                     │
│           "has_docker": True                         │
│         }                                            │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: CHUNK TEXT                                   │
│ RecursiveCharacterTextSplitter.split_documents()     │
│                                                      │
│ Input:  1 document (2000 chars)                     │
│ Output: 4 chunks (500 chars each, 100 overlap)      │
│                                                      │
│ Chunk 1: chars 0-500                                │
│ Chunk 2: chars 400-900    ← 100 char overlap        │
│ Chunk 3: chars 800-1300                             │
│ Chunk 4: chars 1200-1700                            │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: CREATE EMBEDDINGS                            │
│ OpenAIEmbeddings.embed_documents()                   │
│                                                      │
│ Input:  "John Doe Software Engineer Python..."      │
│ Output: [0.023, -0.041, 0.089, ..., 0.012]         │
│         (1536-dimensional vector)                    │
│                                                      │
│ Cost: ~$0.0001 per 1000 tokens                      │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: STORE IN QDRANT                              │
│ QdrantVectorStore.add_documents()                    │
│                                                      │
│ Stored per chunk:                                    │
│ - Vector: [0.023, -0.041, ...]                      │
│ - Payload: {                                         │
│     "page_content": "John Doe...",                  │
│     "source_file": "john_doe.pdf",                  │
│     "skills": ["Python", "Docker"],                 │
│     "experience_years": 5,                           │
│     "has_docker": true                               │
│   }                                                  │
└─────────────────────────────────────────────────────┘
```

### Query Flow (Every search)

```
User Query: "Python developer with Docker experience"
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: EMBED QUERY                                  │
│ OpenAIEmbeddings.embed_query()                       │
│                                                      │
│ Input:  "Python developer with Docker experience"   │
│ Output: [0.045, -0.023, 0.078, ..., 0.034]         │
│         (1536-dimensional vector)                    │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: VECTOR SIMILARITY SEARCH                     │
│ Qdrant.search()                                      │
│                                                      │
│ Algorithm: Cosine Similarity                         │
│ similarity = dot(query_vec, doc_vec) /              │
│              (norm(query_vec) * norm(doc_vec))      │
│                                                      │
│ Returns: Top 10 most similar chunks                  │
│          with similarity scores                      │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: DEDUPLICATE BY SOURCE FILE                   │
│ query_engine.search_candidates()                     │
│                                                      │
│ Input:  10 chunks (may have duplicates)             │
│ Output: 5 unique candidates                          │
│         (one per source file)                        │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: LLM ANALYSIS                                 │
│ ChatOpenAI.invoke()                                  │
│                                                      │
│ Input:  Query + Candidate summaries                 │
│ Output: Natural language answer                      │
│                                                      │
│ Example:                                             │
│ "Based on your query, I recommend:                  │
│  1. John Doe - 5 years Python, Docker expert        │
│  2. Jane Smith - 7 years Python, AWS certified"    │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: RETURN RESULTS                               │
│                                                      │
│ {                                                    │
│   "answer": "Based on your query...",               │
│   "candidates": [                                    │
│     {"source_file": "john_doe.pdf", ...},           │
│     {"source_file": "jane_smith.pdf", ...}          │
│   ]                                                  │
│ }                                                    │
└─────────────────────────────────────────────────────┘
```

---

## API Reference

### Quick Reference

```python
# Initialize components
from src.document_processor import CVDocumentProcessor
from src.vector_store import CVVectorStore
from src.query_engine import CVQueryEngine

# Process CVs
processor = CVDocumentProcessor()
chunks = processor.process_directory("/path/to/cvs")

# Create vector store
store = CVVectorStore()
store.create_collection(chunks)

# Query
engine = CVQueryEngine(store)
result = engine.answer_query("Find Python developers")
print(result["answer"])
print(result["candidates"])
```

### Full API

```python
# ============================================
# CVDocumentProcessor
# ============================================
processor = CVDocumentProcessor(chunk_size=500, chunk_overlap=100)

# Load single file
docs = processor.load_single_document("cv.pdf")

# Load directory
docs = processor.load_directory("/cvs", show_progress=True)

# Process with metadata extraction
chunks = processor.process_documents(docs, extract_metadata=True)

# All-in-one
chunks = processor.process_directory("/cvs")

# ============================================
# CVVectorStore
# ============================================
store = CVVectorStore(
    collection_name="cv_documents",
    use_local=True,
    qdrant_url="http://localhost:6333"
)

# Create collection
store.create_collection(chunks)

# Load existing
store.load_existing()

# Add more documents
store.add_documents(new_chunks)

# Search
results = store.search("query", k=10, filter_dict={"has_docker": True})

# Search with scores
results = store.search_with_scores("query", k=10)
for doc, score in results:
    print(f"{doc.metadata['source_file']}: {score}")

# Get info
info = store.get_collection_info()
print(f"Total documents: {info['points_count']}")

# ============================================
# CVQueryEngine
# ============================================
engine = CVQueryEngine(store)

# Search candidates
candidates = engine.search_candidates(
    query="Docker expert",
    k=10,
    filters={"experience_years_min": 5}
)

# Find best for role
result = engine.find_best_candidates(
    job_requirement="Senior DevOps Engineer",
    k=5
)
print(result["analysis"])

# Compare candidates
result = engine.compare_candidates(
    candidate_files=["john.pdf", "jane.pdf"],
    comparison_criteria="Docker and Kubernetes skills"
)
print(result["comparison"])

# General query
result = engine.answer_query("Who has AI experience?")
print(result["answer"])
```

---

## Configuration

### Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional - Defaults shown
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=cv_documents
USE_LOCAL_QDRANT=true
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### Switching to Docker Qdrant

For production with persistence:

```bash
# 1. Start Qdrant
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# 2. Update .env
USE_LOCAL_QDRANT=false
QDRANT_URL=http://localhost:6333

# 3. Re-index CVs
python ingest.py /path/to/cvs
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No documents found" | Wrong directory or no supported files | Check path, ensure .pdf/.docx/.txt files exist |
| "OPENAI_API_KEY not set" | Missing .env file | Copy .env.example to .env, add key |
| "Storage folder already accessed" | File lock on Windows | Delete data/vector_store folder, or use Docker |
| "No candidates found" | Filters too strict | Remove filters, use semantic search only |
| Empty search results | Collection not created | Run "Index CVs" button first |
| Slow embedding | Many documents | Normal - ~1 min per 100 docs |

### Debug Mode

Run test script to verify system:

```bash
python test_system.py
```

Expected output:
```
[1/3] Loading and processing CVs...
Loaded 58 documents from 58 files
Created 177 chunks from 58 documents

[2/3] Creating embeddings...
Created collection 'cv_documents' with 177 documents

[3/3] Running test queries...
Raw search returned 5 results
Query: Best candidates for Docker deployment
Answer: Based on your query...
```

---

## Cost Estimation

### OpenAI API Costs

| Operation | Model | Cost |
|-----------|-------|------|
| Embedding 1000 CVs | text-embedding-3-small | ~$0.02 |
| Embedding 1000 CVs | text-embedding-3-large | ~$0.13 |
| 1 Query | GPT-4o | ~$0.01-0.03 |
| 1 Query | GPT-4o-mini | ~$0.001 |

### Estimated Monthly Cost

| Usage | Embeddings | Queries | Total |
|-------|------------|---------|-------|
| 1000 CVs, 100 queries/day | $0.02 (one-time) | $30-90 | ~$30-90/month |
| 10000 CVs, 500 queries/day | $0.20 (one-time) | $150-450 | ~$150-450/month |

---

## Glossary

| Term | Definition |
|------|------------|
| **Embedding** | Vector representation of text (1536 numbers) |
| **Vector Database** | Database optimized for similarity search |
| **Chunk** | Small piece of text (500 chars) for indexing |
| **Cosine Similarity** | Measure of angle between vectors (0-1) |
| **RAG** | Retrieval-Augmented Generation - search + LLM |
| **Semantic Search** | Search by meaning, not just keywords |
| **Payload** | Metadata stored with each vector |
| **Collection** | Group of vectors (like a database table) |
