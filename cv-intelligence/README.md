# CV Intelligence System

AI-powered recruitment assistant for searching, filtering, and comparing candidates from CV documents.

## Features

- **Document Processing**: Ingest PDFs, Word docs, and images (with OCR)
- **Semantic Search**: Find candidates using natural language queries
- **Metadata Extraction**: Automatically extract skills, experience, education
- **Smart Filtering**: Filter by experience years, specific skills (Docker, AI/ML, etc.)
- **Candidate Comparison**: Compare multiple candidates side-by-side
- **Best Match Ranking**: Find and rank best candidates for job requirements

## Quick Start

### 1. Install Dependencies

```bash
cd cv-intelligence
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Generate Sample CVs (for testing)

```bash
python generate_samples.py -n 30
```

### 4. Index CVs

```bash
python ingest.py data/sample_cvs
```

### 5. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage Examples

### Search Queries

```
"Python developer with AWS experience"
"Senior engineers with Docker and Kubernetes"
"Machine learning experts with 5+ years experience"
```

### Filtering

- Minimum experience years
- Docker experience required
- AI/ML experience required
- Cloud experience required

### Comparison

Enter comma-separated CV filenames to compare:
```
john_smith_cv.txt, jane_doe_cv.txt
```

## Project Structure

```
cv-intelligence/
├── app.py                    # Streamlit UI
├── ingest.py                 # CLI for document ingestion
├── generate_samples.py       # Sample CV generator
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── src/
│   ├── config.py             # Configuration
│   ├── document_processor.py # Document loading & chunking
│   ├── metadata_extractor.py # CV metadata extraction
│   ├── vector_store.py       # Qdrant vector database
│   └── query_engine.py       # Search & comparison logic
└── data/
    ├── sample_cvs/           # Sample CV documents
    └── vector_store/         # Local Qdrant storage
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `LLM_MODEL` | LLM for analysis | `gpt-4o` |
| `CHUNK_SIZE` | Text chunk size | `500` |
| `CHUNK_OVERLAP` | Chunk overlap | `100` |

## Scaling to 90GB

For production with 90GB of data:

1. **Use Docker Qdrant**:
```bash
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

2. **Update `.env`**:
```
USE_LOCAL_QDRANT=false
QDRANT_URL=http://localhost:6333
```

3. **Batch ingestion** for large datasets:
```bash
# Process in batches of 1000 documents
python ingest.py /path/to/cvs --batch-size 1000
```

## Supported File Types

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Text and scanned (OCR) |
| Word | `.docx`, `.doc` | Full text extraction |
| Images | `.png`, `.jpg`, `.jpeg` | OCR required |

## License

MIT
