#!/usr/bin/env python3
"""CLI for ingesting CV documents."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import CVDocumentProcessor
from src.vector_store import CVVectorStore
from src.config import SAMPLE_CVS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Ingest CV documents into the vector database"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(SAMPLE_CVS_DIR),
        help=f"Directory containing CV documents (default: {SAMPLE_CVS_DIR})"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap (default: 100)"
    )
    parser.add_argument(
        "--collection",
        default="cv_documents",
        help="Qdrant collection name (default: cv_documents)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata extraction"
    )

    args = parser.parse_args()

    # Validate directory
    cv_dir = Path(args.directory)
    if not cv_dir.exists():
        print(f"Error: Directory not found: {cv_dir}")
        sys.exit(1)

    print(f"[DIR] Processing CVs from: {cv_dir}")
    print(f"      Chunk size: {args.chunk_size}")
    print(f"      Chunk overlap: {args.chunk_overlap}")
    print()

    # Process documents
    processor = CVDocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    print("[LOAD] Loading and processing documents...")
    chunks = processor.process_directory(
        str(cv_dir),
        extract_metadata=not args.no_metadata
    )

    if not chunks:
        print("[ERROR] No documents found!")
        sys.exit(1)

    print(f"[OK] Processed {len(chunks)} chunks")
    print()

    # Create vector store
    print("[EMBED] Creating embeddings and storing in vector database...")
    store = CVVectorStore(collection_name=args.collection)
    store.create_collection(chunks)

    # Show stats
    info = store.get_collection_info()
    print()
    print("[DONE] Ingestion complete!")
    print(f"       Collection: {info.get('name')}")
    print(f"       Documents: {info.get('points_count')}")
    print()
    print("[RUN] Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
