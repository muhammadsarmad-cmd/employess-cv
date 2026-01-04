#!/usr/bin/env python3
"""Quick test script - runs everything in memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import CVDocumentProcessor
from src.query_engine import CVQueryEngine
from src.vector_store import CVVectorStore
from src.config import SAMPLE_CVS_DIR


def main():
    print("=" * 60)
    print("CV Intelligence System - Quick Test")
    print("=" * 60)

    # 1. Process documents
    print("\n[1/3] Loading and processing CVs...")
    processor = CVDocumentProcessor()
    chunks = processor.process_directory(str(SAMPLE_CVS_DIR))

    if not chunks:
        print("No documents found! Run: python generate_samples.py")
        return

    # Show sample metadata
    print(f"\nSample chunk metadata:")
    print(f"  Source: {chunks[0].metadata.get('source_file')}")
    print(f"  Skills: {chunks[0].metadata.get('skills', [])[:5]}")
    print(f"  Experience: {chunks[0].metadata.get('experience_years')}")
    print(f"  Has Docker: {chunks[0].metadata.get('has_docker')}")

    # 2. Create vector store (in memory)
    print("\n[2/3] Creating embeddings (this may take a minute)...")
    store = CVVectorStore()
    store.create_collection(chunks)

    # Verify collection
    info = store.get_collection_info()
    print(f"Collection info: {info}")

    # 3. Test raw search first
    print("\n[DEBUG] Testing raw vector search...")
    raw_results = store.search("Docker Kubernetes DevOps", k=5)
    print(f"Raw search returned {len(raw_results)} results")
    if raw_results:
        for i, doc in enumerate(raw_results[:3]):
            print(f"  {i+1}. {doc.metadata.get('source_file')} - {doc.page_content[:100]}...")

    # Test with scores
    print("\n[DEBUG] Testing search with scores...")
    scored_results = store.search_with_scores("Docker Kubernetes DevOps", k=5)
    print(f"Scored search returned {len(scored_results)} results")
    if scored_results:
        for doc, score in scored_results[:3]:
            print(f"  Score: {score:.4f} - {doc.metadata.get('source_file')}")

    # 4. Test query engine
    print("\n[3/3] Testing query engine...")
    engine = CVQueryEngine(store)

    # Test search_candidates directly
    print("\n[DEBUG] Testing search_candidates...")
    candidates = engine.search_candidates("Docker deployment", k=5)
    print(f"search_candidates returned {len(candidates)} results")
    for c in candidates[:3]:
        print(f"  - {c['source_file']}: score={c['relevance_score']}, skills={c['skills'][:3]}")

    # Test full query
    test_queries = [
        "Best candidates for Docker deployment",
        "Candidates with experience in AI and machine learning",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        result = engine.answer_query(query)
        print(f"\nAnswer:\n{result['answer'][:500]}...")

        print(f"\nTop Candidates:")
        for i, c in enumerate(result.get('candidates', [])[:3]):
            print(f"  {i+1}. {c['source_file']} - {c.get('experience_years', 'N/A')} years")
            print(f"     Skills: {', '.join(c.get('skills', [])[:5])}")

    print("\n[DONE] Test complete!")


if __name__ == "__main__":
    main()
