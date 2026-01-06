#!/usr/bin/env python3
"""Create optimized Qdrant collection for CV documents.

Usage:
    python create_collection.py

This script:
1. Initializes the embedding service (detects CPU/GPU automatically)
2. Creates a Qdrant collection with appropriate settings
3. Optimizes for your hardware (laptop vs production)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store import CVVectorStore
from src.embedding_service import LocalEmbeddingService
from src.config import USE_LOCAL_EMBEDDINGS, LOCAL_EMBEDDING_MODEL, QDRANT_COLLECTION


def main():
    print("=" * 60)
    print("CV Intelligence - Collection Creator")
    print("=" * 60)
    print()

    # Determine embedding dimension
    if USE_LOCAL_EMBEDDINGS:
        print(f"📦 Loading embedding model: {LOCAL_EMBEDDING_MODEL}")
        try:
            embedding_service = LocalEmbeddingService()
            vector_dim = embedding_service.get_dimension()
            print(f"✓ Model loaded (dimension={vector_dim})")
            print(f"  Device: {embedding_service.device}")
        except Exception as e:
            print(f"✗ Failed to load embedding model: {e}")
            sys.exit(1)
    else:
        # OpenAI embeddings
        print("📦 Using OpenAI embeddings")
        vector_dim = 1536  # text-embedding-3-small
        print(f"✓ Dimension: {vector_dim}")

    print()

    # Create vector store
    print(f"🗄️  Connecting to Qdrant...")
    try:
        vectorstore = CVVectorStore(use_local=False)
        print(f"✓ Connected to Qdrant")
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        print(f"\n💡 Make sure Qdrant is running:")
        print(f"   docker-compose up -d")
        sys.exit(1)

    print()

    # Create collection
    print(f"📋 Creating collection '{QDRANT_COLLECTION}'...")

    # Auto-detect: use on-disk for large models, in-memory for small ones
    on_disk = vector_dim > 512  # True for BGE-large (1024), False for MiniLM (384)

    vectorstore.create_collection_optimized(
        vector_size=vector_dim,
        shard_number=1,       # Single shard for laptop testing
        replication_factor=1, # No replication for testing
        on_disk=on_disk       # Disk storage for large embeddings
    )

    print(f"✓ Collection created!")
    print()

    # Show collection info
    stats = vectorstore.get_collection_stats()
    print(f"📊 Collection Details:")
    print(f"   Name:       {stats.get('name')}")
    print(f"   Vectors:    {stats.get('points_count', 0)}")
    print(f"   Status:     {stats.get('status')}")
    print(f"   On Disk:    {'Yes' if on_disk else 'No (in memory)'}")
    print()

    print("✅ Collection ready!")
    print()
    print("💡 Next steps:")
    print("   1. Enqueue files:  python enqueue_files.py data/sample_cvs")
    print("   2. Start worker:   python worker.py")
    print("   3. Monitor:        python monitor_progress.py --watch")
    print()
    print(f"🌐 View in dashboard: http://localhost:6333/dashboard")
    print()


if __name__ == "__main__":
    main()
