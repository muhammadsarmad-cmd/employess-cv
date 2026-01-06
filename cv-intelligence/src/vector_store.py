"""Vector store operations with Qdrant."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
import logging

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition,
    MatchValue, Range, PointStruct, OptimizersConfigDiff,
    HnswConfigDiff
)

from .config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, QDRANT_URL,
    QDRANT_COLLECTION, QDRANT_API_KEY, USE_LOCAL_QDRANT
)

logger = logging.getLogger(__name__)


class CVVectorStore:
    """Vector store for CV documents."""

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        use_local: bool = USE_LOCAL_QDRANT,
        qdrant_url: str = QDRANT_URL,
        api_key: Optional[str] = QDRANT_API_KEY
    ):
        self.collection_name = collection_name
        self.use_local = use_local
        self.qdrant_url = qdrant_url

        # Initialize embeddings (for backward compatibility)
        self.embeddings = None
        if OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )

        # Initialize Qdrant client (shared instance)
        if use_local:
            self.client = QdrantClient(location=":memory:")
            logger.info("Using in-memory Qdrant (for testing only)")
        else:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=api_key if api_key else None,
                timeout=60  # Increase timeout for large operations
            )
            logger.info(f"Connected to Qdrant at {qdrant_url}")

        self.vectorstore: Optional[QdrantVectorStore] = None

    def create_collection(self, documents: List[Document]) -> QdrantVectorStore:
        """Create a new collection from documents."""
        # Delete existing collection if exists
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        # Get embedding dimension by embedding a test string
        test_embedding = self.embeddings.embed_query("test")
        vector_size = len(test_embedding)

        # Create collection manually with our client
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Create vector store using our existing client
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        # Add documents
        self.vectorstore.add_documents(documents)

        print(f"Created collection '{self.collection_name}' with {len(documents)} documents")
        return self.vectorstore

    def load_existing(self) -> Optional[QdrantVectorStore]:
        """Load an existing collection."""
        try:
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            return self.vectorstore
        except Exception as e:
            print(f"Could not load collection: {e}")
            return None

    def add_documents(self, documents: List[Document]):
        """Add documents to existing collection."""
        if self.vectorstore is None:
            self.load_existing()

        if self.vectorstore:
            self.vectorstore.add_documents(documents)
            print(f"Added {len(documents)} documents")

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        if self.vectorstore is None:
            self.load_existing()

        if self.vectorstore is None:
            raise ValueError("No vector store available")

        # Build filter
        qdrant_filter = self._build_filter(filter_dict) if filter_dict else None

        # Search
        if qdrant_filter:
            results = self.vectorstore.similarity_search(
                query, k=k, filter=qdrant_filter
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Search and return documents with similarity scores."""
        if self.vectorstore is None:
            self.load_existing()

        if self.vectorstore is None:
            raise ValueError("No vector store available")

        qdrant_filter = self._build_filter(filter_dict) if filter_dict else None

        if qdrant_filter:
            results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=qdrant_filter
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def _build_filter(self, filter_dict: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []

        for key, value in filter_dict.items():
            if key == "experience_years_min":
                conditions.append(
                    FieldCondition(
                        key="experience_years",
                        range=Range(gte=value)
                    )
                )
            elif key == "experience_years_max":
                conditions.append(
                    FieldCondition(
                        key="experience_years",
                        range=Range(lte=value)
                    )
                )
            elif isinstance(value, bool):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            elif isinstance(value, str):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )

        return Filter(must=conditions) if conditions else None

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}

    # === NEW METHODS FOR SCALED ARCHITECTURE ===

    def create_collection_optimized(
        self,
        vector_size: int,
        shard_number: int = 1,
        replication_factor: int = 1,
        on_disk: bool = False
    ):
        """Create collection optimized for large scale.

        Args:
            vector_size: Embedding dimension (384 for MiniLM, 1024 for BGE-large)
            shard_number: Number of shards (1 for laptop, 3+ for cluster)
            replication_factor: Replication factor (1 for laptop, 2+ for HA)
            on_disk: Store vectors on disk (True for large datasets)
        """
        # Delete if exists
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection '{self.collection_name}'")
        except Exception:
            pass

        # Optimize based on scale
        if on_disk or shard_number > 1:
            # Large scale configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    on_disk=on_disk
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
                    on_disk=on_disk    # Store HNSW index on disk
                )
            )
        else:
            # Laptop/testing configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    on_disk=False
                )
            )

        logger.info(f"Created optimized collection '{self.collection_name}' "
                   f"(dim={vector_size}, shards={shard_number}, on_disk={on_disk})")

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> List[str]:
        """Add documents with pre-computed embeddings in batches.

        Args:
            documents: List of LangChain documents
            embeddings: Pre-computed embedding vectors
            batch_size: Batch size for uploads

        Returns:
            List of vector IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) count mismatch")

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
                wait=True  # Wait for last batch
            )

        logger.info(f"Added {len(vector_ids)} documents to collection")
        return vector_ids

    def search_with_embedding(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search using pre-computed query embedding.

        Args:
            query_embedding: Pre-computed query embedding vector
            k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of search results with scores
        """
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
        """Get detailed collection statistics.

        Returns:
            Dictionary with collection stats
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status.value,
                "segments_count": len(info.segments) if info.segments else 0,
                "disk_data_size": getattr(info, 'disk_data_size', None),
                "ram_data_size": getattr(info, 'ram_data_size', None)
            }
        except Exception as e:
            return {"error": str(e)}


# Convenience functions
def create_cv_vectorstore(documents: List[Document]) -> CVVectorStore:
    """Create a new CV vector store."""
    store = CVVectorStore()
    store.create_collection(documents)
    return store


def load_cv_vectorstore() -> CVVectorStore:
    """Load existing CV vector store."""
    store = CVVectorStore()
    store.load_existing()
    return store
