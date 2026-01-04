"""Vector store operations with Qdrant."""

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition,
    MatchValue, Range
)

from .config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, QDRANT_URL,
    QDRANT_COLLECTION, USE_LOCAL_QDRANT
)


class CVVectorStore:
    """Vector store for CV documents."""

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        use_local: bool = USE_LOCAL_QDRANT,
        qdrant_url: str = QDRANT_URL
    ):
        self.collection_name = collection_name
        self.use_local = use_local
        self.qdrant_url = qdrant_url

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )

        # Initialize Qdrant client (shared instance)
        if use_local:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url)

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
