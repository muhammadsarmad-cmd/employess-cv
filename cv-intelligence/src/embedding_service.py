"""Local embedding service using sentence-transformers.

Provides cost-free embeddings using open-source models:
- CPU: all-MiniLM-L6-v2 (384 dim, fast on laptop)
- GPU: BAAI/bge-large-en-v1.5 (1024 dim, production quality)
"""

import os
from typing import List, Optional
import logging

from .config import (
    LOCAL_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE
)

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Generate embeddings locally using sentence-transformers.

    Benefits:
    - Zero API costs
    - Data privacy (no external calls)
    - Fast batch processing
    - Scales from CPU (testing) to GPU (production)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None
    ):
        """Initialize the embedding service.

        Args:
            model_name: Model to use (defaults to config)
            batch_size: Batch size for encoding (defaults to config)
            device: Device to use: 'cpu', 'cuda', or None for auto-detect
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )

        self.model_name = model_name or LOCAL_EMBEDDING_MODEL
        self.batch_size = batch_size or EMBEDDING_BATCH_SIZE
        self.device = device or EMBEDDING_DEVICE

        # Auto-detect GPU if device not specified
        if self.device == "auto":
            self.device = "cuda" if self._gpu_available() else "cpu"

        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Device: {self.device}")

        # Load model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def _gpu_available(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return available
        except ImportError:
            return False

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Cosine similarity ready
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def get_dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Dimension of the embedding vectors
        """
        return self.dimension

    def benchmark(self, num_texts: int = 100, text_length: int = 200) -> dict:
        """Benchmark the embedding service.

        Args:
            num_texts: Number of texts to embed
            text_length: Length of each text (characters)

        Returns:
            Dictionary with benchmark statistics
        """
        import time

        # Generate dummy texts
        dummy_texts = [f"Sample text {i} " * (text_length // 20) for i in range(num_texts)]

        # Time the embedding
        start_time = time.time()
        embeddings = self.embed_batch(dummy_texts)
        elapsed = time.time() - start_time

        return {
            "model": self.model_name,
            "device": self.device,
            "num_texts": num_texts,
            "total_time": round(elapsed, 2),
            "time_per_text": round(elapsed / num_texts, 3),
            "texts_per_second": round(num_texts / elapsed, 1),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0
        }


# LangChain-compatible embeddings wrapper
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """LangChain-compatible wrapper for local embeddings."""

    def __init__(self, service: Optional[LocalEmbeddingService] = None):
        """Initialize with embedding service.

        Args:
            service: LocalEmbeddingService instance (creates new if None)
        """
        self.service = service or LocalEmbeddingService()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of document texts

        Returns:
            List of embeddings
        """
        return self.service.embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query text.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        return self.service.embed_query(text)


def create_embedding_service(
    use_local: bool = True,
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> LocalEmbeddingService:
    """Factory function to create an embedding service.

    Args:
        use_local: Whether to use local embeddings
        model_name: Model name (uses config default if None)
        device: Device to use ('cpu', 'cuda', or 'auto')

    Returns:
        LocalEmbeddingService instance

    Example:
        >>> # CPU testing (laptop)
        >>> service = create_embedding_service(
        ...     model_name="sentence-transformers/all-MiniLM-L6-v2",
        ...     device="cpu"
        ... )
        >>>
        >>> # GPU production (later)
        >>> service = create_embedding_service(
        ...     model_name="BAAI/bge-large-en-v1.5",
        ...     device="cuda"
        ... )
    """
    if not use_local:
        raise ValueError("Only local embeddings supported in this service")

    return LocalEmbeddingService(model_name=model_name, device=device)
