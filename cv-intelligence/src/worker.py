"""Worker process for batch document ingestion.

This worker:
1. Pulls jobs from Redis queue
2. Loads and chunks documents
3. Generates embeddings (local or API)
4. Stores vectors in Qdrant
5. Handles failures with retry logic
"""

import os
import signal
import logging
from pathlib import Path
from typing import List
import uuid

from .config import (
    REDIS_URL, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP, USE_LOCAL_EMBEDDINGS,
    OPENAI_API_KEY, EMBEDDING_MODEL
)
from .job_queue import JobQueue
from .document_processor import CVDocumentProcessor
from .vector_store import CVVectorStore
from .embedding_service import LocalEmbeddingService

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestionWorker:
    """Worker that processes documents from the job queue.

    Features:
    - Graceful shutdown (handles SIGTERM/SIGINT)
    - Automatic retry on failures
    - Progress logging
    - Memory-efficient (processes one file at a time)
    """

    def __init__(self):
        """Initialize the ingestion worker."""
        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self.running = True

        logger.info(f"Initializing worker {self.worker_id}")

        # Initialize components
        try:
            self.queue = JobQueue(redis_url=REDIS_URL)
            logger.info(f"Connected to Redis at {REDIS_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        try:
            self.vectorstore = CVVectorStore(
                collection_name=QDRANT_COLLECTION,
                qdrant_url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                use_local=False  # Always use server mode for workers
            )
            logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        # Initialize embedding service
        if USE_LOCAL_EMBEDDINGS:
            try:
                self.embedding_service = LocalEmbeddingService()
                self.embedding_dim = self.embedding_service.get_dimension()
                logger.info(f"Using local embeddings (dim={self.embedding_dim})")
            except Exception as e:
                logger.error(f"Failed to initialize local embeddings: {e}")
                raise
        else:
            # Use OpenAI embeddings
            from langchain_openai import OpenAIEmbeddings
            self.embedding_service = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            # Test to get dimension
            test_emb = self.embedding_service.embed_query("test")
            self.embedding_dim = len(test_emb)
            logger.info(f"Using OpenAI embeddings (dim={self.embedding_dim})")

        # Document processor
        self.processor = CVDocumentProcessor(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

        logger.info(f"Worker {self.worker_id} initialized successfully")

    def _shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Worker {self.worker_id} received shutdown signal")
        self.running = False

    def run(self):
        """Main worker loop.

        Continuously pulls jobs from the queue and processes them until shutdown.
        """
        logger.info(f"Worker {self.worker_id} started")
        logger.info("Waiting for jobs...")

        while self.running:
            try:
                # Get next job (blocks for up to 30 seconds)
                job = self.queue.get_next_job(self.worker_id, timeout=30)

                if job is None:
                    # No jobs available, continue waiting
                    continue

                logger.info(f"Processing job {job['id']}: {job['file_path']}")

                try:
                    # Process the file
                    vector_ids = self._process_file(job['file_path'])

                    # Mark as completed
                    self.queue.complete_job(job['id'], vector_ids)

                    logger.info(
                        f"Completed job {job['id']}: "
                        f"{len(vector_ids)} vectors created"
                    )

                except Exception as e:
                    # Mark as failed (will retry if attempts < 3)
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"Failed job {job['id']}: {error_msg}")
                    self.queue.fail_job(job['id'], error_msg, retry=True)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

        logger.info(f"Worker {self.worker_id} stopped")

    def _process_file(self, file_path: str) -> List[str]:
        """Process a single file and return vector IDs.

        Args:
            file_path: Path to the CV file

        Returns:
            List of vector IDs created

        Raises:
            Exception: If processing fails
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Process file: load, extract metadata, chunk
        chunks = self.processor.process_file(path, extract_metadata=True)

        if not chunks:
            raise ValueError(f"No chunks generated from file: {path}")

        logger.debug(f"Generated {len(chunks)} chunks from {path.name}")

        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]

        if USE_LOCAL_EMBEDDINGS:
            embeddings = self.embedding_service.embed_batch(texts)
        else:
            # OpenAI batch embedding
            embeddings = self.embedding_service.embed_documents(texts)

        logger.debug(f"Generated {len(embeddings)} embeddings")

        # Add to vector store
        vector_ids = self.vectorstore.add_documents_with_embeddings(
            chunks, embeddings, batch_size=100
        )

        return vector_ids


def main():
    """Entry point for worker process."""
    try:
        worker = IngestionWorker()
        worker.run()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed to start: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
