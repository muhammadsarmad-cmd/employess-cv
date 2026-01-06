"""Redis-based job queue for document processing."""

import json
import hashlib
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum
from datetime import datetime
import redis


class JobStatus(Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobQueue:
    """Manages document processing jobs with Redis.

    Provides reliable job queuing with:
    - Deduplication (skip already processed files)
    - Resume capability (handle worker failures)
    - Progress tracking
    - Retry logic for failed jobs
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize job queue.

        Args:
            redis_url: Redis connection URL
        """
        self.redis = redis.from_url(redis_url, decode_responses=True)

        # Queue keys
        self.queue_key = "cv:jobs:pending"
        self.processing_key = "cv:jobs:processing"
        self.completed_key = "cv:jobs:completed"
        self.failed_key = "cv:jobs:failed"
        self.metadata_prefix = "cv:job:"

    def _file_hash(self, file_path: Path) -> str:
        """Generate unique ID for a file based on its path.

        Args:
            file_path: Path to the file

        Returns:
            12-character hash ID
        """
        return hashlib.md5(str(file_path).encode()).hexdigest()[:12]

    def enqueue_files(self, file_paths: List[Path]) -> Dict[str, int]:
        """Add files to the processing queue.

        Args:
            file_paths: List of file paths to enqueue

        Returns:
            Dictionary with 'added' and 'skipped' counts
        """
        stats = {"added": 0, "skipped": 0}

        for file_path in file_paths:
            job_id = self._file_hash(file_path)

            # Skip if already processed or in queue
            if self._job_exists(job_id):
                stats["skipped"] += 1
                continue

            job_data = {
                "id": job_id,
                "file_path": str(file_path),
                "status": JobStatus.PENDING.value,
                "created_at": self._now(),
                "attempts": 0
            }

            # Store job metadata
            self.redis.set(
                f"{self.metadata_prefix}{job_id}",
                json.dumps(job_data),
                ex=86400 * 7  # Expire after 7 days
            )

            # Add to pending queue
            self.redis.lpush(self.queue_key, job_id)
            stats["added"] += 1

        return stats

    def get_next_job(self, worker_id: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Get next job from queue (blocking).

        Args:
            worker_id: Unique identifier for the worker
            timeout: Seconds to wait for a job (0 = wait forever)

        Returns:
            Job data dictionary or None if no jobs available
        """
        # Atomic move from pending to processing
        result = self.redis.brpoplpush(
            self.queue_key,
            self.processing_key,
            timeout=timeout
        )

        if not result:
            return None

        job_id = result
        job_data = self._get_job(job_id)

        if job_data:
            job_data["status"] = JobStatus.PROCESSING.value
            job_data["worker_id"] = worker_id
            job_data["started_at"] = self._now()
            job_data["attempts"] += 1
            self._save_job(job_data)

        return job_data

    def complete_job(self, job_id: str, vector_ids: List[str]):
        """Mark job as completed.

        Args:
            job_id: Job identifier
            vector_ids: List of vector IDs created from this job
        """
        job_data = self._get_job(job_id)
        if job_data:
            job_data["status"] = JobStatus.COMPLETED.value
            job_data["completed_at"] = self._now()
            job_data["vector_ids"] = vector_ids
            job_data["vector_count"] = len(vector_ids)
            self._save_job(job_data)

            # Move from processing to completed
            self.redis.lrem(self.processing_key, 1, job_id)
            self.redis.sadd(self.completed_key, job_id)

    def fail_job(self, job_id: str, error: str, retry: bool = True, max_attempts: int = 3):
        """Mark job as failed, optionally retry.

        Args:
            job_id: Job identifier
            error: Error message
            retry: Whether to retry the job
            max_attempts: Maximum retry attempts
        """
        job_data = self._get_job(job_id)
        if job_data:
            job_data["status"] = JobStatus.FAILED.value
            job_data["error"] = error
            job_data["failed_at"] = self._now()
            self._save_job(job_data)

            # Remove from processing
            self.redis.lrem(self.processing_key, 1, job_id)

            # Retry if attempts < max
            if retry and job_data["attempts"] < max_attempts:
                self.redis.lpush(self.queue_key, job_id)
            else:
                self.redis.sadd(self.failed_key, job_id)

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics.

        Returns:
            Dictionary with counts for each status
        """
        return {
            "pending": self.redis.llen(self.queue_key),
            "processing": self.redis.llen(self.processing_key),
            "completed": self.redis.scard(self.completed_key),
            "failed": self.redis.scard(self.failed_key)
        }

    def get_failed_jobs(self) -> List[Dict[str, Any]]:
        """Get all failed jobs with details.

        Returns:
            List of failed job data
        """
        failed_ids = self.redis.smembers(self.failed_key)
        return [self._get_job(job_id) for job_id in failed_ids if self._get_job(job_id)]

    def retry_failed_jobs(self) -> int:
        """Retry all failed jobs.

        Returns:
            Number of jobs re-queued
        """
        failed_ids = list(self.redis.smembers(self.failed_key))
        count = 0

        for job_id in failed_ids:
            job_data = self._get_job(job_id)
            if job_data:
                # Reset attempts and re-queue
                job_data["attempts"] = 0
                job_data["status"] = JobStatus.PENDING.value
                self._save_job(job_data)

                self.redis.srem(self.failed_key, job_id)
                self.redis.lpush(self.queue_key, job_id)
                count += 1

        return count

    def clear_all(self):
        """Clear all queues (CAUTION: deletes all job data)."""
        self.redis.delete(self.queue_key)
        self.redis.delete(self.processing_key)
        self.redis.delete(self.completed_key)
        self.redis.delete(self.failed_key)

        # Delete all job metadata
        for key in self.redis.scan_iter(f"{self.metadata_prefix}*"):
            self.redis.delete(key)

    def _job_exists(self, job_id: str) -> bool:
        """Check if job already exists in any queue."""
        return self.redis.exists(f"{self.metadata_prefix}{job_id}") > 0

    def _get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job data from Redis."""
        data = self.redis.get(f"{self.metadata_prefix}{job_id}")
        return json.loads(data) if data else None

    def _save_job(self, job_data: Dict[str, Any]):
        """Save job data to Redis."""
        self.redis.set(
            f"{self.metadata_prefix}{job_data['id']}",
            json.dumps(job_data),
            ex=86400 * 7  # Expire after 7 days
        )

    def _now(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()
