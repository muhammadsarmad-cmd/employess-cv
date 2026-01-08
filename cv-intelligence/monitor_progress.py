#!/usr/bin/env python3
"""Monitor ingestion progress in real-time.

Usage:
    # One-time check
    python monitor_progress.py

    # Continuous monitoring (refreshes every 5 seconds)
    python monitor_progress.py --watch

    # Custom refresh interval
    python monitor_progress.py --watch --interval 10
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.job_queue import JobQueue
from src.vector_store import CVVectorStore
from src.config import REDIS_URL, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def format_number(n):
    """Format large numbers with commas."""
    return f"{n:,}"


def print_stats(queue: JobQueue, vectorstore: CVVectorStore):
    """Print current statistics."""
    # Get stats
    queue_stats = queue.get_stats()
    vector_stats = vectorstore.get_collection_stats()

    total = sum(queue_stats.values())
    completed = queue_stats['completed']
    progress_pct = (completed / total * 100) if total > 0 else 0

    # Print header
    print("=" * 70)
    print(" " * 15 + "CV INTELLIGENCE - INGESTION MONITOR")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Queue stats
    print("📋 JOB QUEUE STATUS")
    print("-" * 70)
    print(f"  Pending:    {format_number(queue_stats['pending']):>10}  (waiting to be processed)")
    print(f"  Processing: {format_number(queue_stats['processing']):>10}  (currently being worked on)")
    print(f"  Completed:  {format_number(queue_stats['completed']):>10}  (successfully processed)")
    print(f"  Failed:     {format_number(queue_stats['failed']):>10}  (processing errors)")
    print()
    print(f"  Total:      {format_number(total):>10}")
    print(f"  Progress:   {progress_pct:>9.1f}%")
    print()

    # Vector store stats
    print("🗄️  VECTOR STORE STATUS")
    print("-" * 70)
    if 'error' not in vector_stats:
        print(f"  Collection: {vector_stats.get('name', 'N/A')}")
        print(f"  Vectors:    {format_number(vector_stats.get('points_count', 0)):>10}")
        print(f"  Status:     {vector_stats.get('status', 'unknown'):>10}")

        if vector_stats.get('disk_data_size'):
            size_mb = vector_stats['disk_data_size'] / (1024 * 1024)
            print(f"  Disk Size:  {size_mb:>9.1f} MB")
    else:
        print(f"  Error: {vector_stats['error']}")
    print()

    # Failed jobs detail
    if queue_stats['failed'] > 0:
        print("⚠️  FAILED JOBS")
        print("-" * 70)
        failed_jobs = queue.get_failed_jobs()
        for job in failed_jobs[:5]:  # Show first 5
            print(f"  • {job.get('file_path', 'unknown')}")
            print(f"    Error: {job.get('error', 'unknown error')}")
            print(f"    Attempts: {job.get('attempts', 0)}")
        if len(failed_jobs) > 5:
            print(f"  ... and {len(failed_jobs) - 5} more")
        print()
        print("  💡 Tip: Use `python retry_failed.py` to retry failed jobs")
        print()

    # Status messages
    print("📌 STATUS")
    print("-" * 70)
    if queue_stats['pending'] > 0:
        print("  ⏳ Jobs in queue, processing in progress...")
    elif queue_stats['processing'] > 0:
        print("  ⚙️  Finishing up current jobs...")
    elif queue_stats['completed'] > 0 and queue_stats['pending'] == 0:
        print("  ✅ All jobs completed!")
    else:
        print("  💤 No jobs in queue")
    print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor CV ingestion progress"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor (refresh every interval)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Watch interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--redis-url",
        default=REDIS_URL,
        help=f"Redis URL (default: {REDIS_URL})"
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help=f"Qdrant URL (default: {QDRANT_URL})"
    )

    args = parser.parse_args()

    # Initialize components
    try:
        queue = JobQueue(redis_url=args.redis_url)
        vectorstore = CVVectorStore(
            qdrant_url=args.qdrant_url,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION,
            use_local=False,
            init_embeddings=False  # Stats only, no embeddings needed
        )
    except Exception as e:
        print(f"Error: Failed to connect to services: {e}")
        sys.exit(1)

    # Monitor
    if args.watch:
        print("Starting continuous monitoring (press Ctrl+C to stop)...\n")
        try:
            while True:
                clear_screen()
                print_stats(queue, vectorstore)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
    else:
        print_stats(queue, vectorstore)


if __name__ == "__main__":
    main()
