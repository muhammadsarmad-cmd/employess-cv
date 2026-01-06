#!/usr/bin/env python3
"""Script to enqueue CV files for processing.

Usage:
    python enqueue_files.py [data_directory]
    python enqueue_files.py --help

Examples:
    # Enqueue files from default directory
    python enqueue_files.py

    # Enqueue from specific directory
    python enqueue_files.py /path/to/cvs

    # Enqueue from sample CVs
    python enqueue_files.py data/sample_cvs
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.job_queue import JobQueue
from src.document_processor import CVDocumentProcessor
from src.config import SAMPLE_CVS_DIR, REDIS_URL


def main():
    parser = argparse.ArgumentParser(
        description="Enqueue CV files for processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enqueue_files.py
  python enqueue_files.py /path/to/cvs
  python enqueue_files.py data/sample_cvs
        """
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(SAMPLE_CVS_DIR),
        help=f"Directory containing CV documents (default: {SAMPLE_CVS_DIR})"
    )
    parser.add_argument(
        "--redis-url",
        default=REDIS_URL,
        help=f"Redis URL (default: {REDIS_URL})"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all queues before enqueueing (CAUTION: deletes all job data)"
    )

    args = parser.parse_args()

    # Validate directory
    cv_dir = Path(args.directory)
    if not cv_dir.exists():
        print(f"Error: Directory not found: {cv_dir}")
        sys.exit(1)

    print("=" * 60)
    print("CV Intelligence - File Enqueuer")
    print("=" * 60)
    print(f"Data directory: {cv_dir}")
    print(f"Redis URL:      {args.redis_url}")
    print()

    # Initialize components
    try:
        queue = JobQueue(redis_url=args.redis_url)
        print("✓ Connected to Redis")
    except Exception as e:
        print(f"✗ Failed to connect to Redis: {e}")
        sys.exit(1)

    # Clear queues if requested
    if args.clear:
        print("\n⚠️  Clearing all queues...")
        queue.clear_all()
        print("✓ All queues cleared")

    # Count files
    print(f"\n📁 Scanning directory...")
    processor = CVDocumentProcessor(data_dir=cv_dir)
    file_count = processor.count_files()
    print(f"   Found {file_count} CV files")

    if file_count == 0:
        print("\n⚠️  No files found. Nothing to enqueue.")
        sys.exit(0)

    # Enqueue files
    print(f"\n🔄 Enqueueing files...")
    file_paths = list(processor.iter_files())
    stats = queue.enqueue_files(file_paths)

    print(f"\n✓ Enqueuing complete:")
    print(f"   Added:   {stats['added']}")
    print(f"   Skipped: {stats['skipped']} (already processed or in queue)")

    # Show queue stats
    queue_stats = queue.get_stats()
    print(f"\n📊 Queue Statistics:")
    print(f"   Pending:    {queue_stats['pending']}")
    print(f"   Processing: {queue_stats['processing']}")
    print(f"   Completed:  {queue_stats['completed']}")
    print(f"   Failed:     {queue_stats['failed']}")

    print(f"\n✅ Ready to process!")
    print(f"\n💡 Next steps:")
    print(f"   1. Start workers:  python worker.py")
    print(f"   2. Monitor progress: python monitor_progress.py --watch")
    print()


if __name__ == "__main__":
    main()
