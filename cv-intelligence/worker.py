#!/usr/bin/env python3
"""Worker process entry point for CV Intelligence ingestion.

Usage:
    python worker.py

The worker will:
1. Connect to Redis and Qdrant
2. Pull jobs from the queue
3. Process CV files (load, chunk, embed, store)
4. Handle failures with automatic retry
5. Run until stopped (Ctrl+C)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.worker import main

if __name__ == "__main__":
    main()
