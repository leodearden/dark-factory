#!/usr/bin/env python3
"""Delete stale _test_mem0_qdrant_integration_* collections from Qdrant.

Intended to run as a cron job.  Always exits 0 (idempotent).
"""

from __future__ import annotations

import sys

PREFIX = '_test_mem0_qdrant_integration_'
QDRANT_URL = 'http://localhost:6333'


def main() -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print('qdrant-client not installed, skipping', file=sys.stderr)
        return

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        collections = client.get_collections().collections
    except Exception as exc:
        print(f'Qdrant unreachable ({exc}), skipping', file=sys.stderr)
        return

    deleted = 0
    for col in collections:
        if col.name.startswith(PREFIX):
            try:
                client.delete_collection(col.name)
                deleted += 1
            except Exception as exc:
                print(f'Failed to delete {col.name}: {exc}', file=sys.stderr)

    if deleted:
        print(f'Deleted {deleted} stale test collection(s)')

    client.close()


if __name__ == '__main__':
    main()
