"""Qdrant error responses, built as the live server words them.

``fused_memory.backends.mem0_client::is_missing_collection_error`` decides by
the wording of Qdrant's absent-collection 404. Every test that fakes that 404
builds it here, so a change in Qdrant's wording is one edit, and no copy can
drift from what the predicate has to match.

Kept out of ``_fm_helpers.py`` for the reason ``_mem0_record_shapes.py`` gives:
one subject per helper module.
"""

import json

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse


def missing_collection_404(collection: str) -> UnexpectedResponse:
    """Qdrant's 404 for an absent *collection*, with the body a live scroll/count returns."""
    body = {'status': {'error': f"Not found: Collection `{collection}` doesn't exist!"}}
    return UnexpectedResponse(404, 'Not Found', json.dumps(body).encode(), httpx.Headers())
