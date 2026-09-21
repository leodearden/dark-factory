"""The two record shapes one stored Qdrant payload can be read back as.

Three test modules need a record shaped the way mem0 hands it back — the
``MemoryService.get_memory`` fingerprint unit tests, the
``cite_memory``-over-the-real-service end-to-end pin, and the
``_fingerprint_from_record`` convergence test.  Built here as a
TRANSFORMATION of a stored payload rather than written out as a literal
record, for one reason: a literal encodes a THIRD-PARTY contract by hand, and
three hand copies of it drift independently — a mem0 upgrade that changed
``promoted_payload_keys`` would turn one red and leave the others quietly
asserting the retired shape.  Deriving all three from the same rule means they
move together, which is the drift the convergence test exists to prevent.

THE RULE, verified against installed mem0 1.0.11
(``mem0/memory/main.py::Memory.get`` / ``::AsyncMemory.get``):

  * ``PROMOTED_PAYLOAD_KEYS`` are copied to the record's TOP LEVEL, and only
    ``if key in payload`` — one absent from the payload stays absent from the
    record, which is why every read of them must be ``.get()``;
  * those same keys are EXCLUDED from ``metadata`` (``core_and_promoted_keys``);
  * every other payload key — ``category`` among them — stays INSIDE
    ``metadata``.

``MemoryService.get_memory_by_id`` applies none of that: it returns the FULL
unprocessed payload under ``metadata``, all keys at one level.  Both readings
live here so the divergence between them is visible in one screen.
"""

from __future__ import annotations

from typing import Any

#: mem0 1.0.11, ``mem0/memory/main.py::Memory.get``.
PROMOTED_PAYLOAD_KEYS = ('user_id', 'agent_id', 'run_id', 'actor_id', 'role')
CORE_KEYS = ('data', 'hash', 'created_at', 'updated_at', 'id')


def mem0_record(payload: dict[str, Any], *, memory_id: str) -> dict[str, Any]:
    """What mem0's ``get`` returns for a point storing *payload*."""
    core_and_promoted = {*CORE_KEYS, *PROMOTED_PAYLOAD_KEYS}
    record: dict[str, Any] = {
        'id': memory_id,
        'memory': payload.get('data'),
        'hash': payload.get('hash'),
        'created_at': payload.get('created_at'),
        'updated_at': payload.get('updated_at'),
        'score': None,
    }
    for key in PROMOTED_PAYLOAD_KEYS:
        if key in payload:
            record[key] = payload[key]
    record['metadata'] = {k: v for k, v in payload.items() if k not in core_and_promoted}
    return record


def raw_record(payload: dict[str, Any], *, memory_id: str) -> dict[str, Any]:
    """What ``MemoryService.get_memory_by_id`` returns: the payload, untouched."""
    return {
        'id': memory_id,
        'content': payload.get('data'),
        'metadata': dict(payload),
    }
