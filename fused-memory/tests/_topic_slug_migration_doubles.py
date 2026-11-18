"""Doubles shared by test_normalize_topic_slugs.py and the test_topic_slug_*.py suites."""
from __future__ import annotations


def rec(memory_id: str, topic: object = None, **meta) -> dict:
    """One scroll-shaped record, the shape ``scroll_all_by_metadata`` yields.

    ``topic=None`` means "no ``topic`` key at all", which is a genuinely
    different case from ``topic`` present-and-``None`` — the caller spells the
    latter explicitly via ``metadata``.
    """
    metadata: dict = dict(meta)
    if topic is not None:
        metadata['topic'] = topic
    return {'id': memory_id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': metadata}


def by_reason(skips: list[dict], reason: str) -> list[dict]:
    return [s for s in skips if s.get('reason') == reason]
