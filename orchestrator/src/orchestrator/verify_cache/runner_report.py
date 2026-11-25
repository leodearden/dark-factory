"""What a runner ships its dispatcher of a verify's cache: the report of its settled units,
since its cache record stays on its host (plans/verify-result-cache-prd.md Amendment A5)."""

from __future__ import annotations

import dataclasses

from orchestrator.verify import VerifyResult
from orchestrator.verify_cache.recorder import CacheRecord, report_of


def with_cache_report(result: VerifyResult) -> VerifyResult:
    """*result* carrying the report of its units for its dispatcher's recorder, each unit's
    admission evaluated here where its leg ran; the report is None when the cache settled nothing."""
    record = result.cache_record
    report = (
        report_of(record, flake_suppressed=result.flake_suppression is not None)
        if isinstance(record, CacheRecord) else None
    )
    return dataclasses.replace(result, cache_report=report)
