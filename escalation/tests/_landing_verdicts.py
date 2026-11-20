"""A hand-built ``branch_work_landed`` answer for the git-authority stub tests.

Both live-ref arms of ``escalation/src/escalation/git_authority.py`` hand their
whole decision to
``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::branch_work_landed``
(task 4648).  That producer shells ``git cherry``, which a ``SimpleNamespace``
``git_ops`` stub cannot answer, so a stub test pins only how an arm maps each
of its reasons.  The real producer runs over real git in the same test files.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


def stub_landing_producer(
    monkeypatch: pytest.MonkeyPatch, reason: str, *, evidence_sha: str | None = None,
) -> AsyncMock:
    """Make ``branch_work_landed`` answer a verdict for *reason*; return the stub.

    *reason* is a ``LandingReason`` spelling; ``'landed'`` accepts, anchored on
    *evidence_sha*.  The stub is patched onto the DEFINING module, which the
    arms' per-call import resolves.
    """
    from orchestrator.merge_lane.landing_evidence import (
        LandingMethod,
        LandingReason,
        LandingVerdict,
    )
    landing_reason = LandingReason(reason)
    producer = AsyncMock(return_value=LandingVerdict(
        accepted=landing_reason is LandingReason.landed,
        evidence_sha=evidence_sha,
        reason=landing_reason,
        probe={'reason': reason, 'method': str(LandingMethod.patch_id)},
        method=LandingMethod.patch_id,
    ))
    monkeypatch.setattr('orchestrator.merge_lane.landing_evidence.branch_work_landed', producer)
    return producer
