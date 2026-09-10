"""The git-authority landing tier, extracted from the escalation MCP server.

PRD ``docs/prds/landed-not-done-recovery.md`` label ζ (task 4887).  Everything
here lived as 4-space-indent siblings inside
``escalation/src/escalation/server.py::create_server``, reachable only by
building a whole MCP server — which is why the periodic landed reconciler that
PRD's leaf η (task 4651) needs could not reuse the fully-guarded verdict
``merge_status`` computes and throws away.

**Where this lives** resolves that PRD's Open question 4 in favour of its own
recommendation: ``orchestrator/pyproject.toml`` already declares ``escalation``
as a workspace dependency (and not the reverse), so η — which lives in
``orchestrator`` — imports this module STATICALLY, with no new runtime
indirection.  ``shared/`` was rejected because it would place a runtime
reverse-import of ``orchestrator.landing_evidence`` inside the package every
other package depends on: strictly worse layering than the status quo, since
the reverse import would then sit in the most widely-imported package rather
than in the one that already tolerates it.

**Two import rules this module must keep.**  It must never import
``escalation.server`` — server.py imports this, and a cycle would turn a
graceful degradation into a server-construction failure.  And its
``orchestrator.landing_evidence`` import is RUNTIME-ONLY, made inside the
fire-safe wrapper in :func:`probe_landing`: ``escalation/pyproject.toml``
declares no orchestrator dependency, and the import resolves at runtime only
because the escalation server is hosted inside the orchestrator process.
Hoisting it to module level would convert that degradation into an import-time
crash.  ``shared.*`` imports are intra-workspace and unconstrained.

**On the two non-landing outcomes.**  ``merge_status`` today renders
:attr:`GitAuthorityOutcome.landed_unconfirmed` and
:attr:`GitAuthorityOutcome.no_signal` IDENTICALLY — as the bare Tier-4
``unknown``.  The distinction is materialized on the verdict rather than on
the wire because the ``merge_status`` response vocabulary belongs to task
**4831** (``plans/merge-status-durable-non-landed-prd.md`` D1/D2), which
consumes this module and maps the two onto its epistemic states.  Do not add
a response field here to carry it.
"""
from __future__ import annotations

import enum
from typing import Any

from shared.merge_state import MergeState


class GitAuthorityOutcome(enum.StrEnum):
    """What git authority was able to establish about a branch's landing.

    A ``StrEnum`` rather than bare strings so a consumer switches on a closed
    set, and for consistency with the two enums this code sits between —
    ``shared.merge_state.MergeState`` above it and
    ``orchestrator.landing_evidence.LandingReason`` below.

    ``landed_unconfirmed`` and ``no_signal`` are DIFFERENT PROPOSITIONS and
    that is the whole reason both exist: the first means git positively
    established a landing (the branch is an ancestor of main, or a merge
    marker was found) but attribution or effect-survival could not be
    confirmed; the second means nothing was established at all.  Collapsing
    them is exactly the conflation task 4831 exists to undo.
    """

    found_on_main = 'found_on_main'
    landed_unconfirmed = 'landed_unconfirmed'
    no_signal = 'no_signal'


def found_on_main_response(request_id: str | None, merge_sha: str) -> dict[str, Any]:
    """Build the git-authority Tier-3.5 done/found_on_main response.

    ``merge_sha`` is a commit ON MAIN on both resolution paths, with one
    explicit exception stated below (task 3103):

    - **Live-branch path** (``is_ancestor`` hit): the citation commit
      discovered by ``validate_landing_evidence`` — a commit on main
      whose subject cites the task.
    - **Deleted-branch path** (``find_merge_marker`` hit): the
      merge-commit SHA found on main via ``git log``.

    Both are effect-present-checked against current main HEAD before
    being returned, so ``merge_sha`` is safe to record as provenance
    as-is.  (Before task 3103 the live-branch path returned the *branch
    tip*, which for a ``--no-ff`` merge is a distinct commit that is not
    on main's first-parent chain — callers were told to prefer the
    deleted-branch path's value.  That caveat no longer applies.)

    **The one exception — ``git.commit_citation_pattern == ''``.**  That
    is the documented per-project opt-out for projects with no citation
    convention (config.py; ``find_task_citation_commit`` honours it by
    returning None for everything, so running the gate would reject
    unconditionally and turn this tier into dead code).  On that setting
    the live-branch path skips the citation gate entirely and
    ``merge_sha`` is the raw BRANCH TIP, neither citation-discovered nor
    effect-present-checked — i.e. exactly the pre-3103 ``--no-ff`` wart,
    deliberately retained as the price of the opt-out (the degeneracy
    guard still applies).  Do not read the paragraphs above as
    unconditional: on such a project a caller stamping ``merge_sha`` as
    provenance is recording a branch tip, and a reverted landing is
    indistinguishable from a live one (review #4).  The opt-out is
    ``''`` only; ``None`` means "use the built-in default pattern" and
    keeps the full guarantee.  Both SKILL.md runbooks carry the same
    exception.
    """
    return {
        'state': MergeState.done,
        'request_id': request_id,
        'generation': 1,
        'kind': 'found_on_main',
        'merge_sha': merge_sha,
        'outcome': 'found_on_main',
    }
