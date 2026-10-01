"""Detects a Stage 1 finding that asserts a live-workflow signal its own payload does not carry.

Reify run 6aa50844 deferred a disposition because "Live-Workflow Signals show
task/5891 is live (worktree, orchestrator)", while the ``### Live-Workflow
Signals`` section of that same payload listed nothing.  There is no
``live_workflow_contradiction`` detector to correct: Stage 1 findings are
LLM-emitted through ``mcp__recon-report__add_finding``, so the check reads the
findings' prose after the fact.

Extraction is deliberately conservative and under-reports.  A citation needs a
``task/<id>`` ref, a signal token and live language in one sentence window with
no negation cue, and a window naming two tasks is skipped as ambiguous, because
a false annotation would itself be the false-signal class this guard catches.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from fused_memory.reconciliation.live_workflow_section import LiveSignal

#: ``content`` is the structured-JSON fallback key of a finding.
_SCANNED_FIELDS = ('description', 'suggested_action', 'content')

#: Prose spellings of a signal besides its rendered value.
_SIGNAL_ALIASES = {'recent commit': LiveSignal.RECENT_COMMIT}

_SIGNAL_BY_SPELLING: dict[str, LiveSignal] = {
    **{signal.value: signal for signal in LiveSignal},
    **_SIGNAL_ALIASES,
}

_SIGNAL_TOKEN = re.compile(
    r'\b('
    + '|'.join(re.escape(spelling) for spelling in sorted(_SIGNAL_BY_SPELLING, key=len, reverse=True))
    + r')\b',
    re.IGNORECASE,
)
_TASK_REF = re.compile(r'\btask/(\d+)\b', re.IGNORECASE)
_LIVE_LANGUAGE = re.compile(r'\blive\b', re.IGNORECASE)
_NEGATION_CUE = re.compile(r'\b(?:no|not|without|absent|none)\b', re.IGNORECASE)
_WINDOW_BREAK = re.compile(r'[.;\n]')


@dataclass(frozen=True)
class LiveWorkflowCitation:
    """A finding's claim that *task_id* is live through *signals*."""

    task_id: str
    signals: frozenset[LiveSignal]


def extract_live_workflow_citations(flag: object) -> tuple[LiveWorkflowCitation, ...]:
    """The live-workflow citations in *flag*'s prose, one per task in first-cited order.

    Any value is accepted: a non-mapping flag, or a scanned field that is not a
    str, contributes nothing.
    """
    if not isinstance(flag, Mapping):
        return ()
    cited: dict[str, set[LiveSignal]] = {}
    for window in _windows(flag):
        citation = _citation_in(window)
        if citation is not None:
            cited.setdefault(citation.task_id, set()).update(citation.signals)
    return tuple(
        LiveWorkflowCitation(task_id, frozenset(signals)) for task_id, signals in cited.items()
    )


def _windows(flag: Mapping) -> Iterator[str]:
    for field in _SCANNED_FIELDS:
        text = flag.get(field)
        if isinstance(text, str):
            yield from _WINDOW_BREAK.split(text)


def _citation_in(window: str) -> LiveWorkflowCitation | None:
    task_ids = {match.group(1) for match in _TASK_REF.finditer(window)}
    signals = frozenset(
        _SIGNAL_BY_SPELLING[match.group(1).lower()] for match in _SIGNAL_TOKEN.finditer(window)
    )
    asserts_live = _LIVE_LANGUAGE.search(window) and not _NEGATION_CUE.search(window)
    if len(task_ids) != 1 or not signals or not asserts_live:
        return None
    return LiveWorkflowCitation(task_ids.pop(), signals)
