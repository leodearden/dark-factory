"""Routing-intent lint guard.

Declaration-only lint at the fused-memory ``submit_task`` boundary: flags —
or, when explicitly enforced, hard-rejects — a ``task_kind='normal'``
submission whose title/description/details DECLARE a different execution
path (e.g. "DO NOT IMPLEMENT", "do not attempt to author a TDD plan",
"no-code", "deterministic; no worktree", "escalate to a human instead of
implementing", "--apply against a live store", "DO NOT COMPLETE" anchors)
while carrying no matching ``task_kind``/``execution_class`` declaration.

## Contract (task 2563)

Unlike the sibling recon-stage-scoped guards (``execution_class_guard``,
``premise_lint_guard``, ``deterministic_task_guard``), this guard is scoped
to ``task_kind='normal'`` submissions from ANY caller — the failure mode it
targets (task 2332-style routing-intent churn) and the dominant filing path
(``planning_mode`` PRD-batch decomposition) are not recon-stage-specific.

``routing_intent_finding`` returns ``None`` (no mismatch) when:
- ``task_kind`` is not ``'normal'`` — a submission that already declares a
  non-normal ``task_kind`` needs no lint.
- ``metadata.execution_class`` is ``'operational'`` or ``'decision'`` (the
  non-``'code_tdd'`` members of
  :data:`fused_memory.reconciliation.recon_self_model.EXECUTION_CLASSES`,
  imported rather than redefined so this exemption set can never drift from
  the canonical vocabulary) — an honest non-code declaration is not a
  mismatch to flag.
- A code-change signal (``fix``/``bug``/``crash``/``refactor``) is present
  anywhere across title/description/details — mirrors
  ``operational_ask_registry``'s ``_CODE_CHANGE_TITLE_SIGNALS`` negative
  guard, the defense against a genuine code task whose prose merely
  MENTIONS routing-adjacent terminology in passing (task 2408). Note that
  "implement" is deliberately NOT one of these signal words here (unlike
  the sibling registry): two of this guard's own markers
  (``do_not_implement``, ``escalate_instead_of_implementing``) contain the
  substring "implement" in their canonical phrasing, so including it would
  make those markers permanently self-suppressing.
- None of the declarative markers match any field.

This module is declaration-only: it never coerces ``task_kind`` or
``execution_class`` (task 2085's rejected-blast-radius decision) — it only
flags (see ``routing_intent_warning``) or, when
:func:`routing_intent_enforced` is ``True``, rejects (see
``routing_intent_reject``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from fused_memory.reconciliation.recon_self_model import EXECUTION_CLASSES

logger = logging.getLogger(__name__)

__all__ = [
    'RoutingIntentFinding',
    'routing_intent_finding',
    'routing_intent_reject',
    'routing_intent_warning',
    'routing_intent_enforced',
]

# Truthy string values for FUSED_ROUTING_INTENT_ENFORCE (case-insensitive,
# whitespace-stripped). Unset, empty, or any other value -> warn-only
# (the default) — see routing_intent_enforced().
_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({'1', 'true', 'yes', 'on'})

_EXEMPT_EXECUTION_CLASSES: frozenset[str] = frozenset(EXECUTION_CLASSES) - {'code_tdd'}

# Code-change signal words that unconditionally suppress a routing-intent
# finding, regardless of which field(s) a marker matched in — mirrors
# operational_ask_registry._CODE_CHANGE_TITLE_SIGNALS's "precision over
# recall" defense against a genuine code task whose prose merely mentions
# routing-adjacent terminology in passing (task 2408). Deliberately excludes
# "implement": two markers below (do_not_implement,
# escalate_instead_of_implementing) contain that substring in their own
# canonical phrasing, so including it would make those markers permanently
# dead code (a real task declaring "DO NOT IMPLEMENT" would always be
# suppressed by its own marker text).
_CODE_CHANGE_SIGNALS_RE = re.compile(r'\b(?:fix|bug|crash|refactor)\w*\b', re.IGNORECASE)

# (compiled regex, marker label, detail text) — each anchored to a
# DECLARATIVE self-description (an imperative directive or an explicit
# "this task is/needs ..." framing), not a bare word mention, so a passing
# reference to routing-adjacent terminology does not trip the lint. Extend
# this table as new declaration phrasings surface.
_ROUTING_MARKERS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r'\bdo[ -]not[ -]implement\b', re.IGNORECASE),
        'do_not_implement',
        'Declares "DO NOT IMPLEMENT" — an explicit non-code directive.',
    ),
    (
        re.compile(
            r'\bdo\s+not\s+attempt\s+to\s+(?:author|write)\s+a(?:\s+TDD)?\s+plan\b',
            re.IGNORECASE,
        ),
        'do_not_author_plan',
        'Declares that no TDD plan should be authored for this task.',
    ),
    (
        re.compile(r'\bno[- ]code\b', re.IGNORECASE),
        'no_code_label',
        'Declares a "no-code" label for this task.',
    ),
    (
        re.compile(r'\bno\s+worktree\b', re.IGNORECASE),
        'no_worktree',
        'Declares the task needs no worktree (the deterministic execution path).',
    ),
    (
        re.compile(
            r'\bescalate\s+to\s+(?:a\s+)?human\s+instead\s+of\s+implementing\b',
            re.IGNORECASE,
        ),
        'escalate_instead_of_implementing',
        'Declares the task should escalate to a human instead of being implemented.',
    ),
    (
        re.compile(
            r'--apply\b[^\n]{0,40}\b(?:live|production)\b[^\n]{0,20}\bstore\b',
            re.IGNORECASE,
        ),
        'apply_live_store',
        'Declares a --apply step against a live/production store (an operational ask).',
    ),
    (
        re.compile(r'\bdo[ -]not[ -]complete\b', re.IGNORECASE),
        'do_not_complete',
        'Declares "DO NOT COMPLETE" — an explicit do-not-complete anchor.',
    ),
)


@dataclass(frozen=True)
class RoutingIntentFinding:
    """One or more declarative routing-intent markers matched in a submission.

    ``markers``: dedup'd matched marker labels, in first-matched order.
    ``details``: dedup'd detail text, one per entry in ``markers`` (same
        order/length).
    ``fields``: dedup'd field names (``'title'``/``'description'``/
        ``'details'``) that contained at least one match, in first-matched
        order.
    """

    markers: tuple[str, ...]
    details: tuple[str, ...]
    fields: tuple[str, ...]


def _parse_metadata(metadata: Any) -> dict:
    """Return *metadata* as a dict (best-effort; unknown shapes -> {}).

    Mirrors ``execution_class_guard._parse_metadata``: None -> {}; empty
    string -> {}; dict -> returned as-is; JSON string -> parsed dict (or {}
    on parse failure / non-dict result); anything else -> {}.
    """
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def routing_intent_finding(
    title: str | None,
    description: str | None,
    details: str | None,
    *,
    task_kind: str,
    metadata: str | dict[str, Any] | None,
) -> RoutingIntentFinding | None:
    """Scan a submission for a routing-intent mismatch; see module docstring.

    Returns ``None`` when: ``task_kind`` is not ``'normal'``;
    ``metadata.execution_class`` is an honest non-code declaration
    (``'operational'``/``'decision'``); a code-change signal is present
    anywhere across the three fields; or no declarative marker matches.
    Otherwise returns a :class:`RoutingIntentFinding` naming every matched
    marker, its detail text, and which field(s) it was found in.

    Each field is scanned INDEPENDENTLY (mirrors ``premise_lint_guard``): a
    match must be wholly contained within a single field.
    """
    if task_kind != 'normal':
        return None

    if _parse_metadata(metadata).get('execution_class') in _EXEMPT_EXECUTION_CLASSES:
        return None

    fields: dict[str, str] = {
        'title': title or '',
        'description': description or '',
        'details': details or '',
    }

    combined = ' '.join(fields.values())
    if _CODE_CHANGE_SIGNALS_RE.search(combined):
        return None

    matched_markers: list[str] = []
    matched_details: list[str] = []
    matched_fields: list[str] = []
    for field_name, text in fields.items():
        if not text:
            continue
        for pattern, label, detail in _ROUTING_MARKERS:
            if pattern.search(text):
                if label not in matched_markers:
                    matched_markers.append(label)
                    matched_details.append(detail)
                if field_name not in matched_fields:
                    matched_fields.append(field_name)

    if not matched_markers:
        return None

    return RoutingIntentFinding(
        markers=tuple(matched_markers),
        details=tuple(matched_details),
        fields=tuple(matched_fields),
    )


def routing_intent_reject(finding: RoutingIntentFinding) -> dict[str, Any]:
    """Return a structured ValidationError dict naming *finding*'s markers.

    Hard-reject payload used when :func:`routing_intent_enforced` is
    ``True`` — mirrors the sibling guards' ``{'error', 'error_type':
    'ValidationError', 'hint'}`` shape (see
    ``execution_class_guard._validation_error``).
    """
    markers = ', '.join(finding.markers)
    detail_text = ' '.join(finding.details)
    return {
        'error': (
            f"task_kind='normal' submission declares a different execution "
            f'path than it is filed under (matched marker(s): {markers}). '
            f'{detail_text}'
        ),
        'error_type': 'ValidationError',
        'hint': (
            "If this is genuinely operational/decision work, set "
            "metadata.execution_class to 'operational' or 'decision', or "
            "resubmit with the matching task_kind (e.g. 'deterministic'). "
            'If it is a genuine code task, remove the conflicting '
            'declaration from the title/description/details.'
        ),
    }


def routing_intent_warning(finding: RoutingIntentFinding) -> dict[str, Any]:
    """Return a non-blocking ``{'routing_intent_warning': {...}}`` payload.

    Mirrors task 2206's ``_attach_possible_scope_mismatch`` advisory-marker
    precedent: a structured, non-blocking signal meant to be merged into the
    submit result rather than a rejection — the returned dict carries no
    top-level ``'error'``/``'error_type'`` key.

    Also emits a greppable ``routing_intent_lint.flagged`` WARNING (the
    census line the enforce-flip decision is meant to be based on — observe
    the flagged rate from logs before flipping
    ``FUSED_ROUTING_INTENT_ENFORCE``).
    """
    logger.warning(
        'routing_intent_lint.flagged task_kind=normal markers=%s fields=%s',
        ','.join(finding.markers),
        ','.join(finding.fields),
    )
    return {
        'routing_intent_warning': {
            'markers': list(finding.markers),
            'fields': list(finding.fields),
            'detail': ' '.join(finding.details),
            'hint': (
                "This task_kind='normal' submission's text declares a "
                "different execution path (see 'markers'). If genuinely "
                'operational/decision work, set metadata.execution_class or '
                'resubmit with the matching task_kind; otherwise remove the '
                'conflicting declaration. Set FUSED_ROUTING_INTENT_ENFORCE=1 '
                'to hard-reject such submissions instead of warning.'
            ),
        },
    }


def routing_intent_enforced() -> bool:
    """Read ``FUSED_ROUTING_INTENT_ENFORCE`` from the environment.

    Returns ``True`` (hard-reject) only when the env var is set to a
    recognized truthy string (case-insensitive, whitespace-stripped: '1',
    'true', 'yes', 'on'). Unset, empty, or any other value -> ``False``
    (warn-only, the default).
    """
    raw = os.environ.get('FUSED_ROUTING_INTENT_ENFORCE', '')
    return raw.strip().lower() in _TRUTHY_ENV_VALUES
