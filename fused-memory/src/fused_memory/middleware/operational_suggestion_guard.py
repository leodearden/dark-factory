"""Operational-suggestion lint guard (warn-only).

Declaration-only lint at the fused-memory ``submit_task`` boundary: flags —
but never rejects, never coerces — a ``task_kind='normal'`` submission whose
title/description/details carry IMPLICIT operational phrasing (restart/
redeploy/reload/confirm/deploy/systemctl) while declaring no code deliverable
and no honest non-code classification.

## Contract (task 2679, part 2)

Unlike ``routing_intent_guard`` (which targets EXPLICIT declarative markers
such as "DO NOT IMPLEMENT" and has an optional hard-reject enforce mode),
this guard targets IMPLICIT operational phrasing and is mandated WARN-ONLY —
there is no enforce/reject path here at all. It is scoped to
``task_kind='normal'`` submissions from ANY caller, mirroring
``routing_intent_guard``'s any-caller scope: the failure mode this guards
against (a files-less operational ask filed as ordinary code work) is not
recon-stage-specific either.

``operational_suggestion_finding`` returns ``None`` (no suggestion) when:
- ``task_kind`` is not ``'normal'`` — a submission that already declares a
  non-normal ``task_kind`` (e.g. ``'deterministic'``) needs no suggestion.
- ``metadata.execution_class`` is ``'operational'`` or ``'decision'`` (the
  non-``'code_tdd'`` members of
  :data:`fused_memory.reconciliation.recon_self_model.EXECUTION_CLASSES`,
  imported rather than redefined so this exemption set can never drift from
  the canonical vocabulary) — an honest non-code declaration is not a
  mismatch to flag.
- ``metadata.files`` is non-empty (parsed via
  :func:`fused_memory.middleware.lock_charter_guard.extract_files`) — a
  declared code deliverable means this is not a files-less operational ask.
- A code-change signal (``fix``/``bug``/``crash``/``implement``) is present
  anywhere across title/description/details — mirrors
  ``operational_ask_registry._CODE_CHANGE_TITLE_SIGNALS``'s word set,
  defined locally here rather than imported across the module boundary
  (this codebase's established convention — see ``routing_intent_guard``,
  ``execution_class_guard``, ``deterministic_task_guard``, and
  ``model_overrides_guard``, which each keep their own local
  ``_parse_metadata`` rather than importing a private helper from a sibling
  module). Unlike ``operational_ask_registry`` (title-only scope) and like
  ``routing_intent_guard`` (combined-text scope), this scans the combined
  title + description + details text — the defense against a genuine code
  task whose prose merely MENTIONS operational terminology in passing.
- None of the operational markers match any field.
- The ONLY markers that matched are ``confirm``/``reload``/``deploy`` (the
  "weak" markers — common enough in ordinary code-task prose to collide,
  e.g. "Confirm the retry logic handles timeouts", "Reload the config
  module after edit", "Deploy config schema field") AND the same field
  also names a code-level artifact (module/schema/field/logic/function/
  class/method/variable/parameter/argument/test) rather than a running
  system. ``restart``/``redeploy``/``systemctl`` are specific enough to
  fire standalone and are never gated this way. This is a precision
  tradeoff, not a completeness guarantee — it targets the concrete
  collision patterns above, not every possible false positive; the
  ``operational_task_suggestion.flagged`` census WARNING rate remains the
  rollout signal to watch for anything it misses.

This module is declaration-only and WARN-ONLY: it never coerces
``task_kind`` or ``execution_class`` and never rejects a submission — see
``operational_suggestion_warning`` for the non-blocking payload shape.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from fused_memory.middleware.lock_charter_guard import extract_files
from fused_memory.reconciliation.recon_self_model import EXECUTION_CLASSES

logger = logging.getLogger(__name__)

__all__ = [
    'OperationalSuggestionFinding',
    'operational_suggestion_finding',
    'operational_suggestion_warning',
]

_EXEMPT_EXECUTION_CLASSES: frozenset[str] = frozenset(
    c for c in EXECUTION_CLASSES if c != 'code_tdd'
)

# Code-change signal words that unconditionally suppress a finding, regardless
# of which field an operational marker matched in -- mirrors
# operational_ask_registry._CODE_CHANGE_TITLE_SIGNALS = ("fix", "bug", "crash",
# "implement"). Defined locally rather than imported across the module
# boundary (see module docstring). Scanned across the combined
# title+description+details text (unlike operational_ask_registry's
# title-only scope), like routing_intent_guard's code-change suppression --
# the defense against a genuine code task whose prose merely mentions
# operational terminology in passing (e.g. "Fix the restart-loop bug").
_CODE_CHANGE_SIGNALS_RE = re.compile(r'\b(?:fix|bug|crash|implement)\w*\b', re.IGNORECASE)

# (compiled regex, marker label) -- each anchored to a whole word so a
# marker does not spuriously fire on a substring occurrence inside an
# unrelated word (e.g. the "deploy" marker does not fire inside "redeploy";
# that phrasing is its own independent marker below). Extend this table as
# new operational-phrasing markers surface.
_OPERATIONAL_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\brestart\b', re.IGNORECASE), 'restart'),
    (re.compile(r'\bredeploy\b', re.IGNORECASE), 'redeploy'),
    (re.compile(r'\breload\b', re.IGNORECASE), 'reload'),
    (re.compile(r'\bconfirm\b', re.IGNORECASE), 'confirm'),
    (re.compile(r'\bdeploy\b', re.IGNORECASE), 'deploy'),
    (re.compile(r'\bsystemctl\b', re.IGNORECASE), 'systemctl'),
)

# task 2679 amendment pass (reviewer_comprehensive robustness finding):
# "confirm"/"reload"/"deploy" are generic enough to occur routinely in
# legitimate code-task prose (see module docstring). Gate these three
# ("weak") markers: a match is suppressed when the SAME field also names a
# code-level artifact rather than a running system -- "restart"/
# "redeploy"/"systemctl" are specific enough to fire standalone and are
# never gated.
_WEAK_MARKER_LABELS: frozenset[str] = frozenset({'confirm', 'reload', 'deploy'})

_CODE_ARTIFACT_RE = re.compile(
    r'\b(?:module|schema|field|logic|function|class|method|variable|parameter|argument|test)\w*\b',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OperationalSuggestionFinding:
    """One or more implicit operational-phrasing markers matched in a submission.

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

    Mirrors ``routing_intent_guard._parse_metadata`` / ``execution_class_guard
    ._parse_metadata``: None -> {}; empty string -> {}; dict -> returned
    as-is; JSON string -> parsed dict (or {} on parse failure / non-dict
    result); anything else -> {}.
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


def operational_suggestion_finding(
    title: str | None,
    description: str | None,
    details: str | None,
    *,
    task_kind: str,
    metadata: str | dict[str, Any] | None,
) -> OperationalSuggestionFinding | None:
    """Scan a submission for implicit operational phrasing; see module docstring.

    Returns ``None`` when: ``task_kind`` is not ``'normal'``;
    ``metadata.execution_class`` is an honest non-code declaration
    (``'operational'``/``'decision'``); ``metadata.files`` is non-empty (a
    declared code deliverable); a code-change signal is present anywhere
    across the three fields; or no operational marker matches. Otherwise
    returns an :class:`OperationalSuggestionFinding` naming every matched
    marker, its detail text, and which field(s) it was found in.

    Each field is scanned INDEPENDENTLY (mirrors ``routing_intent_finding``):
    a match must be wholly contained within a single field.
    """
    if task_kind != 'normal':
        return None

    if _parse_metadata(metadata).get('execution_class') in _EXEMPT_EXECUTION_CLASSES:
        return None

    if extract_files(metadata):
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
    matched_fields: list[str] = []
    for field_name, text in fields.items():
        if not text:
            continue
        field_has_code_artifact = bool(_CODE_ARTIFACT_RE.search(text))
        for pattern, label in _OPERATIONAL_MARKERS:
            if pattern.search(text):
                if label in _WEAK_MARKER_LABELS and field_has_code_artifact:
                    # Generic marker co-occurs with a code-level artifact
                    # noun in this same field (e.g. "Reload the config
                    # module after edit") -- treat as a code task, not an
                    # operational ask. See _WEAK_MARKER_LABELS above.
                    continue
                if label not in matched_markers:
                    matched_markers.append(label)
                if field_name not in matched_fields:
                    matched_fields.append(field_name)

    if not matched_markers:
        return None

    matched_details = tuple(
        f"Declares {label!r} operational phrasing." for label in matched_markers
    )

    return OperationalSuggestionFinding(
        markers=tuple(matched_markers),
        details=matched_details,
        fields=tuple(matched_fields),
    )


def operational_suggestion_warning(finding: OperationalSuggestionFinding) -> dict[str, Any]:
    """Return a non-blocking ``{'operational_suggestion_warning': {...}}`` payload.

    Mirrors ``routing_intent_warning``'s non-blocking advisory-marker shape:
    the returned dict carries no top-level ``'error'``/``'error_type'`` key,
    and is meant to be merged into the ``submit_task`` result rather than
    rejecting it.

    Also emits a greppable ``operational_task_suggestion.flagged`` WARNING —
    the census line for observing how often this guard fires.
    """
    logger.warning(
        'operational_task_suggestion.flagged task_kind=normal markers=%s fields=%s',
        ','.join(finding.markers),
        ','.join(finding.fields),
    )
    return {
        'operational_suggestion_warning': {
            'markers': list(finding.markers),
            'fields': list(finding.fields),
            'detail': ' '.join(finding.details),
            'hint': (
                "This task_kind='normal' submission's text declares "
                "operational phrasing (see 'markers') with no code "
                'deliverable. If this is genuinely operational/decision '
                "work, resubmit with task_kind='deterministic' or set "
                "metadata.execution_class to 'operational' or 'decision'. "
                'This is advisory only -- task_kind and execution_class are '
                'never coerced.'
            ),
        },
    }
