"""Operational-ask registry for the task curator filing-policy gate.

Provides a pure, deterministic guard that the curator consults *before* any
LLM call. Operators add an entry per recurring operational live-data/
live-mutation ask — one whose deliverable is an already-built+unit-tested
script and whose remaining work is dry-run -> human-review -> apply against
live external stores, modifying ZERO repo files. The TDD architect cannot
author a RED->GREEN plan for that shape of work, so a match here routes the
candidate straight to a deterministic PURE-GATE (born-at-L2 milestone gate)
instead of letting it reach the architect and bounce as "unactionable".

A registry entry's title/description substrings can incidentally co-occur in
a legitimate CODE-FIX title too — e.g. "Fix a bug in merge_entities against
the live FalkorDB graph" satisfies the merge_entities_live_graph entry's
anchors even though the task is a bug fix, not an operational ask. Because a
route decision short-circuits the LLM entirely, there is no downstream
classifier to catch that false positive (2085 amendment). ``match_candidate``
therefore checks the candidate's title against ``_CODE_CHANGE_TITLE_SIGNALS``
before consulting any registry entry and refuses to match when one is
present, so such asks still reach the architect.

Execution-class demotion (task δ): a candidate whose
``metadata.execution_class`` is ``"operational"`` or ``"decision"`` (see
``fused_memory.reconciliation.recon_self_model.EXECUTION_CLASSES``) is NO
LONGER routed here — ``match_candidate`` returns ``None`` for it EARLY,
before the ``_CODE_CHANGE_TITLE_SIGNALS`` guard and the substring loop below.
The submit boundary ``operational_routing_guard.inject_operational_routing``
(task β) now owns tagged-ask routing: it coerces every tagged operational/
decision ask to a deterministic PURE-GATE on the submit path, so a second
coercion site in the curator would be lock-step duplication (INV-5). The
early ``None`` fires EVEN IF the tagged candidate also incidentally matches a
substring entry — the boundary is the sole coercion site for tagged asks.

Task 2687 originally routed tagged asks here (before the β boundary existed);
task δ narrows that axis to a SKIP now that the boundary owns it. The title/
description substring entries below are retained ONLY as the untagged-legacy
fallback — for asks that carry NO execution_class, where the boundary has
nothing to key on. ``"code_tdd"`` and ``None`` tagging never engaged this
axis and are unchanged: they fall through to the substring loop (a
``code_tdd`` ask must still reach the TDD architect).

Caveat: the substring fallback below only runs when the registry is
configured and non-empty. ``match_candidate``'s sole caller,
``TaskCurator._maybe_route_deterministic``, short-circuits to ``None``
*before* ever calling ``match_candidate`` when the registry path is
unconfigured or the registry file failed to load (missing, unreadable,
unparseable, or empty) — a deliberate fail-open choice (see that method's
docstring). A tagged operational/decision candidate returns ``None`` here
regardless, so its coverage rests entirely on the β submit boundary, not on
the registry being loaded.

Usage::

    from fused_memory.middleware.operational_ask_registry import (
        OperationalAskEntry,
        load_operational_registry,
        match_candidate,
    )

    entries = load_operational_registry(Path("config/operational_ask_registry.yaml"))
    hit = match_candidate(candidate, entries)
    if hit is not None:
        # return CuratorDecision(action='route_deterministic', justification=...)
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fused_memory.utils.safe_yaml import load_yaml_list_file

if TYPE_CHECKING:
    from fused_memory.middleware.task_curator import CandidateTask

logger = logging.getLogger(__name__)
_LOG_LABEL = "operational_ask_registry"


@dataclass(frozen=True)
class OperationalAskEntry:
    """One entry in the operational-ask registry.

    Matching logic (evaluated case-insensitively):
    - ALL strings in ``title_substrings`` must appear in the candidate title.
    - AT LEAST ONE string in ``description_substrings`` must appear in the
      combined candidate description + details text.
    """

    name: str
    reason: str
    title_substrings: list[str]
    description_substrings: list[str]


def load_operational_registry(path: Path | None) -> list[OperationalAskEntry]:
    """Load the operational-ask registry from a YAML file; never raises.

    File-level failures (unset path, missing / unreadable / undecodable file,
    invalid YAML, non-list document) degrade to [] as specified by
    fused_memory/utils/safe_yaml.py::load_yaml_list_file. Malformed individual
    entries are skipped with one WARNING each.
    """
    entries: list[OperationalAskEntry] = []
    for item in load_yaml_list_file(
        path,
        logger=logger,
        label=_LOG_LABEL,
        consequence="registry disabled",
    ):
        if not isinstance(item, dict):
            logger.warning(
                "%s: skipping non-dict entry in %s: %r",
                _LOG_LABEL, path, item
            )
            continue

        # Validate required fields
        missing = [
            f for f in ("name", "reason", "title_substrings", "description_substrings")
            if f not in item
        ]
        if missing:
            logger.warning(
                "%s: skipping entry missing fields %s in %s: %r",
                _LOG_LABEL, missing, path, item.get("name", "<unnamed>"),
            )
            continue

        title_subs = item["title_substrings"]
        desc_subs = item["description_substrings"]
        if not isinstance(title_subs, list) or not isinstance(desc_subs, list):
            logger.warning(
                "%s: skipping entry %r — title_substrings and "
                "description_substrings must be lists",
                _LOG_LABEL,
                item.get("name", "<unnamed>"),
            )
            continue

        if not title_subs or not desc_subs:
            logger.warning(
                "%s: skipping entry %r — title_substrings and "
                "description_substrings must be non-empty (an empty title_substrings "
                "would match every candidate title via all([]) == True, degrading the "
                "gate to a description-only match)",
                _LOG_LABEL,
                item.get("name", "<unnamed>"),
            )
            continue

        entries.append(
            OperationalAskEntry(
                name=str(item["name"]),
                reason=str(item["reason"]),
                title_substrings=[str(s) for s in title_subs],
                description_substrings=[str(s) for s in desc_subs],
            )
        )

    return entries


# Title phrases that strongly signal the candidate describes a CODE change (a
# bug/crash fix or a new-feature implementation) rather than an operational
# live-data/live-mutation ask. Checked BEFORE any registry entry in
# match_candidate(): a registry entry's positive title/description
# substrings can co-occur incidentally with one of these in a legitimate
# code-fix title — e.g. "Fix a bug in merge_entities against the live
# FalkorDB graph" independently satisfies the merge_entities_live_graph
# entry's anchors even though the task is a code fix, not an operational ask
# (2085 amendment). This is a deliberately conservative, registry-wide
# guard: a false negative here just means a genuinely-operational ask falls
# through to the architect and bounces "unactionable" again (the pre-2085
# status quo, self-healed by adding/adjusting a registry entry) — whereas a
# false positive silently skips the architect and stalls a real code change
# at a born-at-L2 gate a human is unlikely to reject. Precision is favored
# over recall.
_CODE_CHANGE_TITLE_SIGNALS: tuple[str, ...] = ("fix", "bug", "crash", "implement")

# Execution-class values the submit boundary OWNS — a candidate tagged with
# one of these is SKIPPED by match_candidate (returns None early), NOT routed
# (task δ demotion; task 2687 originally routed them here). The submit
# boundary operational_routing_guard.inject_operational_routing (task β)
# coerces every tagged operational/decision ask to a deterministic PURE-GATE
# on the submit path, so re-routing here would be a second coercion site
# (INV-5). Mirrors operational_routing_guard._COERCED_EXECUTION_CLASSES =
# {'operational', 'decision'} for symmetry; spelled out explicitly (NOT
# derived by subtracting 'code_tdd' from recon_self_model.EXECUTION_CLASSES)
# so a future code-oriented execution class added there is not silently swept
# into the skip set. 'code_tdd' is intentionally excluded — a code_tdd-tagged
# candidate falls through to the substring loop below and must still reach the
# TDD architect.
_BOUNDARY_OWNED_EXECUTION_CLASSES: frozenset[str] = frozenset({"operational", "decision"})


def match_candidate(
    candidate: CandidateTask,
    entries: list[OperationalAskEntry],
) -> OperationalAskEntry | None:
    """Return the first matching registry entry for *candidate*, or ``None``.

    Checked FIRST, before anything else: if ``candidate.execution_class``
    (case-insensitively) is ``"operational"`` or ``"decision"``, this returns
    ``None`` EARLY — regardless of *entries*, the ``_CODE_CHANGE_TITLE_SIGNALS``
    guard, or any title/description substring (task δ demotion). Such a tagged
    ask has already been coerced to a deterministic PURE-GATE by the submit
    boundary ``operational_routing_guard.inject_operational_routing`` (task β),
    so re-routing it here would be a second coercion site (INV-5). The early
    ``None`` fires EVEN IF the tagged candidate also matches a substring entry
    below — it does not fall through to the substring loop. ``execution_class``
    of ``"code_tdd"`` or ``None`` does not engage this skip and falls through
    to the substring-only behavior below unchanged (task 2687 originally
    routed tagged asks here; task δ narrowed the axis to a skip once the β
    boundary took ownership).

    For an untagged (or ``"code_tdd"``) candidate, matching is case-insensitive
    string search (the untagged-legacy fallback):
    - ALL strings in ``entry.title_substrings`` must appear in
      ``candidate.title``.
    - AT LEAST ONE string in ``entry.description_substrings`` must appear in
      the combined ``candidate.description + " " + candidate.details``.

    Before consulting any entry, the candidate's title is checked against
    ``_CODE_CHANGE_TITLE_SIGNALS`` — a hit (e.g. "fix", "bug", "crash",
    "implement") unconditionally returns ``None`` regardless of entry
    substrings, since such a title indicates a code change that must still
    reach the TDD architect (2085 amendment: closes the false-positive gap
    where a bug-fix title incidentally satisfies an entry's positive
    anchors — see module docstring).

    Returns the first match in list order (deterministic). Returns ``None``
    when *entries* is empty or no entry matches.

    Pure string operations — no regex, no async, no I/O. Never raises: even a
    malformed (non-string) ``candidate.execution_class`` — e.g. an int or
    list from a corrupt metadata write — degrades to being treated as untagged
    (falls through to the substring fallback, not skipped) rather than raising,
    via the ``str(...)`` coercion below.
    """
    # Tagged operational/decision asks are OWNED by the submit boundary
    # (inject_operational_routing, task β), which already coerced this ask to a
    # deterministic pure-gate; re-routing here would be a second coercion site
    # (INV-5). Return None EARLY — before the substring loop — so a tagged ask
    # that incidentally matches an entry is still not routed. The substring
    # loop below is retained ONLY as the untagged-legacy fallback (task δ).
    execution_class = str(candidate.execution_class or "").strip().lower()
    if execution_class in _BOUNDARY_OWNED_EXECUTION_CLASSES:
        return None

    title_lower = (candidate.title or "").lower()

    if any(signal in title_lower for signal in _CODE_CHANGE_TITLE_SIGNALS):
        return None

    desc_lower = (
        (candidate.description or "") + " " + (candidate.details or "")
    ).lower()

    for entry in entries:
        # All title substrings must match
        if not all(sub.lower() in title_lower for sub in entry.title_substrings):
            continue
        # At least one description substring must match
        if not any(sub.lower() in desc_lower for sub in entry.description_substrings):
            continue
        return entry

    return None
