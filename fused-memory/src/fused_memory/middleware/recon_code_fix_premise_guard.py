"""Premise-verification guard for Stage 2 self-referential recon code-fix task-filing.

Provides a pure, deterministic curator pre-check that drops a self-referential
recon code-fix candidate ONLY WHILE the live source/tests still refute its
premise. Mirrors cancelled_premise_blocklist.py in shape, but where that
module drops matching candidates unconditionally forever, this module
re-verifies against the live source tree on every call — the drop is
self-correcting: if a refactor later makes the premise valid, the candidate
stops matching and proceeds to the architect.

Usage::

    from fused_memory.middleware.recon_code_fix_premise_guard import (
        PremiseEntry,
        load_premise_registry,
    )

    entries = load_premise_registry(Path("config/recon_code_fix_premise_registry.yaml"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceAssertion:
    """One live source/test assertion backing a PremiseEntry.

    Holds iff every ``must_contain`` substring is present in the cited file's
    text AND every ``must_not_contain`` substring is absent from it.
    """

    file: str
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PremiseEntry:
    """One entry in the recon code-fix premise-verification registry.

    Matching logic mirrors cancelled_premise_blocklist.BlocklistEntry
    (evaluated case-insensitively):
    - ALL strings in ``title_substrings`` must appear in the candidate title.
    - AT LEAST ONE string in ``description_substrings`` must appear in the
      combined candidate description + details text.

    Unlike BlocklistEntry, a textual match alone does not drop the candidate:
    ``source_assertions`` must ALSO currently hold against the live source
    tree (see ``verify_premise_refuted``) — the drop is self-correcting.
    """

    name: str
    reason: str
    title_substrings: list[str]
    description_substrings: list[str]
    source_assertions: list[SourceAssertion]


def _coerce_source_assertion(item: object) -> SourceAssertion | None:
    """Coerce one raw YAML mapping into a SourceAssertion, or None if malformed."""
    if not isinstance(item, dict) or "file" not in item:
        return None
    must_contain = item.get("must_contain", [])
    must_not_contain = item.get("must_not_contain", [])
    if not isinstance(must_contain, list) or not isinstance(must_not_contain, list):
        return None
    return SourceAssertion(
        file=str(item["file"]),
        must_contain=[str(s) for s in must_contain],
        must_not_contain=[str(s) for s in must_not_contain],
    )


def load_premise_registry(path: Path | None) -> list[PremiseEntry]:
    """Load the recon code-fix premise-verification registry from a YAML file.

    Returns an empty list (without warning) when *path* is ``None``.
    Returns an empty list and emits one WARNING when the file is missing,
    unreadable, or not valid YAML.
    Skips malformed individual entries with one WARNING each while returning
    the well-formed entries from the same file.

    The function never raises — all failures degrade gracefully to [].
    """
    if path is None:
        return []

    # Missing-file / unreadable
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            "recon_code_fix_premise_guard: file not found: %s — guard disabled", path
        )
        return []
    except OSError as exc:
        logger.warning(
            "recon_code_fix_premise_guard: cannot read %s: %s — guard disabled",
            path, exc,
        )
        return []

    # Parse
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning(
            "recon_code_fix_premise_guard: YAML parse error in %s: %s — guard disabled",
            path, exc,
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "recon_code_fix_premise_guard: expected a YAML list in %s, got %s — guard disabled",
            path, type(data).__name__,
        )
        return []

    entries: list[PremiseEntry] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning(
                "recon_code_fix_premise_guard: skipping non-dict entry in %s: %r", path, item
            )
            continue

        # Validate required fields
        missing = [
            f for f in (
                "name", "reason", "title_substrings", "description_substrings",
                "source_assertions",
            )
            if f not in item
        ]
        if missing:
            logger.warning(
                "recon_code_fix_premise_guard: skipping entry missing fields %s in %s: %r",
                missing, path, item.get("name", "<unnamed>"),
            )
            continue

        title_subs = item["title_substrings"]
        desc_subs = item["description_substrings"]
        raw_assertions = item["source_assertions"]
        if not isinstance(title_subs, list) or not isinstance(desc_subs, list):
            logger.warning(
                "recon_code_fix_premise_guard: skipping entry %r — title_substrings and "
                "description_substrings must be lists",
                item.get("name", "<unnamed>"),
            )
            continue
        if not isinstance(raw_assertions, list):
            logger.warning(
                "recon_code_fix_premise_guard: skipping entry %r — source_assertions must "
                "be a list",
                item.get("name", "<unnamed>"),
            )
            continue

        assertions: list[SourceAssertion] = []
        malformed_assertion = False
        for raw in raw_assertions:
            sa = _coerce_source_assertion(raw)
            if sa is None:
                malformed_assertion = True
                break
            assertions.append(sa)
        if malformed_assertion:
            logger.warning(
                "recon_code_fix_premise_guard: skipping entry %r — malformed "
                "source_assertions entry (each requires a 'file' key)",
                item.get("name", "<unnamed>"),
            )
            continue

        entries.append(
            PremiseEntry(
                name=str(item["name"]),
                reason=str(item["reason"]),
                title_substrings=[str(s) for s in title_subs],
                description_substrings=[str(s) for s in desc_subs],
                source_assertions=assertions,
            )
        )

    return entries
