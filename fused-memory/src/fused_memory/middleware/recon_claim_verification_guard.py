"""Attributed-claim extraction/verification for curator candidates (task 2438).

Provides a pure, deterministic pre-check that extracts specific code-level
claims a candidate attributes to a completed task/commit/ACTION (a metadata
key, a stamp like ``foo=true``, or a named identifier) so they can be
verified against the live source tree + git history before they are
embedded in a filed task's description as fact.

Motivated by a Stage 2 reconciliation incident, but wired (via
``TaskCurator._maybe_flag_unverified_claims``) into the curator's general
``curate`` / ``curate_batch_prepared`` entry points — when
``CuratorConfig.recon_claim_verification_enabled`` is set, it runs over
EVERY candidate the curator sees, not just Stage-2-recon-filed ones; there
is no candidate-origin field to narrow it further (``CandidateTask``'s
closest field, ``spawn_context``, distinguishes review/steward-triage/
planning/manual, not stage2-recon vs. other origins).

Motivating incident: Stage 2 reconciliation filed task 2433 asserting, as
verified fact, that "task 2372 added an ACTION #5 stamp
(metadata.done_provenance_invalidated=true) on task reopen." No such
stamp/token has ever existed — a FABRICATION: a confidently-specific,
plausible-sounding but false claim attributed to another completed task's
implementation. This module extracts exactly the kind of claim that
incident embedded (``extract_attributed_claims``), verifies it against a
caller-supplied probe (``verify_attributed_claims``), and composes both
(``unverified_claims_in_text``). The one impure git-backed probe
(``make_source_and_history_probe``) lives here too, mirroring
recon_code_fix_premise_guard.py's split between pure detection and a
live-state re-check performed fresh on every call.

Usage::

    from fused_memory.middleware.recon_claim_verification_guard import (
        make_source_and_history_probe,
        unverified_claims_in_text,
    )

    probe = make_source_and_history_probe(repo_root)
    unverified = unverified_claims_in_text(candidate_text, probe)
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "AttributedClaim",
    "extract_attributed_claims",
    "make_source_and_history_probe",
    "unverified_claims_in_text",
    "verify_attributed_claims",
]


@dataclass(frozen=True)
class AttributedClaim:
    """One specific code-level claim attributed to a completed task/commit/ACTION.

    ``token`` is the code literal to verify (e.g. a metadata key or stamp
    identifier, with any ``metadata.`` prefix stripped so it matches how the
    identifier would actually appear in source). ``attribution`` is the
    cited anchor text tying the claim to specific prior work (e.g.
    "per task 2372, ACTION #5"). ``span`` is a whitespace-collapsed snippet
    of the surrounding text, for use in advisory/log output.
    """

    token: str
    attribution: str
    span: str


# A specific code-level token: a `metadata.<snake_case>` reference, a
# backtick-quoted dotted/snake_case identifier, or an `<ident>=<value>`
# stamp form. Exactly one of the three named groups captures per match.
_CODE_TOKEN_PATTERN = re.compile(
    r"""
    metadata\.(?P<meta>[A-Za-z_][A-Za-z0-9_]*)
    |
    `(?P<tick>[A-Za-z_][A-Za-z0-9_.]*)`
    |
    \b(?P<stamp>[A-Za-z_][A-Za-z0-9_]*)=(?:true|false|[A-Za-z0-9_]+)\b
    """,
    re.VERBOSE,
)

# An attribution anchor: a citation of specific prior work. Requires a
# NUMBER (task/ACTION) or a plausible commit sha — a bare "task" or "ACTION"
# mention with no identifying number is never treated as an anchor, which is
# what keeps prospective new-work specs ("add a new task for X") from being
# mistaken for retrospective claims about completed work.
_ATTRIBUTION_PATTERN = re.compile(
    r"""
    \bper\s+task\s+\d+\b
    |
    \btask\s+\d+\b
    |
    \bACTION\s*\#\d+\b
    |
    \bcommit\s+[0-9a-f]{7,40}\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Character radius scanned on each side of a code-token match for a
# co-occurring attribution anchor. Deliberately small (roughly sentence-
# scale) — the anchor requirement is what separates a retrospective
# false-fact claim from a prospective new-work spec, so the window must not
# be so wide that it picks up an unrelated task/commit mention elsewhere in
# a long candidate description.
_CO_OCCURRENCE_WINDOW_CHARS = 100


def extract_attributed_claims(text: str) -> list[AttributedClaim]:
    """Extract code-level claims *text* attributes to a completed task/commit/ACTION.

    Emits a claim only when a specific CODE TOKEN co-occurs, within
    :data:`_CO_OCCURRENCE_WINDOW_CHARS` characters, with an ATTRIBUTION
    ANCHOR. A code token with no nearby anchor is a prospective new-work
    spec (e.g. "add a `metadata.foo` field") and is never flagged — only
    the combination signals a retrospective fact claim about prior work.

    Deduplicates by token, keeping the FIRST occurrence that has a
    co-occurring anchor (an earlier anchor-less mention of the same token
    does not suppress a later attributed one). Order is first-seen.

    Pure — no I/O, no verification (see ``verify_attributed_claims`` for
    that). Never raises; empty/whitespace/plain prose text returns ``[]``.
    """
    if not text:
        return []

    claims: list[AttributedClaim] = []
    seen_tokens: set[str] = set()

    for match in _CODE_TOKEN_PATTERN.finditer(text):
        token = match.group('meta') or match.group('tick') or match.group('stamp')
        if not token or token in seen_tokens:
            continue

        window_start = max(0, match.start() - _CO_OCCURRENCE_WINDOW_CHARS)
        window_end = min(len(text), match.end() + _CO_OCCURRENCE_WINDOW_CHARS)
        window = text[window_start:window_end]

        anchors: list[str] = []
        for anchor_match in _ATTRIBUTION_PATTERN.finditer(window):
            anchor_text = anchor_match.group(0)
            if anchor_text not in anchors:
                anchors.append(anchor_text)
        if not anchors:
            continue

        seen_tokens.add(token)
        claims.append(
            AttributedClaim(
                token=token,
                attribution=', '.join(anchors),
                span=' '.join(window.split()),
            )
        )

    return claims


def verify_attributed_claims(
    claims: list[AttributedClaim], probe: Callable[[str], bool],
) -> list[AttributedClaim]:
    """Return the subset of *claims* whose token *probe* reports ABSENT.

    ``probe(token) -> bool`` is the injectable verification seam: it should
    return ``True`` when the token is present somewhere relevant (e.g. the
    live source tree or git history) and ``False`` when it is absent. Only
    tokens the probe reports absent are returned — a claim whose token the
    probe confirms present is filtered out (self-correcting: a legitimate
    reference like ``get_statuses`` verifies clean regardless of how loosely
    it was extracted).

    Pure orchestration over the injected *probe* — no I/O of its own. Never
    raises on its own; a *probe* that raises propagates to the caller.
    """
    return [claim for claim in claims if not probe(claim.token)]


def unverified_claims_in_text(
    text: str, probe: Callable[[str], bool],
) -> list[AttributedClaim]:
    """Extract attributed claims from *text* and return the unverified subset.

    Composes :func:`extract_attributed_claims` and
    :func:`verify_attributed_claims` — mirrors
    ``recon_code_fix_premise_guard.premise_refuted_entry``'s
    detect-then-verify composition shape. Short-circuits to ``[]`` (without
    calling *probe*) when *text* carries no attributed claims at all.
    """
    claims = extract_attributed_claims(text)
    if not claims:
        return []
    return verify_attributed_claims(claims, probe)


# Wall-clock budget for each individual git subprocess call. Generous enough
# for a large repo's grep/pickaxe, but bounded so a wedged git process can
# never hang the caller — a timeout is treated the same as any other git
# error and fails open (see probe() below).
_GIT_PROBE_TIMEOUT_SECS = 10.0

# Pathspecs excluded from BOTH the tree grep and the history pickaxe, so a
# match confined to tests/docs/prompts never counts as "present". Without
# this, a token this very module's own tests/docstrings/prompt guardrail use
# as an EXAMPLE (e.g. the task-2433 incident's `done_provenance_invalidated`)
# would verify "present" the moment those example-carrying files merge —
# silently defeating detection of a genuine future re-fabrication of the
# same token. `glob` magic is required for `**` to cross directory
# boundaries (plain pathspec globs do not span `/`).
_EXCLUDED_PATHSPECS: tuple[str, ...] = (
    ':(exclude,glob)**/tests/**',
    ':(exclude,glob)**/docs/**',
    ':(exclude,glob)**/prompts/**',
)


def _resolve_git_toplevel(repo_root: Path) -> Path:
    """Resolve *repo_root* to its git working-tree top level, if possible.

    ``git grep`` (unlike ``git log --all``) scopes matches to the invocation
    cwd's subtree, not the whole repository — so if *repo_root* is itself a
    package subdirectory rather than the repo root, the two probes below
    would search inconsistent scopes (grep: subtree only; log: everything).
    Probing from the resolved top level keeps both calls' scope consistent
    regardless of what *repo_root* points at.

    Falls back to *repo_root* unchanged on any resolution failure — the
    subsequent grep/log calls already fail open on a git error, so an
    unresolvable root just degrades to the prior cwd-scoped behaviour rather
    than raising. Never raises.
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=repo_root,
            timeout=_GIT_PROBE_TIMEOUT_SECS,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return repo_root
    if result.returncode != 0:
        return repo_root
    toplevel = result.stdout.strip()
    return Path(toplevel) if toplevel else repo_root


def make_source_and_history_probe(repo_root: Path) -> Callable[[str], bool]:
    """Build a ``probe(token) -> bool`` backed by *repo_root*'s live tree + git history.

    Encodes the architect's exact manual refutation check for the task-2433
    incident: ``git grep`` against the working tree, falling back on a tree
    miss to ``git log --all -S<token>`` (the pickaxe) against the FULL
    history. A token found in EITHER is treated as present (``True``) — this
    avoids false-flagging a token that was legitimately REMOVED from the
    tree but still exists in history (e.g. "task N removed X").

    Both the tree grep and the history pickaxe are rooted at *repo_root*'s
    resolved git top level (see :func:`_resolve_git_toplevel`) and exclude
    ``tests/``, ``docs/``, and ``prompts/`` paths (see
    :data:`_EXCLUDED_PATHSPECS`) — a match confined to those paths (e.g. this
    guard's own example tokens in its tests/docstrings) never counts as
    "present".

    Fails OPEN (returns ``True``, i.e. "assume present") on ANY git error —
    a non-zero/non-"no match" exit code, a timeout, or the git binary being
    unavailable — so an infrastructure hiccup (or *repo_root* not being a
    git repository at all) can never itself produce a false "fabrication"
    flag. Never raises.
    """
    resolved_root = _resolve_git_toplevel(repo_root)

    def probe(token: str) -> bool:
        try:
            grep_result = subprocess.run(
                ['git', 'grep', '--quiet', '--fixed-strings', '--',
                 token, *_EXCLUDED_PATHSPECS],
                cwd=resolved_root,
                timeout=_GIT_PROBE_TIMEOUT_SECS,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                'recon_claim_verification_guard: git grep failed for token=%r '
                'under %s — probe fails open: %s',
                token, repo_root, exc,
            )
            return True

        if grep_result.returncode == 0:
            return True
        if grep_result.returncode != 1:
            # Not a clean "no match" (1) — an actual git/infra error (e.g.
            # repo_root is not a git repository). Fail open rather than
            # treat this as a genuine miss.
            logger.warning(
                'recon_claim_verification_guard: git grep errored (exit=%d) for '
                'token=%r under %s — probe fails open: %s',
                grep_result.returncode, token, repo_root, grep_result.stderr.strip(),
            )
            return True

        try:
            log_result = subprocess.run(
                ['git', 'log', '--all', f'-S{token}', '--oneline', '-1',
                 '--', *_EXCLUDED_PATHSPECS],
                cwd=resolved_root,
                timeout=_GIT_PROBE_TIMEOUT_SECS,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                'recon_claim_verification_guard: git log failed for token=%r '
                'under %s — probe fails open: %s',
                token, repo_root, exc,
            )
            return True

        if log_result.returncode != 0:
            logger.warning(
                'recon_claim_verification_guard: git log errored (exit=%d) for '
                'token=%r under %s — probe fails open: %s',
                log_result.returncode, token, repo_root, log_result.stderr.strip(),
            )
            return True

        return bool(log_result.stdout.strip())

    return probe
