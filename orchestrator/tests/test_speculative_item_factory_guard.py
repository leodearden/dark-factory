"""I3 factory grep-guard for SpeculativeItem construction sites (task 1990).

The guard enforces PRD invariant I3 ("pipeline items are constructed only at
whitelisted factory sites; all rebuilds go through ``dataclasses.replace``") as
a structural inventory tripwire: it git-greps ``orchestrator/src/`` for real
construction-site tokens and asserts the per-file inventory matches a pinned
allowlist. A NEW construction — in particular a CAS-loop hand-rebuild that
reconstructs the item field-by-field (the task-1928 dropped-field bug class)
instead of calling ``dataclasses.replace(item, ...)`` — changes the inventory
and fails the build, forcing a deliberate review.

This is deliberately NOT a comment-marker check: it cannot be satisfied by
annotating a construction line, and it locks no wording. Shape correctness is
guarded independently by the RealMergeItem/DecidedItem dataclasses themselves
(illegal cross-variant fields are unrepresentable, task ο); this guard covers
the orthogonal "no un-vetted new construction site" property that a
dataclass's own shape cannot see (a hand-rebuild can drop ``base_sha`` and
still have a valid shape). Task ο (2000) retired the SpeculativeItem product
dataclass in favor of a ``RealMergeItem | DecidedItem`` union — SpeculativeItem
is now a non-constructible type alias, so the inventory tracks the two union
constructors instead and separately asserts zero ``SpeculativeItem(``
constructions remain.

The sibling ``test_no_tracked_editor_backup_files_under_orchestrator`` is a
repo-hygiene guard: a WIP-snapshot commit once force-added a ~4193-line
``*.py.tmp.*`` near-duplicate; a ``.gitignore`` rule alone does not catch an
already-tracked or force-added file, so this test scans the tracked-file list.
"""

from __future__ import annotations

import fnmatch
import subprocess
from collections import Counter
from pathlib import Path

import pytest

_SCOPED_PATH = 'orchestrator/src/orchestrator/'
_BACKUP_GLOBS = ('*.tmp.*', '*.py.tmp.*', '*.orig', '*.rej', '*.bak', '*.swp', '*~')

# Pinned inventory of the whitelisted RealMergeItem / DecidedItem factory
# sites, keyed by repo-relative file. All live in the merger/verifier factory
# helpers of merge_queue.py; CAS-loop rebuilds go through
# ``dataclasses.replace(item, base_sha=...)`` and do not construct at all.
# Bump a count (or add a file) ONLY for a genuine new factory site — never to
# silence a hand-rebuild that should be ``dataclasses.replace`` instead.
#
# Task ο (2000) split the prior 9-site ``SpeculativeItem(`` inventory into its
# REAL (3: real merge success in _merger_loop / _remerge) and DECIDED (6:
# conflict/already_merged/train-handoff/retry-outcome in _merger_loop /
# classify_and_merge / _remerge) constructors — same 9 call sites, reframed
# under the two union constructors that replaced the single product type.
#
# Deliberately a per-file *count*, not just a set of filenames: every
# whitelisted site already lives in this one file, so a bare set-of-files
# check would be permanently satisfied and blind to a new construction
# landing in that same file — including a reintroduced hand-rebuild, which is
# exactly the task-1928 failure mode this guard exists to catch. The
# RealMergeItem/DecidedItem dataclasses validate *shape* (unrepresentability)
# but not *provenance*: a hand-rebuild can drop a field that isn't part of
# the type shape (e.g. counts_against_cap) while still producing a
# shape-valid item, so it would sail through construction undetected. The
# count is what makes that rebuild visible; it is expected to require a
# deliberate bump for genuine new sites, not something to relax away.
_EXPECTED_REAL_MERGE_ITEM_SITES = {
    'orchestrator/src/orchestrator/merge_queue.py': 3,
}
_EXPECTED_DECIDED_ITEM_SITES = {
    'orchestrator/src/orchestrator/merge_queue.py': 6,
}


def _strip_line_comment(code: str) -> str:
    """Return ``code`` with a trailing ``#`` comment removed.

    Tracks single/double-quote state so a ``#`` inside a string literal
    (e.g. an f-string log message built before a real call on the same
    line) is not mistaken for a comment marker — unlike a naive
    ``code.split('#', 1)[0]``, which would truncate at that in-string ``#``
    and silently drop a real construction from the inventory.
    """
    quote: str | None = None
    for i, ch in enumerate(code):
        if quote is not None:
            if ch == quote and code[i - 1] != '\\':
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == '#':
            return code[:i]
    return code


def test_strip_line_comment_handles_hash_inside_string_literal() -> None:
    """Regression for the naive ``code.split('#', 1)[0]`` heuristic this
    replaced: a ``#`` inside a string literal must not truncate the line
    before a real construction later on it (that heuristic would have
    silently dropped the construction from the inventory)."""
    line = 'logger.info("issue #42"); x = SpeculativeItem(request=req)  # spec-factory'
    stripped = _strip_line_comment(line)
    assert 'SpeculativeItem(' in stripped
    assert '# spec-factory' not in stripped


def test_strip_line_comment_strips_plain_trailing_comment() -> None:
    assert _strip_line_comment('foo()  # a comment') == 'foo()  '


def test_strip_line_comment_passthrough_when_no_comment() -> None:
    assert _strip_line_comment('foo(bar, baz)') == 'foo(bar, baz)'


def _is_match_case_pattern(code: str) -> bool:
    """Return True if ``code`` is a ``case Token():`` structural-pattern arm.

    A ``match``/``case`` pattern like ``case RealMergeItem():`` (see
    ``item_merge_wt``'s exhaustive match) tests the value's type — it calls no
    constructor at runtime — so it must not be miscounted as a construction
    site alongside real ``Token(...)`` calls.
    """
    return code.lstrip().startswith('case ')


def _real_construction_hits(repo_root: Path, token: str) -> list[str]:
    """Return ``path:lineno:code`` git-grep hits that are actual ``token(``
    constructions, excluding lines where the token only appears inside a
    trailing comment (so a comment/docstring mention is not miscounted as a
    construction) or as a ``case token():`` structural-pattern match arm."""
    result = subprocess.run(
        ['git', 'grep', '-n', f'{token}(', '--', _SCOPED_PATH],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f'git grep failed (exit {result.returncode}): {result.stderr}')

    hits: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # git grep -n → "path:lineno:code"
        try:
            path, lineno, code = line.split(':', 2)
        except ValueError:
            continue
        # Drop pure-comment mentions: keep only hits where the construction
        # token precedes any '#'. Production constructor lines have no '#'
        # before the call; a comment like "# build a token(...)" does.
        code_before_comment = _strip_line_comment(code)
        if _is_match_case_pattern(code_before_comment):
            continue
        if f'{token}(' in code_before_comment:
            hits.append(f'{path}:{lineno}:{code.strip()}')
    return hits


def test_real_merge_item_constructed_only_at_whitelisted_factory_sites(
    repo_root: Path | None,
) -> None:
    """The per-file inventory of ``RealMergeItem(`` constructions under
    orchestrator/src/ must match the pinned whitelist.

    Guards I3: a future hand-rebuild that bypasses ``dataclasses.replace(...)``
    and reconstructs a RealMergeItem field-by-field (the task-1928 bug class)
    appears as an inventory delta and fails here — forcing the author to either
    use ``dataclasses.replace`` or deliberately extend the whitelist.
    """
    if repo_root is None:
        pytest.skip('not running inside a git checkout')

    hits = _real_construction_hits(repo_root, 'RealMergeItem')
    inventory = dict(Counter(hit.split(':', 1)[0] for hit in hits))

    assert inventory == _EXPECTED_REAL_MERGE_ITEM_SITES, (
        'RealMergeItem construction inventory drifted from the pinned factory '
        f'whitelist (I3).\n  expected: {_EXPECTED_REAL_MERGE_ITEM_SITES}\n  actual:   {inventory}\n'
        'A new/removed construction site changed this. If you added a CAS-loop or '
        'other rebuild, use dataclasses.replace(item, ...) so no field is silently '
        'dropped. If this is a genuine new factory site, update '
        '_EXPECTED_REAL_MERGE_ITEM_SITES above.\nConstruction sites found:\n'
        + '\n'.join(hits)
    )


def test_decided_item_constructed_only_at_whitelisted_factory_sites(
    repo_root: Path | None,
) -> None:
    """The per-file inventory of ``DecidedItem(`` constructions under
    orchestrator/src/ must match the pinned whitelist.

    Guards I3: a future hand-rebuild that bypasses ``dataclasses.replace(...)``
    and reconstructs a DecidedItem field-by-field (the task-1928 bug class)
    appears as an inventory delta and fails here — forcing the author to either
    use ``dataclasses.replace`` or deliberately extend the whitelist.
    """
    if repo_root is None:
        pytest.skip('not running inside a git checkout')

    hits = _real_construction_hits(repo_root, 'DecidedItem')
    inventory = dict(Counter(hit.split(':', 1)[0] for hit in hits))

    assert inventory == _EXPECTED_DECIDED_ITEM_SITES, (
        'DecidedItem construction inventory drifted from the pinned factory '
        f'whitelist (I3).\n  expected: {_EXPECTED_DECIDED_ITEM_SITES}\n  actual:   {inventory}\n'
        'A new/removed construction site changed this. If you added a CAS-loop or '
        'other rebuild, use dataclasses.replace(item, ...) so no field is silently '
        'dropped. If this is a genuine new factory site, update '
        '_EXPECTED_DECIDED_ITEM_SITES above.\nConstruction sites found:\n'
        + '\n'.join(hits)
    )


def test_no_speculative_item_constructions_remain_in_src(repo_root: Path | None) -> None:
    """Task ο retired the SpeculativeItem product dataclass in favor of the
    RealMergeItem/DecidedItem union: zero real ``SpeculativeItem(``
    constructions may remain under orchestrator/src/ — SpeculativeItem is now
    a non-constructible ``RealMergeItem | DecidedItem`` union type alias."""
    if repo_root is None:
        pytest.skip('not running inside a git checkout')

    hits = _real_construction_hits(repo_root, 'SpeculativeItem')
    assert not hits, (
        'Found SpeculativeItem( construction(s) in src; SpeculativeItem is now a '
        'RealMergeItem | DecidedItem union alias and must not be constructed directly:\n'
        + '\n'.join(hits)
    )


def test_no_tracked_editor_backup_files_under_orchestrator(repo_root: Path | None) -> None:
    """No editor/tool backup file should ever be git-tracked under orchestrator/.

    A `.gitignore` rule alone only stops an *untracked* backup from being
    casually `git add`ed — it does nothing for a file that is already
    tracked or was force-added (`git add -f`), which is exactly how
    ``test_merge_queue_concurrent_verify.py.tmp.2390297.d4b2fa0bada8`` (a
    ~4193-line stale near-duplicate) slipped in via a WIP-snapshot commit.
    This test scans the tracked file list itself so re-adding a similar
    backup file is caught regardless of .gitignore.
    """
    if repo_root is None:
        pytest.skip('not running inside a git checkout')

    result = subprocess.run(
        ['git', 'ls-files', '--', 'orchestrator/'],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f'git ls-files failed (exit {result.returncode}): {result.stderr}')

    tracked = [ln for ln in result.stdout.splitlines() if ln.strip()]
    offenders = [
        path
        for path in tracked
        if any(fnmatch.fnmatch(Path(path).name, pattern) for pattern in _BACKUP_GLOBS)
    ]
    assert not offenders, (
        f'Found {len(offenders)} tracked editor/tool backup file(s) under orchestrator/ '
        f'matching {_BACKUP_GLOBS!r} (these must never be committed):\n' + '\n'.join(offenders)
    )
