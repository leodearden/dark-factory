"""Repo-wide gate: every ``TaskConfigDir`` site carries an archival disposition.

Task 3271 root cause, stated as it stood then: ``archive_task_transcripts`` had
exactly TWO non-test call sites, against an unbounded and growing number of
``TaskConfigDir`` construction sites — and nothing asserted that ratio. (Both
numbers have since moved; see :data:`_ARCHIVAL_NAMES` for the archival entry
points this gate tracks today. The ratio argument is what generalises, not the
counts.) A per-investigation
config dir added to ``dry_run_unblock`` (commit 7a07c40820) was therefore
``rmtree``'d with its transcript intact and nobody noticed, because the
teardown backstop in ``GitOps.cleanup_worktree`` composes
``f'claude-config-{branch}'`` literally and silently never matched the
``-unblock`` suffix.

Fixing that one site does not stop the next one. This gate makes the audit
DURABLE: each construction site must appear in
``config_dir_archival_allowlist.AUDITED_SITES`` with a recorded disposition
(``ARCHIVED`` / ``UNARCHIVED_BY_DESIGN`` / ``UNARCHIVED_GAP``) and a rationale,
so an eighth site cannot be added without someone deciding — on the record —
whether its transcripts are worth keeping.

Mirrors the established triad in this directory: ``silent_fallthrough_scan``
(scanner) + ``silent_fallthrough_allowlist`` (pure-data baseline record) +
``test_silent_fallthrough_gate`` (assertions). Nothing new is invented.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest
from config_dir_archival_allowlist import ARCHIVED, AUDITED_SITES, DISPOSITIONS
from silent_fallthrough_scan import (
    _build_parent_map,
    _compute_qualname,
    iter_first_party_files,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Site keys omit lineno deliberately: a line number drifts on every unrelated
# edit above the site, which would make this gate a nuisance ratchet that
# reviewers learn to re-bless blindly. (path, qualname) changes only when the
# site genuinely moves between functions.
_SiteKey = tuple[str, str]

# The archival entry points, as NAMES. Both count: a site that reaches either
# one has kept its transcripts, which is the only thing an ARCHIVED
# disposition claims.
#
# ``archive_before_delete`` joined the set in task 3619, which made archival a
# PRECONDITION of the config-dir delete rather than a step before it. Recording
# only the older name would have made this gate read the new, stronger teardown
# as an archival REGRESSION — the exact false alarm that tempts a reviewer to
# downgrade a disposition and lose the transcripts for real.
_ARCHIVAL_NAMES = frozenset({'archive_task_transcripts', 'archive_before_delete'})


def _scan(repo_root: Path) -> tuple[list[_SiteKey], list[_SiteKey]]:
    """Return ``(construction_sites, archival_reference_sites)``.

    One AST walk yields both halves of the audit:

    * every ``TaskConfigDir(...)`` construction — the thing that must carry a
      recorded disposition;
    * every reference to one of the :data:`_ARCHIVAL_NAMES` — the thing that
      makes an ``ARCHIVED`` disposition TRUE rather than merely asserted.

    The second is deliberately a NAME scan, not a call scan, and stays one
    across both call shapes this repo has used. Before task 3619 the producers
    passed the archiver as a VALUE to ``asyncio.to_thread(...)``, which
    ``ast.Call(func=Name)`` would have matched in none of them; 3619 collapsed
    the teardown sites to direct synchronous calls, which a call scan WOULD
    match. A name scan is correct under either, so the gate does not silently
    go blind the next time a call site changes shape. A
    ``from ... import archive_before_delete`` produces an ``ast.alias``, not an
    ``ast.Name``, so module-level imports do not contaminate the result —
    which is what lets the gate attribute archival to the enclosing FUNCTION
    rather than the module. Inside workflow.py, ``TaskWorkflow._invoke`` and
    ``TaskWorkflow._archive_then_cleanup_config_dir`` name an archiver and are
    reported; ``TaskWorkflow._cleanup_config_dir`` and
    ``TaskWorkflow._recycle_config_dir`` reach archival only by CALLING that
    helper, so they are not.

    That indirection is a real limit, not an oversight, and it is why an
    ``archives_in`` names the helper rather than its callers: the scan sees
    name references, not a call graph, so archival reached through a
    ``self._helper()`` hop is credited where the archiver is actually named.
    The gate therefore proves a hook EXISTS and is reachable from the recorded
    site's teardown path; it does not prove every path into that helper is
    covered. Tests of the call sites themselves carry that half.

    ``iter_first_party_files`` already encodes the 7 scope roots, the
    ``tests``/``mem0``/``graphiti``/``conftest.py`` exclusions, and a
    sentinel-dir validation of *repo_root* that RAISES rather than yielding a
    vacuously-empty scan.

    Read/parse errors are deliberately NOT swallowed: a first-party source file
    that cannot be read or parsed is a real breakage, and silently skipping it
    would let a construction site hide behind it — the exact silent-degradation
    shape this repo's gates exist to prevent.
    """
    sites: list[_SiteKey] = []
    archival_refs: list[_SiteKey] = []
    for py_file in iter_first_party_files(repo_root):
        tree = ast.parse(py_file.read_text(encoding='utf-8'), filename=str(py_file))
        parent_map = _build_parent_map(tree)
        rel = py_file.relative_to(repo_root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _ARCHIVAL_NAMES:
                archival_refs.append((rel, _compute_qualname(node, parent_map)))
                continue
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                callee = func.id
            elif isinstance(func, ast.Attribute):
                callee = func.attr
            else:
                continue
            if callee != 'TaskConfigDir':
                continue
            sites.append((rel, _compute_qualname(node, parent_map)))
    return sites, archival_refs


@pytest.fixture(scope='module')
def scan_result() -> tuple[list[_SiteKey], list[_SiteKey]]:
    return _scan(_REPO_ROOT)


@pytest.fixture(scope='module')
def scanned_sites(scan_result) -> list[_SiteKey]:
    return scan_result[0]


@pytest.fixture(scope='module')
def archival_ref_sites(scan_result) -> set[_SiteKey]:
    return set(scan_result[1])


def _allowlist_keys() -> list[_SiteKey]:
    return [(e['path'], e['qualname']) for e in AUDITED_SITES]


class TestEverySiteIsAudited:
    """The ratchet: scanned sites and audited sites must correspond exactly."""

    def test_every_construction_site_is_audited(self, scanned_sites):
        """A new TaskConfigDir site must record its archival disposition."""
        # Counter, not set: two byte-distinct constructions can share one
        # (path, qualname) key — UsageGate.__init__ legitimately has two. Set
        # membership would let a THIRD site added to an already-listed function
        # pass silently, which is precisely the hole this gate exists to close.
        missing = Counter(scanned_sites) - Counter(_allowlist_keys())
        assert not missing, (
            'Unaudited TaskConfigDir construction site(s):\n'
            + '\n'.join(
                f'  {path}  ::  {qualname}  (x{count})'
                for (path, qualname), count in sorted(missing.items())
            )
            + '\n\nA TaskConfigDir holds an agent CLAUDE_CONFIG_DIR, so whatever '
              'destroys it destroys that agent session\'s transcripts with it. '
              'Record this site in shared/tests/config_dir_archival_allowlist.py '
              'with what destroys the dir and a disposition of ARCHIVED / '
              'UNARCHIVED_BY_DESIGN / UNARCHIVED_GAP plus a rationale. Decide '
              'whether the transcripts are worth keeping — do not silence the gate.'
        )

    def test_no_stale_allowlist_entries(self, scanned_sites):
        """A removed site must not leave a lying record behind."""
        stale = Counter(_allowlist_keys()) - Counter(scanned_sites)
        assert not stale, (
            'Stale config_dir_archival_allowlist entries — no TaskConfigDir '
            'construction found at:\n'
            + '\n'.join(
                f'  {path}  ::  {qualname}  (x{count})'
                for (path, qualname), count in sorted(stale.items())
            )
            + '\n\nThe site moved or was removed; update the entry rather than '
              'leaving a record that documents code which no longer exists.'
        )

    def test_every_entry_is_well_formed(self):
        """The schema contract the AST cross-check consumes.

        Two checks only, both about SHAPE rather than wording: ``disposition``
        is one of the known values, and ``archives_in`` is present exactly for
        ``ARCHIVED`` entries. The second is the load-bearing one —
        :class:`TestDispositionsMatchTheCode` re-derives ``archives_in`` from
        the AST, so a malformed or missing field there would silently narrow
        the falsifiable half of the gate.

        The free-text fields (``destroyed_by``, ``rationale``, ``follow_up``)
        are deliberately NOT asserted: a non-empty check on prose passes on any
        string and constrains no code.
        """
        for entry in AUDITED_SITES:
            key = (entry.get('path'), entry.get('qualname'))
            assert entry.get('disposition') in DISPOSITIONS, (
                f'{key}: disposition must be one of {sorted(DISPOSITIONS)}, '
                f'got {entry.get("disposition")!r}'
            )
            if entry['disposition'] == ARCHIVED:
                assert entry.get('archives_in'), (
                    f'{key}: an ARCHIVED entry must name where the archival '
                    f'actually happens, as a tuple of (path, qualname) pairs. '
                    f'Without it the disposition is an unfalsifiable claim.'
                )
            else:
                assert 'archives_in' not in entry, (
                    f'{key}: archives_in is meaningful only for ARCHIVED; a '
                    f'{entry["disposition"]} entry that names an archival site '
                    f'is self-contradictory.'
                )


class TestGateSelfIntegrity:
    """The gate must not pass vacuously."""

    def test_scan_finds_the_known_sites(self, scanned_sites):
        """Floor guard: an import/glob regression that finds nothing must FAIL.

        Mirrors test_silent_fallthrough_gate.py's "is repo_root correct?" guard.
        Seven sites exist today; the floor sits at six so ordinary churn does
        not trip it while a collapsed scan still does.
        """
        assert len(scanned_sites) >= 6, (
            f'Only {len(scanned_sites)} TaskConfigDir sites found — is repo_root '
            f'correct? ({_REPO_ROOT})'
        )


class TestDispositionsMatchTheCode:
    """The record must be falsifiable, not merely self-consistent.

    Comparing two constants committed in the same diff proves nothing: deleting
    a producer hook would leave an ``ARCHIVED`` label sitting there, green and
    lying. These tests re-derive from the AST which functions actually reference
    one of the :data:`_ARCHIVAL_NAMES` and cross-check both directions, so the
    record and the code cannot silently disagree.
    """

    def test_archived_entries_really_archive(self, archival_ref_sites):
        """Every ARCHIVED entry's declared archival site must exist in the code."""
        broken: list[str] = []
        for entry in AUDITED_SITES:
            if entry['disposition'] != ARCHIVED:
                continue
            for path, qualname in entry.get('archives_in', ()):
                if (path, qualname) not in archival_ref_sites:
                    broken.append(
                        f'  {entry["path"]} :: {entry["qualname"]}  claims '
                        f'ARCHIVED via {path} :: {qualname} — but no reference '
                        f'to any of {sorted(_ARCHIVAL_NAMES)} was found there'
                    )
        assert not broken, (
            'Audit record disagrees with the code — an ARCHIVED disposition '
            'whose archival hook is gone:\n' + '\n'.join(broken)
            + '\n\nEither the hook was removed (restore it — nothing else '
              'archives these transcripts) or it moved (update archives_in). '
              'Do NOT downgrade the disposition to make this pass without '
              'first checking whether the transcripts are still being kept.\n'
              f'Archival references found in the tree: {sorted(archival_ref_sites)}'
        )

    def test_unarchived_entries_really_do_not_archive(self, archival_ref_sites):
        """An UNARCHIVED_* site whose own function archives is a stale record.

        Checked against the enclosing FUNCTION, not the module. workflow.py
        holds four of the audited qualnames and only two of them name an
        archiver (``TaskWorkflow._invoke`` and
        ``TaskWorkflow._archive_then_cleanup_config_dir``), so a module-level
        check would credit every config dir built anywhere in that file with
        archival that only some of its functions perform — laundering a real
        gap into a false pass.
        """
        contradictions = [
            f'  {e["path"]} :: {e["qualname"]}  is recorded {e["disposition"]}'
            for e in AUDITED_SITES
            if e['disposition'] != ARCHIVED
            and (e['path'], e['qualname']) in archival_ref_sites
        ]
        assert not contradictions, (
            'Audit record disagrees with the code — a site recorded as NOT '
            f'archiving now references one of {sorted(_ARCHIVAL_NAMES)}:\n'
            + '\n'.join(contradictions)
            + '\n\nIf the gap was closed, promote the entry to ARCHIVED and give '
              'it an archives_in; if not, the reference needs explaining.'
        )

    def test_archival_reference_scan_is_not_vacuous(self, archival_ref_sites):
        """Floor guard for the second scan, mirroring the site-count floor.

        A Name-scan that silently found nothing would make
        ``test_unarchived_entries_really_do_not_archive`` pass vacuously. (The
        ARCHIVED direction self-guards — it fails loudly on an empty scan — but
        the UNARCHIVED direction does not, so it needs this.)
        """
        assert len(archival_ref_sites) >= 5, (
            f'Only {len(archival_ref_sites)} archival reference site(s) found: '
            f'{sorted(archival_ref_sites)}. Five are known — '
            f'TaskWorkflow._invoke and run_dry_run_unblock (producer hooks), '
            f'GitOps.cleanup_worktree and '
            f'TaskWorkflow._archive_then_cleanup_config_dir (teardown, task '
            f'3619), and Harness._sweep_orphaned_transcripts (the boot-time '
            f'SIGKILL-tail sweeper, task 3619) — is repo_root correct? '
            f'({_REPO_ROOT})'
        )
