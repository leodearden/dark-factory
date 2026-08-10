"""Repo-wide gate: every ``TaskConfigDir`` site carries an archival disposition.

Task 3271 root cause. ``archive_task_transcripts`` has exactly TWO non-test
call sites, against an unbounded and growing number of ``TaskConfigDir``
construction sites — and nothing asserted that ratio. A per-investigation
config dir added to ``dry_run_unblock`` (commit 7a07c40820) was therefore
``rmtree``'d with its transcript intact and nobody noticed, because the
teardown backstop composes ``f'claude-config-{branch}'`` literally
(git_ops.py:11780) and silently never matched the ``-unblock`` suffix.

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
from config_dir_archival_allowlist import AUDITED_SITES, DISPOSITIONS
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


def _scan_task_config_dir_sites(repo_root: Path) -> list[_SiteKey]:
    """Return a site key per ``TaskConfigDir(...)`` construction in first-party source.

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
    for py_file in iter_first_party_files(repo_root):
        tree = ast.parse(py_file.read_text(encoding='utf-8'), filename=str(py_file))
        parent_map = _build_parent_map(tree)
        for node in ast.walk(tree):
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
            sites.append((
                py_file.relative_to(repo_root).as_posix(),
                _compute_qualname(node, parent_map),
            ))
    return sites


@pytest.fixture(scope='module')
def scanned_sites() -> list[_SiteKey]:
    return _scan_task_config_dir_sites(_REPO_ROOT)


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
        """Each record must actually carry the audit it claims to."""
        for entry in AUDITED_SITES:
            key = (entry.get('path'), entry.get('qualname'))
            assert entry.get('disposition') in DISPOSITIONS, (
                f'{key}: disposition must be one of {sorted(DISPOSITIONS)}, '
                f'got {entry.get("disposition")!r}'
            )
            for field in ('destroyed_by', 'rationale'):
                assert entry.get(field), (
                    f'{key}: non-empty {field!r} required — an entry without it '
                    f'records nothing and defeats the audit.'
                )
            if entry['disposition'] == 'UNARCHIVED_GAP':
                assert entry.get('follow_up'), (
                    f'{key}: an UNARCHIVED_GAP must name its follow-up, or the '
                    f'gap is just an accepted loss with extra steps.'
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

    def test_dry_run_unblock_disposition_is_archived(self):
        """Task 3271's fix is pinned in the audit record, not only in behaviour.

        A revert of the producer hook in run_dry_run_unblock's finally must fail
        HERE as well as in test_dry_run_unblock.py — the record and the code
        cannot silently disagree about whether that site archives.
        """
        matches = [
            e for e in AUDITED_SITES
            if e['path'] == 'orchestrator/src/orchestrator/dry_run_unblock.py'
            and e['qualname'] == 'run_dry_run_unblock'
        ]
        assert len(matches) == 1, (
            f'Expected exactly one audited dry_run_unblock site, got {matches}'
        )
        assert matches[0]['disposition'] == 'ARCHIVED', (
            'run_dry_run_unblock archives its per-investigation config dir '
            '(task 3271). If that hook was removed, restore it rather than '
            'downgrading this record: the cleanup_worktree backstop composes '
            "f'claude-config-{branch}' literally and can never cover a "
            "'-unblock' dir, so nothing else archives that transcript."
        )
