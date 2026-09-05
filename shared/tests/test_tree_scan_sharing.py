"""The first-party tree is read and parsed ONCE per session, and shared.

WHY THIS FILE EXISTS (task 4520). Two gate modules each grew their own
whole-first-party-tree scan: ``test_silent_fallthrough_gate.tree_scan_data``
and ``test_config_dir_archival_gate._scan``. Neither was wrong on its own, but
together they read the same 461 files twice, parsed them three times, and built
922 AST parent maps where 24 suffice — and because ``pytest-timeout`` arms its
timer over the WHOLE runtest protocol (``func_only=False``), that duplicated
work is charged to whichever single test item happens to trigger the fixture.
Measured at 23.68s of setup against a 60s budget: 2.5x headroom, which a loaded
host consumes, producing a burst of ERROR-at-setup that looks like flake and is
actually arithmetic.

THE CONTRACT PINNED HERE. ``silent_fallthrough_scan.parse_first_party_tree`` is
the ONE place the first-party tree is read and parsed; both gates walk the ASTs
it hands out. Every assertion in this file counts WORK (``ast.parse`` calls,
``Path.read_text`` calls, parent maps built) rather than wall-clock time — a
timing assertion on a host whose load average swung between 70 and 289 during
the investigation would be the exact flaky shape this task exists to remove.

Mirrors this directory's established gate idiom (``silent_fallthrough_scan`` +
``silent_fallthrough_allowlist`` + ``test_silent_fallthrough_gate``;
``test_safe_io.TestNoRegrownAtomicWriters``): scan, name the offender, and
carry a floor so the guard cannot pass vacuously.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest
import silent_fallthrough_scan as sfs
from silent_fallthrough_scan import iter_first_party_files

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers — a fabricated repo root, so the provider's contract can be pinned
# without re-reading (or evicting the memo for) the real 461-file tree.
# ---------------------------------------------------------------------------


def _make_fake_repo(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    """Build a minimal repo root that satisfies iter_first_party_files' sentinels.

    ``iter_first_party_files`` validates ``shared/src`` and ``orchestrator/src``
    and RAISES if they are absent, so both are created even when no file is
    placed in them.
    """
    root = tmp_path / 'repo'
    (root / 'shared' / 'src').mkdir(parents=True)
    (root / 'orchestrator' / 'src').mkdir(parents=True)
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding='utf-8')
    return root


@pytest.fixture
def isolated_parse_cache(monkeypatch):
    """Give the test a private, empty provider memo.

    Deliberately NOT ``_reset_parse_cache()``: clearing the real memo mid-session
    would make the next consumer re-read and re-parse all 461 files, resurrecting
    the very cost this task removes. Swapping the dict leaves the session's warm
    entry for ``_REPO_ROOT`` untouched and restores it on teardown.
    """
    cache: dict = {}
    monkeypatch.setattr(sfs, '_PARSE_CACHE', cache)
    return cache


class _WorkCounter:
    """Count ``ast.parse`` / ``Path.read_text`` calls made by the provider."""

    def __init__(self, monkeypatch):
        self.parses: list[str] = []
        self.reads: list[str] = []
        real_parse = ast.parse
        real_read_text = Path.read_text

        def counting_parse(source, filename='<unknown>', *args, **kwargs):
            self.parses.append(str(filename))
            return real_parse(source, filename, *args, **kwargs)

        def counting_read_text(self_path, *args, **kwargs):
            self.reads.append(str(self_path))
            return real_read_text(self_path, *args, **kwargs)

        monkeypatch.setattr(sfs.ast, 'parse', counting_parse)
        monkeypatch.setattr(sfs.Path, 'read_text', counting_read_text)


# ---------------------------------------------------------------------------
# Step 1 — the shared provider's contract
# ---------------------------------------------------------------------------


class TestParseFirstPartyTreeContract:
    """One immutable record per first-party file, carrying tree XOR syntax_error."""

    def test_one_record_per_first_party_file(self, tmp_path, isolated_parse_cache):
        """Enumeration is delegated verbatim to iter_first_party_files."""
        root = _make_fake_repo(tmp_path, {
            'shared/src/shared/alpha.py': 'x = 1\n',
            'orchestrator/src/orchestrator/beta.py': 'y = 2\n',
            'orchestrator/src/orchestrator/tests/gamma.py': 'z = 3\n',   # excluded
            'orchestrator/src/orchestrator/test_delta.py': 'w = 4\n',    # excluded
            'orchestrator/src/orchestrator/conftest.py': 'v = 5\n',      # excluded
        })
        records = sfs.parse_first_party_tree(root)
        assert [r.path for r in records] == list(iter_first_party_files(root))
        assert sorted(r.relpath for r in records) == [
            'orchestrator/src/orchestrator/beta.py',
            'shared/src/shared/alpha.py',
        ]

    def test_record_field_shapes(self, tmp_path, isolated_parse_cache):
        """path absolute, relpath posix-relative, source verbatim, tree XOR error."""
        root = _make_fake_repo(tmp_path, {'shared/src/shared/alpha.py': 'x = 1\n'})
        (record,) = sfs.parse_first_party_tree(root)
        assert isinstance(record.path, Path) and record.path.is_absolute()
        assert record.relpath == 'shared/src/shared/alpha.py'
        assert record.path == root / record.relpath
        assert record.source == 'x = 1\n'
        assert isinstance(record.tree, ast.Module)
        assert record.syntax_error is None
        assert (record.tree is None) != (record.syntax_error is None), (
            'a record must carry exactly one of tree / syntax_error'
        )

    def test_records_are_immutable(self, tmp_path, isolated_parse_cache):
        """A consumer walking the shared ASTs must not be able to rewrite a record."""
        root = _make_fake_repo(tmp_path, {'shared/src/shared/alpha.py': 'x = 1\n'})
        records = sfs.parse_first_party_tree(root)
        assert isinstance(records, tuple)
        with pytest.raises(AttributeError):
            records[0].source = 'mutated'

    def test_syntax_error_is_recorded_not_raised(self, tmp_path, isolated_parse_cache):
        """test_no_unparseable_files reports every bad file at once — so record, don't raise."""
        root = _make_fake_repo(tmp_path, {
            'shared/src/shared/ok.py': 'x = 1\n',
            'shared/src/shared/broken.py': 'def f(:\n',
        })
        by_rel = {r.relpath: r for r in sfs.parse_first_party_tree(root)}
        broken = by_rel['shared/src/shared/broken.py']
        assert isinstance(broken.syntax_error, SyntaxError)
        assert broken.tree is None
        assert broken.source == 'def f(:\n'
        assert isinstance(by_rel['shared/src/shared/ok.py'].tree, ast.Module)

    def test_undecodable_file_propagates(self, tmp_path, isolated_parse_cache):
        """A first-party file that cannot be DECODED is a real breakage — stay loud.

        The archival gate's documented contract: silently skipping an unreadable
        file would let a TaskConfigDir construction site hide behind it.
        """
        root = _make_fake_repo(tmp_path, {
            'shared/src/shared/binary.py': b'x = "\xff\xfe not utf-8"\n',
        })
        with pytest.raises(UnicodeDecodeError):
            sfs.parse_first_party_tree(root)

    def test_unreadable_file_propagates(self, tmp_path, isolated_parse_cache, monkeypatch):
        """An OSError on read propagates too — same loudness contract as decode."""
        root = _make_fake_repo(tmp_path, {'shared/src/shared/alpha.py': 'x = 1\n'})
        real_read_text = Path.read_text

        def boom(self_path, *args, **kwargs):
            if self_path.name == 'alpha.py':
                raise PermissionError('unreadable first-party file')
            return real_read_text(self_path, *args, **kwargs)

        monkeypatch.setattr(sfs.Path, 'read_text', boom)
        with pytest.raises(PermissionError):
            sfs.parse_first_party_tree(root)


class TestParseFirstPartyTreeIsMemoized:
    """The whole point: the tree is read once and parsed once per session."""

    def test_second_call_returns_the_identical_object(self, tmp_path, isolated_parse_cache):
        root = _make_fake_repo(tmp_path, {'shared/src/shared/alpha.py': 'x = 1\n'})
        first = sfs.parse_first_party_tree(root)
        assert sfs.parse_first_party_tree(root) is first

    def test_memo_key_is_the_resolved_root(self, tmp_path, isolated_parse_cache):
        """An unresolved spelling of the same root must hit the same entry."""
        root = _make_fake_repo(tmp_path, {'shared/src/shared/alpha.py': 'x = 1\n'})
        first = sfs.parse_first_party_tree(root)
        assert sfs.parse_first_party_tree(root / 'shared' / '..') is first
        assert len(isolated_parse_cache) == 1

    def test_each_file_is_read_once_and_parsed_once(self, tmp_path, isolated_parse_cache,
                                                    monkeypatch):
        """Across TWO calls, each file is read once and ast.parse'd once."""
        root = _make_fake_repo(tmp_path, {
            'shared/src/shared/alpha.py': 'x = 1\n',
            'orchestrator/src/orchestrator/beta.py': 'y = 2\n',
        })
        counter = _WorkCounter(monkeypatch)
        sfs.parse_first_party_tree(root)
        sfs.parse_first_party_tree(root)
        assert len(counter.parses) == 2, f'expected 2 parses, got {counter.parses}'
        assert len(counter.reads) == 2, f'expected 2 reads, got {counter.reads}'
        assert sorted(Path(p).name for p in counter.parses) == ['alpha.py', 'beta.py']

    def test_reset_parse_cache_forces_a_reparse(self, tmp_path, isolated_parse_cache,
                                                monkeypatch):
        """The test hook is real: after a reset the tree is read and parsed again."""
        root = _make_fake_repo(tmp_path, {'shared/src/shared/alpha.py': 'x = 1\n'})
        first = sfs.parse_first_party_tree(root)
        sfs._reset_parse_cache()
        assert isolated_parse_cache == {}
        counter = _WorkCounter(monkeypatch)
        second = sfs.parse_first_party_tree(root)
        assert second is not first
        assert len(counter.parses) == 1
        assert len(counter.reads) == 1


class TestSessionFixtureSharesTheProvider:
    """conftest's ``first_party_tree`` hands out the memoized real-tree records."""

    def test_fixture_is_the_memoized_provider_result(self, first_party_tree):
        assert first_party_tree is sfs.parse_first_party_tree(_REPO_ROOT)

    def test_fixture_covers_the_real_tree(self, first_party_tree):
        """Floor guard, mirroring the gates' own anti-vacuity floors."""
        assert len(first_party_tree) > 150, (
            f'Only {len(first_party_tree)} first-party files parsed — is repo_root '
            f'correct? ({_REPO_ROOT})'
        )
        assert all(r.syntax_error is None for r in first_party_tree), (
            'unparseable first-party file(s): '
            + ', '.join(r.relpath for r in first_party_tree if r.syntax_error)
        )


# ---------------------------------------------------------------------------
# Step 3 — the parent map is built LAZILY, and a pre-parsed tree is reusable
# ---------------------------------------------------------------------------

#: A signature-(a) violation nested two scopes deep, so a degraded qualname
#: resolution shows up as a wrong ANSWER rather than merely a missing one.
_SIG_A_SOURCE = '''class Harness:
    async def run(self):
        value, _ = await get_statuses()
        return value
'''

#: A signature-(b) violation, likewise nested.
_SIG_B_SOURCE = '''class Loader:
    def load(self):
        try:
            return parse()
        except Exception:
            return None
'''

#: A file with neither signature — the 451-of-461 case in the real tree.
_CLEAN_SOURCE = '''class Quiet:
    def run(self):
        try:
            return compute()
        except KeyError as exc:
            logger.warning('missed: %s', exc)
            raise
'''


class TestParentMapIsLazy:
    """97.8% of the real tree yields no violation — so build no parent map for it.

    Measured: 10 of 461 first-party files produce at least one violation, while
    the pre-4520 scanner built a parent map for every one of them (3.73s per
    full pass, done twice per session).
    """

    def test_a_clean_file_builds_no_parent_map(self, monkeypatch):
        """The parent map is needed only to name a violation's enclosing scope."""
        def boom(_tree):
            raise AssertionError(
                'a file with no violation must not pay for a parent map'
            )

        monkeypatch.setattr(sfs, '_build_parent_map', boom)
        assert sfs.find_violations(_CLEAN_SOURCE, 'fake/quiet.py') == []

    def test_a_syntax_error_builds_no_parent_map(self, monkeypatch):
        """find_violations still returns [] on SyntaxError, without any map work."""
        def boom(_tree):
            raise AssertionError('an unparseable file must not pay for a parent map')

        monkeypatch.setattr(sfs, '_build_parent_map', boom)
        assert sfs.find_violations('def f(:\n', 'fake/broken.py') == []

    def test_signature_a_qualname_and_hash_survive_laziness(self):
        """Pinned against the pre-4520 values: laziness must not degrade qualname."""
        (violation,) = sfs.find_violations(_SIG_A_SOURCE, 'fake/module.py')
        assert violation.signature == 'a'
        assert violation.qualname == 'Harness.run'
        assert violation.content_hash == '4d5edf203564'

    def test_signature_b_qualname_and_hash_survive_laziness(self):
        """Pinned against the pre-4520 values (the other half of the ratchet key)."""
        (violation,) = sfs.find_violations(_SIG_B_SOURCE, 'fake/module.py')
        assert violation.signature == 'b'
        assert violation.qualname == 'Loader.load'
        assert violation.content_hash == '92b98f5b67f9'

    def test_module_scope_violation_still_resolves(self):
        """The <module> fallback needs the map too — it must still be built."""
        (violation,) = sfs.find_violations('x, _ = get_statuses()\n', 'fake/module.py')
        assert violation.qualname == '<module>'

    def test_the_map_is_built_at_most_once_per_file(self, monkeypatch):
        """Two violations in one file share one parent map, as they did before."""
        calls = []
        real = sfs._build_parent_map

        def counting(tree):
            calls.append(tree)
            return real(tree)

        monkeypatch.setattr(sfs, '_build_parent_map', counting)
        source = _SIG_A_SOURCE + '\n' + _SIG_B_SOURCE
        assert len(sfs.find_violations(source, 'fake/module.py')) == 2
        assert len(calls) == 1, f'expected one parent map, built {len(calls)}'


class TestFindViolationsInTree:
    """The already-parsed entry point the shared provider feeds."""

    @pytest.mark.parametrize(
        'source',
        [_SIG_A_SOURCE, _SIG_B_SOURCE, _CLEAN_SOURCE],
        ids=['sig-a', 'sig-b', 'clean'],
    )
    def test_matches_find_violations_exactly(self, source):
        tree = ast.parse(source, filename='fake/module.py')
        assert sfs.find_violations_in_tree(tree, 'fake/module.py', source) == (
            sfs.find_violations(source, 'fake/module.py')
        )

    def test_source_argument_is_optional(self):
        """The scanner works off the tree; *source* is context, not input."""
        tree = ast.parse(_SIG_B_SOURCE, filename='fake/module.py')
        assert sfs.find_violations_in_tree(tree, 'fake/module.py') == (
            sfs.find_violations(_SIG_B_SOURCE, 'fake/module.py')
        )

    def test_filename_is_what_lands_in_the_record(self):
        """The gate keys on relpath, so the caller's filename must be honoured."""
        tree = ast.parse(_SIG_A_SOURCE, filename='ignored.py')
        (violation,) = sfs.find_violations_in_tree(tree, 'shared/src/shared/x.py')
        assert violation.filename == 'shared/src/shared/x.py'


# ---------------------------------------------------------------------------
# Step 5 — the silent-fallthrough gate consumes the shared provider
# ---------------------------------------------------------------------------


def _prefix_tree_scan_reference(records):
    """Recompute the gate's inputs with the PRE-4520 loop's plumbing.

    Reproduces the old fixture statement for statement — the same
    ``iter_first_party_files`` enumeration and order, the same
    ``str(filepath.relative_to(REPO_ROOT))`` filename spelling (NOT the
    provider's posix ``relpath``, which is the thing being changed), the same
    ``f'{path}: {e}'`` parse-failure message, the same skip-a-broken-file
    structure — and then compares that against what the rewritten gate
    produces. So every piece of PLUMBING the rewrite touches is re-derived
    independently, whole-tree.

    ONE THING IS DELIBERATELY NOT RE-DERIVED: the ``ast.parse`` itself. The old
    loop reached the scanner through ``find_violations(source, rel)``, which
    since task 4520 is a two-line wrapper that parses and then calls
    ``find_violations_in_tree(tree, filename, source)`` — the exact function
    the new gate calls. ``source`` is unused by the scan and ``filename``
    reaches only ``Violation.filename`` (which IS re-derived here, in the old
    spelling), so a second ``ast.parse`` of the same bytes could only prove
    that ``ast.parse`` is deterministic. On this tree that ceremony measured
    18.87s in a single test item under load — against the 60s budget whose
    thin headroom is the entire defect this task exists to remove, and worse
    than the 23.68s item it replaces once the provider's own setup lands on
    the same item. Re-parsing is therefore done only where it could actually
    differ: :meth:`TestSilentFallthroughGateUsesTheSharedTree.
    test_recorded_violations_survive_a_real_reparse` pushes every file that
    produces a violation back through the full ``find_violations`` entry point.

    The read is not re-derived either: the provider reads with STRICT utf-8,
    so a file the old ``errors='replace'`` loop would have papered over now
    RAISES rather than yielding different text (``test_undecodable_file_
    propagates``).
    """
    files = list(iter_first_party_files(_REPO_ROOT))
    by_path = {record.path: record for record in records}
    violations = []
    parse_failures = []
    for filepath in files:
        record = by_path[filepath]
        rel = str(filepath.relative_to(_REPO_ROOT))
        if record.syntax_error is not None:
            parse_failures.append(f'{filepath}: {record.syntax_error}')
            continue
        violations.extend(
            sfs.find_violations_in_tree(record.tree, rel, record.source)
        )
    return files, violations, parse_failures


class TestSilentFallthroughGateUsesTheSharedTree:
    """tree_scan_data walks the shared ASTs and parses nothing itself."""

    def test_building_the_scan_data_does_no_io_and_no_parsing(
        self, first_party_tree, monkeypatch
    ):
        """With the memo warm, the gate's own work must be a WALK and nothing more."""
        import test_silent_fallthrough_gate as gate

        counter = _WorkCounter(monkeypatch)
        data = gate._build_tree_scan_data(first_party_tree)
        assert counter.parses == [], (
            f'tree_scan_data re-parsed {len(counter.parses)} file(s): '
            f'{counter.parses[:5]}'
        )
        assert counter.reads == [], (
            f'tree_scan_data re-read {len(counter.reads)} file(s): '
            f'{counter.reads[:5]}'
        )
        assert len(data.files) == len(first_party_tree)

    def test_parse_failure_message_shape_is_preserved(self):
        """test_no_unparseable_files prints these — keep the pre-4520 shape.

        Non-vacuous by construction: the real tree has zero parse failures
        today, so whole-tree parity alone would assert ``[] == []``.
        """
        import test_silent_fallthrough_gate as gate

        source = 'def f(:\n'
        try:
            ast.parse(source, filename='fake/broken.py')
        except SyntaxError as exc:
            error = exc
        record = sfs.ParsedFile(
            path=Path('/repo/shared/src/shared/broken.py'),
            relpath='shared/src/shared/broken.py',
            source=source,
            tree=None,
            syntax_error=error,
        )
        data = gate._build_tree_scan_data((record,))
        assert data.parse_failures == [f'{record.path}: {error}']
        assert data.violations == []

    def test_output_parity_with_the_prefix_computation(self, first_party_tree):
        """files, violation_key_counts and parse_failures are unchanged.

        Whole-tree, against an independent re-derivation of the pre-4520
        loop's plumbing — including its ``str(relative_to)`` filename spelling,
        which the rewrite replaces with the provider's posix ``relpath`` and
        which ``violation_key`` keys on. See
        :func:`_prefix_tree_scan_reference` for exactly what is and is not
        re-derived, and why.
        """
        import test_silent_fallthrough_gate as gate

        expected_files, expected_violations, expected_failures = (
            _prefix_tree_scan_reference(first_party_tree)
        )
        data = gate._build_tree_scan_data(first_party_tree)

        assert data.files == expected_files
        assert data.parse_failures == expected_failures
        expected_counts = Counter(sfs.violation_key(v) for v in expected_violations)
        assert sorted(data.violation_key_counts.items()) == sorted(
            expected_counts.items()
        ), (
            'the consolidated scan sees a different violation multiset than the '
            'pre-4520 read-then-parse-then-find_violations loop did'
        )
        assert sum(expected_counts.values()) > 0, (
            'parity would be vacuous with no violations in the tree — the '
            'allowlist records 14 today'
        )

    def test_recorded_violations_survive_a_real_reparse(self, first_party_tree):
        """The one place a second ast.parse could differ — so do it there, only there.

        For every file that actually produces a violation, push the source back
        through the full pre-4520 ``find_violations`` entry point (which parses
        from scratch) and require the identical records. Fourteen files today,
        microseconds; re-parsing the other 447 would only re-prove that
        ``ast.parse`` is deterministic.
        """
        producing = [
            record for record in first_party_tree
            if record.tree is not None
            and sfs.find_violations_in_tree(record.tree, record.relpath, record.source)
        ]
        assert producing, (
            'no first-party file produces a violation — this cross-check would '
            'be vacuous; the allowlist records 14 today'
        )
        for record in producing:
            rel = str(record.path.relative_to(_REPO_ROOT))
            assert sfs.find_violations(record.source, rel) == (
                sfs.find_violations_in_tree(record.tree, rel, record.source)
            ), f'a real re-parse of {rel} yields different violations'
