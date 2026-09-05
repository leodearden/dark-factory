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
