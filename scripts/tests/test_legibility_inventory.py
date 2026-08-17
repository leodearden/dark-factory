"""Tests for scripts/legibility/inventory.py — session enumeration (PRD §5.2 point 2).

``inventory.encode_cwd`` mirrors ``orchestrator.session_registry.encode_cwd``,
the canonical cwd encoding (``/``, ``.`` and ``_`` all map to ``-``, and case
is preserved); ``TestEncoderLockstep`` below holds every in-repo copy of that
rule to the canonical AND to real on-disk dir names. A project's agents span many encoded
dirs (57 for dark-factory today: main checkout + ``.worktrees``/
``.claude-worktrees`` children), so membership is resolved from the
session's REAL ``cwd`` (read from a transcript line) via path-component
semantics (``Path.is_relative_to``) — never a raw string prefix match on
the encoded dir name, which would over-include a sibling project sharing
the same literal prefix (e.g. ``dark-factory-cockpit``).

Imported as ``from legibility import inventory`` (PEP-420 namespace
package; see test_legibility_config.py's module docstring for the import
mechanics).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import subprocess
from collections.abc import Callable
from datetime import date as dt_date
from functools import lru_cache
from pathlib import Path

import pytest
from legibility import digest
from legibility import inventory as mod

from orchestrator import session_registry

# Repo root from scripts/tests/ — the same parents[2] derivation
# scripts/tests/conftest.py already uses.
SPAWN_SCRIPT = Path(__file__).resolve().parents[2] / 'skills' / 'spawn' / 'spawn-claude.sh'

MAIN_CWD = '/home/leo/src/dark-factory'
WORKTREE_CWD = '/home/leo/src/dark-factory/.worktrees/2573'
COCKPIT_CWD = '/home/leo/src/dark-factory-cockpit'

# OBSERVED, not guessed (task 3272). Every right-hand side below is a real
# directory name read off a live ``~/.claude/projects`` tree, or a cwd
# confirmed against one. The rule was derived empirically from 738
# (encoded-dir, decoded-cwd) pairs sampled from that tree: the only
# substitutions observed were ``.`` -> ``-``, ``/`` -> ``-`` and
# ``_`` -> ``-``, and the only non-alphanumeric characters appearing in ANY
# sampled cwd were ``- . / _`` — so the three-character rule is complete
# over the observed domain (it reproduces all 738 pairs; the former
# two-character ``/``+``.`` rule mismatched 492 of them).
#
# These are STRING LITERALS on purpose. They must never be produced by
# calling ``encode_cwd`` (or any mirror of it): a fixture built with the
# function under test moves in lockstep with a bug in that function and can
# never detect it, which is exactly why the missing ``_`` rule survived a
# fully green suite. See TestEncoderLockstep below.
REAL_ENCODED_DIR_PAIRS: tuple[tuple[str, str], ...] = (
    (MAIN_CWD, '-home-leo-src-dark-factory'),
    (WORKTREE_CWD, '-home-leo-src-dark-factory--worktrees-2573'),
    (
        '/home/leo/src/dark-factory/.eval-worktrees/df_task_12/run-5383f6a8',
        '-home-leo-src-dark-factory--eval-worktrees-df-task-12-run-5383f6a8',
    ),
    (
        '/home/leo/src/reify/.claude/worktrees/printer-design-v01',
        '-home-leo-src-reify--claude-worktrees-printer-design-v01',
    ),
    # Pins CASE PRESERVATION: the encoder does NOT lowercase. This dir name
    # exists on disk with its capitals intact, ruling out a case-folding step.
    ('/opt/Auto-Claude/resources/backend', '-opt-Auto-Claude-resources-backend'),
    (
        '/home/leo/src/warm-lanes/worktrees/_lane-39',
        '-home-leo-src-warm-lanes-worktrees--lane-39',
    ),
    ('/media/leo/data_lv_1/leo/reify-build', '-media-leo-data-lv-1-leo-reify-build'),
)


# ---------------------------------------------------------------------------
# Corruption scaffolding — the damage a fire-and-forget archive writer really
# produces on a plain ``.jsonl`` corpus (a write interrupted by a killed unit,
# or a flipped stored byte). Kept local to this file rather than hoisted into
# conftest: scripts/tests/conftest.py is sys.path bootstrap only, and each test
# module here already carries its own write helpers.
# ---------------------------------------------------------------------------

_UNDECODABLE_BODY = b'{"type": "user", "seq": 0}\n{"type": "user", "t": "\xff\xfe"}\n'
"""A JSONL body whose SECOND line carries a raw 0xFF — invalid UTF-8.

The first line is well-formed on purpose: a reader that degraded this
per-LINE rather than per-FILE would be visibly distinguishable here (it
would yield record 0 and skip record 1) instead of silently passing.
"""


def _write_undecodable_plain(path: Path) -> Path:
    """Write a plain ``.jsonl`` whose payload is not valid UTF-8.

    The reader opens under strict ``encoding='utf-8'``, so a single flipped
    stored byte makes byte 0xFF meet the text wrapper and raise
    ``UnicodeDecodeError`` — a ``ValueError`` subclass, which therefore
    escapes an ``except OSError`` degrade path unless the reader normalizes
    it. This is why the normalized message says "undecodable transcript
    bytes" rather than labelling it a compression failure.
    """
    path.write_bytes(_UNDECODABLE_BODY)
    return path


class TestEncodeCwd:
    def test_plain_path(self):
        assert mod.encode_cwd(MAIN_CWD) == '-home-leo-src-dark-factory'

    def test_worktrees_child_maps_slash_and_dot(self):
        # Two of the three characters; see test_underscore_maps_to_dash for
        # the third. A leading '.' on a path component yields a doubled '--'
        # (one dash from the preceding '/', one from the '.').
        assert mod.encode_cwd(WORKTREE_CWD) == '-home-leo-src-dark-factory--worktrees-2573'

    def test_underscore_maps_to_dash(self):
        # The character the mirror used to miss (task 3272). Two thirds of the
        # real project dirs sampled contain an underscore.
        assert mod.encode_cwd('/media/leo/data_lv_1/leo/reify-build') == (
            '-media-leo-data-lv-1-leo-reify-build'
        )

    def test_round_trips_real_on_disk_dir_names(self):
        """Every encoding matches a dir name observed on a live ~/.claude/projects tree.

        Table-driven over REAL_ENCODED_DIR_PAIRS, whose expected values are
        hard-coded literals rather than encoder output — the only kind of
        assertion that can catch an encoder which is self-consistently wrong.
        """
        for cwd, expected_dir in REAL_ENCODED_DIR_PAIRS:
            assert mod.encode_cwd(cwd) == expected_dir, cwd

    def test_cockpit_sibling_shares_literal_prefix(self):
        # This is exactly why a raw string-prefix match over-includes: the
        # encoded cockpit dir name starts with the encoded main dir name.
        encoded_main = mod.encode_cwd(MAIN_CWD)
        encoded_cockpit = mod.encode_cwd(COCKPIT_CWD)
        assert encoded_cockpit.startswith(encoded_main)
        assert encoded_cockpit != encoded_main


def _load_sibling_test_module(name: str):
    """Import a sibling scripts/tests module by file path.

    ``scripts/tests`` is not on ``sys.path`` (its conftest inserts
    ``scripts/`` and ``scripts/legibility``, not itself), so a bare
    ``import test_legibility_nightly`` would not resolve under the suite's
    ``--import-mode=importlib`` collection. Loading by path is the sanctioned
    equivalent and avoids restructuring the nightly fixture.
    """
    spec = importlib.util.spec_from_file_location(
        f'_lockstep_{name}', Path(__file__).parent / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _bash_encode_cwd_source() -> str:
    """Extract spawn-claude.sh's ``_encode_cwd`` function block verbatim.

    spawn-claude.sh is not sourceable — it runs ``set -u`` and an argument-count
    usage check that exits 2 long before ``_encode_cwd``'s definition, and past
    that point proceeds straight to launching a terminal. Rather than put a
    test-only "library mode" branch into a production launcher on the incident
    path, the four-line pure function is lifted out by an anchored regex and
    eval'd in a throwaway shell (see :func:`_bash_encode_cwd`).

    The regex anchors on exactly two properties, and it is worth stating them
    as MEASURED rather than as remembered — a rule restated from memory in a
    comment is the mechanism task 3272 traced this whole bug class to:

      - the definition starts at column 0 as ``_encode_cwd()`` — the
        ``function _encode_cwd`` form does not match, nor does an indented one;
      - the body ends at a line that is a bare ``}`` at column 0, with no
        earlier column-0 ``}`` inside it (the match is non-greedy, so an
        earlier one truncates the extraction).

    Nothing else is required. Verified against this file: ``\\s*`` spans
    newlines, so ``()`` and ``{`` need NOT share a line (a brace-on-next-line
    reformat still matches), and ``re.search`` scans the whole script, so the
    function may be MOVED anywhere in it without breaking extraction. Neither a
    move nor a brace-style reformat trips the assert below, and this docstring
    must not claim they do.

    The assert on a miss is load-bearing: if the function is renamed or its
    definition stops starting at column 0, a tolerant ``if m:`` would return
    nothing and the whole lockstep would pass VACUOUSLY — the exact class of
    silent coverage loss this guard exists to prevent. A loud extraction
    failure is the correct outcome.
    """
    source = SPAWN_SCRIPT.read_text()
    match = re.search(r'^_encode_cwd\(\)\s*\{\n(?:.*\n)*?^\}$', source, re.M)
    assert match is not None, (
        f'could not extract the _encode_cwd() function block from {SPAWN_SCRIPT}. '
        'Either it was renamed, or its definition no longer starts at column 0 '
        'as `_encode_cwd()` (the `function _encode_cwd` form does not match), or '
        'its body no longer ends at a bare column-0 `}`. Fix the regex here '
        'rather than letting TestEncoderLockstep silently stop covering the '
        'bash copy.'
    )
    return match.group(0)


def _bash_encode_cwd(cwd: str) -> str:
    """Run spawn-claude.sh's ``_encode_cwd`` on ``cwd`` and return its raw stdout.

    stdout is returned UNSTRIPPED on purpose: the bash function ends in
    ``printf '%s'`` (no trailing newline), so an exact ``==`` comparison is
    valid and keeps a stray-whitespace regression detectable — stripping would
    mask one.

    Every guard in this bridge is deliberately loud-on-failure, and the two
    below close the only paths that were not. ``timeout`` is not optional
    belt-and-braces: an unavailable or wedged ``bash`` with no deadline HANGS
    the suite instead of failing it, which would be the single silent
    degradation in an otherwise fail-loud mechanism. And the returncode is
    checked explicitly rather than with ``check=True``, because
    ``CalledProcessError``'s default message does not include captured stderr —
    a bash-level syntax error introduced while editing ``_encode_cwd`` would
    surface as a bare "returned non-zero exit status 2" with the actual
    diagnostic hidden in an unprinted attribute.
    """
    result = subprocess.run(
        ['bash', '-c', _bash_encode_cwd_source() + '\n_encode_cwd "$1"', 'bash', cwd],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f'bash _encode_cwd exited {result.returncode} for cwd {cwd!r}. '
        f'stderr: {result.stderr!r}'
    )
    return result.stdout


@lru_cache(maxsize=1)
def _mirrors() -> tuple[tuple[str, Callable[[str], str]], ...]:
    """(label, callable) for every in-repo copy of the cwd encoding.

    Not only the Python ones: a copy in another language enters through a
    bridge that presents it as a plain ``Callable[[str], str]``, so the
    lockstep assertions need no special case for it. ``spawn-claude.sh``'s
    bash copy is the first such entry (see :func:`_bash_encode_cwd`).

    Cached, and resolved ONCE per session rather than per assertion row, for
    two reasons that now compound: naming the nightly mirror requires exec'ing
    that whole test module, which is not written to be executed repeatedly,
    and the bash bridge reads and regex-scans spawn-claude.sh. An uncached
    call inside the ``REAL_ENCODED_DIR_PAIRS`` loop paid both once per row
    (growing with the table). Any future module-scope side effect there now
    costs one execution, not N. Note the caching is of the mirror REGISTRY and
    of the extracted bash SOURCE, never of an encoding result — every
    assertion row still calls each encoder for real.

    As of task 3464 no in-repo copy is deliberately omitted. If you add one,
    add it here; if you add one that cannot be pinned to the canonical, say so
    in :class:`TestEncoderLockstep`'s SCOPE note rather than leaving it
    unlisted, which would let that docstring imply coverage it lacks.
    """
    nightly_tests = _load_sibling_test_module('test_legibility_nightly')
    return (
        ('legibility.inventory.encode_cwd', mod.encode_cwd),
        ('legibility.digest._encode_cwd', digest._encode_cwd),
        ('test_legibility_nightly._encode_cwd', nightly_tests._encode_cwd),
        ('skills/spawn/spawn-claude.sh:_encode_cwd', _bash_encode_cwd),
    )


class TestEncoderLockstep:
    """Every in-repo copy of the cwd encoding must agree with the canonical (task 3272).

    The rule is duplicated five times across the repo (four Python, one bash),
    and EVERY ONE was found to be missing the same character (``_`` -> ``-``)
    at once — the four Python copies in task 3272, the bash copy in task 3464.
    The old ``inventory.encode_cwd`` docstring asserted the mirrors were "kept
    in lockstep with the canonical implementation" — a claim nothing checked,
    and which was false in fact.

    This class replaces that aspiration with an enforced invariant. Each
    mirror is asserted equal to BOTH:

      - ``session_registry.encode_cwd``, the canonical — so a mirror that
        drifts from it fails loudly; and
      - the hard-coded ``REAL_ENCODED_DIR_PAIRS`` dir names — so the
        canonical drifting from REALITY fails too.

    The second assertion is the load-bearing one. A mirror-only check would
    have passed cleanly on the pre-3272 tree, because all four copies were
    consistently wrong together. The same defect explains why 37 green tests
    never caught it: every fixture built its session dirs by calling the
    encoder under test, so the fixtures tracked the bug. Only literals read
    off a real ``~/.claude/projects`` tree can detect an encoder that is
    self-consistently wrong.

    Every in-repo copy is now inside this guard — the two that task 3272 had
    to leave outside it were closed by task 3464:

      - ``skills/spawn/spawn-claude.sh``'s ``_encode_cwd`` (bash) is covered
        via :func:`_bash_encode_cwd`, a subprocess bridge: the function block
        is lifted out of the script with an anchored regex and eval'd in a
        throwaway shell, then registered as an ordinary :func:`_mirrors`
        entry. 3272 left it out for want of a mechanism (no Python test can
        import bash), not for want of will. The extraction ASSERTS on a miss,
        so losing the function fails loudly instead of silently dropping
        coverage and letting this class pass vacuously — see
        :func:`_bash_encode_cwd_source` for precisely what that extraction does
        and does not anchor on.
      - ``tests/scripts/test_spawn_claude.py``'s fixture is no longer a copy
        of the rule at all: it now CALLS ``session_registry.encode_cwd``, so
        it is pinned to the canonical by construction and needs no entry
        here. (It could never have been listed as a mirror anyway — it names
        the dir a fake ``claude`` creates for spawn-claude.sh's own probe to
        find, so it was pinned to the BASH copy, and while bash was wrong,
        asserting it equal to the canonical would have asserted the wrong
        thing. Both moved in one commit, as this note previously required.)

    SCOPE — what this still does NOT cover. This class verifies encoder
    AGREEMENT and nothing more. That spawn-claude.sh's ``_started_evidence``
    (or any other caller) USES the encoded value correctly — as a lookup key
    against a directory that exists, at the right moment — is outside it;
    ``test_spawn_claude.py``'s ``test_transcript_appearance_suppresses_flag``
    is what exercises that end to end. Nor does it extend the rule's
    validated domain: the pairs below are complete only over the punctuation
    actually observed (``- . / _``), per ``session_registry.encode_cwd``.

    Best of all is not to add a copy: prefer CALLING
    ``session_registry.encode_cwd``, as the spawn test's fixture now does.
    Where a copy is genuinely unavoidable, add it to :func:`_mirrors` — being
    in another language is no longer an exemption, since a bridge can present
    it as a plain callable. Only if a copy truly cannot be pinned to the
    canonical, say so in the SCOPE note above rather than leaving it silently
    unlisted and letting this docstring imply coverage it lacks.
    """

    def test_every_mirror_agrees_with_canonical_and_with_reality(self):
        canonical = session_registry.encode_cwd
        mirrors = _mirrors()
        for cwd, expected_dir in REAL_ENCODED_DIR_PAIRS:
            # The canonical itself must match the real on-disk dir name.
            assert canonical(cwd) == expected_dir, f'canonical drifted from reality: {cwd}'
            for label, mirror in mirrors:
                got = mirror(cwd)
                assert got == canonical(cwd), f'{label} drifted from canonical: {cwd}'
                assert got == expected_dir, f'{label} drifted from reality: {cwd}'


def _write_session(dir_path: Path, session_id: str, cwd: str, timestamp: str = '2026-07-13T10:00:00.000Z'):
    dir_path.mkdir(parents=True, exist_ok=True)
    session_path = dir_path / f'{session_id}.jsonl'
    lines = [
        {'type': 'user', 'cwd': cwd, 'timestamp': timestamp, 'message': {'content': 'hello'}},
    ]
    session_path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
    return session_path


class TestIsMember:
    """is_member uses Path.is_relative_to path-component semantics."""

    def test_main_dir_is_member(self):
        assert mod.is_member(MAIN_CWD, [MAIN_CWD]) is True

    def test_worktree_child_is_member(self):
        assert mod.is_member(WORKTREE_CWD, [MAIN_CWD]) is True

    def test_cockpit_sibling_is_not_member(self):
        assert mod.is_member(COCKPIT_CWD, [MAIN_CWD]) is False


class TestProjectDirMembershipResolution:
    """End-to-end: a tmp projects_root with main + worktree + cockpit-sibling
    encoded dirs. Membership resolution (iter_project_dirs + is_member on
    each session's real cwd) includes the main dir and worktree child but
    excludes the cockpit sibling — even though the cockpit dir's encoded
    name is a candidate under the cheap prefix pre-filter.
    """

    def _build_tree(self, tmp_path: Path) -> Path:
        projects_root = tmp_path / 'projects'
        _write_session(projects_root / '-home-leo-src-dark-factory', 'main-session', MAIN_CWD)
        _write_session(
            projects_root / '-home-leo-src-dark-factory--worktrees-2573',
            'worktree-session',
            WORKTREE_CWD,
        )
        _write_session(
            projects_root / '-home-leo-src-dark-factory-cockpit',
            'cockpit-session',
            COCKPIT_CWD,
        )
        return projects_root

    def test_iter_project_dirs_over_includes_cockpit_as_a_candidate(self, tmp_path):
        # The cheap encoded-prefix pre-filter is intentionally imprecise —
        # confirms the design premise that a further real-cwd check is needed.
        projects_root = self._build_tree(tmp_path)
        dirs = {d.name for d in mod.iter_project_dirs(projects_root, [MAIN_CWD])}
        assert '-home-leo-src-dark-factory-cockpit' in dirs

    def test_enumerate_membership_excludes_cockpit_includes_worktree(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        candidate_dirs = mod.iter_project_dirs(projects_root, [MAIN_CWD])
        members = []
        for project_dir in candidate_dirs:
            for session_path in project_dir.glob('*.jsonl'):
                cwd = mod.session_cwd(session_path)
                if cwd is not None and mod.is_member(cwd, [MAIN_CWD]):
                    members.append(session_path.stem)
        assert set(members) == {'main-session', 'worktree-session'}
        assert 'cockpit-session' not in members


class TestSessionCwd:
    def test_reads_cwd_from_first_matching_line(self, tmp_path):
        session_path = _write_session(tmp_path, 'sess', MAIN_CWD)
        assert mod.session_cwd(session_path) == MAIN_CWD

    def test_returns_none_when_no_cwd_anywhere(self, tmp_path):
        # Mirrors real ~/.claude/projects stub files: metadata-only lines
        # (ai-title/agent-name/queue-operation) carry no 'cwd' at all.
        session_path = tmp_path / 'stub.jsonl'
        lines = [
            {'type': 'ai-title', 'aiTitle': 'x', 'sessionId': 'stub'},
            {'type': 'agent-name', 'agentName': 'x', 'sessionId': 'stub'},
        ]
        session_path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
        assert mod.session_cwd(session_path) is None

    def test_returns_none_for_unreadable_path(self, tmp_path):
        assert mod.session_cwd(tmp_path / 'does-not-exist.jsonl') is None


class TestPublicIterJsonLines:
    """``iter_json_lines`` is PUBLIC — the single low-level transcript reader.

    The memory-eval retro corpus extractor
    (``fused-memory/scripts/memory_eval_transcript_corpus.py``) consumes it
    from a DIFFERENT package, which is why the name is public: a cross-package
    consumer of an underscore name is a standing invitation for the next
    author to copy the function instead — the outcome the reuse invariant
    exists to prevent.
    """

    RECORDS = [
        {'type': 'user', 'cwd': MAIN_CWD, 'seq': 1},
        {'type': 'assistant', 'seq': 2},
    ]

    def _lines(self) -> str:
        # A blank line and a syntactically-corrupt line interleaved between the
        # two good records: both are LINE-level degradations a fire-and-forget
        # writer really produces, and neither may abort the read.
        return (
            json.dumps(self.RECORDS[0])
            + '\n\n'
            + '{"type": "user", "message": {broken\n'
            + json.dumps(self.RECORDS[1])
            + '\n'
        )

    def test_public_name_exists(self):
        assert callable(mod.iter_json_lines)

    def test_plain_jsonl_skips_blank_and_corrupt_lines(self, tmp_path):
        path = tmp_path / 'session.jsonl'
        path.write_text(self._lines(), encoding='utf-8')
        assert list(mod.iter_json_lines(path)) == self.RECORDS


class TestIterJsonLinesCorruptionShapes:
    """The file-level corruption shape must raise ``OSError``.

    With the gzip container gone, the container-damage shapes (bad magic,
    truncated stream, corrupt body) are gone with it, and one shape survives::

        undecodable byte  -> UnicodeDecodeError  ("codec can't decode byte 0xff")

    It does not derive from ``OSError``, so unnormalized it escapes every
    consumer's documented ``except OSError`` degrade path — ``sampling.py``,
    ``check_transcript_persistence.py``, and the cross-package corpus
    extractor alike — and aborts the whole walk with a traceback. A flipped
    stored byte in live fleet runtime state produces exactly it, so it is not
    a theoretical shape.

    These tests pin the contract the reader's docstring advertises: one
    documented degrade path covers every way a FILE can be unreadable.
    """

    def test_undecodable_plain_jsonl_raises_oserror(self, tmp_path):
        # The one surviving file-level shape: the reader opens under strict
        # utf-8, so a bad byte aborts the read and must normalize to OSError.
        undecodable = _write_undecodable_plain(tmp_path / 'undecodable.jsonl')
        with pytest.raises(OSError):
            list(mod.iter_json_lines(undecodable))

    def test_the_decode_shape_names_the_offending_byte(self, tmp_path):
        # With the gzip container gone this is the ONE file-level shape left,
        # so the disclosed reason can no longer be triaged by contrast. What
        # has to survive is the actionable detail: the offending byte, which
        # is what sends an operator to the right place in the file rather than
        # to audit a compressor that is no longer in the picture.
        undecodable = _write_undecodable_plain(tmp_path / 'undecodable.jsonl')

        with pytest.raises(OSError) as undecodable_exc:
            list(mod.iter_json_lines(undecodable))

        message = str(undecodable_exc.value)
        assert 'gzip' not in message
        assert '0xff' in message.lower()

    def test_corrupt_line_in_a_valid_file_still_degrades_silently(self, tmp_path):
        # The other half of the split, and the one a too-broad fix would
        # destroy: a well-formed transcript whose LAST line is half-written
        # (the line-level analogue of a truncated file) still yields every
        # parseable record and still does NOT raise. If the file-level wrap
        # swallowed the parse loop as well, this read would start raising and
        # the coverage counters would double-count ordinary trailing debris as
        # unreadable files.
        good = [{'type': 'user', 'seq': 1}, {'type': 'assistant', 'seq': 2}]
        body = (
            json.dumps(good[0]) + '\n'
            + '\n'
            + '{"type": "user", "message": {broken\n'
            + json.dumps(good[1]) + '\n'
            + '{"type": "assistant", "message": {"content": "cut mid-writ'
        )
        path = tmp_path / 'trailing-partial.jsonl'
        path.write_text(body, encoding='utf-8')

        assert list(mod.iter_json_lines(path)) == good


class TestResolveAgentTranscriptRoots:
    """resolve_agent_transcript_roots joins each relative root against
    project_root (so mining is independent of the process CWD) and returns
    an already-absolute root unchanged — always as pathlib.Path instances."""

    PROJECT_ROOT = '/home/leo/src/dark-factory'

    def test_relative_root_resolved_against_project_root(self):
        roots = mod.resolve_agent_transcript_roots(
            self.PROJECT_ROOT, ['data/orchestrator/agent-transcripts']
        )
        assert roots == [
            Path('/home/leo/src/dark-factory/data/orchestrator/agent-transcripts')
        ]

    def test_absolute_root_returned_unchanged(self):
        roots = mod.resolve_agent_transcript_roots(
            self.PROJECT_ROOT, ['/var/lib/agent-transcripts']
        )
        assert roots == [Path('/var/lib/agent-transcripts')]

    def test_empty_roots_returns_empty_list(self):
        assert mod.resolve_agent_transcript_roots(self.PROJECT_ROOT, []) == []

    def test_result_elements_are_paths(self):
        roots = mod.resolve_agent_transcript_roots(
            self.PROJECT_ROOT, ['data/orchestrator/agent-transcripts', '/abs/root']
        )
        assert roots and all(isinstance(r, Path) for r in roots)


class TestEnumerateSessions:
    """enumerate_sessions aggregates across every matching encoded dir
    (never one-dir-per-project), filters by first-timestamp UTC date,
    stamps real size_bytes, and skips non-.jsonl / empty / fully-malformed
    files without raising."""

    TARGET_DATE = dt_date(2026, 7, 13)

    def _build_tree(self, tmp_path: Path) -> Path:
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        worktree_dir = projects_root / '-home-leo-src-dark-factory--worktrees-2573'
        main_dir.mkdir(parents=True)
        worktree_dir.mkdir(parents=True)

        # Target-date session in the main dir.
        _write_session(main_dir, 'main-target', MAIN_CWD, timestamp='2026-07-13T09:00:00.000Z')
        # Different-date session in the main dir — must be excluded.
        _write_session(
            main_dir, 'main-other-date', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z'
        )
        # Target-date session in the worktree dir — proves aggregation
        # across multiple encoded dirs, not just the main one.
        _write_session(
            worktree_dir, 'worktree-target', WORKTREE_CWD, timestamp='2026-07-13T11:00:00.000Z'
        )

        # A non-.jsonl file: excluded by the *.jsonl glob itself.
        (main_dir / 'notes.txt').write_text('not a transcript')
        # An empty .jsonl file: must be skipped, not raise.
        (main_dir / 'empty.jsonl').write_text('')
        # A fully-malformed .jsonl file (no valid JSON line at all, so no
        # cwd/timestamp is derivable): must be skipped, not raise.
        (main_dir / 'garbage.jsonl').write_text('not json\n{{{broken\n')

        return projects_root

    def test_returns_only_target_date_sessions(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert {r.path.stem for r in records} == {'main-target', 'worktree-target'}

    def test_aggregates_across_multiple_encoded_dirs(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert {r.encoded_dir for r in records} == {
            '-home-leo-src-dark-factory',
            '-home-leo-src-dark-factory--worktrees-2573',
        }

    def test_size_bytes_matches_real_file_size(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert records  # sanity: the fixture does produce records
        for record in records:
            assert record.size_bytes == record.path.stat().st_size

    def test_excludes_different_date_session(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        assert 'main-other-date' not in {r.path.stem for r in records}

    def test_skips_non_jsonl_empty_and_malformed_without_raising(self, tmp_path):
        projects_root = self._build_tree(tmp_path)
        records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        names = {r.path.stem for r in records}
        assert 'notes' not in names
        assert 'empty' not in names
        assert 'garbage' not in names


class TestEnumerateArchiveRoots:
    """enumerate_sessions additionally walks agent_transcript_roots — the
    archived fleet-transcript tree written by shared.transcript_archive in
    the production nested layout ``<archive>/<task_id>/<enc>/<sid>.jsonl``
    — recursively, gated solely by :func:`is_member` on each session's REAL
    cwd. The empty-roots path is byte-identical to today (the archive loop
    simply does not execute).
    """

    TARGET_DATE = dt_date(2026, 7, 13)
    WT_ENC = '-home-leo-src-dark-factory--worktrees-2573'

    def _build_archive(self, root: Path) -> Path:
        # Production nested layout: <archive>/<task_id>/<enc>/<sid>.jsonl
        enc_dir = root / '2573' / self.WT_ENC
        _write_session(
            enc_dir, 'archived-session', WORKTREE_CWD, timestamp='2026-07-13T09:00:00.000Z'
        )
        _write_session(
            enc_dir, 'plain-session', WORKTREE_CWD, timestamp='2026-07-13T10:00:00.000Z'
        )
        # A non-member cockpit cwd under its own task-id/enc dir: is_member
        # is false, so it is excluded even though it is inside the archive.
        _write_session(
            root / '9999' / '-home-leo-src-dark-factory-cockpit',
            'cockpit-session', COCKPIT_CWD, timestamp='2026-07-13T09:30:00.000Z',
        )
        return root

    def test_enumerates_nested_archive_sessions(self, tmp_path):
        archive = self._build_archive(tmp_path / 'archive')
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert {r.path.name for r in records} == {
            'archived-session.jsonl', 'plain-session.jsonl',
        }

    def test_archive_record_fields(self, tmp_path):
        archive = self._build_archive(tmp_path / 'archive')
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        archived = next(r for r in records if r.path.name == 'archived-session.jsonl')
        assert archived.encoded_dir == self.WT_ENC
        assert archived.cwd == WORKTREE_CWD
        assert archived.date == self.TARGET_DATE
        assert archived.size_bytes == archived.path.stat().st_size

    def test_non_member_cockpit_session_excluded(self, tmp_path):
        archive = self._build_archive(tmp_path / 'archive')
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert 'cockpit-session.jsonl' not in {r.path.name for r in records}

    def test_empty_agent_transcript_roots_is_byte_identical(self, tmp_path):
        # A tree with BOTH a projects-root session and a populated archive.
        projects_root = tmp_path / 'projects'
        _write_session(
            projects_root / '-home-leo-src-dark-factory', 'main-session', MAIN_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        self._build_archive(tmp_path / 'archive')

        # No agent_transcript_roots kwarg at all == today's behavior.
        default_records = mod.enumerate_sessions(projects_root, [MAIN_CWD], self.TARGET_DATE)
        # Explicit empty tuple == same (the archive loop does not execute).
        empty_records = mod.enumerate_sessions(
            projects_root, [MAIN_CWD], self.TARGET_DATE, agent_transcript_roots=(),
        )
        assert {r.path.name for r in default_records} == {'main-session.jsonl'}
        assert {r.path.name for r in empty_records} == {'main-session.jsonl'}

    def test_absent_archive_root_yields_nothing_and_does_not_raise(self, tmp_path):
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[tmp_path / 'does-not-exist'],
        )
        assert records == []


class TestResidualGzIsAnnounced:
    """The archive walk only enumerates ``*.jsonl``, so anything still gzipped
    is not skipped-with-a-reason — it is not seen at all.

    That window is real and accepted: the destructive migration sweep is a
    human-operated step (OPERATIONS.md §13), so between this merge and the
    operator's run the corpus under-reports. What is NOT acceptable is the gap
    being invisible, since its duration is bounded only by someone remembering.
    So the walk counts what it cannot see and says so — a count, never a read,
    and it disappears on its own once the migration has run.
    """

    TARGET_DATE = dt_date(2026, 7, 13)
    WT_ENC = '-home-leo-src-dark-factory--worktrees-2573'
    LOGGER = 'legibility.inventory'

    def _archive_with_residue(self, tmp_path: Path, *, residual: int) -> Path:
        root = tmp_path / 'archive'
        enc_dir = root / '2573' / self.WT_ENC
        _write_session(
            enc_dir, 'migrated-session', WORKTREE_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        for i in range(residual):
            # NOT valid gzip on purpose: the count must never open these, so
            # bytes no decompressor would accept are the honest fixture.
            (enc_dir / f'un-migrated-{i}.jsonl.gz').write_bytes(b'not gzip either')
        return root

    def _enumerate(self, tmp_path: Path, archive: Path):
        return mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )

    def test_residual_gz_is_counted_and_announced_once(self, tmp_path, caplog):
        archive = self._archive_with_residue(tmp_path, residual=2)

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            records = self._enumerate(tmp_path, archive)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]
        message = warnings[0].getMessage()
        assert '2' in message
        assert '.jsonl.gz' in message
        # Actionable on its own: it names the fix, not just the symptom.
        assert 'migrate_transcript_archive_gunzip.py' in message
        # And the residue costs nothing else: the migrated session still lands.
        assert {r.path.name for r in records} == {'migrated-session.jsonl'}

    def test_a_fully_migrated_archive_says_nothing(self, tmp_path, caplog):
        archive = self._archive_with_residue(tmp_path, residual=0)

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            self._enumerate(tmp_path, archive)

        # The signal is self-clearing — an operator who ran the sweep must not
        # keep being told about a gap that no longer exists.
        assert [r.getMessage() for r in caplog.records] == []

    def test_count_residual_gz_never_opens_a_file(self, tmp_path):
        # The fixtures are undecompressable, so a count that tried to read
        # them would raise rather than answer. No gzip branch comes back.
        archive = self._archive_with_residue(tmp_path, residual=3)
        assert mod.count_residual_gz(archive) == 3

    def test_count_residual_gz_on_an_absent_root_is_zero(self, tmp_path):
        # Same posture as the walk itself: an archive root that does not exist
        # yet is normal (the tree is git-ignored), not an error.
        assert mod.count_residual_gz(tmp_path / 'does-not-exist') == 0


class TestEnumerateSessionsInRange:
    """enumerate_sessions_in_range walks the projects tree ONCE and keeps
    every session whose date falls in the inclusive ``[start_date, end_date]``
    window — the single-walk O(total_files) replacement for calling
    :func:`enumerate_sessions` once per calendar date (which re-opens each
    file window_days times, O(window_days × files))."""

    START_DATE = dt_date(2026, 7, 12)
    END_DATE = dt_date(2026, 7, 14)

    def test_inclusive_boundaries_both_kept(self, tmp_path):
        # A session dated == start_date AND one dated == end_date are BOTH
        # kept: the window is inclusive on both ends.
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        _write_session(main_dir, 'at-start', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z')
        _write_session(main_dir, 'at-end', MAIN_CWD, timestamp='2026-07-14T23:00:00.000Z')
        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.path.stem for r in records} == {'at-start', 'at-end'}

    def test_out_of_range_excluded(self, tmp_path):
        # One day before start and one day after end are BOTH excluded; a
        # mid-window session is kept.
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        _write_session(main_dir, 'before-start', MAIN_CWD, timestamp='2026-07-11T09:00:00.000Z')
        _write_session(main_dir, 'after-end', MAIN_CWD, timestamp='2026-07-15T09:00:00.000Z')
        _write_session(main_dir, 'in-range', MAIN_CWD, timestamp='2026-07-13T09:00:00.000Z')
        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.path.stem for r in records} == {'in-range'}

    def test_aggregates_across_multiple_encoded_dirs(self, tmp_path):
        # Mirrors TestEnumerateSessions: aggregation spans every matching
        # encoded dir, not just the main one.
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        worktree_dir = projects_root / '-home-leo-src-dark-factory--worktrees-2573'
        _write_session(main_dir, 'main-in-range', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z')
        _write_session(
            worktree_dir, 'worktree-in-range', WORKTREE_CWD, timestamp='2026-07-14T11:00:00.000Z'
        )
        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.encoded_dir for r in records} == {
            '-home-leo-src-dark-factory',
            '-home-leo-src-dark-factory--worktrees-2573',
        }

    def test_single_walk_opens_each_in_range_file_exactly_once(self, tmp_path, monkeypatch):
        # Three in-range dates across the window. A spy that wraps + delegates
        # to _session_cwd_and_date (the single gz-decompress/open point)
        # records each path it is called with: every in-range file must be
        # passed EXACTLY ONCE — proving the range enumerator is O(total_files),
        # not O(window_days × files) (the per-date loop would open each file 3×).
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        _write_session(main_dir, 'day12', MAIN_CWD, timestamp='2026-07-12T09:00:00.000Z')
        _write_session(main_dir, 'day13', MAIN_CWD, timestamp='2026-07-13T09:00:00.000Z')
        _write_session(main_dir, 'day14', MAIN_CWD, timestamp='2026-07-14T09:00:00.000Z')

        real = mod._session_cwd_and_date
        opened = []

        def spy(path):
            opened.append(path)
            return real(path)

        monkeypatch.setattr(mod, '_session_cwd_and_date', spy)

        records = mod.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], self.START_DATE, self.END_DATE
        )
        assert {r.path.stem for r in records} == {'day12', 'day13', 'day14'}
        # Exactly one open per in-range file — no per-date re-walk.
        for stem in ('day12', 'day13', 'day14'):
            path = main_dir / f'{stem}.jsonl'
            assert opened.count(path) == 1, f'{stem} opened {opened.count(path)}× (want 1)'
        # And no extra opens beyond the three in-range files.
        assert len(opened) == 3


class TestArchiveEncPrefilter:
    """The archive-roots walk cheaply pre-filters by the encoded ``<enc>``
    directory — the archive-root-relative ``parts[1]`` — mirroring
    :func:`iter_project_dirs`' superset pre-filter, so a proven-foreign
    ``<enc>`` is skipped WITHOUT a gz-decompress. :func:`is_member` on the
    real cwd remains the SOLE membership authority for lossy false-positives
    (e.g. a ``-cockpit`` sibling that string-startswith the prefix). ``<enc>``
    is ``parts[1]`` for BOTH the main (``<task>/<enc>/<sid>.jsonl``) and
    subagent (``<task>/<enc>/<sid>/subagents/agent-*.jsonl``) layouts —
    never ``session_path.parent.name`` (== ``'subagents'`` for the subagent
    variant, which would wrongly drop every subagent transcript)."""

    TARGET_DATE = dt_date(2026, 7, 13)
    WT_ENC = '-home-leo-src-dark-factory--worktrees-2573'
    OTHER_CWD = '/home/leo/src/other-project'
    OTHER_ENC = '-home-leo-src-other-project'

    @staticmethod
    def _install_open_spy(monkeypatch) -> list[Path]:
        """Wrap+delegate to _session_cwd_and_date (the single gz-decompress/
        open point), recording every path it is called with."""
        real = mod._session_cwd_and_date
        opened: list[Path] = []

        def spy(path):
            opened.append(path)
            return real(path)

        monkeypatch.setattr(mod, '_session_cwd_and_date', spy)
        return opened

    def test_foreign_enc_excluded_and_never_opened(self, tmp_path, monkeypatch):
        # (a) A foreign <enc> (does NOT startswith the encoded MAIN prefix) is
        # excluded AND its path is never passed to the reader — skipped without
        # a gz-decompress by the cheap <enc> pre-filter.
        archive = tmp_path / 'archive'
        _write_session(
            archive / '2573' / self.OTHER_ENC, 'foreign', self.OTHER_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        foreign_path = archive / '2573' / self.OTHER_ENC / 'foreign.jsonl'
        assert records == []
        assert foreign_path not in opened

    def test_member_enc_kept_and_opened(self, tmp_path, monkeypatch):
        # (b) A member <enc> is kept AND its path WAS passed to the reader.
        archive = tmp_path / 'archive'
        member_path = _write_session(
            archive / '2573' / self.WT_ENC, 'member', WORKTREE_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert {r.path.name for r in records} == {'member.jsonl'}
        assert member_path in opened

    def test_lossy_cockpit_false_positive_is_opened_then_is_member_rejected(
        self, tmp_path, monkeypatch
    ):
        # (c) A -cockpit <enc> string-startswith the encoded main prefix (a
        # LOSSY false-positive), so the superset pre-filter admits it as a
        # candidate — it IS opened — but is_member on the real cwd rejects it.
        # The pre-filter is a superset filter; is_member is the sole authority.
        archive = tmp_path / 'archive'
        cockpit_enc = mod.encode_cwd(COCKPIT_CWD)
        cockpit_path = _write_session(
            archive / '9999' / cockpit_enc, 'cockpit', COCKPIT_CWD,
            timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert records == []
        assert cockpit_path in opened

    def test_subagent_layout_member_kept_and_opened(self, tmp_path, monkeypatch):
        # (d) Subagent layout: <archive>/<task>/<enc>/<sid>/subagents/agent-x.jsonl.
        # <enc> is parts[1] (the member WT_ENC), NOT parent.name (== 'subagents',
        # which never encoded-prefix-matches a cwd and would drop EVERY subagent
        # transcript). The member subagent file is kept + opened, and its
        # encoded_dir is the real <enc>, not 'subagents'.
        archive = tmp_path / 'archive'
        sub_dir = archive / '2573' / self.WT_ENC / 'cafe-sid' / 'subagents'
        sub_path = _write_session(
            sub_dir, 'agent-x', WORKTREE_CWD, timestamp='2026-07-13T09:00:00.000Z',
        )
        opened = self._install_open_spy(monkeypatch)
        records = mod.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], self.TARGET_DATE,
            agent_transcript_roots=[archive],
        )
        assert {r.path.name for r in records} == {'agent-x.jsonl'}
        assert sub_path in opened
        record = next(r for r in records if r.path.name == 'agent-x.jsonl')
        assert record.encoded_dir == self.WT_ENC
