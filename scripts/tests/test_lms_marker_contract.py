"""The delivered-check contract for task 3713 (LME-alpha).

PRD-MARKER:local-memory-models-eval serving

This is not a documentation meta-test.  The orchestrator gates this task on
one command::

    git grep -E 'PRD[-]MARKER:local-memory-models-eval serving' -- scripts/

A grep is a cheap gate with an expensive failure mode: it passes as soon as
ONE file matches.  So the day a ninth module lands here unmarked, the gate
stays green while covering strictly less than it did — the check narrows
silently, and nothing anywhere says so.  These tests are what turns that
silent narrowing into a red suite.

The enumeration comes from `git ls-files`, never a filesystem glob, for the
same reason: an untracked scratch file in the working tree would otherwise
either pad the count or fail the sweep depending on nothing but local state,
and the artifact that ships is the COMMITTED one.

The marker is spelled out literally here.  The PRD capability manifest
deliberately spells it `PRD[-]MARKER` so it can never satisfy its own check;
this file is on the other side of that line — it is one of the marked
artifacts, so it carries the real thing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

#: The literal every artifact of this task must carry.
MARKER = 'PRD-MARKER:local-memory-models-eval serving'

#: The delivered-check pattern verbatim from the task metadata.  `[-]` is a
#: one-character class matching a literal hyphen, which is how the manifest
#: writes the marker without matching it.
DELIVERED_CHECK_PATTERN = r'PRD[-]MARKER:local-memory-models-eval serving'

SERVING_DIR = 'scripts/local-model-serving'
TEST_GLOB = 'scripts/tests/test_lms_*.py'

#: Every artifact this task commits.  Pinned as an explicit inventory rather
#: than derived from the directory listing: a derived list would shrink
#: silently along with the directory, so deleting a module would keep the test
#: green while narrowing the gate — the exact failure this file exists to stop.
EXPECTED_SERVING_FILES = {
    'README.md',
    'arms.yaml',
    'install-lms-units.sh',
    'lms-arm@.service',
    'lms_ctl.py',
    'lms_fetch_weights.py',
    'lms_healthcheck.py',
    'lms_manifest.py',
    'lms_serve.py',
    'lms_vram.py',
}


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ['git', *args],
        cwd=root, capture_output=True, text=True, check=True, timeout=60,
    ).stdout


def _repo_root() -> Path:
    return Path(_git('rev-parse', '--show-toplevel').strip())


def _tracked(*pathspecs: str) -> list[str]:
    return sorted(line for line in _git('ls-files', *pathspecs).splitlines() if line)


def _marked_files() -> set[str]:
    """The files the orchestrator's delivered check would actually find."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ['git', 'grep', '-lE', DELIVERED_CHECK_PATTERN, '--', 'scripts/'],
        cwd=root, capture_output=True, text=True, check=False, timeout=60,
    )
    # git grep exits 1 on "no matches", which is a legitimate (and here,
    # failing) outcome rather than an error to raise on.
    assert result.returncode in (0, 1), result.stderr
    return {line for line in result.stdout.splitlines() if line}


# ---------------------------------------------------------------------------
# The enumeration must not be vacuous
# ---------------------------------------------------------------------------


def test_the_serving_directory_has_committed_files():
    """A pathspec matching nothing must FAIL, not pass by having nothing to
    check.  Every per-file assertion below is a no-op over an empty list."""
    tracked = _tracked(SERVING_DIR)

    assert tracked, f'no committed files under {SERVING_DIR}'
    assert len(tracked) >= len(EXPECTED_SERVING_FILES)


def test_this_tasks_test_modules_are_committed():
    tracked = _tracked(TEST_GLOB)

    assert tracked
    assert 'scripts/tests/test_lms_marker_contract.py' in tracked


def test_the_committed_serving_inventory_is_complete():
    """Pins the substrate itself.  A deleted module would otherwise leave the
    grep gate green over a smaller slate."""
    present = {Path(p).name for p in _tracked(SERVING_DIR)}

    assert present >= EXPECTED_SERVING_FILES, (
        f'missing from {SERVING_DIR}: {sorted(EXPECTED_SERVING_FILES - present)}'
    )


# ---------------------------------------------------------------------------
# Every artifact carries the marker
# ---------------------------------------------------------------------------


def test_every_committed_serving_file_carries_the_marker():
    root = _repo_root()
    unmarked = [
        path for path in _tracked(SERVING_DIR)
        if MARKER not in (root / path).read_text(errors='replace')
    ]

    assert not unmarked, (
        f'these committed files carry no {MARKER!r} line: {unmarked}. The '
        "orchestrator's delivered check is a grep, so it stays green while "
        'covering less — add the marker rather than narrowing the gate.'
    )


def test_every_lms_test_module_carries_the_marker():
    root = _repo_root()
    unmarked = [
        path for path in _tracked(TEST_GLOB)
        if MARKER not in (root / path).read_text(errors='replace')
    ]

    assert not unmarked, f'unmarked test modules: {unmarked}'


def test_the_marker_is_the_resolved_literal_not_the_escaped_pattern():
    """`PRD[-]MARKER` is how the capability manifest writes the marker WITHOUT
    matching it.  An artifact copying that spelling would look marked to a
    reader and be invisible to the gate."""
    root = _repo_root()
    escaped = [
        path for path in _tracked(SERVING_DIR)
        if 'PRD[-]MARKER' in (root / path).read_text(errors='replace')
    ]

    assert not escaped, (
        f'{escaped} spell the marker as the escaped pattern, which the '
        'delivered-check grep will never match'
    )


# ---------------------------------------------------------------------------
# The real gate command, run for real
# ---------------------------------------------------------------------------


def test_the_delivered_check_grep_finds_exactly_this_tasks_files():
    """Runs the orchestrator's own command rather than re-implementing it.

    Re-implementing the search in Python would test this file's regex, not the
    gate.  Equality (not merely non-empty) is the point: a hit OUTSIDE this
    task's substrate means the marker has been copied somewhere it does not
    describe, which makes the gate report on a file it does not gate.
    """
    expected = set(_tracked(SERVING_DIR)) | set(_tracked(TEST_GLOB))

    found = _marked_files()

    assert found, (
        'the delivered check found nothing at all — this task would be '
        'recorded as delivering no serving substrate'
    )
    assert found == expected, (
        f'unexpected marked files: {sorted(found - expected)}; '
        f'missing marked files: {sorted(expected - found)}'
    )


# ---------------------------------------------------------------------------
# The check itself has teeth
# ---------------------------------------------------------------------------


def test_an_unmarked_file_is_detected(tmp_path):
    """Proves the assertion above can fail.  A marker check that cannot go red
    is indistinguishable from no check at all."""
    marked = tmp_path / 'marked.py'
    marked.write_text(f'# {MARKER}\n')
    unmarked = tmp_path / 'unmarked.py'
    unmarked.write_text('# nothing here\n')

    missing = [p.name for p in sorted(tmp_path.iterdir())
               if MARKER not in p.read_text()]

    assert missing == ['unmarked.py']


@pytest.mark.parametrize('impostor', ['PRD[-]MARKER:local-memory-models-eval serving',
                                      'PRD-MARKER:local-memory-models-eval',
                                      'PRD-MARKER: local-memory-models-eval serving'])
def test_a_near_miss_spelling_does_not_satisfy_the_marker(impostor):
    """The gate matches one exact string.  A near miss reads as marked to a
    human and is invisible to the grep, which is the worst of both."""
    assert MARKER not in impostor
