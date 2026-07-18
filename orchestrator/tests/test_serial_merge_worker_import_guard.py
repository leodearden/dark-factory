"""Guard test: no NEW imports of the retired-worker test fixture
(``orchestrator/tests/_serial_merge_worker.py``) in orchestrator/tests/*.py
(merge-queue-reliability PRD, task omicron, scope 6).

The legacy serial ``MergeWorker`` was retired from production (task nu/R7b;
see ``test_merge_worker_retired.py``) and now survives only as a frozen
test-local reference fixture, ``_serial_merge_worker.py``, imported by bare
module name (the flat orchestrator/tests/ convention -- see
``conftest.py``'s ``sys.path`` insertion of the tests directory). Nine
existing test files import it today; this guard freezes that surface so no
NEW file starts depending on the retired fixture going forward. As later
cleanup migrates an existing importer off the fixture, ``ALLOWLIST`` should
shrink to match -- see ``test_allowlist_has_no_stale_entries``.

This test uses AST parsing (not regex) so comments and docstrings that
merely mention the module name -- including the fixture's own module
docstring, and other test files' retirement notes -- are never mistaken for
a real import statement.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# _serial_worker_importers(source) -- inline-fixture unit tests.
# ---------------------------------------------------------------------------


def test_serial_worker_importers_flags_module_level_import() -> None:
    """A bare `import _serial_merge_worker` is flagged."""
    source = "import _serial_merge_worker\n"
    assert _serial_worker_importers(source) == [1]


def test_serial_worker_importers_flags_from_import() -> None:
    """A `from _serial_merge_worker import MergeWorker` is flagged."""
    source = "from _serial_merge_worker import MergeWorker\n"
    assert _serial_worker_importers(source) == [1]


def test_serial_worker_importers_flags_function_local_import() -> None:
    """A function-local import is flagged too, not just module-level."""
    source = (
        "def test_something():\n"
        "    from _serial_merge_worker import MergeWorker\n"
        "    return MergeWorker\n"
    )
    assert _serial_worker_importers(source) == [2]


def test_serial_worker_importers_ignores_docstring_mention() -> None:
    """A comment/docstring merely mentioning the module name is NOT flagged
    -- proves the detector is AST-based, not text/regex."""
    source = (
        '"""See _serial_merge_worker.py for the retired worker reference."""\n'
        "# from _serial_merge_worker import MergeWorker  example only\n"
    )
    assert _serial_worker_importers(source) == []


def test_serial_worker_importers_ignores_unrelated_import() -> None:
    """An import of a different bare-name test helper is NOT flagged."""
    source = "from _merge_queue_harness import drive_verify_and_advance\n"
    assert _serial_worker_importers(source) == []


def test_serial_worker_importers_returns_empty_on_syntax_error() -> None:
    """A transiently-unparseable source returns [] rather than raising --
    a syntax error in some other sibling file must never crash the ratchet."""
    source = "def broken(:\n"
    assert _serial_worker_importers(source) == []


# ---------------------------------------------------------------------------
# Required fixture: demonstrate fail-on-new.
# ---------------------------------------------------------------------------


def test_guard_flags_synthetic_new_importer() -> None:
    """A brand-new `_serial_merge_worker` import, under a synthetic filename
    not in ALLOWLIST, must be reported as a violation."""
    synthetic_source = "from _serial_merge_worker import MergeWorker\n"
    synthetic_file = '__synthetic_not_in_allowlist__.py'

    is_violation = (
        bool(_serial_worker_importers(synthetic_source)) and synthetic_file not in ALLOWLIST
    )

    assert is_violation, (
        f'expected the synthetic new import in {synthetic_file!r} to be '
        'flagged as a violation'
    )


# ---------------------------------------------------------------------------
# Tree-scan: the actual ratchet.
# ---------------------------------------------------------------------------


def test_no_new_serial_merge_worker_imports() -> None:
    """No orchestrator test file may introduce a NEW import of the retired
    `_serial_merge_worker` fixture beyond the frozen ALLOWLIST."""
    offenders: list[str] = []

    for py_file in sorted(_TESTS_DIR.rglob('*.py')):
        if py_file.name == _THIS_FILE:
            continue  # skip the guard itself (its docstring/fixtures mention the pattern)
        if '__pycache__' in py_file.parts:
            continue
        rel = str(py_file.relative_to(_TESTS_DIR))
        if rel in ALLOWLIST:
            continue
        source = py_file.read_text(encoding='utf-8')
        for lineno in _serial_worker_importers(source):
            offenders.append(f'{rel}:{lineno}')

    if offenders:
        offender_list = '\n  '.join(offenders)
        raise AssertionError(
            'New import(s) of the retired _serial_merge_worker test fixture '
            'found that are not in ALLOWLIST. The retired serial MergeWorker '
            'is a frozen historical artifact (task nu/R7b) -- write new '
            'tests against SpeculativeMergeWorker instead. If a new test '
            'genuinely needs the serial-specific surface, add the filename '
            'to ALLOWLIST with justification.\n'
            f'\nOffending sites:\n  {offender_list}'
        )


def test_allowlist_has_no_stale_entries() -> None:
    """Every ALLOWLIST filename must still actually import the fixture --
    the ratchet self-tightens as later cleanup migrates tests off it."""
    stale: list[str] = []
    for filename in sorted(ALLOWLIST):
        path = _TESTS_DIR / filename
        assert path.is_file(), f'ALLOWLIST entry {filename!r} does not exist in {_TESTS_DIR}'
        source = path.read_text(encoding='utf-8')
        if not _serial_worker_importers(source):
            stale.append(filename)

    assert not stale, (
        f'ALLOWLIST entries no longer import _serial_merge_worker -- remove '
        f'them (ratchet should shrink): {stale}'
    )
