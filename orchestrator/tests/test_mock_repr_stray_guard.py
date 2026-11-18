"""conftest.py's ``_no_mock_repr_stray_entries`` fails the test that leaves a new
``<...Mock ...>`` entry in the CWD and removes that entry, while entries that
predate the test are neither blamed nor touched.

Both arms run the probe module in a REAL nested pytest session whose CWD is a
fresh tmp directory, so the deliberate leak never lands in the checkout.  The
control arm disables conftest.py and proves the probe genuinely leaks; the
treatment arm is the shipped configuration, read back per testcase from its
JUnit report.
"""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from _probe_session import run_probe_session

_PROBE = Path(__file__).with_name('_mock_repr_stray_probe.py')
_KILL_SECS = 60
_MOCK_REPR_GLOB = '<*Mock *'
_LEAKING_TESTS = ('test_leaks_a_mock_repr_sqlite_file', 'test_leaks_a_mock_repr_directory_tree')
_CLEAN_TEST = 'test_leaves_the_cwd_alone'
_GUARD_CAUSE_PHRASE = 'mock-repr'
_PRE_EXISTING_STRAY = "<MagicMock name='pre-existing' id='0'>"


def _nested_session(tmp_path: Path, cwd: Path, report: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run_probe_session(
        _PROBE, basetemp=tmp_path / 'probe-basetemp', kill_secs=_KILL_SECS,
        backstop_secs=_KILL_SECS * 4, cwd=cwd, extra=(f'--junitxml={report}', *extra),
    )


def _fresh_cwd(tmp_path: Path) -> Path:
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    return cwd


def _mock_repr_names(directory: Path) -> set[str]:
    return {entry.name for entry in directory.glob(_MOCK_REPR_GLOB)}


def _mock_repr_kinds(directory: Path) -> set[str]:
    return {'dir' if entry.is_dir() else 'file' for entry in directory.glob(_MOCK_REPR_GLOB)}


def _testcase_children(report: Path) -> dict[str, list[ET.Element]]:
    return {
        case.attrib['name']: list(case)
        for case in ET.parse(report).getroot().iter('testcase')
    }


def test_without_the_guard_the_probe_leaves_a_mock_repr_file_and_directory(
    tmp_path: Path,
) -> None:
    cwd = _fresh_cwd(tmp_path)
    run = _nested_session(tmp_path, cwd, tmp_path / 'report.xml', '--noconftest')
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
    assert _mock_repr_kinds(cwd) == {'file', 'dir'}, (
        'the probe must leak a <MagicMock ...> file AND directory when no guard '
        'runs, or the treatment arm passes vacuously for one removal branch; '
        f'CWD holds {sorted(p.name for p in cwd.iterdir())}'
    )


def test_with_the_guard_the_leak_fails_its_test_and_only_new_entries_are_removed(
    tmp_path: Path,
) -> None:
    cwd = _fresh_cwd(tmp_path)
    (cwd / _PRE_EXISTING_STRAY).touch()
    report = tmp_path / 'report.xml'

    run = _nested_session(tmp_path, cwd, report)

    diagnostics = run.stdout[-3000:] + run.stderr[-2000:]
    assert run.returncode != 0, diagnostics
    assert report.is_file(), diagnostics
    children = _testcase_children(report)
    assert {*_LEAKING_TESTS, _CLEAN_TEST} <= children.keys(), diagnostics
    for name in _LEAKING_TESTS:
        leak_errors = [c for c in children[name] if c.tag == 'error']
        assert leak_errors, diagnostics
        error_text = ''.join(leak_errors[0].itertext()) + leak_errors[0].attrib.get('message', '')
        assert f'_mock_repr_stray_probe.py::{name}' in error_text, error_text
        assert _GUARD_CAUSE_PHRASE in error_text, error_text
    assert not [c for c in children[_CLEAN_TEST] if c.tag in {'failure', 'error'}], diagnostics
    assert _mock_repr_names(cwd) == {_PRE_EXISTING_STRAY}, sorted(p.name for p in cwd.iterdir())
