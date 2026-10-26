"""A uv venv moved after it was built is retired so ``uv run`` rebuilds it; nothing else is touched."""

from __future__ import annotations

import errno
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from _venv_fixtures import make_venv, stale_script

from orchestrator.relocated_venv import Disposition, RelocatedVenv, retire_relocated_venvs

_LOGGER = 'orchestrator.relocated_venv'


@pytest.fixture
def root(tmp_path: Path) -> Path:
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    return checkout


@pytest.fixture
def old_build(tmp_path: Path) -> Path:
    return tmp_path / 'old-checkout' / '.venv'


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _LOGGER and r.levelno == logging.WARNING
    ]


def test_relocated_uv_venv_is_retired_and_reported(
    root: Path, old_build: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    venv = make_venv(root / '.venv', built_at=old_build)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        found = retire_relocated_venvs(root)

    assert found == (RelocatedVenv(
        path=venv, built_at=(old_build,), scripts=('mypy', 'pytest'),
        disposition=Disposition.RETIRED,
    ),)
    assert not venv.exists()
    assert any(str(venv) in m and str(old_build) in m for m in _warnings(caplog))


def test_healthy_venv_is_left_intact(root: Path) -> None:
    venv = make_venv(root / '.venv')
    sentinel = venv / 'sentinel'
    sentinel.write_text('keep')

    assert retire_relocated_venvs(root) == ()
    assert sentinel.read_text() == 'keep'


def test_checkout_without_a_venv_reports_nothing(root: Path) -> None:
    assert retire_relocated_venvs(root) == ()


def test_relocated_venv_not_managed_by_uv_is_kept(
    root: Path, old_build: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    venv = make_venv(root / '.venv', built_at=old_build, uv_managed=False)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        found = retire_relocated_venvs(root)

    assert [f.disposition for f in found] == [Disposition.KEPT_NOT_UV_MANAGED]
    assert (venv / 'bin' / 'pytest').exists()
    assert any(str(venv) in m for m in _warnings(caplog))


def test_venv_of_the_running_interpreter_is_kept(root: Path, old_build: Path) -> None:
    venv = make_venv(root / '.venv', built_at=old_build)

    found = retire_relocated_venvs(root, running_prefix=venv)

    assert [f.disposition for f in found] == [Disposition.KEPT_RUNNING_INTERPRETER]
    assert (venv / 'bin' / 'pytest').exists()


def test_script_naming_a_missing_system_interpreter_is_not_a_relocation(root: Path) -> None:
    venv = make_venv(root / '.venv')
    stale_script(venv, 'legacy-tool', '/nonexistent-7077/usr/bin/python3')

    assert retire_relocated_venvs(root) == ()
    assert (venv / 'bin' / 'legacy-tool').exists()


def test_run_dir_venvs_are_inspected_alongside_the_root_venv(
    root: Path, old_build: Path,
) -> None:
    root_venv = make_venv(root / '.venv')
    pkg_venv = make_venv(root / 'pkg' / '.venv', built_at=old_build)

    found = retire_relocated_venvs(root, ('pkg',))

    assert [(f.path, f.disposition) for f in found] == [(pkg_venv, Disposition.RETIRED)]
    assert not pkg_venv.exists()
    assert (root_venv / 'bin' / 'pytest').exists()


@pytest.fixture
def unremovable_relocated_venv(root: Path, old_build: Path) -> Iterator[Path]:
    venv = make_venv(root / '.venv', built_at=old_build)
    bin_dir = venv / 'bin'
    bin_dir.chmod(0o555)
    yield venv
    bin_dir.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores the read-only bin dir')
def test_failed_removal_is_reported_not_raised(
    root: Path, unremovable_relocated_venv: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    venv = unremovable_relocated_venv

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        found = retire_relocated_venvs(root)

    assert [(f.path, f.disposition) for f in found] == [(venv, Disposition.RETIRE_FAILED)]
    assert (venv / 'bin' / 'pytest').exists()
    assert any(
        str(venv) in m and f'[Errno {errno.EACCES}]' in m for m in _warnings(caplog)
    )


def test_symlinked_venv_is_ignored(root: Path, tmp_path: Path, old_build: Path) -> None:
    target = make_venv(tmp_path / 'elsewhere' / '.venv', built_at=old_build)
    link = root / '.venv'
    os.symlink(target, link)

    assert retire_relocated_venvs(root) == ()
    assert link.is_symlink()
    assert (target / 'bin' / 'pytest').exists()
