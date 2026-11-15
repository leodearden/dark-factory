"""scripts/expected_user_timers.py: the declared registry of user timers this repo expects armed."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from expected_user_timers import ARMED_ELSEWHERE, EXPECTED_TIMERS, ExpectedTimer, InstallerArg
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

CHECKOUT = Path(__file__).resolve().parents[2]
GIT_TIMEOUT_SECS = 60


@pytest.mark.parametrize(
    ('unit_file', 'installer_arg', 'rule'),
    [
        ('scripts/foo.service', InstallerArg.NONE, '.timer'),
        ('scripts/foo@.timer', InstallerArg.NONE, 'template'),
        ('scripts/foo.timer', InstallerArg.PROJECT_ID, 'template'),
        ('scripts/foo.timer', InstallerArg.PROJECT_ROOT, 'template'),
    ],
)
def test_construction_rejects_a_broken_entry_naming_the_rule_and_values(
    unit_file: str, installer_arg: InstallerArg, rule: str,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        ExpectedTimer(unit_file=unit_file, installer='scripts/install-foo.sh', installer_arg=installer_arg)
    message = str(excinfo.value)
    assert rule in message
    assert unit_file in message
    if unit_file.endswith('.timer'):
        assert installer_arg.name in message


def test_unit_name_of_a_singleton_is_its_basename() -> None:
    timer = ExpectedTimer('scripts/fused-memory-link-heal.timer', 'scripts/install-link-heal-timer.sh')
    assert timer.unit_name(project_id='dark_factory') == 'fused-memory-link-heal.timer'
    assert not timer.is_template


def test_unit_name_of_a_template_binds_the_project_instance() -> None:
    timer = ExpectedTimer(
        'scripts/quality-snapshot@.timer', 'scripts/install-quality-snapshot-timer.sh', InstallerArg.PROJECT_ROOT,
    )
    assert timer.is_template
    assert timer.unit_name(project_id='dark_factory') == 'quality-snapshot@dark_factory.timer'


def test_unit_name_of_a_template_refuses_an_empty_project_id() -> None:
    timer = ExpectedTimer(
        'scripts/legibility-trickle@.timer', 'scripts/legibility/install-trickle-timer.sh', InstallerArg.PROJECT_ID,
    )
    with pytest.raises(ValueError, match='legibility-trickle@.timer'):
        timer.unit_name(project_id='')


@pytest.mark.parametrize(
    ('unit_file', 'installer_arg', 'expected_args'),
    [
        ('scripts/x.timer', InstallerArg.NONE, ()),
        ('scripts/x@.timer', InstallerArg.PROJECT_ID, ('dark_factory',)),
        ('scripts/x@.timer', InstallerArg.PROJECT_ROOT, ('/r/dark-factory',)),
    ],
)
def test_installer_argv_is_the_absolute_installer_and_its_one_argument(
    unit_file: str, installer_arg: InstallerArg, expected_args: tuple[str, ...],
) -> None:
    timer = ExpectedTimer(unit_file, 'scripts/install-x.sh', installer_arg)
    argv = timer.installer_argv(repo_root=Path('/r/dark-factory'), project_id='dark_factory')
    assert argv == ('/r/dark-factory/scripts/install-x.sh', *expected_args)


def _committed_timers() -> set[str]:
    listing = subprocess.run(
        ['git', '-C', str(CHECKOUT), 'ls-files', '-z', '--', '*.timer'],
        capture_output=True, text=True, check=True, timeout=GIT_TIMEOUT_SECS,
    )
    return {path for path in listing.stdout.split('\0') if path}


def test_every_committed_timer_is_declared_expected_or_armed_elsewhere() -> None:
    committed = _committed_timers()
    expected = {timer.unit_file for timer in EXPECTED_TIMERS}
    elsewhere = set(ARMED_ELSEWHERE)
    assert committed, f'git ls-files found no *.timer under {CHECKOUT}; the guard would pass vacuously'
    unclassified = sorted(committed - expected - elsewhere)
    assert not unclassified, (
        f'committed timers in neither EXPECTED_TIMERS nor ARMED_ELSEWHERE: {unclassified}; add an '
        f'EXPECTED_TIMERS entry naming the installer that arms each one, or an ARMED_ELSEWHERE reason '
        f'in scripts/expected_user_timers.py'
    )
    stale = sorted((expected | elsewhere) - committed)
    assert not stale, f'scripts/expected_user_timers.py names timers git does not track: {stale}'
    assert not expected & elsewhere, f'declared both expected and armed elsewhere: {sorted(expected & elsewhere)}'


def test_no_two_declared_timers_share_a_unit_name() -> None:
    """systemd knows a unit by its basename alone, so two timers sharing one would overwrite each other."""
    declared = [timer.unit_file for timer in EXPECTED_TIMERS] + list(ARMED_ELSEWHERE)
    unit_names = [Path(unit_file).name for unit_file in declared]
    clashes = sorted(f for f in declared if unit_names.count(Path(f).name) > 1)
    assert not clashes, f'declared timers sharing a unit name: {clashes}'


@pytest.mark.parametrize('timer', EXPECTED_TIMERS, ids=lambda timer: timer.unit_file)
def test_each_declared_installer_exists_is_executable_and_names_its_timer(timer: ExpectedTimer) -> None:
    unit_file = CHECKOUT / timer.unit_file
    installer = CHECKOUT / timer.installer
    assert unit_file.is_file(), f'{timer.unit_file} is declared but absent from the checkout'
    assert installer.is_file(), f'{timer.installer} (installer of {timer.unit_file}) is absent'
    assert os.access(installer, os.X_OK), f'{timer.installer} is not executable'
    assert Path(timer.unit_file).name in installer.read_text(), (
        f'{timer.installer} never names {Path(timer.unit_file).name}, so it cannot be what arms it'
    )


@pytest.mark.parametrize('unit_file', sorted(ARMED_ELSEWHERE))
def test_each_timer_armed_elsewhere_states_its_reason(unit_file: str) -> None:
    reason = ARMED_ELSEWHERE[unit_file]
    assert isinstance(reason, str)
    assert reason.strip()
