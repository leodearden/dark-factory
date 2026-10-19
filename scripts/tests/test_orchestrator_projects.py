"""Tests for ``scripts/orchestrator_projects.py``: which project roots the
factory's orchestrator units run, read from a tmp units dir of real unit-file
text. Nothing here reads ``~/.config/systemd/user``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import orchestrator_projects
import pytest
from orchestrator_projects import ProjectTarget, UnitScan, resolve_targets, scan_units

CONFIG_NAME = 'dark-factory-orchestrator.yaml'


def _unit(exec_start: str) -> str:
    return (
        '[Unit]\n'
        'Description=an orchestrator\n'
        '\n'
        '[Service]\n'
        'ExecStartPre=/home/leo/bin/wait-for-port.py --timeout 280 127.0.0.1:8002\n'
        f'ExecStart={exec_start}\n'
        'Restart=on-failure\n'
        '\n'
        '[Install]\n'
        'WantedBy=default.target\n'
    )


def _config(root: Path) -> Path:
    return root / CONFIG_NAME


@pytest.fixture
def units_dir(tmp_path: Path) -> Path:
    units = tmp_path / 'units'
    units.mkdir()
    (units / 'orchestrator-alpha.service').write_text(
        '[Unit]\n'
        'Description=alpha\n'
        '\n'
        '[Service]\n'
        'ExecStart=/home/leo/.local/bin/uv run --frozen --project orchestrator \\\n'
        f'    orchestrator run --config {_config(tmp_path / "alpha")}\n'
    )
    (units / 'orchestrator-beta.service').write_text(
        _unit(f'/usr/bin/uv run orchestrator run --config={_config(tmp_path / "beta")}')
    )
    (units / 'orchestrator-gamma.service').write_text(
        _unit(f'/usr/bin/uv run orchestrator run --config {_config(tmp_path / "gamma")}')
    )
    dropins = units / 'orchestrator-gamma.service.d'
    dropins.mkdir()
    (dropins / 'override.conf').write_text(
        '[Service]\n'
        'ExecStart=\n'
        f'ExecStart=/usr/bin/uv run orchestrator run --config {_config(tmp_path / "gamma2")}\n'
    )
    (units / 'orchestrator-watchdog.service').write_text(
        _unit('/home/leo/src/dark-factory/scripts/orchestrator-watchdog.py')
    )
    (units / 'orchestrator-alpha.socket').write_text(
        '[Socket]\nListenStream=127.0.0.1:8103\n'
    )
    (units / 'orchestrator-watchdog.timer').write_text('[Timer]\nOnUnitActiveSec=5min\n')
    (units / 'orchestrator-x.service.bak-1').write_text(
        _unit(f'/usr/bin/uv run orchestrator run --config {_config(tmp_path / "bak")}')
    )
    return units


def _root(tmp_path: Path, name: str) -> Path:
    return (tmp_path / name).resolve()


class TestScanUnits:
    def test_maps_each_configured_root_to_its_units(self, units_dir: Path, tmp_path: Path) -> None:
        scan = scan_units(units_dir)

        assert dict(scan.roots) == {
            _root(tmp_path, 'alpha'): ('orchestrator-alpha.service',),
            _root(tmp_path, 'beta'): ('orchestrator-beta.service',),
            _root(tmp_path, 'gamma2'): ('orchestrator-gamma.service',),
        }
        assert scan.problems == ()

    def test_every_root_is_absolute(self, units_dir: Path) -> None:
        assert all(root.is_absolute() for root in scan_units(units_dir).roots)

    def test_two_units_naming_one_root_give_one_root(self, tmp_path: Path) -> None:
        units = tmp_path / 'units'
        units.mkdir()
        command = f'/usr/bin/uv run orchestrator run --config {_config(tmp_path / "shared")}'
        (units / 'orchestrator-one.service').write_text(_unit(command))
        (units / 'orchestrator-two.service').write_text(_unit(command))

        scan = scan_units(units)

        assert dict(scan.roots) == {
            _root(tmp_path, 'shared'): ('orchestrator-one.service', 'orchestrator-two.service'),
        }

    def test_a_missing_units_dir_is_a_problem_never_a_clean_scan(self, tmp_path: Path) -> None:
        missing = tmp_path / 'no-such-units'

        scan = scan_units(missing)

        assert dict(scan.roots) == {}
        assert len(scan.problems) == 1
        assert str(missing) in scan.problems[0]

    def test_an_unreadable_unit_is_a_problem_and_the_rest_still_scan(
        self, units_dir: Path, tmp_path: Path
    ) -> None:
        (units_dir / 'orchestrator-bad.service').mkdir()

        scan = scan_units(units_dir)

        assert len(scan.problems) == 1
        assert 'orchestrator-bad.service' in scan.problems[0]
        assert set(scan.roots) == {
            _root(tmp_path, 'alpha'),
            _root(tmp_path, 'beta'),
            _root(tmp_path, 'gamma2'),
        }


class TestResolveTargets:
    def test_no_explicit_roots_gives_every_scanned_root_sorted(
        self, units_dir: Path, tmp_path: Path
    ) -> None:
        targets = resolve_targets(scan_units(units_dir), None)

        assert targets == (
            ProjectTarget(_root(tmp_path, 'alpha'), ('orchestrator-alpha.service',)),
            ProjectTarget(_root(tmp_path, 'beta'), ('orchestrator-beta.service',)),
            ProjectTarget(_root(tmp_path, 'gamma2'), ('orchestrator-gamma.service',)),
        )
        assert [target.sweep_label for target in targets] == [
            'sweep: orchestrator-alpha.service',
            'sweep: orchestrator-beta.service',
            'sweep: orchestrator-gamma.service',
        ]
        assert all(target.has_sweep for target in targets)

    def test_explicit_roots_are_exactly_the_run_set_deduplicated_in_order(
        self, units_dir: Path, tmp_path: Path
    ) -> None:
        alpha = _root(tmp_path, 'alpha')
        other = _root(tmp_path, 'other')

        targets = resolve_targets(scan_units(units_dir), [alpha, other, alpha])

        assert targets == (
            ProjectTarget(alpha, ('orchestrator-alpha.service',)),
            ProjectTarget(other, ()),
        )
        assert targets[1].has_sweep is False
        assert targets[1].sweep_label == 'no sweep: surfaced only'

    def test_a_relative_explicit_root_is_resolved_absolute(
        self, units_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        (target,) = resolve_targets(scan_units(units_dir), [Path('alpha')])

        assert target == ProjectTarget(_root(tmp_path, 'alpha'), ('orchestrator-alpha.service',))

    def test_a_home_relative_explicit_root_is_resolved_absolute(
        self, units_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('HOME', str(tmp_path))

        (target,) = resolve_targets(scan_units(units_dir), [Path('~/beta')])

        assert target == ProjectTarget(_root(tmp_path, 'beta'), ('orchestrator-beta.service',))

    def test_an_empty_scan_with_no_explicit_roots_has_no_targets(self) -> None:
        assert resolve_targets(UnitScan(roots={}, problems=()), None) == ()


class TestArguments:
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        orchestrator_projects.add_arguments(parser)
        return parser

    def test_defaults(self) -> None:
        args = self._parser().parse_args([])

        assert args.project_root is None
        assert args.units_dir == orchestrator_projects.DEFAULT_UNITS_DIR

    def test_project_root_repeats_and_units_dir_is_a_path(self) -> None:
        args = self._parser().parse_args(
            ['--project-root', '/a', '--project-root', '/b', '--units-dir', '/u']
        )

        assert args.project_root == [Path('/a'), Path('/b')]
        assert args.units_dir == Path('/u')

    def test_targets_from_args_applies_the_target_policy(
        self, units_dir: Path, tmp_path: Path
    ) -> None:
        other = _root(tmp_path, 'other')
        args = self._parser().parse_args(
            ['--units-dir', str(units_dir), '--project-root', str(other)]
        )

        targets, scan = orchestrator_projects.targets_from_args(args)

        assert targets == (ProjectTarget(other, ()),)
        assert scan == scan_units(units_dir)

    def test_the_default_units_dir_is_the_user_systemd_dir(self) -> None:
        expected = Path.home() / '.config' / 'systemd' / 'user'

        assert expected == orchestrator_projects.DEFAULT_UNITS_DIR
