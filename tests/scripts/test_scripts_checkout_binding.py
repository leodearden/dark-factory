"""``scripts/_checkout.py`` binds the importing script's own checkout.

Contract: ``plans/review-all-s25-scripts-instruments-prd.md`` C1/D2. Every test
runs a real interpreter against a COPY of the module laid out as a checkout under
``tmp_path``, with a decoy ``shared``/``sampler`` on ``PYTHONPATH`` outside it;
nothing here imports ``_checkout`` in-process.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_REAL_PYPROJECT = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
_STUB_MEMBERS: Mapping[str, str] = MappingProxyType({'shared': 'shared', 'sampler': 'sampler'})
_DECOY_PACKAGES = ('shared', 'sampler')
_BIND = 'import _checkout as _checkout\n'
_PRINT_MEMBER_FILES = 'import sampler, shared\nprint(shared.__file__)\nprint(sampler.__file__)\n'


def _decoy(tmp_path: Path) -> Path:
    decoy = tmp_path / 'decoy'
    for package in _DECOY_PACKAGES:
        (decoy / package).mkdir(parents=True)
        (decoy / package / '__init__.py').write_text('', encoding='utf-8')
    return decoy


def _layout(
    tmp_path: Path,
    *,
    pyproject: str | None = _REAL_PYPROJECT,
    scripts: tuple[str, ...] = ('_checkout.py',),
    member_pkgs: Mapping[str, str] = _STUB_MEMBERS,
) -> Path:
    checkout = tmp_path / 'checkout'
    (checkout / 'scripts').mkdir(parents=True)
    if pyproject is not None:
        (checkout / 'pyproject.toml').write_text(pyproject, encoding='utf-8')
    for name in scripts:
        shutil.copy2(REPO_ROOT / 'scripts' / name, checkout / 'scripts' / name)
    for member, package in member_pkgs.items():
        package_dir = checkout / member / 'src' / package
        package_dir.mkdir(parents=True)
        (package_dir / '__init__.py').write_text('', encoding='utf-8')
    _decoy(tmp_path)
    return checkout


def _env(decoy: Path) -> dict[str, str]:
    dropped = {'PYTHONPATH', 'PYTHONSAFEPATH', 'VIRTUAL_ENV'}
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env['PYTHONPATH'] = str(decoy)
    env['PYTHONNOUSERSITE'] = '1'
    return env


def _run_probe(checkout: Path, source: str) -> subprocess.CompletedProcess[str]:
    probe = checkout / 'scripts' / 'probe.py'
    probe.write_text(source, encoding='utf-8')
    return subprocess.run(
        [sys.executable, '-s', str(probe)],
        env=_env(checkout.parent / 'decoy'),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _lies_under(path: str, directory: Path) -> bool:
    return Path(path).resolve().is_relative_to(directory.resolve())


@pytest.mark.parametrize('bound', [True, False], ids=['bound', 'unbound-control'])
def test_binding_beats_a_decoy_the_environment_resolves(tmp_path: Path, bound: bool) -> None:
    checkout = _layout(tmp_path)
    proc = _run_probe(checkout, (_BIND if bound else '') + _PRINT_MEMBER_FILES)

    assert proc.returncode == 0, proc.stderr
    shared_file, sampler_file = proc.stdout.splitlines()
    if bound:
        assert _lies_under(shared_file, checkout / 'shared' / 'src'), shared_file
        assert _lies_under(sampler_file, checkout / 'sampler' / 'src'), sampler_file
    else:
        assert _lies_under(shared_file, tmp_path / 'decoy'), shared_file
        assert _lies_under(sampler_file, tmp_path / 'decoy'), sampler_file


def test_member_src_is_the_declared_members_with_a_src_in_order_and_prefixes_sys_path(
    tmp_path: Path,
) -> None:
    checkout = _layout(tmp_path)
    proc = _run_probe(
        checkout,
        'import json, sys\n'
        + _BIND
        + 'member_src = [str(p) for p in _checkout.MEMBER_SRC]\n'
        "print(json.dumps({'root': str(_checkout.ROOT), 'member_src': member_src,"
        " 'prefix': sys.path[:len(member_src)]}))\n",
    )

    assert proc.returncode == 0, proc.stderr
    reported = json.loads(proc.stdout)
    root = checkout.resolve()
    assert reported['root'] == str(root)
    assert reported['member_src'] == [str(root / 'sampler' / 'src'), str(root / 'shared' / 'src')]
    assert reported['prefix'] == reported['member_src']


def test_reloading_the_binding_leaves_sys_path_unchanged(tmp_path: Path) -> None:
    checkout = _layout(tmp_path)
    proc = _run_probe(
        checkout,
        'import importlib, sys\n'
        + _BIND
        + 'before = list(sys.path)\n'
        'importlib.reload(_checkout)\n'
        'print(sys.path == before)\n',
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'True'


_UNBINDABLE_PYPROJECTS = {
    'absent': None,
    'invalid-toml': 'members = [',
    'no-workspace-table': '[project]\nname = "x"\n',
    'empty-members': '[tool.uv.workspace]\nmembers = []\n',
    'members-not-a-list': '[tool.uv.workspace]\nmembers = "shared"\n',
    'glob-member': '[tool.uv.workspace]\nmembers = ["pkgs/*", "shared"]\n',
}

_REPORT_BINDING_FAILURE = (
    'import json, sys\n'
    'try:\n'
    '    import _checkout as _checkout\n'
    'except ImportError as exc:\n'
    "    print(json.dumps({'name': type(exc).__name__,"
    " 'runtime_error': isinstance(exc, RuntimeError), 'path': exc.path, 'message': str(exc)}))\n"
    '    sys.exit(3)\n'
)


@pytest.mark.parametrize(
    'pyproject', _UNBINDABLE_PYPROJECTS.values(), ids=_UNBINDABLE_PYPROJECTS.keys()
)
def test_an_unbindable_checkout_refuses_to_import(tmp_path: Path, pyproject: str | None) -> None:
    checkout = _layout(tmp_path, pyproject=pyproject)
    proc = _run_probe(checkout, _REPORT_BINDING_FAILURE)

    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    reported = json.loads(proc.stdout)
    pyproject_path = str(checkout.resolve() / 'pyproject.toml')
    assert reported['name'] == 'CheckoutBindingError'
    assert reported['runtime_error'] is True
    assert reported['path'] == pyproject_path
    assert pyproject_path in reported['message']
