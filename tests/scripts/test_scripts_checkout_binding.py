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
_DECOY = 'decoy'
_BIND = 'import _checkout as _checkout\n'
_PRINT_MEMBER_FILES = 'import sampler, shared\nprint(shared.__file__)\nprint(sampler.__file__)\n'


def _decoy(tmp_path: Path) -> Path:
    decoy = tmp_path / _DECOY
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


def _python(
    args: list[str], *, decoy: Path, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-s', *args],
        cwd=cwd,
        env=_env(decoy),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _run_probe(checkout: Path, source: str) -> subprocess.CompletedProcess[str]:
    probe = checkout / 'scripts' / 'probe.py'
    probe.write_text(source, encoding='utf-8')
    return _python([str(probe)], decoy=checkout.parent / _DECOY)


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
        assert _lies_under(shared_file, tmp_path / _DECOY), shared_file
        assert _lies_under(sampler_file, tmp_path / _DECOY), sampler_file


_MEMBERS_ONE_WITHOUT_SRC = '[tool.uv.workspace]\nmembers = ["shared", "no-src", "sampler"]\n'


def test_member_src_is_the_declared_members_with_a_src_in_order_and_prefixes_sys_path(
    tmp_path: Path,
) -> None:
    checkout = _layout(tmp_path, pyproject=_MEMBERS_ONE_WITHOUT_SRC)
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
    assert reported['member_src'] == [str(root / 'shared' / 'src'), str(root / 'sampler' / 'src')]
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
    'non-string-member': '[tool.uv.workspace]\nmembers = ["shared", 1]\n',
    'empty-string-member': '[tool.uv.workspace]\nmembers = ["shared", ""]\n',
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


_BOUND_SCRIPTS = (
    'merge_lane_metrics',
    'quality_metrics_snapshot',
    'inline_suppression_classify',
    'inline_suppressions',
    'gen_dashboard_task_vocab',
    'suite_census',
    'trial_module_tagger_haiku',
)

_IMPORT_THEN_PRINT_MEMBER_FILES = (
    'import importlib, sys; importlib.import_module(sys.argv[1]); '
    'import sampler, shared; print(shared.__file__); print(sampler.__file__)'
)


@pytest.mark.parametrize('module', _BOUND_SCRIPTS)
def test_a_script_imports_the_members_of_its_own_checkout(tmp_path: Path, module: str) -> None:
    proc = _python(
        ['-c', _IMPORT_THEN_PRINT_MEMBER_FILES, module],
        decoy=_decoy(tmp_path),
        cwd=REPO_ROOT / 'scripts',
    )

    assert proc.returncode == 0, proc.stderr
    shared_file, sampler_file = proc.stdout.splitlines()[-2:]
    assert _lies_under(shared_file, REPO_ROOT / 'shared' / 'src'), shared_file
    assert _lies_under(sampler_file, REPO_ROOT / 'sampler' / 'src'), sampler_file


_SAFE_IO_REPORTS_ITS_FILE = "import sys\nprint('safe_io:' + __file__, file=sys.stderr)\n"


def test_merge_lane_metrics_run_from_a_copy_uses_the_copys_shared(tmp_path: Path) -> None:
    checkout = _layout(
        tmp_path, scripts=('_checkout.py', 'merge_lane_metrics.py', 'source_measures.py')
    )
    for shared_package in (checkout / 'shared' / 'src' / 'shared', tmp_path / _DECOY / 'shared'):
        (shared_package / 'safe_io.py').write_text(_SAFE_IO_REPORTS_ITS_FILE, encoding='utf-8')

    proc = _python(
        [str(checkout / 'scripts' / 'merge_lane_metrics.py'), '--help'], decoy=tmp_path / _DECOY
    )

    assert proc.returncode == 0, proc.stderr
    loaded = [
        line.removeprefix('safe_io:')
        for line in proc.stderr.splitlines()
        if line.startswith('safe_io:')
    ]
    assert len(loaded) == 1, proc.stderr
    assert _lies_under(loaded[0], checkout / 'shared' / 'src'), loaded[0]


def test_the_ratchet_commit_gate_reports_an_unbindable_checkout_as_an_instrument_failure(
    tmp_path: Path,
) -> None:
    gate_scripts = (
        '_checkout.py',
        'merge_lane_metrics.py',
        'source_measures.py',
        'check_staged_ratchet_raise.py',
    )
    checkout = _layout(tmp_path, pyproject=None, scripts=gate_scripts)

    proc = _python(
        [str(checkout / 'scripts' / 'check_staged_ratchet_raise.py'), '--root', str(checkout)],
        decoy=tmp_path / _DECOY,
    )

    assert proc.returncode == 2, proc.stderr
    assert 'Traceback' not in proc.stderr
    assert 'CheckoutBindingError' in proc.stderr
    assert str(checkout.resolve() / 'pyproject.toml') in proc.stderr
    assert 'scripts/_checkout.py' in proc.stderr
