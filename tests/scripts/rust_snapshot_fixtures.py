"""The test harness of ``scripts/rust_metrics_snapshot.py``: fixture rules, tokei and cargo shims, a Cargo repo.

The shims go first on PATH, so a verify host needs neither binary.

IMPORT ME, DO NOT COPY ME. Importable from ``tests/scripts/test_*.py`` because
``tests/scripts/conftest.py`` puts this directory on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from git_listing import git

#: Two areas, one per fixture crate; nothing excluded beyond docs/.
FIXTURE_RULES_TOML = """\
exclude = '^docs/'
test = '(^|/)tests?/'
lizard_excludes = ['./target/*']
fix_density_since = 2026-07-06

[[area]]
pattern = '^crates/core/'
area = 'core'

[[area]]
pattern = '^crates/app/'
area = 'app'

[inferred]
"""


def write_fixture_rules(path: Path) -> Path:
    """The fixture rules written to *path*, its parent created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FIXTURE_RULES_TOML, encoding='utf-8')
    return path


#: Set to '1' in the environment, the tokei shim commits in its cwd before counting: a merge landing mid-measurement.
MOVES_HEAD_ENV = 'FAKE_TOKEI_MOVES_HEAD'

_TOKEI_SHIM = '''
import json
import os
import subprocess
import sys
from pathlib import Path


def _stats(text):
    lines = text.splitlines()
    comments = sum(1 for line in lines if line.strip().startswith('//'))
    blanks = sum(1 for line in lines if not line.strip())
    return {'code': len(lines) - comments - blanks, 'comments': comments, 'blanks': blanks, 'blobs': {}}


def _report():
    reports = [
        {'name': f'./{path.as_posix()}', 'stats': _stats(path.read_text(encoding='utf-8'))}
        for path in sorted(Path('.').rglob('*.rs'))
        if '.git' not in path.parts
    ]
    return {'Rust': {'reports': reports}, 'Total': {}}


args = sys.argv[1:]
if args == ['--version']:
    print(f'tokei {VERSION} compiled with serialization support: json')
elif args == ['-f', '-o', 'json', '.']:
    if os.environ.get(MOVES_HEAD_ENV) == '1':
        subprocess.run(
            ['git', '-c', 'user.name=f', '-c', 'user.email=f@e.invalid',
             'commit', '--allow-empty', '--no-verify', '-q', '-m', 'moved'],
            check=True,
        )
    print(json.dumps(_report()))
else:
    sys.exit(f'tokei shim: unexpected argv {args}')
'''

_CARGO_SHIM = '''
import json
import sys
from pathlib import Path

args = sys.argv[1:]
if args == ['metadata', '--no-deps', '--format-version', '1']:
    packages = [
        {'name': manifest.parent.name, 'manifest_path': str(manifest.resolve())}
        for manifest in sorted(Path('.').glob('crates/*/Cargo.toml'))
    ]
    print(json.dumps({'packages': packages}))
else:
    sys.exit(f'cargo shim: unexpected argv {args}')
'''


def _write_shim(path: Path, body: str) -> None:
    path.write_text(f'#!{sys.executable}\n{body}', encoding='utf-8')
    path.chmod(0o755)


def tool_shims(bin_dir: Path, *, tokei_version: str = '15.0.0') -> Path:
    """Executable ``tokei`` and ``cargo`` stand-ins in *bin_dir*, answering only the instrument's argv.

    tokei counts every ``.rs`` file under its cwd (``//`` lines as comments), first
    committing there when ``MOVES_HEAD_ENV`` is '1'; cargo lists one package per
    ``crates/<name>/Cargo.toml`` under its cwd.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_shim(bin_dir / 'tokei', f'VERSION = {tokei_version!r}\nMOVES_HEAD_ENV = {MOVES_HEAD_ENV!r}\n{_TOKEI_SHIM}')
    _write_shim(bin_dir / 'cargo', _CARGO_SHIM)
    return bin_dir


#: A two-crate Cargo workspace: core's lib holds two ifs and a pub(crate) fn, app uses core.
RUST_WORKSPACE_FILES: dict[str, str] = {
    'Cargo.toml': '[workspace]\nmembers = ["crates/core", "crates/app"]\n',
    'crates/core/Cargo.toml': '[package]\nname = "core"\nversion = "0.1.0"\n',
    'crates/core/src/lib.rs': (
        '// The fixture library.\n'
        'pub fn classify(a: bool, b: bool) -> u8 {\n'
        '    let mut x = 0;\n'
        '    if a {\n'
        '        x = 1;\n'
        '    }\n'
        '    if b {\n'
        '        x = 2;\n'
        '    }\n'
        '    x\n'
        '}\n'
        '\n'
        'pub(crate) fn helper() -> u8 {\n'
        '    1\n'
        '}\n'
    ),
    'crates/core/tests/t.rs': '#[test]\nfn classifies() {\n    assert_eq!(core::classify(true, false), 1);\n}\n',
    'crates/app/Cargo.toml': '[package]\nname = "app"\nversion = "0.1.0"\n',
    'crates/app/src/main.rs': 'use core::classify;\n\nfn main() {\n    let _ = classify(true, false);\n}\n',
}

#: Two module cycles in core, a <-> b and c <-> d, both entered from lib.rs, so the order
#: Tarjan's walk finds them in follows the iteration order of lib.rs's successor set.
RUST_CYCLE_FILES: dict[str, str] = {
    'crates/core/src/lib.rs': 'mod a;\nmod b;\nmod c;\nmod d;\n\nuse crate::a::A;\nuse crate::c::C;\n',
    'crates/core/src/a.rs': 'use crate::b::B;\n\npub struct A;\n',
    'crates/core/src/b.rs': 'use crate::a::A;\n\npub struct B;\n',
    'crates/core/src/c.rs': 'use crate::d::D;\n\npub struct C;\n',
    'crates/core/src/d.rs': 'use crate::c::C;\n\npub struct D;\n',
}

_COMMITTER = ('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid')


def rust_workspace_repo(tmp_path: Path, files: dict[str, str] = RUST_WORKSPACE_FILES) -> Path:
    """A committed Cargo workspace of *files* under *tmp_path* (no pyproject.toml) whose one branch is main."""
    root = tmp_path / 'repo'
    for relpath, text in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, *_COMMITTER, 'commit', '--no-verify', '-q', '-m', 'base')
    git(root, 'branch', '-M', 'main')
    return root
