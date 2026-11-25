"""The test harness of ``scripts/rust_metrics_snapshot.py``: fixture rules, and tokei and cargo shims.

The shims go first on PATH, so a verify host needs neither binary.

IMPORT ME, DO NOT COPY ME. Importable from ``tests/scripts/test_*.py`` because
``tests/scripts/conftest.py`` puts this directory on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

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


_TOKEI_SHIM = '''
import json
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

    tokei counts every ``.rs`` file under its cwd (``//`` lines as comments);
    cargo lists one package per ``crates/<name>/Cargo.toml`` under its cwd.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_shim(bin_dir / 'tokei', f'VERSION = {tokei_version!r}\n{_TOKEI_SHIM}')
    _write_shim(bin_dir / 'cargo', _CARGO_SHIM)
    return bin_dir
