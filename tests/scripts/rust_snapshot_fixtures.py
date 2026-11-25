"""The test harness of ``scripts/rust_metrics_snapshot.py``: a minimal rules file for a two-crate fixture.

IMPORT ME, DO NOT COPY ME. Importable from ``tests/scripts/test_*.py`` because
``tests/scripts/conftest.py`` puts this directory on ``sys.path``.
"""
from __future__ import annotations

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
