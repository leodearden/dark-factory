"""The canonical lock-charter vectors that shared.locking's two test suites pin.

Test-support module, NOT production code. Consumed by
``shared/tests/test_locking.py``'s corpus-coverage tests and by the drift
guards in ``tests/repo_guards/test_lock_charter_reify_parity.py``, which must
pin the SAME lists rather than two copies of them. Importable bare
(``from lock_charter_vectors import ...``) because ``shared/tests/conftest.py``
and ``tests/repo_guards/conftest.py`` both insert this ``cross_suite_helpers/``
directory onto ``sys.path``. The names keep their underscore spelling because
operator-facing messages name them, e.g.
``fused-memory/src/fused_memory/middleware/lock_charter_guard.py::_allowlist_lockstep``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Drift guard — pins shared.locking.FILE_EXTENSIONS to the canonical vector.
#
# Update _CANONICAL_EXTENSIONS AND FILE_EXTENSIONS together when the allowlist
# changes.  Also update the verbatim copy in
# fused-memory/src/fused_memory/middleware/lock_charter_guard.py AND its
# corresponding _CANONICAL_EXTENSIONS in
# fused-memory/tests/test_lock_charter_guard.py.
# ---------------------------------------------------------------------------

_CANONICAL_EXTENSIONS = [
    'c', 'cc', 'cjs', 'conf', 'cpp', 'css', 'csv', 'cts', 'cxx',
    'diff', 'envrc', 'example', 'example-systemd-config',
    'gcode', 'gitattributes', 'gitignore', 'gitkeep', 'gitmodules', 'golden', 'grammar',
    'h', 'hh', 'hpp', 'html',
    'icns', 'ico',
    'jq', 'js', 'json', 'jsonc', 'jsonl', 'jsx',
    'lock', 'log',
    'manifest', 'md', 'mjs', 'mts',
    'npmrc', 'png', 'py', 'python-version',
    'ri', 'rs', 'scss', 'service', 'sh', 'step', 'stl', 'svg',
    'template', 'timer', 'tombstones', 'toml', 'ts', 'tsv', 'tsx', 'txt', 'typed',
    'yaml', 'yml',
]

# ---------------------------------------------------------------------------
# Extension-less tracked FILES (dark_factory #3248).
#
# The second half of the classifier: real tracked files whose final segment
# carries NO extension at all, so the FILE_EXTENSIONS lookup can never reach
# them.  Before #3248 every one of these classified as a DIRECTORY, which made
# `hooks/project-checks` undeclarable in metadata.files (γ rejected it with a
# LockCharterViolation) and — worse, because it was silent — made a task
# declaring ONLY such paths strip to an empty charter and fall through to the
# per-task `task-<id>` synthetic lock, holding no lock on the file it edits.
#
# Update _CANONICAL_EXTENSIONLESS AND EXTENSIONLESS_FILENAMES together when the
# vector changes.  Also update the verbatim copies in
# fused-memory/src/fused_memory/middleware/lock_charter_guard.py AND
# fused-memory/tests/test_lock_charter_guard.py.
#
# EVIDENCE — `git ls-files -s` over BOTH repos, re-measured 2026-08-09 (and
# identical to the 2026-07-31 measurement): every tracked path whose final
# segment contains no '.'.  dark-factory contributes Dockerfile LICENSE
# pre-commit pre-merge-commit project-checks; reify adds cargo
# cargo-audit-orphans reference-transaction.  Union = the 8 below.
# ---------------------------------------------------------------------------

_CANONICAL_EXTENSIONLESS = [
    'Dockerfile',
    'LICENSE',
    'cargo',
    'cargo-audit-orphans',
    'pre-commit',
    'pre-merge-commit',
    'project-checks',
    'reference-transaction',
]
