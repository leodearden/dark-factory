"""The governed exception lists of shared/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import config_dir_archival_allowlist
import silent_fallthrough_allowlist

from shared.governed_exceptions import Debt, TaskRef, governed_exceptions
from shared.task_metadata import _BLESSED_METADATA_KEYS, _LEGACY_RETRY_LEDGER_COUNTER_KEYS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _occurrence_keys(sites: Iterable[tuple[str, ...]]) -> list[str]:
    """One key per site, the n-th repeat of an identical site suffixed '#<n>'."""
    seen: Counter[str] = Counter()
    keys = []
    for site in sites:
        key = '::'.join(site)
        seen[key] += 1
        keys.append(key if seen[key] == 1 else f'{key}#{seen[key]}')
    return keys


_RUFF_TABLES = (('tool', 'ruff', 'lint'), ('tool', 'ruff'))
_RUFF_GLOBAL_IGNORES = ('ignore', 'extend-ignore')
_RUFF_PER_FILE_IGNORES = ('per-file-ignores', 'extend-per-file-ignores')


def _table(config: Mapping, path: tuple[str, ...]) -> Mapping:
    for part in path:
        config = config.get(part, {})
    return config


def _workspace_pyprojects(root: Path) -> list[Path]:
    root_pyproject = root / 'pyproject.toml'
    workspace = _table(tomllib.loads(root_pyproject.read_text()), ('tool', 'uv', 'workspace'))
    return [root_pyproject, *(root / member / 'pyproject.toml' for member in workspace.get('members', []))]


def _global_ignore_keys(table: tuple[str, ...], section: Mapping) -> set[str]:
    dotted = '.'.join(table)
    return {
        f'{dotted}.{setting}::{code}'
        for setting in _RUFF_GLOBAL_IGNORES
        for code in section.get(setting, [])
    }


def _per_file_ignore_keys(relpath: str, table: tuple[str, ...], section: Mapping) -> set[str]:
    dotted = '.'.join(table)
    return {
        f'{relpath}::{dotted}.{setting}::{glob}::{code}'
        for setting in _RUFF_PER_FILE_IGNORES
        for glob, codes in section.get(setting, {}).items()
        for code in codes
    }


def ruff_suppression_keys(root: Path) -> list[str]:
    """Every ruff rule a workspace pyproject.toml turns off, as sorted declaration keys."""
    keys: set[str] = set()
    for pyproject in _workspace_pyprojects(root):
        config = tomllib.loads(pyproject.read_text())
        relpath = pyproject.relative_to(root).as_posix()
        for table in _RUFF_TABLES:
            section = _table(config, table)
            keys |= _global_ignore_keys(table, section)
            keys |= _per_file_ignore_keys(relpath, table, section)
    return sorted(keys)


SILENT_FALLTHROUGH_ALLOWLIST = governed_exceptions(
    'shared.tests.silent_fallthrough_allowlist.ALLOWLIST_ENTRIES',
    'A pre-existing site that swallows an error into an empty value or a discarded '
    'error slot, tolerated by the silent-fallthrough gate until it is fixed.',
    _occurrence_keys(entry[:3] for entry in silent_fallthrough_allowlist.ALLOWLIST_ENTRIES),
    default=Debt(TaskRef(5603)),
    default_covers=12,
)

CONFIG_DIR_ARCHIVAL_EXCEPTIONS = governed_exceptions(
    'shared.tests.config_dir_archival_allowlist.AUDITED_SITES',
    'A TaskConfigDir construction site whose dir is destroyed without its transcripts '
    'being archived first, either by design or as a recorded gap.',
    _occurrence_keys(
        (site['path'], site['qualname'])
        for site in config_dir_archival_allowlist.AUDITED_SITES
        if site['disposition'] != config_dir_archival_allowlist.ARCHIVED
    ),
    default=Debt(TaskRef(5603)),
    default_covers=6,
)

BLESSED_TASK_METADATA_KEYS = governed_exceptions(
    'shared.task_metadata._BLESSED_METADATA_KEYS',
    'A task-metadata key outside the typed TaskMetadata schema that the metadata census '
    'accepts without an unknown-key warning.',
    sorted(_BLESSED_METADATA_KEYS),
    default=Debt(TaskRef(5603)),
    default_covers=52,
)

LEGACY_RETRY_LEDGER_COUNTER_KEYS = governed_exceptions(
    'shared.task_metadata._LEGACY_RETRY_LEDGER_COUNTER_KEYS',
    'A RetryLedger counter that legacy metadata blobs still carry as a top-level key '
    'instead of nested under retry_ledger, tolerated rather than reported.',
    list(_LEGACY_RETRY_LEDGER_COUNTER_KEYS),
    default=Debt(TaskRef(5603)),
    default_covers=2,
)

RUFF_CONFIG_SUPPRESSIONS = governed_exceptions(
    'workspace.pyproject_toml.ruff_ignores',
    'A ruff rule turned off in a workspace pyproject.toml, globally or for a file glob, '
    'instead of being fixed at its sites or disposed inline.',
    ruff_suppression_keys(REPO_ROOT),
    default=Debt(TaskRef(5603)),
    default_covers=2,
)


def test_repeated_sites_keep_distinct_keys() -> None:
    assert _occurrence_keys([('a.py', 'f', 'h'), ('a.py', 'f', 'h'), ('b.py', 'g', 'h')]) == [
        'a.py::f::h',
        'a.py::f::h#2',
        'b.py::g::h',
    ]


def test_ruff_suppression_keys_come_from_every_workspace_pyproject(tmp_path: Path) -> None:
    (tmp_path / 'pyproject.toml').write_text(
        "[tool.uv.workspace]\nmembers = ['a', 'b']\n\n[tool.ruff.lint]\nignore = ['E501']\n"
    )
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'pyproject.toml').write_text(
        "[tool.ruff.lint]\nignore = ['E501']\nper-file-ignores = {'tests/*.py' = ['F811']}\n"
    )
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'pyproject.toml').write_text("[tool.ruff]\nextend-ignore = ['B008']\n")

    assert ruff_suppression_keys(tmp_path) == [
        'a/pyproject.toml::tool.ruff.lint.per-file-ignores::tests/*.py::F811',
        'tool.ruff.extend-ignore::B008',
        'tool.ruff.lint.ignore::E501',
    ]
