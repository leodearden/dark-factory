"""The governed exception lists of shared/, declared under INV-12.

Each binding below is a declaration in the sense of
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D4/D5: it names a list,
states the rule a new entry is judged against, and gives every entry a
disposition. The call is evaluated when this module is collected, so an entry
with no disposition fails this suite before any test runs.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from shared.governed_exceptions import (
    Debt,
    MalformedDeclaration,
    Policy,
    TaskRef,
    governed_exceptions,
)
from shared.task_metadata import _BLESSED_METADATA_KEYS, _LEGACY_RETRY_LEDGER_COUNTER_KEYS
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]


_RUFF_TABLES = (('tool', 'ruff', 'lint'), ('tool', 'ruff'))
_RUFF_GLOBAL_IGNORES = ('ignore', 'extend-ignore')
_RUFF_PER_FILE_IGNORES = ('per-file-ignores', 'extend-per-file-ignores')
_RUFF_ONLY_CONFIG_NAMES = ('.ruff.toml', 'ruff.toml')


def _table(config: Mapping, path: tuple[str, ...]) -> Mapping:
    for part in path:
        config = config.get(part, {})
    return config


def _unreadable_workspace(root: Path, path: Path, fault: str) -> MalformedDeclaration:
    return MalformedDeclaration(
        f'RUFF_CONFIG_SUPPRESSIONS cannot derive its keys: {path.relative_to(root).as_posix()} '
        f'{fault}. Extend ruff_suppression_keys in shared/tests/test_shared_governed_lists.py '
        'to read the workspace as it now stands.',
        value=path,
    )


def _member_dirs(root: Path, workspace: Mapping) -> list[Path]:
    excluded = {path for pattern in workspace.get('exclude', []) for path in root.glob(pattern)}
    dirs = []
    for pattern in workspace.get('members', []):
        matches = sorted(path for path in root.glob(pattern) if path.is_dir())
        if not matches:
            raise _unreadable_workspace(
                root, root / 'pyproject.toml', f'declares the member {pattern!r}, which names no directory'
            )
        dirs += [path for path in matches if path not in excluded]
    return dirs


def _ruff_pyproject(root: Path, config_dir: Path) -> Path:
    for name in _RUFF_ONLY_CONFIG_NAMES:
        if (config_dir / name).exists():
            raise _unreadable_workspace(
                root, config_dir / name, 'exists, and ruff reads it instead of the pyproject.toml beside it'
            )
    if not (config_dir / 'pyproject.toml').is_file():
        raise _unreadable_workspace(root, config_dir, 'has no pyproject.toml')
    return config_dir / 'pyproject.toml'


def _workspace_pyprojects(root: Path) -> list[Path]:
    workspace = _table(tomllib.loads((root / 'pyproject.toml').read_text()), ('tool', 'uv', 'workspace'))
    return [_ruff_pyproject(root, config_dir) for config_dir in [root, *_member_dirs(root, workspace)]]


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


BLESSED_TASK_METADATA_KEYS = governed_exceptions(
    'shared.task_metadata._BLESSED_METADATA_KEYS',
    'A task-metadata key outside the typed TaskMetadata schema that the metadata census '
    'accepts without an unknown-key warning.',
    sorted(_BLESSED_METADATA_KEYS),
    default=Policy('inv12-task-metadata-blessed-keys'),
    default_covers=56,
    dispositions={
        'cancel_dispositions': Policy('inv12-task-metadata-blessed-keys'),
        'gate_escalated_preconditions': Policy('inv12-task-metadata-blessed-keys'),
        'predicate_recheck': Policy('inv12-task-metadata-blessed-keys'),
        'predicate_verdict': Policy('inv12-task-metadata-blessed-keys'),
        'pure_gate_from': Policy('inv12-task-metadata-blessed-keys'),
        'superseded_by': Policy('inv12-task-metadata-blessed-keys'),
        'trigger_chain': Policy('inv12-task-metadata-blessed-keys'),
    },
)

LEGACY_RETRY_LEDGER_COUNTER_KEYS = governed_exceptions(
    'shared.task_metadata._LEGACY_RETRY_LEDGER_COUNTER_KEYS',
    'A RetryLedger counter that legacy metadata blobs still carry as a top-level key '
    'instead of nested under retry_ledger, tolerated rather than reported.',
    list(_LEGACY_RETRY_LEDGER_COUNTER_KEYS),
    default=Debt(TaskRef(7135)),
    default_covers=2,
)

RUFF_CONFIG_SUPPRESSIONS = governed_exceptions(
    'workspace.pyproject_toml.ruff_ignores',
    'A ruff rule turned off in a workspace pyproject.toml, globally or for a file glob, '
    'instead of being fixed at its sites or disposed inline.',
    ruff_suppression_keys(REPO_ROOT),
    dispositions={
        'tool.ruff.lint.ignore::E501': Policy('inv12-ruff-e501-line-length'),
        'orchestrator/pyproject.toml::tool.ruff.lint.per-file-ignores::tests/**/*.py::F811': Debt(TaskRef(7137)),
        'orchestrator/pyproject.toml::tool.ruff.lint.per-file-ignores::src/**/*.py::TID251': Debt(TaskRef(8622)),
    },
)


def _write_tree(root: Path, files: Mapping[str, str]) -> None:
    for relpath, text in files.items():
        (root / relpath).parent.mkdir(parents=True, exist_ok=True)
        (root / relpath).write_text(text)


def test_ruff_suppression_keys_come_from_every_workspace_pyproject(tmp_path: Path) -> None:
    _write_tree(tmp_path, {
        'pyproject.toml': (
            "[tool.uv.workspace]\nmembers = ['a', 'b']\n\n[tool.ruff.lint]\nignore = ['E501']\n"
            "per-file-ignores = {'conftest.py' = ['E402']}\n"
        ),
        'a/pyproject.toml': (
            "[tool.ruff.lint]\nignore = ['E501']\nper-file-ignores = {'tests/*.py' = ['F811']}\n"
        ),
        'b/pyproject.toml': (
            "[tool.ruff]\nextend-ignore = ['B008']\n"
            "extend-per-file-ignores = {'scripts/*.py' = ['T201']}\n"
        ),
    })

    assert ruff_suppression_keys(tmp_path) == [
        'a/pyproject.toml::tool.ruff.lint.per-file-ignores::tests/*.py::F811',
        'b/pyproject.toml::tool.ruff.extend-per-file-ignores::scripts/*.py::T201',
        'pyproject.toml::tool.ruff.lint.per-file-ignores::conftest.py::E402',
        'tool.ruff.extend-ignore::B008',
        'tool.ruff.lint.ignore::E501',
    ]


def test_ruff_suppression_keys_expand_glob_members_less_the_excluded(tmp_path: Path) -> None:
    _write_tree(tmp_path, {
        'pyproject.toml': (
            "[tool.uv.workspace]\nmembers = ['packages/*']\nexclude = ['packages/scratch']\n"
        ),
        'packages/a/pyproject.toml': "[tool.ruff.lint]\nignore = ['E501']\n",
        'packages/b/pyproject.toml': "[tool.ruff.lint]\nignore = ['B008']\n",
        'packages/scratch/pyproject.toml': "[tool.ruff.lint]\nignore = ['ALL']\n",
        'packages/README.md': 'not a member\n',
    })

    assert ruff_suppression_keys(tmp_path) == [
        'tool.ruff.lint.ignore::B008',
        'tool.ruff.lint.ignore::E501',
    ]


@pytest.mark.parametrize(
    ('files', 'fault'),
    [
        pytest.param(
            {'a/pyproject.toml': '', 'ruff.toml': "[lint]\nignore = ['E501']\n"},
            'ruff.toml exists',
            id='root-ruff-toml',
        ),
        pytest.param(
            {'a/pyproject.toml': '', 'a/.ruff.toml': '[lint]\n'},
            'a/.ruff.toml exists',
            id='member-dot-ruff-toml',
        ),
        pytest.param({'a/setup.cfg': ''}, 'a has no pyproject.toml', id='member-without-pyproject'),
        pytest.param(
            {}, "pyproject.toml declares the member 'a', which names no directory", id='member-naming-no-directory'
        ),
    ],
)
def test_ruff_suppression_keys_refuse_a_workspace_they_cannot_read(
    tmp_path: Path, files: Mapping[str, str], fault: str
) -> None:
    _write_tree(tmp_path, {'pyproject.toml': "[tool.uv.workspace]\nmembers = ['a']\n", **files})

    with pytest.raises(MalformedDeclaration, match=re.escape(fault)):
        ruff_suppression_keys(tmp_path)
