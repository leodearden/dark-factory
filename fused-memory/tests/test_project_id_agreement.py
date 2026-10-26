"""fused-memory resolves a project id exactly as ``shared.project_id`` does.

Every fused-memory entry point that turns a raw id or a root into a stored id
must agree with the one rule, and its folded form must agree with the cockpit
fold (invariant I2). Contract: plans/review-all-s14-shared-leaves-prd.md (B4).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
import yaml
from pydantic import ValidationError
from shared.project_id import fold_project_token, normalize_project_id, project_id_for_root

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.models.scope import (
    ORCHESTRATOR_CONFIG_NAME,
    Scope,
    read_declared_project_id,
    resolve_project_id,
)
from fused_memory.utils.validation import PathShapedProjectIdError, canonicalize_project_id

RAW_IDS = ['Know-Live', 'dark--factory', 'Dark_Factory', 'df', 'reify']

PATH_SHAPED_ID = '-home-leo-x'

raw_ids = pytest.mark.parametrize('raw', RAW_IDS)


def _root_declaring(root: Path, project_id: str) -> Path:
    root.mkdir(parents=True)
    (root / ORCHESTRATOR_CONFIG_NAME).write_text(
        yaml.safe_dump({'project_id': project_id}), encoding='utf-8',
    )
    return root


@pytest_asyncio.fixture
async def backend(tmp_path):
    started = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await started.start()
    yield started
    await started.close()


@raw_ids
def test_canonicalize_project_id_is_the_shared_rule(raw):
    assert canonicalize_project_id(raw) == normalize_project_id(raw)


@raw_ids
def test_scope_stores_the_shared_rule(raw):
    assert Scope(project_id=raw).project_id == normalize_project_id(raw)


@raw_ids
@pytest.mark.parametrize('suffix', ['', '/'], ids=['bare', 'trailing-slash'])
def test_resolve_project_id_agrees_with_project_id_for_root(raw, suffix):
    root = f'/srv/{raw}{suffix}'
    assert resolve_project_id(root) == project_id_for_root(root)


@raw_ids
def test_a_declared_project_id_is_read_through_the_shared_rule(raw, tmp_path):
    root = _root_declaring(tmp_path / 'checkout', raw)
    assert read_declared_project_id(root) == normalize_project_id(raw)


@raw_ids
@pytest.mark.asyncio
async def test_a_qualified_dependency_stores_the_shared_rule(raw, backend, tmp_path):
    project_root = str(tmp_path / 'proj')
    await backend.add_task(project_root=project_root, title='a')

    await backend.add_dependency('1', f'{raw}:13', project_root=project_root)

    task = await backend.get_task('1', project_root)
    assert task['metadata']['external_deps'] == [f'{normalize_project_id(raw)}:13']


@raw_ids
def test_the_canonical_id_folds_like_the_raw_token(raw):
    assert fold_project_token(canonicalize_project_id(raw)) == fold_project_token(raw)


class TestAPathShapedIdIsStillRejected:
    def test_canonicalize_project_id_raises(self):
        with pytest.raises(PathShapedProjectIdError):
            canonicalize_project_id(PATH_SHAPED_ID)

    def test_scope_refuses_it(self):
        with pytest.raises(ValidationError):
            Scope(project_id=PATH_SHAPED_ID)

    def test_a_manifest_declaring_it_is_ignored(self, tmp_path):
        root = _root_declaring(tmp_path / 'checkout', PATH_SHAPED_ID)
        assert read_declared_project_id(root) is None
