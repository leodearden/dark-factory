"""commit_planning's ``manifest_root`` argument and the ``sidecar_not_found`` report.

Driven through the MCP tool on a real git layout: ``main`` is the repository's
main checkout and the task store's root, ``wt`` is a linked worktree on the
branch that holds a PRD family not yet on main.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from _fastmcp_tool_result import tool_result_dict
from _fm_git_repos import _init_git_repo, git_stdout
from _hermetic_task_stack import hermetic_task_stack
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)]

_PRD_PATH = 'plans/foo-prd.md'
_SIDECAR_REL = 'plans/foo-prd.capability-manifest.yaml'
_CHECK_NAME = 'future_symbol_lands'
#: Never committed anywhere, so an ``expect: present`` grep for it is the
#: forward-looking shape the polarity lint leaves alone.
_FUTURE_SYMBOL = 'FutureSymbol'


@pytest.fixture
def repo_layout(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / 'main'
    main.mkdir()
    _init_git_repo(main)
    wt = tmp_path / 'wt'
    git_stdout(main, 'worktree', 'add', '-q', '-b', 'prd/foo', str(wt))
    return main, wt


def write_prd_family(root: Path, stem: str, label: str) -> Path:
    """Write ``plans/<stem>-prd.md`` and its two sidecar twins; return the YAML sidecar."""
    plans = root / 'plans'
    plans.mkdir(parents=True, exist_ok=True)
    (plans / f'{stem}-prd.md').write_text(f'# {stem} PRD\n', encoding='utf-8')
    (plans / f'{stem}-prd.capability-manifest.md').write_text(
        f'# {stem} capability manifest\n', encoding='utf-8',
    )
    sidecar = plans / f'{stem}-prd.capability-manifest.yaml'
    sidecar.write_text(
        f'prd: plans/{stem}-prd.md\n'
        f'schema_version: 1\n'
        f'tasks:\n'
        f'  - label: {label}\n'
        f'    task_id: null\n'
        f'    title: Land {_FUTURE_SYMBOL}\n'
        f'    capabilities:\n'
        f'      - name: {_CHECK_NAME}\n'
        f"        binding: 'grep for {_FUTURE_SYMBOL}'\n"
        f'        verdict: PASS\n'
        f'        delivered_check:\n'
        f'          kind: grep\n'
        f"          pattern: '{_FUTURE_SYMBOL}'\n"
        f'          expect: present\n'
        f'          paths:\n'
        f'            - seed.txt\n',
        encoding='utf-8',
    )
    return sidecar


@contextlib.asynccontextmanager
async def task_stack(main: Path) -> AsyncIterator[Any]:
    """The hermetic MCP server over *main*, its corpus indexing stubbed out.

    A ``pending`` commit indexes the batch into the curator corpus, which
    reaches a live embedder; that is not this file's subject.
    """
    async with hermetic_task_stack(main) as (server, interceptor):
        with patch.object(interceptor, 'index_committed_tasks', AsyncMock()):
            yield server


async def call(server: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return tool_result_dict(await server._tool_manager.call_tool(tool, args))


async def submit_planned(server: Any, root: Path, metadata: dict[str, Any]) -> str:
    submitted = await call(server, 'submit_task', {
        'project_root': str(root),
        'title': f"Producer for {metadata.get('prd_task_label')}",
        'planning_mode': True,
        'metadata': metadata,
    })
    assert submitted.get('status') == 'deferred', submitted
    return str(submitted['task_id'])


def _producer_metadata(prd_path: str = _PRD_PATH, label: str = 'alpha') -> dict[str, Any]:
    return {'files': ['seed.txt'], 'prd_path': prd_path, 'prd_task_label': label}


async def test_prd_family_only_in_a_worktree_is_stamped_there(
    repo_layout: tuple[Path, Path],
) -> None:
    main, wt = repo_layout
    sidecar = write_prd_family(wt, 'foo', 'alpha')
    git_stdout(wt, 'add', '-A')
    git_stdout(wt, 'commit', '-q', '--no-verify', '-m', 'prd: foo')

    async with task_stack(main) as server:
        task_id = await submit_planned(server, main, _producer_metadata())

        result = await call(server, 'commit_planning', {
            'project_root': str(main), 'task_ids': task_id, 'manifest_root': str(wt),
        })

        assert result['manifest_stamping'] == {
            'path': _SIDECAR_REL, 'stamped': ['alpha'], 'missing_labels': [], 'errors': [],
        }, result
        assert 'manifest_stamping_action_required' not in result
        task = await call(server, 'get_task', {'id': task_id, 'project_root': str(main)})
        assert task['status'] == 'pending'
        assert [check['name'] for check in task['metadata']['delivered_checks']] == [_CHECK_NAME]

    reloaded = yaml.safe_load(sidecar.read_text(encoding='utf-8'))
    assert reloaded['tasks'][0]['task_id'] == int(task_id)
    assert git_stdout(wt, 'status', '--porcelain') == f'M {_SIDECAR_REL}'
    assert not (main / 'plans').exists()


#: ``(tmp_path, main, wt) -> (manifest_root, sidecars that must stay untouched)``.
CaseBuilder = Callable[[Path, Path, Path], tuple[str, list[Path]]]


async def assert_refused_before_anything(
    server: Any, root: Path, task_id: str, result: dict[str, Any], reason_code: str,
) -> None:
    assert result.get('error_type') == 'ManifestRootRefused', result
    assert result['reason_code'] == reason_code, result
    assert 'manifest_stamping' not in result
    task = await call(server, 'get_task', {'id': task_id, 'project_root': str(root)})
    assert task['status'] == 'deferred'
    assert 'delivered_checks' not in task['metadata']


def _other_repository(tmp_path: Path, main: Path, wt: Path) -> tuple[str, list[Path]]:
    other = tmp_path / 'other'
    other.mkdir()
    _init_git_repo(other)
    return str(other), [write_prd_family(other, 'foo', 'alpha')]


def _plain_directory(tmp_path: Path, main: Path, wt: Path) -> tuple[str, list[Path]]:
    plain = tmp_path / 'plain'
    return str(plain), [write_prd_family(plain, 'foo', 'alpha')]


def _missing_directory(tmp_path: Path, main: Path, wt: Path) -> tuple[str, list[Path]]:
    return str(tmp_path / 'missing'), []


def _relative_path(tmp_path: Path, main: Path, wt: Path) -> tuple[str, list[Path]]:
    return 'wt', [write_prd_family(wt, 'foo', 'alpha')]


def _worktree_subdirectory(tmp_path: Path, main: Path, wt: Path) -> tuple[str, list[Path]]:
    sidecar = write_prd_family(wt, 'foo', 'alpha')
    return str(wt / 'plans'), [sidecar]


def _main_checkout_alias(tmp_path: Path, main: Path, wt: Path) -> tuple[str, list[Path]]:
    alias = tmp_path / 'alias'
    alias.symlink_to(main, target_is_directory=True)
    return str(alias), [write_prd_family(main, 'foo', 'alpha')]


@pytest.mark.parametrize(
    ('build_case', 'target_status', 'reason_code'),
    [
        pytest.param(_other_repository, 'pending', 'other_repository', id='other-repository'),
        pytest.param(_plain_directory, 'pending', 'not_a_worktree', id='plain-directory'),
        pytest.param(_missing_directory, 'pending', 'not_a_worktree', id='missing-directory'),
        pytest.param(_relative_path, 'pending', 'not_a_worktree', id='relative-path'),
        pytest.param(
            _worktree_subdirectory, 'pending', 'not_a_worktree', id='worktree-subdirectory',
        ),
        pytest.param(_main_checkout_alias, 'pending', 'main_checkout', id='main-checkout-alias'),
        pytest.param(
            _other_repository, 'cancelled', 'other_repository', id='other-repository-cancelled',
        ),
    ],
)
async def test_manifest_root_that_is_not_a_worktree_of_the_repo_is_refused(
    tmp_path: Path, repo_layout: tuple[Path, Path], build_case: CaseBuilder,
    target_status: str, reason_code: str,
) -> None:
    main, wt = repo_layout
    manifest_root, sidecars = build_case(tmp_path, main, wt)
    snapshot = {sidecar: sidecar.read_bytes() for sidecar in sidecars}

    async with task_stack(main) as server:
        task_id = await submit_planned(server, main, _producer_metadata())

        result = await call(server, 'commit_planning', {
            'project_root': str(main), 'task_ids': task_id, 'target_status': target_status,
            'manifest_root': manifest_root,
        })

        await assert_refused_before_anything(server, main, task_id, result, reason_code)
        assert manifest_root in result['error']

    assert {sidecar: sidecar.read_bytes() for sidecar in sidecars} == snapshot


async def test_manifest_root_is_refused_when_project_root_is_not_a_git_checkout(
    tmp_path: Path, repo_layout: tuple[Path, Path],
) -> None:
    _, wt = repo_layout
    store = tmp_path / 'store'
    store.mkdir()
    sidecar = write_prd_family(wt, 'foo', 'alpha')
    snapshot = sidecar.read_bytes()

    async with task_stack(store) as server:
        task_id = await submit_planned(server, store, _producer_metadata())

        result = await call(server, 'commit_planning', {
            'project_root': str(store), 'task_ids': task_id, 'manifest_root': str(wt),
        })

        await assert_refused_before_anything(server, store, task_id, result, 'other_repository')
        assert f'project_root {str(store)!r} is not a git checkout' in result['error']

    assert sidecar.read_bytes() == snapshot


async def test_manifest_root_is_refused_when_git_cannot_run(
    tmp_path: Path, repo_layout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    main, wt = repo_layout
    sidecar = write_prd_family(wt, 'foo', 'alpha')
    snapshot = sidecar.read_bytes()

    async with task_stack(main) as server:
        task_id = await submit_planned(server, main, _producer_metadata())

        with monkeypatch.context() as env:
            env.setenv('PATH', str(tmp_path / 'no-git-here'))
            result = await call(server, 'commit_planning', {
                'project_root': str(main), 'task_ids': task_id, 'manifest_root': str(wt),
            })

        await assert_refused_before_anything(server, main, task_id, result, 'probe_failed')

    assert sidecar.read_bytes() == snapshot


@pytest.mark.parametrize('searched', ['main', 'wt'], ids=['manifest-root-omitted', 'worktree'])
async def test_absent_sidecar_is_reported_with_the_root_searched(
    repo_layout: tuple[Path, Path], searched: str,
) -> None:
    main, wt = repo_layout
    root = {'main': main, 'wt': wt}[searched]
    args = {'project_root': str(main)}
    if searched == 'wt':
        args['manifest_root'] = str(wt)

    async with task_stack(main) as server:
        task_id = await submit_planned(server, main, _producer_metadata())

        result = await call(server, 'commit_planning', {**args, 'task_ids': task_id})

        assert result['manifest_stamping'] == {
            'path': None,
            'stamped': [],
            'missing_labels': [],
            'errors': [],
            'sidecar_not_found': {'root': str(root.resolve()), 'paths': [_SIDECAR_REL]},
        }, result
        assert 'manifest_stamping_action_required' not in result
        task = await call(server, 'get_task', {'id': task_id, 'project_root': str(main)})
        assert task['status'] == 'pending'


async def test_absent_sidecar_beside_a_found_one_is_reported(
    repo_layout: tuple[Path, Path],
) -> None:
    main, wt = repo_layout
    write_prd_family(wt, 'foo', 'alpha')

    async with task_stack(main) as server:
        alpha = await submit_planned(server, main, _producer_metadata(label='alpha'))
        beta = await submit_planned(
            server, main, _producer_metadata(prd_path='plans/bar-prd.md', label='beta'),
        )

        result = await call(server, 'commit_planning', {
            'project_root': str(main), 'task_ids': f'{alpha},{beta}', 'manifest_root': str(wt),
        })

        report = result['manifest_stamping']
        assert report['path'] == _SIDECAR_REL, result
        assert report['stamped'] == ['alpha']
        assert report['sidecar_not_found'] == {
            'root': str(wt.resolve()), 'paths': ['plans/bar-prd.capability-manifest.yaml'],
        }
