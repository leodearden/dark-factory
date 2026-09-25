"""The per-task prose read behind the Tasks tab's Task Detail pane (task 5815).

The ACTIVE_TASKS list no longer carries a task's description/details; the pane
fetches them for the ONE selected task. That read is addressed by the row's own
uid, so the label resolution and the uid format below are the seam the list
rows and the prose route share.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
from _dashboard_helpers import mcp_init_response, mcp_notify_response, mcp_tool_response
from fastapi import FastAPI

from dashboard.api.task_prose import router as task_prose_router
from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import project_roots_for_label, task_uid
from dashboard.data.memory import reset_sessions
from dashboard.data.tasks import (
    TaskNotFound,
    TaskProse,
    TaskReadOffline,
    fetch_task_prose,
)

_URL_1 = 'http://127.0.0.1:9101'
_URL_2 = 'http://127.0.0.1:9102'


@pytest.fixture(autouse=True)
def _clean_sessions():
    reset_sessions()
    yield
    reset_sessions()


@dataclass(frozen=True)
class _Post:
    port: int
    method: str
    params: dict | None


@dataclass
class _FusedMemory:
    """A MockTransport handler standing in for fused-memory, one answer per port.

    ``answers`` maps a port to the inner ``get_task`` result that URL returns,
    to an exception instance it raises on every post, or to an
    ``asyncio.Event`` its ``tools/call`` waits on (set by nothing: a hang).
    Every post is recorded in ``posts``, so a test can assert which URLs were
    contacted and exactly what ``get_task`` was sent — or that nothing was sent
    at all.
    """

    answers: dict[int, dict | Exception | asyncio.Event]
    posts: list[_Post] = field(default_factory=list)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        port = request.url.port
        assert port is not None
        body = json.loads(request.content)
        method = body.get('method', '')
        self.posts.append(_Post(port, method, body.get('params')))
        answer = self.answers[port]
        if isinstance(answer, Exception):
            raise answer
        if method == 'initialize':
            return mcp_init_response(body.get('id', 1))
        if method.startswith('notifications/'):
            return mcp_notify_response()
        if isinstance(answer, asyncio.Event):
            await answer.wait()
            raise AssertionError('nothing sets the hang event')
        return mcp_tool_response(answer, body.get('id', 1))

    def tool_calls(self) -> list[_Post]:
        return [post for post in self.posts if post.method == 'tools/call']

    def ports_contacted(self) -> set[int]:
        return {post.port for post in self.posts}


def _project(tmp_path: Path, *urls: str, name: str = 'proj') -> tuple[Path, DashboardConfig]:
    (tmp_path / name).mkdir()
    config = DashboardConfig(project_root=tmp_path / name, fused_memory_urls=list(urls))
    return config.project_root, config


async def _read(fused_memory: _FusedMemory, config: DashboardConfig, root: Path, task_id: int):
    async with httpx.AsyncClient(transport=httpx.MockTransport(fused_memory)) as client:
        return await fetch_task_prose(client, config, root, task_id)


def _prose_path(uid: str) -> str:
    """The route path for a row uid, each segment percent-encoded as the client does."""
    return '/api/v2/dashboard/task/' + '/'.join(
        quote(segment, safe='') for segment in uid.split('/')
    )


async def _get_prose(
    config: DashboardConfig, fused_memory: _FusedMemory, path: str,
) -> httpx.Response:
    """GET *path* from an app serving ONLY the prose router, fused-memory mocked."""
    app = FastAPI()
    app.include_router(task_prose_router)
    app.state.config = config
    async with httpx.AsyncClient(transport=httpx.MockTransport(fused_memory)) as mcp_client:
        app.state.http_client = mcp_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://test',
        ) as client:
            return await client.get(path)


class TestProjectRootsForLabel:
    """A row's project label resolves back to the configured root(s) it names."""

    def test_a_label_matching_one_root_returns_that_root(self, tmp_path):
        primary = tmp_path / 'alpha'
        other = tmp_path / 'beta'
        for root in (primary, other):
            root.mkdir()
        config = DashboardConfig(project_root=primary, known_project_roots=[other])

        assert project_roots_for_label(config, 'beta') == [other.resolve()]

    def test_an_unknown_label_returns_no_root(self, tmp_path):
        primary = tmp_path / 'alpha'
        primary.mkdir()
        config = DashboardConfig(project_root=primary)

        assert project_roots_for_label(config, 'gamma') == []

    def test_roots_sharing_a_basename_all_come_back_primary_first(self, tmp_path):
        """The label is a BASENAME, so it can name two roots; neither is dropped."""
        primary = tmp_path / 'a' / 'proj'
        other = tmp_path / 'b' / 'proj'
        for root in (primary, other):
            root.mkdir(parents=True)
        config = DashboardConfig(project_root=primary, known_project_roots=[other])

        assert project_roots_for_label(config, 'proj') == [
            primary.resolve(), other.resolve(),
        ]

    def test_task_uid_is_the_rows_identity(self):
        assert task_uid('proj', 19) == 'proj/T-19'


class TestFetchTaskProse:
    """One task's description/details, read through fused-memory's get_task."""

    async def test_success_returns_the_prose_from_one_get_task_call(self, tmp_path):
        root, config = _project(tmp_path, _URL_1)
        fused_memory = _FusedMemory({9101: {'id': '19', 'description': 'd', 'details': 't'}})

        read = await _read(fused_memory, config, root, 19)

        assert read == TaskProse(description='d', details='t')
        [call] = fused_memory.tool_calls()
        assert call.params is not None
        assert call.params['name'] == 'get_task'
        assert call.params['arguments'] == {'id': '19', 'project_root': str(root)}

    @pytest.mark.parametrize('row', [
        {'id': '19', 'title': 'x', 'description': None, 'details': None},
        {'id': '19', 'title': 'x'},
    ], ids=['none', 'absent'])
    async def test_missing_prose_normalises_to_empty_strings(self, tmp_path, row):
        root, config = _project(tmp_path, _URL_1)

        read = await _read(_FusedMemory({9101: row}), config, root, 19)

        assert read == TaskProse(description='', details='')

    async def test_a_missing_task_is_a_definitive_answer_not_an_outage(self, tmp_path):
        """TaskNotFoundError stops the fan-out: the next URL would say the same."""
        root, config = _project(tmp_path, _URL_1, _URL_2)
        message = 'No tasks found for ID(s): 19'
        fused_memory = _FusedMemory({
            9101: {'error': message, 'error_type': 'TaskNotFoundError'},
            9102: {'id': '19', 'description': 'd', 'details': 't'},
        })

        read = await _read(fused_memory, config, root, 19)

        assert isinstance(read, TaskNotFound)
        assert message in read.detail
        assert 9102 not in fused_memory.ports_contacted(), (
            'a missing task is a definitive answer; asking the next URL treats '
            'it as a soft failure and reports an absent task as an outage'
        )

    async def test_a_generic_tool_error_falls_through_to_the_next_url(self, tmp_path):
        root, config = _project(tmp_path, _URL_1, _URL_2)
        fused_memory = _FusedMemory({
            9101: {'error': 'boom', 'error_type': 'TaskmasterError'},
            9102: {'id': '19', 'description': 'd', 'details': 't'},
        })

        read = await _read(fused_memory, config, root, 19)

        assert read == TaskProse(description='d', details='t')

    async def test_an_empty_result_is_a_soft_failure_not_empty_prose(self, tmp_path):
        """An unparseable tool result must not render as 'this task has no prose'."""
        root, config = _project(tmp_path, _URL_1)

        read = await _read(_FusedMemory({9101: {}}), config, root, 19)

        assert isinstance(read, TaskReadOffline)
        assert 'empty result' in read.detail

    async def test_every_url_unreachable_is_offline_naming_each_url(self, tmp_path):
        root, config = _project(tmp_path, _URL_1, _URL_2)
        fused_memory = _FusedMemory({
            9101: httpx.ConnectError('refused'),
            9102: httpx.ConnectError('refused'),
        })

        read = await _read(fused_memory, config, root, 19)

        assert isinstance(read, TaskReadOffline)
        assert _URL_1 in read.detail
        assert _URL_2 in read.detail

    def test_the_wire_shape_is_exactly_description_and_details(self):
        prose = TaskProse(description='d', details='t')

        assert prose.to_wire() == {'description': 'd', 'details': 't'}


class TestTaskProseRoute:
    """GET /api/v2/dashboard/task/{project}/T-{task_id}: one task's prose, by row uid."""

    @pytest.mark.parametrize('name', ['proj', 'my proj#1'], ids=['plain', 'url-hostile'])
    async def test_the_rows_uid_round_trips_to_get_task(self, tmp_path, name):
        """The route template is task_uid's format: the uid addresses the right task."""
        root, config = _project(tmp_path, _URL_1, name=name)
        fused_memory = _FusedMemory({9101: {'id': '19', 'description': 'd', 'details': 't'}})
        uid = task_uid(root.name, 19)

        response = await _get_prose(config, fused_memory, _prose_path(uid))

        assert response.status_code == 200
        assert response.json() == {f'TASK_PROSE:{uid}': {'description': 'd', 'details': 't'}}
        [call] = fused_memory.tool_calls()
        assert call.params is not None
        assert call.params['arguments'] == {'id': '19', 'project_root': str(root)}

    async def test_an_unknown_project_is_404_and_never_asks_fused_memory(self, tmp_path):
        _root, config = _project(tmp_path, _URL_1)
        fused_memory = _FusedMemory({9101: {'id': '19'}})

        response = await _get_prose(config, fused_memory, _prose_path('nope/T-19'))

        assert response.status_code == 404
        assert response.json() == {'error': 'unknown_project', 'project': 'nope'}
        assert fused_memory.posts == []

    async def test_a_label_naming_two_roots_is_409_and_never_asks_fused_memory(self, tmp_path):
        """Picking one would show another task's prose under a correct-looking title."""
        primary = tmp_path / 'a' / 'proj'
        other = tmp_path / 'b' / 'proj'
        for root in (primary, other):
            root.mkdir(parents=True)
        config = DashboardConfig(
            project_root=primary, known_project_roots=[other], fused_memory_urls=[_URL_1],
        )
        fused_memory = _FusedMemory({9101: {'id': '19'}})

        response = await _get_prose(config, fused_memory, _prose_path('proj/T-19'))

        assert response.status_code == 409
        assert response.json() == {
            'error': 'ambiguous_project',
            'project': 'proj',
            'roots': [str(primary.resolve()), str(other.resolve())],
        }
        assert fused_memory.posts == []

    async def test_a_missing_task_is_404_carrying_fused_memorys_message(self, tmp_path):
        _root, config = _project(tmp_path, _URL_1)
        message = 'No tasks found for ID(s): 19'
        fused_memory = _FusedMemory({9101: {'error': message, 'error_type': 'TaskNotFoundError'}})

        response = await _get_prose(config, fused_memory, _prose_path('proj/T-19'))

        assert response.status_code == 404
        assert response.json() == {'error': 'task_not_found', 'detail': message}

    async def test_fused_memory_unreachable_is_502_with_a_detail(self, tmp_path):
        _root, config = _project(tmp_path, _URL_1)
        fused_memory = _FusedMemory({9101: httpx.ConnectError('refused')})

        response = await _get_prose(config, fused_memory, _prose_path('proj/T-19'))

        assert response.status_code == 502
        body = response.json()
        assert body['error'] == 'fused_memory_unreachable'
        assert _URL_1 in body['detail']

    async def test_a_hung_read_is_cut_at_the_budget_as_504(self, tmp_path, monkeypatch, caplog):
        """Bounded: neither a hang nor a 500, and the expiry leaves a journal line."""
        monkeypatch.setattr('dashboard.api.task_prose._TASK_PROSE_BUDGET', 0.05)
        _root, config = _project(tmp_path, _URL_1)
        fused_memory = _FusedMemory({9101: asyncio.Event()})

        with caplog.at_level(logging.WARNING, logger='dashboard.api.task_prose'):
            async with asyncio.timeout(5):
                response = await _get_prose(config, fused_memory, _prose_path('proj/T-19'))

        assert response.status_code == 504
        body = response.json()
        assert body['error'] == 'budget_exceeded'
        assert body['detail']
        assert fused_memory.tool_calls(), 'the read must have reached get_task before hanging'
        [record] = [r for r in caplog.records if r.name == 'dashboard.api.task_prose']
        assert 'proj' in record.getMessage()
        assert '19' in record.getMessage()
