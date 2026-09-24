"""The per-task prose read behind the Tasks tab's Task Detail pane (task 5815).

The ACTIVE_TASKS list no longer carries a task's description/details; the pane
fetches them for the ONE selected task. That read is addressed by the row's own
uid, so the label resolution and the uid format below are the seam the list
rows and the prose route share.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from _dashboard_helpers import mcp_init_response, mcp_notify_response, mcp_tool_response

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
    or to an exception instance it raises on every post. Every post is recorded
    in ``posts``, so a test can assert which URLs were contacted and exactly
    what ``get_task`` was sent — or that nothing was sent at all.
    """

    answers: dict[int, dict | Exception]
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
        return mcp_tool_response(answer, body.get('id', 1))

    def tool_calls(self) -> list[_Post]:
        return [post for post in self.posts if post.method == 'tools/call']

    def ports_contacted(self) -> set[int]:
        return {post.port for post in self.posts}


def _project(tmp_path: Path, *urls: str) -> tuple[Path, DashboardConfig]:
    root = tmp_path / 'proj'
    root.mkdir()
    return root, DashboardConfig(project_root=root, fused_memory_urls=list(urls))


async def _read(fused_memory: _FusedMemory, config: DashboardConfig, root: Path, task_id: int):
    async with httpx.AsyncClient(transport=httpx.MockTransport(fused_memory)) as client:
        return await fetch_task_prose(client, config, root, task_id)


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
