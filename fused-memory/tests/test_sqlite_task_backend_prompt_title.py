"""A title derived from a prompt's first line is clipped at a word with the counted marker, never silently."""

from __future__ import annotations

from os.path import commonprefix

import pytest
import pytest_asyncio
from _clip_assertions import assert_elided

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def project_root(tmp_path):
    return str(tmp_path / 'proj')


async def _stored(backend: SqliteTaskBackend, project_root: str, **fields) -> dict:
    added = await backend.add_task(project_root=project_root, **fields)
    return await backend.get_task(added['id'], project_root=project_root)


@pytest.mark.asyncio
async def test_a_long_first_line_is_clipped_at_a_word_with_a_counted_marker(backend, project_root):
    first_line = ('Investigate why ' + 'the frobinator stalls under load ' * 10).rstrip()
    prompt = first_line + '\n\nBody text'
    task = await _stored(backend, project_root, prompt=prompt)
    title = task['title']
    assert len(title) <= 200
    assert_elided(title, first_line)
    head = commonprefix([title, first_line])
    assert first_line[len(head)].isspace()
    assert task['description'] == prompt


@pytest.mark.asyncio
async def test_a_short_first_line_is_the_title_verbatim(backend, project_root):
    task = await _stored(backend, project_root, prompt='Build a frobinator\n\nBody')
    assert task['title'] == 'Build a frobinator'


@pytest.mark.asyncio
async def test_an_explicit_title_is_never_clipped(backend, project_root):
    title = 'word ' * 100 + 'end'
    task = await _stored(backend, project_root, title=title, description='d')
    assert task['title'] == title
