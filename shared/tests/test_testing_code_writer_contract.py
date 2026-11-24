"""Contract of shared/src/shared/testing_code_writer_contract.py: status words, write sites, census verdict, test existence."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from shared.task_statuses import TaskStatus
from shared.testing_code_writer_contract import (
    MCP_TOOL_CALLERS,
    WRITE_TOOLS,
    CensusVerdict,
    WriteSite,
    judge_census,
    missing_tests,
    task_status_words_in,
    write_sites,
)


def _sites(source: str, relpath: str = 'pkg/mod.py') -> tuple[WriteSite, ...]:
    return write_sites(ast.parse(textwrap.dedent(source)), relpath)


# ---------------------------------------------------------------------------
# task_status_words_in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('status', list(TaskStatus))
def test_every_task_status_is_found_as_a_word(status: TaskStatus) -> None:
    assert task_status_words_in(f'the task is {status} now') == (status.value,)


@pytest.mark.parametrize(('text', 'expected'), [
    pytest.param('Task 7 is in-progress.', ('in-progress',), id='hyphenated-status'),
    pytest.param('parked as merge-deferred', ('merge-deferred',), id='longest-hyphenated-wins'),
    pytest.param('Blocked by upstream', ('blocked',), id='case-insensitive'),
    pytest.param('DONE; then pending', ('done', 'pending'), id='every-occurrence-in-order'),
    pytest.param('transitioned to done;', ('done',), id='trailing-punctuation'),
])
def test_status_words_are_reported_in_order(text: str, expected: tuple[str, ...]) -> None:
    assert task_status_words_in(text) == expected


@pytest.mark.parametrize('text', [
    pytest.param('the branch was abandoned', id='inside-a-longer-word'),
    pytest.param('done_count=3', id='identifier-suffix'),
    pytest.param('the reviewer approved', id='status-as-word-prefix'),
    pytest.param('Completed: Steps completed 3/4', id='completion-template-words'),
    pytest.param('', id='empty'),
])
def test_clean_text_has_no_status_words(text: str) -> None:
    assert task_status_words_in(text) == ()


# ---------------------------------------------------------------------------
# write_sites
# ---------------------------------------------------------------------------


def test_the_tool_name_sets_are_the_four_writes_and_the_named_callers() -> None:
    assert WRITE_TOOLS == frozenset({'add_memory', 'add_episode', 'add_system_record', 'update_memory'})
    assert MCP_TOOL_CALLERS == frozenset({'post_mcp_tool_call', 'call_mcp_tool', 'call_tool', '_call_tool'})


@pytest.mark.parametrize('tool', sorted(WRITE_TOOLS))
def test_a_direct_write_call_is_a_site_on_any_receiver(tool: str) -> None:
    source = f"""
    async def plain():
        await {tool}(content='x')

    async def attribute(self):
        await self.memory_service.{tool}(content='x')
    """
    assert _sites(source) == (
        WriteSite(path='pkg/mod.py', function='plain', callee=tool),
        WriteSite(path='pkg/mod.py', function='attribute', callee=tool),
    )


def test_function_is_the_dotted_enclosing_chain_including_closures() -> None:
    source = """
    memory.add_memory(content='top')

    class Writer:
        async def write(self):
            async def _inner():
                await self.svc.add_episode(content='x')
            await _inner()

        def sync_write(self):
            self.svc.update_memory(memory_id='m')
    """
    assert _sites(source) == (
        WriteSite(path='pkg/mod.py', function='<module>', callee='add_memory'),
        WriteSite(path='pkg/mod.py', function='Writer.write._inner', callee='add_episode'),
        WriteSite(path='pkg/mod.py', function='Writer.sync_write', callee='update_memory'),
    )


def test_an_mcp_caller_passing_a_write_tool_literal_is_a_site_named_by_the_tool() -> None:
    source = """
    async def write(client, url, session):
        await post_mcp_tool_call(client, url, 'add_memory', {'content': 'x'})
        await session.call_tool('add_system_record', {})
        await call_mcp_tool(url, name='add_episode')
    """
    assert _sites(source) == (
        WriteSite(path='pkg/mod.py', function='write', callee='add_memory'),
        WriteSite(path='pkg/mod.py', function='write', callee='add_system_record'),
        WriteSite(path='pkg/mod.py', function='write', callee='add_episode'),
    )


def test_an_mcp_caller_passing_a_module_level_tool_constant_is_a_site() -> None:
    source = """
    WRITE_TOOL = 'update_memory'
    READ_TOOL = 'search'

    class Store:
        async def heal(self, arguments):
            await self._call_tool(WRITE_TOOL, arguments)
            await self._call_tool(READ_TOOL, arguments)
    """
    assert _sites(source) == (
        WriteSite(path='pkg/mod.py', function='Store.heal', callee='update_memory'),
    )


def test_an_mcp_caller_passing_a_read_tool_is_not_a_site() -> None:
    assert _sites("post_mcp_tool_call(client, url, 'search', {})\n") == ()


@pytest.mark.parametrize('source', [
    pytest.param('''
        def f():
            """Calls add_memory on the service."""
            # memory_service.add_memory(...) used to live here
            return 1
        ''', id='docstring-and-comment'),
    pytest.param(
        "journal.add_run_action(run_id, 'write', 'memory', 'add_memory', {})\n",
        id='journal-row-naming-the-tool'),
    pytest.param("record(operation='add_memory')\n", id='operation-keyword'),
    pytest.param('''
        async def add_memory(content):
            return content
        ''', id='a-definition-named-add_memory'),
])
def test_mentions_that_are_not_calls_to_a_write_are_not_sites(source: str) -> None:
    assert _sites(source) == ()


# ---------------------------------------------------------------------------
# judge_census
# ---------------------------------------------------------------------------

_SITE = WriteSite(path='a.py', function='f', callee='add_memory')
_KEY = ('a.py', 'f', 'add_memory')
_OTHER_KEY = ('b.py', 'g', 'add_episode')


def test_census_agrees_when_found_counts_equal_listed_counts() -> None:
    verdict = judge_census([_SITE, _SITE], {_KEY: 2})
    assert verdict == CensusVerdict(unlisted=(), stale=())


def test_a_site_found_more_times_than_listed_is_unlisted() -> None:
    assert judge_census([_SITE, _SITE], {_KEY: 1}).unlisted == (_KEY,)
    assert judge_census([_SITE], {}).unlisted == (_KEY,)


def test_a_key_listed_more_times_than_found_is_stale() -> None:
    verdict = judge_census([_SITE], {_KEY: 2, _OTHER_KEY: 1})
    assert verdict.unlisted == ()
    assert verdict.stale == (_KEY, _OTHER_KEY)


def test_the_census_verdict_is_frozen() -> None:
    verdict = judge_census([], {})
    with pytest.raises(AttributeError):
        verdict.unlisted = (_KEY,)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# missing_tests
# ---------------------------------------------------------------------------


def _write_tests_tree(root: Path) -> None:
    (root / 'pkg').mkdir()
    (root / 'pkg' / 'test_mod.py').write_text(textwrap.dedent("""
        def test_top():
            pass

        class TestGroup:
            async def test_member(self):
                pass

            class TestNested:
                def test_deep(self):
                    pass
    """))


def test_existing_node_ids_are_not_missing(tmp_path: Path) -> None:
    _write_tests_tree(tmp_path)
    present = (
        'pkg/test_mod.py::test_top',
        'pkg/test_mod.py::TestGroup::test_member',
        'pkg/test_mod.py::TestGroup::TestNested::test_deep',
        'pkg/test_mod.py::test_top[param-id]',
    )
    assert missing_tests(present, tmp_path) == ()


def test_node_ids_naming_an_absent_file_or_function_are_missing(tmp_path: Path) -> None:
    _write_tests_tree(tmp_path)
    absent = (
        'pkg/test_gone.py::test_top',
        'pkg/test_mod.py::test_never_written',
        'pkg/test_mod.py::TestGroup::test_top',
        'pkg/test_mod.py::TestMissing::test_member',
    )
    assert missing_tests(('pkg/test_mod.py::test_top', *absent), tmp_path) == absent
