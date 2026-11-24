"""Contract of shared/src/shared/testing_code_writer_contract.py: status words, write sites, census verdict, test existence, census rows."""

from __future__ import annotations

import ast
import dataclasses
import textwrap
from pathlib import Path

import pytest

from shared.task_statuses import TaskStatus
from shared.testing_code_writer_contract import (
    MCP_TOOL_CALLERS,
    TESTED_DISPOSITIONS,
    WRITE_TOOLS,
    CensusRow,
    CensusVerdict,
    Disposition,
    WriteSite,
    census_failure_message,
    census_row_problems,
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
    assert frozenset({'add_memory', 'add_episode', 'add_system_record', 'update_memory'}) == WRITE_TOOLS
    assert frozenset({'post_mcp_tool_call', 'call_mcp_tool', 'call_tool', '_call_tool'}) == MCP_TOOL_CALLERS


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


@pytest.mark.parametrize('declaration', [
    pytest.param("WRITE_TOOL = 'update_memory'", id='plain'),
    pytest.param("WRITE_TOOL: Final = 'update_memory'", id='final'),
    pytest.param("WRITE_TOOL: str = 'update_memory'", id='typed'),
])
def test_an_mcp_caller_passing_a_module_level_tool_constant_is_a_site(declaration: str) -> None:
    source = f"""
    {declaration}
    READ_TOOL = 'search'
    UNSET_TOOL: str

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
    field = 'unlisted'
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(verdict, field, (_KEY,))


def test_an_agreeing_verdict_has_no_failure_message() -> None:
    verdict = judge_census([_SITE], {_KEY: 1})
    assert verdict.agrees
    assert census_failure_message(verdict, Path('pkg/src')) == ''


def test_the_failure_message_names_the_root_each_unlisted_site_and_each_stale_row() -> None:
    verdict = judge_census([_SITE], {_OTHER_KEY: 1})
    message = census_failure_message(verdict, Path('pkg/src'))
    assert not verdict.agrees
    assert f'under pkg/src that CENSUS does not list: {[_KEY]}' in message
    assert f'CENSUS rows with fewer sites than listed: {[_OTHER_KEY]}' in message


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


# ---------------------------------------------------------------------------
# census rows
# ---------------------------------------------------------------------------

_PRESENT_TEST = 'pkg/test_mod.py::test_top'


def test_only_code_writers_and_model_prose_need_a_test() -> None:
    assert {Disposition.CODE_WRITER, Disposition.MODEL_PROSE} == TESTED_DISPOSITIONS


def test_the_census_row_is_frozen() -> None:
    row = CensusRow(count=1, disposition=Disposition.RELAY, reason='relays a caller')
    field = 'count'
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(row, field, 2)


def test_a_census_of_well_formed_rows_has_no_problems(tmp_path: Path) -> None:
    _write_tests_tree(tmp_path)
    census = {
        _KEY: CensusRow(1, Disposition.CODE_WRITER, 'writes a note', tests=(_PRESENT_TEST,)),
        _OTHER_KEY: CensusRow(1, Disposition.STORE_LAYER, 'the store'),
        ('c.py', 'h', 'add_memory'): CensusRow(1, Disposition.RETIRING, 'deleted by task 10087'),
    }
    assert census_row_problems(census, tmp_path) == ()


@pytest.mark.parametrize('disposition', sorted(TESTED_DISPOSITIONS))
def test_a_tested_disposition_naming_no_test_is_a_problem(tmp_path: Path, disposition: Disposition) -> None:
    [problem] = census_row_problems({_KEY: CensusRow(1, disposition, 'writes a note')}, tmp_path)
    assert problem.key == _KEY


def test_a_row_naming_a_test_that_does_not_exist_is_a_problem(tmp_path: Path) -> None:
    _write_tests_tree(tmp_path)
    row = CensusRow(1, Disposition.MODEL_PROSE, 'forwards prose', tests=(_PRESENT_TEST, 'pkg/test_mod.py::test_gone'))
    [problem] = census_row_problems({_KEY: row}, tmp_path)
    assert problem.key == _KEY
    assert 'pkg/test_mod.py::test_gone' in str(problem)


def test_a_retiring_row_must_name_the_task_that_retires_it(tmp_path: Path) -> None:
    [problem] = census_row_problems({_KEY: CensusRow(1, Disposition.RETIRING, 'goes away soon')}, tmp_path)
    assert problem.key == _KEY
