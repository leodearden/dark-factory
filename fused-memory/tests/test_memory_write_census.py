"""Census of every memory-write call site under fused-memory/src/fused_memory, for C5 of plans/live-task-facts-at-recall-prd.md.

C5: memory written by code links its subject task structurally and writes no
task-status value. Each write site is listed under its (path, enclosing
qualname, write tool) key with its count and disposition. A code writer or a
model-prose forwarder names the behaviour test that holds it to C5. The
instruments are shared/src/shared/testing_code_writer_contract.py's, read by
the orchestrator's census too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _ast_guard import parse_python_module
from shared.testing_code_writer_contract import (
    WRITE_TOOLS,
    CensusKey,
    CensusRow,
    WriteSite,
    census_row_problems,
    judge_census,
    write_sites,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

TESTS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TESTS_DIR.parent / 'src' / 'fused_memory'

CENSUS: dict[CensusKey, CensusRow] = {}


def _write_sites_under(root: Path) -> tuple[WriteSite, ...]:
    sites: list[WriteSite] = []
    for path in sorted(root.rglob('*.py')):
        source = path.read_text(encoding='utf-8')
        if any(tool in source for tool in WRITE_TOOLS):
            sites.extend(write_sites(parse_python_module(path), path.relative_to(root).as_posix()))
    return tuple(sites)


def _listed_counts() -> dict[CensusKey, int]:
    return {key: row.count for key, row in CENSUS.items()}


def test_every_write_site_is_listed() -> None:
    verdict = judge_census(_write_sites_under(SRC_ROOT), _listed_counts())

    assert verdict.unlisted == (), (
        f'write sites under {SRC_ROOT} that CENSUS does not list: {list(verdict.unlisted)}. '
        'List each with its count, disposition and reason; a CODE_WRITER or MODEL_PROSE row '
        "also names the behaviour test, in the writer's own module test file, that holds it to C5."
    )
    assert verdict.stale == (), f'CENSUS rows with fewer sites than listed: {list(verdict.stale)}'


def test_every_row_keeps_the_census_rules() -> None:
    assert census_row_problems(CENSUS, TESTS_DIR) == ()


def test_an_unlisted_write_site_is_named(tmp_path: Path) -> None:
    (tmp_path / 'notes').mkdir()
    (tmp_path / 'notes' / 'writer.py').write_text(
        'async def write_note(memory_service):\n'
        "    await memory_service.add_memory(content='a note', category='observations_and_summaries')\n",
        encoding='utf-8',
    )

    verdict = judge_census(_write_sites_under(tmp_path), _listed_counts())

    assert verdict.unlisted == (('notes/writer.py', 'write_note', 'add_memory'),)
