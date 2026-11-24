"""Census of every memory-write call site under orchestrator/src/orchestrator, for C5 of plans/live-task-facts-at-recall-prd.md.

C5: memory written by code links its subject task structurally and writes no
task-status value. Each write site is listed under its (path, enclosing
qualname, write tool) key with its count and disposition. A code writer or a
model-prose forwarder names the behaviour test that holds it to C5. The
instruments are shared/src/shared/testing_code_writer_contract.py's, read by
fused-memory's census too; the sites come from the shared tree pass.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import pytest
from _tree_scan import (
    ORCHESTRATOR_SRC,
    ORCHESTRATOR_SRC_DIR,
    TESTS_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    python_files,
)
from shared.testing_code_writer_contract import (
    WRITE_TOOLS,
    CensusKey,
    CensusRow,
    Disposition,
    WriteSite,
    census_failure_message,
    census_row_problems,
    judge_census,
    write_sites,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

_WRITERS = 'test_workflow_memory_writers.py'

CENSUS: dict[CensusKey, CensusRow] = {
    ('workflow.py', 'TaskWorkflow._write_completion_to_memory', 'add_memory'): CensusRow(
        1, Disposition.CODE_WRITER, 'the completion record, carrying the registered completion_record_metadata',
        tests=(f'{_WRITERS}::TestCompletionRecord::test_it_carries_the_registered_marker_and_names_no_status',),
    ),
    ('workflow.py', 'TaskWorkflow._write_decisions_to_memory', 'add_memory'): CensusRow(
        1, Disposition.MODEL_PROSE, "forwards the architect's decisions, each declaring and naming its task",
        tests=(
            f'{_WRITERS}::TestDecisionNotes::test_each_note_declares_its_task_and_names_it',
            f'{_WRITERS}::TestDecisionNotes::test_a_non_numeric_task_id_declares_nothing',
        ),
    ),
    ('workflow.py', 'TaskWorkflow._write_suggestions_to_memory', 'add_memory'): CensusRow(
        1, Disposition.MODEL_PROSE, "forwards reviewer suggestions as conventions; they have no subject task",
        tests=(f'{_WRITERS}::TestSuggestionNotes::test_they_forward_the_reviewers_prose_and_declare_no_task',),
    ),
}


def _write_sites_relative_to(root: Path) -> Callable[[ParsedModule], tuple[WriteSite, ...]]:
    resolved_root = root.resolve()

    def extract(module: ParsedModule) -> tuple[WriteSite, ...]:
        if not module.may_name(*WRITE_TOOLS):
            return ()
        return write_sites(module.tree, module.path.relative_to(resolved_root).as_posix())

    return extract


ANALYSIS = TreeAnalysis(
    name=Path(__file__).stem, files=ORCHESTRATOR_SRC, extract=_write_sites_relative_to(ORCHESTRATOR_SRC_DIR),
)
TREE_ANALYSES = (ANALYSIS,)


def _found(tree_pass: TreePass, analysis: TreeAnalysis[tuple[WriteSite, ...]]) -> list[WriteSite]:
    return [site for sites in tree_pass.facts(analysis).values() for site in sites]


def _listed_counts() -> dict[CensusKey, int]:
    return {key: row.count for key, row in CENSUS.items()}


def test_every_write_site_is_listed(tree_pass: TreePass) -> None:
    verdict = judge_census(_found(tree_pass, ANALYSIS), _listed_counts())

    assert verdict.agrees, census_failure_message(verdict, ORCHESTRATOR_SRC_DIR)


def test_every_row_keeps_the_census_rules() -> None:
    assert census_row_problems(CENSUS, TESTS_DIR) == ()


def test_an_unlisted_write_site_is_named(tmp_path: Path) -> None:
    (tmp_path / 'writer.py').write_text(
        'async def write_note(client, url):\n'
        "    await post_mcp_tool_call(client, url, 'add_memory', {'content': 'a note'})\n",
        encoding='utf-8',
    )
    fixture = TreeAnalysis(
        name='fixture', files=functools.partial(python_files, tmp_path), extract=_write_sites_relative_to(tmp_path),
    )

    verdict = judge_census(_found(TreePass.run([fixture]), fixture), _listed_counts())

    assert verdict.unlisted == (('writer.py', 'write_note', 'add_memory'),)
