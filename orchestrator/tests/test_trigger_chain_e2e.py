"""The completion-driven trigger chain, end to end (plans/completion-driven-triggers-prd.md §4 boundary rows 1-16).

Every hop is production code: the snippet's own submit shapes, the fused-memory
submit guards behind a live MCP server over a real SqliteTaskBackend, a real
Scheduler time gate, a real DeterministicRunner executing the real factory
script, and a real EscalationQueue.  Only the host processes are stood in for.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip('fused_memory')

from _git_fixtures import seed_repo
from _trigger_chain_doc_shapes import (
    DocShapeError,
    authoring_metadata_exemplars,
    concretize,
    snippet_submit_calls,
    submit_calls,
)
from fused_memory.middleware.deterministic_task_guard import (
    deterministic_task_error,
    inject_task_kind,
)
from shared.repo_paths import DARK_FACTORY_ROOT_ENV
from shared.task_metadata import parse_metadata

pytestmark = [pytest.mark.usefixtures('code_default_config')]


@pytest.fixture
def factory_root(repo_root: Path | None, monkeypatch: pytest.MonkeyPatch) -> Path:
    if repo_root is None:
        pytest.skip('not running inside a dark-factory checkout')
    monkeypatch.setenv(DARK_FACTORY_ROOT_ENV, str(repo_root))
    return repo_root


def _doc_placeholders(project_root: Path) -> dict[str, str]:
    first_check = datetime.now(UTC) + timedelta(seconds=86400)
    return {
        'skill': 'review',
        'run_id': 'review-lambda-20261112-1',
        'project_root': str(project_root),
        'project_id': 'lambda_fixture',
        'now + 86400 s, ISO-8601 UTC': first_check.isoformat(),
    }


def _assert_accepted(task_kind: str, metadata: dict, project_root: Path) -> None:
    assert deterministic_task_error(task_kind, metadata, str(project_root)) is None
    _, warnings = parse_metadata(
        inject_task_kind(metadata, task_kind), direction='write', enforce=True
    )
    assert warnings == []


class TestDocShapes:
    def test_every_snippet_submit_call_passes_the_guard_and_the_model(
        self, factory_root: Path, tmp_path: Path
    ) -> None:
        project_root = seed_repo(tmp_path / 'project')
        placeholders = _doc_placeholders(project_root)
        calls = [concretize(call, placeholders) for call in snippet_submit_calls(factory_root)]

        assert {call['metadata']['trigger_chain']['role'] for call in calls} == {
            'completion_gate',
            'human_gate',
        }
        for call in calls:
            _assert_accepted(call['task_kind'], call['metadata'], project_root)

    def test_every_authoring_section_6_json_exemplar_passes(
        self, factory_root: Path, tmp_path: Path
    ) -> None:
        project_root = seed_repo(tmp_path / 'project')
        exemplars = authoring_metadata_exemplars(factory_root)

        assert exemplars
        for exemplar in exemplars:
            concrete = concretize(exemplar, _doc_placeholders(project_root))
            _assert_accepted(concrete['task_kind'], concrete, project_root)

    def test_a_block_that_cannot_be_extracted_fails_loudly(self) -> None:
        non_literal = "```python\nsubmit_task(project_root='x', metadata=some_name)\n```\n"
        with pytest.raises(DocShapeError, match=r'submit_task.*metadata'):
            submit_calls(non_literal)

        no_call = "```python\nadd_dependency(id='1', depends_on='2')\n```\n"
        with pytest.raises(DocShapeError, match='submit_task'):
            submit_calls(no_call)

        with pytest.raises(DocShapeError, match='<mystery>'):
            concretize({'title': 'about <mystery>'}, {'skill': 'review'})
