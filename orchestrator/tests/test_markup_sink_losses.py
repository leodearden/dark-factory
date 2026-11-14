"""A record the markup escalation sink cannot queue leaves one durable loss line (task 7128).

plan-tools and verdict-tools are per-agent stdio subprocesses, and their stderr
reaches no durable sink. So the sink's ERROR line for a lost record was the
whole account of that loss: nothing preserved it and nothing could count it.
``markup_sink.make_escalation_sink`` now hands every such record to an injected
``loss_sink``, and each server wires in the markup journal it already uses as
its fact sink. These rows read the loss back off that journal.

The PAYLOAD floor is unchanged: the queue or nothing (task 4584). A loss line
names the loss and the payload's size, never the payload.

Subjects are NUMERIC throughout: ``shared.eval_lane.eval_lane_provenance``
drops fixture-grammar subjects, and an eval-lane record is deliberately not
filed, which is not a loss.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _plan_markup_fixtures
import pytest
from _git_checkouts import git
from _markup_corpus import corpus_specimen
from _markup_sink_rig import QUEUE_LOSS_SHAPES, RecordingQueue, seed_plan
from escalation.models import Escalation
from fastmcp import Client
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import (
    FACT_MARKUP_DETECTED,
    MARKUP_UNRECOVERED_PARAM_ERROR_TYPE,
    OUTCOMES,
)
from shared.project_layout import escalation_queue_dir

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import (
    markup_journal,
    markup_sink,
    plan_markup_stamp,
    plan_tools,
    verdict_tools,
)
from orchestrator.review_issue_contract import REVIEW_ISSUE_OFF_CONTRACT_FACT

#: pytest finds a fixture by its name in the requesting module's namespace.
isolate_the_refusal_memo = _plan_markup_fixtures.isolate_the_refusal_memo

SUBJECT = '7128'
RAW = 'THE ONLY SURVIVING COPY'
REVIEWER_ROLE = 'test_analyst'

#: A real ``submit_review_verdict`` payload the guard refuses, driven as ``summary``.
VERDICT_SPECIMEN = 'toolu_012YjuXbKZAMwNAo9WR4Pvjx'

#: A real ``add_design_decision.decision`` payload the guard refuses.
PLAN_SPECIMEN = 'toolu_01CX8okLpKgz5auenVYQzY23'


@pytest.fixture(autouse=True)
def _clear_pending_stamps():
    """plan-tools' stamp buffer is process-global; one row must not inflate the next."""
    plan_markup_stamp.clear_pending()
    yield
    plan_markup_stamp.clear_pending()


def unrepairable_value(tool_use_id: str) -> str:
    """The committed corpus payload for *tool_use_id*, its outcome asserted."""
    record = corpus_specimen(tool_use_id)
    assert record['expected_outcome'] == 'unrepairable'
    return record['value']


def journal_lines(root: Path, server_label: str) -> list[dict[str, Any]]:
    path = markup_journal.journal_path(root, server_label)
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def lines_of(root: Path, server_label: str, fact: str) -> list[dict[str, Any]]:
    return [line for line in journal_lines(root, server_label) if line.get('fact') == fact]


def loss_lines(root: Path, server_label: str) -> list[dict[str, Any]]:
    return lines_of(root, server_label, markup_sink.MARKUP_RECORD_LOST_FACT)


def succeeds_once(root: Path) -> Callable[[Path], Path | None]:
    """A project-root resolver that answers *root* once and ``None`` ever after."""
    answers = [root]
    return lambda worktree: answers.pop() if answers else None


def block_the_queue(root: Path) -> None:
    """Put a regular file where the queue directory belongs, so it cannot open."""
    queue_dir = escalation_queue_dir(root)
    queue_dir.parent.mkdir(parents=True, exist_ok=True)
    queue_dir.write_text('not a directory\n', encoding='utf-8')


def task_artifacts(worktree: Path) -> TaskArtifacts:
    artifacts = TaskArtifacts(worktree)
    artifacts.init(SUBJECT, 'Test task', 'A test')
    seed_plan(artifacts, SUBJECT)
    return artifacts


# ---------------------------------------------------------------------------
# The sink's own contract, driven directly.
# ---------------------------------------------------------------------------


def verdict_journal(root: Path):
    return markup_journal.make_fact_journal(
        worktree=root,
        server_label=verdict_tools.MARKUP_SINK_SPEC.server_label,
        subject_task_id=lambda: SUBJECT,
        resolve_root=lambda _wt: root,
    )


def sink_over(
    root: Path,
    *,
    resolve_root: Callable[[Path], Path | None],
    open_channel: Callable[[Path], Any] = lambda _root: None,
    loss_sink: Callable[[dict[str, Any]], Any] | None = None,
):
    return markup_sink.make_escalation_sink(
        worktree=root,
        spec=verdict_tools.MARKUP_SINK_SPEC,
        subject_task_id=lambda: SUBJECT,
        resolve_root=resolve_root,
        open_channel=open_channel,
        loss_sink=verdict_journal(root) if loss_sink is None else loss_sink,
    )


def residue_record(
    error_type: str = markup_sink.MARKUP_RESIDUE_ERROR_TYPE,
) -> dict[str, Any]:
    return {
        'error_type': error_type,
        'tool': 'submit_review_verdict',
        'field': 'issues',
        'raw_value': RAW,
    }


def storm_record() -> dict[str, Any]:
    return {
        'error_type': markup_sink.MARKUP_STORM_ERROR_TYPE,
        'count': 7,
        'threshold': 5,
        'window_seconds': 3600,
        'outcome': 'repaired',
        'project': None,
        'callers': [],
    }


#: Both residue kinds: a refusal's, and FORWARD_REPAIR's dropped slice.
RESIDUE_KINDS = [
    pytest.param(markup_sink.MARKUP_RESIDUE_ERROR_TYPE, id='refused'),
    pytest.param(MARKUP_UNRECOVERED_PARAM_ERROR_TYPE, id='dropped-slice'),
]


class TestEveryLostRecordIsJournalled:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('error_type', RESIDUE_KINDS)
    @pytest.mark.parametrize(('resolve_root', 'open_channel', 'cause'), QUEUE_LOSS_SHAPES)
    async def test_a_residue_the_queue_cannot_take_leaves_one_loss_line(
        self, tmp_path, resolve_root, open_channel, cause, error_type,
    ):
        sink = sink_over(tmp_path, resolve_root=resolve_root, open_channel=open_channel)

        assert await sink(residue_record(error_type)) is None

        (line,) = journal_lines(tmp_path, 'verdict-tools')
        assert line['fact'] == markup_sink.MARKUP_RECORD_LOST_FACT
        assert line['outcome'] == markup_sink.MARKUP_RECORD_LOST_OUTCOME
        assert line['cause'] == cause.value
        assert {key: line[key] for key in (
            'error_type', 'tool', 'param', 'raw_value_chars', 'subject_task_id', 'server',
        )} == {
            'error_type': error_type,
            'tool': 'submit_review_verdict',
            'param': 'issues',
            'raw_value_chars': len(RAW),
            'subject_task_id': SUBJECT,
            'server': 'verdict-tools',
        }
        journal = markup_journal.journal_path(tmp_path, 'verdict-tools').read_bytes()
        assert RAW.encode('utf-8') not in journal, 'the payload floor is the queue or nothing'

    @pytest.mark.asyncio
    async def test_a_lost_burst_alarm_leaves_a_payload_free_loss_line(self, tmp_path):
        sink = sink_over(tmp_path, resolve_root=lambda worktree: None)

        assert await sink(storm_record()) is None

        (line,) = journal_lines(tmp_path, 'verdict-tools')
        assert {key: line[key] for key in (
            'fact', 'outcome', 'cause', 'error_type', 'tool', 'param', 'raw_value_chars',
        )} == {
            'fact': markup_sink.MARKUP_RECORD_LOST_FACT,
            'outcome': markup_sink.MARKUP_RECORD_LOST_OUTCOME,
            'cause': 'project_root_unresolved',
            'error_type': 'mcp_markup_storm',
            'tool': None,
            'param': None,
            'raw_value_chars': None,
        }

    @pytest.mark.asyncio
    async def test_a_sink_that_fails_outright_is_journalled_as_sink_failed(
        self, tmp_path, caplog,
    ):
        def exploding_root(worktree: Path) -> Path | None:
            raise RuntimeError('the resolver blew up')

        sink = sink_over(tmp_path, resolve_root=exploding_root)

        with caplog.at_level(logging.INFO, logger='orchestrator.mcp.markup_sink'):
            assert await sink(residue_record()) is None

        (line,) = loss_lines(tmp_path, 'verdict-tools')
        assert line['cause'] == markup_sink.LossCause.SINK_FAILED.value
        (error,) = [
            r for r in caplog.records
            if r.name == 'orchestrator.mcp.markup_sink' and r.levelno >= logging.ERROR
        ]
        assert isinstance(error.exc_info[1], RuntimeError)
        assert 'submit_review_verdict.issues' in error.getMessage(), (
            'a sink failure must give the same account of the loss as every other cause'
        )
        assert f'{len(RAW)}-char raw value' in error.getMessage()

    @pytest.mark.asyncio
    async def test_a_filed_record_is_not_a_loss(self, tmp_path):
        queue = RecordingQueue()
        sink = sink_over(
            tmp_path,
            resolve_root=lambda worktree: worktree,
            open_channel=lambda root: (Escalation, queue),
        )

        assert await sink(residue_record()) == 'esc-verdict-tools-markup-residue-1'

        assert len(queue.submitted) == 1
        assert not markup_journal.journal_path(tmp_path, 'verdict-tools').exists()

    @pytest.mark.asyncio
    async def test_a_failing_loss_sink_costs_only_its_own_line(self, tmp_path, caplog):
        handed: list[dict[str, Any]] = []

        async def unwritable(record: dict[str, Any]) -> None:
            handed.append(record)
            raise OSError('the journal is unwritable')

        sink = sink_over(tmp_path, resolve_root=lambda worktree: None, loss_sink=unwritable)

        with caplog.at_level(logging.DEBUG, logger='orchestrator.mcp.markup_sink'):
            assert await sink(residue_record()) is None

        assert len(handed) == 1
        logged = [r for r in caplog.records if r.name == 'orchestrator.mcp.markup_sink']
        errors = [r for r in logged if r.levelno >= logging.ERROR]
        assert len(errors) == 1, [r.getMessage() for r in errors]
        (failure,) = [
            r for r in logged
            if r.exc_info is not None and isinstance(r.exc_info[1], OSError)
        ]
        assert failure.levelno < logging.ERROR, (
            'the loss already has its ERROR line; a second would double-count one event'
        )


class TestTheLossVocabularyIsItsOwn:
    """verdict-tools' attribution_source tells an operator to select markup
    lines by ``outcome``, and a line cut to the journal's floor keys keeps
    nothing else to tell its kind by."""

    def test_the_loss_outcome_is_no_other_lines_outcome(self):
        assert markup_sink.MARKUP_RECORD_LOST_OUTCOME not in OUTCOMES
        assert markup_sink.MARKUP_RECORD_LOST_OUTCOME not in {
            outcome.value for outcome in verdict_tools.IssueIntakeOutcome
        }

    def test_the_loss_fact_is_no_other_lines_fact(self):
        assert markup_sink.MARKUP_RECORD_LOST_FACT != FACT_MARKUP_DETECTED
        assert markup_sink.MARKUP_RECORD_LOST_FACT != REVIEW_ISSUE_OFF_CONTRACT_FACT


# ---------------------------------------------------------------------------
# Through the REAL servers.
# ---------------------------------------------------------------------------


async def refuse_a_verdict(
    artifacts: TaskArtifacts, resolve_project_root: Callable[[Path], Path | None],
) -> dict[str, Any]:
    server = verdict_tools.create_server(
        artifacts, REVIEWER_ROLE, resolve_project_root=resolve_project_root,
    )
    async with Client(server) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool('submit_review_verdict', {
                'reviewer': REVIEWER_ROLE,
                'verdict': 'ISSUES_FOUND',
                'issues': [],
                'summary': unrepairable_value(VERDICT_SPECIMEN),
            })
    return json.loads(str(excinfo.value))


async def refuse_a_decision(artifacts: TaskArtifacts) -> dict[str, Any]:
    server = plan_tools.create_server(artifacts)
    async with Client(server) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool('add_design_decision', {
                'decision': unrepairable_value(PLAN_SPECIMEN),
            })
    return json.loads(str(excinfo.value))


class TestVerdictToolsJournalsItsLosses:
    @pytest.mark.asyncio
    async def test_an_unopenable_queue_is_journalled_beside_the_fact(self, tmp_path):
        artifacts = task_artifacts(tmp_path)
        block_the_queue(tmp_path)

        payload = await refuse_a_verdict(artifacts, lambda worktree: tmp_path)

        assert payload['escalation_id'] is None
        (fact,) = lines_of(tmp_path, 'verdict-tools', FACT_MARKUP_DETECTED)
        (loss,) = loss_lines(tmp_path, 'verdict-tools')
        assert fact['outcome'] == 'unrepairable'
        assert {key: loss[key] for key in ('cause', 'tool', 'param', 'raw_value_chars')} == {
            'cause': 'queue_unopened',
            'tool': 'submit_review_verdict',
            'param': 'summary',
            'raw_value_chars': len(unrepairable_value(VERDICT_SPECIMEN)),
        }
        assert fact['subject_task_id'] == loss['subject_task_id'] == SUBJECT

    @pytest.mark.asyncio
    async def test_the_fact_journal_is_the_loss_sink(self, tmp_path):
        """The fact line lands first and memoizes the root, so the escalation
        sink's own failed resolution is still journalled. A separately built
        journal would re-resolve, get nothing, and write nothing."""
        artifacts = task_artifacts(tmp_path)

        await refuse_a_verdict(artifacts, succeeds_once(tmp_path))

        (loss,) = loss_lines(tmp_path, 'verdict-tools')
        assert loss['cause'] == 'project_root_unresolved'


class TestPlanToolsJournalsItsLosses:
    @pytest.mark.asyncio
    async def test_an_unopenable_queue_is_journalled_beside_the_fact(self, tmp_path):
        checkout = tmp_path / 'repo'
        checkout.mkdir()
        git('init', cwd=checkout)
        root = checkout.resolve()
        artifacts = task_artifacts(checkout)
        block_the_queue(root)

        payload = await refuse_a_decision(artifacts)

        assert payload['escalation_id'] is None
        (fact,) = lines_of(root, 'plan-tools', FACT_MARKUP_DETECTED)
        (loss,) = loss_lines(root, 'plan-tools')
        assert fact['outcome'] == 'unrepairable'
        assert {key: loss[key] for key in ('cause', 'tool', 'param', 'raw_value_chars')} == {
            'cause': 'queue_unopened',
            'tool': 'add_design_decision',
            'param': 'decision',
            'raw_value_chars': len(unrepairable_value(PLAN_SPECIMEN)),
        }
        assert fact['subject_task_id'] == loss['subject_task_id'] == SUBJECT

    @pytest.mark.asyncio
    async def test_the_fact_journal_is_the_loss_sink(self, monkeypatch, tmp_path):
        artifacts = task_artifacts(tmp_path)
        monkeypatch.setattr(plan_tools, '_markup_project_root', succeeds_once(tmp_path))

        await refuse_a_decision(artifacts)

        (loss,) = loss_lines(tmp_path, 'plan-tools')
        assert loss['cause'] == 'project_root_unresolved'
