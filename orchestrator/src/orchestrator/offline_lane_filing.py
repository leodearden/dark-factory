"""The ``submit_task`` argument blocks of the offline lane's fix tasks (PRD 5.5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

MAX_TESTS_SHOWN = 10
FAILURE_LINE_MAX_CHARS = 300
HEAD_PREFIX_CHARS = 12
SPAWN_CONTEXT = 'offline_lane_red'
MERGE_LANE = 'normal'
UNKNOWN_HEAD = 'unknown'


@dataclass(frozen=True)
class EvidenceNote:
    source: str
    node: str | None
    text: str


@dataclass(frozen=True)
class CountedPassEvidence:
    pass_id: str
    head: str
    started_at: datetime
    mode: str
    detect_first_line: str | None
    confirm_first_line: str | None
    record_dir: str


@dataclass(frozen=True)
class TestEvidence:
    node: str
    kind: str
    passes: tuple[CountedPassEvidence, ...]
    first_red_head: str
    last_green_head: str | None
    notes: tuple[EvidenceNote, ...]
    prior_task_ids: tuple[str, ...]


def _head_prefix(head: str) -> str:
    return head[:HEAD_PREFIX_CHARS]


def _bulleted(items: Sequence[str]) -> str:
    return '\n'.join(f'  - {item}' for item in items)


def _distinct_passes(passes: Sequence[CountedPassEvidence]) -> list[CountedPassEvidence]:
    by_id = {p.pass_id: p for p in passes}
    return sorted(by_id.values(), key=lambda p: (p.started_at, p.pass_id))


def _distinct_in_order(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _pass_lines(counted: CountedPassEvidence) -> list[str]:
    phases = (('detect', counted.detect_first_line), ('confirm', counted.confirm_first_line))
    return [
        f'  pass {counted.pass_id} {phase:<7}: {line[:FAILURE_LINE_MAX_CHARS]}'
        for phase, line in phases
        if line is not None
    ]


def _test_block(test: TestEvidence) -> str:
    lines = [f'{test.node}   kind: {test.kind}']
    for counted in test.passes:
        lines.extend(_pass_lines(counted))
    return '\n'.join(lines)


def _tests_section(tests: Sequence[TestEvidence]) -> str:
    shown = [_test_block(test) for test in tests[:MAX_TESTS_SHOWN]]
    hidden = len(tests) - MAX_TESTS_SHOWN
    if hidden > 0:
        shown.append(f'and {hidden} more')
    return '\n\n'.join(shown)


def _passes_table(passes: Sequence[CountedPassEvidence]) -> str:
    rows = [
        f'  {p.pass_id} | {p.head} | {p.started_at.isoformat()} | {p.mode}' for p in passes
    ]
    return 'Counted passes (pass id | head | started | outcome mode):\n' + '\n'.join(rows)


def _notes_section(tests: Sequence[TestEvidence]) -> str | None:
    rows = [
        f'  [{note.source}] {note.node or "(all reproduced tests)"}: {note.text}'
        for test in tests
        for note in test.notes
    ]
    return 'Hook notes:\n' + '\n'.join(rows) if rows else None


def _prior_tasks_section(tests: Sequence[TestEvidence]) -> str | None:
    rows = [
        f'  {test.node}: {", ".join(test.prior_task_ids)}' for test in tests if test.prior_task_ids
    ]
    return 'Prior lane tasks for these tests:\n' + '\n'.join(rows) if rows else None


def _owners_section(tests: Sequence[TestEvidence], owners: Mapping[str, str]) -> str | None:
    rows = [f'  {test.node}: task {owners[test.node]}' for test in tests if test.node in owners]
    return 'Live de-flake owners:\n' + '\n'.join(rows) if rows else None


def _record_dirs_section(passes: Sequence[CountedPassEvidence]) -> str:
    return 'Record directories:\n' + _bulleted(_distinct_in_order([p.record_dir for p in passes]))


def _join_sections(sections: Sequence[str | None]) -> str:
    return '\n\n'.join(section for section in sections if section is not None)


def _earliest_first_red(tests: Sequence[TestEvidence]) -> TestEvidence:
    return min(tests, key=lambda test: min(p.started_at for p in test.passes))


def _suspect_range(test: TestEvidence) -> str:
    return f'{test.last_green_head or UNKNOWN_HEAD}..{test.first_red_head}'


def _reproduce_text(command: str, reproduce: str | None) -> str:
    return reproduce if reproduce is not None else f'run the whole command: {command}'


def _lane_metadata(
    *,
    command: str,
    leg: str,
    nodes: Sequence[str],
    first_red_head: str,
    last_green_head: str | None,
    passes: Sequence[CountedPassEvidence],
    prior_task_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        'spawn_context': SPAWN_CONTEXT,
        'merge_lane': MERGE_LANE,
        'x_offline_lane': {
            'command': command,
            'leg': leg,
            'nodes': list(nodes),
            'first_red_head': first_red_head,
            'last_green_head': last_green_head,
            'pass_ids': [p.pass_id for p in passes],
            'prior_task_ids': list(prior_task_ids),
        },
    }


def _require_evidence(tests: Sequence[TestEvidence]) -> None:
    if not tests or any(not test.passes for test in tests):
        raise ValueError('a lane fix task needs at least one test, each with a counted pass')


def build_lane_fix_task_arguments(
    *,
    command: str,
    leg: str,
    priority: str,
    tests: Sequence[TestEvidence],
    reproduce: str | None,
    deflake_owners: Mapping[str, str] = {},
) -> dict[str, Any]:
    """The test-level fix task of PRD 5.5.

    Tests render in the order given. The suspect range and the title's head come from
    the test whose first counted pass started earliest. No ``project_root`` and no
    ``metadata.files``, as ``flake_ledger.py::build_deflake_task_arguments`` explains.
    """
    _require_evidence(tests)
    passes = _distinct_passes([p for test in tests for p in test.passes])
    earliest = _earliest_first_red(tests)
    prior_task_ids = _distinct_in_order([t for test in tests for t in test.prior_task_ids])
    title = (
        f'offline-lane[{command}]: {len(tests)} test(s) red on {len(passes)} passes '
        f'since {_head_prefix(earliest.first_red_head)} — {tests[0].node}'
    )
    shown_nodes = [test.node for test in tests[:MAX_TESTS_SHOWN]]
    hidden = len(tests) - len(shown_nodes)
    description = _join_sections([
        f'The offline lane reproduced a red run of `{command}` (leg: {leg}).',
        'Tests:\n' + _bulleted(shown_nodes + ([f'and {hidden} more'] if hidden > 0 else [])),
        f'Suspect range (last all-green head..first red head): {_suspect_range(earliest)}',
        f'Reproduce: {_reproduce_text(command, reproduce)}',
    ])
    details = _join_sections([
        _tests_section(tests),
        _passes_table(passes),
        _notes_section(tests),
        _prior_tasks_section(tests),
        _owners_section(tests, deflake_owners),
        _record_dirs_section(passes),
    ])
    return {
        'title': title,
        'description': description,
        'details': details,
        'priority': priority,
        'planning_mode': True,
        'metadata': _lane_metadata(
            command=command,
            leg=leg,
            nodes=[test.node for test in tests],
            first_red_head=earliest.first_red_head,
            last_green_head=earliest.last_green_head,
            passes=passes,
            prior_task_ids=prior_task_ids,
        ),
    }


def _command_pass_block(counted: CountedPassEvidence) -> str:
    return '\n'.join([f'pass {counted.pass_id} head {counted.head}', *_pass_lines(counted)])


def build_lane_command_task_arguments(
    *,
    command: str,
    leg: str,
    priority: str,
    head: str,
    passes: Sequence[CountedPassEvidence],
    reason: str,
) -> dict[str, Any]:
    """The command-level fix task: the command exits non-zero and no test reproduces."""
    counted = _distinct_passes(passes)
    description = _join_sections([
        f'The offline lane saw `{command}` (leg: {leg}) exit non-zero with no reproducible '
        f'failing test since {head}.',
        f'Why no test could be confirmed: {reason}',
        f'Reproduce: {_reproduce_text(command, None)}',
    ])
    details = _join_sections([
        '\n\n'.join(_command_pass_block(c) for c in counted),
        _passes_table(counted),
        _record_dirs_section(counted),
    ])
    return {
        'title': (
            f'offline-lane[{command}]: exits non-zero with no reproducible failing test '
            f'since {_head_prefix(head)}'
        ),
        'description': description,
        'details': details,
        'priority': priority,
        'planning_mode': True,
        'metadata': _lane_metadata(
            command=command,
            leg=leg,
            nodes=[],
            first_red_head=head,
            last_green_head=None,
            passes=counted,
            prior_task_ids=[],
        ),
    }
