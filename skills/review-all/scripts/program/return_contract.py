"""How a /review-all brief states the return block: its schema, a full example, the inbox path and the check line.

Everything is read from ``state.py``'s constants, so a brief and
``state.validate_return`` cannot disagree (``plans/review-all-program-tooling-prd.md``
decision 13). Every top-level return key has one example builder; the module
refuses to import while any key of ``state.RETURN_KEYS`` lacks one.
"""
from __future__ import annotations

import json
import shlex
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import state

APPLY_RETURN = Path(__file__).resolve().with_name('apply_return.py')
PLACEHOLDER_TASK_ID = 0
"""Task ids in an example: the form check needs an integer, and no store task carries id 0."""
SEAM = '<seam id>'


@dataclass(frozen=True)
class ExampleIdentity:
    """The program facts an example names, so every field it carries passes the form checks."""

    run_id: str
    stream: str
    pass_: state.Pass
    item: str | None
    session: state.Session
    prd_path: str
    keys: tuple[str, ...]
    other_stream: str


def _sibling_path(prd_path: str, suffix: str) -> str:
    prd = PurePosixPath(prd_path)
    return str(prd.with_name(prd.stem + suffix))


_Builder = Callable[[ExampleIdentity], Any]

_EXAMPLES: Mapping[str, _Builder] = types.MappingProxyType({
    'schema': lambda i: state.RETURN_SCHEMA,
    'run_id': lambda i: i.run_id,
    'stream': lambda i: i.stream,
    'pass': lambda i: str(i.pass_),
    'session': lambda i: {'kind': str(i.session.kind), 'title': i.session.title},
    'item': lambda i: i.item,
    'prd_path': lambda i: i.prd_path,
    'manifest_path': lambda i: _sibling_path(i.prd_path, '.capability-manifest.md'),
    'sidecar_path': lambda i: _sibling_path(i.prd_path, '.capability-manifest.yaml'),
    'commits': lambda i: [],
    'open_questions': lambda i: [{
        'id': 'Q1', 'question': '<a design-level question the PRD cannot settle alone>',
        'options': ['<option A>', '<option B>'], 'recommendation': '<option A>',
        'needs_human': False, 'lead_rulable': False, 'source': str(state.QuestionSource.AUTHOR),
    }],
    'rulings_in_session': lambda i: [{
        'id': 'R1', 'question': '<the question the human ruled on in this session>',
        'ruling': '<option A>', 'options': ['<option A>', '<option B>'],
        'consequence': '<what the ruling changes>',
    }],
    'seam_dependencies': lambda i: [{'stream': i.other_stream, 'contract': SEAM}],
    'xrefs': lambda i: [{'to': i.other_stream, 'text': f'<what {i.other_stream} must know>'}],
    'lead_actions': lambda i: [{
        'op': str(state.LeadActionOp.EDGE), 'task': PLACEHOLDER_TASK_ID,
        'target': [PLACEHOLDER_TASK_ID], 'text': '<why the lead should act>', 'needs_human': False,
    }],
    'notes': lambda i: f'<one paragraph for the lead, at most {state.NOTES_CAP} characters>',
    'touches': lambda i: ['<repo-relative path the PRD changes>'],
    'filed': lambda i: [{
        'label': 'α', 'task_id': PLACEHOLDER_TASK_ID, 'priority': str(state.Priority.HIGH),
        'keys': list(i.keys[:1]), 'gate_dep': True, 'adopted': False,
    }],
    'held': lambda i: [{
        'label': 'β', 'task_id': PLACEHOLDER_TASK_ID, 'waits_on': i.other_stream,
        'contract': SEAM, 'release_owner': i.other_stream,
    }],
    'released': lambda i: [{
        'task_id': PLACEHOLDER_TASK_ID, 'onto': [PLACEHOLDER_TASK_ID],
        'contract_as_delivered': '<the seam contract as delivered>',
    }],
    'commit_planning': lambda i: {
        'committed': [PLACEHOLDER_TASK_ID], 'excluded': [], 'stamping_errors': [],
    },
    'other_filed': lambda i: [{'task_id': PLACEHOLDER_TASK_ID, 'why': '<why it is filed>'}],
    'moves': lambda i: [{
        'from': '<repo-relative path moved>', 'to': '<repo-relative path it moves to>',
        'symbols': ['<symbol moved>'],
    }],
})


def _require_every(members: Iterable[str], covered: Mapping[str, Any], what: str) -> None:
    missing = sorted(member for member in members if member not in covered)
    if missing:
        raise state.StateError(f'{what} misses {", ".join(missing)}')


_require_every(
    {key for keys in state.RETURN_KEYS.values() for key in keys.required | keys.optional},
    _EXAMPLES, '_EXAMPLES',
)


def _carried(pass_: state.Pass) -> frozenset[str]:
    keys = state.RETURN_KEYS[pass_]
    return keys.required | keys.optional


def example(identity: ExampleIdentity) -> dict[str, Any]:
    """A full return of the identity's pass: every key it may carry, each with a form-valid value."""
    carried = _carried(identity.pass_)
    return {key: build(identity) for key, build in _EXAMPLES.items() if key in carried}


def _spans(names: Iterable[str]) -> str:
    return ', '.join(f'`{name}`' for name in sorted(names)) or 'none'


def _vocabulary_lines(carried: frozenset[str]) -> list[str]:
    lines = [f'`session.kind`: {_spans(state.SessionKind)}']
    if 'open_questions' in carried:
        lines.append(f'`open_questions[].source`: {_spans(state.QuestionSource)}')
    if 'lead_actions' in carried:
        ops = ', '.join(f'`{op}` (target: {state.LEAD_ACTION_TARGETS[op]})' for op in state.LeadActionOp)
        lines.append(f'`lead_actions[].op`: {ops}')
    if 'filed' in carried:
        lines.append(f'`filed[].priority`: {_spans(state.Priority)}')
    if 'notes' in carried:
        lines.append(f'`notes`: at most {state.NOTES_CAP} characters')
    if 'commits' in carried:
        lines.append('`commits`: always `[]`; a seat or session never commits')
    return lines


def schema_lines(pass_: state.Pass) -> list[str]:
    """The pass's schema as bullets: RETURN_SCHEMA, its keys from RETURN_KEYS, and their vocabularies."""
    keys = state.RETURN_KEYS[pass_]
    lines = [
        f'`schema`: `{state.RETURN_SCHEMA}`',
        f'required keys: {_spans(keys.required)}',
        f'optional keys (an empty list may be given): {_spans(keys.optional)}',
    ]
    if keys.non_empty:
        lines.append(f'must not be empty: {_spans(keys.non_empty)}')
    return [f'- {line}' for line in lines + _vocabulary_lines(keys.required | keys.optional)]


def check_argv(home: state.Home, stream: str, pass_: state.Pass, item: str | None) -> list[str]:
    """``apply_return.py --check <inbox file>``, with ``--amend`` once returns/ holds this return.

    That is ``apply_return.py::_prior_returns``'s already-applied rule, read from the same file.
    """
    argv = [str(APPLY_RETURN), '--check', str(home.inbox_file(stream, pass_, item))]
    if home.returns_file(stream, pass_, item).exists():
        argv.append('--amend')
    return argv


def section_lines(home: state.Home, identity: ExampleIdentity) -> list[str]:
    """The body of a brief's return section for *identity*'s stream and pass."""
    inbox = home.inbox_file(identity.stream, identity.pass_, identity.item)
    argv = check_argv(home, identity.stream, identity.pass_, identity.item)
    return [
        f'Write ONE JSON object to `{inbox}`, and nowhere else.',
        '',
        *schema_lines(identity.pass_),
        '',
        'A full example; replace every `<…>` value and every task id 0:',
        '',
        '```json',
        json.dumps(example(identity), indent=2, ensure_ascii=False),
        '```',
        '',
        'Then check it, and end only once it prints `check ok`:',
        '',
        '```sh',
        shlex.join(argv),
        '```',
        '',
        'A form error: fix it in the file and re-run the check. '
        'A contradiction: stop and tell the lead.',
    ]
