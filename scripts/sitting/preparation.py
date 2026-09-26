"""The prepared judgement per open item: options with their ramifications, a recommendation or an explicit no-lean, and the agent's gate facts.

Why the store exists: a sitting is read-only until Leo answers, so a
recommendation cannot live on the escalation before apply (``x_prepared`` is
stamped only AT apply). It must still survive from the nightly Fable run to the
interactive sitting, and this file is that bridge. It is the preparer's own
state under ``data/sitting/``, never a store of record.

Every invariant is enforced at construction, so the nightly run and an
interactive session meet the same checks whichever path they write through.
"""
from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from shared.safe_io import atomic_write_text
from sitting.gates import UNKNOWN, Fact
from sitting.inventory import ESC_ID_RE, ItemKey, key_str, parse_key, parse_stamp

DEFAULT_PREPARATION_PATH = Path('data', 'sitting', 'preparation.json')
STANDING_KINDS = frozenset({'pin', 'hold', 'leo_owned', 'owned'})
GATE_FACT_NAMES: tuple[str, ...] = ('ruling', 'names_this_record', 'executed', 'session_terminated')

_QUOTED_RE = re.compile(r'"[^"]*"|`[^`]*`|“[^”]*”')
_SENTENCE_END_RE = re.compile(r'[.?](?=\s|$)')
_TASK_ID_RE = re.compile(r'\d+(?:\.\d+)*')


class PreparationStoreCorrupt(Exception):
    """The store at :attr:`path` exists but cannot be read; raised rather than read as empty."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f'{path}: preparation store is unreadable ({reason}); fix or move it aside before recording')


@dataclass(frozen=True)
class Option:
    label: str
    text: str
    ramification: str

    def __post_init__(self) -> None:
        _require_text('option label', self.label)
        _require_text(f'option {self.label} text', self.text)
        _require_text(f'option {self.label} ramification', self.ramification)


@dataclass(frozen=True)
class Recommended:
    option_label: str
    evidence_chain: str

    def __post_init__(self) -> None:
        _require_text('recommended option label', self.option_label)
        _require_text('recommendation evidence chain', self.evidence_chain)


@dataclass(frozen=True)
class NoLean:
    reason: str

    def __post_init__(self) -> None:
        _require_text('no-lean reason', self.reason)


@dataclass(frozen=True)
class TaskStatusIs:
    task_id: str
    statuses: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text('release task id', self.task_id)
        if not isinstance(self.statuses, tuple) or not self.statuses:
            raise ValueError('a task-status release names at least one status')
        for status in self.statuses:
            _require_text('release status', status)


@dataclass(frozen=True)
class EscalationClosed:
    esc_id: str

    def __post_init__(self) -> None:
        _require_escalation_id(self.esc_id)


@dataclass(frozen=True)
class Manual:
    text: str

    def __post_init__(self) -> None:
        _require_text('manual release', self.text)


ReleasePredicate = TaskStatusIs | EscalationClosed | Manual


@dataclass(frozen=True)
class Standing:
    """Why an item is not put to Leo now, and what would release it."""

    kind: str
    owner: str
    release_predicate: ReleasePredicate
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in STANDING_KINDS:
            raise ValueError(f'standing kind {self.kind!r} is not one of {sorted(STANDING_KINDS)}')
        _require_text('standing owner', self.owner)
        if not isinstance(self.release_predicate, TaskStatusIs | EscalationClosed | Manual):
            raise ValueError(f'release predicate must be TaskStatusIs, EscalationClosed or Manual, '
                             f'not {self.release_predicate!r}')
        _require_text('standing evidence', self.evidence)


@dataclass(frozen=True)
class Cites:
    escalation_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for esc_id in self.escalation_ids:
            _require_escalation_id(esc_id)
        for task_id in self.task_ids:
            if not isinstance(task_id, str) or not _TASK_ID_RE.fullmatch(task_id):
                raise ValueError(f'a cited task id is a bare id, not {task_id!r}')


@dataclass(frozen=True)
class GateFacts:
    """The agent-supplied facts for carve-out gates 1-4 and ``pins_recovery``; all default to unknown."""

    ruling: Fact = UNKNOWN
    names_this_record: Fact = UNKNOWN
    executed: Fact = UNKNOWN
    session_terminated: Fact = UNKNOWN
    pins_recovery: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in GATE_FACT_NAMES:
            if not isinstance(getattr(self, name), Fact):
                raise ValueError(f'gate fact {name} must be a gates.Fact')
        if self.pins_recovery is not None and not all(isinstance(pin, str) for pin in self.pins_recovery):
            raise ValueError('pins_recovery is a tuple of strings, or None when unknown')


@dataclass(frozen=True)
class Preparation:
    item_key: ItemKey
    question: str
    options: tuple[Option, ...]
    recommendation: Recommended | NoLean
    on_apply: str
    prepared_at: str
    prepared_by: str
    cites: Cites = field(default_factory=Cites)
    gate_facts: GateFacts = field(default_factory=GateFacts)
    standing: Standing | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item_key, tuple) or parse_key(key_str(self.item_key)) != self.item_key:
            raise ValueError(f'item_key {self.item_key!r} is not an open-item key')
        _require_single_sentence(self.question)
        labels = [option.label for option in self.options]
        if not all(isinstance(option, Option) for option in self.options) or len(set(labels)) != len(labels):
            raise ValueError(f'options must be Options with distinct labels, got {labels}')
        if not labels and self.standing is None:
            raise ValueError('a question put to Leo offers at least one option')
        if not isinstance(self.recommendation, Recommended | NoLean):
            raise ValueError('recommendation must be Recommended or NoLean')
        if isinstance(self.recommendation, Recommended) and self.recommendation.option_label not in labels:
            raise ValueError(f'recommended option {self.recommendation.option_label!r} is not one of {labels}')
        if not isinstance(self.on_apply, str) or (not self.on_apply.strip() and self.standing is None):
            raise ValueError('on_apply states what changes on apply')
        if parse_stamp(self.prepared_at) is None:
            raise ValueError(f'prepared_at {self.prepared_at!r} is not an ISO-8601 timestamp')
        _require_text('prepared_by', self.prepared_by)
        if not isinstance(self.cites, Cites) or not isinstance(self.gate_facts, GateFacts):
            raise ValueError('cites and gate_facts must be Cites and GateFacts')
        if self.standing is not None and not isinstance(self.standing, Standing):
            raise ValueError('standing must be a Standing or None')


@dataclass(frozen=True)
class PreparationStore:
    entries: Mapping[str, Preparation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, prep in self.entries.items():
            if key != key_str(prep.item_key):
                raise ValueError(f'store key {key!r} does not match its preparation {prep.item_key!r}')
        object.__setattr__(self, 'entries', MappingProxyType(dict(self.entries)))

    def get(self, item_key: ItemKey) -> Preparation | None:
        return self.entries.get(key_str(item_key))


def load(path: Path | str) -> PreparationStore:
    """The store at *path*; empty when absent, :class:`PreparationStoreCorrupt` when unreadable."""
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return PreparationStore()
    except (OSError, ValueError) as exc:
        raise PreparationStoreCorrupt(path, str(exc)) from exc
    try:
        data = _fields(json.loads(text), 'preparation store', {'preparations'})
        preparations = [from_json_payload(payload) for payload in _list(data['preparations'], 'preparations')]
    except (TypeError, ValueError) as exc:
        raise PreparationStoreCorrupt(path, str(exc)) from exc
    return PreparationStore({key_str(prep.item_key): prep for prep in preparations})


def record(path: Path | str, payloads: Iterable[Mapping[str, Any]]) -> PreparationStore:
    """Validate every payload, then merge them in (newer ``prepared_at`` wins per key) in one locked atomic write."""
    incoming = [from_json_payload(payload) for payload in payloads]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        merged = _merge(load(path), incoming)
        atomic_write_text(path, _encode_store(merged))
    return merged


def prune(store: PreparationStore, live_keys: Iterable[ItemKey]) -> tuple[PreparationStore, tuple[ItemKey, ...]]:
    """The store without items that are no longer open, and the keys it dropped."""
    live = {key_str(key) for key in live_keys}
    dropped = tuple(prep.item_key for key, prep in sorted(store.entries.items()) if key not in live)
    return PreparationStore({key: prep for key, prep in store.entries.items() if key in live}), dropped


def to_json_payload(prep: Preparation) -> dict[str, Any]:
    return {
        'item': list(prep.item_key),
        'question': prep.question,
        'options': [{'label': o.label, 'text': o.text, 'ramification': o.ramification} for o in prep.options],
        'recommendation': (
            {'no_lean': prep.recommendation.reason} if isinstance(prep.recommendation, NoLean)
            else {'option': prep.recommendation.option_label, 'evidence_chain': prep.recommendation.evidence_chain}
        ),
        'on_apply': prep.on_apply,
        'cites': {'escalations': list(prep.cites.escalation_ids), 'tasks': list(prep.cites.task_ids)},
        'gate_facts': {
            **{name: _fact_payload(getattr(prep.gate_facts, name)) for name in GATE_FACT_NAMES},
            'pins_recovery': None if prep.gate_facts.pins_recovery is None else list(prep.gate_facts.pins_recovery),
        },
        'standing': None if prep.standing is None else standing_to_json(prep.standing),
        'prepared_at': prep.prepared_at,
        'prepared_by': prep.prepared_by,
    }


def from_json_payload(obj: object) -> Preparation:
    """Decode the shape :func:`to_json_payload` emits, strictly: an unknown or missing key is refused by name."""
    data = _fields(
        obj, 'preparation',
        {'item', 'question', 'options', 'recommendation', 'on_apply', 'prepared_at', 'prepared_by'},
        {'cites', 'gate_facts', 'standing'},
    )
    return Preparation(
        item_key=parse_key(json.dumps(_list(data['item'], 'item'))),
        question=data['question'],
        options=tuple(_option(option) for option in _list(data['options'], 'options')),
        recommendation=_recommendation(data['recommendation']),
        on_apply=data['on_apply'],
        prepared_at=data['prepared_at'],
        prepared_by=data['prepared_by'],
        cites=_cites(data.get('cites')),
        gate_facts=_gate_facts(data.get('gate_facts')),
        standing=None if data.get('standing') is None else standing_from_json(data['standing']),
    )


def standing_to_json(standing: Standing) -> dict[str, Any]:
    release = standing.release_predicate
    if isinstance(release, TaskStatusIs):
        release_payload: dict[str, Any] = {
            'task_status_is': {'task_id': release.task_id, 'statuses': list(release.statuses)},
        }
    elif isinstance(release, EscalationClosed):
        release_payload = {'escalation_closed': release.esc_id}
    else:
        release_payload = {'manual': release.text}
    return {'kind': standing.kind, 'owner': standing.owner, 'release': release_payload, 'evidence': standing.evidence}


def standing_from_json(obj: object) -> Standing:
    data = _fields(obj, 'standing', {'kind', 'owner', 'release', 'evidence'})
    return Standing(data['kind'], data['owner'], _release(data['release']), data['evidence'])


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be non-empty text')


def _require_escalation_id(value: object) -> None:
    if not isinstance(value, str) or not ESC_ID_RE.fullmatch(value):
        raise ValueError(f'{value!r} is not an escalation id')


def _require_single_sentence(question: object) -> None:
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must be non-empty text')
    if len(_SENTENCE_END_RE.findall(_QUOTED_RE.sub('', question))) > 1:
        raise ValueError(f'the question is one sentence: {question!r}')


def _fields(obj: object, where: str, required: set[str], optional: frozenset[str] | set[str] = frozenset()) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f'{where} must be a JSON object')
    unknown = sorted(set(obj) - required - set(optional))
    if unknown:
        raise ValueError(f'{where}: unknown keys {unknown}')
    missing = sorted(required - set(obj))
    if missing:
        raise ValueError(f'{where}: missing keys {missing}')
    return obj


def _list(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f'{where} must be a JSON array')
    return value


def _option(obj: object) -> Option:
    data = _fields(obj, 'option', {'label', 'text', 'ramification'})
    return Option(data['label'], data['text'], data['ramification'])


def _recommendation(obj: object) -> Recommended | NoLean:
    if isinstance(obj, dict) and 'no_lean' in obj:
        return NoLean(_fields(obj, 'recommendation', {'no_lean'})['no_lean'])
    data = _fields(obj, 'recommendation', {'option', 'evidence_chain'})
    return Recommended(data['option'], data['evidence_chain'])


def _cites(obj: object) -> Cites:
    if obj is None:
        return Cites()
    data = _fields(obj, 'cites', set(), {'escalations', 'tasks'})
    return Cites(
        escalation_ids=tuple(_list(data.get('escalations', []), 'cites.escalations')),
        task_ids=tuple(_list(data.get('tasks', []), 'cites.tasks')),
    )


def _fact(obj: object, where: str) -> Fact:
    data = _fields(obj, where, {'held', 'evidence'}, {'source_kind'})
    held, evidence, source_kind = data['held'], data['evidence'], data.get('source_kind', '')
    if held is not None and not isinstance(held, bool):
        raise ValueError(f'{where}.held is true, false or null')
    if not isinstance(evidence, str) or not isinstance(source_kind, str):
        raise ValueError(f'{where}: evidence and source_kind are strings')
    return Fact(held=held, evidence=evidence, source_kind=source_kind)


def _gate_facts(obj: object) -> GateFacts:
    if obj is None:
        return GateFacts()
    data = _fields(obj, 'gate_facts', set(), {*GATE_FACT_NAMES, 'pins_recovery'})
    facts = {name: _fact(data[name], f'gate_facts.{name}') for name in GATE_FACT_NAMES if name in data}
    pins = data.get('pins_recovery')
    return GateFacts(**facts, pins_recovery=None if pins is None else tuple(_list(pins, 'pins_recovery')))


def _release(obj: object) -> ReleasePredicate:
    if not isinstance(obj, dict) or len(obj) != 1:
        raise ValueError('release is an object with exactly one of task_status_is, escalation_closed, manual')
    ((kind, value),) = obj.items()
    if kind == 'task_status_is':
        data = _fields(value, 'release.task_status_is', {'task_id', 'statuses'})
        return TaskStatusIs(data['task_id'], tuple(_list(data['statuses'], 'release.task_status_is.statuses')))
    if kind == 'escalation_closed':
        return EscalationClosed(value)
    if kind == 'manual':
        return Manual(value)
    raise ValueError(f'unknown release predicate {kind!r}')


def _fact_payload(fact: Fact) -> dict[str, Any]:
    return {'held': fact.held, 'evidence': fact.evidence, 'source_kind': fact.source_kind}


def _prepared_at(prep: Preparation) -> datetime:
    stamp = parse_stamp(prep.prepared_at)
    if stamp is None:
        raise ValueError(f'prepared_at {prep.prepared_at!r} is not an ISO-8601 timestamp')
    return stamp


def _merge(store: PreparationStore, incoming: Iterable[Preparation]) -> PreparationStore:
    entries = dict(store.entries)
    for prep in incoming:
        key = key_str(prep.item_key)
        current = entries.get(key)
        if current is None or _prepared_at(prep) >= _prepared_at(current):
            entries[key] = prep
    return PreparationStore(entries)


def _encode_store(store: PreparationStore) -> str:
    payloads = [to_json_payload(prep) for _, prep in sorted(store.entries.items())]
    return json.dumps({'preparations': payloads}, indent=2, ensure_ascii=False) + '\n'


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    with open(path.with_name(path.name + '.lock'), 'a', encoding='utf-8') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
