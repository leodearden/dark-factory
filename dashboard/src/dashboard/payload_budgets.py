"""The dashboard's payload budgets: the declared table, and how it is read.

A budget is the most wire bytes one REQUEST, a GET path plus its query, may
serve. The table is package data, ``payload_budgets.json`` beside this module,
and two enforcement points read it: the pre-merge fleet sweep
(``dashboard/tests/test_payload_budget_sweep.py``) and the live predicate
(``dashboard/src/dashboard/payload_check.py``).

This module is pure. It reads the table and nothing else: no network, and no
import of the app.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard
from urllib.parse import urlencode

TABLE_PATH = Path(__file__).with_name('payload_budgets.json')
TABLE_REPO_PATH = 'dashboard/src/dashboard/payload_budgets.json'
"""The table's repo-relative path: the form a remedy names and ``git show`` reads."""
SCHEMA_VERSION = 1
API_PREFIX = '/api/v2/dashboard/'

MAX_SLACK_RATIO = 2.0
"""Headroom is at most 1.25x live and the fleet is calibrated to at least 0.625x live."""
SLACK_FLOOR_BYTES = 32_768
"""Twice the 16 KiB headroom floor, so a small endpoint's headroom never reads as slack."""

_LIST_DESCENT = '[]'
_COLLAPSED_KEY = '{}'
_COLLAPSED_KEYS = '{}(keys)'

_TABLE_KEYS = frozenset({'_README', 'schema_version', 'endpoints', 'exempt'})
_ENDPOINT_KEYS = frozenset({'path', 'query', 'data_keyed_maps', 'history'})
_CHANGE_KEYS = frozenset({'budget_bytes', 'task', 'reason'})
_EXEMPTION_KEYS = frozenset({'path', 'reason'})
_ABSENT = object()

RequestKey = tuple[str, tuple[tuple[str, str], ...]]
"""A request's identity: its path, and its query as sorted (name, value) pairs."""


class PayloadBudgetTableError(Exception):
    """The budget table cannot be used, and no part of it is handed back.

    An empty or partial table would compare clean against every payload, so a
    table that cannot be read IS the finding (INV-11). ``source`` names the
    file first, because it is the one thing the reader cannot derive.
    """

    def __init__(self, source: str, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(
            f'refusing the payload budget table at {source}: {reason}. An empty or '
            'partial table would compare clean against every payload, so none is '
            'returned.'
        )


@dataclass(frozen=True)
class BudgetChange:
    """One recorded budget: the bytes, the task that set it, and why."""

    budget_bytes: int
    task: str
    reason: str


@dataclass(frozen=True)
class EndpointBudget:
    """The budget for one request, with the append-only history that set it."""

    path: str
    query: Mapping[str, str]
    data_keyed_maps: tuple[tuple[str, ...], ...]
    history: tuple[BudgetChange, ...]

    @property
    def budget_bytes(self) -> int:
        return self.history[-1].budget_bytes

    @property
    def request_key(self) -> RequestKey:
        return (self.path, tuple(sorted(self.query.items())))

    def url_suffix(self) -> str:
        return f'{self.path}?{urlencode(self.query)}' if self.query else self.path


@dataclass(frozen=True)
class Exemption:
    """A templated route that carries no budget, and the reviewed reason why."""

    path: str
    reason: str


@dataclass(frozen=True)
class BudgetTable:
    """Every budgeted request in file order, plus the exempt routes."""

    endpoints: tuple[EndpointBudget, ...]
    exempt: tuple[Exemption, ...]


def load_budget_table(path: Path = TABLE_PATH) -> BudgetTable:
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise PayloadBudgetTableError(str(path), f'it cannot be read ({exc})') from exc
    return parse_budget_table(text, source=str(path))


def parse_budget_table(text: str, *, source: str = '<inline>') -> BudgetTable:
    """Parse and validate a budget table, refusing at the FIRST bad field.

    Fields are checked in a fixed order, so a refusal names the one field that
    is wrong rather than whatever a constructor happened to trip over first.
    """
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise PayloadBudgetTableError(source, f'it is not valid JSON ({exc})') from exc
    return _Reader(source).table(raw)


@dataclass(frozen=True)
class _Reader:
    source: str

    def refuse(self, field: str, problem: str) -> PayloadBudgetTableError:
        return PayloadBudgetTableError(self.source, f'{field} {problem}')

    def table(self, raw: object) -> BudgetTable:
        top = self.json_object(raw, 'the top level', _TABLE_KEYS)
        version = top.get('schema_version', _ABSENT)
        if type(version) is not int or version != SCHEMA_VERSION:
            raise self.refuse(
                'schema_version',
                f'is {_shown(version)}; this build reads only {SCHEMA_VERSION}',
            )
        endpoints = tuple(
            self.endpoint(item, f'endpoints[{i}]')
            for i, item in enumerate(self.json_list(top, 'endpoints', non_empty=True))
        )
        self.require_unique_requests(endpoints)
        exempt = tuple(
            self.exemption(item, f'exempt[{i}]')
            for i, item in enumerate(self.json_list(top, 'exempt', non_empty=False))
        )
        self.require_disjoint(endpoints, exempt)
        return BudgetTable(endpoints=endpoints, exempt=exempt)

    def json_object(self, raw: object, field: str, allowed: frozenset[str]) -> dict[str, object]:
        if not isinstance(raw, dict):
            raise self.refuse(field, f'is {_shown(raw)}, not a JSON object')
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise self.refuse(field, f'carries unknown key(s) {unknown}; allowed: {sorted(allowed)}')
        return raw

    def json_list(self, parent: dict[str, object], name: str, *, non_empty: bool) -> list[object]:
        value = parent.get(name, _ABSENT)
        if not isinstance(value, list) or (non_empty and not value):
            wanted = 'a non-empty list' if non_empty else 'a list'
            raise self.refuse(name, f'is {_shown(value)}, not {wanted}')
        return value

    def endpoint(self, raw: object, field: str) -> EndpointBudget:
        item = self.json_object(raw, field, _ENDPOINT_KEYS)
        return EndpointBudget(
            path=self.api_path(item.get('path', _ABSENT), f'{field}.path'),
            query=self.query(item.get('query', _ABSENT), f'{field}.query'),
            data_keyed_maps=self.key_paths(
                item.get('data_keyed_maps', _ABSENT), f'{field}.data_keyed_maps',
            ),
            history=self.history(item.get('history', _ABSENT), f'{field}.history'),
        )

    def exemption(self, raw: object, field: str) -> Exemption:
        item = self.json_object(raw, field, _EXEMPTION_KEYS)
        return Exemption(
            path=self.api_path(item.get('path', _ABSENT), f'{field}.path'),
            reason=self.text(item.get('reason', _ABSENT), f'{field}.reason'),
        )

    def api_path(self, value: object, field: str) -> str:
        if not isinstance(value, str) or not value.startswith(API_PREFIX):
            raise self.refuse(field, f'is {_shown(value)}, not a path under {API_PREFIX}')
        return value

    def text(self, value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise self.refuse(field, f'is {_shown(value)}, not a non-blank string')
        return value

    def query(self, value: object, field: str) -> Mapping[str, str]:
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            raise self.refuse(field, f'is {_shown(value)}, not an object of string values')
        return MappingProxyType(dict(value))

    def key_paths(self, value: object, field: str) -> tuple[tuple[str, ...], ...]:
        if not isinstance(value, list) or not all(_is_key_path(p) for p in value):
            raise self.refuse(
                field,
                f'is {_shown(value)}, not a list of key paths (each a non-empty '
                'list of non-blank strings)',
            )
        return tuple(tuple(p) for p in value)

    def history(self, value: object, field: str) -> tuple[BudgetChange, ...]:
        if not isinstance(value, list) or not value:
            raise self.refuse(field, f'is {_shown(value)}, not a non-empty list')
        return tuple(self.change(item, f'{field}[{i}]') for i, item in enumerate(value))

    def change(self, raw: object, field: str) -> BudgetChange:
        item = self.json_object(raw, field, _CHANGE_KEYS)
        budget = item.get('budget_bytes', _ABSENT)
        if type(budget) is not int or budget <= 0:
            raise self.refuse(f'{field}.budget_bytes', f'is {_shown(budget)}, not a positive integer')
        return BudgetChange(
            budget_bytes=budget,
            task=self.text(item.get('task', _ABSENT), f'{field}.task'),
            reason=self.text(item.get('reason', _ABSENT), f'{field}.reason'),
        )

    def require_unique_requests(self, endpoints: tuple[EndpointBudget, ...]) -> None:
        seen: set[RequestKey] = set()
        for i, entry in enumerate(endpoints):
            if entry.request_key in seen:
                raise self.refuse(
                    f'endpoints[{i}]',
                    f'is a duplicate request {entry.url_suffix()}; each (path, query) '
                    'may appear once',
                )
            seen.add(entry.request_key)

    def require_disjoint(
        self, endpoints: tuple[EndpointBudget, ...], exempt: tuple[Exemption, ...],
    ) -> None:
        budgeted = {entry.path for entry in endpoints}
        for i, exemption in enumerate(exempt):
            if exemption.path in budgeted:
                raise self.refuse(
                    f'exempt[{i}].path',
                    f'is {exemption.path}, which also carries a budget; a path is '
                    'budgeted or exempt, never both',
                )


def _is_key_path(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(segment, str) and segment.strip() for segment in value)
    )


def _shown(value: object) -> str:
    if value is _ABSENT:
        return 'missing'
    rendered = json.dumps(value)
    return rendered if len(rendered) <= 60 else f'{rendered[:57]}...'


@dataclass(frozen=True)
class FieldBytes:
    """The wire bytes one LEAF field carries, summed over every occurrence."""

    path: str
    bytes: int


def field_breakdown(
    payload: object, data_keyed_maps: Iterable[Sequence[str]] = (),
) -> tuple[FieldBytes, ...]:
    """Attribute *payload*'s bytes to its leaf fields, largest first.

    A leaf is a dict key whose value is a scalar, an empty container, or a list
    holding no dict. It is charged its JSON-encoded key, the colon, and its
    compact value. List descent renders as ``[]``. A dict whose key path is in
    *data_keyed_maps* has its keys collapsed to ``{}``, so a field spread
    across project labels or dates sums into one leaf. Where such a map's
    values are containers, its keys and their colons are charged to one
    ``{}(keys)`` leaf beside them, so a map grown by its keys still shows.
    Nothing is inferred: two NAMED fields that happen to share a shape stay
    apart.
    """
    totals: Counter[tuple[str, ...]] = Counter()
    keyed = frozenset(tuple(key_path) for key_path in data_keyed_maps)
    if _holds_a_dict(payload):
        _account_list(payload, (_LIST_DESCENT,), keyed, totals)
    else:
        _account_list([payload], (), keyed, totals)
    ranked = sorted(
        ((_rendered(path), size) for path, size in totals.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(FieldBytes(path=path, bytes=size) for path, size in ranked)


def nodes_at(payload: object, key_path: Sequence[str]) -> Iterator[object]:
    """Every node *key_path* reaches in *payload*: ``[]`` descends a list, ``{}`` every map value."""
    if not key_path:
        yield payload
        return
    segment, rest = key_path[0], key_path[1:]
    if segment == _LIST_DESCENT and isinstance(payload, list):
        children = payload
    elif segment == _COLLAPSED_KEY and isinstance(payload, dict):
        children = list(payload.values())
    elif isinstance(payload, dict) and segment in payload:
        children = [payload[segment]]
    else:
        children = []
    for child in children:
        yield from nodes_at(child, rest)


def render_overage(
    entry: EndpointBudget,
    measured_bytes: int,
    breakdown: tuple[FieldBytes, ...],
    top_n: int = 8,
) -> str:
    last = entry.history[-1]
    ratio = measured_bytes / entry.budget_bytes
    return '\n'.join([
        f'{entry.url_suffix()} served {measured_bytes:,} bytes against a budget of '
        f'{entry.budget_bytes:,} ({ratio:.2f}x the budget).',
        f'Largest leaf fields ({min(top_n, len(breakdown))} of {len(breakdown)}):',
        *(
            f'  {field.bytes:>12,}  {100 * field.bytes / measured_bytes:5.1f}%  {field.path}'
            for field in breakdown[:top_n]
        ),
        f'The budget was last set by task {last.task}: {last.reason}',
        'Remedy: shrink the payload, OR APPEND a history entry {budget_bytes, task, '
        f'reason}} to this request in {TABLE_REPO_PATH} naming your task and why the '
        'larger size is right. Never edit a recorded entry.',
    ])


def slack_violation(budget_bytes: int, measured_bytes: int) -> bool:
    """True when the budget sits further above the measurement than headroom explains.

    That is either a fixture that stopped reaching its data, or a real shrink
    nobody recorded; both are failures of the budget, not of the payload.
    """
    allowance = max(MAX_SLACK_RATIO * measured_bytes, measured_bytes + SLACK_FLOOR_BYTES)
    return budget_bytes > allowance


def _encoded(value: object) -> bytes:
    """*value* serialized exactly as Starlette's JSONResponse does, so leaf bytes are wire bytes."""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, indent=None, separators=(',', ':'),
    ).encode('utf-8')


def _holds_a_dict(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list) and any(isinstance(item, dict) for item in value)


def _account_list(
    items: list[object],
    path: tuple[str, ...],
    keyed: frozenset[tuple[str, ...]],
    totals: Counter[tuple[str, ...]],
) -> None:
    for item in items:
        if isinstance(item, dict):
            _account_dict(item, path, keyed, totals)
        else:
            totals[path] += len(_encoded(item))


def _account_dict(
    node: dict[str, object],
    path: tuple[str, ...],
    keyed: frozenset[tuple[str, ...]],
    totals: Counter[tuple[str, ...]],
) -> None:
    collapse = path in keyed
    for key, value in node.items():
        member = (*path, _COLLAPSED_KEY if collapse else key)
        key_bytes = len(_encoded(key)) + 1
        if isinstance(value, dict) and value:
            _account_dict(value, member, keyed, totals)
        elif _holds_a_dict(value):
            _account_list(value, (*member, _LIST_DESCENT), keyed, totals)
        else:
            totals[member] += key_bytes + len(_encoded(value))
            continue
        if collapse:
            totals[(*path, _COLLAPSED_KEYS)] += key_bytes


def _rendered(path: tuple[str, ...]) -> str:
    text = ''
    for segment in path:
        if segment == _LIST_DESCENT or not text:
            text += segment
        else:
            text += f'.{segment}'
    return text or '(the whole payload)'
