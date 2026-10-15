"""The dashboard's payload budgets: the declared table, and how it is read.

A budget is the most wire bytes one REQUEST, a GET path plus its query, may
serve. The table is package data, ``payload_budgets.json`` beside this module,
and two enforcement points read it: the pre-merge fleet sweep
(``dashboard/tests/test_payload_budget_sweep.py``) and the live predicate
(``dashboard/payload_check.py``).

This module is pure. It reads the table and nothing else: no network, and no
import of the app.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlencode

TABLE_PATH = Path(__file__).with_name('payload_budgets.json')
SCHEMA_VERSION = 1
API_PREFIX = '/api/v2/dashboard/'

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
