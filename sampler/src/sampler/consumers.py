"""Attribute the host's cgroup use to consumers.

PRD ``plans/per-consumer-resource-recording-prd.md`` D2. A *consumer* is a
role and an optional key. Members are leaf cgroups, named by ONE ordered rule
table (``LEAF_RULES``, first match wins); every inner cgroup adds a residual,
named by ``classify_residual``. The consumer's string form in the store's
``leaf`` column is produced and parsed only by ``encode_consumer`` /
``parse_consumer``.

A cgroup path is handled as its tuple of segments below the cgroup root; the
root is ``()``. Verify scopes are recognised only through
``shared.verify_scope_tag``.

This module names no metric: ``sampler.metrics`` imports it, never the reverse.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

from shared.verify_scope_tag import parse_verify_scope_unit

__all__ = [
    'LEAF_RULES',
    'Consumer',
    'LeafRule',
    'Role',
    'classify_member',
    'classify_residual',
    'encode_consumer',
    'parse_consumer',
]

Segments = tuple[str, ...]


class Role(StrEnum):
    SERVICE = 'service'
    AGENTS = 'agents'
    VERIFY = 'verify'
    PROJECT = 'project'
    SHARED = 'shared'
    INTERACTIVE = 'interactive'
    SYSTEM = 'system'
    OTHER = 'other'
    EXITED = 'exited'
    KERNEL = 'kernel'


@dataclass(frozen=True)
class Consumer:
    """A role and its key; only the kernel is unkeyed."""

    role: Role
    key: str | None = None

    def __post_init__(self) -> None:
        if (self.key is None) != (self.role is Role.KERNEL):
            raise ValueError(f'{self.role} consumer with key {self.key!r}')
        if self.key == '':
            raise ValueError(f'{self.role} consumer with an empty key')

    def __lt__(self, other: Consumer) -> bool:
        return (self.role, self.key or '') < (other.role, other.key or '')


_KEY_SEPARATOR = '@'


def encode_consumer(consumer: Consumer) -> str:
    """``<role>@<key>``, or ``<role>`` for the unkeyed kernel."""
    if consumer.key is None:
        return consumer.role.value
    return f'{consumer.role.value}{_KEY_SEPARATOR}{consumer.key}'


def parse_consumer(text: str) -> Consumer:
    """The inverse of ``encode_consumer``; splits on the FIRST '@' (keys may hold '@')."""
    role_text, separator, key = text.partition(_KEY_SEPARATOR)
    try:
        return Consumer(Role(role_text), key if separator else None)
    except ValueError as exc:
        raise ValueError(f'not a consumer name: {text!r} ({exc})') from exc


_ROOT_ANCHOR = '/'
_FIXED_ANCHORS = frozenset(
    {'system.slice', 'user.slice', 'app.slice', 'session.slice', 'background.slice'}
)
_USER_ANCHOR = re.compile(r'user-\d+\.slice|user@\d+\.service')
_ORCHESTRATOR_SERVICE = re.compile(r'orchestrator-.+\.service')
_WATCHDOG = 'orchestrator-watchdog.service'
_DF_PROJECT_SLICE = re.compile(r'df-.+\.slice')
_REIFY_SLICE = 'reify.slice'
_AGENT_CHILD_PREFIX = 'agent-'
_FACTORY_SERVICES = frozenset(
    {
        'sccache.service',
        'fused-memory.service',
        'dark-factory-dashboard.service',
        _WATCHDOG,
        'reify-warm-lane-gc.service',
        'reify-jobserver.service',
        'pytest-jobserver.service',
        'cliproxyapi.service',
        'dark-factory-load-sampler.service',
        'app-quality\\x2dsnapshot.slice',
    }
)
_T3CODE_UNITS = frozenset({'t3code.service', 'app-t3code.slice'})
_T3CODE_SCOPE_PREFIX = 'app-com.t3tools.T3Code-'
_TERMINAL_PREFIX = 'app-org.kde.konsole-'
_DOCKER_SCOPE = re.compile(r'docker-.+\.scope')


def _is_anchor(name: str) -> bool:
    return name in _FIXED_ANCHORS or _USER_ANCHOR.fullmatch(name) is not None


def _is_orchestrator_service(name: str) -> bool:
    return name != _WATCHDOG and _ORCHESTRATOR_SERVICE.fullmatch(name) is not None


def _is_project_slice(name: str) -> bool:
    return name == _REIFY_SLICE or _DF_PROJECT_SLICE.fullmatch(name) is not None


def _nearest_anchor(segments: Segments) -> str:
    """The deepest anchor on the path, the cgroup itself included; the root is '/'."""
    return next((name for name in reversed(segments) if _is_anchor(name)), _ROOT_ANCHOR)


def _first_index(segments: Segments, predicate: Callable[[str], bool]) -> int | None:
    return next((i for i, name in enumerate(segments) if predicate(name)), None)


def _service_key(segments: Segments, service_index: int) -> str:
    """The project slice the service sits in, else the service unit itself."""
    enclosing = [name for name in segments[:service_index] if _is_project_slice(name)]
    return enclosing[-1] if enclosing else segments[service_index]


def _orchestrator_service(segments: Segments) -> Consumer | None:
    index = _first_index(segments, _is_orchestrator_service)
    if index is None:
        return None
    child = segments[index + 1] if index + 1 < len(segments) else ''
    role = Role.AGENTS if child.startswith(_AGENT_CHILD_PREFIX) else Role.SERVICE
    return Consumer(role, _service_key(segments, index))


def _verify_scope(segments: Segments) -> Consumer | None:
    for name in segments:
        unit = parse_verify_scope_unit(name)
        if unit is not None:
            return Consumer(Role.VERIFY, unit.tag)
    return None


def _project_slice(segments: Segments) -> Consumer | None:
    index = _first_index(segments, _is_project_slice)
    return None if index is None else Consumer(Role.PROJECT, segments[index])


def _factory_service(segments: Segments) -> Consumer | None:
    index = _first_index(segments, _FACTORY_SERVICES.__contains__)
    return None if index is None else Consumer(Role.SHARED, segments[index])


def _t3code(segments: Segments) -> Consumer | None:
    if any(name in _T3CODE_UNITS or name.startswith(_T3CODE_SCOPE_PREFIX) for name in segments):
        return Consumer(Role.INTERACTIVE, 't3code')
    return None


def _terminals(segments: Segments) -> Consumer | None:
    if any(name.startswith(_TERMINAL_PREFIX) for name in segments):
        return Consumer(Role.INTERACTIVE, 'terminals')
    return None


def _docker(segments: Segments) -> Consumer | None:
    if len(segments) >= 2 and segments[0] == 'system.slice' and _DOCKER_SCOPE.fullmatch(segments[1]):
        return Consumer(Role.SYSTEM, 'docker')
    return None


def _nearest_anchor_rule(segments: Segments) -> Consumer:
    return Consumer(Role.OTHER, _nearest_anchor(segments))


class LeafRule(NamedTuple):
    name: str
    match: Callable[[Segments], Consumer | None]


LEAF_RULES: tuple[LeafRule, ...] = (
    LeafRule('orchestrator_service', _orchestrator_service),
    LeafRule('verify_scope', _verify_scope),
    LeafRule('project_slice', _project_slice),
    LeafRule('factory_service', _factory_service),
    LeafRule('t3code', _t3code),
    LeafRule('terminals', _terminals),
    LeafRule('docker', _docker),
    LeafRule('nearest_anchor', _nearest_anchor_rule),
)
"""D2's leaf rules in order; the last is the fallback and always matches."""


def _first_named(segments: Segments, rules: tuple[LeafRule, ...]) -> Consumer | None:
    return next(
        (consumer for rule in rules if (consumer := rule.match(segments)) is not None), None
    )


def classify_member(segments: Segments) -> Consumer:
    """The consumer a leaf cgroup's use belongs to."""
    consumer = _first_named(segments, LEAF_RULES)
    assert consumer is not None, 'the nearest_anchor rule always matches'
    return consumer


def classify_residual(segments: Segments) -> Consumer:
    """The consumer an inner cgroup's residual belongs to (D2's residual order)."""
    if not segments:
        return Consumer(Role.KERNEL)
    own = segments[-1]
    if _is_orchestrator_service(own):
        return Consumer(Role.AGENTS, _service_key(segments, len(segments) - 1))
    if _is_project_slice(own):
        return Consumer(Role.PROJECT, own)
    named = _first_named(segments, LEAF_RULES[:-1])
    return named if named is not None else Consumer(Role.EXITED, _nearest_anchor(segments))
