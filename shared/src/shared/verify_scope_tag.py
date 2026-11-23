"""The verify scope-tag rule: build a project's tag, build and parse a scope name.

Every project's verify scopes share one ``systemctl --user`` namespace, so a
scope's unit name ``df-verify-<tag>-<instance>.scope`` carries a per-project
tag ``<slug>-<hash8>``: the project root's basename made unit-name safe, and
8 hex of the sha1 of its resolved path. PRD
``plans/per-consumer-resource-recording-prd.md`` D2 keys the load recorder's
``verify`` consumer by that tag, and C2's hold report computes it from a root.

The rule is copied verbatim from
``orchestrator/src/orchestrator/verify.py::_scope_tag_for``; until leaf 4
(task 6355) switches verify.py to this module,
``orchestrator/tests/test_verify_scope_tag_cross_check.py`` pins the two
together.

Stdlib-only: a member of
``shared/tests/test_pure_stdlib_leaves.py::PURE_STDLIB_LEAVES``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    'VerifyScopeUnit',
    'parse_verify_scope_unit',
    'scope_tag_for',
    'verify_scope_unit_name',
]

_TAG = r'[a-z0-9-]+-[0-9a-f]{8}'
_INSTANCE = r'[0-9a-f]{12}'
_UNIT_NAME = re.compile(rf'^df-verify-(?P<tag>{_TAG})-(?P<instance>{_INSTANCE})\.scope$')


@dataclass(frozen=True)
class VerifyScopeUnit:
    tag: str
    instance: str


def scope_tag_for(project_root: Path) -> str:
    """``<slug>-<hash8>`` for a project root; the same root always gives the same tag."""
    resolved = Path(project_root).resolve()
    slug = re.sub(r'[^a-z0-9-]+', '-', resolved.name.lower()).strip('-')
    slug = slug[:32] or 'proj'
    digest = hashlib.sha1(str(resolved).encode()).hexdigest()[:8]
    return f'{slug}-{digest}'


def verify_scope_unit_name(tag: str, instance: str) -> str:
    """``df-verify-<tag>-<instance>.scope``; refuses a name the parser would not read."""
    if not re.fullmatch(_TAG, tag):
        raise ValueError(f'not a verify scope tag: {tag!r}')
    if not re.fullmatch(_INSTANCE, instance):
        raise ValueError(f'not a 12-hex verify scope instance: {instance!r}')
    return f'df-verify-{tag}-{instance}.scope'


def parse_verify_scope_unit(name: str) -> VerifyScopeUnit | None:
    """The tag and instance of a verify scope's unit name; ``None`` for any other name.

    The fixed-width hex tail makes the split unambiguous for slugs holding '-'.
    """
    match = _UNIT_NAME.match(name)
    if match is None:
        return None
    return VerifyScopeUnit(tag=match['tag'], instance=match['instance'])
