"""shared.capability_manifest — the capability-manifest sidecar schema + loader.

Pydantic v2 models for the machine-readable capability-manifest sidecar
(``<prd-stem>.capability-manifest.yaml``, ``schema_version`` 1) that
accompanies a PRD's ``.capability-manifest.md`` (see
``plans/capability-delivered-checks-prd.md`` §Contract). ``commit_planning``
(fused-memory, task γ) is the canonical writer of the sidecar's ``task_id``
field and of a producer task's ``metadata.delivered_checks``; the scheduler
(orchestrator, task δ) is the canonical reader of ``metadata.delivered_checks``
at the dispatch gate.

This module also registers the ``metadata.delivered_checks`` sub-model with
``shared.task_metadata``'s W10 extension point
(:func:`shared.task_metadata.register_metadata_submodel`) — see
:data:`DeliveredCheckMeta`'s registration call at the bottom of this module.
Registration is a side-effect of importing *this* module, mirroring
``shared/src/shared/deploy_state.py`` (not done at ``shared.task_metadata``
import time, so ``shared/tests/test_task_metadata.py``'s
``TestSubmodelRegistry`` stub registrations are undisturbed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    'CapabilityManifestDoc',
    'DeliveredCheck',
    'ManifestCapability',
    'ManifestTask',
    'load_capability_manifest',
    'parse_capability_manifest',
]


def _check_kind_conditional_fields(
    *,
    kind: str,
    pattern: str | None,
    expect: str | None,
    paths: list[str],
    script: str | None,
    args: list[str],
    timeout_secs: int | None,
) -> None:
    """Shared grep/script/manual cross-field validation (PRD §Contract).

    Raises ``ValueError`` naming ``kind`` + the offending field, so the
    structured ``ValidationError`` a caller sees names the entry. Shared
    between :class:`DeliveredCheck` (sidecar; allows ``kind='manual'``) and
    :class:`DeliveredCheckMeta` (``metadata.delivered_checks`` entries; its
    own ``kind`` type excludes ``'manual'``, so that branch is unreachable
    there but kept here so both models enforce identical grep/script rules).
    """
    if kind == 'grep':
        if not pattern:
            raise ValueError(f'DeliveredCheck: pattern is required when kind={kind!r}.')
        if expect is None:
            raise ValueError(f'DeliveredCheck: expect is required when kind={kind!r}.')
        if script is not None:
            raise ValueError(f'DeliveredCheck: script must not be set when kind={kind!r}.')
        if args:
            raise ValueError(f'DeliveredCheck: args must not be set when kind={kind!r}.')
        if timeout_secs is not None:
            raise ValueError(f'DeliveredCheck: timeout_secs must not be set when kind={kind!r}.')
    elif kind == 'script':
        if not script:
            raise ValueError(f'DeliveredCheck: script is required when kind={kind!r}.')
        if timeout_secs is None or timeout_secs <= 0:
            raise ValueError(
                f'DeliveredCheck: timeout_secs is required and must be > 0 when kind={kind!r}.'
            )
        if pattern is not None:
            raise ValueError(f'DeliveredCheck: pattern must not be set when kind={kind!r}.')
        if expect is not None:
            raise ValueError(f'DeliveredCheck: expect must not be set when kind={kind!r}.')
        if paths:
            raise ValueError(f'DeliveredCheck: paths must not be set when kind={kind!r}.')
    else:  # manual
        if pattern is not None:
            raise ValueError(f'DeliveredCheck: pattern must not be set when kind={kind!r}.')
        if expect is not None:
            raise ValueError(f'DeliveredCheck: expect must not be set when kind={kind!r}.')
        if paths:
            raise ValueError(f'DeliveredCheck: paths must not be set when kind={kind!r}.')
        if script is not None:
            raise ValueError(f'DeliveredCheck: script must not be set when kind={kind!r}.')
        if args:
            raise ValueError(f'DeliveredCheck: args must not be set when kind={kind!r}.')
        if timeout_secs is not None:
            raise ValueError(f'DeliveredCheck: timeout_secs must not be set when kind={kind!r}.')


class DeliveredCheck(BaseModel):
    """A single sidecar ``delivered_check`` — the gate δ evaluates (PRD §Contract).

    ``kind`` discriminates three mutually exclusive shapes: ``'grep'``
    (``pattern`` + ``expect``, optional ``paths``), ``'script'`` (``script``
    + ``timeout_secs``, optional ``args``), and ``'manual'`` (no check
    fields — excluded from the gate; optional free-text ``reason``).
    ``extra='forbid'`` — this is a strict authoring schema whose deliverable
    is rejecting malformed/typo'd entries, unlike ``shared.task_metadata``'s
    ``extra='allow'`` round-trip sub-models.
    """

    model_config = ConfigDict(extra='forbid')

    kind: Literal['grep', 'script', 'manual']
    pattern: str | None = None
    expect: Literal['present', 'absent'] | None = None
    paths: list[str] = Field(default_factory=list)
    script: str | None = None
    args: list[str] = Field(default_factory=list)
    timeout_secs: int | None = None
    reason: str | None = None

    @model_validator(mode='after')
    def _check_fields(self) -> DeliveredCheck:
        _check_kind_conditional_fields(
            kind=self.kind,
            pattern=self.pattern,
            expect=self.expect,
            paths=self.paths,
            script=self.script,
            args=self.args,
            timeout_secs=self.timeout_secs,
        )
        return self


class ManifestCapability(BaseModel):
    """A single capability asserted for a manifest task (PRD §Contract).

    ``verdict`` is the authoring-time G3/G6 verdict; PASS is required to
    queue (enforced upstream of this schema, not here). ``delivered_check``
    is optional — omitted (or ``kind='manual'``) excludes the capability
    from the automated gate.
    """

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1)
    binding: str
    verdict: Literal['PASS', 'FAIL']
    delivered_check: DeliveredCheck | None = None


class ManifestTask(BaseModel):
    """A single manifest task block — one per PRD Greek-label task (PRD §Contract).

    ``task_id`` is ``int | None``: ``None`` at authoring time, stamped by
    ``commit_planning`` — never author-supplied for a real batch. ``title``
    is a human aid, not load-bearing.
    """

    model_config = ConfigDict(extra='forbid')

    label: str = Field(min_length=1)
    task_id: int | None = None
    title: str | None = None
    capabilities: list[ManifestCapability]


class CapabilityManifestDoc(BaseModel):
    """The full capability-manifest sidecar document (PRD §Contract).

    ``schema_version`` is pinned to ``1`` — the only version this loader
    validates today. The doc-level validator enforces label non-empty +
    uniqueness across ``tasks``: uniqueness can only be checked with the
    whole task list in view (an individual :class:`ManifestTask`'s own
    ``min_length=1`` on ``label`` already rejects an empty label on its
    own, so that branch here is a defense-in-depth restatement of the same
    rule at doc granularity).
    """

    model_config = ConfigDict(extra='forbid')

    prd: str
    schema_version: Literal[1]
    tasks: list[ManifestTask]

    @model_validator(mode='after')
    def _check_labels(self) -> CapabilityManifestDoc:
        seen: set[str] = set()
        for task in self.tasks:
            if not task.label:
                raise ValueError('CapabilityManifestDoc: task label must be non-empty.')
            if task.label in seen:
                raise ValueError(f'CapabilityManifestDoc: duplicate task label {task.label!r}.')
            seen.add(task.label)
        return self


def parse_capability_manifest(data: dict) -> CapabilityManifestDoc:
    """Validate a raw ``dict`` (already YAML/JSON-decoded) as a manifest doc.

    A thin ``model_validate`` delegator — the convenience entry point for a
    caller that already holds the decoded mapping (e.g. ``commit_planning``,
    which reads the sidecar off disk itself). Raises ``pydantic.ValidationError``
    on any malformed shape; never swallows.
    """
    return CapabilityManifestDoc.model_validate(data)


def load_capability_manifest(path: str | Path) -> CapabilityManifestDoc:
    """Load and validate a capability-manifest sidecar YAML file.

    Errors propagate uncaught — a missing file raises ``FileNotFoundError``,
    malformed YAML syntax raises ``yaml.YAMLError``, and a malformed document
    (per PRD §Contract) raises ``pydantic.ValidationError`` naming the
    offending entry. No silent swallow: callers (the commit_planning
    stamper, this PRD's own CI fixture test) must observe every failure.
    """
    with open(path, encoding='utf-8') as fh:
        data = yaml.safe_load(fh)
    return parse_capability_manifest(data)
