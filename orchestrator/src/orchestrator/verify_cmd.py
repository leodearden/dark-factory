"""Structured verify-command model (PRD: plans/verify-plan-prd.md task β).

Replaces verify.py's raw-shell-string find/replace-surgery command model
(``_scope_command``, ``_strip_directory_flag``, ``_strip_leading_cd``,
``_reproject_bare_uv_run``, ``_force_serial_pytest``, ``_scope_cargo_workspace``,
``_maybe_govern_merge_cmd``'s bash-wrap) with a structured, serializable
``VerifyCmd`` model:

- ``parse_config_command(raw)`` tokenizes a config-level command string once.
- ``render(cmd)`` is the single shell-string producer (the inverse of parse
  for well-formed, non-OPAQUE commands).
- A set of pure ``VerifyCmd -> VerifyCmd`` mutators (``scope_to``,
  ``strip_cwd``, ``reproject``, ``cargo_scope``, ``serial_pytest``,
  ``govern_cpu``) replace the old string-surgery helpers.

``ToolKind`` is a ``StrEnum`` — mirrors verify_categories.FailureCategory
(task α) so tool identity is JSON-serialisable and ``str(ToolKind.X) == 'x'``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class ToolKind(StrEnum):
    """The verify tools ``parse_config_command`` recognises, plus OPAQUE."""

    PYTEST = 'pytest'
    RUFF = 'ruff'
    PYRIGHT = 'pyright'
    CARGO_TEST = 'cargo_test'
    CARGO_CLIPPY = 'cargo_clippy'
    NPX = 'npx'
    OPAQUE = 'opaque'


@dataclass(frozen=True)
class VerifyCmd:
    """A structured verify command: either fully-structured or raw-retained.

    Fully-structured (``raw is None``, ``tool`` is not OPAQUE): ``base_flags``/
    ``targets`` are populated and mutators operate on them directly; ``render``
    reassembles the shell string from these fields.

    Raw-retained (``raw is not None``): either OPAQUE (genuinely unparseable
    or unrecognised — every mutator no-ops, see P1) or a RECOGNISED-BUT-
    UNSTRUCTURABLE multi-segment chain (a cargo or pytest ``&&``-chain that
    ``parse_config_command`` couldn't safely split into one tool invocation).
    For the latter, ``tool`` names the chain's dominant tool so the matching
    chain-aware mutator (``cargo_scope`` / ``serial_pytest``) can still act —
    via a localised regex rewrite of ``raw`` — while every other mutator
    no-ops. ``render`` returns ``raw`` (as mutated) unchanged otherwise.

    ``wrappers`` holds zero or more argv-prefix markers rendered by
    ``render``: the sentinel ``'npx'`` (set by ``parse_config_command`` when
    the original command was ``npx``-fronted, e.g. ``npx pyright``) is
    rendered as an innermost prefix right before the tool head; any other
    entry (set by ``govern_cpu``) is treated as a resolved cpu-governed-exec
    path and wraps the *entire* rendered command as an outermost
    ``/bin/bash -c`` payload.
    """

    tool: ToolKind
    uv_project: str | None = None
    cwd_rel: str | None = None
    base_flags: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    wrappers: tuple[str, ...] = ()
    raw: str | None = None
