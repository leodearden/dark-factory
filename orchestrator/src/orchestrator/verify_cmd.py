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

import shlex
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


# Shell chain-delimiter tokens. shlex.split has no concept of shell operators —
# it only tokenizes on whitespace and quotes — so `&&`/`||`/`;`/`|` survive as
# ordinary (unquoted) tokens in the split output. Their presence means *raw*
# is a multi-segment chain that parse_config_command cannot safely decompose
# into one tool invocation's base_flags/targets (see module docstring's
# RECOGNISED-BUT-UNSTRUCTURABLE discussion).
_CHAIN_OPERATOR_TOKENS = frozenset({'&&', '||', ';', '|'})

# Canonical head phrase rendered for each structured ToolKind. CARGO_TEST/
# CARGO_CLIPPY intentionally exclude this — cargo's rest-tokens are carried
# unsplit in `targets` (see _parse_single_segment) since cargo's flag grammar
# (`--exclude <value>`, a trailing `-- <passthrough>`) isn't the simple
# "flags-then-paths" shape the dash/non-dash split assumes.
_TOOL_HEAD: dict[ToolKind, str] = {
    ToolKind.PYTEST: 'pytest',
    ToolKind.RUFF: 'ruff check',
    ToolKind.PYRIGHT: 'pyright',
    ToolKind.CARGO_TEST: 'cargo test',
    ToolKind.CARGO_CLIPPY: 'cargo clippy',
    ToolKind.NPX: 'npx',
}


def parse_config_command(raw: str) -> VerifyCmd:
    """Parse a config-level command string into a VerifyCmd.

    Tokenizes *raw* with ``shlex.split`` (POSIX quoting rules). Unbalanced
    quotes or an empty command classify OPAQUE with *raw* retained verbatim.

    A command containing a shell chain operator (``&&``, ``||``, ``;``, ``|``)
    as a literal token is a multi-segment chain: recognised chains (pytest or
    cargo test/clippy — the two chain-aware mutators, serial_pytest and
    cargo_scope) retain *raw* under that ToolKind; anything else chained is
    OPAQUE. A single-segment command is decomposed into a leading ``cd <dir>
    &&``, an optional ``uv run [--project X|--directory X]`` wrapper, the
    tool head, and the remaining base_flags/targets. An unrecognised head
    (single-segment) is also OPAQUE.
    """
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return VerifyCmd(tool=ToolKind.OPAQUE, raw=raw)
    if not tokens:
        return VerifyCmd(tool=ToolKind.OPAQUE, raw=raw)
    return _parse_single_segment(raw, tokens)


def _parse_chain(raw: str, tokens: list[str]) -> VerifyCmd:
    """Classify a multi-segment chain by scanning for a chain-aware tool.

    Only pytest and cargo test/clippy are chain-aware (serial_pytest and
    cargo_scope respectively know how to rewrite *raw* in place via a
    localised regex); any other chain shape — including a recognised-looking
    head followed by an unrelated segment (e.g. a ruff-check chain with a
    second, unrelated script clause) — is OPAQUE, so it runs unscoped rather
    than risk the historical find/replace-surgery mangling.
    """
    if 'pytest' in tokens:
        return VerifyCmd(tool=ToolKind.PYTEST, raw=raw)
    for i in range(len(tokens) - 1):
        if tokens[i] != 'cargo':
            continue
        if tokens[i + 1] == 'test':
            return VerifyCmd(tool=ToolKind.CARGO_TEST, raw=raw)
        if tokens[i + 1] == 'clippy':
            return VerifyCmd(tool=ToolKind.CARGO_CLIPPY, raw=raw)
    return VerifyCmd(tool=ToolKind.OPAQUE, raw=raw)


def _parse_single_segment(raw: str, tokens: list[str]) -> VerifyCmd:
    """Decompose a single-segment (non-chained) command into a VerifyCmd."""
    idx = 0
    cwd_rel: str | None = None

    # Peel a leading `cd <dir> &&`.
    if tokens[idx : idx + 1] == ['cd'] and tokens[idx + 2 : idx + 3] == ['&&']:
        cwd_rel = tokens[idx + 1]
        idx += 3

    # Peel a `uv run [--project X | --directory X]` wrapper. uv_project is
    # tri-state: None = no uv wrapper at all; '' = bare `uv run <tool>`
    # (uv-wrapped, no explicit project — see reproject()); non-empty = an
    # explicit --project was given.
    uv_project: str | None = None
    if tokens[idx : idx + 2] == ['uv', 'run']:
        idx += 2
        uv_project = ''
        if tokens[idx : idx + 1] == ['--project'] and len(tokens) > idx + 1:
            uv_project = tokens[idx + 1]
            idx += 2
        elif tokens[idx : idx + 1] == ['--directory'] and len(tokens) > idx + 1:
            cwd_rel = tokens[idx + 1]
            idx += 2

    rest = tokens[idx:]

    # A chain operator anywhere in what's left (beyond the single leading
    # cd/uv-run peeled above) means *raw* is a multi-segment command, not one
    # tool invocation — classify by scanning the ORIGINAL, unpeeled tokens.
    # The speculative cwd_rel/uv_project computed above are discarded: a
    # raw-retained chain carries no structured fields, only `tool` + `raw`.
    if any(tok in _CHAIN_OPERATOR_TOKENS for tok in rest):
        return _parse_chain(raw, tokens)

    if not rest:
        return VerifyCmd(tool=ToolKind.OPAQUE, raw=raw)

    head = rest[0]
    wrappers: tuple[str, ...] = ()
    if head == 'pytest':
        tool = ToolKind.PYTEST
        rest = rest[1:]
    elif head == 'ruff' and rest[1:2] == ['check']:
        tool = ToolKind.RUFF
        rest = rest[2:]
    elif head == 'pyright':
        tool = ToolKind.PYRIGHT
        rest = rest[1:]
    elif head == 'cargo' and rest[1:2] == ['test']:
        tool = ToolKind.CARGO_TEST
        rest = rest[2:]
    elif head == 'cargo' and rest[1:2] == ['clippy']:
        tool = ToolKind.CARGO_CLIPPY
        rest = rest[2:]
    elif head == 'npx':
        if rest[1:2] == ['pyright']:
            tool = ToolKind.PYRIGHT
            wrappers = ('npx',)
            rest = rest[2:]
        else:
            tool = ToolKind.NPX
            rest = rest[1:]
    else:
        return VerifyCmd(tool=ToolKind.OPAQUE, raw=raw)

    if tool in (ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY):
        # Cargo's flag grammar isn't "flags then paths": `--exclude <value>`
        # is a 2-token flag, and a trailing `-- <passthrough>` must stay
        # after any inserted `-p <crate>` pair (see cargo_scope). Carrying
        # the tokens unsplit in `targets` lets cargo_scope manipulate them
        # positionally instead of guessing a flags/targets split.
        base_flags: tuple[str, ...] = ()
        targets: tuple[str, ...] = tuple(rest)
    else:
        base_flags = tuple(t for t in rest if t.startswith('-'))
        targets = tuple(t for t in rest if not t.startswith('-'))

    return VerifyCmd(
        tool=tool,
        uv_project=uv_project,
        cwd_rel=cwd_rel,
        base_flags=base_flags,
        targets=targets,
        wrappers=wrappers,
        raw=None,
    )
