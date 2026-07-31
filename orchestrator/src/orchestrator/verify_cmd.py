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

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
    or unrecognised — every mutator except ``govern_cpu`` no-ops, see P1) or
    a RECOGNISED-BUT-UNSTRUCTURABLE multi-segment chain (a cargo or pytest
    ``&&``-chain that ``parse_config_command`` couldn't safely split into
    one tool invocation).
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

# Genuinely value-taking pytest flags that consume a SEPARATE following
# token (as opposed to a boolean flag, or a `--flag=value` single token).
# Used by _split_pytest_args to bind a value flag to its value as an
# adjacent pair inside base_flags at parse time, so a later base_flags
# append (apply_pytest_numprocesses, serial_pytest, with_junitxml) can never
# be inserted between the flag and its value (task 2727). This set must
# stay CLOSED to only value-taking flags — listing a boolean flag (e.g.
# -x/-s/-v/-q/-l) here would make the walk swallow the following target
# token, a silent, worse failure than the stranded-value bug this fixes.
_PYTEST_VALUE_FLAGS = frozenset({
    '-k', '-m', '-p', '-o', '-c', '-n', '-W',
    '--maxfail', '--tb', '--rootdir', '--override-ini',
    '--deselect', '--ignore', '--ignore-glob',
})

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


def _split_pytest_args(rest: list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split pytest's remaining tokens into ``(base_flags, targets)``.

    Walks *rest* left-to-right: a token that is a known separate-token value
    flag (``_PYTEST_VALUE_FLAGS``) is emitted into ``base_flags`` together
    with its following token (the value), and the walk advances past both —
    binding the pair contiguously so a later ``base_flags`` append can never
    land between a flag and its value. Any other dash-prefixed token is
    classified as a bare flag; everything else is a target. A listed value
    flag with no following token (malformed/truncated input) falls back to
    the bare-flag classification rather than indexing past the end.
    """
    base_flags: list[str] = []
    targets: list[str] = []
    i = 0
    n = len(rest)
    while i < n:
        tok = rest[i]
        if tok in _PYTEST_VALUE_FLAGS and i + 1 < n:
            base_flags.append(tok)
            base_flags.append(rest[i + 1])
            i += 2
        elif tok.startswith('-'):
            base_flags.append(tok)
            i += 1
        else:
            targets.append(tok)
            i += 1
    return tuple(base_flags), tuple(targets)


# Unquoted tokens that mean *raw* is doing shell control flow beyond a plain
# left-to-right `&&` chain. `split_chain_tail` refuses to carry a tail across
# any of them: a `||` alternative, a `;` sequence, a `|` pipe or a `( ... )`
# subshell all make "everything after segment 0" something other than "further
# independent commands that would have run anyway".
_NON_AND_CHAIN_TOKENS = frozenset({'||', ';', '|', '(', ')'})


def _has_shell_grouping_or_substitution(raw: str) -> bool:
    """True if *raw* contains an ACTIVE grouping or command-substitution construct.

    Character-level companion to the ``_NON_AND_CHAIN_TOKENS`` check, which is
    token-EQUALITY based and therefore only sees a paren that ``shlex`` happens
    to isolate as its own whitespace-separated token. An unspaced subshell
    (``(ruff check src/ && echo x)``) tokenizes as ``'(ruff'`` / ``'x)'`` and a
    command substitution (``$(git ls-files && echo x)``, ``` `...` ```)
    tokenizes as one opaque token — neither is caught by equality, yet both
    contain an `&&` that ``split_top_level_and`` (which tracks quote state
    only) will happily treat as a top-level split point. Carrying a tail out of
    one truncates the head mid-construct and emits an unbalanced shell string,
    i.e. a spurious RED verify rather than a missed check.

    Scans quote-aware, mirroring ``split_top_level_and``'s state machine:

    * outside quotes — any ``(``, ``)`` or backtick is active;
    * inside double quotes — ``$(`` and backtick are still substitutions and
      are rejected, but a bare paren is literal there (``-k "test_a(1)"``) and
      is allowed;
    * inside single quotes — nothing is active.

    Deliberately conservative: a false positive only sends ``split_chain_tail``
    down its REJECT path, which returns the untouched original and restores the
    exact pre-gate behaviour. A false negative corrupts a command.
    """
    i = 0
    n = len(raw)
    quote: str | None = None
    while i < n:
        ch = raw[i]
        if quote is None:
            if ch == '\\':
                i += 2
                continue
            if ch in ('"', "'"):
                quote = ch
                i += 1
                continue
            if ch in ('(', ')', '`'):
                return True
            i += 1
            continue
        if quote == '"':
            if ch == '\\':
                i += 2
                continue
            if ch == '`':
                return True
            if ch == '$' and raw[i + 1 : i + 2] == '(':
                return True
        if ch == quote:
            quote = None
        i += 1
    return False


def split_top_level_and(raw: str) -> list[str]:
    """Split *raw* on `&&` at shell quote depth 0, returning segments VERBATIM.

    A character-scan state machine tracking single-quote / double-quote state
    and backslash escapes (POSIX rules, matching ``shlex.split``: a backslash
    escapes outside quotes and inside double quotes, but is literal inside
    single quotes). A `&&` inside quotes is an argument value — pytest's
    ``-k 'a && b'`` is the real case — not a chain operator, so it is not a
    split point.

    Segments keep every byte between separators, boundary whitespace
    included, so ``'&&'.join(split_top_level_and(raw)) == raw`` exactly. That
    losslessness is the point: ``split_chain_tail``'s caller re-emits the tail
    verbatim rather than re-rendering it, and can only do so safely if the
    decomposition consumed nothing but the separators themselves.
    """
    segments: list[str] = []
    start = 0
    i = 0
    quote: str | None = None
    n = len(raw)
    while i < n:
        ch = raw[i]
        if quote is None:
            if ch == '\\':
                i += 2
                continue
            if ch in ('"', "'"):
                quote = ch
                i += 1
                continue
            if ch == '&' and raw[i + 1 : i + 2] == '&':
                segments.append(raw[start:i])
                i += 2
                start = i
                continue
            i += 1
            continue
        # Inside quotes: only a double-quote context honours backslash escapes.
        if quote == '"' and ch == '\\':
            i += 2
            continue
        if ch == quote:
            quote = None
        i += 1
    segments.append(raw[start:])
    return segments


# Keywords whose slot may carry a preserved `&&` tail. An ALLOWLIST, not a
# denylist, and that direction is the point (task 3218).
#
# `'pytest'` is deliberately ABSENT. A preserved tail makes the caller's
# result RECOGNISED-BUT-UNSTRUCTURABLE (``VerifyCmd.raw is not None``), and
# both `with_junitxml` and `with_pytest_timeout` are documented no-ops on
# that shape — so the tail would be bought at the price of SILENTLY dropping
# the `--junitxml` report that drives `_extract_failing_test_ids_from_junit`,
# flake confirmation and the per-test timeout floor. An unscoped sibling
# checker is not worth that trade in the test slot; in the lint/type slots
# there is nothing to lose, since neither mutator applies there.
#
# The DEFAULT for an unlisted keyword is therefore NO preservation — exactly
# the pre-task-3061 truncate-at-keyword behaviour. A verify slot added later
# cannot silently acquire this degradation by existing: it has to opt in
# here, explicitly, which is the fail-safe direction.
#
# `'uv run'` is listed for `verify._reproject_str`, whose tail preservation
# is load-bearing rather than merely nice: without it a chained lint command
# re-parses OPAQUE and the `--project` injection is silently dropped, which
# the depless workspace-root project turns into exit 127 (task 2036).
_TAIL_PRESERVING_KEYWORDS = frozenset({'ruff check', 'pyright', 'uv run'})


def _segment_invokes_tool(segment: str, keyword: str) -> bool:
    """True if *segment* actually INVOKES *keyword*'s tool at an argv-head position.

    ``split_chain_tail``'s later-segment check (task 3218 part 2). Replaces a
    plain ``keyword in segment`` substring test, which could not tell a real
    invocation from the tool's name merely OCCURRING in the segment — as it
    does inside a sibling checker's script path
    (``python3 scripts/check_pyright_config.py``) or a quoted flag value
    (``--tool "ruff check"``).

    An argv-head position is index 0, or the index just past a recognised
    wrapper prefix:

    * ``uv run`` followed by any run of ``--project X`` / ``--directory X``
      pairs, in either order, both optional — mirroring
      ``_parse_single_segment``'s peel loop, so the gate's notion of "where a
      tool head can begin" is the same as the parser's;
    * ``npx``;
    * ``python`` / ``python3`` followed by ``-m``.

    Index 0 is a head position BEFORE any wrapper is peeled, which is what
    keeps the ``'uv run'`` keyword (``verify._reproject_str``'s) matching
    segment 0 of a ``uv run ... ruff check ...`` command.

    ``shlex.split`` raising ``ValueError`` returns True: an undecodable
    segment counts as a MATCH, so the gate rejects and the pre-3218
    disposition is restored. Conservative by construction.

    **Why tightening this is safe — the two error directions are not
    symmetric.** The old substring test OVER-rejects: a legitimate sibling
    checker is dropped, so a real check never runs, which is a possible false
    GREEN — the bug class the tail-preservation gate exists to close.
    Argv-head matching can only UNDER-reject, and only for a same-tool
    fan-out behind a wrapper this module does not recognise (``poetry run
    ruff check b/``); the consequence there is that clause running UNSCOPED,
    which is a SUPERSET of the checks that would otherwise have run and can
    never produce a false GREEN. It also cannot misresolve relative paths,
    because ``split_chain_tail``'s condition 4 already rejects any chain
    containing a ``cd`` token — the property that makes an unscoped tail safe
    in the first place. Under-rejection is the strictly safer failure
    direction, which is what licenses the precise test over the blunt one.
    """
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return True

    head_positions = {0}
    idx = 0
    if tokens[idx : idx + 2] == ['uv', 'run']:
        idx += 2
        while True:
            if tokens[idx : idx + 1] in (['--project'], ['--directory']) and len(tokens) > idx + 1:
                idx += 2
            else:
                break
        head_positions.add(idx)
    elif tokens[idx : idx + 1] == ['npx']:
        head_positions.add(idx + 1)
    elif tokens[idx : idx + 1] in (['python'], ['python3']) and tokens[idx + 1 : idx + 2] == ['-m']:
        head_positions.add(idx + 2)

    kw_tokens = keyword.split()
    return any(tokens[i : i + len(kw_tokens)] == kw_tokens for i in head_positions)


def split_chain_tail(raw: str, keyword: str) -> tuple[str, str]:
    """Split *raw* into a *keyword*-bearing head and a preservable trailing chain.

    Returns ``(segments[0], tail)`` when the gate below ACCEPTS — ``tail`` is
    every byte after segment 0, so it carries its own leading `&&` and
    ``head + tail == raw`` exactly. Returns ``(raw, '')`` on every REJECT:
    deliberately the WHOLE original string, so a caller's existing
    truncate-at-*keyword* algorithm then runs on an untouched input and its
    output stays byte-identical to the pre-gate behaviour BY CONSTRUCTION.
    (Rejecting to ``(segments[0], '')`` would silently truncate — precisely
    the class of bug this helper exists to fix.)

    **The rule for a preserved tail: it RUNS UNSCOPED AND VERBATIM.** It is
    never re-parsed, re-rendered, or narrowed to the caller's file list. This
    repo's real trailing clauses are the reason: they are bare
    ``python3 fused-memory/scripts/check_*.py <dir>`` invocations that take a
    whole DIRECTORY (``fused-memory/tests``) and are single-pass AST/text
    scans asserting a whole-directory invariant. Narrowing them to the
    touched files would be WRONG, not merely wasteful — the invariant is over
    the directory, not over a diff — and running them unscoped costs
    essentially nothing. They also have no structured ``VerifyCmd`` form
    (they parse OPAQUE), so re-rendering is not even expressible.

    The gate, cheapest condition first — ACCEPT requires ALL of:

    1. ``shlex.split(raw)`` succeeds (an unbalanced quote means the string is
       not safely decomposable at all);
    2. no token in ``_NON_AND_CHAIN_TOKENS`` — the chain is plain `&&`, with
       no ``||`` / ``;`` / ``|`` / ``(`` / ``)`` control flow;
    3. ``not _has_shell_grouping_or_substitution(raw)`` — the character-level
       companion to (2). Condition 2 is token-EQUALITY based, so it only sees
       a paren ``shlex`` isolated as its own whitespace-separated token; an
       unspaced subshell ``(ruff check src/ && echo x)`` or a substitution
       ``$(git ls-files && echo x)`` slips past it while still hiding an `&&`
       that ``split_top_level_and`` would split on, truncating the head
       mid-construct into an unbalanced — instantly RED — shell string;
    4. no ``cd`` token anywhere;
    5. ``len(split_top_level_and(raw)) - 1 == tokens.count('&&')`` — the
       quote-aware splitter and ``shlex``'s tokenizer must agree on how many
       `&&` operators there are. On disagreement the gate bails rather than
       risk corrupting a quoted `&&`. (Note this one does NOT catch a nested
       substitution: both sides count the nested `&&` alike, so condition 3
       is the only thing standing between that input and a mangled command.);
    6. at least two segments (nothing to preserve otherwise);
    7. *keyword* occurs in ``segments[0]`` and in NO later segment.

    Conditions 4 and 7 are what distinguish a SIBLING-CHECKER chain (a
    different tool, no cwd sequencing — safe and desirable to preserve) from
    a SAME-TOOL FAN-OUT (which must keep being truncated), and both are load-
    bearing against real configs:

    * ``dark-factory-orchestrator.yaml:51`` is ``cd fused-memory && npx
      pyright && cd ../orchestrator && npx pyright && cd ../dashboard && npx
      pyright``. Preserving that tail would (a) run pyright fully UNSCOPED
      over two more subprojects, defeating scoping entirely, and (b) break
      correctness — the caller applies ``strip_cwd``, which removes the
      leading ``cd fused-memory``, so a surviving ``cd ../orchestrator``
      would resolve relative to the worktree root and escape the repo. The
      ``cd``-token rejection (4) stops it; the duplicate-``pyright``
      rejection (7) independently stops it too.
    * ``dark-factory-orchestrator.yaml:41`` is an 8-segment ``cd X && uv run
      pytest`` fan-out with a ``( ... )`` subshell — rejected by (2), (3),
      (4) and (7) alike.

    A ``cd`` token anywhere is disqualifying rather than only in the tail:
    once any segment shifts the shell's cwd, every later segment's relative
    paths depend on that sequencing, so a tail lifted out of it cannot be
    replayed after ``strip_cwd`` has flattened the head.

    CAVEAT for callers: condition 4 sees only the INPUT SPELLING. A uv
    ``--directory X`` head is not a ``cd`` token here, but ``render()``
    re-emits it as a leading ``cd X &&``. A caller that re-renders a parsed
    head WITHOUT ``strip_cwd`` must therefore additionally refuse to carry a
    tail when ``parsed.cwd_rel is not None`` — see ``verify._reproject_str``,
    the only such caller. The two scopers apply ``strip_cwd``, so their
    ``cwd_rel`` is always ``None`` by render time.
    """
    if keyword not in _TAIL_PRESERVING_KEYWORDS:
        return raw, ''
    try:
        tokens = shlex.split(raw)
    except ValueError:
        return raw, ''
    if any(tok in _NON_AND_CHAIN_TOKENS for tok in tokens):
        return raw, ''
    if _has_shell_grouping_or_substitution(raw):
        return raw, ''
    if 'cd' in tokens:
        return raw, ''
    segments = split_top_level_and(raw)
    if len(segments) - 1 != tokens.count('&&'):
        return raw, ''
    if len(segments) < 2:
        return raw, ''
    if keyword not in segments[0]:
        return raw, ''
    if any(_segment_invokes_tool(segment, keyword) for segment in segments[1:]):
        return raw, ''
    return segments[0], raw[len(segments[0]) :]


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

    # Peel a `uv run [--project X] [--directory X]` wrapper — both flags may
    # appear together (in either order; real per-subproject commands carry
    # both: `uv run --project orchestrator --directory orchestrator pyright
    # ...`), so this loops rather than peeling at most one. uv_project is
    # tri-state: None = no uv wrapper at all; '' = bare `uv run <tool>`
    # (uv-wrapped, no explicit project — see reproject()); non-empty = an
    # explicit --project was given.
    uv_project: str | None = None
    if tokens[idx : idx + 2] == ['uv', 'run']:
        idx += 2
        uv_project = ''
        while True:
            if tokens[idx : idx + 1] == ['--project'] and len(tokens) > idx + 1:
                uv_project = tokens[idx + 1]
                idx += 2
            elif tokens[idx : idx + 1] == ['--directory'] and len(tokens) > idx + 1:
                cwd_rel = tokens[idx + 1]
                idx += 2
            else:
                break

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
    elif tool is ToolKind.PYTEST:
        # Pytest is the only tool whose base_flags-appending mutators
        # (apply_pytest_numprocesses, serial_pytest, with_junitxml) run
        # after parse, so it's the only tool where a naive dash-prefix
        # split's stranded value could later be split from its flag by an
        # inserted append. RUFF/PYRIGHT/NPX keep the naive split below —
        # their coincidental round-trip holds since nothing is ever
        # inserted between their base_flags and targets.
        base_flags, targets = _split_pytest_args(rest)
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


def render(cmd: VerifyCmd) -> str:
    """Render a VerifyCmd back into a shell command string.

    The single shell-string producer — the inverse of ``parse_config_command``
    for well-formed, non-OPAQUE, non-chained commands (P2).

    Computes an *inner* command first: OPAQUE commands and RECOGNISED-BUT-
    UNSTRUCTURABLE multi-segment chains (``cmd.raw is not None`` in both
    cases) use ``raw`` verbatim — there is no structured state to reassemble,
    and for a chain any mutation happens as a localised in-place rewrite of
    ``raw`` itself (``cargo_scope`` / ``serial_pytest``), not through the
    fields rendered below. Otherwise the inner command is assembled, in
    canonical order: a leading ``cd <cwd_rel> &&`` (if set), a ``uv run
    [--project <uv_project>]`` wrapper (if uv-wrapped), the ``npx`` sentinel
    (if present in ``wrappers``), the tool's canonical head, then
    ``base_flags`` and ``targets`` — each value token ``shlex.quote``d for
    shell safety. ``strip_cwd``/``reproject`` normalise both
    ``--directory``/``--project`` and a leading ``cd`` into ``cwd_rel``/
    ``uv_project``; rendering ``cwd_rel`` as a leading ``cd`` unconditionally
    is a documented normalisation (argv-equivalent, not always byte-identical
    to a ``--directory``-form input).

    Finally, if ``wrappers`` carries a non-``'npx'`` entry (set by
    ``govern_cpu`` — a resolved cpu-governed-exec path), the inner command
    computed above is wrapped as an outermost ``<exec> --role merge --
    /bin/bash -c <quoted-inner>``: the ``shlex.quote``d inner payload keeps
    any shell operators (``&&``, ``|``, ...) inside it intact, including for
    a raw-retained chain OR an OPAQUE command (migrates
    ``_maybe_govern_merge_cmd``'s bash-wrap, which unconditionally wrapped
    ANY non-``None`` command for ``role=='merge'`` — see ``govern_cpu``'s
    docstring for why OPAQUE is the one deliberate exemption from P1).

    Invariant asserts (defensive — no mutator other than ``govern_cpu``
    produces these states; they catch a hand-constructed VerifyCmd that
    bypassed a no-op guard): **P1** OPAQUE must never have been mutated
    except via ``govern_cpu`` — ``uv_project``/``cwd_rel`` stay at their
    parse-time defaults and ``targets`` stays empty; ``wrappers`` MAY
    legitimately carry a ``govern_cpu`` entry (merge-role cpu governance
    must still apply to an OPAQUE/arbitrary-shell command — see
    ``govern_cpu``). **P3** ANY raw-retained command (OPAQUE or a
    recognised-but-unstructurable chain) carries no meaningful
    ``cwd_rel``/``targets`` — neither field is legitimately settable once
    ``raw`` is retained (``cargo_scope``/``serial_pytest`` rewrite ``raw``
    itself instead), so render() never has a reason to treat ``targets`` as
    worktree-root-relative there. ``wrappers`` is exempt from the P3 check
    too — a raw-retained command legitimately carries a ``govern_cpu``
    wrapper (the "legitimate wrapper context").

    A raw-retained return therefore drops the structured ``targets`` a
    scoper produced; scoping provenance is recorded on
    ``verify_plan.PlannedRun.scoped_targets`` instead — see that field's
    docstring for why P3 was kept rather than relaxed to carry it (task
    3219).
    """
    if cmd.raw is not None:
        assert cmd.cwd_rel is None and not cmd.targets, (
            'render: a raw-retained VerifyCmd (OPAQUE or an unstructurable '
            'chain) must not carry cwd_rel/targets (P3)'
        )
        if cmd.tool is ToolKind.OPAQUE:
            assert cmd.uv_project is None, (
                'render: OPAQUE VerifyCmd must never be mutated other than '
                'via govern_cpu (P1)'
            )
        inner = cmd.raw
    else:
        inner = _render_structured(cmd)
    govern_exec = next((w for w in cmd.wrappers if w != 'npx'), None)
    if govern_exec is None:
        return inner
    return f'{shlex.quote(govern_exec)} --role merge -- /bin/bash -c {shlex.quote(inner)}'


def _render_structured(cmd: VerifyCmd) -> str:
    """Assemble the shell string for a non-raw-retained (structured) VerifyCmd."""
    segments: list[str] = []
    if cmd.cwd_rel is not None:
        segments.append(f'cd {shlex.quote(cmd.cwd_rel)} &&')
    if cmd.uv_project is not None:
        if cmd.uv_project:
            segments.append(f'uv run --project {shlex.quote(cmd.uv_project)}')
        else:
            segments.append('uv run')
    if 'npx' in cmd.wrappers:
        segments.append('npx')
    segments.append(_TOOL_HEAD[cmd.tool])
    segments.extend(shlex.quote(flag) for flag in cmd.base_flags)
    segments.extend(shlex.quote(target) for target in cmd.targets)
    return ' '.join(segments)


def scope_to(cmd: VerifyCmd, files: list[str]) -> VerifyCmd:
    """Return *cmd* with ``targets`` replaced by *files* (worktree-root-relative).

    ``tool``/``base_flags``/``uv_project``/``cwd_rel``/``wrappers`` are left
    untouched — only ``targets`` is replaced, so dash-prefixed flags parsed
    into ``base_flags`` never leak into (or out of) the new target list
    (migrates the historical ``_scope_command`` dash-token regression).

    A no-op (returns *cmd* unchanged) when *files* is empty, when
    ``cmd.tool is ToolKind.OPAQUE`` (P1), or when ``cmd.raw is not None`` (a
    recognised-but-unstructurable chain — ``targets`` has no meaning there;
    ``cargo_scope``/``serial_pytest`` rewrite ``raw`` directly instead).
    """
    if cmd.tool is ToolKind.OPAQUE or cmd.raw is not None or not files:
        return cmd
    return replace(cmd, targets=tuple(files))


def strip_cwd(cmd: VerifyCmd) -> VerifyCmd:
    """Return *cmd* with ``cwd_rel`` cleared to ``None``.

    Unifies the two historical cwd-shifting forms — a leading ``cd <dir> &&``
    (old ``_strip_leading_cd``) and a uv ``--directory <dir>`` flag (old
    ``_strip_directory_flag``) — since ``parse_config_command`` already
    folds both into the single ``cwd_rel`` field. ``uv_project`` (``--project``)
    is left untouched: it selects the venv, independent of cwd.

    A no-op on ``cmd.tool is ToolKind.OPAQUE`` (P1) or a raw-retained chain
    (``cmd.raw is not None``) — neither carries a meaningful ``cwd_rel``.
    """
    if cmd.tool is ToolKind.OPAQUE or cmd.raw is not None:
        return cmd
    return replace(cmd, cwd_rel=None)


def reproject(cmd: VerifyCmd, project: str) -> VerifyCmd:
    """Return *cmd* with ``uv_project`` set to *project* (regression ef68777a17).

    Applies only to a bare ``uv run <tool>`` — ``uv_project == ''`` (uv-wrapped,
    no explicit ``--project``/``--directory`` yet; see the tri-state note on
    ``VerifyCmd.uv_project``) and ``cwd_rel is None``. No-op when: not
    uv-wrapped at all (``uv_project is None``); an explicit ``--project`` is
    already set (``uv_project`` non-empty); an explicit ``--directory`` is
    already set (``cwd_rel is not None`` — the structural equivalent of
    05c2d87a72's clause-scoped "don't second-guess an explicit uv context"
    guard); OPAQUE (P1); or a raw-retained chain. Idempotent: reprojecting an
    already-reprojected command is a no-op (uv_project is then non-empty).
    """
    if cmd.tool is ToolKind.OPAQUE or cmd.raw is not None:
        return cmd
    if cmd.uv_project != '' or cmd.cwd_rel is not None:
        return cmd
    return replace(cmd, uv_project=project)


# Cargo subcommands whose --workspace flag we know how to rewrite. Other
# cargo subcommands (doc, bench, ...) are left alone to avoid semantic drift.
# Moved from verify.py's _scope_cargo_workspace / _CARGO_SUBCMDS.
_CARGO_SUBCMDS = ('test', 'clippy', 'check', 'build', 'run')

# Matches `cargo <subcmd> ...--workspace` where `...` does not cross a shell
# delimiter (&&, ||, ;, |), so a chained non-cargo command is left alone.
_CARGO_WORKSPACE_RE = re.compile(
    r'(cargo\s+(?:' + '|'.join(_CARGO_SUBCMDS) + r')\b[^&|;]*?)' r'\s--workspace\b',
)

# Matches `--exclude <name>` (or `--exclude=name`) inside the same cargo
# subcommand segment — invalid once --workspace is replaced with -p <crate>
# (cargo rejects "--exclude can only be used together with --workspace").
_CARGO_EXCLUDE_RE = re.compile(
    r'(cargo\s+(?:' + '|'.join(_CARGO_SUBCMDS) + r')\b[^&|;]*?)' r'\s--exclude(?:\s+|=)\S+',
)


def cargo_scope(cmd: VerifyCmd, crates: list[str]) -> VerifyCmd:
    """Return *cmd* with ``cargo ... --workspace`` rewritten to ``-p <crate>`` per crate.

    No-ops when *cmd* isn't a cargo ToolKind (covers OPAQUE — P1) or *crates*
    is empty. For a structured cargo command (``cmd.raw is None``), rewrites
    ``targets`` in place: ``--workspace`` is replaced (at its own position)
    by ``-p c1 -p c2 ...`` and every ``--exclude``/``--exclude=value`` pair
    is dropped, preserving any trailing ``-- <passthrough>`` after the
    inserted crate flags (regression fd4758fcff). For a raw-retained cargo
    chain (e.g. the reify 4-segment ``gated.sh cargo test ... && ...
    --workspace --exclude ...``), applies the same rewrite as a localised
    regex substitution on ``raw`` — gated segments with no ``--workspace``
    are left byte-identical (regression: reify test A4).
    """
    if cmd.tool not in (ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY):
        return cmd
    if not crates:
        return cmd
    if cmd.raw is not None:
        return _cargo_scope_raw(cmd, crates)
    return _cargo_scope_structured(cmd, crates)


def _cargo_scope_raw(cmd: VerifyCmd, crates: list[str]) -> VerifyCmd:
    raw = cmd.raw
    assert raw is not None
    if '--workspace' not in raw:
        return cmd
    p_flags = ' '.join(f'-p {c}' for c in crates)
    new_raw = _CARGO_WORKSPACE_RE.sub(lambda m: f'{m.group(1)} {p_flags}', raw)
    # Loop until stable to handle multiple --exclude flags on one command.
    prev = None
    while prev != new_raw:
        prev = new_raw
        new_raw = _CARGO_EXCLUDE_RE.sub(lambda m: m.group(1), new_raw)
    return replace(cmd, raw=new_raw)


def _cargo_scope_structured(cmd: VerifyCmd, crates: list[str]) -> VerifyCmd:
    tokens = list(cmd.targets)
    if '--workspace' not in tokens:
        return cmd
    ws_idx = tokens.index('--workspace')
    p_flags: list[str] = []
    for crate in crates:
        p_flags.extend(('-p', crate))
    spliced = tokens[:ws_idx] + p_flags + tokens[ws_idx + 1 :]

    cleaned: list[str] = []
    skip_next = False
    for tok in spliced:
        if skip_next:
            skip_next = False
            continue
        if tok == '--exclude':
            skip_next = True
            continue
        if tok.startswith('--exclude='):
            continue
        cleaned.append(tok)
    return replace(cmd, targets=tuple(cleaned))


# Matches a pytest invocation up to (but not including) the next shell chain
# operator (&&, ||, ;) or end of string — the span serial_pytest's raw-chain
# path rewrites. Word-bounded so it doesn't match inside 'pytest_xdist' etc.
# Moved from verify.py's _PYTEST_INVOCATION_RE / _force_serial_pytest.
_PYTEST_INVOCATION_RE = re.compile(r'\bpytest\b[^&|;]*')


def _append_to_raw_pytest_invocations(raw: str, suffix: str) -> str:
    """Return *raw* with *suffix* appended to every pytest invocation.

    Shared rewrite closure for ``serial_pytest``/``apply_pytest_numprocesses``'s
    raw-retained (chained) path: matches each ``_PYTEST_INVOCATION_RE`` span,
    strips trailing whitespace, appends *suffix*, then re-attaches the
    trailing whitespace so an immediately-following chain operator (e.g.
    `` && ``) survives untouched. *suffix* should include its own leading
    space (e.g. ``' -n 16'``).
    """
    def _rewrite(match: re.Match[str]) -> str:
        segment = match.group(0)
        stripped = segment.rstrip()
        trailing = segment[len(stripped) :]
        return f'{stripped}{suffix}{trailing}'

    return _PYTEST_INVOCATION_RE.sub(_rewrite, raw)


def serial_pytest(cmd: VerifyCmd) -> VerifyCmd:
    """Return *cmd* with the serial-recovery flags applied to every pytest invocation.

    Appends ``-p no:xdist -o addopts=`` (clears any pyproject-level
    ``addopts``, e.g. ``-n auto`` — the ``-o addopts=""`` workaround task
    2045 proved recovers a shared-venv-mutation transient; ``-p no:xdist``
    is belt-and-suspenders) to a structured command's ``base_flags``, or —
    for a raw-retained pytest chain — to every ``pytest`` invocation's
    arguments in ``raw`` via a localised regex rewrite (moved from
    ``_force_serial_pytest``), so each chained invocation recovers
    independently. No-ops unless ``cmd.tool is ToolKind.PYTEST`` (covers
    OPAQUE and every other tool — P1).
    """
    if cmd.tool is not ToolKind.PYTEST:
        return cmd
    if cmd.raw is not None:
        return replace(
            cmd, raw=_append_to_raw_pytest_invocations(cmd.raw, " -p no:xdist -o addopts=''")
        )
    return replace(cmd, base_flags=(*cmd.base_flags, '-p', 'no:xdist', '-o', 'addopts='))


def _is_serial_forced(cmd: VerifyCmd) -> bool:
    """True when *cmd* has been forced serial (xdist plugin disabled).

    ``serial_pytest`` disables xdist by appending ``-p no:xdist`` — to
    ``base_flags`` for a structured command, or into every ``pytest``
    invocation in ``raw`` for a chain. With the xdist plugin disabled pytest
    does not register the ``-n``/``--numprocesses`` option, so injecting
    ``-n`` afterwards makes pytest exit with ``unrecognized arguments: -n``.
    ``apply_pytest_numprocesses`` consults this to stay a no-op on any
    already-serial command (the env-transient and flaky-scoped recovery
    re-runs both pass such commands back through the injection site).

    ``no:xdist`` is checked across both ``base_flags`` and ``targets``: a
    freshly ``serial_pytest``-ed structured command carries the ``-p
    no:xdist`` pair in ``base_flags``, and since ``_parse_single_segment``
    binds a separate-token value flag like ``-p`` to its following value at
    parse time (task 2727), re-parsing that rendered string at the injection
    site (verify.py's recovery re-runs render then feeds the command back
    through ``parse_config_command``) keeps ``no:xdist`` in ``base_flags``
    too — the ``targets`` check is now defensive-only (guards a
    hand-constructed or pre-fix-shaped ``VerifyCmd``) but harmless to keep.
    """
    if cmd.raw is not None:
        return 'no:xdist' in cmd.raw
    return 'no:xdist' in cmd.base_flags or 'no:xdist' in cmd.targets


def apply_pytest_numprocesses(cmd: VerifyCmd, n: str) -> VerifyCmd:
    """Return *cmd* with a `-n <n>` pytest-xdist worker-count flag applied.

    Appends ``-n <n>`` to a structured command's ``base_flags``, or — for a
    raw-retained pytest chain — to every ``pytest`` invocation's arguments in
    ``raw`` via the same localised regex rewrite ``serial_pytest`` uses, so
    each chained invocation gets its own cap independently.

    A no-op (returns *cmd* unchanged) unless ``cmd.tool is ToolKind.PYTEST``
    (covers OPAQUE and every other tool — P1), and also when *n* is ``''`` or
    ``'auto'`` — the byte-identical guard: the pyproject ``-n auto`` addopts
    already picks a worker count, so there is nothing to override — and when
    the command has already been forced serial (``-p no:xdist``): with xdist
    disabled the ``-n`` option is unregistered, so injecting it would fail the
    run with ``unrecognized arguments: -n``. The serial-recovery re-runs
    (env-transient at verify.py's env-recovery retry, flaky-scoped isolated
    re-run) build their command via ``serial_pytest`` and then pass it back
    through the same injection site, so this guard is what keeps the ``-n``
    knob from breaking those recovery paths.
    """
    if cmd.tool is not ToolKind.PYTEST or n in {'', 'auto'} or _is_serial_forced(cmd):
        return cmd
    if cmd.raw is not None:
        return replace(cmd, raw=_append_to_raw_pytest_invocations(cmd.raw, f' -n {n}'))
    return replace(cmd, base_flags=(*cmd.base_flags, '-n', n))


def with_junitxml(cmd: VerifyCmd, junit_path: str) -> VerifyCmd:
    """Return *cmd* with a ``--junitxml <junit_path>`` flag appended to ``base_flags``.

    Task μ (verify-scope-inversion-prd.md): structured field edit, never a
    regex, mirroring ``apply_pytest_numprocesses``'s structured branch. A
    no-op (returns *cmd* unchanged) unless ``cmd.tool is ToolKind.PYTEST and
    cmd.raw is None`` — covers OPAQUE and every other tool (P1), and,
    deliberately UNLIKE ``apply_pytest_numprocesses``/``serial_pytest``, also
    covers a raw-retained pytest chain (``cmd.raw is not None``): there is no
    regex-rewrite branch here, so a recognised-but-unstructurable
    ``&&``-chained pytest command is left byte-identical rather than
    rewritten. Callers degrade gracefully (no junit collected for that run —
    B3) rather than risk a mis-scoped injection into an unstructured shell
    string.

    *junit_path* should be an absolute path: the rendered command may run
    with a shifted cwd (a structured command's own ``cd <cwd_rel> &&``), so a
    relative ``--junitxml`` value would land in the wrong directory.
    """
    if cmd.tool is not ToolKind.PYTEST or cmd.raw is not None:
        return cmd
    return replace(cmd, base_flags=(*cmd.base_flags, '--junitxml', junit_path))


def with_pytest_timeout(cmd: VerifyCmd, secs: int) -> VerifyCmd:
    """Return *cmd* with a ``--timeout <secs>`` per-test timeout appended to ``base_flags``.

    PRD task α (cpu-load-robust-verify-prd.md): structured field edit, never a
    regex, copying ``with_junitxml``'s structure exactly. A no-op (returns
    *cmd* unchanged) unless ``cmd.tool is ToolKind.PYTEST and cmd.raw is
    None`` — covers OPAQUE and every other tool (P1), and, like
    ``with_junitxml`` (and deliberately UNLIKE
    ``apply_pytest_numprocesses``/``serial_pytest``), also covers a
    raw-retained pytest chain (``cmd.raw is not None``): there is no
    regex-rewrite branch here, so a recognised-but-unstructurable
    ``&&``-chained pytest command is left byte-identical rather than
    rewritten.

    The α confirm gate injects this AFTER ``serial_pytest``'s
    ``-p no:xdist -o addopts=`` recovery form: the pyproject per-test
    ``timeout=60`` default lives in ``[tool.pytest.ini_options]``, NOT in
    ``addopts``, so ``-o addopts=`` does not clear it. Without a GENEROUS
    explicit override the isolated confirm re-run could itself starve into a
    false non-suppression (never masking a real red is a hard constraint, but
    masking a genuine flake as a real red just because the confirm run was
    also starved defeats the gate's purpose).
    """
    if cmd.tool is not ToolKind.PYTEST or cmd.raw is not None:
        return cmd
    return replace(cmd, base_flags=(*cmd.base_flags, '--timeout', str(secs)))


def govern_cpu(cmd: VerifyCmd, exec_path: str | None) -> VerifyCmd:
    """Return *cmd* with a cpu-governed-exec wrapper appended to ``wrappers``.

    ``render`` recognises any ``wrappers`` entry other than the ``'npx'``
    sentinel as a resolved cpu-governed-exec path and wraps the *entire*
    rendered command — structured, raw-retained chain, or OPAQUE alike — as
    an outermost ``<exec> --role merge -- /bin/bash -c <quoted-inner>``
    (migrates ``_maybe_govern_merge_cmd``'s bash-wrap; the ``shlex.quote``d
    inner payload preserves shell operators like ``&&`` inside a
    raw-retained/OPAQUE command intact).

    A no-op only when *exec_path* is falsy (``''``/``None`` — governance not
    configured/resolved). Unlike every other mutator, ``govern_cpu`` does
    NOT no-op on ``cmd.tool is ToolKind.OPAQUE`` — this is the sole
    deliberate exemption from P1. The historical ``_maybe_govern_merge_cmd``
    bash-wrapped ANY non-``None`` command for ``role=='merge'`` regardless
    of shape, and OPAQUE is exactly the "arbitrary/unparseable shell" case
    that bash-wrap exists for (e.g. dark_factory's multi-clause
    ``lint_command``/``type_check_command`` chains, which parse OPAQUE —
    see ``_parse_chain`` — yet must still receive merge-weighted cgroup
    scope like every other merge verify command).
    """
    if not exec_path:
        return cmd
    return replace(cmd, wrappers=(*cmd.wrappers, exec_path))
