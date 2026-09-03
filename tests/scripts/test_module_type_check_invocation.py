"""Repo-wide guard: no discovered module config shells a checker through npx.

WHAT AND WHY, and it is a measurement, not a style preference. esc-3473-2
recorded a bare ``npx pyright`` re-resolving through the shared, mutable,
concurrently-written npm cache under ``$HOME`` on EVERY invocation, and once
turning a clean 0-error type leg RED on a transient npm-cache write failure
(npm could not write ``~/.npm/_logs``) with no real type errors in the tree.
Because ``verify.run_full_verification`` asyncio-gathers over ALL
``module_configs.values()`` and the repo root sets
``merge_verify_breadth: "full"``, that is a fleet-wide false-red blocking
every merge, review checkpoint and main-tip sweep — on branches with no
defect. Task 3842 fixed ``tests/scripts/``, task 4358 fixed ``scripts/`` (the
last holdout), and this task (4369) promotes the invariant to all nine
configs ``orchestrator.config._discover_module_configs`` returns, so a TENTH
module landing later is covered automatically with no edit here.

GENERALISED FROM, not imported from:
``test_scripts_module_config.py::test_type_gates_resolve_pyright_without_npx``,
which stays scoped to its two non-workspace-member configs and is
deliberately RETAINED (see that file's corrected PLACEMENT paragraph). That
scoped assertion makes five claims; only two promote repo-wide:

  * the exact-token ``npx`` ban — TRUE for all nine configs, all three
    guarded command fields.
  * the pyright segment begins ``uv run`` — also TRUE for all nine, and kept
    rather than dropped because the ban alone is insufficient: satisfying it
    with a bare ``pyright <dir>`` resolves off PATH — where pyright does not
    exist at the worktree root, only inside a member venv (``verify.py``'s
    ``_FALLBACK_UV_PROJECT = 'shared'`` encodes the same pairing) — trading a
    flaky red for a command-not-found red.

Left behind, deliberately: the ``--project <member>`` selector requirement,
the ``[tool.uv.workspace].members`` check, and the post-anchor pyright
config-redirect ban. The seven workspace members declare
``uv run --directory <member> pyright src/ tests/``, not ``--project`` — a
repo-wide ``--project`` requirement would be RED for seven of nine configs on
a tree with no defect. Those three claims stay with the scoped sibling, which
already checks them correctly for the only two configs where ``--project`` is
the real spelling.

THE ROOT-CONFIG CARVE-OUT. ``dark-factory-orchestrator.yaml``'s own
``type_check_command`` IS npx-fronted, on purpose — a seven-clause
``cd <member> && npx pyright && ...`` chain — and
``test_pyright_version_pin.py::test_the_fleet_chain_stays_bare_npx_pyright``
pins it in exactly that shape, because the pyright version pin lives in the
root ``package.json`` rather than at the invocation (``npx pyright@1.1.408``
would break the bare-clause invariant that guard depends on).
``_discover_module_configs`` returns nine entries with no root or
``__fallback__`` key, so the loop below never sees that command — but that is
easy to get WRONG by "helpfully" widening this guard to the root config too,
which would put two correct guards into head-on collision. Don't.

PLACEMENT: ``tests/scripts/`` rather than a per-module ``tests/`` directory,
for the reason ``test_module_verify_budgets.py``'s own PLACEMENT docstring
gives — this directory is ADDITIONALLY covered by its own registered module
config (``tests/scripts/orchestrator.yaml``), so a guard living here cannot
be silenced by an edit to the very command it asserts about. A NEW file,
never appended to the budgets guard, which is about budgets.

IMPORTS: ``verify_command_invariants`` and ``orchestrator.verify_cmd`` are
shared, non-test / production modules, so importing them is unconstrained by
the family's no-cross-import convention — that convention (task 4320
corrected the older, broader phrasing of it) bans importing a SIBLING TEST
FILE, which couples two guards that must be able to fail independently. No
symbol is imported from ``test_scripts_module_config.py`` or
``test_module_verify_budgets.py``, and neither imports this file.

Cited by SYMBOL (``path::symbol``), never ``file:line`` — both sibling guards
record that their own line pins had rotted at HEAD, ``CLAUDE.md`` mandates
it, and ``test_line_pin_policy.py`` is a live guard in this same directory.
"""
from __future__ import annotations

import shlex

from orchestrator import verify_cmd


def _npx_fronted_segments(cmd: str) -> list[str]:
    """The `&&`-chained segments of *cmd* whose shlex tokens contain the exact token `npx`.

    Scans EVERY top-level ``&&`` segment of *cmd* (via the quote-aware,
    documented-lossless production splitter
    ``orchestrator.verify_cmd.split_top_level_and`` — never a naive
    ``str.split('&&')``), not just the segment that invokes the checker.
    Chaining is an established pattern in these configs, not hypothetical:
    every workspace member's ``lint_command`` chains a
    ``python3 .../check_bare_magicmock_config.py <dir>`` gate after
    ``ruff check``, and an npx-fronted TAIL clause
    (``uv run --directory x pyright src/ && npx tsc``) would re-introduce the
    exact npm-cache dependency this scan bans while the checker's own segment
    stays clean — anchoring the scan to the checker segment would leave a
    silent hole.

    Matched on the EXACT shlex TOKEN ``npx`` in each segment, never on a
    substring: a path argument like ``tools/npx-shim/`` or a flag value like
    ``npx.json`` merely CONTAINS the letters and must not be flagged.
    ``pnpx`` is a distinct token and is deliberately out of scope — there is
    no pnpm or bun in this repo, so banning it would assert about an
    unmeasured failure mode.

    Returns the OFFENDING SEGMENTS VERBATIM (not stripped): a non-leading
    segment carries the whitespace ``split_top_level_and`` preserves ahead of
    it, and callers that need the bare command compare with ``.strip()``.
    Returns ``[]`` when *cmd* is clean.

    A segment ``shlex`` cannot tokenise raises a named ``AssertionError``
    naming *cmd* and the offending segment, never a bare
    ``ValueError: No closing quotation`` — the same diagnostic discipline
    ``verify_command_invariants.anchor_split`` uses. This scan is asked to
    CERTIFY THE ABSENCE of ``npx`` across the whole command, so an
    unparseable segment cannot be silently skipped the way
    ``optional_token_segment`` skips one among several candidates: skipping
    here would let a segment this scan cannot read pass as though it had been
    checked and found clean.
    """
    offending: list[str] = []
    for segment in verify_cmd.split_top_level_and(cmd):
        try:
            tokens = shlex.split(segment)
        except ValueError as exc:
            raise AssertionError(
                f'cannot tokenise a `&&`-chained segment of {cmd!r} while '
                f'scanning for npx-fronted checker invocations: {exc}; '
                f'segment: {segment!r}'
            ) from exc
        if 'npx' in tokens:
            offending.append(segment)
    return offending


def test_the_npx_scan_reads_exact_tokens_in_every_chain_segment() -> None:
    """`_npx_fronted_segments` must scan every `&&`-chained segment by exact shlex token.

    Eight lettered cases, each pinning a distinct way a naive scan could get
    this wrong.
    """
    # (a) The historical `scripts` violation — a bare, unchained npx-fronted
    # command. scripts/orchestrator.yaml declared exactly this shape before
    # task 4358.
    result = _npx_fronted_segments('npx pyright scripts/')
    assert len(result) == 1, (
        f'expected exactly one npx-fronted segment in the bare historical '
        f'`npx pyright scripts/` command, got {result!r}'
    )

    # (b) The npx-fronted clause is the SECOND `&&` clause — proves the scan is
    # not anchored to the head clause. This is the root fleet chain's own
    # clause shape (`cd <member> && npx pyright`).
    result = _npx_fronted_segments('cd scripts && npx pyright')
    assert len(result) == 1 and result[0].strip() == 'npx pyright', (
        f'expected the SECOND `&&` clause to be reported as npx-fronted, got '
        f'{result!r} — a scan anchored to the head clause would miss it '
        'entirely, and this is exactly the shape '
        '`dark-factory-orchestrator.yaml`\'s own fleet chain uses'
    )

    # (c) The npx-fronted clause is a TAIL clause after a clean `uv run`
    # checker segment — proves the scan is not anchored to the checker
    # keyword either.
    result = _npx_fronted_segments(
        'uv run --directory cockpit pyright src/ tests/ && npx tsc'
    )
    assert len(result) == 1 and result[0].strip() == 'npx tsc', (
        f'expected the npx-fronted TAIL clause to be reported even though the '
        f'checker\'s own segment is clean `uv run`, got {result!r} — a scan '
        'anchored to the segment that invokes the checker (rather than every '
        'chained segment) would miss this entirely'
    )

    # (d) A flagged npx invocation is still npx.
    result = _npx_fronted_segments('npx --yes pyright')
    assert len(result) == 1, (
        f'a flagged `npx --yes pyright` is still npx-fronted, got {result!r}'
    )

    # (e) The real, already-fixed `scripts` type_check_command must not be
    # flagged.
    assert _npx_fronted_segments('uv run --project shared pyright scripts/') == [], (
        'the real, already-fixed `scripts` type_check_command '
        '(`uv run --project shared pyright scripts/`) must not be flagged'
    )

    # (f) The real workspace-member shape must not be flagged.
    assert (
        _npx_fronted_segments('uv run --directory cockpit pyright src/ tests/') == []
    ), (
        'the real member shape (`uv run --directory <member> ...`) must not '
        'be flagged'
    )

    # (g) SUBSTRING LOOKALIKES — the exact-token discipline the task asks for.
    # A `'npx' in cmd` substring test reports both of these as violations; a
    # shlex exact-token test must not.
    assert (
        _npx_fronted_segments('uv run --project shared pyright tools/npx-shim/') == []
    ), (
        "a path argument that merely CONTAINS the substring 'npx' "
        "('tools/npx-shim/') must not be flagged — the token is "
        "'tools/npx-shim/', not 'npx'"
    )
    assert (
        _npx_fronted_segments(
            'uv run --project shared pyright --outputjson npx.json'
        )
        == []
    ), (
        "a flag VALUE that merely contains 'npx' ('npx.json') must not be "
        "flagged either — the token is 'npx.json', not 'npx'"
    )

    # (h) `pnpx` is a different binary and is deliberately out of scope (see
    # design decisions: no pnpm/bun in this repo, so banning pnpx would assert
    # about an unmeasured failure mode).
    assert _npx_fronted_segments('pnpx pyright') == [], (
        "'pnpx' is a distinct token from 'npx' and this scan bans the exact "
        "token 'npx' only, by deliberate scope limit"
    )
