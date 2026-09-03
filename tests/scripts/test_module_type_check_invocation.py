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
from collections.abc import Callable

from orchestrator.config import ModuleConfig

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


# ALL THREE fields are guarded, not just the type gate this task's title
# names. The npm-cache dependency esc-3473-2 measured is a property of HOW a
# command resolves its binaries, not of WHICH checker it runs — an
# npx-fronted lint_command or test_command carries the identical
# shared-mutable-cache hazard. Only the type gate has any npx HISTORY
# (scripts/ and tests/scripts/ were both once npx-fronted pyright
# invocations); the other two fields are pure regression prevention and were
# measured clean on all nine discovered configs at planning time.
_GUARDED_COMMAND_FIELDS = ('type_check_command', 'lint_command', 'test_command')


def _npx_fronted_fields(mc: ModuleConfig) -> dict[str, list[str]]:
    """Map each guarded command field of *mc* to its offending npx-fronted segments.

    Iterates :data:`_GUARDED_COMMAND_FIELDS`, skipping a falsy value — a
    module config declaring no command in a given field is a legitimate
    state (``verify`` renders it as a SKIPPED PlannedRun, not a violation),
    not something :func:`_npx_fronted_segments` has an opinion about. Returns
    only the fields with at least one offending segment, so a clean config
    (or one declaring no commands at all) yields ``{}``.
    """
    offenders: dict[str, list[str]] = {}
    for field in _GUARDED_COMMAND_FIELDS:
        cmd = getattr(mc, field)
        if not cmd:
            continue
        segments = _npx_fronted_segments(cmd)
        if segments:
            offenders[field] = segments
    return offenders


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


def test_the_field_scan_covers_every_guarded_command_field() -> None:
    """`_npx_fronted_fields` must scan type_check_command, lint_command AND test_command.

    Six lettered cases, built on fabricated `ModuleConfig`s (a dataclass with
    a required `prefix` plus optional command fields).
    """
    # (a) npx in type_check_command only.
    mc = ModuleConfig(prefix='fake', type_check_command='npx pyright scripts/')
    offenders = _npx_fronted_fields(mc)
    assert set(offenders) == {'type_check_command'}, (
        f'expected only type_check_command flagged, got {offenders!r}'
    )
    assert [s.strip() for s in offenders['type_check_command']] == [
        'npx pyright scripts/'
    ], f'expected the offending segment itself to be reported, got {offenders!r}'

    # (b) npx in lint_command only — the field the task calls "arguably" in
    # scope; pinning it here is what stops the scan quietly covering only the
    # type gate.
    mc = ModuleConfig(prefix='fake', lint_command='npx eslint src/')
    offenders = _npx_fronted_fields(mc)
    assert set(offenders) == {'lint_command'}, (
        f'expected only lint_command flagged, got {offenders!r}'
    )

    # (c) npx in test_command only.
    mc = ModuleConfig(prefix='fake', test_command='npx jest')
    offenders = _npx_fronted_fields(mc)
    assert set(offenders) == {'test_command'}, (
        f'expected only test_command flagged, got {offenders!r}'
    )

    # (d) npx in a chained lint TAIL clause, mirroring the real chained
    # lint_command shape every workspace member declares (e.g. cockpit's
    # `ruff check ... && python3 .../check_bare_magicmock_config.py ...`).
    mc = ModuleConfig(
        prefix='fake',
        lint_command='uv run --directory x ruff check src/ && npx some-linter',
    )
    offenders = _npx_fronted_fields(mc)
    assert set(offenders) == {'lint_command'}, (
        f'a chained TAIL clause must be flagged even though the head clause '
        f'is clean `uv run`, got {offenders!r}'
    )
    assert [s.strip() for s in offenders['lint_command']] == ['npx some-linter'], (
        f'expected the TAIL clause itself to be reported, got {offenders!r}'
    )

    # (e) all three fields None -> {}, not a crash. A module config declaring
    # no commands at all is a legitimate state (verify renders a falsy
    # command as a SKIPPED PlannedRun, not a violation).
    mc = ModuleConfig(prefix='fake')
    assert _npx_fronted_fields(mc) == {}, (
        f'a module config with no commands declared must scan clean with no '
        f'crash, got {_npx_fronted_fields(mc)!r}'
    )

    # (f) THE FALSE-POSITIVE FLOOR: the three REAL commands measured on this
    # tree for cockpit/orchestrator.yaml must not be flagged.
    mc = ModuleConfig(
        prefix='cockpit',
        test_command='uv run --directory cockpit pytest tests/ --tb=short -q',
        lint_command=(
            'uv run --directory cockpit ruff check src/ tests/ && '
            'python3 fused-memory/scripts/check_bare_magicmock_config.py cockpit/tests'
        ),
        type_check_command='uv run --directory cockpit pyright src/ tests/',
    )
    assert _npx_fronted_fields(mc) == {}, (
        f"cockpit's real, measured commands must not be flagged, got "
        f'{_npx_fronted_fields(mc)!r}'
    )


def test_no_discovered_module_config_shells_a_guarded_command_through_npx(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """No module config `_discover_module_configs` returns may shell a guarded command through npx.

    THE FIRST REPO-WIDE GUARD. The invariant it asserts is already GREEN on
    all nine configs (measured at planning time), so this is regression
    prevention: task 3842 fixed ``tests/scripts/``, task 4358 fixed
    ``scripts/`` (the last holdout), and nothing since has reintroduced an
    npx-fronted command anywhere `_discover_module_configs` looks.

    Takes the `discover_module_configs` fixture as a CALLABLE and calls it
    inside the test body (never hoisted to setup) — see that fixture's own
    docstring in conftest.py for why.

    Does NOT cover ``dark-factory-orchestrator.yaml``, the repo root config,
    which is deliberately npx-fronted (see this module's docstring, "THE
    ROOT-CONFIG CARVE-OUT") and is pinned in that exact shape by
    ``test_pyright_version_pin.py::test_the_fleet_chain_stays_bare_npx_pyright``.
    `_discover_module_configs` never returns that config, so this loop cannot
    see it.
    """
    discovered = discover_module_configs()

    # ANTI-VACUITY FLOOR, asserted FIRST. Without it, a regression in the
    # production walk would shrink `discovered` and the loop below would pass
    # VACUOUSLY on whatever remained — the same hole
    # `test_module_verify_budgets.py::test_every_discovered_module_config_declares_its_own_verify_budget`
    # closes for the sibling guard. SUBSET, not equality: a newly-registered
    # module config must be covered by the loop automatically, with no edit
    # here.
    missing = KNOWN_MODULE_CONFIG_PREFIXES - set(discovered)
    assert not missing, (
        f'the production walk (config._discover_module_configs) failed to '
        f'resolve known module config(s) {sorted(missing)} — discovery has '
        f'regressed, and the npx-ban loop below would pass vacuously on the '
        f'shrunken set. Discovered: {sorted(discovered)}'
    )

    for prefix, mc in sorted(discovered.items()):
        offenders = _npx_fronted_fields(mc)
        assert not offenders, (
            f'{prefix}/orchestrator.yaml shells {sorted(offenders)} through '
            f'npx: {offenders!r}. MEASURED (esc-3473-2): a bare `npx pyright` '
            f're-resolves through the shared, mutable, concurrently-written '
            f'npm cache under $HOME on EVERY invocation, and once turned a '
            f'clean 0-error type leg RED on a transient npm-cache write '
            f'failure (npm could not write ~/.npm/_logs) with no real defect '
            f'in the tree. Because verify.run_full_verification '
            f'asyncio-gathers over ALL module_configs and this repo\'s root '
            f'sets merge_verify_breadth: "full", that is a FLEET-WIDE '
            f'false-red blocking every merge, review checkpoint and main-tip '
            f'sweep — on a branch with no defect. Remedy: resolve the '
            f'checker through `uv run --directory {prefix} ...` if {prefix} '
            f'is a [tool.uv.workspace] member, else `uv run --project '
            f'shared ...`'
        )
