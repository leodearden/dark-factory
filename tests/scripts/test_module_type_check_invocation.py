"""Placeholder — task 4369 step-2 replaces this with the full module docstring.

This file intentionally carries only the failing self-check for
``_npx_fronted_segments`` until step-2 implements that helper.
"""
from __future__ import annotations


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
