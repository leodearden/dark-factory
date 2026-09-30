"""Substrate the plan-tools and verdict-tools markup suites BOTH need.

Three suites in this package describe MCP tool-call envelope markup, so all
three need the same two things and neither may hold a second copy of either.

SENTINEL BUILDERS. A raw envelope literal in a test file corrupts the very
tool call that edits it — the Write/Edit argument terminates early, truncating
the file and silently dropping that call's sibling arguments (the rationale
recorded at ``shared/src/shared/toolcall_markup.py`` lines 52-62). So every
specimen is BUILT from :func:`closer` / :func:`param_opener`, whose angle bracket
comes from ``chr(60)``, and :func:`assert_no_raw_sentinels` enforces that on a
module's own bytes at import — including on THIS module's, at the bottom.

SCHEMA READING. :func:`type_alternatives` is the one reading of what a
``dict | None`` parameter declaration MEANS, shared so a future change in how
fastmcp renders it cannot be applied to one suite and missed in the other,
leaving that suite silently asserting nothing.

Everything here is derived from ``shared.toolcall_markup.ENVELOPE_LITERALS``,
which stays the single owner of the literal set (INV-5); this module adds the
two structural prefixes a built specimen uses and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.toolcall_markup import ENVELOPE_LITERALS

#: The opening angle bracket, spelled so it never appears verbatim in a file.
LT = chr(60)


def closer(name: str) -> str:
    """Build the closing tag for *name* (the mis-close shape the harness emits)."""
    return LT + '/' + name + '>'


def param_opener(name: str) -> str:
    """Build the canonical opening tag for parameter *name*."""
    return LT + 'parameter name="' + name + '">'


#: The bare invoke closer — the terminator that trails a last-parameter leak.
INVOKE_CLOSER = closer('invoke')


def assert_no_raw_sentinels(module_file: str) -> None:
    """Fail at IMPORT if *module_file*'s own bytes carry a raw envelope literal.

    Checked against ``ENVELOPE_LITERALS`` plus the two structural prefixes every
    built specimen uses, so a builder's output spelled out by hand is caught
    even when it is not itself one of the enumerated literals.

    Takes the caller's ``__file__`` rather than reading its own: the check is
    about the bytes of the file being EDITED, and a shared implementation that
    scanned itself would report every suite as clean.
    """
    path = Path(module_file)
    source = path.read_text(encoding='utf-8')
    for sequence in (*ENVELOPE_LITERALS, LT + '/', LT + 'parameter '):
        if sequence in source:
            raise AssertionError(
                f'{path.name} contains a RAW envelope sentinel '
                f'({sequence!r}). Build it from closer()/param_opener() instead '
                '— a verbatim literal here corrupts the tool call that writes '
                'this file. See _markup_helpers.'
            )


def type_alternatives(schema: dict[str, Any]) -> set[str]:
    """The JSON-Schema type names *schema* accepts, however it spells them.

    ``anyOf`` branches and a list-valued ``type`` are the two renderings a
    ``dict | None`` annotation can produce, and which one a given fastmcp emits
    is its business, not this contract's. Reading both keeps a row asserting
    what the schema MEANS rather than how the library formats it.
    """
    if isinstance(schema.get('anyOf'), list):
        branch_types = (
            branch.get('type') for branch in schema['anyOf'] if isinstance(branch, dict)
        )
        return {name for name in branch_types if isinstance(name, str)}
    declared = schema.get('type')
    if isinstance(declared, list):
        return {name for name in declared if isinstance(name, str)}
    return {declared} if isinstance(declared, str) else set()


assert_no_raw_sentinels(__file__)
