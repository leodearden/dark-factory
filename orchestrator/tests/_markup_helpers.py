"""Substrate the plan-tools and verdict-tools markup suites BOTH need.

Three suites in this package describe MCP tool-call envelope markup, so all
three need the same two things and neither may hold a second copy of either.

SENTINEL BUILDERS. A raw envelope literal in a test file corrupts the very
tool call that edits it — the Write/Edit argument terminates early, truncating
the file and silently dropping that call's sibling arguments (the rationale
recorded in the "Sentinel-literal hazard" section of
``shared/src/shared/toolcall_markup.py``). So every specimen is BUILT from
:func:`closer` / :func:`param_opener`, whose angle bracket comes from
``chr(60)``. The source rule is enforced repo-wide by
``tests/scripts/test_no_raw_envelope_literal.py::test_no_markup_handling_file_spells_a_raw_envelope_literal``.

SCHEMA READING. :func:`type_alternatives` is the one reading of what a
``dict | None`` parameter declaration MEANS, shared so a future change in how
fastmcp renders it cannot be applied to one suite and missed in the other,
leaving that suite silently asserting nothing.
"""

from __future__ import annotations

from typing import Any

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
