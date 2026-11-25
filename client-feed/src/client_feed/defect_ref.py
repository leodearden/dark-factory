"""The one ref rule for a client's defect report (plans/client-feed-prd.md §14.2 decision 11): portal_ref mints it, parse_portal_ref and portal_ref_prefix read it back, and check_ref is the one ref of the grammar it never mints; kept out of the vendorable contract."""

from __future__ import annotations

from typing import Final, NamedTuple

from shared.defects import DefectOrigin, MalformedDefectRef, mint_portal_ref, parse_ref

CHECK_BLOCK: Final = 'check'


class PortalReport(NamedTuple):
    """The block and report a portal ref names."""

    block: int
    report: int


def portal_ref(project_id: str, block: int, report: int) -> str:
    """`portal:<project_id>:<block>:<report>`; the filer and the publisher both mint through this function."""
    if type(block) is not int or block < 1:
        origin = DefectOrigin.CLIENT
        raise MalformedDefectRef(
            f'{origin} ref component block={block!r} must be an int of at least 1'
            ' (bool is excluded even though it is an int subclass)',
            origin=origin,
            value=block,
        )
    return mint_portal_ref(project_id, str(block), report)


def parse_portal_ref(ref: str, *, project_id: str) -> PortalReport | None:
    """The report a portal ref of `project_id` names; None for any ref portal_ref would not mint for this project."""
    try:
        parsed = parse_ref(ref)
    except MalformedDefectRef:
        return None
    if parsed.origin is not DefectOrigin.CLIENT or parsed.component('project') != project_id:
        return None
    block = parsed.component('block')
    if not (block.isascii() and block.isdecimal()):
        return None
    named = PortalReport(block=int(block), report=int(parsed.component('n')))
    return named if named.block >= 1 and portal_ref(project_id, *named) == ref else None


def portal_ref_prefix(project_id: str) -> str:
    """The text every portal ref of `project_id` begins with: what a store query pre-filters on before parse_portal_ref confirms each value."""
    return portal_ref(project_id, 1, 1).removesuffix('1:1')


def check_ref(project_id: str) -> str:
    """`portal:<project_id>:check:1`: a client ref of the project whose block is a token, so no report and no carrier is ever named by it."""
    return mint_portal_ref(project_id, CHECK_BLOCK, 1)
