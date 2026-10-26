"""Slice a recon stage prompt down to its ``## Live-Workflow Authority`` region.

The region runs from that heading to the next ``## `` heading, or to the end
of the prompt. Pins scoped to the region read only this slice, so a token that
appears elsewhere in a long stage prompt cannot satisfy them. The shared rules
block (``reconciliation/live_workflow_section.py::render_live_workflow_authority_rules``)
sits inside the region and uses ``###`` headings only, so it never ends the
slice early.

Lives outside conftest.py, for the same reason ``_fm_helpers`` does, so test
files can ``from _live_workflow_region import live_workflow_region``.
"""

from __future__ import annotations

LIVE_WORKFLOW_REGION_HEADER = '## Live-Workflow Authority'


def live_workflow_region(prompt: str) -> str:
    assert prompt.count(LIVE_WORKFLOW_REGION_HEADER) == 1, (
        f'{LIVE_WORKFLOW_REGION_HEADER!r} must appear exactly once; otherwise '
        'the slice silently reads the wrong region, or none'
    )
    _, _, after = prompt.partition(LIVE_WORKFLOW_REGION_HEADER)
    end = after.find('\n## ')
    return after if end == -1 else after[:end]
