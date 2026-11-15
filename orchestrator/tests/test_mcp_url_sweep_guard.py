"""Guard: no swept source file builds an MCP URL with a trailing slash (task 4023).

The fused-memory server 307-redirects ``/mcp/`` to ``/mcp`` and a bare httpx
client discards the payload silently.  Split out of test_mcp_post_transport.py,
which keeps the transport-level regression tests for the five fixed POSTs, so
this repo-wide sweep no longer widens those tests to the guard-family ceiling.
The detector itself is ``_mcp_url_scan``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _mcp_url_scan import SweepFindings, sweep_parsed
from _tree_scan import (
    REPO_ROOT,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    python_files,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

#: Directories holding raw-POST source.  Scoped, not repo-wide, on purpose:
#: ``*/tests`` are excluded wholesale because FastMCP's
#: ``StreamableHttpTransport(f'{base}/mcp/')`` fixtures legitimately use the
#: slashed mount (see the guard's docstring).  ``escalation/src`` and
#: ``shared/src`` are in the list even though both are clean today —
#: ``escalation/src`` hosts an MCP server, and ``shared/src`` now hosts the
#: primitive the fix routes through, so both are exactly where a sixth site
#: would appear.  Measured while adding them: 0 hits each.
SWEEP_DIRS = (
    'orchestrator/src',
    'fused-memory/src',
    'dashboard/src',
    'escalation/src',
    'shared/src',
    'scripts',
    'fused-memory/scripts',
)

#: Anti-vacuity floor on the files swept, ~25% below the 1106 measured on
#: 2026-10-25 (task 4600): a sweep that silently reads nothing must fail rather
#: than report a clean tree.  Raise it as the tree grows; never lower it to
#: silence a failure.
_MIN_SWEPT_FILES = 830


def _swept_files() -> tuple[Path, ...]:
    return python_files(*(REPO_ROOT / rel_dir for rel_dir in SWEEP_DIRS))


def _sweep(module: ParsedModule) -> SweepFindings:
    return sweep_parsed(module.tree, module.source)


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=_swept_files, extract=_sweep)
TREE_ANALYSES = (ANALYSIS,)


def _sweep_findings(tree_pass: TreePass) -> dict[str, SweepFindings]:
    """Return ``{relpath: findings}`` for every swept file with anything to report.

    Shared by the guard and the allow-marker check below so both read the
    same pass — the markers and the sweep they exempt from must never be able
    to disagree about what the tree contains.  Refuses a vacuous sweep first.
    An unparseable file fails the pass loudly, naming it.
    """
    swept = tree_pass.facts(ANALYSIS)
    assert len(swept) >= _MIN_SWEPT_FILES, (
        f'the MCP URL sweep read only {len(swept)} files under {SWEEP_DIRS}, '
        f'expected at least {_MIN_SWEPT_FILES}: a sweep that reads nothing '
        'reports a clean tree.'
    )
    return {
        path.relative_to(REPO_ROOT).as_posix(): findings
        for path, findings in swept.items()
        if findings != SweepFindings()
    }


def test_no_raw_post_builds_a_trailing_slash_mcp_url(tree_pass: TreePass):
    """No source file may build an MCP URL with a trailing slash.

    Implements the "grep the slash, not the flag" rule: the slash is the
    NECESSARY condition for the silent loss, and ``follow_redirects`` only
    determines whether the loss is fatal.  A future site can reintroduce this
    without touching any of the five methods fixed here, so the guard is
    repo-scoped rather than attached to them.

    DELIBERATELY OUT OF SCOPE: FastMCP ``StreamableHttpTransport(f'{base}/mcp/')``
    uses under ``escalation/tests/`` and ``fused-memory/tests/``.  ``/mcp/`` is
    FastMCP's canonical ASGI mount — ``escalation/tests/conftest.py:201``
    explicitly waits on it to prove the app finished mounting — and that client
    follows redirects natively, so those sites carry no silent-discard risk.
    Rewriting live fixtures to chase a cosmetic match would risk breaking
    readiness gating for zero benefit.

    THE SLASH IS NECESSARY, NOT SUFFICIENT, and this guard only covers the
    necessary half.  A sixth site could drop the slash and still lose its
    payload by omitting the ``Accept`` header (the live server answers 406) or
    ``follow_redirects=True``.  Those three parts are held together by
    ``shared.mcp_post.post_mcp_tool_call`` / ``open_mcp_client``, which apply
    them as a unit; this guard is the backstop for a site that bypasses the
    primitive entirely.

    DETECTION IS BY PARSING, NOT SUBSTRING.  This guard originally matched the
    text ``}/mcp/'``, chosen over a bare ``/mcp/`` because that also matches
    the phrase ``scheduler/mcp/usage_gate/cost_store`` in merge_queue.py
    docstrings (:1163, :2467) and would have been permanently unsatisfiable.
    But the narrow anchor bought that immunity by hard-coding one spelling —
    it required a brace immediately before the slash and a single quote after,
    so double-quoted, plain-literal, percent-format and concatenated URLs all
    passed silently.  ``_mcp_url_scan`` matches the parsed literal's TAIL
    instead, which catches every spelling and is still immune to the prose.

    MEASURED both directions, against real historical source: run over
    ``git show 2633a244a6:<path>`` the detector fires on exactly the 8 pre-fix
    defect sites (workflow.py 15173/15215/15247/15286, merge_queue.py 16031,
    and the three operator scripts); run over this tree it flags only three
    deliberate lines, each exempted by an inline
    ``# mcp-url-sweep: allow <reason>`` marker on the line itself.  A guard
    that cannot fail on the defect it was written for is worth nothing.
    """
    offenders = [
        f'{rel}:{lineno}: {text}'
        for rel, findings in sorted(_sweep_findings(tree_pass).items())
        for lineno, text in findings.offenders
    ]

    assert offenders == [], (
        'these build an MCP URL with a trailing slash; the server 307-redirects '
        'it and a bare httpx client discards the payload silently. Use '
        'shared.mcp_post.post_mcp_tool_call (or mcp_endpoint_url) instead. If a '
        'line is genuinely not a fetched URL (an MCP config entry for a '
        'redirect-following client, or a display/log string), put '
        '"# mcp-url-sweep: allow <reason>" on that same line, or after the '
        'closing quotes of a string that spans several lines:\n  '
        + '\n  '.join(offenders)
    )


def test_every_sweep_allow_marker_earns_its_place(tree_pass: TreePass):
    """An allow marker that outlives its offending line must fail, not lurk.

    Each ``# mcp-url-sweep: allow <reason>`` marker is a hole in the guard
    above.  If the line that justified one is fixed, the hole stays open and
    silently covers whatever a future edit puts there.  Requiring every
    marker to still sit on a LIVE hit turns that into a failing test the
    moment it goes stale — the marker gets removed instead of masking a
    regression.

    A malformed marker is reported here too: it exempts nothing, so the guard
    above already fails on its line, and this names the marker as the reason.
    """
    findings_by_file = sorted(_sweep_findings(tree_pass).items())
    stale = [
        f'{rel}:{lineno}: {comment}'
        for rel, findings in findings_by_file
        for lineno, comment in findings.stale_markers
    ]
    malformed = [
        f'{rel}:{lineno}: {comment}'
        for rel, findings in findings_by_file
        for lineno, comment in findings.malformed_markers
    ]

    assert (stale, malformed) == ([], []), (
        'STALE markers sit on a line that no longer builds a slashed MCP URL, so '
        'they now only hide future regressions. Delete them:\n  '
        + '\n  '.join(stale)
        + '\nMALFORMED markers must read "# mcp-url-sweep: allow <reason>" with a '
        'non-empty reason; until they do they exempt nothing:\n  '
        + '\n  '.join(malformed)
    )
