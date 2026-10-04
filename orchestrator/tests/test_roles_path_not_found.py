"""Anchor/contract test for `PATH_NOT_FOUND_GUIDANCE` (task 5970, reify census #7920).

The block tells an agent that a not-found error on a path it expected to exist
falsifies the path, so the next call must search for the file by name rather
than hand the same guess to another tool.

Every assertion here is an existence, containment, count or index check against
a NAMED constant, never a prose pin. `_role_splice_contract.py` holds the
rationale for that rule and the mechanical half of each check.
"""

from __future__ import annotations

from _role_splice_contract import MARKDOWN_HEADING, assert_brace_free, assert_nonempty

from orchestrator.agents.path_not_found_guidance import PATH_NOT_FOUND_GUIDANCE


def test_guidance_is_nonempty():
    """The sole guard against every containment test passing on an emptied constant."""
    assert_nonempty(
        'PATH_NOT_FOUND_GUIDANCE',
        PATH_NOT_FOUND_GUIDANCE,
        remedy=(
            'Restore the task-5970 guidance: without it an agent whose expected '
            'path came back not-found retries the same guess with another tool '
            'instead of searching for the file.'
        ),
    )


def test_guidance_is_brace_free():
    """Held brace-free so it stays safe at any future interpolating splice site."""
    assert_brace_free(
        'PATH_NOT_FOUND_GUIDANCE',
        PATH_NOT_FOUND_GUIDANCE,
        remedy='Describe any brace form in words; never spell it out.',
    )


def test_guidance_opens_its_own_section():
    """The block opens with its own ``\\n## `` heading."""
    assert PATH_NOT_FOUND_GUIDANCE.startswith(MARKDOWN_HEADING), (
        'PATH_NOT_FOUND_GUIDANCE does not start with MARKDOWN_HEADING, so where '
        'it is spliced it would read as an unheaded continuation of whichever '
        'block precedes it. Give it a leading blank line and a `## ` heading.'
    )
