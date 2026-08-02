"""Shared test kit: the capacity-failure skip guard.

Canonical home for the capacity-failure detection helper that was previously
copied byte-for-byte into ``test_cli_invoke_integration.py`` (as
``_CAPACITY_FAILURE_MARKERS`` / ``_looks_like_capacity_failure(AgentResult)``)
and ``test_usage_gate.py`` (same list, ``_looks_like_capacity_failure(str)``).
Both copies feed live ``pytest.skip`` call sites in real-CLI integration tests,
so a divergence between them is a test that silently skips — or silently goes
red — on one path and not the other.  Two entry points remain because the call
sites genuinely differ in what they hold: an ``AgentResult`` in the cli_invoke
suite, a pre-combined ``f'{stdout}\\n{stderr}'`` string in the usage-gate probe
guard.  One policy, two calling conventions.

Consumers import by bare module name (``from _capacity_skip import ...``) --
``conftest.py`` prepends ``shared/tests`` to ``sys.path`` (same convention as
``_usage_gate_test_helpers.py``).  The leading underscore keeps pytest from
collecting this module as a test file; its contract tests live in
``test_capacity_skip.py``.
"""

from __future__ import annotations

from shared.cli_invoke import AgentResult

CAPACITY_FAILURE_MARKERS: tuple[str, ...] = (
    ' capped',          # leading space prevents matching 'uncapped' as a false positive
    'rate limit',
    'account unavailable',  # narrowed from bare 'unavailable' to avoid generic network errors
    'out of extra usage',
    'usage limit',
    "you've hit your usage",   # narrowed prefix to avoid matching innocuous "you've hit a snag" phrasing
    "you've used all",         # narrowed prefix to avoid matching innocuous "you've used the wrong format" phrasing
)


def looks_like_capacity_failure(text: str) -> bool:
    """Return True when *text* looks like a Claude CLI capacity / quota failure.

    Case-insensitive substring match of *text* against a small focused list of
    markers drawn from real Claude CLI cap messages.  (The verbatim strings
    those markers were derived from lived in ``shared.usage_gate`` inline
    comments until task 2129 moved the tables to
    ``shared.invocation_outcome`` — ``CAP_HIT_PREFIXES`` /
    ``CAP_CONFIRM_KEYWORDS`` / ``NEAR_CAP_PREFIXES``.)

    **Conservative bias (fail loudly when uncertain).** This helper is used
    at ``pytest.skip`` call sites, so a false positive — skipping on a real
    regression — is the exact failure mode we are trying to prevent. The list
    is therefore intentionally small and obvious; anything not matching a
    well-known capacity signal falls through to an ``assert`` that fails the
    test loudly.

    **Purpose-built list, not an import of the production detector.** The
    production cap detector (``usage_gate.detect_cap_hit``, which delegates to
    ``invocation_outcome.classify_invocation(..., strict_confirm=True)``)
    requires BOTH a prefix AND a confirm-keyword match — a strict combined
    policy designed to avoid marking healthy accounts as capped. Re-using those
    lists here would either collapse the combined check to a loose OR (pulling
    in confirm keywords like ``"resets"`` as standalone signals) or miss real
    cap messages that arrive without the expected prefix. A purpose-built
    substring list is the correct shape for this use-case.
    """
    haystack = text.lower()
    return any(marker in haystack for marker in CAPACITY_FAILURE_MARKERS)


def result_looks_like_capacity_failure(result: AgentResult) -> bool:
    """``looks_like_capacity_failure`` over an ``AgentResult``.

    Inspects both ``result.output`` and ``result.stderr`` — a cap message can
    arrive on either stream depending on how the CLI failed.
    """
    return looks_like_capacity_failure(f'{result.output}\n{result.stderr}')
