"""RED scaffolding for the CONTRIBUTING.md <-> ``lint_command`` mirror guard (task 3558).

Step-1 of the plan. These cases pin the CONTRACT of the marker extractor
``_documented_lint_command`` before it exists. The full module docstring —
provenance, placement rationale and the out-of-scope note — lands with the
implementation in step-2, and the live drift assertion in step-3.

MEASURED RED at this commit: ``_documented_lint_command`` is undefined, so every
case below fails with ``NameError``. That is the intended RED — the extractor's
behaviour is specified here first.

SYNTHETIC FIXTURES ON PURPOSE. Every case below runs against a hand-written
markdown string, never against the real CONTRIBUTING.md, so these cases stay
stable under any future edit to that file's content. The one test that reads the
real file is the live drift assertion (step-3), which is a different invariant.

The fixtures spell the ``lint-command-mirror`` marker literals out in full rather
than interpolating the implementation's constants. That is deliberate: a rename
of a constant must not be able to silently keep a broken parser agreeing with its
own fixtures.
"""
from __future__ import annotations

import pytest

# (a) Happy path. Modelled on the real CONTRIBUTING.md block: a fenced ```bash
# example ABOVE the marker (backticks outside the slice), and — the measured
# hazard — TWO inline-code spans inside the begin comment's own prose. The real
# marker committed in pre-1 really does carry `ruff check` and `lint_command` in
# its explanatory text, so an extractor that scanned the raw slice for "the one
# backtick span" would pick up comment prose, not the command.
_HAPPY_DOC = """\
- **Tests** run per-package with `pytest`, e.g.:
  ```bash
  cd orchestrator && uv run pytest tests/ --timeout=300
  ```
<!-- lint-command-mirror:begin
     Mirrors the `ruff check` leg of `lint_command` in
     dark-factory-orchestrator.yaml. Pinned by
     tests/scripts/test_contributing_lint_command_drift.py. -->
- **Lint**: `uv run ruff check alpha beta gamma.py`
<!-- lint-command-mirror:end -->
- **Type-check** (pyright), run from each configured package directory.
"""

_HAPPY_COMMAND = "uv run ruff check alpha beta gamma.py"

# (b) No marker at all — someone deleted it, or renamed the file's section.
# Note this doc still CONTAINS a plausible-looking lint bullet: the extractor
# must not fall back to "find something that looks right".
_NO_MARKER_DOC = """\
- **Lint**: `uv run ruff check alpha beta`

Then, before submitting:

2. `uv run ruff check <touched packages>`.
"""

# (c) Two begin markers — e.g. a section duplicated in a bad merge. Picking the
# first silently pins one of two mirrors and lets the other rot unwatched.
_DUPLICATE_MARKER_DOC = """\
<!-- lint-command-mirror:begin -->
- **Lint**: `uv run ruff check alpha beta`
<!-- lint-command-mirror:end -->

## Some later section

<!-- lint-command-mirror:begin -->
- **Lint**: `uv run ruff check delta epsilon`
<!-- lint-command-mirror:end -->
"""

# (d) Decoy immunity. The generic `uv run ruff check <touched packages>` span
# appears BOTH before and after the marker block, so neither a "first span" nor
# a "last span" heuristic can pass this by accident. This is a MEASURED hazard,
# not a hypothetical: CONTRIBUTING.md really carries that generic span in its
# quality-gates list, and it must stay generic and stay unpinned.
_DECOY_DOC = """\
1. `uv run ruff check <touched packages>` — before you start.

<!-- lint-command-mirror:begin
     Mirrors the `ruff check` leg of `lint_command`. -->
- **Lint**: `uv run ruff check alpha beta gamma.py`
<!-- lint-command-mirror:end -->

2. `uv run ruff check <touched packages>` — and again before submitting.
"""


def test_documented_lint_command_extracts_the_marked_span() -> None:
    """(a) The marked inline-code span is returned, markdown and backticks stripped."""
    assert _documented_lint_command(_HAPPY_DOC) == _HAPPY_COMMAND  # noqa: F821


def test_documented_lint_command_ignores_backticks_in_the_marker_prose() -> None:
    """(a') The begin comment's own inline-code spans are not the command.

    Split out from (a) because it is the specific failure an extractor that
    slices on the marker LITERAL (rather than on the end of the begin comment)
    exhibits: it would return ``ruff check`` — the first span of the comment's
    explanatory prose — which is a plausible-looking string, so the mistake
    would not announce itself.
    """
    extracted = _documented_lint_command(_HAPPY_DOC)  # noqa: F821
    assert extracted == _HAPPY_COMMAND
    assert extracted not in ("ruff check", "lint_command")
    assert extracted.startswith("uv run ruff check ")


def test_documented_lint_command_raises_when_the_marker_is_missing() -> None:
    """(b) A missing marker FAILS LOUDLY — never '' or None.

    This is the vacuity hazard, and it is the whole reason the guard exists: an
    extractor that silently returns nothing turns every downstream assertion
    green while pinning nothing at all, which is strictly worse than having no
    guard, because the check still reports success.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_lint_command(_NO_MARKER_DOC)  # noqa: F821

    # The message must tell a human who deleted the marker exactly what to
    # restore and where — the marker literal and the file it belongs in.
    message = str(excinfo.value)
    assert "lint-command-mirror:begin" in message
    assert "CONTRIBUTING.md" in message


def test_documented_lint_command_raises_on_a_duplicated_marker() -> None:
    """(c) Two marker blocks FAIL LOUDLY rather than silently picking the first.

    Silently taking the first would leave the second mirror unpinned and free to
    drift — the exact failure this task closes, reintroduced one level down.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_lint_command(_DUPLICATE_MARKER_DOC)  # noqa: F821

    assert "lint-command-mirror:begin" in str(excinfo.value)


def test_documented_lint_command_is_immune_to_the_generic_ruff_decoy() -> None:
    """(d) Only the MARKED span is returned, never the generic instruction.

    CONTRIBUTING.md's quality-gates list carries a deliberately generic
    ``uv run ruff check <touched packages>``. Pinning that to the live config
    would be wrong twice over: it would fail immediately, and "fixing" it would
    destroy a correct, audience-appropriate instruction.
    """
    assert _documented_lint_command(_DECOY_DOC) == _HAPPY_COMMAND  # noqa: F821
