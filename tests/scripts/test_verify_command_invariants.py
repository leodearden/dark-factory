"""Unit tests for the shared command-parsing helpers in ``verify_command_invariants``.

The trio these helpers replace was, until task 3745, exercised only INDIRECTLY —
through whatever live config each of the four sibling guards happened to parse
(``test_root_lint_covers_nonmember_py.py``, ``test_scripts_module_config.py``,
``test_contributing_lint_command_drift.py``,
``test_skills_module_config_decision.py``). The single exception was
``test_ruff_targets_reads_flag_values_as_flags_not_paths``. That is a poor net
for a parser: a live command exercises one shape, so the branches that only fire
on a DIFFERENT shape (two matching segments, an absent anchor, an unbalanced
quote, a value-taking flag) were unasserted, and the four copies drifted apart
precisely there.

So the assertions below are on SYNTHETIC commands, deliberately. They pin the
parser's contract independently of what any config currently says, which is what
lets the four guards keep asserting about the live configs without also having to
double as the parser's test suite.

TWO EQUIVALENCES ARE PINNED HERE AS EXECUTABLE STATEMENTS, because they are the
whole reason ONE implementation can serve four call sites that today have
different semantics:

  * ``positional_targets`` with an EMPTY ``value_flags`` set is byte-for-byte the
    naive ``-``-prefix filter that ``test_root_lint_covers_nonmember_py.py`` and
    ``test_scripts_module_config.py`` use — phantoms included. The
    phantom-admitting expectation below is therefore an assertion about what
    those two files do TODAY, not an aspiration.
  * ``covers``' slash-tolerant name set AGREES with the slashless form on
    ``posixpath.normpath``-ed targets — the only kind
    ``test_skills_module_config_decision.py`` ever passes — while slash
    tolerance is genuinely load-bearing for the raw tokens
    ``test_root_lint_covers_nonmember_py.py`` passes.

Flat, namespace-less import: ``tests/scripts/test_pytest_workspace_collection.py``
forbids ``from tests.*``, ``from conftest import`` and package-relative imports
in every tests directory, and ``tests/scripts/conftest.py`` puts this directory
on ``sys.path`` for exactly this reason (pytest's ``--import-mode=importlib``
deliberately does not).

ASSERTION-MESSAGE WORDING IS NOT PINNED, on purpose. The four migrated guards
contain no ``match=`` and exactly one ``pytest.raises``, which targets an
unrelated Markdown extractor — so the diagnostics here are free to be reworded.
What IS pinned is that the message carries the caller's ``label`` and the
offending command, since that is what keeps
``test_contributing_lint_command_drift.py``'s live-vs-documented failures
distinguishable from each other.
"""
from __future__ import annotations

import pytest
import verify_command_invariants as vci

_RUFF = "ruff check"
_PYTEST = "pytest"

# A real-shaped chain: the live repo-root lint_command is a ruff leg followed by
# a `python3 .../check_bare_magicmock_config.py <dir>` gate, and the tail leg's
# own directory arguments are exactly what a whole-string tokenisation would
# misread as ruff lint targets.
_CHAINED_LINT = (
    "uv run ruff check alpha beta && "
    "python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests"
)


# ---------------------------------------------------------------------------
# required_segment
# ---------------------------------------------------------------------------


def test_required_segment_returns_only_the_matching_leg_of_an_and_chain() -> None:
    """The tail leg's arguments must not reach the caller as the checker's."""
    assert vci.required_segment(_CHAINED_LINT, _RUFF) == "uv run ruff check alpha beta"


def test_required_segment_strips_the_leading_space_the_splitter_preserves() -> None:
    """The returned segment is STRIPPED, and that is load-bearing, not cosmetic.

    ``verify_cmd.split_top_level_and`` returns segments VERBATIM by documented
    contract (``'&&'.join(segments) == raw``), so the second leg of an ``&&``
    chain arrives carrying the space that preceded it.
    ``test_contributing_lint_command_drift.py`` compares its live segment as a
    RAW STRING against the command documented in CONTRIBUTING.md, so an
    unstripped return makes that mirror fail on whitespace alone.
    """
    assert vci.required_segment("ls a && ruff check x", _RUFF) == "ruff check x"


def test_required_segment_asserts_when_no_segment_matches() -> None:
    with pytest.raises(AssertionError):
        vci.required_segment("python3 fused-memory/scripts/check.py shared/tests", _RUFF)


def test_required_segment_asserts_when_two_segments_match() -> None:
    """Exactly-one, not first-match: two ruff legs mean the caller's model is wrong."""
    with pytest.raises(AssertionError):
        vci.required_segment("ruff check a && ruff check b", _RUFF)


def test_required_segment_assertion_message_names_the_label_and_the_command() -> None:
    """``label`` distinguishes otherwise-identical live and documented failures.

    ``test_contributing_lint_command_drift.py`` parses TWO commands with the same
    keyword — the live ``lint_command`` and the one documented in CONTRIBUTING.md
    — so a message that named neither would leave a reader unable to tell which
    side of the mirror broke.
    """
    with pytest.raises(AssertionError) as excinfo:
        vci.required_segment("echo nothing here", _RUFF, label="the documented command")
    message = str(excinfo.value)
    assert "the documented command" in message
    assert "echo nothing here" in message


# ---------------------------------------------------------------------------
# optional_token_segment
# ---------------------------------------------------------------------------


def test_optional_token_segment_matches_the_bare_token_not_a_substring() -> None:
    """``pytest-timeout`` and ``--rootdir=/x/pytest`` are not pytest invocations."""
    assert vci.optional_token_segment("uv run pytest-timeout --rootdir=/x/pytest a", _PYTEST) is None


def test_optional_token_segment_returns_the_first_match_in_a_multi_clause_chain() -> None:
    """First-match, NOT exactly-one — the repo-root fleet test_command really is a chain.

    It is a seven-clause ``cd <dir> && uv run pytest ...``, so an exactly-one
    contract would assert on the real command this helper exists to read.
    """
    chain = (
        "cd escalation && uv run pytest tests/ -q && "
        "cd ../orchestrator && uv run pytest tests/ -q"
    )
    assert vci.optional_token_segment(chain, _PYTEST) == "uv run pytest tests/ -q"


def test_optional_token_segment_returns_none_rather_than_asserting() -> None:
    """The measured task-3554 contract: a non-pytest command contributes no targets.

    Asserting instead would make the first module to declare ``cargo test`` fail
    its caller's ratchets with a message naming an unrelated module — which
    invites suppressing the guard rather than fixing anything.
    """
    assert vci.optional_token_segment("cargo test --workspace", _PYTEST) is None


def test_optional_token_segment_skips_a_segment_it_cannot_tokenise() -> None:
    """An unbalanced quote is skipped, not propagated as a bare ValueError.

    MEASURED shape, not invented: ``split_top_level_and`` is quote-aware, so an
    unterminated quote swallows every later ``&&`` into its own segment — the
    second segment below is ``" foo 'bar && uv run pytest tests/"``, which
    ``shlex.split`` rejects with ``No closing quotation``.
    """
    assert vci.optional_token_segment("echo && foo 'bar && uv run pytest tests/", _PYTEST) is None


def test_optional_token_segment_strips_the_verbatim_segment() -> None:
    assert vci.optional_token_segment("ls a && uv run pytest tests/", _PYTEST) == (
        "uv run pytest tests/"
    )
