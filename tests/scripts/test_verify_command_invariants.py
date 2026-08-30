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

import pathlib
import posixpath
import shlex

import pytest
import verify_command_invariants as vci

# Spelled as LITERALS on purpose, even though ``vci`` now exports the same three
# strings. This file is the module's oracle, so it must be able to disagree with
# it; ``test_exported_keywords_are_the_strings_the_guards_parse`` below is the
# one place the two are compared, and it is what makes the guards' aliasing of
# ``vci.RUFF`` / ``vci.PYRIGHT`` / ``vci.PYTEST`` a checked contract rather than
# a convention.
_RUFF = "ruff check"
_PYRIGHT = "pyright"
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
# exported keywords
# ---------------------------------------------------------------------------


def test_exported_keywords_are_the_strings_the_guards_parse() -> None:
    """The four guards alias these rather than restating the literal.

    The keyword is not per-caller policy — it is what SELECTS the anchor
    (``keyword.split()[-1]``), so it belongs to the shared contract. Before the
    task-3745 amendment pass ``"ruff check"`` was spelled in four files under
    three names (``_RUFF_KEYWORD`` twice, ``_RUFF`` once), which is the same
    N-copy shape this module exists to close.

    Compared against this file's own literals, which are deliberately NOT
    aliases: an edit to the exported constant that the guards silently inherit
    surfaces here as a disagreement with the oracle every other test uses.
    """
    assert (vci.RUFF, vci.PYRIGHT, vci.PYTEST) == (_RUFF, _PYRIGHT, _PYTEST)


def test_exported_keywords_select_the_anchor_each_guard_needs() -> None:
    """Why the keyword is shared: ruff anchors on a SUBCOMMAND, the others do not.

    ``ruff check <targets>`` puts the anchor one token past the program name,
    while ``pyright <targets>`` and ``pytest <targets>`` anchor on the program
    name itself. One rule over these three constants reproduces all three.
    """
    assert [k.split()[-1] for k in (vci.RUFF, vci.PYRIGHT, vci.PYTEST)] == [
        "check",
        "pyright",
        "pytest",
    ]


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


# ---------------------------------------------------------------------------
# anchor_split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("segment", "keyword", "pre", "post"),
    [
        ("uv run ruff check alpha beta", _RUFF, ["uv", "run", "ruff"], ["alpha", "beta"]),
        (
            "uv run --project shared pyright scripts/",
            "pyright",
            ["uv", "run", "--project", "shared"],
            ["scripts/"],
        ),
        (
            "uv run --directory orchestrator pytest tests/ -q",
            _PYTEST,
            ["uv", "run", "--directory", "orchestrator"],
            ["tests/", "-q"],
        ),
    ],
)
def test_anchor_split_derives_the_anchor_from_the_keywords_last_token(
    segment: str, keyword: str, pre: list[str], post: list[str]
) -> None:
    """One rule — ``keyword.split()[-1]`` — reproduces all three live anchors.

    ``ruff check`` anchors on ``check``, ``pyright`` and ``pytest`` on
    themselves. Before this module the three were spelled three ways, including
    two hand-rolled ``tokens.index("check")`` calls, which is how a caller ends
    up with a slice a sibling caller does not have (task 4358).

    The anchor belongs to NEITHER half: everything before it is the WRAPPER's
    (``uv``'s ``--project shared`` selects the ENVIRONMENT the binary resolves
    from), everything after is the CHECKER's own argv.
    """
    assert vci.anchor_split(segment, keyword) == (pre, post)


def test_anchor_split_asserts_when_the_anchor_is_absent() -> None:
    """Without the anchor the checker's own arguments cannot be located at all.

    A bare ``ValueError`` from ``list.index`` would say so far less clearly, and
    would not name which command failed.
    """
    with pytest.raises(AssertionError) as excinfo:
        vci.anchor_split("uv run ruff format alpha", _RUFF, label="the live lint_command")
    message = str(excinfo.value)
    assert "the live lint_command" in message
    assert "uv run ruff format alpha" in message


def test_anchor_split_reports_an_untokenisable_segment_with_its_label() -> None:
    """An unbalanced quote is an AssertionError naming the label, not a ValueError.

    Unlike ``optional_token_segment``, which SKIPS a clause it cannot tokenise,
    this helper's caller has already committed to this segment being the one — so
    the failure must say WHICH command would not parse. That matters because one
    caller parses human-edited prose: ``test_contributing_lint_command_drift.py``
    extracts its command from a CONTRIBUTING.md bullet, where a stray apostrophe
    is ordinary input rather than a programming error, and a bare
    ``ValueError: No closing quotation`` names neither the label nor the bullet.
    """
    with pytest.raises(AssertionError) as excinfo:
        vci.anchor_split("uv run ruff check alpha'", _RUFF, label="the documented command")
    message = str(excinfo.value)
    assert "the documented command" in message
    assert "uv run ruff check alpha'" in message


def test_anchor_split_locates_a_path_spelled_anchor() -> None:
    """``path_anchor=True`` finds an anchor spelled as the PATH it was invoked by.

    This is the shape the live repo-root ``lint_command``'s tail leg actually
    has: the magicmock checker is run as ``python3
    fused-memory/scripts/check_bare_magicmock_config.py <dirs>``, so the anchor
    ``check_bare_magicmock_config.py`` is never a bare token. Without this
    parameter the exact-token rule cannot locate it at all, which is what kept
    ``test_fallback_verify_config.py::_lint_leg_targets`` on a private copy of
    this parser (task 3883).
    """
    segment = "python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests"
    assert vci.anchor_split(segment, "check_bare_magicmock_config.py", path_anchor=True) == (
        ["python3"],
        ["shared/tests"],
    )


def test_anchor_split_still_asserts_on_a_path_spelled_anchor_by_default() -> None:
    """The DEFAULT is unchanged: a path-spelled anchor is not an exact token.

    Load-bearing rather than incidental. ``path_anchor`` only ever ADDS
    candidate positions, so pinning that the default rejects the path spelling
    is what guarantees task 3745's four callers cannot be silently widened by
    the fifth caller's needs.
    """
    segment = "python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests"
    with pytest.raises(AssertionError):
        vci.anchor_split(segment, "check_bare_magicmock_config.py")


def test_anchor_split_path_anchor_matches_a_whole_component_not_a_suffix() -> None:
    """A whole path COMPONENT, never a raw string suffix.

    ``scripts/x_check_bare_magicmock_config.py`` is a DIFFERENT file whose name
    merely ends with the anchor, so a raw ``str.endswith`` would report another
    program's arguments as this checker's targets. "Compare by exact element,
    never substring-match" is the contract this whole module exists to enforce,
    and ``path_anchor`` must not smuggle in an exception to it.
    """
    with pytest.raises(AssertionError):
        vci.anchor_split(
            "python3 scripts/x_check_bare_magicmock_config.py a b",
            "check_bare_magicmock_config.py",
            path_anchor=True,
        )


def test_positional_targets_propagates_the_tokenisation_failure() -> None:
    """The guard is at the choke point, so every caller reaching it inherits it."""
    with pytest.raises(AssertionError) as excinfo:
        vci.positional_targets("ruff check 'alpha", _RUFF, label="the live lint_command")
    assert "the live lint_command" in str(excinfo.value)


# ---------------------------------------------------------------------------
# positional_targets
# ---------------------------------------------------------------------------


def test_positional_targets_excludes_everything_before_the_anchor() -> None:
    """uv's positional ``shared`` is an environment name, not a checked path.

    This is the documented reason the extractor anchors where it does rather
    than filtering the whole segment.
    """
    assert vci.positional_targets("uv run --project shared pytest tests/", _PYTEST) == ["tests/"]


def test_positional_targets_with_no_value_flags_is_the_naive_dash_prefix_filter() -> None:
    """THE EQUIVALENCE that lets one extractor serve all four call sites.

    ``test_root_lint_covers_nonmember_py.py`` and ``test_scripts_module_config.py``
    both use a bare ``[t for t in tail if not t.startswith('-')]`` today, phantoms
    and all: ``--select E,F`` donates ``E,F`` and ``--line-length 100`` donates
    ``100`` as though they were paths. With an empty ``value_flags`` set the
    consume-next flag can never become True, so the loop reduces to exactly that
    filter — and asserting the PHANTOM-ADMITTING result here is what proves the
    migration changed nothing for those two files.

    What a phantom then COSTS its caller does not partition by which callers
    supply a set. Against a coverage check it is inert (an extra target can only
    make ``covers`` pass spuriously, never fail), but against a target-EXISTS
    assertion it is a misleading red naming a flag value as a missing path — and
    ``test_root_lint_covers_nonmember_py.py`` both omits the set AND asserts
    existence. Preserving its behaviour bit-for-bit was this task's requirement;
    the residual exposure is recorded on that guard's own ``_ruff_targets``, not
    resolved here.
    """
    segment = "ruff check --select E,F --line-length 100 --fix alpha beta.py"
    assert vci.positional_targets(segment, _RUFF) == ["E,F", "100", "alpha", "beta.py"]


@pytest.mark.parametrize(
    ("segment", "keyword"),
    [
        ("ruff check alpha beta.py", _RUFF),
        ("uv run ruff check --fix alpha", _RUFF),
        ("ruff check --select E,F --line-length 100 --fix alpha beta.py", _RUFF),
        ("uv run --project shared pyright scripts/", "pyright"),
        ("uv run --project shared pytest tests/scripts/ --tb=short -q --timeout=300", _PYTEST),
        ("uv run --directory orchestrator pytest tests/ -k some_expr", _PYTEST),
        ("pytest", _PYTEST),
    ],
)
def test_positional_targets_agrees_with_the_pre_migration_filter(segment: str, keyword: str) -> None:
    """The same equivalence, checked against an INDEPENDENT oracle over many shapes.

    The right-hand side is the pre-migration expression transcribed literally
    from ``_ruff_targets`` / ``_targets``, not a call back into this module, so
    the two can genuinely disagree.
    """
    tokens = shlex.split(segment)
    anchor = keyword.split()[-1]
    naive = [t for t in tokens[tokens.index(anchor) + 1:] if not t.startswith("-")]
    assert vci.positional_targets(segment, keyword) == naive


def test_positional_targets_consumes_a_supplied_value_flags_following_token() -> None:
    """With the caller's value-flag set supplied, a flag VALUE is not a target.

    Byte-identical to ``test_contributing_lint_command_drift.py``'s existing
    ``test_ruff_targets_reads_flag_values_as_flags_not_paths`` expectation. That
    guard needs this because it reports a target that does not exist on disk, so
    a phantom ``E,F`` surfaces as "names 'E,F', which does not exist" — a red
    verify with a misleading diagnosis on a change that broke nothing.
    """
    segment = "ruff check --select E,F --line-length 100 --fix alpha beta.py"
    targets = vci.positional_targets(
        segment, _RUFF, value_flags=frozenset({"--select", "--line-length"})
    )
    assert targets == ["alpha", "beta.py"]


def test_positional_targets_needs_no_entry_for_the_equals_spelling() -> None:
    """``--flag=value`` is one shlex token and the ``-`` prefix drops it whole."""
    segment = "uv run --project shared pytest tests/scripts/ --timeout=300 -q"
    assert vci.positional_targets(segment, _PYTEST) == ["tests/scripts/"]


def test_positional_targets_threads_path_anchor_to_the_anchor_split() -> None:
    """The fifth caller reaches ``anchor_split``'s new parameter through here.

    Over ``_CHAINED_LINT``, already documented above as modelling the live
    repo-root ``lint_command`` — a ruff leg followed by a ``python3
    .../check_bare_magicmock_config.py <dir>`` gate whose checker is named by
    PATH. Reusing that fixture rather than inventing one keeps the pin tied to
    the shape the live command actually has.
    """
    segment = vci.required_segment(_CHAINED_LINT, "check_bare_magicmock_config.py")
    targets = vci.positional_targets(
        segment, "check_bare_magicmock_config.py", path_anchor=True
    )
    assert targets == ["shared/tests"]


def test_positional_targets_default_still_rejects_a_path_spelled_anchor() -> None:
    """The default is pinned at THIS layer too, not only at ``anchor_split``.

    ``positional_targets`` is what the four task-3745 callers actually call, so
    a default that leaked here would widen them even with ``anchor_split``'s own
    default intact.
    """
    segment = vci.required_segment(_CHAINED_LINT, "check_bare_magicmock_config.py")
    with pytest.raises(AssertionError):
        vci.positional_targets(segment, "check_bare_magicmock_config.py")


def test_positional_targets_propagates_the_anchor_assertion_with_the_label() -> None:
    with pytest.raises(AssertionError) as excinfo:
        vci.positional_targets("uv run ruff format alpha", _RUFF, label="the documented command")
    assert "the documented command" in str(excinfo.value)


# ---------------------------------------------------------------------------
# covers
# ---------------------------------------------------------------------------


def _slashless_is_collected(rel_path: str, targets: list[str]) -> bool:
    """``_is_collected`` from ``test_skills_module_config_decision.py``, transcribed.

    An INDEPENDENT oracle for the agreement check below — deliberately a literal
    copy of the pre-migration body rather than a call into the module under test,
    so the two can genuinely disagree.
    """
    candidate = pathlib.PurePosixPath(rel_path)
    names = {rel_path}
    for parent in candidate.parents:
        if parent == pathlib.PurePosixPath("."):
            continue
        names.add(parent.as_posix())
    return any(t in names for t in targets)


def test_covers_matches_an_exact_target() -> None:
    assert vci.covers("conftest.py", ["conftest.py", "df_pytest_isolation.py"])


def test_covers_treats_an_ancestor_directory_as_real_coverage() -> None:
    """``ruff check skills`` and ``pytest tests/scripts/`` both traverse the directory."""
    assert vci.covers("skills/factory-init/scripts/find_escalation_port.py", ["skills"])


def test_covers_is_exact_element_never_substring() -> None:
    """THE property the whole trio exists to provide, and the one ``in cmd`` destroys.

    ``'scripts'`` is a substring of ``'tests/scripts/test_x.py'``, so a naive
    containment test would report a file as linted by a command that never sees
    it — a guard that passes vacuously, which is the silent failure direction.
    """
    assert not vci.covers("tests/scripts/test_x.py", ["scripts"])


def test_covers_tolerates_a_trailing_slash_on_the_target() -> None:
    """Slash tolerance is ``test_root_lint_covers_nonmember_py.py``'s requirement.

    Its targets are RAW command tokens, so ``ruff check a/`` yields the target
    ``'a/'`` with the slash still attached. The slashless form (which the skills
    guard can afford, because ``posixpath.normpath`` has already stripped it)
    returns False here — so the slash-TOLERANT form is the one that must survive
    the unification, and this pair is the measurement that says so.
    """
    assert vci.covers("a/b.py", ["a/"])
    assert not _slashless_is_collected("a/b.py", ["a/"])


def test_covers_does_not_let_the_root_parent_cover_everything() -> None:
    """``PurePosixPath('a/b.py').parents`` ends at ``'.'``, which is skipped.

    Without the skip a ``.`` target — or any command run from the repo root that
    happened to name it — would vacuously cover every path in the repo.
    """
    assert not vci.covers("a/b.py", ["."])


@pytest.mark.parametrize(
    "rel_path",
    [
        "conftest.py",
        "tests/scripts/test_x.py",
        "skills/factory-init/scripts/find_escalation_port.py",
        "orchestrator/tests/test_verify.py",
        "a/b.py",
        "scripts/legibility/codebook.py",
    ],
)
@pytest.mark.parametrize(
    "targets",
    [
        [],
        ["tests/scripts"],
        ["tests"],
        ["skills"],
        ["conftest.py"],
        ["orchestrator/tests"],
        ["scripts"],
        ["tests/scripts", "skills", "conftest.py"],
        ["does/not/match"],
    ],
)
def test_covers_agrees_with_the_slashless_form_on_normpathed_targets(
    rel_path: str, targets: list[str]
) -> None:
    """THE SUPERSET EQUIVALENCE that lets one function replace both copies.

    ``test_skills_module_config_decision.py`` only ever passes targets that have
    been through ``posixpath.normpath``, so they carry no trailing slash. On
    exactly that input the slash-tolerant name set and the slashless one agree,
    while the tolerant one additionally handles the raw tokens
    ``test_root_lint_covers_nonmember_py.py`` passes. Superset, not replacement:
    neither caller regresses.

    Every target below is already normpath-stable, which is what makes this an
    assertion about the inputs the skills guard actually produces rather than
    about arbitrary strings (``['a/']`` disagrees, on purpose — see the trailing
    slash test above).
    """
    assert all(posixpath.normpath(t) == t for t in targets), "oracle inputs must be normpath-stable"
    assert vci.covers(rel_path, targets) == _slashless_is_collected(rel_path, targets)


# ---------------------------------------------------------------------------
# flag_args
# ---------------------------------------------------------------------------


def test_flag_args_catches_both_the_spaced_and_equals_spellings() -> None:
    tokens = shlex.split("ruff check a --exclude b --extend-exclude=c --force-exclude")
    prefixes = ("--exclude", "--extend-exclude", "--force-exclude")
    assert vci.flag_args(tokens, prefixes) == ["--exclude", "--extend-exclude=c", "--force-exclude"]


def test_flag_args_returns_empty_when_no_flag_is_present() -> None:
    tokens = shlex.split("uv run ruff check scripts/ tests/scripts/")
    assert vci.flag_args(tokens, ("--exclude", "--extend-exclude", "--force-exclude")) == []


def test_flag_args_scope_is_the_callers_choice_not_a_default() -> None:
    """THE executable statement of why this takes TOKENS rather than a segment.

    ``uv run --project <member> pyright <dir>`` and ``pyright --project <file>
    <dir>`` contain the SAME CHARACTERS naming two unrelated things: uv's
    PRE-anchor ``--project`` selects the member ENVIRONMENT and narrows nothing,
    while pyright's POST-anchor ``--project`` redirects the CONFIG FILE and can
    relax ``typeCheckingMode`` wholesale. Only POSITION distinguishes them.

    So the two live callers pass different token lists ON PURPOSE:
    ``test_root_lint_covers_nonmember_py.py``'s ``_ruff_exclude_flags`` scans the
    WHOLE segment, and task 4358 deliberately narrowed
    ``test_scripts_module_config.py``'s ``_narrowing_flag_args`` to the
    post-anchor slice. A shared helper that took a segment would have to pick
    one scope and would regress whichever caller it did not pick.
    """
    segment = "uv run --project shared pyright scripts/"
    prefixes = ("--skip", "-p", "--project")
    assert vci.flag_args(shlex.split(segment), prefixes) == ["--project"]
    assert vci.flag_args(vci.anchor_split(segment, "pyright")[1], prefixes) == []
