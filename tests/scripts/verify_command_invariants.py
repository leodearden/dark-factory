"""Shared verify/lint/type command parsing, imported by more than one guard.

This module holds NO test functions of its own. Its own unit tests live in
``test_verify_command_invariants.py`` beside it.

WHY IT EXISTS. A trio of command-parsing helpers — pick the ``&&`` segment that
invokes the checker, extract that checker's positional targets, decide whether a
path is covered by one of them — was hand-maintained in FIVE copies:

  * ``test_root_lint_covers_nonmember_py.py`` (``_ruff_segment`` /
    ``_ruff_targets`` / ``_ruff_exclude_flags`` / ``_is_covered``),
  * ``test_scripts_module_config.py`` (``_segment`` / ``_anchor_split`` /
    ``_targets`` / ``_narrowing_flag_args``),
  * ``test_contributing_lint_command_drift.py`` (``_ruff_segment`` /
    ``_ruff_targets``),
  * ``test_skills_module_config_decision.py`` (``_pytest_segment`` /
    ``_pytest_collected_dirs`` / ``_is_collected``),
  * ``test_fallback_verify_config.py`` (``_lint_leg_targets``).

They had ALREADY DRIFTED, which is the point: the skills guard's own module
docstring named the drift (``_is_covered`` normalising trailing slashes with
``rstrip('/')`` where ``_is_collected`` used ``posixpath.normpath``) and filed
the extraction as [tkt_0RS47G1QXJ5XDPH4T0HKKA1A9S], deferring it only because it
needed edits to sibling guards. Task 4358 then paid for the same defect class a
second time INSIDE one file, where ``_narrowing_flag_args`` had missed a slice
``_targets`` had held from the start. Task 3745 is that extraction, over the
first four; the fifth was outside its fixed scope and was closed separately by
task 3883.

THAT FIFTH COPY IS THE CAUTIONARY ONE. ``_lint_leg_targets`` was the copy whose
prose the OTHER guards cited as canonical for the whole-path-TOKENS rule — so
the file DEFINING the convention was the last one still implementing it
privately, and had drifted from it: its anchor fallback was a raw
``str.endswith``, which is the substring match this module's contract forbids.
Being the documentation home is not the same as being the implementation home,
and the citations now point at :func:`positional_targets` instead.

IMPORT ME, DO NOT COPY ME. The NEXT copy is not a shortcut, it is the next
drift — and drift here does not fail loudly. Every one of these helpers feeds a
coverage assertion, so a copy that silently parses one token differently
degrades to a guard that passes vacuously. If a caller needs behaviour this
module does not have, add a PARAMETER here rather than a variant there —
``path_anchor`` (task 3883) is what that looks like in practice.

THIS MODULE PARSES; IT DOES NOT DECIDE. The five call sites deliberately do not
agree on semantics, and unifying them would be a behaviour change at four of
them. So everything that is POLICY stays with its caller and arrives as an
argument: which flags consume the following token (``_RUFF_FLAGS_TAKING_A_VALUE``
in the CONTRIBUTING guard, ``_PYTEST_VALUE_FLAGS`` in the skills guard, none in
the other three), which flag prefixes narrow a target set
(``_NARROWING_FLAGS``), whether flag scanning covers the whole segment or only
the checker's own post-anchor arguments, and the skills guard's
``--directory`` base resolution / ``posixpath.normpath`` / target-EXISTS layer.
What is shared is only the mechanics of reading a shell command. ``path_anchor``
is NOT an exception to this: it records how the checker was SPELLED on the
command line, which is a fact about the text being parsed, not a policy about
the checker.

Importable from ``tests/scripts/test_*.py`` only because
``tests/scripts/conftest.py`` puts this directory on ``sys.path`` — pytest's
``--import-mode=importlib`` (set in pyproject.toml ``addopts``) deliberately does
not, and there is no ``tests/scripts/__init__.py``. Same mechanism, and same
precedent, as ``systemd_unit_invariants.py``, ``setup_host_parsing.py`` and
``setup_host_sections.py``.
"""
from __future__ import annotations

import pathlib
import shlex

from orchestrator import verify_cmd

# The checker keyword each guard parses, single-sourced HERE rather than
# restated per caller. The keyword is not policy: it is what SELECTS the anchor
# (:func:`anchor_split` takes ``keyword.split()[-1]``), so it is part of this
# module's contract. ``ruff check`` anchors on its ``check`` SUBCOMMAND;
# ``pyright`` and ``pytest`` are invoked as ``<program> <targets>`` with no
# subcommand at all, so each anchors on the program name itself.
#
# Before task 3745's amendment pass the literal ``"ruff check"`` was spelled in
# four files under three names. That is the same N-copy shape the rest of this
# module exists to close: were ruff ever to spell its subcommand differently,
# one edit here beats four coordinated ones, and a caller left behind would not
# fail loudly — it would fail to MATCH a segment and take the exactly-one
# assertion, or match and anchor in the wrong place.
RUFF = "ruff check"
PYRIGHT = "pyright"
PYTEST = "pytest"


def _where(label: str | None, fallback: str) -> str:
    """How a diagnostic names the thing being parsed.

    *label* when the caller supplied one, else the raw text. Callers that parse
    two commands with the SAME keyword need the label to tell the two failures
    apart — ``test_contributing_lint_command_drift.py`` reads both the live
    ``lint_command`` and the one documented in CONTRIBUTING.md.
    """
    return label if label is not None else repr(fallback)


def required_segment(cmd: str, keyword: str, *, label: str | None = None) -> str:
    """The one ``&&``-chained segment of *cmd* that invokes *keyword*.

    Uses the production splitter ``verify_cmd.split_top_level_and`` (quote-aware)
    rather than a naive ``str.split('&&')``. Chaining is an ESTABLISHED pattern
    in this repo's configs, not a hypothetical: every subproject's
    ``lint_command`` chains a ``python3 .../check_*.py <dir>`` gate after
    ``ruff check``. Extracting the checker's own segment FIRST is what keeps the
    target assertions honest — tokenising the whole chain would read ``&&``,
    ``python3`` and the tail gate's own directory arguments as lint targets.

    EXACTLY-ONE, asserted. Two matching segments mean the caller's model of the
    command is wrong, and picking either one would silently assert about half a
    command. Callers for which zero matches is a legitimate answer want
    :func:`optional_token_segment` instead.

    Matched by SUBSTRING, because the keyword may be a phrase (``ruff check``)
    whose tokens are not adjacent to each other in any useful sense here.

    The result is ``.strip()``ed. ``split_top_level_and`` is documented LOSSLESS
    (``'&&'.join(segments) == raw``), so a non-leading segment arrives carrying
    the whitespace that preceded it. Stripping is load-bearing rather than
    cosmetic: ``test_contributing_lint_command_drift.py`` compares this return
    value as a RAW STRING against the command documented in CONTRIBUTING.md, and
    every other caller feeds it straight to ``shlex.split``, for which the strip
    is a no-op.
    """
    segments = verify_cmd.split_top_level_and(cmd)
    matching = [s for s in segments if keyword in s]
    assert len(matching) == 1, (
        f"expected exactly one `{keyword}` segment in {_where(label, cmd)}, got "
        f"{matching!r}; full command: {cmd!r}"
    )
    return matching[0].strip()


def optional_token_segment(cmd: str, keyword: str) -> str | None:
    """The first ``&&``-chained segment of *cmd* whose TOKENS contain *keyword*.

    ``None`` when there is none, and RETURNING ``None`` RATHER THAN ASSERTING is
    the contract, not a convenience (task 3554, measured). A module whose
    ``test_command`` runs something other than pytest contributes no pytest
    targets — that is the correct semantic, not an error. Asserting would make
    the first module to declare ``cargo test`` fail its caller's ratchets with a
    message naming an unrelated module and saying nothing about the invariant
    under test, which invites suppressing the guard instead of fixing anything.
    That module is not hypothetical: ``verify._has_source_files`` already keys on
    ``.py`` AND ``.rs``.

    FIRST match, not exactly-one: the repo-root fleet ``test_command`` is a
    seven-clause ``cd <dir> && uv run pytest ...`` chain, so an exactly-one
    contract would assert on the real command this helper exists to read.

    Matched on the BARE TOKEN after ``shlex.split``, never on a substring: a
    segment mentioning ``pytest-timeout``, or a ``--rootdir=/x/pytest`` value, is
    not a pytest invocation.

    A segment ``shlex`` cannot tokenise (an unbalanced quote) is SKIPPED rather
    than allowed to propagate a bare ``ValueError`` — an unparseable clause is
    one this helper has no opinion about, and a ``ValueError`` from inside a
    coverage assertion says nothing about coverage.

    The returned segment is stripped, matching :func:`required_segment`.
    """
    for segment in verify_cmd.split_top_level_and(cmd):
        try:
            tokens = shlex.split(segment)
        except ValueError:  # unbalanced quotes in a segment — not parseable
            continue
        if keyword in tokens:
            return segment.strip()
    return None


def anchor_split(
    segment: str, keyword: str, *, path_anchor: bool = False, label: str | None = None
) -> tuple[list[str], list[str]]:
    """*segment* split at the checker anchor into ``(pre, post)`` token lists.

    The ANCHOR is the last whitespace-separated token of *keyword*: ``check`` for
    ``ruff check``, ``pyright`` for ``pyright``, ``pytest`` for ``pytest``. One
    rule reproduces every anchor the four callers use, two of which spelled it as
    a hand-rolled ``tokens.index("check")``.

    THE ANCHOR BELONGS TO NEITHER HALF, and the split is the whole point.
    Everything BEFORE it is the WRAPPER's: in
    ``uv run --project shared pyright scripts/`` the first four tokens are uv's,
    and ``--project`` there selects the member ENVIRONMENT the binary resolves
    from. Everything AFTER it is the CHECKER's own argv, where the identically
    spelled ``--project`` would instead redirect pyright's CONFIG FILE. Only
    POSITION tells the two apart, so a caller that reads the whole segment reads
    a category error.

    Sole home for the anchor rule and the anchor-presence assertion. That is
    deliberate: this file's callers already paid once for the alternative, when
    ``test_scripts_module_config.py``'s ``_narrowing_flag_args`` had missed the
    slice its sibling ``_targets`` held from the start (task 4358).

    Takes an already-extracted SEGMENT rather than a whole command, so the
    segment-selection policy stays the caller's choice — :func:`required_segment`
    for the ruff and pyright legs, :func:`optional_token_segment` for pytest.

    A segment ``shlex`` cannot tokenise raises an ``AssertionError`` naming the
    label and the segment, never a bare ``ValueError: No closing quotation``.
    This is the tokenisation choke point for every caller that reaches it
    through :func:`required_segment`, and one of them parses HUMAN-EDITED PROSE:
    ``test_contributing_lint_command_drift.py`` extracts its command from a
    CONTRIBUTING.md bullet, so a stray apostrophe there is ordinary input, not a
    programming error. An unbalanced quote in that bullet must say WHICH command
    would not tokenise — a raw ``ValueError`` traceback names neither the label
    nor the bullet, which is precisely the diagnostic failure this module
    exists to prevent. Note the contrast with
    :func:`optional_token_segment`, which SKIPS such a segment: there, an
    unparseable clause is one among several and the helper has no opinion about
    it; here, the caller has already committed to this segment being the one.

    *path_anchor* additionally accepts the anchor spelled as the PATH the
    checker was INVOKED BY. It exists because a checker need not be on ``PATH``
    at all: the live repo-root ``lint_command`` runs ``python3
    fused-memory/scripts/check_bare_magicmock_config.py <dirs>``, so its anchor
    ``check_bare_magicmock_config.py`` is never a bare token and the exact-token
    rule cannot locate it. That is what kept
    ``test_fallback_verify_config.py::_lint_leg_targets`` on a private copy of
    this parser until task 3883.

    THE DEFAULT IS False, and the four task-3745 callers therefore stay on
    strict exact-token matching — this parameter only ever ADDS candidate
    positions, so no existing caller's parse can change. Pinned rather than
    claimed, by
    ``test_verify_command_invariants.py::test_anchor_split_still_asserts_on_a_path_spelled_anchor_by_default``.

    A WHOLE PATH COMPONENT, never a raw string suffix. ``token.endswith("/" +
    anchor)`` IS the component test for a POSIX shell token, and the distinction
    is not pedantry: a bare ``token.endswith(anchor)`` is a SUBSTRING match, so
    ``scripts/x_check_bare_magicmock_config.py`` — a different file whose name
    merely ends with the anchor — would match and report another program's
    arguments as this checker's targets. "Compare by exact element, never
    substring-match" is the contract this module exists to enforce, and
    ``path_anchor`` must not smuggle in an exception to it. Deliberately NOT
    ``pathlib``/``posixpath``: a normalisation layer would import ``..``
    resolution and absolute-path semantics this parser has no opinion about.
    """
    anchor = keyword.split()[-1]
    try:
        tokens = shlex.split(segment)
    except ValueError as exc:
        raise AssertionError(
            f"cannot tokenise the `{keyword}` segment of {_where(label, segment)}: "
            f"{exc}; segment: {segment!r}"
        ) from exc
    at = None
    for i, token in enumerate(tokens):
        if token == anchor or (path_anchor and token.endswith("/" + anchor)):
            at = i
            break
    assert at is not None, (
        f"no `{anchor}` token in the `{keyword}` segment of "
        f"{_where(label, segment)}, so the checker's own arguments cannot be "
        f"located; segment: {segment!r}"
    )
    return tokens[:at], tokens[at + 1:]


def positional_targets(
    segment: str,
    keyword: str,
    *,
    value_flags: frozenset[str] = frozenset(),
    path_anchor: bool = False,
    label: str | None = None,
) -> list[str]:
    """The positional path arguments *keyword* checks, as whole TOKENS.

    Callers must compare against these by exact element and never substring-match
    the raw command. That rule is backed by a MEASURED counterexample rather than
    a hypothetical: ``'shared' in cmd`` is already TRUE of the live repo-root
    ``lint_command`` via the ``check_bare_magicmock_config.py`` TAIL leg's
    ``shared/tests`` argument, so a substring test passes vacuously for a path
    the ruff leg never checks. Likewise ``'scripts/'`` is a substring of
    ``'tests/scripts/'``, so a copy-pasted sibling command satisfies a naive
    ``'scripts/' in cmd`` for both.

    *value_flags* is the caller's POLICY, and its default is the reason one
    implementation serves four call sites unchanged. With it EMPTY, ``consume``
    can never become True and the loop below reduces byte-for-byte to
    ``[t for t in post if not t.startswith('-')]`` — exactly what
    ``test_root_lint_covers_nonmember_py.py`` and ``test_scripts_module_config.py``
    do today, phantom flag values included. Supplying a set (the CONTRIBUTING
    guard's ``_RUFF_FLAGS_TAKING_A_VALUE``, the skills guard's
    ``_PYTEST_VALUE_FLAGS``) drops the following token instead. The
    ``--flag=value`` spelling needs no entry either way: ``shlex`` keeps it as
    one token and the ``-`` prefix drops it whole.

    WHAT THIS GUARANTEES, stated no more strongly than it holds: every returned
    token was a positional argument after the anchor. It is NOT a proof that no
    unlisted value-taking flag exists — an unknown one still donates its value as
    a PHANTOM target, and nothing here can tell that phantom from a real path.

    A PHANTOM IS NOT UNIFORMLY HARMLESS, and which callers can tolerate one does
    NOT partition by which ones supply a set. Against a COVERAGE check a phantom
    is inert: :func:`covers` over an extra target can only ever pass spuriously,
    never fail. But a caller that ALSO asserts each target exists on disk reads
    the list for exactly what a phantom adds, and goes red naming a flag value as
    a missing path — a misleading diagnosis on a change that broke nothing.
    Measured across the four callers: THREE assert existence
    (``test_root_lint_covers_nonmember_py.py``,
    ``test_contributing_lint_command_drift.py``,
    ``test_skills_module_config_decision.py``) while only two of those three
    supply a set; ``test_scripts_module_config.py`` supplies none and asserts no
    existence. The remaining exposure is root_lint's, and it is recorded on that
    guard's own ``_ruff_targets`` rather than papered over here.

    Discarding unrecognised tokens instead is not an option: that would silently
    shrink the target list and re-open the false-pass hazard the parsing exists
    to close. Widening the set is the caller's call, not this module's.

    *path_anchor* is passed straight through to :func:`anchor_split`, for the
    fifth caller ``test_fallback_verify_config.py::_lint_leg_targets`` (task
    3883), whose ``check_bare_magicmock_config.py`` leg names the checker by
    PATH. It is ORTHOGONAL to *value_flags*, and reading it as another policy
    knob would be a category error: *value_flags* is per-checker POLICY (which
    flags eat the following token — a fact about the checker's CLI), while
    *path_anchor* is a fact about how the checker was SPELLED on this particular
    command line. The same checker invoked by bare name and by path wants the
    same *value_flags* and different *path_anchor*.
    """
    post = anchor_split(segment, keyword, path_anchor=path_anchor, label=label)[1]
    targets: list[str] = []
    consume = False
    for token in post:
        if consume:
            consume = False
            continue
        if token.startswith("-"):
            consume = token in value_flags
            continue
        targets.append(token)
    return targets


def covers(rel_path: str, targets: list[str]) -> bool:
    """True if *rel_path* or one of its ancestor directories is an exact target.

    Ancestor-directory coverage is REAL coverage: ``ruff check skills`` and
    ``pytest tests/scripts/`` both traverse the directory. Membership is tested
    ELEMENT-WISE against the extracted target tokens, never by substring —
    ``'scripts'`` is a substring of ``'tests/scripts/test_x.py'``, so a naive
    containment test reports a file as checked by a command that never sees it,
    and a guard that passes vacuously is the failure mode this whole module
    exists to prevent.

    THE NAME SET IS SLASH-TOLERANT, and that is a decision with a measurement
    behind it rather than an incidental spelling. It unifies two copies that had
    drifted: ``_is_covered`` normalised trailing slashes with ``rstrip('/')``,
    ``_is_collected`` with ``posixpath.normpath``. The tolerant form is a strict
    SUPERSET of the slashless one, so it regresses neither caller:

      * ``covers('a/b.py', ['a/'])`` is True, where the slashless form is False.
        That case is load-bearing for ``test_root_lint_covers_nonmember_py.py``,
        whose targets are RAW command tokens that may still carry the slash the
        author typed.
      * On ``posixpath.normpath``-ed targets — the only kind
        ``test_skills_module_config_decision.py`` ever passes — the two agree
        exactly (measured across 6 rel_paths x 9 target lists, zero
        disagreements; pinned by
        ``test_covers_agrees_with_the_slashless_form_on_normpathed_targets``).

    So do NOT "simplify" this back to the narrower set: the extra spellings are
    the reason one function can serve both callers.

    The root ``'.'`` parent is skipped, so a ``'.'`` target does not vacuously
    cover every path in the repo.
    """
    candidate = pathlib.PurePosixPath(rel_path)
    names = {rel_path, rel_path.rstrip("/"), rel_path + "/"}
    for parent in candidate.parents:
        if parent == pathlib.PurePosixPath("."):
            continue
        names.add(parent.as_posix())
        names.add(parent.as_posix() + "/")
    return any(t in names for t in targets)


def flag_args(tokens: list[str], prefixes: tuple[str, ...]) -> list[str]:
    """Those *tokens* beginning with any of *prefixes*, in order.

    Both spellings are caught by construction: ``--exclude foo`` and
    ``--exclude=foo``. *prefixes* is always supplied by the caller and is never
    hardcoded here — which flags narrow a checker's target set is per-checker
    POLICY (``_NARROWING_FLAGS`` in ``test_scripts_module_config.py``, ruff's
    three exclude spellings in ``test_root_lint_covers_nonmember_py.py``).

    THE SCOPE IS DELIBERATELY THE CALLER'S, which is why this takes an
    already-tokenised LIST rather than a segment. ``uv run --project <member>
    pyright <dir>`` and ``pyright --project <file> <dir>`` contain the same
    characters naming two unrelated things: uv's PRE-anchor ``--project`` selects
    the member ENVIRONMENT the binary resolves from and narrows nothing, while
    pyright's POST-anchor ``--project`` redirects the CONFIG FILE and can relax
    ``typeCheckingMode`` wholesale. Only POSITION distinguishes them.

    The two live callers therefore pass different lists on purpose:
    ``_ruff_exclude_flags`` in ``test_root_lint_covers_nonmember_py.py`` passes
    ``shlex.split(segment)`` — the whole segment — while task 4358 deliberately
    narrowed ``_narrowing_flag_args`` in ``test_scripts_module_config.py`` to
    ``anchor_split(...)[1]``. A helper that took a segment would have to pick one
    scope and would silently regress whichever caller it did not pick, so the
    choice stays visible at the call site.
    """
    return [t for t in tokens if t.startswith(prefixes)]
