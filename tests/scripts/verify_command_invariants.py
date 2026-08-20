"""Shared verify/lint/type command parsing, imported by more than one guard.

This module holds NO test functions of its own. Its own unit tests live in
``test_verify_command_invariants.py`` beside it.

WHY IT EXISTS. A trio of command-parsing helpers — pick the ``&&`` segment that
invokes the checker, extract that checker's positional targets, decide whether a
path is covered by one of them — was hand-maintained in FOUR copies:

  * ``test_root_lint_covers_nonmember_py.py`` (``_ruff_segment`` /
    ``_ruff_targets`` / ``_ruff_exclude_flags`` / ``_is_covered``),
  * ``test_scripts_module_config.py`` (``_segment`` / ``_anchor_split`` /
    ``_targets`` / ``_narrowing_flag_args``),
  * ``test_contributing_lint_command_drift.py`` (``_ruff_segment`` /
    ``_ruff_targets``),
  * ``test_skills_module_config_decision.py`` (``_pytest_segment`` /
    ``_pytest_collected_dirs`` / ``_is_collected``).

They had ALREADY DRIFTED, which is the point: the skills guard's own module
docstring named the drift (``_is_covered`` normalising trailing slashes with
``rstrip('/')`` where ``_is_collected`` used ``posixpath.normpath``) and filed
the extraction as [tkt_0RS47G1QXJ5XDPH4T0HKKA1A9S], deferring it only because it
needed edits to sibling guards. Task 4358 then paid for the same defect class a
second time INSIDE one file, where ``_narrowing_flag_args`` had missed a slice
``_targets`` had held from the start. Task 3745 is that extraction.

IMPORT ME, DO NOT COPY ME. A fifth copy is not a shortcut, it is the next
drift — and drift here does not fail loudly. Every one of these helpers feeds a
coverage assertion, so a copy that silently parses one token differently
degrades to a guard that passes vacuously. If a caller needs behaviour this
module does not have, add a PARAMETER here rather than a variant there.

THIS MODULE PARSES; IT DOES NOT DECIDE. The four call sites deliberately do not
agree on semantics, and unifying them would be a behaviour change at three of
them. So everything that is POLICY stays with its caller and arrives as an
argument: which flags consume the following token (``_RUFF_FLAGS_TAKING_A_VALUE``
in the CONTRIBUTING guard, ``_PYTEST_VALUE_FLAGS`` in the skills guard, neither
in the other two), which flag prefixes narrow a target set
(``_NARROWING_FLAGS``), whether flag scanning covers the whole segment or only
the checker's own post-anchor arguments, and the skills guard's
``--directory`` base resolution / ``posixpath.normpath`` / target-EXISTS layer.
What is shared is only the mechanics of reading a shell command.

Importable from ``tests/scripts/test_*.py`` only because
``tests/scripts/conftest.py`` puts this directory on ``sys.path`` — pytest's
``--import-mode=importlib`` (set in pyproject.toml ``addopts``) deliberately does
not, and there is no ``tests/scripts/__init__.py``. Same mechanism, and same
precedent, as ``systemd_unit_invariants.py``, ``setup_host_parsing.py`` and
``setup_host_sections.py``.
"""
from __future__ import annotations

import shlex

from orchestrator import verify_cmd


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
