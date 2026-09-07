"""Drift guard over the merge-state vocabularies' hand-maintained prose copies.

Task 4829 (PRD ``plans/merge-status-durable-non-landed-prd.md``, decision D3,
behaviour B9). ``shared/src/shared/merge_state.py`` is the single normative copy
of the two merge wire vocabularies — ``MergeState`` (the POLL vocabulary of
``merge_status`` / ``merge_cancel``'s ``state``) and ``MergeSubmitStatus`` (the
SUBMIT vocabulary of ``merge_request``'s ``status``). Several committed
artifacts nonetheless restate parts of those vocabularies inline, and none of
them auto-extends when a member lands.

MEASURED, not predicted. Every restatement below was read off this tree before
this guard existed:

  * ``skills/escalation-watcher/SKILL.md`` documented the POLL terminal set as
    ``done | conflict | blocked | already_merged`` — three defects in one line.
    ``already_merged`` is SUBMIT-only (``_map_terminal_state`` collapses it to
    ``done``, as that same file says forty lines further down), and ``abandoned``
    and ``superseded`` were both missing even though the same file documents
    ``merge_status`` returning ``state: 'abandoned'``. The file contradicted
    itself.
  * ``escalation/src/escalation/server.py``'s ``merge_cancel`` docstring
    enumerated five states for a function that demonstrably returns six
    (``superseded`` was missing).
  * ``skills/merge-queue/SKILL.md`` and ``skills/unblock-low-risk/SKILL.md``
    both named a submit status ``failed`` that appears NOWHERE in the server
    (the real value is ``error``), while omitting six real ones.

This module derives both vocabularies ONCE from the normative module and
cross-checks every restatement against it. It never stores its own snapshot of
the members: a hardcoded list here would be one more lock-step copy, stale on
the next member exactly like the prose sites were.

THE PINNED SITES live in ``PINNED_SITES``. Each carries its value list INLINE —
a runbook an agent executes at run time must SHOW the values, not point at a
Python file — wrapped in a ``merge-state-vocab`` marked span naming the
partition the list must equal. Same convention as ``CONTRIBUTING.md``'s
``lint-command-mirror`` block and ``skills/prd/references/gates.md``'s
``inv-trigger-shapes`` span.

WHAT THIS GUARD DELIBERATELY DOES NOT PIN.

  * ARM RULES — a loop or branch that deliberately handles a NARROWED subset of
    a vocabulary. ``skills/merge-queue/SKILL.md``'s resumed-poll loop (which
    drops ``superseded`` from its terminal set on purpose) and
    ``skills/unblock/SKILL.md``'s ``terminal_resumed`` tuple are rules about
    which states a particular arm may stop on, not copies of a vocabulary.
    Pinning them to a partition would demand they re-add the very member they
    exist to exclude. PRD delta owns them.
  * ``skills/orchestrate/SKILL.md``'s halted-merge row (see
    ``_UNPINNED_RULE_SITES``) — the same shape, one file wider: it enumerates
    the five statuses ``orchestrator/src/orchestrator/merge_queue.py``'s
    ``_map_advance_failure`` can return that halt the queue, a rule owned by the
    orchestrator rather than a copy of ``SUBMIT_TERMINAL``.
  * REASONING PROSE around each list — which state means what, how to react.
    PRD delta rewrites it, and pinning wording would go red on an editorial
    edit while a genuinely stale VALUE list stayed green.
  * The mapping between the two vocabularies. That belongs to
    ``escalation/src/escalation/server.py::_map_terminal_state``; a copy here
    would be exactly the second home the normative module exists to remove.

PLACEMENT IS LOAD-BEARING. ``scripts/tests/`` modules must import NO first-party
package — that is what lets ``uv run --project shared pytest scripts/tests/``
(``scripts/orchestrator.yaml``'s ``test_command``) satisfy them on a freshly
synced verify worktree. This module is stdlib-only (``importlib.util``, ``os``,
``re``, ``subprocess``, ``pathlib``) plus ``pytest``, and it loads
``shared/src/shared/merge_state.py`` BY ABSOLUTE PATH rather than importing
``shared.merge_state``. That second point is load-bearing twice over: an agent
Bash session typically inherits the MAIN checkout's ``VIRTUAL_ENV`` (see
``CLAUDE.md``, "Locating installed code"), so a plain import could pin THIS
worktree's SKILL.md files against MAIN's vocabulary.

EXTRACTOR CONTRACT. Every extractor below raises a loud ``AssertionError``
naming its ``source`` rather than returning an empty result. An extractor that
silently yields nothing turns every downstream drift assertion green while
pinning nothing at all — strictly worse than no guard, because the check still
reports success. Extractors are unit-tested against HAND-WRITTEN fixture text,
never the live artifacts, so those tests stay stable under any future edit; the
live assertions re-read every committed artifact fresh.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Hand-written fixture text. NEVER the live artifacts: an extractor unit test
# that read a real SKILL.md would go red on any future edit to that file, which
# trains readers to edit the guard instead of the drift.
# ---------------------------------------------------------------------------

_FIXTURE_SOURCE = "fixture-site.md"

# The markdown carrier, with the marker form step-10 writes into the SKILL.md
# files: a multi-line begin comment carrying the source-of-truth pointer, and a
# bare end comment. Two decoys are deliberate — the begin comment names the
# partition (a token an over-broad span would swallow), and the trailing prose
# names `already_merged`, a real vocabulary member that sits OUTSIDE the span.
_MARKDOWN_SPAN = """\
# fixture site

<!-- merge-state-vocab:begin partition=TERMINAL_STATES
     Mirrors shared/src/shared/merge_state.py::TERMINAL_STATES. Pinned by
     scripts/tests/test_merge_state_vocabulary_consistency.py — extend the enum
     and this line goes red until it matches. -->
**Terminal states** (`done`, `conflict`, `blocked`, `abandoned`, `superseded`) — stop polling.
<!-- merge-state-vocab:end -->

Elsewhere in the file, outside every span, the prose discusses `already_merged`.
"""

# The Python-docstring carrier: bare marker lines inside a docstring, where an
# HTML comment would be nonsense. This is the shape
# `escalation/src/escalation/server.py::merge_cancel` carries.
_DOCSTRING_SPAN = '''\
def merge_cancel() -> dict:
    """Cancel a pending merge request.

    Returns:
        state (str) — Coarse terminal state; always a member of
                      ``shared.merge_state.MergeState``.
                      merge-state-vocab:begin partition=CANCEL_STATES
                      'done' | 'conflict' | 'blocked' | 'abandoned' | 'superseded' | 'unknown'
                      merge-state-vocab:end
                      Source of truth: shared/src/shared/merge_state.py::CANCEL_STATES.
    """
'''

# The third literal style, and the single-line begin-comment form: bare set
# notation inside a fenced block, markers outside the fence.
_SET_NOTATION_SPAN = """\
<!-- merge-state-vocab:begin partition=LIVE_STATES -->
```python
LIVE = {"queued", "verifying", "gate", "finalizing"}
```
<!-- merge-state-vocab:end -->
"""

_TWO_SPANS = """\
<!-- merge-state-vocab:begin partition=LIVE_STATES -->
Live: `queued`, `verifying`, `gate`, `finalizing`
<!-- merge-state-vocab:end -->

<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""

# ── seeded violations ──────────────────────────────────────────────────────
# Each is a shape a prose list actually developed on this tree, or the shape a
# future edit would develop. B9 binds the REJECTION, so each must be OBSERVED
# to fail rather than asserted by reading the code.

# Today's `skills/escalation-watcher/SKILL.md` line, defect for defect.
_SPAN_MISSING_MEMBER = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`
<!-- merge-state-vocab:end -->
"""

_SPAN_EXTRA_MEMBER = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`, `already_merged`
<!-- merge-state-vocab:end -->
"""

# A value from the OTHER vocabulary. `wip_halted` is SUBMIT-only, so a POLL span
# naming it is claiming `merge_status` can return it — it cannot.
_SPAN_WRONG_VOCABULARY = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`, `wip_halted`
<!-- merge-state-vocab:end -->
"""

_SPAN_ABSENT = """\
# fixture site

A registered file whose span was deleted in an edit. It still discusses
`done`, `conflict` and `superseded` in prose, so a content heuristic would
think it was fine.
"""

_SPAN_UNTERMINATED = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
"""

_SPAN_NESTED = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`
<!-- merge-state-vocab:begin partition=LIVE_STATES -->
Live: `queued`, `verifying`, `gate`, `finalizing`
<!-- merge-state-vocab:end -->
"""

_SPAN_UNOPENED_END = """\
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""

_SPAN_UNKNOWN_PARTITION = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATUSES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""

_SPAN_NO_PARTITION = """\
<!-- merge-state-vocab:begin -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""


# ---------------------------------------------------------------------------
# load_vocabulary
# ---------------------------------------------------------------------------


def test_load_vocabulary_exposes_every_partition_the_spans_can_name() -> None:
    """The loader hands back every partition a `partition=` marker may cite.

    Named explicitly rather than derived from the module's `__all__`: the point
    of the guard is that a prose span can only pin a partition this module
    knows how to compare, so the set of citable names is part of the contract,
    not an accident of what the vocabulary module happens to export.
    """
    partitions = load_vocabulary()

    assert set(partitions) == {
        "LIVE_STATES",
        "TERMINAL_STATES",
        "OUTCOME_STATES",
        "EPISTEMIC_STATES",
        "POLL_STOP_STATES",
        "CANCEL_STATES",
        "SUBMIT_TERMINAL",
        "SUBMIT_NON_TERMINAL",
    }
    for name, values in partitions.items():
        assert values, f"{name} loaded empty — every downstream comparison would be vacuous"
        assert all(isinstance(value, str) for value in values), (
            f"{name} must load as plain `str` values: spans carry text, and a "
            f"comparison of text against enum members would never be equal"
        )


def test_load_vocabulary_values_covers_both_vocabularies() -> None:
    """The union of the partitions is the token filter `extract_values` uses.

    It must span BOTH vocabularies. If it covered only `MergeState`, a POLL span
    that named a SUBMIT-only value would have that value filtered out before the
    comparison and the drift would read GREEN — the seeded wrong-vocabulary case
    below is exactly that failure.
    """
    values = vocabulary_values(load_vocabulary())

    assert {"done", "abandoned", "superseded", "no_record"} <= values, "POLL members missing"
    assert {"already_merged", "wip_halted", "attached", "error"} <= values, "SUBMIT members missing"
    assert "failed" not in values, (
        "`failed` is a value the runbooks invented; it appears nowhere in "
        "escalation/src/escalation/server.py (the real value is `error`)"
    )


def test_load_vocabulary_raises_when_the_vocabulary_module_is_missing(tmp_path: Path) -> None:
    """A missing normative module RAISES rather than loading an empty vocabulary.

    An empty vocabulary makes every span comparison compare nothing against
    nothing, which passes — the guard would report its strongest verdict while
    having read no vocabulary at all.
    """
    missing = tmp_path / "merge_state.py"

    with pytest.raises(AssertionError) as excinfo:
        load_vocabulary(path=missing)

    assert str(missing) in str(excinfo.value)


def test_load_vocabulary_raises_when_a_partition_is_absent(tmp_path: Path) -> None:
    """A renamed or deleted partition names ITSELF in the failure.

    This is the drift shape a refactor of the vocabulary module produces: the
    spans still cite `TERMINAL_STATES` while the module no longer defines it.
    """
    stub = tmp_path / "merge_state.py"
    stub.write_text(
        "LIVE_STATES = frozenset({'queued'})\n"
        "OUTCOME_STATES = frozenset({'queued'})\n"
        "EPISTEMIC_STATES = frozenset({'unknown'})\n"
        "POLL_STOP_STATES = frozenset({'done'})\n"
        "CANCEL_STATES = frozenset({'done'})\n"
        "SUBMIT_TERMINAL = frozenset({'done'})\n"
        "SUBMIT_NON_TERMINAL = frozenset({'queued'})\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError) as excinfo:
        load_vocabulary(path=stub)

    assert "TERMINAL_STATES" in str(excinfo.value)


def test_load_vocabulary_raises_when_the_vocabulary_module_will_not_execute(
    tmp_path: Path,
) -> None:
    """A broken normative module fails as a NAMED loader failure.

    Without the wrapper the reader gets a bare `SyntaxError` from a synthetic
    module name, with nothing tying it to this guard or to the file it tried to
    load.
    """
    broken = tmp_path / "merge_state.py"
    broken.write_text("LIVE_STATES = frozenset({\n", encoding="utf-8")

    with pytest.raises(AssertionError) as excinfo:
        load_vocabulary(path=broken)

    assert str(broken) in str(excinfo.value)


def test_load_vocabulary_does_not_import_the_shared_package(tmp_path: Path) -> None:
    """Loading is BY PATH: the stub above wins over any installed `shared`.

    The property that matters is that the guard reads the vocabulary from the
    tree it was pointed at. A `import shared.merge_state` would resolve through
    whatever editable install is on `sys.path` — in a task worktree, typically
    the MAIN checkout — and would silently pin this worktree's SKILL.md files
    against another tree's vocabulary.
    """
    stub = tmp_path / "merge_state.py"
    stub.write_text(
        "\n".join(
            f"{name} = frozenset({{'sentinel_value'}})"
            for name in (
                "LIVE_STATES",
                "TERMINAL_STATES",
                "OUTCOME_STATES",
                "EPISTEMIC_STATES",
                "POLL_STOP_STATES",
                "CANCEL_STATES",
                "SUBMIT_TERMINAL",
                "SUBMIT_NON_TERMINAL",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_vocabulary(path=stub)["TERMINAL_STATES"] == frozenset({"sentinel_value"})


# ---------------------------------------------------------------------------
# extract_spans
# ---------------------------------------------------------------------------


def test_extract_spans_reads_the_markdown_comment_carrier() -> None:
    """The HTML-comment carrier, with the begin comment's own body EXCLUDED.

    The begin comment names this module's path and the partition; a span that
    swallowed it would pin the pointer rather than the list. The trailing prose
    naming `already_merged` is outside the span and must not be read either —
    span placement is what keeps the extracted set honest.
    """
    spans = extract_spans(_MARKDOWN_SPAN, source=_FIXTURE_SOURCE)

    assert [name for name, _ in spans] == ["TERMINAL_STATES"]
    body = spans[0][1]
    assert "superseded" in body
    assert "Mirrors shared/src/shared/merge_state.py" not in body, "begin comment leaked in"
    assert "already_merged" not in body, "out-of-span prose leaked in"


def test_extract_spans_reads_the_bare_marker_carrier_in_a_docstring() -> None:
    """Bare marker lines inside a Python docstring, where `<!-- -->` is nonsense.

    Both carriers matter: four pinned sites are markdown and one is a docstring
    in `escalation/src/escalation/server.py`, and a guard that handled only one
    carrier would leave the other unpinned.
    """
    spans = extract_spans(_DOCSTRING_SPAN, source="server.py")

    assert [name for name, _ in spans] == ["CANCEL_STATES"]
    body = spans[0][1]
    assert "'superseded'" in body
    assert "Source of truth" not in body, "trailing pointer prose leaked in"


def test_extract_spans_reads_a_single_line_begin_comment() -> None:
    """The begin comment may close on its own line; the body starts after `-->`."""
    spans = extract_spans(_SET_NOTATION_SPAN, source=_FIXTURE_SOURCE)

    assert [name for name, _ in spans] == ["LIVE_STATES"]
    assert "finalizing" in spans[0][1]


def test_extract_spans_reads_every_span_in_order() -> None:
    """A file carries several spans; each is returned with its own partition."""
    spans = extract_spans(_TWO_SPANS, source=_FIXTURE_SOURCE)

    assert [name for name, _ in spans] == ["LIVE_STATES", "TERMINAL_STATES"]
    assert "verifying" in spans[0][1]
    assert "abandoned" in spans[1][1]


@pytest.mark.parametrize(
    ("text", "case", "expected_fragment"),
    [
        pytest.param(_SPAN_ABSENT, "no span at all", _FIXTURE_SOURCE, id="no-span"),
        pytest.param(_SPAN_UNTERMINATED, "unterminated span", "TERMINAL_STATES", id="unterminated"),
        pytest.param(_SPAN_NESTED, "nested span", "LIVE_STATES", id="nested"),
        pytest.param(_SPAN_UNOPENED_END, "end without begin", _FIXTURE_SOURCE, id="unopened-end"),
        pytest.param(
            _SPAN_UNKNOWN_PARTITION,
            "partition= names no partition",
            "TERMINAL_STATUSES",
            id="unknown-partition",
        ),
        pytest.param(_SPAN_NO_PARTITION, "no partition= at all", _FIXTURE_SOURCE, id="no-partition"),
    ],
)
def test_extract_spans_rejects_every_malformed_span(
    text: str, case: str, expected_fragment: str
) -> None:
    """Each way a span can be malformed fails LOUDLY, naming the offender.

    A missing span is the highest-value case: it is what an edit that deletes
    the markers while keeping the list produces, and returning `[]` for it would
    turn the site's whole drift check into a vacuous pass.
    """
    with pytest.raises(AssertionError) as excinfo:
        extract_spans(text, source=_FIXTURE_SOURCE)

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message or "server.py" in message, f"{case}: {message!r}"
    assert expected_fragment in message, f"{case}: message must name the offender: {message!r}"


# ---------------------------------------------------------------------------
# extract_values
# ---------------------------------------------------------------------------


_TERMINAL_LITERAL_STYLES = [
    pytest.param(
        "Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`",
        "markdown backticks",
        id="backticks",
    ),
    pytest.param(
        "TERMINAL = {\"done\", \"conflict\", 'blocked', 'abandoned', \"superseded\"}",
        "python string literals",
        id="python-literals",
    ),
    pytest.param(
        "state is done | conflict | blocked | abandoned | superseded",
        "bare pipe notation",
        id="pipes",
    ),
    pytest.param(
        "{done, conflict, blocked, abandoned, superseded}",
        "bare set notation",
        id="bare-set",
    ),
]


@pytest.mark.parametrize(("body", "case"), _TERMINAL_LITERAL_STYLES)
def test_extract_values_reads_every_literal_style_these_files_use(body: str, case: str) -> None:
    """All four styles present across the pinned sites yield the same set.

    The sites are not uniform and must not be forced to be: a SKILL.md renders
    backticks, a Python docstring writes quoted literals, and a fenced code
    block carries bare set notation. A guard that read only one style would
    silently pin nothing in the other files.
    """
    values = extract_values(body, vocabulary_values(load_vocabulary()))

    assert values == {"done", "conflict", "blocked", "abandoned", "superseded"}, case


def test_extract_values_keeps_only_vocabulary_members() -> None:
    """Surrounding prose words are dropped; only declared members survive.

    This is what lets the spans stay readable — a marked list may carry ordinary
    English inside it without every noun becoming a phantom member.
    """
    body = "**Terminal states** (`done`, `conflict`) — the poll loop stops here."

    assert extract_values(body, vocabulary_values(load_vocabulary())) == {"done", "conflict"}


def test_extract_values_keeps_a_member_of_the_OTHER_vocabulary() -> None:
    """A SUBMIT-only value inside a POLL span SURVIVES extraction.

    Load-bearing, and the reason the filter is the union of both vocabularies
    rather than the span's own partition: filtering by partition would drop the
    intruder before the comparison, and a span claiming `merge_status` returns
    `wip_halted` would read green. Surviving extraction is what makes it show up
    as an EXTRA value in `assert_span_matches`.
    """
    body = "Terminal: `done`, `wip_halted`"

    assert extract_values(body, vocabulary_values(load_vocabulary())) == {"done", "wip_halted"}


def test_extract_values_raises_when_it_extracts_nothing() -> None:
    """An empty extraction RAISES rather than returning an empty set.

    An empty set compares unequal to every partition, so this case would be
    caught downstream — but as "the list lost all five members", which sends the
    reader to fix a list that is fine. The real defect is a misplaced `:end`
    marker, and the message must say so.
    """
    with pytest.raises(AssertionError):
        extract_values("prose with no members at all", vocabulary_values(load_vocabulary()))


# ---------------------------------------------------------------------------
# assert_span_matches
# ---------------------------------------------------------------------------


def test_assert_span_matches_accepts_an_exact_match() -> None:
    """The green case: the span's set equals its partition exactly."""
    partitions = load_vocabulary()

    assert_span_matches(
        partitions["TERMINAL_STATES"],
        partitions["TERMINAL_STATES"],
        source=_FIXTURE_SOURCE,
        partition_name="TERMINAL_STATES",
    )


@pytest.mark.parametrize(
    ("text", "case", "expected_repr"),
    [
        pytest.param(
            _SPAN_MISSING_MEMBER,
            "a member the list forgot — today's escalation-watcher line",
            repr(["superseded"]),
            id="missing-member",
        ),
        pytest.param(
            _SPAN_EXTRA_MEMBER,
            "a SUBMIT-only value in a POLL list — also today's escalation-watcher line",
            repr(["already_merged"]),
            id="extra-member",
        ),
        pytest.param(
            _SPAN_WRONG_VOCABULARY,
            "a value from the other vocabulary entirely",
            repr(["wip_halted"]),
            id="wrong-vocabulary",
        ),
    ],
)
def test_assert_span_matches_rejects_every_drift_shape(
    text: str, case: str, expected_repr: str
) -> None:
    """Each drift shape fires, quoting the REPR of the offending values.

    A repr is contractually load-bearing: a reader cannot act on "the list has
    drifted" without being told WHICH value, so a message that stopped carrying
    it would have genuinely regressed. Pinning wording instead makes an
    editorial reword red while a message degraded to uselessness stays green.

    Missing and extra are reported SEPARATELY because the fixes differ: a
    missing value means the list is stale, an extra one usually means a value
    from the sibling vocabulary was pasted in.
    """
    partitions = load_vocabulary()
    (partition_name, body), = extract_spans(text, source=_FIXTURE_SOURCE)
    values = extract_values(body, vocabulary_values(partitions))

    with pytest.raises(AssertionError) as excinfo:
        assert_span_matches(
            values,
            partitions[partition_name],
            source=_FIXTURE_SOURCE,
            partition_name=partition_name,
        )

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message, f"{case}: {message!r}"
    assert partition_name in message, f"{case}: {message!r}"
    assert expected_repr in message, f"{case}: message must quote the offending value: {message!r}"


def test_assert_span_matches_rejects_an_empty_value_set() -> None:
    """An empty set is refused up front, as a non-vacuity backstop.

    `extract_values` already raises on an empty extraction; this is the second
    line of defence for any future caller that builds a value set another way.
    """
    partitions = load_vocabulary()

    with pytest.raises(AssertionError):
        assert_span_matches(
            frozenset(),
            partitions["TERMINAL_STATES"],
            source=_FIXTURE_SOURCE,
            partition_name="TERMINAL_STATES",
        )


# ---------------------------------------------------------------------------
# find_unregistered_sites
# ---------------------------------------------------------------------------


def _write_fixture_site(directory: Path, name: str, tokens: list[str]) -> Path:
    """A throwaway markdown file naming `tokens`, for the registry scan's tests.

    Written under `tmp_path` rather than into the repo: the scan's own fixtures
    must not be discoverable BY the live scan, or `test_registry_is_complete`
    would go red on this module's test data.
    """
    body = "\n".join(f"- `{token}` — restated here" for token in tokens)
    path = directory / name
    path.write_text(f"# fixture site\n\n{body}\n", encoding="utf-8")
    return path


_DISCRIMINATING_SAMPLE = ["abandoned", "superseded", "verifying", "finalizing", "already_merged"]

# Every one of these is a real vocabulary member AND an ordinary English word or
# a task status. A file naming six of them is discussing merges, not enumerating
# the vocabulary — which is why they are excluded from the count.
_AMBIGUOUS_SAMPLE = ["done", "blocked", "queued", "conflict", "unknown", "error"]


def test_find_unregistered_sites_is_empty_when_every_site_is_registered(tmp_path: Path) -> None:
    """The green case, including the two shapes that must NOT be reported.

    A registered file over the threshold is fine (that is the point of the
    registry), and an unregistered file UNDER the threshold is fine too — it
    mentions states without enumerating them.
    """
    enumerating = _write_fixture_site(tmp_path, "enumerating.md", _DISCRIMINATING_SAMPLE)
    under = _write_fixture_site(tmp_path, "under-threshold.md", _DISCRIMINATING_SAMPLE[:3])

    assert (
        find_unregistered_sites(
            files=[enumerating, under],
            registry={str(enumerating): "enumerates the poll vocabulary"},
            threshold=4,
        )
        == []
    )


def test_find_unregistered_sites_reports_a_new_unpinned_site(tmp_path: Path) -> None:
    """A NEW enumeration site nobody registered is exactly what this scan is for.

    A drift guard whose site list is hand-maintained reproduces the defect it
    exists to prevent: it reads green while a freshly written runbook restates
    the vocabulary and drifts on the next member.
    """
    registered = _write_fixture_site(tmp_path, "registered.md", _DISCRIMINATING_SAMPLE)
    newcomer = _write_fixture_site(tmp_path, "newcomer.md", _DISCRIMINATING_SAMPLE[:4])

    assert find_unregistered_sites(
        files=[registered, newcomer],
        registry={str(registered): "enumerates the poll vocabulary"},
        threshold=4,
    ) == [str(newcomer)]


def test_find_unregistered_sites_counts_distinct_tokens_only(tmp_path: Path) -> None:
    """A file naming one state six times is discussing it, not enumerating.

    Counting occurrences would flag every doc that treats a single state in
    depth — noise that trains readers to register files to silence the guard,
    which is how a guard stops meaning anything.
    """
    repeated = _write_fixture_site(tmp_path, "repeated.md", ["superseded"] * 6)

    assert find_unregistered_sites(files=[repeated], registry={}, threshold=4) == []


def test_find_unregistered_sites_ignores_the_ambiguous_tokens(tmp_path: Path) -> None:
    """Six AMBIGUOUS members do not make an enumeration.

    `done`, `blocked`, `queued`, `conflict`, `unknown` and `error` are all task
    statuses or ordinary prose in this repo, so counting them would flag most of
    `skills/` — which is why the count runs over the DISCRIMINATING subset.
    """
    ambiguous = _write_fixture_site(tmp_path, "ambiguous.md", _AMBIGUOUS_SAMPLE)

    assert find_unregistered_sites(files=[ambiguous], registry={}, threshold=4) == []


def test_find_unregistered_sites_skips_the_declared_rule_sites(tmp_path: Path) -> None:
    """An explicitly declared ARM-RULE site is not reported as unregistered.

    A rule site enumerates a deliberately NARROWED subset (which statuses halt
    the queue, which states a particular loop may stop on). Pinning one to a
    partition would demand it re-add the very members it exists to exclude, so
    it is excluded by name, with its reason recorded beside it, rather than
    silently falling under the threshold.
    """
    rule_site = _write_fixture_site(tmp_path, "rule-site.md", _DISCRIMINATING_SAMPLE)

    assert (
        find_unregistered_sites(
            files=[rule_site],
            registry={},
            rule_sites={str(rule_site): "a narrowed halt-status rule, not a vocabulary copy"},
            threshold=4,
        )
        == []
    )


def test_find_unregistered_sites_raises_on_an_empty_scan() -> None:
    """An empty file list RAISES rather than returning `[]`.

    `[]` from an empty scan is indistinguishable from "every site is
    registered" — an over-broad pathspec would report the guard's strongest
    possible result while having read nothing at all.
    """
    with pytest.raises(AssertionError):
        find_unregistered_sites(files=[], registry={}, threshold=4)


def test_tracked_skill_markdown_fails_loudly_without_the_git_oracle(tmp_path: Path) -> None:
    """The tracked-file oracle raises when it cannot run, with no filesystem fallback.

    The scan's verdict must come from TRACKED files only, so that an untracked
    scratch file cannot flip it. Falling back to a filesystem walk on a failed
    oracle would restore that exact hazard in precisely the situation nobody is
    watching for it. `git` on PATH is not guaranteed either: a verify
    subprocess's PATH is rewritten by
    `orchestrator/src/orchestrator/verify.py::_target_subprocess_env`.
    """
    with pytest.raises(RuntimeError) as excinfo:
        tracked_skill_markdown(root=tmp_path)

    assert str(tmp_path) in str(excinfo.value)
