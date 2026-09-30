"""The DURABLE journal a plan-tools markup rejection lands in.

Task 4744. The measurement that opened it, on 2026-08-25::

    journalctl --user --since 2026-08-22 | grep 'markup guard:'   ->  0 lines

against 35 REAL plan-tools rejections in ``data/orchestrator/agent-transcripts/``
over the same span. plan-tools is a per-agent STDIO SUBPROCESS whose stderr the
CLI agent that spawned it consumes, so the ``logger.warning`` inside
``MarkupGuardMiddleware._emit_fact`` — the only per-call record of WHO leaked —
reaches no durable sink at all. A plan-tools storm escalation therefore asks a
human to "identify the leaking caller from the guard's own log lines", which is
unfollowable by construction.

The subject under test is :mod:`orchestrator.mcp.markup_journal`: the sink that
turns that fact record into an append-only JSONL line an operator can grep after
the fact. The END-TO-END wiring (that plan-tools' registered guard actually
reaches it) is pinned in ``test_plan_tools_markup_guard.py``, against the real
server; nothing here re-derives detection, repair or policy, which are owned by
``shared.toolcall_markup`` / ``shared.mcp_markup_middleware`` and pinned by
their own tests.

## Async marker

Every async test carries an explicit ``@pytest.mark.asyncio``. orchestrator does
NOT set ``asyncio_mode = auto``.

## Sentinel-literal hazard — every specimen is BUILT, never written verbatim

This module describes MCP tool-call envelope markup, so it is exactly the file
that must not contain any of it literally. The rationale is the one recorded at
``shared/src/shared/toolcall_markup.py`` lines 52-62 and repeated in
``test_plan_tools_markup_guard.py``: an agent editing a file that holds a raw
envelope literal has to emit that literal INSIDE its own tool-call argument,
which reproduces the very over-consumption defect under test.

So every specimen is assembled from :func:`_close` / :func:`_open_param`, which
build their angle bracket from ``chr(60)``, and :func:`_assert_no_raw_sentinels`
enforces that on this module's OWN BYTES at import — checked against
``shared.toolcall_markup.ENVELOPE_LITERALS``, the single owner of the literal
set (INV-5), plus the two structural prefixes.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from shared.mcp_markup_middleware import FACT_MARKUP_DETECTED
from shared.toolcall_markup import ENVELOPE_LITERALS

from orchestrator.mcp import markup_journal, markup_sink

# ---------------------------------------------------------------------------
# Sentinel BUILDERS — the only way markup enters this module.
# ---------------------------------------------------------------------------

#: The opening angle bracket, spelled so it never appears verbatim in the file.
_LT = chr(60)


def _close(name: str) -> str:
    """Build the closing tag for *name* (the mis-close shape the harness emits)."""
    return _LT + '/' + name + '>'


def _open_param(name: str) -> str:
    """Build the canonical opening tag for parameter *name*."""
    return _LT + 'parameter name="' + name + '">'


#: ``shared.toolcall_markup.ENVELOPE_LITERALS`` (the single owner of the literal
#: set, INV-5) plus the two structural prefixes every built specimen uses, so a
#: builder output spelled out by hand is caught even when it is not itself one
#: of the enumerated literals. Applied to this module's OWN BYTES at import, and
#: to the JOURNAL FILE the sink writes — the same predicate, two artifacts.
_FORBIDDEN_SEQUENCES = (*ENVELOPE_LITERALS, _LT + '/', _LT + 'parameter ')


def _assert_no_raw_sentinels() -> None:
    """Fail at IMPORT if this file's own bytes carry a raw envelope literal."""
    source = Path(__file__).read_text(encoding='utf-8')
    for sequence in _FORBIDDEN_SEQUENCES:
        if sequence in source:
            raise AssertionError(
                f'{Path(__file__).name} contains a RAW envelope sentinel '
                f'({sequence!r}). Build it from _close()/_open_param() instead '
                '— a verbatim literal here corrupts the tool call that writes '
                'this file. See the module docstring.'
            )


_assert_no_raw_sentinels()


# ---------------------------------------------------------------------------
# The fact record — the shape ``MarkupGuardMiddleware._emit_fact`` builds.
# ---------------------------------------------------------------------------

#: The measured plan-tools leak: ``add_design_decision.decision`` mis-closed and
#: swallowing its ``rationale`` sibling. ``pattern`` and ``misclose`` are the
#: leaked markup itself, which is why they are BUILT rather than written.
_MISCLOSE = _close('decision')
_PATTERN = _open_param('rationale')


def make_fact(**overrides: Any) -> dict[str, Any]:
    """One ``markup_detected`` record, keyed exactly as the middleware keys it.

    Spelled here rather than imported so a middleware key rename shows up as a
    failing assertion in the module that consumes the record, instead of
    silently re-shaping the journal's own contract.
    """
    fact = {
        'fact': FACT_MARKUP_DETECTED,
        'tool': 'add_design_decision',
        'param': 'decision',
        'pattern': _PATTERN,
        'misclose': _MISCLOSE,
        'outcome': 'rejected',
        'recovered_params': ['rationale'],
        # Structurally None on this boundary: ``_identity`` reads only
        # arguments named agent_id / project_root / project_id, and no
        # plan-tools tool declares any of the three.
        'agent_id': None,
        'project': None,
    }
    fact.update(overrides)
    return fact


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def journal_lines(root: Path, label: str = 'plan-tools') -> list[dict[str, Any]]:
    """Every record in the journal for *label* under *root*, parsed."""
    path = markup_journal.journal_path(root, label)
    text = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def worktree_with_no_name() -> str:
    """A path whose ``.name`` is empty, so the attribution ladder runs out.

    The filesystem root is the only such path, and it is used as a VALUE here —
    nothing is read from or written to it, because ``resolve_root`` is stubbed.
    """
    return os.sep


def build_sink(
    tmp_path: Path,
    *,
    worktree: Path | None = None,
    server_label: str = 'plan-tools',
    subject: str = '4744',
    **kwargs: Any,
):
    """A journal sink whose project root is *tmp_path* — no git, no subprocess."""
    return markup_journal.make_fact_journal(
        worktree=worktree if worktree is not None else tmp_path / 'lane',
        server_label=server_label,
        subject_task_id=lambda: subject,
        resolve_root=kwargs.pop('resolve_root', lambda _worktree: tmp_path),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# (f) — the PATH is pinned by a test, not by prose.
# ---------------------------------------------------------------------------


class TestTheJournalPathIsFixed:
    """An operator is told ONE exact path to grep, so it cannot drift."""

    def test_the_dirname_is_the_operator_facing_constant(self):
        assert markup_journal.MARKUP_JOURNAL_DIRNAME == 'data/orchestrator/markup-guard'

    def test_journal_path_composes_root_dirname_and_label(self, tmp_path):
        assert markup_journal.journal_path(tmp_path, 'plan-tools') == (
            tmp_path / 'data' / 'orchestrator' / 'markup-guard' / 'plan-tools.jsonl'
        )
        assert markup_journal.journal_path(tmp_path, 'verdict-tools') == (
            tmp_path / 'data' / 'orchestrator' / 'markup-guard' / 'verdict-tools.jsonl'
        )


# ---------------------------------------------------------------------------
# (a)/(b)/(c)/(d) — the core record.
# ---------------------------------------------------------------------------


class TestOneEventIsOneLine:
    """The journal is per-EVENT, which is what the storm summary is not."""

    @pytest.mark.asyncio
    async def test_a_single_fact_appends_exactly_one_line(self, tmp_path):
        sink = build_sink(tmp_path)

        await sink(make_fact())

        path = markup_journal.journal_path(tmp_path, 'plan-tools')
        assert path.exists(), 'the journal directory is created lazily, on first write'
        assert len(path.read_text(encoding='utf-8').splitlines()) == 1

    @pytest.mark.asyncio
    async def test_the_line_carries_the_facts_own_keys_unchanged(self, tmp_path):
        sink = build_sink(tmp_path)
        fact = make_fact()

        await sink(fact)

        (line,) = journal_lines(tmp_path)
        for key, value in fact.items():
            assert line[key] == value, f'{key} was not journalled unchanged'

    @pytest.mark.asyncio
    async def test_the_line_carries_the_identity_envelope(self, tmp_path):
        """The whole point: a durable line that NAMES the leaking task."""
        worktree = tmp_path / 'lane-4744'
        worktree.mkdir()
        sink = build_sink(tmp_path, worktree=worktree)

        await sink(make_fact())

        (line,) = journal_lines(tmp_path)
        assert line['server'] == 'plan-tools'
        assert line['subject_task_id'] == '4744'
        assert line['worktree'] == str(worktree)
        assert line['pid'] == os.getpid()
        assert 'ts' in line

    @pytest.mark.asyncio
    async def test_null_identity_fields_are_recorded_as_the_nulls_they_are(self, tmp_path):
        """(b) ``agent_id`` and ``project`` are structurally None here.

        Present-and-null, never absent: a consumer must never have to tell
        "no identity on this boundary" apart from "that emitter forgot the key",
        which is the same reason the middleware emits a null ``misclose``.
        """
        sink = build_sink(tmp_path)

        await sink(make_fact())

        (line,) = journal_lines(tmp_path)
        assert 'agent_id' in line and line['agent_id'] is None
        assert 'project' in line and line['project'] is None

    @pytest.mark.asyncio
    async def test_the_timestamp_is_iso_8601_utc(self, tmp_path):
        """(c) An operator correlating with a transcript needs a real clock."""
        sink = build_sink(tmp_path, now=lambda: 1_756_000_000.0)

        await sink(make_fact())

        (line,) = journal_lines(tmp_path)
        parsed = datetime.fromisoformat(line['ts'])
        assert parsed.tzinfo is not None, 'a naive timestamp is not correlatable'
        assert parsed.utcoffset() == UTC.utcoffset(None)
        assert parsed == datetime.fromtimestamp(1_756_000_000.0, tz=UTC)

    @pytest.mark.asyncio
    async def test_the_sink_returns_the_journal_path_as_a_locator(self, tmp_path):
        """(d) Same contract as the escalation sink's id: a locator or None."""
        sink = build_sink(tmp_path)
        expected = str(markup_journal.journal_path(tmp_path, 'plan-tools'))

        first = await sink(make_fact())
        second = await sink(make_fact())

        assert first == expected
        assert second == expected
        assert len(journal_lines(tmp_path)) == 2, 'append, never overwrite'


# ---------------------------------------------------------------------------
# (e) — CONCURRENCY, the real correctness pin.
# ---------------------------------------------------------------------------


class TestConcurrentAppendsNeverLoseALine:
    """Many plan-tools subprocesses append to ONE file, and a leak is bursty.

    This is what forbids any read-modify-write implementation: "read the file,
    concatenate, rewrite" passes every row above and silently destroys other
    writers' lines here — the same class of data loss the whole guard exists to
    prevent.
    """

    def test_eight_writers_twenty_records_each_yields_one_hundred_sixty_lines(
        self, tmp_path
    ):
        sink = build_sink(tmp_path)
        writers, per_writer = 8, 20

        def run(writer: int) -> None:
            async def drive() -> None:
                for seq in range(per_writer):
                    await sink(make_fact(param=f'w{writer}-{seq}'))

            asyncio.run(drive())

        with ThreadPoolExecutor(max_workers=writers) as pool:
            for future in [pool.submit(run, w) for w in range(writers)]:
                future.result()

        lines = journal_lines(tmp_path)
        assert len(lines) == writers * per_writer, (
            'a lost line is a lost attribution — the O_APPEND single write is '
            'what makes concurrent per-agent subprocesses safe'
        )
        for line in lines:
            assert line['fact'] == FACT_MARKUP_DETECTED
            assert line['server'] == 'plan-tools'
            assert line['subject_task_id'] == '4744'
        assert len({line['param'] for line in lines}) == writers * per_writer, (
            'every distinct record survived, not merely the right count of them'
        )


# ---------------------------------------------------------------------------
# The journal FILE is itself a file an agent can safely read.
# ---------------------------------------------------------------------------


class TestTheJournalNeverHoldsARawEnvelopeLiteral:
    """The escaping contract, applied to the ARTIFACT rather than to a source file.

    This journal records envelope literals BY CONSTRUCTION — the ``pattern`` and
    ``misclose`` fields ARE the leaked markup. A file holding them verbatim is a
    file no agent can safely read or edit: pulling it into a tool-call argument
    reproduces the exact over-consumption defect the journal exists to record,
    at the one artifact an operator is told to open. That is the hazard
    ``shared/src/shared/toolcall_markup.py`` warns about and the reason the
    committed specimen corpus escapes every literal the same way.
    """

    def test_the_fixture_really_carries_markup_to_escape(self):
        """(c) A control: the test cannot pass by being handed a clean specimen."""
        assert any(literal in _PATTERN for literal in ENVELOPE_LITERALS), (
            'the pattern specimen must be a real envelope literal, or (a) below '
            'is vacuous'
        )
        assert _MISCLOSE.startswith(_LT + '/'), (
            'the mis-close specimen must carry the structural closing prefix'
        )

    @pytest.mark.asyncio
    async def test_the_written_file_holds_no_raw_sentinel(self, tmp_path):
        """(a) The same predicate ``_assert_no_raw_sentinels`` applies to source."""
        sink = build_sink(tmp_path)

        await sink(make_fact())

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_text(
            encoding='utf-8'
        )
        for sequence in _FORBIDDEN_SEQUENCES:
            assert sequence not in raw, (
                f'the journal holds a RAW envelope sentinel ({sequence!r}) — '
                'reading this file into a tool-call argument would reproduce '
                'the defect it records'
            )
        assert _LT not in raw, (
            'no opening angle bracket at all: a partial escape leaves a shape '
            'a future literal could slip through'
        )

    @pytest.mark.asyncio
    async def test_the_escape_is_lossless_not_sanitisation(self, tmp_path):
        """(b) Escaped on the way to disk, IDENTICAL on the way back out.

        A journal that mangled the pattern would answer "something leaked" while
        destroying which literal it was — the one field that says what the
        upstream harness bug actually emitted.
        """
        sink = build_sink(tmp_path)
        fact = make_fact()

        await sink(fact)

        (line,) = journal_lines(tmp_path)
        assert line['pattern'] == _PATTERN
        assert line['misclose'] == _MISCLOSE

    @pytest.mark.asyncio
    async def test_an_angle_bracket_anywhere_in_the_record_is_escaped(self, tmp_path):
        """The replacement is blanket, so a NEW field carrying markup is covered too.

        In JSON output the opening angle bracket can only ever occur inside a
        string, so escaping every occurrence is both sufficient and safe — it
        can never touch the structural punctuation.
        """
        sink = build_sink(tmp_path)
        smuggled = 'prose ' + _close('rationale') + ' more prose'

        await sink(make_fact(some_future_key=smuggled))

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_text(
            encoding='utf-8'
        )
        assert _LT not in raw
        (line,) = journal_lines(tmp_path)
        assert line['some_future_key'] == smuggled


# ---------------------------------------------------------------------------
# The degraded paths — a journal outage must never become an outage of its own.
# ---------------------------------------------------------------------------


class TestTheJournalNeverRaises:
    """The middleware calls this channel PURELY ADDITIVELY.

    The call's outcome — rejected, repaired, refused — is already decided by the
    time ``_emit_fact`` runs, so a journal failure must cost an operator
    visibility and never turn a working guard into an outage. Same contract
    ``markup_sink.make_escalation_sink`` keeps, asserted the same way: every row
    awaits the sink DIRECTLY, so nothing upstream can be masking a raise.
    """

    @pytest.mark.asyncio
    async def test_an_unresolvable_project_root_costs_a_line_not_the_call(self, tmp_path):
        """(a) git answered nothing — the commonest transient failure."""
        sink = build_sink(tmp_path, resolve_root=lambda _worktree: None)

        assert await sink(make_fact()) is None
        assert not (tmp_path / 'data').exists(), 'nothing half-written'

    @pytest.mark.asyncio
    async def test_a_raising_project_root_resolver_is_contained(self, tmp_path):
        """(b) The default resolver never raises, but an injected one may."""

        def boom(_worktree: Path) -> Path:
            raise OSError('git could not be forked')

        sink = build_sink(tmp_path, resolve_root=boom)

        assert await sink(make_fact()) is None

    @pytest.mark.asyncio
    async def test_an_uncreatable_journal_directory_is_contained(self, tmp_path):
        """(c) The parent cannot be made — here because the root is a FILE."""
        root = tmp_path / 'not-a-directory'
        root.write_text('', encoding='utf-8')
        sink = build_sink(tmp_path, resolve_root=lambda _worktree: root)

        assert await sink(make_fact()) is None

    @pytest.mark.asyncio
    async def test_an_unwritable_journal_file_is_contained(self, tmp_path):
        """(c) The write itself fails — here because the path is a directory."""
        markup_journal.journal_path(tmp_path, 'plan-tools').mkdir(parents=True)
        sink = build_sink(tmp_path)

        assert await sink(make_fact()) is None


class TestAttributionFailureNeverCostsTheRecord:
    """Losing a line because attribution failed is the opposite of the point.

    The whole task is that these events reach a durable sink at all; a record
    that says "somebody leaked from add_design_decision at 16:52:34" is strictly
    more than the nothing that was written before.
    """

    @pytest.mark.asyncio
    async def test_a_raising_thunk_still_writes_the_line(self, tmp_path):
        """(d) It falls back to the worktree name, which is a task lane's id."""
        worktree = tmp_path / '4744'
        worktree.mkdir()

        def boom() -> str:
            raise RuntimeError('the plan could not be read')

        sink = markup_journal.make_fact_journal(
            worktree=worktree,
            server_label='plan-tools',
            subject_task_id=boom,
            resolve_root=lambda _worktree: tmp_path,
        )

        assert await sink(make_fact()) is not None
        (line,) = journal_lines(tmp_path)
        assert line['subject_task_id'] == '4744'

    @pytest.mark.asyncio
    async def test_the_bottom_of_the_ladder_is_an_explicit_sentinel(self, tmp_path):
        """(e) Never None and never absent — an explicit "nobody knows"."""
        anonymous = Path(worktree_with_no_name())
        sink = markup_journal.make_fact_journal(
            worktree=anonymous,
            server_label='plan-tools',
            subject_task_id=lambda: '',
            resolve_root=lambda _worktree: tmp_path,
        )

        await sink(make_fact())

        (line,) = journal_lines(tmp_path)
        assert line['subject_task_id'] == markup_sink.MARKUP_UNATTRIBUTED_SUBJECT
        assert line['subject_task_id'] == 'unattributed'


class TestMemoizationIsAsymmetric:
    """Successes are cached; failures are RETRIED. The asymmetry is the point.

    The reasoning is ``markup_sink.make_escalation_sink``'s, verbatim: a failed
    git resolution is transient by nature — a fork failure under load, an EINTR,
    a timeout, an index.lock storm — and caching one would permanently disable
    the journal for every later event on this server, losing exactly the records
    this module exists to preserve.
    """

    @pytest.mark.asyncio
    async def test_a_failed_resolution_is_retried_on_the_next_record(self, tmp_path):
        """(f) And the second record LANDS once the transient failure clears."""
        calls: list[Path] = []
        answers: list[Path | None] = [None, tmp_path]

        def resolve(worktree: Path) -> Path | None:
            calls.append(worktree)
            return answers.pop(0)

        sink = build_sink(tmp_path, resolve_root=resolve)

        assert await sink(make_fact(param='first')) is None
        assert await sink(make_fact(param='second')) is not None

        assert len(calls) == 2, 'a cached failure would silence the journal forever'
        (line,) = journal_lines(tmp_path)
        assert line['param'] == 'second'

    @pytest.mark.asyncio
    async def test_a_successful_resolution_is_memoized(self, tmp_path):
        """(f) One ``git rev-parse`` per server, not one per leaked call."""
        calls: list[Path] = []

        def resolve(worktree: Path) -> Path:
            calls.append(worktree)
            return tmp_path

        sink = build_sink(tmp_path, resolve_root=resolve)

        await sink(make_fact(param='first'))
        await sink(make_fact(param='second'))

        assert len(calls) == 1
        assert [line['param'] for line in journal_lines(tmp_path)] == ['first', 'second']


# ---------------------------------------------------------------------------
# The BOUND — what makes the O_APPEND atomicity premise sound, not assumed.
# ---------------------------------------------------------------------------


class TestTheLineIsBounded:
    """A single ``os.write`` on an O_APPEND fd is atomic only while it is SMALL.

    Every row above rests on that atomicity, and the payloads this guard sees
    are measured in tens of KB. So the bound is ENFORCED here rather than
    assumed: an over-budget record degrades to a shorter, still-valid,
    still-identity-carrying line, which is strictly better than either a torn
    line or a silently dropped one.
    """

    @pytest.mark.asyncio
    async def test_a_huge_pattern_still_yields_one_bounded_line(self, tmp_path):
        """(a) The measured payloads reach tens of KB; the line must not."""
        sink = build_sink(tmp_path)
        huge = 'x' * 10_000

        await sink(make_fact(pattern=huge, misclose=huge))

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_bytes()
        assert raw.count(b'\n') == 1, 'one event is one line, however big it was'
        assert len(raw) <= markup_journal.MARKUP_JOURNAL_MAX_LINE_BYTES
        (line,) = journal_lines(tmp_path)
        assert line['tool'] == 'add_design_decision', 'still a complete record'

    @pytest.mark.asyncio
    async def test_a_truncated_field_is_a_prefix_and_is_disclosed(self, tmp_path):
        """(b) A trimmed value must be tellable from a short one."""
        sink = build_sink(tmp_path)
        huge = 'y' * 10_000

        await sink(make_fact(pattern=huge))

        (line,) = journal_lines(tmp_path)
        cap = markup_journal.MARKUP_JOURNAL_MAX_FIELD_CHARS
        assert line['pattern'] == huge[:cap], (
            'a PREFIX of what was sent, never a rewrite — the same discipline '
            "the middleware's own repair keeps"
        )
        assert 'pattern' in line['journal_truncated']
        assert 'misclose' not in line['journal_truncated'], (
            'a field that fit must not be reported as trimmed, or the marker '
            'stops meaning anything'
        )

    @pytest.mark.asyncio
    async def test_a_huge_recovered_params_list_is_capped_and_disclosed(self, tmp_path):
        """(c) The one list field the middleware emits, bounded by COUNT."""
        sink = build_sink(tmp_path)

        await sink(make_fact(recovered_params=[f'p{i}' for i in range(5_000)]))

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_bytes()
        assert len(raw) <= markup_journal.MARKUP_JOURNAL_MAX_LINE_BYTES
        (line,) = journal_lines(tmp_path)
        assert len(line['recovered_params']) == markup_journal.MARKUP_JOURNAL_MAX_LIST_ITEMS
        assert 'recovered_params' in line['journal_truncated']

    @pytest.mark.asyncio
    async def test_the_overflow_floor_keeps_a_record_rather_than_dropping_it(
        self, tmp_path
    ):
        """(d) A shape a future middleware could grow — many keys, each small.

        Per-field caps cannot bound this one, so there has to be a floor under
        them. A record must never be dropped and must never be written torn:
        dropping it is the exact fail-soft this whole PRD exists to end.
        """
        sink = build_sink(tmp_path)
        sprawl = {f'future_key_{i}': f'value-{i}' for i in range(2_000)}

        await sink(make_fact(**sprawl))

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_bytes()
        assert raw.count(b'\n') == 1
        assert len(raw) <= markup_journal.MARKUP_JOURNAL_MAX_LINE_BYTES
        (line,) = journal_lines(tmp_path)
        assert line['journal_overflow'] is True, (
            'the marker is what tells an operator this line is the floor and '
            'not the whole record'
        )
        for key in ('ts', 'server', 'subject_task_id', 'tool', 'param', 'outcome'):
            assert key in line, f'{key} is identity — the floor must still carry it'
        assert line['subject_task_id'] == '4744'
        assert line['tool'] == 'add_design_decision'
        assert line['outcome'] == 'rejected'

    @pytest.mark.asyncio
    async def test_an_ordinary_record_is_nowhere_near_the_bound(self, tmp_path):
        """(e) A regression pin: the caps must never tighten onto the normal path.

        The measured leak is a realistic fact, and if the ordinary case ever
        starts brushing the bound the journal quietly stops carrying the very
        fields it was built to carry.
        """
        sink = build_sink(tmp_path)

        await sink(make_fact())

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_bytes()
        assert len(raw) < markup_journal.MARKUP_JOURNAL_MAX_LINE_BYTES // 2
        (line,) = journal_lines(tmp_path)
        assert 'journal_truncated' not in line, (
            'an untrimmed record carries no trim marker at all, so the key '
            "means \"something was cut\" rather than \"this emitter ran\""
        )
        assert 'journal_overflow' not in line


# ---------------------------------------------------------------------------
# The DISCLOSURE MARKER means what the constants say it means.
# ---------------------------------------------------------------------------


class TestEveryCutIsDisclosed:
    """``journal_truncated`` is ABSENT on an untouched record — so a consumer
    reading a line with no marker is entitled to treat every value in it as
    verbatim. That entitlement is the whole worth of the marker, and it is only
    as good as the narrowest axis that raises it.
    """

    @pytest.mark.asyncio
    async def test_an_over_long_list_item_is_disclosed_not_only_a_long_list(
        self, tmp_path
    ):
        """The COUNT axis and the LENGTH axis both raise the marker.

        A list short enough to keep every entry but whose entries are each cut
        to the field cap is still a rewritten value. Reporting only the count
        axis would hand a consumer a silently shortened item under a line that
        claims nothing was cut.
        """
        sink = build_sink(tmp_path)
        cap = markup_journal.MARKUP_JOURNAL_MAX_FIELD_CHARS

        await sink(make_fact(recovered_params=['z' * 5_000]))

        (line,) = journal_lines(tmp_path)
        assert len(line['recovered_params']) == 1, 'the COUNT axis was never hit'
        assert line['recovered_params'][0] == 'z' * cap, 'a prefix, not a rewrite'
        assert 'recovered_params' in line[markup_journal.MARKUP_JOURNAL_TRUNCATED_KEY]

    @pytest.mark.asyncio
    async def test_an_over_long_subject_is_capped_and_disclosed(self, tmp_path):
        """The ENVELOPE is capped on the same terms as the fact's own fields.

        ``subject_task_id`` comes from the injected thunk — for plan-tools,
        ``plan.json``'s agent-written ``task_id`` — so nothing upstream bounds
        it, and it is a FLOOR key: unbounded, it would push even the
        identity-only line past the byte bound.
        """
        sink = build_sink(tmp_path, subject='4744' * 10_000)
        cap = markup_journal.MARKUP_JOURNAL_MAX_FIELD_CHARS

        await sink(make_fact())

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_bytes()
        assert len(raw) <= markup_journal.MARKUP_JOURNAL_MAX_LINE_BYTES
        (line,) = journal_lines(tmp_path)
        assert line['subject_task_id'] == ('4744' * 10_000)[:cap]
        assert 'subject_task_id' in line[markup_journal.MARKUP_JOURNAL_TRUNCATED_KEY]

    @pytest.mark.asyncio
    async def test_an_over_long_subject_cannot_burst_the_overflow_floor(self, tmp_path):
        """Both degradations at once: a sprawling record AND a huge floor key.

        The floor is what the byte bound rests on when everything else has been
        cut away, so it is MEASURED after it is built rather than assumed to
        fit.
        """
        sink = build_sink(tmp_path, subject='s' * 50_000)
        sprawl = {f'future_key_{i}': f'value-{i}' for i in range(2_000)}

        await sink(make_fact(**sprawl))

        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_bytes()
        assert raw.count(b'\n') == 1
        assert len(raw) <= markup_journal.MARKUP_JOURNAL_MAX_LINE_BYTES
        (line,) = journal_lines(tmp_path)
        assert line[markup_journal.MARKUP_JOURNAL_OVERFLOW_KEY] is True
        assert line['tool'] == 'add_design_decision', 'still identity-carrying'


# ---------------------------------------------------------------------------
# The IDENTITY ENVELOPE is the journal's own, and cannot be overwritten.
# ---------------------------------------------------------------------------


class TestTheEnvelopeCannotBeShadowed:
    """Four of the five envelope keys are ``MARKUP_JOURNAL_FLOOR_KEYS``.

    The floor exists to guarantee that a line always says WHO wrote it, WHEN
    and from WHERE. A middleware record that grew a key named ``server`` or
    ``ts`` — the middleware owns its own fact vocabulary and has grown keys
    before — must not be able to overwrite that guarantee from the outside.
    """

    @pytest.mark.asyncio
    async def test_a_fact_key_cannot_displace_the_server_label(self, tmp_path):
        sink = build_sink(tmp_path, server_label='plan-tools')

        await sink(make_fact(server='NOT-plan-tools'))

        (line,) = journal_lines(tmp_path)
        assert line['server'] == 'plan-tools', (
            'the journal names the server it IS, not the one a record claims'
        )

    @pytest.mark.asyncio
    async def test_a_fact_cannot_forge_the_timestamp_or_the_subject(self, tmp_path):
        sink = build_sink(tmp_path, subject='4744')

        await sink(make_fact(ts='bogus', subject_task_id='999', pid=1, worktree='/x'))

        (line,) = journal_lines(tmp_path)
        assert line['subject_task_id'] == '4744'
        assert line['pid'] == os.getpid()
        assert line['worktree'] != '/x'
        assert datetime.fromisoformat(line['ts']).tzinfo is not None, (
            'the real clock, not the forged value'
        )

    @pytest.mark.asyncio
    async def test_the_collision_is_logged_rather_than_silently_resolved(
        self, tmp_path, caplog
    ):
        """A name collision means the two vocabularies have started to overlap
        — an operator should hear about it, not discover it in a diff.
        """
        sink = build_sink(tmp_path)

        with caplog.at_level('WARNING'):
            await sink(make_fact(server='NOT-plan-tools'))

        assert any(
            'server' in record.getMessage() for record in caplog.records
        ), 'the shadowed key is named in the warning, not just resolved away'


# ---------------------------------------------------------------------------
# A record is DEGRADED, never dropped — the fail-soft this PRD exists to end.
# ---------------------------------------------------------------------------


class TestAnUnencodableRecordIsStillWritten:
    """The middleware emits str / None / list-of-str today.

    So did the key count, before the overflow floor was built for a shape it
    could grow into. A value ``json`` cannot render must cost that ONE FIELD
    its exact type — never the line, which would leave the event exactly where
    this module found it: in a stack trace on the stderr nobody retains.
    """

    @pytest.mark.asyncio
    async def test_a_non_serializable_value_keeps_the_record(self, tmp_path):
        sink = build_sink(tmp_path)

        assert await sink(make_fact(recovered_params={'rationale'})) is not None

        (line,) = journal_lines(tmp_path)
        assert line['tool'] == 'add_design_decision'
        assert line['subject_task_id'] == '4744'
        assert 'rationale' in str(line['recovered_params']), (
            'rendered, not dropped — the field survives as text'
        )

    @pytest.mark.asyncio
    async def test_an_unencodable_key_falls_to_the_identity_floor(self, tmp_path):
        """``default=`` covers a value; nothing covers a non-string KEY.

        So the floor catches it, exactly as it catches an over-budget record.
        """
        sink = build_sink(tmp_path)
        # Annotated ``dict[Any, Any]`` because the SPECIMEN IS THE WRONG SHAPE:
        # the int key is the whole point, and the sink's parameter is
        # ``dict[str, Any]``, so the literal cannot be passed inline.
        record: dict[Any, Any] = make_fact(**{'ok': object()}) | {7: 'int key'}

        assert await sink(record) is not None

        (line,) = journal_lines(tmp_path)
        assert line[markup_journal.MARKUP_JOURNAL_OVERFLOW_KEY] is True
        for key in markup_journal.MARKUP_JOURNAL_FLOOR_KEYS:
            assert key in line, f'{key} is identity — the floor must carry it'
        assert line['tool'] == 'add_design_decision'


# ---------------------------------------------------------------------------
# A SHORT WRITE must not leave an unterminated line for the next appender.
# ---------------------------------------------------------------------------


class TestAShortWriteIsCompleted:
    """The one corruption shape this format cannot survive.

    ``write(2)`` may report a short count — classically near ENOSPC, or on a
    large write interrupted by a signal. A partial record carries no trailing
    newline, so the next process's append concatenates onto it: one unparseable
    line, two events silently merged, and every consumer here splits on
    newlines.
    """

    @pytest.mark.asyncio
    async def test_a_short_write_still_lands_one_complete_line(self, tmp_path, monkeypatch):
        real_write = os.write
        calls: list[int] = []
        armed: set[int] = set()

        def dribble(fd: int, data: bytes) -> int:
            # Only THIS module's records are short-written, and only until the
            # record is finished. ``os.write`` is process-global, and a blanket
            # 16-byte cap on every fd for the length of a test is a far bigger
            # blast radius than the behaviour under test. A journal line is a
            # whole JSON object ending in a newline; its REMAINDERS are not, so
            # the fd stays armed until the last chunk lands.
            if data.startswith(b'{') and data.endswith(b'\n'):
                armed.add(fd)
            if fd not in armed:
                return real_write(fd, data)
            calls.append(len(data))
            written = real_write(fd, data[:16])
            if written >= len(data):
                armed.discard(fd)
            return written

        monkeypatch.setattr(os, 'write', dribble)
        sink = build_sink(tmp_path)

        assert await sink(make_fact()) is not None

        assert len(calls) > 1, 'the fixture must actually short-write'
        raw = markup_journal.journal_path(tmp_path, 'plan-tools').read_bytes()
        assert raw.endswith(b'\n'), 'an unterminated line poisons the next append'
        assert raw.count(b'\n') == 1
        (line,) = journal_lines(tmp_path)
        assert line['tool'] == 'add_design_decision', 'nothing was lost in the middle'

    @pytest.mark.asyncio
    async def test_a_write_that_makes_no_progress_is_contained(self, tmp_path, monkeypatch):
        """A zero-count write would spin forever — it is an error, not a retry."""
        real_write = os.write
        monkeypatch.setattr(
            os,
            'write',
            lambda fd, data: 0 if data.startswith(b'{') else real_write(fd, data),
        )
        sink = build_sink(tmp_path)

        assert await sink(make_fact()) is None, 'contained, like any write failure'


# ---------------------------------------------------------------------------
# ONE attribution ladder, shared with the escalation sink on the same boundary.
# ---------------------------------------------------------------------------


class TestTheAttributionLadderIsShared:
    """The two channels a boundary guard emits through are asserted to name the
    same subject. Two copies of the ladder could drift in opposite directions
    while both kept passing their own tests — the INV-5 sibling duplication
    ``markup_sink``'s own header rules against.
    """

    @pytest.mark.asyncio
    async def test_the_journal_uses_markup_sinks_ladder(self, tmp_path, monkeypatch):
        seen: list[str] = []
        monkeypatch.setattr(
            markup_sink,
            'resolve_subject',
            lambda thunk, worktree, *, what: seen.append(what) or 'stubbed',
        )
        sink = build_sink(tmp_path)

        await sink(make_fact())

        (line,) = journal_lines(tmp_path)
        assert line['subject_task_id'] == 'stubbed', (
            'the journal calls the shared helper rather than its own copy'
        )
        assert seen == ['journal line'], 'and names its own record kind for the log'

    def test_the_ladder_falls_from_the_thunk_to_the_worktree_to_the_sentinel(self):
        def boom() -> str:
            raise RuntimeError('no plan.json yet')

        assert markup_sink.resolve_subject(
            lambda: '4744', Path('/lanes/4744'), what='record',
        ) == '4744'
        assert markup_sink.resolve_subject(
            lambda: '', Path('/lanes/4744'), what='record',
        ) == '4744'
        assert markup_sink.resolve_subject(
            boom, Path('/lanes/4744'), what='record',
        ) == '4744', 'a raising thunk falls to the ladder rather than costing the record'
        assert markup_sink.resolve_subject(
            boom, Path(worktree_with_no_name()), what='record',
        ) == markup_sink.MARKUP_UNATTRIBUTED_SUBJECT
