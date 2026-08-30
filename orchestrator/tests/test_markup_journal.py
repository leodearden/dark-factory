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

from orchestrator.mcp import markup_journal

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


def _assert_no_raw_sentinels() -> None:
    """Fail at IMPORT if this file's own bytes carry a raw envelope literal.

    Checked against ``shared.toolcall_markup.ENVELOPE_LITERALS`` (the single
    owner of the literal set, INV-5) plus the two structural prefixes every
    built specimen uses, so a builder output spelled out by hand is caught even
    when it is not itself one of the enumerated literals.
    """
    source = Path(__file__).read_text(encoding='utf-8')
    forbidden = (*ENVELOPE_LITERALS, _LT + '/', _LT + 'parameter ')
    for sequence in forbidden:
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
