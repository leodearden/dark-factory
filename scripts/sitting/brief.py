"""Render the sitting brief: numbered decisions, the standing footer, done items, carve-out closes and docket rows.

Pure string computation with no I/O and no clock (``generated_at`` is injected),
the contract of ``orchestrator/src/orchestrator/digest.py::render_digest_markdown``.
Release predicates are re-probed by the caller, which passes each measured
outcome in as a :class:`ReleaseProbe`.

Every structured citation renders through :func:`cite`. Within each rendered
entry, the first mention of an escalation id in free text is glossed as well
(amendment B2). An id inside a fenced block is glossed on a line before the
fence, so the fence itself stays verbatim.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from sitting.gates import RECOMMEND_ONLY, Verdict
from sitting.inventory import ESC_ID_RE, Glossary, ItemKey, OpenItem
from sitting.ownership import OwnershipFinding
from sitting.payloads import ApplyPayload
from sitting.preparation import Manual, NoLean, Preparation, Standing, TaskStatusIs

AUDIT_SAMPLE_EVERY = 5
"""``docs/escalation-standing-policy.md``: Leo audits at least one in five autonomous rulings."""

DOCKET_MIN_ITEMS = 6
DOCKET_MIN_HEAVY_ITEMS = 3
DOCKET_HEAVY_OPTIONS = 3
"""The docket-page threshold Leo ratified in the task's second 2026-09-25 amendment."""

STANDING_LABELS = {'pin': 'PIN', 'hold': 'HOLD', 'leo_owned': 'Leo-owned', 'owned': 'owned elsewhere'}

_FREE_ESC_RE = re.compile(r'(?<![\w-])' + ESC_ID_RE.pattern)
_LINE_HAZARD_RE = re.compile(r'^(\s{0,3})(#|`{3,}|~{3,}|=+\s*$|-+\s*$)')
_BLANK_LINE_RE = re.compile(r'\n\s*\n')


@dataclass(frozen=True)
class Citation:
    """An id to gloss, with the scope it is unique in: the queue dir for an escalation, the project for a task."""

    kind: Literal['esc', 'task']
    id: str
    scope: str


@dataclass(frozen=True)
class ReleaseProbe:
    """A release predicate as the caller re-probed it: the state measured, and whether that releases the item."""

    observed: str
    released: bool
    measured_at: str


@dataclass(frozen=True)
class BriefEntry:
    number: int
    item: OpenItem
    preparation: Preparation | None
    ownership: OwnershipFinding
    payloads: tuple[ApplyPayload, ...] = ()

    def __post_init__(self) -> None:
        if self.ownership.item_key != self.item.key:
            raise ValueError(f'item {self.number}: the ownership finding describes {self.ownership.item_key}')
        if self.ownership.owned:
            raise ValueError(f'item {self.number} is owned; it belongs in the standing footer')
        if self.preparation is not None and self.preparation.item_key != self.item.key:
            raise ValueError(f'item {self.number}: the preparation describes {self.preparation.item_key}')
        if self.preparation is not None and self.preparation.standing is not None:
            raise ValueError(f'item {self.number} is prepared as standing; it belongs in the standing footer')


@dataclass(frozen=True)
class StandingEntry:
    number: int
    item: OpenItem
    standing: Standing
    release: ReleaseProbe | None
    routing: ApplyPayload | None = None

    def __post_init__(self) -> None:
        if isinstance(self.standing.release_predicate, Manual) and self.release is not None:
            raise ValueError(f'item {self.number}: a Manual release is not machine-probeable and carries no probe')


@dataclass(frozen=True)
class DoneEntry:
    number: int
    key: ItemKey
    done_at: str


@dataclass(frozen=True)
class CloseRecord:
    item: OpenItem
    verdict: Verdict
    payloads: tuple[ApplyPayload, ...] = ()

    def __post_init__(self) -> None:
        if not self.verdict.all_held:
            raise ValueError(f'a close needs all six gates held; missed {list(self.verdict.missed_gates)}')


def cite(ref: Citation, glossary: Glossary) -> str:
    label = ref.id if ref.kind == 'esc' else f'task {ref.id}'
    return f'{label} ({_gloss(ref, glossary)})'


def render_brief(
    numbered: Sequence[BriefEntry],
    standing: Sequence[StandingEntry],
    done: Sequence[DoneEntry],
    *,
    glossary: Glossary,
    generated_at: str,
) -> str:
    decisions = [_entry_lines(entry, glossary) for entry in sorted(numbered, key=_number)]
    footer = [_standing_lines(entry, glossary) for entry in sorted(standing, key=_number)]
    finished = [[_done_line(entry, glossary)] for entry in sorted(done, key=_number)]
    return _document([
        '# Sitting brief', '', f'generated {generated_at}', '',
        *_section('Decisions needed', decisions, 'No decisions needed.'),
        *_section('Standing / no action', footer, 'Nothing standing.'),
        *_section('Done', finished, 'Nothing done yet.'),
    ])


def render_closes(closes: Sequence[CloseRecord], *, glossary: Glossary) -> str:
    """Each close with its six gates and their evidence verbatim; every ``AUDIT_SAMPLE_EVERY``-th is sampled."""
    recommend_only = all(close.verdict.demoted_by == RECOMMEND_ONLY for close in closes)
    title = 'Recommended closes (recommend-only)' if recommend_only else 'Closes'
    blocks = [_close_lines(n, close, glossary) for n, close in enumerate(closes, start=1)]
    return _document(_section(title, blocks, 'No closes.'))


def docket_reason(entries: Sequence[BriefEntry], *, multi_sitting: bool) -> str | None:
    """Why the docket page is warranted, or None below the threshold."""
    if multi_sitting:
        return 'the decisions span more than one sitting'
    if len(entries) >= DOCKET_MIN_ITEMS:
        return f'{len(entries)} decisions (threshold {DOCKET_MIN_ITEMS})'
    heavy = [entry.number for entry in entries if _is_heavy(entry)]
    if len(heavy) >= DOCKET_MIN_HEAVY_ITEMS:
        return (f'{len(heavy)} heavy decisions, each with {DOCKET_HEAVY_OPTIONS}+ multi-paragraph ramifications '
                f'(items {", ".join(map(str, heavy))})')
    return None


def needs_docket(entries: Sequence[BriefEntry], *, multi_sitting: bool) -> bool:
    return docket_reason(entries, multi_sitting=multi_sitting) is not None


def docket_rows(entries: Sequence[BriefEntry]) -> list[dict[str, Any]]:
    """One row per numbered entry; ``decision`` and ``note`` are the cells Leo fills."""
    return [
        {
            'item': entry.number,
            'esc_id': entry.item.escalation_id or entry.item.decision_id or '',
            'issue': entry.preparation.question if entry.preparation else entry.item.text,
            'options': [
                {'label': label, 'text': text, 'ramification': ramification}
                for label, text, ramification in _options(entry)
            ],
            'recommendation': _recommendation_cell(entry.preparation),
            'decision': '',
            'note': '',
        }
        for entry in sorted(entries, key=_number)
    ]


def options_by_label(entry: BriefEntry) -> dict[str, str]:
    """``{label: text}`` exactly as the brief shows the entry's options, for resolving Leo's answers."""
    return {label: text for label, text, _ in _options(entry)}


class _Glosser:
    """Glosses each escalation id on its first mention within one rendered entry."""

    def __init__(self, queue_dir: str, glossary: Glossary) -> None:
        self._queue_dir = queue_dir
        self._glossary = glossary
        self._seen: set[str] = set()

    def cite(self, ref: Citation) -> str:
        if ref.kind == 'esc':
            self._seen.add(ref.id)
        return cite(ref, self._glossary)

    def label(self, key: ItemKey) -> str:
        if key[0] != 'esc':
            return f'decision {key[-1]}'
        _, queue_dir, esc_id = key
        return self.cite(Citation('esc', esc_id, queue_dir))

    def text(self, text: str) -> str:
        """Prose with markdown hazards neutralised and each unseen escalation id glossed in place."""
        return _FREE_ESC_RE.sub(self._first_mention, _neutralise(text))

    def fenced(self, text: str, info: str = '') -> str:
        """*text* verbatim in a fence, preceded by a gloss line for any escalation id not yet mentioned."""
        unseen = [esc_id for esc_id in dict.fromkeys(_FREE_ESC_RE.findall(text)) if esc_id not in self._seen]
        glosses = [f'ids below: {", ".join(self.cite(Citation("esc", e, self._queue_dir)) for e in unseen)}']
        return '\n'.join([*(glosses if unseen else []), _fence_safe(text, info)])

    def _first_mention(self, match: re.Match[str]) -> str:
        esc_id = match.group(0)
        return esc_id if esc_id in self._seen else self.cite(Citation('esc', esc_id, self._queue_dir))


def _gloss(ref: Citation, glossary: Glossary) -> str:
    if ref.kind == 'esc':
        scoped = glossary.escalations.get(ref.scope)
        no_scope, no_id = f'queue {ref.scope} not glossed', f'not in queue {ref.scope}'
    else:
        scoped = glossary.tasks.get(ref.scope)
        no_scope = f'no tasks.db titles read for {ref.scope or "an unknown project"}'
        no_id = f'not in the {ref.scope} tasks.db'
    if scoped is None:
        return f'no gloss: {no_scope}'
    if ref.id not in scoped:
        return f'no gloss: {no_id}'
    return ' '.join(scoped[ref.id].split()) or 'no gloss: its summary is empty'


def _entry_lines(entry: BriefEntry, glossary: Glossary) -> list[str]:
    item, prep = entry.item, entry.preparation
    g = _Glosser(item.queue_dir, glossary)
    lines = [f'### {entry.number}. {g.label(item.key)}', '']
    if prep is None:
        lines += ['_awaiting preparation_', '', g.text(item.text), '', *_option_lines(entry, g)]
    else:
        lines += [g.text(prep.question), '', *_option_lines(entry, g), '', _recommendation_line(prep, g)]
        lines += ['', f'**On apply:** {g.text(prep.on_apply)}']
    lines += _supersedes_lines(entry.payloads, g)
    if prep is not None:
        lines += _cites_lines(prep, item, g)
    return [*lines, '', *_ownership_lines(entry.ownership, g), *_payload_lines(entry.payloads, g)]


def _options(entry: BriefEntry) -> list[tuple[str, str, str]]:
    """``(label, text, ramification)``: the prepared options, or the raw ones lettered A, B, ... with none."""
    if entry.preparation is not None:
        return [(o.label, o.text, o.ramification) for o in entry.preparation.options]
    return [(_raw_label(n), text, '') for n, text in enumerate(entry.item.options)]


def _raw_label(index: int) -> str:
    return chr(ord('A') + index) if index < 26 else str(index + 1)


def _option_lines(entry: BriefEntry, g: _Glosser) -> list[str]:
    lines = []
    for label, text, ramification in _options(entry):
        lines.append(f'- **{label}** — {_indent(g.text(text), "  ")}')
        if ramification:
            lines.append(f'  - _ramification:_ {_indent(g.text(ramification), "    ")}')
    return lines


def _recommendation_line(prep: Preparation, g: _Glosser) -> str:
    recommendation = prep.recommendation
    if isinstance(recommendation, NoLean):
        return f'No lean — {g.text(recommendation.reason)}'
    return f'**Recommendation:** {recommendation.option_label} — {g.text(recommendation.evidence_chain)}'


def _recommendation_cell(prep: Preparation | None) -> str:
    if prep is None:
        return 'awaiting preparation'
    if isinstance(prep.recommendation, NoLean):
        return f'No lean — {prep.recommendation.reason}'
    return prep.recommendation.option_label


def _supersedes_lines(payloads: Iterable[ApplyPayload], g: _Glosser) -> list[str]:
    lines = []
    for payload in payloads:
        if payload.supersedes:
            lines += ['', f'`{payload.tool}` replaces, verbatim:', g.fenced(payload.supersedes)]
    return lines


def _cites_lines(prep: Preparation, item: OpenItem, g: _Glosser) -> list[str]:
    refs = [Citation('esc', esc_id, item.queue_dir) for esc_id in prep.cites.escalation_ids]
    refs += [Citation('task', task_id, item.project) for task_id in prep.cites.task_ids]
    return ['', f'Cites: {", ".join(g.cite(ref) for ref in refs)}'] if refs else []


def _ownership_lines(finding: OwnershipFinding, g: _Glosser) -> list[str]:
    """Amendment A's line: which probes came back empty, which mention the item, and which could not be read."""
    if not finding.probes:
        return ['ownership: not swept']
    mentions = [r for r in finding.probes if r.status == 'mentions']
    unavailable = [r for r in finding.probes if r.status == 'unavailable']
    parts = [f'checked {", ".join(finding.empty_probes)}: empty'] if finding.empty_probes else []
    parts += [f'{r.probe}: mentions in {r.owner}' for r in mentions]
    parts += [f'could not check {r.probe}: {" ".join(r.evidence.split())}' for r in unavailable]
    prefix = 'no owner found, but the sweep was partial' if unavailable else 'nothing owns this'
    lines = [g.text(f'{prefix}: {"; ".join(parts)}')]
    for result in mentions:
        lines += ['', *(f'> {line}' for line in g.text(result.evidence).splitlines())]
    return lines


def _payload_lines(payloads: Iterable[ApplyPayload], g: _Glosser) -> list[str]:
    lines = []
    for payload in payloads:
        body = json.dumps({'tool': payload.tool, 'args': payload.args}, indent=2, ensure_ascii=False)
        ask = ' (put to Leo before applying)' if payload.put_to_leo else ''
        lines += ['', f'Apply `{payload.tool}`{ask}:', g.fenced(body, 'json')]
    return lines


def _standing_lines(entry: StandingEntry, glossary: Glossary) -> list[str]:
    g = _Glosser(entry.item.queue_dir, glossary)
    standing = entry.standing
    lines = [
        f'- **{entry.number}.** {g.label(entry.item.key)} — {STANDING_LABELS[standing.kind]}, '
        f'owner {g.text(standing.owner)}',
        f'  - release: {_release_text(entry, g)}',
        f'  - evidence: {_indent(g.text(standing.evidence), "    ")}',
    ]
    if entry.routing is not None:
        lines += _payload_lines([entry.routing], g)
    return lines


def _release_text(entry: StandingEntry, g: _Glosser) -> str:
    predicate, probe = entry.standing.release_predicate, entry.release
    if isinstance(predicate, Manual):
        return f'not machine-probeable — re-probe by hand: {g.text(predicate.text)}'
    if isinstance(predicate, TaskStatusIs):
        wanted = f'{g.cite(Citation("task", predicate.task_id, entry.item.project))} reaches ' + ' or '.join(
            predicate.statuses)
    else:
        wanted = f'{g.cite(Citation("esc", predicate.esc_id, entry.item.queue_dir))} closes'
    if probe is None:
        return f'{wanted} — not measured'
    outcome = 'RELEASED' if probe.released else 'still held'
    return f'{wanted} — measured {probe.observed} at {probe.measured_at}: {outcome}'


def _done_line(entry: DoneEntry, glossary: Glossary) -> str:
    queue_dir = entry.key[1] if entry.key[0] == 'esc' else ''
    return f'- **{entry.number}.** {_Glosser(queue_dir, glossary).label(entry.key)} — done at {entry.done_at}'


def _close_lines(n: int, close: CloseRecord, glossary: Glossary) -> list[str]:
    g = _Glosser(close.item.queue_dir, glossary)
    sample = f' — AUDIT SAMPLE (1 in {AUDIT_SAMPLE_EVERY})' if n % AUDIT_SAMPLE_EVERY == 0 else ''
    lines = [f'### Close {n}: {g.label(close.item.key)}{sample}']
    for gate in close.verdict.gates:
        note = f' — {g.text(gate.note)}' if gate.note else ''
        state = 'held' if gate.held else 'MISSED'
        lines += ['', f'gate {gate.number} {gate.name}: {state}{note}', g.fenced(gate.evidence)]
    return [*lines, *_payload_lines(close.payloads, g)]


def _is_heavy(entry: BriefEntry) -> bool:
    if entry.preparation is None:
        return False
    paragraphs = [o for o in entry.preparation.options if _BLANK_LINE_RE.search(o.ramification)]
    return len(paragraphs) >= DOCKET_HEAVY_OPTIONS


def _number(entry: BriefEntry | StandingEntry | DoneEntry) -> int:
    return entry.number


def _section(title: str, blocks: list[list[str]], empty: str) -> list[str]:
    body = [line for block in blocks for line in (*block, '')] or [empty, '']
    return [f'## {title}', '', *body]


def _document(lines: list[str]) -> str:
    return '\n'.join(lines).rstrip('\n') + '\n'


def _neutralise(text: str) -> str:
    """Escape each line markdown would read as a heading, a fence or a setext underline."""
    return '\n'.join(_LINE_HAZARD_RE.sub(r'\1\\\2', line) for line in text.strip().splitlines())


def _indent(text: str, prefix: str) -> str:
    return f'\n{prefix}'.join(text.splitlines())


def _fence_safe(text: str, info: str = '') -> str:
    """*text* in a code fence longer than any backtick run inside it, so no content can close it early."""
    longest = max((len(run) for run in re.findall(r'`+', text)), default=0)
    fence = '`' * max(3, longest + 1)
    return f'{fence}{info}\n{text}\n{fence}'
