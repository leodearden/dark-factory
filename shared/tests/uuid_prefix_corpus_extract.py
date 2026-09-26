"""Generate `fixtures/uuid_prefix_corpus.jsonl` from the position paper's scan.

NEVER HAND-WRITE THE CORPUS. It is only ever produced by running this module,
for the same reason ``toolcall_markup_corpus_extract.py`` says so: every
expectation in it is the DETECTOR'S OWN ANSWER, machine-generated, and a
hand-edited row would silently assert something no run ever produced.

WHAT THE CORPUS IS FOR. It pins REGRESSION-STABILITY and DETERMINISM across
200+ real token contexts. It does NOT pin correctness — correctness lives in
the hand-authored grammar rows in ``test_uuid_prefix.py``, which a human
decided against PRD §4-C1. A machine-generated corpus that claimed correctness
would be a detector agreeing with itself.

PROVENANCE, cited and not re-measured (INV-9). The rows come from the
truncated-uuid position paper's scan of live Mem0/Graphiti stores and
read-only task-db snapshots: 47,209 8-hex occurrences and 23,631 full uuids,
with each distinct 8-hex token classified A/B/C/E. That scan is NOT re-runnable
— the stores it read have moved on and the snapshots are gone — so its two
pickles are the only source, and the COMMITTED corpus is the durable artifact.
The pickles are read from ``--occurrences`` / ``--token-class``; see the
fixture README for where they came from and where to put them.

DETERMINISM. Selection is a fixed per-stratum quota off a stable sort key, with
no RNG anywhere, so a re-run over equal input is byte-identical. Two rows are
dropped rather than guessed at: one whose token sits at window index 0 (the
paper's +/-40 truncation cut the true predecessor, so its glue verdict is
unknowable) and one whose token's position in its window is ambiguous.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from shared.uuid_prefix import find_prefix_tokens

#: The paper's classification of each distinct 8-hex token. A: a genuine
#: truncated reference to a live id. B: a reference that is also a live id in
#: its own right. C: a reference the scan could not adjudicate. E: not a
#: reference at all — a hash, a checksum, a colour, an offset. D is absent from
#: the scan's output and therefore from this vocabulary.
PAPER_CLASSES = ('A', 'B', 'C', 'E')

#: What immediately precedes the token in its window. The C1 grammar rejects a
#: token glued to a preceding ``-`` or ``_``, because ``recon-<hex>`` and
#: ``episode_<hex>`` are composite ids rather than citations, so these three
#: strata are the ones the corpus exists to keep honest.
GLUES = ('plain', 'hyphen', 'underscore')

#: Which of the paper's two occurrence tables a row came from. Full uuids are
#: sampled too, because "a full uuid is never a token" is a grammar claim that
#: deserves real specimens and not only the hand-authored row.
KINDS = ('hex8', 'uuid36')

#: One declared row schema, so a drifted writer fails a test rather than
#: producing a corpus whose rows disagree about their own shape.
ROW_KEYS = (
    'expected_own_span',
    'expected_tokens',
    'field',
    'glue',
    'kind',
    'offset',
    'source_id',
    'source_type',
    'token',
    'token_class',
    'window',
)

#: How many rows each stratum contributes at most. A stratum with fewer
#: candidates contributes all of them: the paper found only four class-B
#: occurrences in the entire corpus, and a quota that refused a short stratum
#: would drop the rarest evidence there is.
QUOTA = 24

#: The paper's window is ``text[start-40:end+40]``, so an untruncated one puts
#: the token at exactly this index. Named because two separate reads depend on
#: it and a bare 40 in either would be unreadable.
WINDOW_RADIUS = 40

DEFAULT_OUT = Path(__file__).resolve().parent / 'fixtures' / 'uuid_prefix_corpus.jsonl'

#: Where the rescued pickles are expected to live. The scan's own scratch
#: directory is under ``/tmp`` and therefore ephemeral; the README records the
#: original path for whoever still has it.
DEFAULT_PICKLE_DIR = Path.home() / '.cache' / 'df-uuid-prefix-corpus'


class Stratum(NamedTuple):
    """The declared coverage cell a row belongs to.

    Structured rather than an encoded ``'A-hyphen'`` string: the quota, the
    row's own columns and the coverage assertions all read these fields
    directly, and none of them has to parse anything (heuristic 12).
    """

    kind: str
    token_class: str | None
    glue: str


def token_offset(window: str, token: str) -> int | None:
    """Where *token* sits in *window*, or ``None`` when that is not knowable.

    The paper stored the window but not the offset, so it is recovered here.
    An untruncated window puts the token at :data:`WINDOW_RADIUS` exactly; a
    window truncated at its start (the token was within 40 characters of the
    beginning of the source text) puts it earlier, and then the position is
    only unambiguous if the token occurs once.

    ``None`` means REFUSE, not "assume the first match": a token appearing
    twice in one window would otherwise get an expectation generated against
    the wrong span, and every assertion downstream would agree with it.
    """
    if window[WINDOW_RADIUS : WINDOW_RADIUS + len(token)] == token:
        return WINDOW_RADIUS
    if window.count(token) == 1:
        return window.index(token)
    return None


def glue_of(window: str, offset: int) -> str:
    """Which glue stratum the character before *offset* puts this row in."""
    return {'-': 'hyphen', '_': 'underscore'}.get(window[offset - 1], 'plain')


def expectations(window: str, offset: int, token: str) -> dict[str, Any]:
    """The detector's own answers for one window — the machine-generated half.

    Two of them, because they answer different questions. ``expected_own_span``
    is the verdict for the paper's EXACT span, which is what the glue strata
    exist to check; ``expected_tokens`` is the whole document-ordered tuple,
    which is what makes an unrelated drift in the same window visible.
    """
    found = [
        {'token': t.token, 'start': t.start, 'end': t.end}
        for t in find_prefix_tokens({'content': window})
    ]
    span = (offset, offset + len(token))
    return {
        'expected_own_span': any((t['start'], t['end']) == span for t in found),
        'expected_tokens': found,
    }


def _row(record: Mapping[str, Any], window: str, offset: int, stratum: Stratum) -> dict[str, Any]:
    token = record['token']
    return {
        'source_type': record['source_type'],
        'source_id': record['source_id'],
        'field': record['field'],
        'token': token,
        'window': window,
        'offset': offset,
        'kind': stratum.kind,
        'token_class': stratum.token_class,
        'glue': stratum.glue,
        **expectations(window, offset, token),
    }


def hex8_rows(
    records: Iterable[Mapping[str, Any]], token_class: Mapping[str, str]
) -> Iterator[tuple[Stratum, dict[str, Any]]]:
    """Candidate rows from the paper's 8-hex occurrence table.

    The window is the paper's own ``text[start-40:end+40]`` slice, kept
    verbatim — a REAL context with the token embedded in it, which is the whole
    point of replaying a corpus rather than more synthetic strings.
    """
    for record in records:
        window, token = record['window'], record['token']
        offset = token_offset(window, token)
        if not offset:
            # 0 is dropped alongside None: the +/-40 truncation cut the true
            # predecessor, so the glue verdict for that span is unknowable and
            # an expectation there would be a guess wearing a measurement's
            # clothes.
            continue
        paper_class = token_class.get(token)
        if paper_class not in PAPER_CLASSES:
            continue
        stratum = Stratum('hex8', paper_class, glue_of(window, offset))
        yield stratum, _row(record, window, offset, stratum)


def uuid36_rows(records: Iterable[Mapping[str, Any]]) -> Iterator[tuple[Stratum, dict[str, Any]]]:
    """Candidate rows from the paper's full-uuid table.

    That table stored a wider ``context`` and no window, so a comparable
    +/-40 window is cut here. ``token_class`` is ``None``: the paper classified
    8-hex tokens, and a full uuid has no class to carry — inventing one would
    put a value in the column that no measurement produced.
    """
    for record in records:
        context, token = record['context'], record['token']
        if context.count(token) != 1:
            continue
        found = context.index(token)
        start = max(0, found - WINDOW_RADIUS)
        window = context[start : found + len(token) + WINDOW_RADIUS]
        offset = found - start
        if not offset:
            continue
        stratum = Stratum('uuid36', None, glue_of(window, offset))
        yield stratum, _row(record, window, offset, stratum)


def _sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """The stable selection order. No RNG, so a re-run is byte-identical."""
    return (row['source_type'], row['source_id'], row['field'], row['token'], row['offset'])


def select(
    candidates: Iterable[tuple[Stratum, dict[str, Any]]], quota: int = QUOTA
) -> list[dict[str, Any]]:
    """Take up to *quota* rows per stratum, in the stable order, then sort.

    Stratified rather than a head slice of the whole table, because the table
    is 70% class E and a head slice would carry almost no glue evidence at all
    — the strata that pin the rule the grammar is most likely to lose.
    """
    by_stratum: dict[Stratum, list[dict[str, Any]]] = {}
    for stratum, row in candidates:
        by_stratum.setdefault(stratum, []).append(row)
    chosen = [
        row
        for stratum in sorted(by_stratum)
        for row in sorted(by_stratum[stratum], key=_sort_key)[:quota]
    ]
    return sorted(chosen, key=_sort_key)


# ---------------------------------------------------------------------------
# Serialization.
# ---------------------------------------------------------------------------


def write_corpus(records: Sequence[Mapping[str, Any]], out: Path | str) -> None:
    """Write *records* as JSONL with every ``\\x3c`` escaped as ``\\u003c``.

    ``sort_keys=True`` plus the fixed separators make repeated runs over equal
    input byte-identical, which is what makes a regenerated corpus reviewable:
    the diff is the rows that changed and nothing else.

    The escaping is standard JSON — ``json.loads`` decodes it transparently —
    and it is what keeps the emitted text free of an opening angle bracket, so
    a row quoted back into a tool call cannot carry an envelope literal. The
    markup corpus is written under the same rule.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record, sort_keys=True, separators=(', ', ': ')).replace('\x3c', '\\u003c')
        for record in records
    ]
    out.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')


def load_corpus(path: Path | str) -> list[dict[str, Any]]:
    """Read a corpus JSONL back. ``\\u003c`` decodes transparently."""
    text = Path(path).read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the uuid-prefix specimen corpus from the position paper's scan.",
        epilog=(
            'The two pickles are NOT in the repo and NOT re-derivable: the scan read '
            'live stores and snapshots that no longer exist. Point --occurrences and '
            '--token-class at your rescued copies; the fixture README records where '
            'the originals were found.'
        ),
    )
    parser.add_argument('--occurrences', type=Path, default=DEFAULT_PICKLE_DIR / 'occurrences.pkl')
    parser.add_argument('--token-class', type=Path, default=DEFAULT_PICKLE_DIR / 'token_class.pkl')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--quota', type=int, default=QUOTA)
    parser.add_argument('--minimum-rows', type=int, default=200)
    return parser


def _load_pickle(path: Path) -> Any:
    with path.open('rb') as handle:
        return pickle.load(handle)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    for path in (args.occurrences, args.token_class):
        if not path.exists():
            print(f'ERROR: {path} not found — nothing written', file=sys.stderr)
            return 1

    occurrences = _load_pickle(args.occurrences)['occurrences']
    token_class = _load_pickle(args.token_class)['token_class']

    candidates = [
        *hex8_rows(occurrences['hex8'], token_class),
        *uuid36_rows(occurrences['uuid36']),
    ]
    rows = select(candidates, args.quota)

    if len(rows) < args.minimum_rows:
        # LOUD, not silent. A short corpus still passes its own replay — every
        # expectation would agree — while quietly covering far less than the
        # file claims, which is the one failure this generator can produce that
        # nothing downstream would catch.
        print(
            f'ERROR: {len(rows)} rows selected, below the {args.minimum_rows} floor '
            '— nothing written (check the pickles and --quota)',
            file=sys.stderr,
        )
        return 1

    write_corpus(rows, args.out)
    print(f'wrote {len(rows)} rows to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
