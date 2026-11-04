"""A predicate check's raw output, read into the two records the runner keeps:
a bounded ``done_provenance`` note and the structured ``predicate_verdict``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = [
    'PREDICATE_NOTE_MAX_PAYLOAD_CHARS',
    'predicate_verdict',
    'summarize_predicate_output',
]


# The payload budget.  Sits safely inside the 500-char `max_note_chars` cap
# that fused-memory's `_format_outcome_echo` applies downstream (tasks
# 2049/2054/2080), so a surviving note is not re-truncated there.
PREDICATE_NOTE_MAX_PAYLOAD_CHARS = 400

# Log shape: a leading ISO/logging timestamp, OR a standalone level token
# anywhere in the line.  Used only to REJECT a tier-2 candidate line.
_LOG_LINE_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}'
    r'|\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b'
)


def summarize_predicate_output(out: object, *, rc: int) -> str:
    """Summarize a predicate check's raw output into a bounded provenance note.

    Anything the extractor does not recognize is dropped, leaving the bare
    verdict prefix — an unrecognized shape therefore yields a less
    informative note, never a corrupted one.

    Motivating specimen — task 2902.  ``before_done_script.run_script`` merges
    stderr into stdout and returns ``decode()[-2000:]``, so a chatty predicate
    script's server-log noise reached ``done_provenance.note`` verbatim: a
    1999-char blob starting mid-token, carrying FalkorDB identity-scan
    WARNINGs for an unrelated project and ``httpx`` request lines.

    Why this matters beyond tidiness: ``note`` is not a private field.
    fused-memory's reconciliation ``_format_outcome_echo`` reads
    ``done_provenance['note']`` and appends it to the ``"Task '<title>'
    completed."`` Mem0 write, so whatever lands here is INGESTED INTO MEMORY.
    Raw subprocess output is dropped at this seam for that reason alone; it
    stays recoverable in the orchestrator log, which is not memory-ingested.

    Extraction runs two tiers with DIFFERENT guarantees — do not read the
    whole of it as an allowlist.  Tier 1 (the script's own trailing JSON
    block) IS a true allowlist: only a parseable structured payload survives,
    and every preceding log line is excluded STRUCTURALLY rather than by
    pattern-matching.  Tier 2 (a single clean final line) is a best-effort
    heuristic guarded by a log-shape DENYLIST (``_LOG_LINE_RE``), and it
    inherits a denylist's weakness: a log line under a formatter nobody
    anticipated still reaches the note.  Concretely, a ``%(name)s
    %(message)s`` line such as ``httpx HTTP Request: GET http://...`` carries
    neither timestamp nor level token and is kept verbatim.

    Tier 2 cannot be tightened into a grammar-based allowlist without
    discarding the very verdicts it exists to preserve: the real in-repo
    predicate outputs ``check ok: 0 flakes`` and
    ``check_merge_flakiness: ... -- invariant holds`` are prose-shaped and
    structurally indistinguishable from a formatter-less log line.  The
    residual exposure is therefore bounded rather than closed: at most ONE
    line, at most ``PREDICATE_NOTE_MAX_PAYLOAD_CHARS`` of it — never the
    multi-KB blob task 2902 stamped.  ``scripts/scan_provenance_note_log_leaks.py``
    shares this blind spot (its discriminator also requires a timestamp), so
    a predicate script that wants its verdict preserved intact should emit
    trailing JSON — tier 1 is the only tier with a real guarantee.

    The denylist rejection is otherwise deliberately CONSERVATIVE and will
    drop an otherwise clean final line that merely contains a standalone
    ``INFO``/``ERROR`` token.  Losing a payload is the safe failure
    direction: the verdict prefix always survives, and a caller that must
    keep the raw output logs it before calling this — this function returns
    a note, it does not decide what else is kept.

    An oversized payload is replaced WHOLESALE with a marker naming the
    dropped size — never sliced.  Task 2054 found a raw ``note[:N]`` slice
    downstream cutting mid-token (garbling ``8679,8680`` into ``8679,868``);
    a sliced JSON object is worse still, unparseable yet still
    structured-looking to a reader.

    *rc* is rendered as both a word and a number, and the word is DERIVED
    from the number (``passed`` iff ``rc == 0``).  A hardcoded ``passed``
    would let a caller with a non-zero rc stamp the self-contradictory note
    ``predicate check passed (rc=2)``; the text cannot contradict the value.
    """
    verdict = f'predicate check {"passed" if rc == 0 else "failed"} (rc={rc})'
    parsed = _parse_predicate_payload(out)
    if parsed is None:
        return verdict
    payload = parsed.text
    if len(payload) > PREDICATE_NOTE_MAX_PAYLOAD_CHARS:
        payload = (
            f'<verdict payload elided: {len(payload)} chars exceeds '
            f'{PREDICATE_NOTE_MAX_PAYLOAD_CHARS}-char cap '
            f'— see the orchestrator log>'
        )
    return f'{verdict}: {payload}'


@dataclass(frozen=True)
class _PredicatePayload:
    """The recognized payload of a predicate's output: a trailing JSON block or a clean line.

    Exactly one of ``json_value`` (tier 1) and ``line`` (tier 2) is set.
    """

    json_value: dict | list | None = None
    line: str | None = None

    def __post_init__(self) -> None:
        if (self.json_value is None) == (self.line is None):
            raise ValueError(
                '_PredicatePayload needs exactly one of json_value and line '
                f'(got json_value={self.json_value!r}, line={self.line!r})'
            )

    @property
    def text(self) -> str:
        """The single-line rendering the provenance note carries."""
        if self.line is not None:
            return self.line
        return json.dumps(self.json_value, separators=(',', ':'))


def _parse_predicate_payload(out: object) -> _PredicatePayload | None:
    """Return the recognized structured payload in *out*, or None.

    Tier 1 — a trailing JSON block: walk backwards to the last line whose
    strip opens a ``{``/``[``, and try to parse from there to end of output.
    Its rendering re-dumps it with compact separators so the note stays a
    single line.

    Tier 2 — the last non-blank line, iff it carries no log shape
    (``_LOG_LINE_RE``).  This is what preserves the real in-repo predicate
    verdicts (``check ok: 0 flakes``, ``-- invariant holds``,
    ``measured_median_ms=…``).

    Otherwise None: an unrecognized shape yields no payload at all.
    """
    lines = out.splitlines() if isinstance(out, str) else []
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].strip().startswith(('{', '[')):
            continue
        try:
            parsed = json.loads('\n'.join(lines[index:]))
        except ValueError:
            continue
        return _PredicatePayload(json_value=parsed)

    for line in reversed(lines):
        candidate = line.strip()
        if not candidate:
            continue
        return None if _LOG_LINE_RE.search(candidate) else _PredicatePayload(line=candidate)
    return None


def predicate_verdict(out: object) -> dict | None:
    """The structured verdict a predicate's output carries: its trailing JSON OBJECT.

    Stamped as ``metadata.predicate_verdict`` (plans/completion-driven-triggers-prd.md
    C1).  A clean line or a JSON array is never a verdict.  An object whose
    compact rendering exceeds the note's cap is recorded by size alone.
    """
    payload = _parse_predicate_payload(out)
    if payload is None or not isinstance(payload.json_value, dict):
        return None
    rendered = payload.text
    if len(rendered) > PREDICATE_NOTE_MAX_PAYLOAD_CHARS:
        return {'elided': True, 'chars': len(rendered)}
    return payload.json_value

