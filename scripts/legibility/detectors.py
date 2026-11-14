"""Per-record confusion-signal detectors, shared by the sampler and the digest.

One record in, one :class:`RecordHits` out (:func:`record_hits`): which of
the record's native carriers count as this session's own dialogue, which
pattern of which signal class each one holds, and what every structured
error in it means. Both readers reduce these hits, and both take a
session's agent class from :func:`classify_agent_class`, so neither can
disagree with the other (plans/confusion-reduction-prd.md §7.2.4). Imports
only :mod:`legibility.reingestion`.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from legibility.reingestion import (
    TRICKLE_CODER_PROMPT_OPENER,
    has_non_human_origin,
    is_reingested_content,
)


class SignalClass(StrEnum):
    """The signal classes, in the frontmatter's ``signal_counts`` order."""

    TOOL_ERROR = 'tool_error'
    SELF_CORRECT = 'self_correct'
    NOT_FOUND = 'not_found'
    DF_GUARD = 'df_guard'
    INTERRUPT = 'interrupt'
    DESIGNED_OUTCOME = 'designed_outcome'


CONFUSION_CLASSES: frozenset[SignalClass] = frozenset(SignalClass) - {SignalClass.DESIGNED_OUTCOME}
"""The classes that are agent confusion. A designed outcome is reported, but
a bounded-poll ceiling is a designed loop-continuation, so it never adds to
a confusion score (PRD §7.2.1)."""


class CarrierKind(StrEnum):
    """A record's native text carriers, in the order the digest lists hits."""

    TOOL_RESULT = 'tool_result'
    ASSISTANT_TEXT = 'assistant_text'
    USER_TEXT = 'user_text'


def message_content(record: Mapping[str, Any]) -> Any:
    """Return a record's ``message.content``, or None if absent/malformed."""
    message = record.get('message')
    if not isinstance(message, dict):
        return None
    return message.get('content')


def user_turn_text(record: Mapping[str, Any]) -> str | None:
    """The text a user record's own turn carries.

    String content is the turn. A block list contributes only its 'text'
    blocks, joined by newlines, so a record holding only tool_result blocks
    (an answer TO the agent, not a turn) has no text.
    """
    content = message_content(record)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block_text for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
            and isinstance(block_text := block.get('text'), str)
        ]
        if texts:
            return '\n'.join(texts)
    return None


def _flatten_content(content: Any) -> str:
    """Flatten a tool_result ``content`` field (a string or a block list) to
    text. Blocks without a 'text' field (e.g. the ``tool_reference`` blocks
    ToolSearch results carry) contribute nothing rather than being guessed
    at."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block_text for block in content
            if isinstance(block, dict) and isinstance(block_text := block.get('text'), str)
        ]
        return '\n'.join(parts)
    return ''


def _raw_carriers(record: Mapping[str, Any]) -> Iterator[tuple[CarrierKind, str]]:
    """Yield *record*'s native text carriers, unfiltered, in block order.

    A user record yields each tool_result block's flattened content and then,
    unless it is a sidechain turn, its own turn text (isMeta injections
    included). An assistant record yields each 'text' block. NEVER a
    tool_use ``input``: an assistant-authored file mutation is not a native
    signal carrier, so a decoy string planted there is excluded by
    construction (PRD Sec 13.2), nor a 'thinking' block.
    """
    content = message_content(record)
    if record.get('type') == 'user':
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    yield CarrierKind.TOOL_RESULT, _flatten_content(block.get('content'))
        if not record.get('isSidechain'):
            text = user_turn_text(record)
            if text:
                yield CarrierKind.USER_TEXT, text
    elif record.get('type') == 'assistant' and isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text')
                if isinstance(text, str):
                    yield CarrierKind.ASSISTANT_TEXT, text


def is_non_dialogue_carrier(record: Mapping[str, Any], text: str) -> bool:
    """The ONE question asked of a carrier before it counts as this
    session's own dialogue: the record half is provenance, the text half is
    content. The record half covers every carrier the record holds,
    tool_results included.

    Dropping is WHOLE-carrier on every carrier kind, so a genuine literal
    printed beside a harness prompt goes with it. That cost is accepted on
    measurement: across the 2026-09-23 dark-factory corpus the harness rule
    dropped 59 non-user carriers with signal hits, each one foreign material
    (a Read of a module holding the marker literals, a task record, a dump
    of another session). Merely NAMING a harness heading drops nothing,
    because the heading rules are line-anchored.
    """
    return has_non_human_origin(record) or is_reingested_content(text)


DECOY_MARKER = '# decoy-fail'
"""Inline same-line sentinel: a fixture/test author appends this to a line
to explicitly declare "this is not a real signal, don't count it", without
having to relocate the string out of its otherwise-native carrier. Matched
case-insensitively, like every other pattern in this module."""


def _strip_decoy_lines(text: str) -> str:
    """Return *text* with every line containing DECOY_MARKER removed, so a
    same-line decoy marker suppresses only that line's occurrence."""
    marker = DECOY_MARKER.lower()
    lines = [line for line in text.split('\n') if marker not in line.lower()]
    return '\n'.join(lines)


def _line_context(text: str, pos: int) -> str:
    """Return the newline-delimited line of *text* containing offset *pos*."""
    start = text.rfind('\n', 0, pos) + 1  # rfind returns -1 when absent -> 0
    end = text.find('\n', pos)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


SELF_CORRECTION_PATTERNS: tuple[str, ...] = (
    "that's wrong",
    'let me fix',
    'my mistake',
    'i was wrong',
    'actually,',
    'correction:',
)
"""Curated self-correction markers, matched case-insensitively against
assistant TEXT blocks only (native-carrier scoping)."""

NOT_FOUND_PATTERNS: tuple[str, ...] = (
    'no such file or directory',
    'modulenotfounderror',
    'command not found',
    'does not exist',
    'not found in',
)
"""Matched case-insensitively against tool_result content only."""

DF_GUARD_PATTERNS: tuple[str, ...] = (
    'blocked:',
    'darkfactorypathscopeviolation',
    'done_gate_missing_files',
    'known-false premise about recon internals',
)
"""Real dark-factory guard TRIP literals -- never the bare, lowercase
category-name mention (e.g. ``category="scope_violation"``). Grounded in
the actual enforcement code: PathGuardVerdict's ``DarkFactoryPathScopeViolation``
error_type (fused_memory/middleware/path_scope_guard.py), the phantom-done
gate's ``done_gate_missing_files`` error code
(fused_memory/middleware/task_interceptor.py, orchestrator/scheduler.py),
and premise_lint_guard's fixed error-message prefix. ``scope_violation`` and
``phantom-done`` alone are common bare mentions (category labels, design-intent
prose/docstrings) and are deliberately NOT matched."""

INTERRUPT_PATTERN = 'request interrupted by user'
"""The literal Claude-Code-injected marker for a user-interrupted tool call
(e.g. "[Request interrupted by user for tool use]")."""


@dataclass(frozen=True)
class TextSignal:
    """A text-pattern signal: its lowercase literals, and the carrier kinds
    they are matched in."""

    patterns: tuple[str, ...]
    carriers: frozenset[CarrierKind]


TEXT_SIGNALS: Mapping[SignalClass, TextSignal] = {
    SignalClass.SELF_CORRECT: TextSignal(
        SELF_CORRECTION_PATTERNS, frozenset({CarrierKind.ASSISTANT_TEXT}),
    ),
    SignalClass.NOT_FOUND: TextSignal(NOT_FOUND_PATTERNS, frozenset({CarrierKind.TOOL_RESULT})),
    SignalClass.DF_GUARD: TextSignal(DF_GUARD_PATTERNS, frozenset(CarrierKind)),
    SignalClass.INTERRUPT: TextSignal((INTERRUPT_PATTERN,), frozenset({CarrierKind.USER_TEXT})),
}
"""Every text-pattern signal class. The other two classes come from
structured errors (:func:`error_hits`)."""

_ERROR_CLASSES = frozenset({SignalClass.TOOL_ERROR, SignalClass.DESIGNED_OUTCOME})


def _check_every_class_is_detected() -> None:
    missing = set(SignalClass) - set(TEXT_SIGNALS) - _ERROR_CLASSES
    doubled = set(TEXT_SIGNALS) & _ERROR_CLASSES
    if missing or doubled:
        raise RuntimeError(
            'every SignalClass needs exactly one detector: '
            f'undetected {sorted(missing)}, detected twice {sorted(doubled)}'
        )


_check_every_class_is_detected()


EXIT_CODE_PATTERN = re.compile(
    r'exit\s*=\s*(\d+)|exit\s+code\s+(\d+)|exit\s+status\s+(\d+)',
    re.IGNORECASE,
)
"""The three exit-code spellings observed in tool_result content: the
machine-readable ``exit=<rc>`` form emitted by scripts/watcher-rearm.sh
(:228-237) and the two prose forms ('exit code N', 'exit status N') shells
and harness wrappers produce. Case-insensitive, like every other pattern in
this module."""

BOUNDED_WAIT_EXIT_CODE = 124
"""``timeout(1)``'s ceiling exit -- the status it returns when the command
was still running when the time limit expired (as opposed to the command's
own status). In this repo a 124 is overwhelmingly produced by a deliberately
bounded poll or wait, not by a broken command."""


def _exit_codes(
    text: str, *, ignoring: tuple[tuple[int, int], ...] = (),
) -> list[int]:
    """Every exit code mentioned in *text*, in order of appearance.

    Spans in *ignoring* are skipped, which is how
    :func:`classify_error_content` tells a code a declaration already
    accounts for (the ``exit=124`` inside ``WATCHER_REARM_OUTCOME:
    CEILING exit=124``) from an UNACCOUNTED one appearing elsewhere in
    the same blob.
    """
    return [
        int(next(g for g in match.groups() if g is not None))
        for match in EXIT_CODE_PATTERN.finditer(text)
        if not any(start <= match.start() < end for start, end in ignoring)
    ]


def _extract_exit_code(text: str) -> int | None:
    """Return the FIRST exit code mentioned in *text*, or None.

    The positional fallback for content that carries no self-declaration
    to key off. :func:`classify_error_content` prefers the code attached
    to the DECIDING declaration and only falls back here when a
    declaration names none.
    """
    codes = _exit_codes(text)
    return codes[0] if codes else None


DESIGNED_OUTCOME_PATTERNS: tuple[tuple[re.Pattern[str], frozenset[str], str], ...] = (
    (
        re.compile(
            r'WATCHER_REARM_OUTCOME:\s*(?P<outcome>[A-Za-z_]+)'
            r'(?:\s+exit\s*=\s*(?P<code>\d+))?',
            re.IGNORECASE,
        ),
        frozenset({'ceiling', 'fired'}),
        'watcher-rearm-declared',
    ),
)
"""Table of (compiled regex, designed tokens, label) triples recognising a
self-DECLARED outcome, so a future contract is a one-line addition. The
regex matches EVERY outcome of its family -- designed and undesigned
alike -- and names two groups: ``outcome`` (the declared token, compared
case-insensitively against the designed set) and the optional ``code``
the declaration carries. Matching the whole family, not just the designed
tokens, is what lets :func:`classify_error_content` see a dissenting
declaration instead of stopping at the first agreeable one.

Seeded with scripts/watcher-rearm.sh's marker (:228-237), which emits four
outcomes and means different things by them: FIRED (exit=0, :228) and
CEILING (exit=124, :231) are designed loop-continuation outcomes, while
KILLED (137|143|144, :234) and ERROR (the catch-all, :237) are REAL
failures. Only the first two are in the designed set -- deliberately.
Suppressing all four because they share a marker token would trade one
over-count for an under-count and hide genuine watcher breakage, the
opposite of the 07-31 census cluster 1.3 finding's intent.

Two traps documented in that script's header (:52-73) bound what this
table can be relied on to mean:
  * usage/guard failures (exit 2 -- missing DARK_FACTORY_ROOT, unresolved
    queue dir, bad flag) return BEFORE the watcher is ever invoked and emit
    NO marker at all (:59-63), so absence of a marker is NOT absence of
    failure -- which is why a bare exit code still classifies below.
  * a SIGTERM delivered to the watcher is converted to a clean exit 0 by
    its own handler, surfacing as ``FIRED exit=0`` with EMPTY stdout
    (:66-73). It is still a designed outcome by the shell contract; a
    caller needing to know whether an escalation actually fired must check
    stdout, which is beyond what a digest signal tally can see.
"""


def _declared_exit_code(match: re.Match[str], text: str) -> int | None:
    """The code a declaration *match* carries, else the first in *text*."""
    declared = match.groupdict().get('code')
    if declared is not None:
        return int(declared)
    return _extract_exit_code(text)


def classify_error_content(content: Any) -> tuple[int | None, str | None]:
    """Classify one structured error's content as (exit_code, designed label).

    THE one classifier of the genuine/designed split (task 3610).
    *content* is a raw tool_result ``content`` field: a string or a list of
    blocks. ``label`` is None for a genuine failure; ``exit_code`` is the
    code of whatever DECIDED that verdict, so the digest renders the code
    that explains the line it is on.

    Two tiers, most authoritative first:
      1. a machine-readable self-declaration matching DESIGNED_OUTCOME_PATTERNS;
      2. a bare BOUNDED_WAIT_EXIT_CODE with no declaration at all.
    The label differs per tier so a digest reader can tell which rule fired.

    A designed verdict must hold for the WHOLE content, never just its
    first agreeable marker: one tool_result routinely carries several
    outcomes (a Bash call looping scripts/watcher-rearm.sh, or tailing its
    stderr log beside a failing command). So a tier-1 declaration
    classifies as designed only when every declaration of its family is a
    designed token AND every exit code the content mentions is one a
    declaration accounts for (or is itself the bounded-wait ceiling); a
    dissenting declaration or an unaccounted code decides genuine and
    supplies the reported code. Tier 2 applies the same rule one level
    down: a bare 124 classifies only when it is the ONLY code present.
    Every ambiguity therefore fails toward GENUINE -- the over-count
    direction, which costs one weight-1.0 noise signal, rather than the
    under-count that would hide a real failure (see
    DESIGNED_OUTCOME_PATTERNS).

    Tier 2 is a deliberate precision/recall tradeoff, and a deliberately
    cheap one to be wrong about: misclassifying a genuine un-designed
    timeout as designed costs exactly one weight-1.0 noise signal
    (tool_error is joint-lowest in the digest's SIGNAL_WEIGHTS) and never a
    gold user turn. Tightening to declaration-only, should a future census
    show real suppressions, is deleting the tier-2 branch -- the table above
    is what keeps the decision reversible rather than diffuse.
    """
    text = _flatten_content(content)
    for pattern, designed_tokens, label in DESIGNED_OUTCOME_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        designed = [
            match for match in matches
            if match.group('outcome').lower() in designed_tokens
        ]
        dissenting = [match for match in matches if match not in designed]
        # A code a DESIGNED declaration carries is explained by it; any
        # other code in the blob is unexplained evidence of a failure.
        unaccounted = [
            code
            for code in _exit_codes(text, ignoring=tuple(m.span() for m in designed))
            if code != BOUNDED_WAIT_EXIT_CODE
        ]
        if dissenting or unaccounted:
            declared = dissenting[0].groupdict().get('code') if dissenting else None
            if declared is not None:
                return int(declared), None
            return (unaccounted[0] if unaccounted else _extract_exit_code(text)), None
        return _declared_exit_code(matches[-1], text), label

    codes = _exit_codes(text)
    if codes and all(code == BOUNDED_WAIT_EXIT_CODE for code in codes):
        return BOUNDED_WAIT_EXIT_CODE, 'bounded-wait-ceiling'
    undesigned = [code for code in codes if code != BOUNDED_WAIT_EXIT_CODE]
    if undesigned:
        return undesigned[0], None
    return None, None


@dataclass(frozen=True)
class ErrorHit:
    """One structured error: a user-record tool_result block whose
    ``is_error`` flag is truthy -- never a substring match on its content
    (the decoy-FAIL suppression decision, PRD Sec 13.2)."""

    tool_use_id: Any
    content: str
    exit_code: int | None
    designed_outcome: str | None

    @property
    def signal_class(self) -> SignalClass:
        if self.designed_outcome is None:
            return SignalClass.TOOL_ERROR
        return SignalClass.DESIGNED_OUTCOME


def error_hits(record: Mapping[str, Any]) -> tuple[ErrorHit, ...]:
    """Every structured error in *record*, classified, in block order.

    Provenance does not filter these: the flag is structure the harness set,
    not text a foreign session could have quoted.
    """
    content = message_content(record)
    if record.get('type') != 'user' or not isinstance(content, list):
        return ()
    hits = []
    for block in content:
        if not (isinstance(block, dict) and block.get('type') == 'tool_result'):
            continue
        if not block.get('is_error'):
            continue
        text = _flatten_content(block.get('content'))
        exit_code, designed_outcome = classify_error_content(text)
        hits.append(ErrorHit(block.get('tool_use_id'), text, exit_code, designed_outcome))
    return tuple(hits)


@dataclass(frozen=True)
class PatternHit:
    """One text-pattern match: the lowercase *pattern*, the kind of carrier
    it was found in, and the stripped line holding its first occurrence."""

    pattern: str
    carrier: CarrierKind
    line: str


@dataclass(frozen=True)
class RecordHits:
    """Every signal one record holds: pattern hits per text class (every
    TEXT_SIGNALS class is a key) and its structured errors."""

    patterns: Mapping[SignalClass, tuple[PatternHit, ...]]
    errors: tuple[ErrorHit, ...]

    def count(self, signal: SignalClass) -> int:
        if signal in self.patterns:
            return len(self.patterns[signal])
        return sum(1 for error in self.errors if error.signal_class is signal)


def record_hits(record: Mapping[str, Any]) -> RecordHits:
    """Detect every signal class in one record.

    Each raw carrier is asked :func:`is_non_dialogue_carrier` once; a kept
    carrier is decoy-stripped and lowercased once, and every text class that
    reads its kind matches its patterns there, one hit per pattern found.
    """
    found: dict[SignalClass, list[PatternHit]] = {signal: [] for signal in TEXT_SIGNALS}
    for carrier, text in _raw_carriers(record):
        if is_non_dialogue_carrier(record, text):
            continue
        stripped = _strip_decoy_lines(text)
        lowered = stripped.lower()
        for signal, table in TEXT_SIGNALS.items():
            if carrier not in table.carriers:
                continue
            for pattern in table.patterns:
                pos = lowered.find(pattern)
                if pos != -1:
                    found[signal].append(PatternHit(pattern, carrier, _line_context(stripped, pos)))
    return RecordHits(
        patterns={signal: tuple(hits) for signal, hits in found.items()},
        errors=error_hits(record),
    )


class AgentClass(StrEnum):
    """The sampler's strata, plus UNKNOWN for an empty transcript.
    LEGIBILITY_INSTRUMENT is a session the legibility pipeline ran for
    itself, not a factory agent."""

    RECON = 'recon'
    CURATOR_CLASSIFIER = 'curator-classifier'
    WATCHER = 'watcher'
    ORCHESTRATED_TASK = 'orchestrated-task'
    INTERACTIVE = 'interactive'
    LEGIBILITY_INSTRUMENT = 'legibility-instrument'
    UNKNOWN = 'unknown'


WORKTREE_DIR_MARKERS: tuple[str, ...] = (
    '--worktrees-', '--claude-worktrees-', '-warm-lanes-worktrees-', '-reify-build-worktrees-',
)
"""Encoded-dir substrings of an orchestrated worktree. 'build-worktrees' is a
generic phrase, so it is qualified with reify's name; 'warm-lanes-worktrees'
is distinctive on its own."""


@dataclass(frozen=True)
class AgentMarker:
    """Lowercase literals that mark *agent_class*: any one of them, or all of
    them together when *require_all*."""

    agent_class: AgentClass
    literals: tuple[str, ...]
    require_all: bool = False

    def matches(self, sighted: AbstractSet[str]) -> bool:
        found = (literal in sighted for literal in self.literals)
        return all(found) if self.require_all else any(found)


AGENT_CLASS_MARKERS: tuple[AgentMarker, ...] = (
    # orchestrator/src/orchestrator/dry_run_unblock.py injects the
    # 'Task ID: {task_id}' / 'Worktree: {worktree}' preamble; its two lines
    # must co-occur, since a bare 'Worktree:' is ordinary prose here.
    AgentMarker(AgentClass.ORCHESTRATED_TASK, ('task id:', 'worktree:'), require_all=True),
    # fused_memory/reconciliation/prompts/stage{1,2,3}.py all say 'operating
    # in sleep mode'; the run headers and memory_consolidator open recon runs.
    AgentMarker(AgentClass.RECON, (
        'operating in sleep mode', '## reconciliation run', '## stage 2: task-knowledge sync',
        'memory_consolidator',
    )),
    # escalation/src/escalation/authority.py::_WATCHER_AUTO_IDENTITY, and the
    # skills/recon-escalation-watcher slash command.
    AgentMarker(AgentClass.WATCHER, ('escalation-watcher-auto', 'recon-escalation-watcher')),
    # fused_memory/middleware/task_curator.py's system prompt, the code-module
    # classifier in orchestrator/src/orchestrator/harness.py, and
    # orchestrator/src/orchestrator/agents/triage.py::TRIAGE_SYSTEM_PROMPT.
    AgentMarker(AgentClass.CURATOR_CLASSIFIER, (
        'task curator for the dark-factory orchestrator', 'code module classifier',
        'review suggestion classifier',
    )),
)
"""The one agent-class marker table, in precedence order."""

_MARKER_LITERALS = frozenset(literal for marker in AGENT_CLASS_MARKERS for literal in marker.literals)

INSTRUMENT_PROMPT_OPENERS: tuple[str, ...] = (TRICKLE_CODER_PROMPT_OPENER,)
"""Lowercase openings of the prompts the legibility pipeline sends its own
model calls. A session whose first primary user turn opens with one is
LEGIBILITY_INSTRUMENT; the same text anywhere else is foreign material."""


def is_primary_user_turn(record: Mapping[str, Any]) -> bool:
    """A user turn of the session itself: not a sidechain, not an isMeta injection."""
    return (
        record.get('type') == 'user'
        and not record.get('isSidechain')
        and not record.get('isMeta')
    )


def _sighted(text: str, literals: AbstractSet[str]) -> frozenset[str]:
    lowered = text.lower()
    return frozenset(literal for literal in literals if literal in lowered)


def _marked_class(sighted: AbstractSet[str]) -> AgentClass | None:
    return next(
        (marker.agent_class for marker in AGENT_CLASS_MARKERS if marker.matches(sighted)),
        None,
    )


def _first_turn_class(text: str) -> AgentClass | None:
    if text.lstrip().lower().startswith(INSTRUMENT_PROMPT_OPENERS):
        return AgentClass.LEGIBILITY_INSTRUMENT
    return _marked_class(_sighted(text, _MARKER_LITERALS))


@dataclass(frozen=True)
class AgentClassEvidence:
    """What a session's records have shown so far about its agent class.

    Start from :meth:`for_path`, fold each record in with :meth:`observe` as
    it is read, and read :attr:`agent_class` at the end, so no caller ever
    holds a transcript's records together. *first_turn_text* is None until
    the first primary user turn arrives, and '' when that turn holds no text.
    """

    in_worktree: bool
    has_records: bool = False
    first_turn_text: str | None = None
    first_turn_class: AgentClass | None = None
    carrier_literals: frozenset[str] = frozenset()

    @classmethod
    def for_path(cls, path: Path) -> AgentClassEvidence:
        """No records yet, from a transcript at *path*: its encoded dir (the
        parent of *path*) says whether it ran in an orchestrated worktree."""
        return cls(in_worktree=any(marker in path.parent.name for marker in WORKTREE_DIR_MARKERS))

    def observe(self, record: Mapping[str, Any]) -> AgentClassEvidence:
        """This evidence with *record*, the session's next record, folded in.

        A marker literal counts once sighted in any raw carrier, so a
        require-all marker whose literals sit in different carriers matches.
        """
        evidence = self if self.has_records else replace(self, has_records=True)
        if evidence.first_turn_text is None and is_primary_user_turn(record):
            text = user_turn_text(record) or ''
            evidence = replace(
                evidence,
                first_turn_text=text,
                first_turn_class=_first_turn_class(text),
            )
        unsighted = _MARKER_LITERALS - evidence.carrier_literals
        if evidence.in_worktree or evidence.first_turn_class is not None or not unsighted:
            return evidence
        found = frozenset().union(*(_sighted(text, unsighted) for _, text in _raw_carriers(record)))
        if not found:
            return evidence
        return replace(evidence, carrier_literals=evidence.carrier_literals | found)

    @property
    def agent_class(self) -> AgentClass:
        """The session's agent class.

        In order: no records is UNKNOWN; a worktree encoded dir is
        ORCHESTRATED_TASK; then a first user turn opening with an
        INSTRUMENT_PROMPT_OPENERS entry is LEGIBILITY_INSTRUMENT; then the
        marker table over the raw first user turn, then over every raw
        carrier. The carriers are read unfiltered
        on purpose: the class is decided BY injected markers, which the
        dialogue filter exists to drop. No marker is INTERACTIVE.
        """
        if not self.has_records:
            return AgentClass.UNKNOWN
        if self.in_worktree:
            return AgentClass.ORCHESTRATED_TASK
        if self.first_turn_class is not None:
            return self.first_turn_class
        marked = _marked_class(self.carrier_literals)
        return AgentClass.INTERACTIVE if marked is None else marked


def classify_agent_class(path: Path, records: Iterable[Mapping[str, Any]]) -> AgentClass:
    """The agent class of the session at *path*: :class:`AgentClassEvidence`
    with every one of *records* folded in."""
    evidence = AgentClassEvidence.for_path(path)
    for record in records:
        evidence = evidence.observe(record)
    return evidence.agent_class
