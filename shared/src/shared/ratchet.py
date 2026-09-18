"""The multiset ratchet kernel: compare a measured multiset against a baseline.

**What this module is.**  A ratchet is a gate that permits equality and
shrinkage and refuses growth.  This is the MULTISET idiom of that gate: the
thing being ratcheted is a bag of opaque keys with multiplicities, and the
check is ``excess(current, baseline) == {}``.  INV-12's inline-suppression
half is its first consumer (``plans/inv12-exceptions-owned-or-ratified-prd.md``
Contract **Kernel**, decision **D7**), where the key is a content hash of the
line bearing a suppression, so an edited line is a new key and an untouched
one is not.

**D13, stated plainly because a reviewer will read this module as a fourth
copy of something.**  There are two ratchet idioms in this repo and each gets
ONE home.  ``scripts/merge_lane_metrics.py`` is the SCALAR idiom — named
measures whose numbers may not rise — and it is deliberately neither ported,
imported, nor edited here; tasks own it separately, and its renderer is a
hybrid shaped by a different on-disk schema (per-path sections, an
insertion-ordered top level).  Its conventions are MATCHED below — a
``_README`` first key, a ``schema_version``, a ``params`` block whose mismatch
refuses, ``complete is True`` gating comparison, preconditions ordered ahead of
every comparison, one entry per line in sorted order, and rendering the whole
text before touching the destination.  Converging the two renderers is a later
task's to make; reading this as an oversight would be wrong.

``shared/tests/silent_fallthrough_scan.py::reconcile_against_allowlist`` and
the loop-blocking ledger are the natural future adopters of this kernel.  The
PRD names them and deliberately does not schedule them: their unblessed half
returns site OBJECTS in original order, which is not Counter subtraction and
cannot be expressed as one without changing what they report.

**What it deliberately does NOT do.**

* *It never interprets a key.*  Keys are opaque identity tokens the scanner
  supplies and this module only ever compares for equality.  A reader weighing
  ``Counter[str]`` against "structured data instead of meaningful strings"
  should note that the heuristic forbids ad-hoc PARSERS of internal values,
  and there is none here: JSON object keys are strings by the format's
  definition, so the rendering exists whoever owns it, and D7's key is
  content-addressed and already opaque by design.  Making the kernel generic
  over a key type with a caller-supplied codec would widen a deliberately
  narrow interface for one consumer that does not need it.
* *It has no absorb, widen, or write-baseline verb, by construction.*
  :func:`tighten` is the only baseline-producing function and its result is a
  subset of the baseline's keys because ``Counter.__and__`` is the pointwise
  minimum.  "No function can add a key to an existing baseline" is therefore a
  structural property of the arithmetic, not a check someone could forget.
* *It never fails soft.*  A partial enumeration and a mismatched params block
  both REFUSE rather than compare, and an unreadable baseline refuses rather
  than reading as empty.  An empty baseline compares clean against everything,
  so a soft return at any of those three points would report a breach as a
  clean tree (INV-11 ``no-silent-fail-soft``).

**Consumers.**  INV-12's inline-suppression scanner today.

Intentionally NOT re-exported from ``shared/__init__.py``.  Consumers import
via the fully-qualified path, consistent with the ``task_statuses`` /
``mcp_envelope`` / ``neutral_cwd`` / ``config_dir`` sub-module convention that
``shared/tests/test_public_api.py`` pins.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from shared.safe_io import atomic_write_text

__all__ = [
    'BASELINE_README',
    'BaselineUnusable',
    'Enumeration',
    'IncompleteEnumeration',
    'ParamsMismatch',
    'RatchetError',
    'SCHEMA_VERSION',
    'dump',
    'excess',
    'load',
    'slack',
    'tighten',
]

# The JSON scalar types a params value may be, or be a tuple of.  Not a
# preference: the params block is written to and read from a JSON file
# verbatim, so a value JSON cannot express would not survive the round trip
# and would make every post-reload comparison refuse with a spurious mismatch.
_JSON_SCALARS = (str, int, float, bool, type(None))

_ABSENT = object()
"""The marker for a params key that is not there at all.

None cannot double as that marker, because None is a legal params VALUE (it is
in :data:`_JSON_SCALARS`).  A bare ``object()`` is unequal to everything but
itself, and both sides of a params comparison draw their keys from the union of
the two blocks, so an absent-versus-absent pair can never arise.
"""

SCHEMA_VERSION: int = 1
"""The KERNEL's on-disk format version, deliberately separate from the params
block, which is the SCANNER's measurement parameters.  Two orthogonal reasons a
file can be incomparable get two fields and two distinct refusal messages:
folding the version into params would make a kernel format change read as a
scanner parameter change to whoever sees the red."""

BASELINE_README: str = (
    'This file is a MACHINE-GENERATED ratchet baseline. Its only legal diff is '
    'DELETIONS: entries leave as the thing they permit is removed, and nothing '
    'is ever added back. It is regenerated by tightening it against a fresh '
    'scan, never hand-edited and never regenerated merely to make a red test '
    'green — doing that silently widens the gate for every later change. This '
    'paragraph is re-emitted from the kernel on every write, so editing it here '
    'does nothing.'
)
"""The paragraph emitted as the first key of every dumped baseline.

WHY THE RULE LIVES IN THE FILE.  A baseline keyed by content digests is
unreviewable by content — a reviewer looking at the diff sees hex — so the only
handle they have on it is a statement, at the top of the file they opened, of
what a legal change to it looks like.  The
``scripts/merge_lane_metrics.py::BASELINE_README`` convention, matched rather
than imported (D13)."""


def _found(raw: Mapping[str, object], name: str) -> str:
    """Render *name*'s value for a refusal message, or the word ``absent``.

    A field that is missing and a field that is present and wrong are different
    mistakes with different fixes, and a message that renders the first as
    ``None`` sends the reader looking for a null they never wrote.  Both places
    that distinction is reported share this one renderer: the file boundary in
    :func:`_require_baseline_shape`, and the params comparison in
    :class:`ParamsMismatch`, where a null is a value the operator may genuinely
    have written.  Returns an ALREADY-REPR'D string, so a caller must not
    re-apply ``!r``.
    """
    return repr(raw[name]) if name in raw else 'absent'


class RatchetError(Exception):
    """Base for every refusal this kernel raises.

    A base is the right shape HERE, and deliberately not in
    ``shared.governed_exceptions``, because the discriminator is whether the
    caller's reaction is the same.  Every member of this family means one
    thing — *you cannot compare against this* — and maps to the same exit 2, so
    a consumer wants one ``except`` clause.  The governed-exceptions faults map
    to different exit codes and different operator actions, which is why they
    share no base.
    """


class ParamsMismatch(RatchetError):
    """Two enumerations were produced under different measurement parameters.

    Attributes:
        current_params: The current side's params block.
        baseline_params: The baseline's params block.
        differing: The keys on which the two disagree, sorted.
    """

    def __init__(
        self,
        *,
        current_params: Mapping[str, object],
        baseline_params: Mapping[str, object],
        differing: tuple[str, ...],
    ) -> None:
        self.current_params = current_params
        self.baseline_params = baseline_params
        self.differing = differing
        rendered = ', '.join(
            f'{key}: baseline={_found(baseline_params, key)} '
            f'current={_found(current_params, key)}'
            for key in differing
        )
        super().__init__(
            'refusing to compare two enumerations measured under different parameters — '
            f'{rendered}. Two enumerations produced under different parameters are not two '
            'measurements of the same thing, so comparing them reports a wall of downstream '
            'violations instead of the one named cause.'
        )


class IncompleteEnumeration(RatchetError):
    """At least one side skipped something, so a comparison would measure low.

    Attributes:
        side: Which side is incomplete — ``'current'``, ``'baseline'`` or
            ``'current and baseline'``.
        unreadable: The named entries the incomplete side(s) skipped.
    """

    def __init__(self, *, side: str, unreadable: tuple[str, ...]) -> None:
        self.side = side
        self.unreadable = unreadable
        super().__init__(
            f'refusing to compare a PARTIAL enumeration: {side} skipped '
            f'{len(unreadable)} entr(y/ies) — {list(unreadable)}. A scan that skipped '
            'entries measures LOWER than the truth, so comparing it would read as a clean '
            'tree, or worse as an improvement worth writing into the baseline (INV-11).'
        )


class BaselineUnusable(RatchetError):
    """The baseline at this path cannot be compared against.

    THE NAME STATES THE CONSUMER-RELEVANT FACT RATHER THAN A CAUSE.  Absent,
    unreadable, malformed and wrong-schema all mean one identical thing to a
    caller — *you cannot compare against this, and no comparison you could run
    would mean anything* — so splitting the family by cause would hand every
    consumer four ``except`` clauses that all do the same thing.  The cause is
    still there for whoever is fixing the file; it is in the message and on
    :attr:`cause`, where it informs without having to be dispatched on.

    Attributes:
        path: The baseline this refusal is about, named FIRST in the message.
            An operator holding a red gate is usually holding several
            baselines, and the one thing they cannot derive from the rest of
            the message is which file to open.
        reason: What was wrong with it, in one clause.
        cause: The underlying exception when the refusal came from one
            (``OSError``, a decode failure, or the shape ``ValueError``
            :class:`Enumeration` raises), else ``None``.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike[str],
        reason: str,
        cause: Exception | None = None,
    ) -> None:
        self.path = path
        self.reason = reason
        self.cause = cause
        super().__init__(
            f'refusing to compare against the baseline at {path}: {reason}. A baseline this '
            'kernel cannot read IS the finding — returning an empty one instead would compare '
            'clean against everything and report a breach as a clean tree (INV-11).'
        )


@dataclass(frozen=True)
class Enumeration:
    """One measured multiset, its parameters, and how honest it is.

    Attributes:
        counts: The multiset, keyed by opaque identity token.  Every value is
            at least 1: a multiset carries no zero entries, Counter arithmetic
            would drop them anyway, and a zero on disk would make :func:`dump`
            and :func:`load` non-canonical.
        params: The SCANNER's measurement parameters — what it swept for and
            how it keyed the results.  Kept separate from
            :data:`SCHEMA_VERSION`, which is the KERNEL's file format: they are
            two orthogonal reasons a file can be incomparable, and folding them
            together would make a kernel format change read as a scanner
            parameter change to whoever sees the red.
        complete: Whether the scan read everything it set out to.  ``complete``
            plus a NAMED ``unreadable`` list is the repo's settled INV-11 shape,
            not an invention — ``scripts/merge_lane_metrics.py::Enumeration``
            and ``fused-memory/scripts/census_memory_metadata.py`` both carry
            it.  A bare count would lose the names a refusal message needs.
        unreadable: What the scan skipped, by name.  Non-empty is a
            contradiction with ``complete=True`` and is refused rather than
            silently corrected: flipping the flag would repair a caller's bug
            without telling anyone.
    """

    counts: Mapping[str, int]
    params: Mapping[str, object]
    complete: bool = True
    unreadable: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        counts = dict(self.counts)
        for key, count in counts.items():
            if not isinstance(key, str):
                raise ValueError(
                    f'Enumeration: counts key {key!r} must be a str — keys are opaque '
                    'identity tokens, and JSON object keys are strings by definition.'
                )
            if type(count) is not int:
                raise ValueError(
                    f'Enumeration: counts[{key!r}]={count!r} must be an int, not '
                    f'{type(count).__name__} (bool is excluded even though it is an int '
                    'subclass — True would silently read as a multiplicity of 1).'
                )
            if count < 1:
                raise ValueError(
                    f'Enumeration: counts[{key!r}]={count!r} must be at least 1. A multiset '
                    'carries no zero entries; Counter arithmetic drops them anyway, so a zero '
                    'would make dump/load non-canonical.'
                )

        params = {}
        for key, value in dict(self.params).items():
            if not isinstance(key, str):
                raise ValueError(
                    f'Enumeration: params key {key!r} must be a str — the params block is '
                    'written to and read from a JSON object, whose keys are strings by the '
                    "format's definition. A non-str key survives dump only by being "
                    'rewritten as its JSON spelling, so the reloaded block differs from the '
                    'one in memory and every later comparison refuses with a spurious '
                    'mismatch — or sorts against a str and raises a bare TypeError, which '
                    'is outside the RatchetError family a consumer catches.'
                )
            params[key] = _normalised_param(key, value)
        unreadable = tuple(self.unreadable)
        if self.complete is True and unreadable:
            raise ValueError(
                f'Enumeration: complete=True with unreadable={list(unreadable)}. A scan that '
                'skipped entries is not complete; this contradiction is refused rather than '
                'corrected, because flipping the flag would repair the caller silently and '
                'the next line probably reports the run as clean (INV-11).'
            )

        object.__setattr__(self, 'counts', MappingProxyType(counts))
        object.__setattr__(self, 'params', MappingProxyType(params))
        object.__setattr__(self, 'unreadable', unreadable)


def _is_json_scalar(value: object) -> bool:
    """Whether JSON can express *value* as a literal.

    ``float`` is admitted only when FINITE, and that exclusion is the same rule
    :data:`_JSON_SCALARS` states rather than a second one.  ``json.dumps`` writes
    ``nan`` and ``inf`` as the bare tokens ``NaN`` and ``Infinity``, which RFC 8259
    has no literal for — so the committed baseline a human is told to review is a
    file strict readers reject — and ``nan != nan``, so a params block carrying one
    refuses every later comparison with a spurious mismatch.  That is precisely the
    round-trip failure the scalar restriction exists to prevent, so letting the
    value through the type check would leave the rule half-enforced.
    """
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, _JSON_SCALARS)


def _normalised_param(key: str, value: object) -> object:
    """Check *value* is JSON-expressible and normalise a sequence to a tuple."""
    if _is_json_scalar(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if not _is_json_scalar(item):
                raise ValueError(
                    f'Enumeration: params[{key!r}] contains {item!r}, which JSON cannot '
                    'express as a literal. The params block round-trips through a JSON file '
                    'verbatim.'
                )
        return tuple(value)
    raise ValueError(
        f'Enumeration: params[{key!r}]={value!r} is a {type(value).__name__}, which JSON '
        'cannot express as a literal, and it is not a sequence of such values either. The '
        'params block round-trips through a JSON file verbatim, so a value JSON cannot '
        'express would make every later comparison refuse with a spurious params mismatch. '
        'A non-finite float is excluded for exactly that reason: JSON has no literal for '
        'NaN or Infinity, and nan != nan.'
    )


def _require_comparable(current: Enumeration, baseline: Enumeration) -> None:
    """Refuse two enumerations that are not two measurements of the same thing.

    THE ONE MECHANISM, with three call sites.  It is the first statement of
    :func:`excess`, :func:`slack` and :func:`tighten` rather than a public
    ``assert_comparable()`` the caller is trusted to run, and that is the whole
    reason the kernel's signatures take Enumerations instead of the bare
    Counters the Contract's prose names.  A forgettable precondition is exactly
    the single missed check INV-11 forbids: a caller who skipped it would
    compare a partial scan against the baseline and read green, which is the
    one failure mode this instrument exists to make impossible.  Making the
    guard structural costs one wrapper type and removes the failure mode
    entirely.

    The two checks run in a FIXED order, so the red a caller sees is
    reproducible when both faults are present.  Completeness is first because
    it is the fault that says the measurement never really happened; a params
    mismatch between two enumerations, at least one of which is a partial scan,
    is not the more useful thing to report.

    ``complete is True`` is an IDENTITY test, not truthiness: an absent or
    unpopulated flag must refuse rather than read as "probably fine", which is
    the spelling ``scripts/merge_lane_metrics.py::_require_complete_enumeration``
    uses for the same reason.

    The params comparison passes :data:`_ABSENT` to both ``.get`` calls because
    the params block is the one mapping in this module where None is a legal
    VALUE rather than a sentinel for missing; a bare ``.get`` would let a params
    block that GAINED or LOST a null-valued key compare as matching and proceed,
    which is the silent fail-soft this refusal exists to prevent.
    """
    incomplete = tuple(
        side
        for side, enumeration in (('current', current), ('baseline', baseline))
        if enumeration.complete is not True
    )
    if incomplete:
        raise IncompleteEnumeration(
            side=' and '.join(incomplete),
            unreadable=tuple(
                entry
                for enumeration in (current, baseline)
                if enumeration.complete is not True
                for entry in enumeration.unreadable
            ),
        )

    differing = tuple(
        sorted(
            key
            for key in set(current.params) | set(baseline.params)
            if current.params.get(key, _ABSENT) != baseline.params.get(key, _ABSENT)
        )
    )
    if differing:
        raise ParamsMismatch(
            current_params=current.params,
            baseline_params=baseline.params,
            differing=differing,
        )


def excess(current: Enumeration, baseline: Enumeration) -> Counter[str]:
    """What *current* has beyond *baseline* — the VIOLATION REPORT.

    Empty means the gate passes: equality and shrinkage are both green, and
    ``Counter``'s saturating ``-`` drops non-positive results, so neither needs
    a clamp.

    Its keys may include keys the baseline never had.  That is not a leak in
    the no-add-key property — it IS the finding, and the caller renders it as
    one.  Nothing in this module ever writes a report back to a baseline.

    Raises:
        RatchetError: The two enumerations are not comparable — see
            :func:`_require_comparable`.
    """
    _require_comparable(current, baseline)
    return Counter(current.counts) - Counter(baseline.counts)


def slack(current: Enumeration, baseline: Enumeration) -> Counter[str]:
    """What *baseline* still permits and *current* does not use — the HEADROOM.

    Non-empty slack is what files a tighten task: an un-tightened baseline lets
    an identical line in where one was removed, so the headroom is a standing
    invitation nobody meant to leave open.  The mirror of :func:`excess`, and
    saturating for the same reason.

    Raises:
        RatchetError: The two enumerations are not comparable — see
            :func:`_require_comparable`.
    """
    _require_comparable(current, baseline)
    return Counter(baseline.counts) - Counter(current.counts)


def tighten(current: Enumeration, baseline: Enumeration) -> Counter[str]:
    """The pointwise minimum — THE ONLY baseline-producing function.

    ``Counter.__and__`` IS the pointwise minimum, so the result's keys are a
    subset of the baseline's by construction.  That is how "no function can add
    a key to an existing baseline" is enforced STRUCTURALLY rather than by a
    check someone could forget to call: a key absent from the baseline has
    multiplicity 0 there, and the minimum of anything and 0 is 0.

    Idempotent by the same property: tightening against an already-tightened
    baseline changes nothing.

    Raises:
        RatchetError: The two enumerations are not comparable — see
            :func:`_require_comparable`.  A baseline tightened against a
            PARTIAL scan would delete every key the scan failed to reach, which
            is the widening that refusal exists to prevent.
    """
    _require_comparable(current, baseline)
    return Counter(current.counts) & Counter(baseline.counts)


def _render_counts(counts: Mapping[str, int]) -> str:
    """Render the counts block with ONE key per line, in sorted order.

    Written here rather than hoisted out of
    ``scripts/merge_lane_metrics.py::_render_section`` because D13 puts that
    file out of scope for this batch and other tasks are about to change it,
    and because its renderer is a hybrid shaped by a different on-disk schema.
    Converging the two is a later task's to make; this is a recorded choice,
    not an oversight.

    ``json.dumps(indent=2)`` would put each key on its own line too, so the
    reason this exists at all is the SORT plus the fixed two-space shift that
    keep two runs over equal input byte-identical.
    """
    entries = [f'    {json.dumps(key)}: {count}' for key, count in sorted(counts.items())]
    if not entries:
        return '  "counts": {}'
    return '  "counts": {\n' + ',\n'.join(entries) + '\n  }'


def dump(enumeration: Enumeration, path: str | os.PathLike[str]) -> None:
    """Write *enumeration* to *path* as the committed baseline's exact bytes.

    THE WHOLE TEXT IS RENDERED BEFORE THE DESTINATION IS TOUCHED, and the write
    goes through ``shared.safe_io.atomic_write_text`` — the tmp+rename writer
    every such site in the repo delegates to.  A truncated baseline is a
    WIDENED ratchet: the one failure mode where a partial write silently
    loosens a gate instead of breaking it, so a reader of the half-file sees
    fewer permitted entries and every missing key reads as a fresh violation —
    or, after a tighten, as an improvement worth keeping.

    Owning the write rather than exposing a pure ``render() -> str`` keeps that
    discipline here instead of making it every future adopter's to rediscover,
    and keeps the public surface at two functions.

    The parent directory is created when absent (``mkdir=True``), so seeding a
    baseline at a path that does not exist yet needs no ceremony.
    """
    blocks = [
        f'  "_README": {json.dumps(BASELINE_README)}',
        f'  "schema_version": {json.dumps(SCHEMA_VERSION)}',
        # allow_nan=False is the second, redundant enforcement of the rule
        # _is_json_scalar states: params is the one block that can carry a float,
        # and json.dumps would otherwise emit the non-RFC-8259 tokens NaN and
        # Infinity into a file a reviewer is told to read.
        '  "params": ' + json.dumps(
            dict(enumeration.params), indent=2, sort_keys=True, allow_nan=False
        ).replace('\n', '\n  '),
        f'  "complete": {json.dumps(enumeration.complete)}',
        '  "unreadable": ' + json.dumps(list(enumeration.unreadable), indent=2).replace(
            '\n', '\n  '
        ),
        _render_counts(enumeration.counts),
    ]
    atomic_write_text(path, '{\n' + ',\n'.join(blocks) + '\n}\n', mkdir=True)


def _require_baseline_shape(raw: object, path: str | os.PathLike[str]) -> None:
    """Refuse anything that is not shaped like a baseline, naming the field.

    Explicit and ordered, ahead of :class:`Enumeration`, so the refusal names
    the one field that is wrong rather than whatever the constructor happened
    to trip over first.

    SHAPE, NEVER CONTENT.  A key a human added by hand loads without complaint,
    and that is not a hole: :func:`tighten` is the only baseline-producing
    function and its result is a subset of the baseline, so an added key cannot
    widen anything and survives exactly until the next tighten.  Policing it
    here would be a second, weaker enforcement point for a property the
    arithmetic already guarantees.
    """
    if not isinstance(raw, dict):
        raise BaselineUnusable(
            path=path,
            reason=f'its top level is a {type(raw).__name__}, not a JSON object',
        )

    version = raw.get('schema_version')
    if type(version) is not int or version != SCHEMA_VERSION:
        declared = _found(raw, 'schema_version')
        raise BaselineUnusable(
            path=path,
            reason=(
                f'its schema_version is {declared} and this build reads only schema_version '
                f'{SCHEMA_VERSION}'
            ),
        )

    for name in ('params', 'counts'):
        if not isinstance(raw.get(name), dict):
            raise BaselineUnusable(
                path=path,
                reason=f'its {name} block is {_found(raw, name)}, not a JSON object',
            )

    if not isinstance(raw.get('complete'), bool):
        raise BaselineUnusable(
            path=path,
            reason=(
                f'its complete flag is {_found(raw, "complete")}, not a JSON boolean — an '
                'enumeration that does not say how honest it is cannot be compared'
            ),
        )

    unreadable = raw.get('unreadable', [])
    if not isinstance(unreadable, list) or not all(isinstance(name, str) for name in unreadable):
        raise BaselineUnusable(
            path=path,
            reason=f'its unreadable list is {_found(raw, "unreadable")}, not a list of strings',
        )


def load(path: str | os.PathLike[str]) -> Enumeration:
    """Read the baseline at *path*, or refuse to hand back anything at all.

    THE POLARITY, stated here rather than pointed at, because it is the
    opposite of the one a whole-tree sweep guard takes.  A sweep that cannot
    read one file out of thousands may reasonably carry on and name it in
    ``unreadable``.  This instrument is asked to compare against ONE fixed,
    named artifact, so a baseline it cannot read IS the finding, and there is
    no degraded answer available to return instead: an empty baseline compares
    clean against EVERYTHING, so a soft return would report a breach as a clean
    tree (INV-11 ``no-silent-fail-soft``).  Absent, undecodable, unparseable,
    misshapen and wrong-schema therefore all converge on one refusal.

    Any inbound ``_README`` is DROPPED rather than carried: :func:`dump` always
    re-emits the constant, which is what stops an edited one surviving a round
    trip and softening the rule in the file that publishes it.

    Raises:
        BaselineUnusable: Always, for every one of those ways of failing to get
            an Enumeration out of this path.  The cause travels on the message
            and on ``.cause`` rather than in the exception type, because the
            caller does the same thing in every case.
    """
    location = Path(path)
    try:
        raw = json.loads(location.read_text(encoding='utf-8'))
        _require_baseline_shape(raw, location)
        return Enumeration(
            counts=raw['counts'],
            params=raw['params'],
            complete=raw['complete'],
            unreadable=tuple(raw.get('unreadable', ())),
        )
    # json.JSONDecodeError and UnicodeDecodeError are both ValueError subclasses,
    # as is the value fault Enumeration.__post_init__ raises; BaselineUnusable is
    # not, so a refusal raised above travels out of here unwrapped.
    except (OSError, ValueError) as exc:
        raise BaselineUnusable(
            path=location, reason=f'{type(exc).__name__}: {exc}', cause=exc
        ) from exc
