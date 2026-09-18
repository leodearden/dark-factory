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

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = [
    'Enumeration',
    'IncompleteEnumeration',
    'ParamsMismatch',
    'RatchetError',
]

# The JSON scalar types a params value may be, or be a tuple of.  Not a
# preference: the params block is written to and read from a JSON file
# verbatim, so a value JSON cannot express would not survive the round trip
# and would make every post-reload comparison refuse with a spurious mismatch.
_JSON_SCALARS = (str, int, float, bool, type(None))


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
            f'{key}: baseline={baseline_params.get(key)!r} current={current_params.get(key)!r}'
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

        params = {key: _normalised_param(key, value) for key, value in dict(self.params).items()}
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


def _normalised_param(key: str, value: object) -> object:
    """Check *value* is JSON-expressible and normalise a sequence to a tuple."""
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, _JSON_SCALARS):
                raise ValueError(
                    f'Enumeration: params[{key!r}] contains {item!r}, which is not a JSON '
                    'scalar. The params block round-trips through a JSON file verbatim.'
                )
        return tuple(value)
    raise ValueError(
        f'Enumeration: params[{key!r}]={value!r} is a {type(value).__name__}, which is not a '
        'JSON scalar or a sequence of them. The params block round-trips through a JSON file '
        'verbatim, so a value JSON cannot express would make every later comparison refuse '
        'with a spurious params mismatch.'
    )
