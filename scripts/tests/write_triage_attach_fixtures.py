"""Standalone stand-ins for ``fused_memory.server.write_triage``.

Shared by BOTH gate test files — ``test_check_write_triage_attach_consumption.py``
drives the consumption probe against them directly, and
``test_check_write_triage_flip_preconditions.py`` lays them into its hermetic
gate repos — so they live here rather than in either one (SPOT).

WHY THE FIXTURES ARE STANDALONE. The probe is pointed at a ``--src-root`` and
imports ``fused_memory.server.write_triage`` out of a bare directory tree. The
real module imports ``shared.storm_counter``, ``fused_memory.models.enums`` and
two sibling server modules, which between them pull in pydantic; a fixture that
did the same would make this suite depend on the fused-memory virtualenv rather
than running under whatever interpreter collected it. So the module text below
may import nothing but the stdlib — the same constraint, for the same reason, as
the judge fixtures in ``test_check_write_triage_flip_preconditions.py``.

WHAT IS MIRRORED FROM MAIN, and why it has to be. ``decide_band`` reproduces
main's max-cosine selection AND ``_canonical_id_of``'s child hoisting, because
the hoist is what makes the band's canonical an id that belongs to no candidate
in the slate — which is exactly what separates "the attach followed the judge's
designation" from "the attach used the band's winner". A fixture that skipped
the hoist would let a positional bug read as a correct fix.
"""
from __future__ import annotations

from pathlib import Path

#: Everything above ``triage_write``. Variants differ ONLY in that function, so
#: each tail below can be read as the one thing it changes.
TRIAGE_PREAMBLE = r'''"""Standalone stand-in for fused_memory.server.write_triage.

Dependency-free by construction: the probe imports this out of a bare directory
tree, so it may import nothing but the stdlib.

decide_band mirrors main's EXACTLY, including _canonical_id_of's hoist of a
CHILD winner to its parent_id — so on the probe's fixture slate the band's
canonical is an id no candidate carries as its own.
"""
from __future__ import annotations

from dataclasses import dataclass

OUTCOME_STORED = 'stored'
OUTCOME_RESTATED = 'restated'
OUTCOME_AMENDED = 'amended'
OUTCOME_CONTESTED = 'contested'
OUTCOME_JUDGE = 'judge'

TRIAGE_OUTCOMES = frozenset({
    OUTCOME_STORED,
    OUTCOME_RESTATED,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
})

PARENT_ID_KEY = 'parent_id'
CHILD_KINDS = frozenset({'sighting', 'amendment'})

_DEFAULT_CANDIDATE_K = 20


@dataclass(frozen=True)
class BandDecision:
    outcome: str
    canonical_id: str | None
    similarity: float | None
    t_high: float | None
    t_low: float | None


class TriageFailOpenCounter:
    """Counts fail-opens. Same record()/drain_storm()/live_count() surface as
    main's, minus the rolling-window burst detection the probe never reads."""

    def __init__(self, time_provider=None):
        self._records = []
        self._pending_storm = None

    def record(self, *, project=None):
        self._records.append(project)
        return None

    def drain_storm(self):
        storm, self._pending_storm = self._pending_storm, None
        return storm

    def live_count(self):
        return len(self._records)


def _cosine_of(result):
    value = (getattr(result, 'metadata', None) or {}).get('store_score')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _canonical_id_of(result):
    """The id a write attaches to for *result*, HOISTING a child to its parent."""
    meta = getattr(result, 'metadata', None) or {}
    if meta.get('kind') in CHILD_KINDS:
        parent_id = meta.get(PARENT_ID_KEY)
        if isinstance(parent_id, str) and parent_id:
            return parent_id
    return result.id


def decide_band(results, *, t_high, t_low):
    scored = []
    for result in results or ():
        cosine = _cosine_of(result)
        if cosine is not None:
            scored.append((cosine, result))
    if not scored:
        return BandDecision(OUTCOME_STORED, None, None, t_high, t_low)
    similarity, best = max(scored, key=lambda pair: pair[0])
    if t_low is None or similarity < t_low:
        return BandDecision(OUTCOME_STORED, None, None, t_high, t_low)
    if t_high is not None and similarity >= t_high:
        return BandDecision(
            OUTCOME_RESTATED, _canonical_id_of(best), similarity, t_high, t_low,
        )
    return BandDecision(
        OUTCOME_JUDGE, _canonical_id_of(best), similarity, t_high, t_low,
    )


def _write_triage_attr(memory_service, attr):
    config = getattr(memory_service, 'config', None)
    write_triage = getattr(config, 'write_triage', None)
    return getattr(write_triage, attr, None)


def resolve_write_triage_enabled(memory_service):
    value = _write_triage_attr(memory_service, 'enabled')
    return value if isinstance(value, bool) else False


def resolve_candidate_k(memory_service):
    value = _write_triage_attr(memory_service, 'candidate_k')
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return _DEFAULT_CANDIDATE_K


def _coerce_band(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def resolve_bands(memory_service):
    return (
        _coerce_band(_write_triage_attr(memory_service, 't_high')),
        _coerce_band(_write_triage_attr(memory_service, 't_low')),
    )


async def retrieve_candidates(memory_service, content, project_id, k):
    return await memory_service.search(
        query=content,
        project_id=project_id,
        categories=[
            'observations_and_summaries',
            'preferences_and_norms',
            'procedural_knowledge',
        ],
        stores=['mem0'],
        limit=k,
        anchor_topics=False,
    )


def _record_fail_open(counter, project_id, exc, *, stage):
    """Count one fail-open. Never raises — main's contract C1."""
    try:
        return counter.record(project=project_id)
    except Exception:
        return None


async def _stub_judge(*, memory_service, content, project_id, decision,
                      candidates=(), **_announced):
    """main's default judge. Tolerates an announced attach target.

    Swallowing an unknown kwarg is what a real judge under option (b) would
    have to do anyway, and without it an announcing variant that fell back to
    this stub would raise -- turning a fixture the probe never exercises that
    way into a fail-open, and measuring the stub instead of the variant.
    """
    return OUTCOME_STORED


async def _band_and_candidates(memory_service, content, project_id, counter):
    """main's force-store -> retrieve -> band prologue, shared by every variant.

    Returns ``(decision, candidates)``. An outcome other than OUTCOME_JUDGE
    means the caller returns the decision unchanged, exactly as main does.
    """
    try:
        k = resolve_candidate_k(memory_service)
        t_high, t_low = resolve_bands(memory_service)
        results = await retrieve_candidates(memory_service, content, project_id, k)
        if getattr(results, 'degraded', False):
            _record_fail_open(
                counter, project_id,
                RuntimeError('search degraded'),
                stage='retrieve',
            )
            return BandDecision(OUTCOME_STORED, None, None, t_high, t_low), ()
        return decide_band(results, t_high=t_high, t_low=t_low), results
    except Exception as exc:
        _record_fail_open(counter, project_id, exc, stage='retrieve')
        return BandDecision(OUTCOME_STORED, None, None, None, None), ()


def _forced(allow_near_duplicate, caller_owns_attach_keys):
    return allow_near_duplicate or caller_owns_attach_keys


def _slate_ids(candidates):
    return [getattr(c, 'id', None) for c in candidates or ()]


def _attach_designated(candidate_id, candidates, decision):
    """Honour the designation, after validating it against the slate."""
    if candidate_id in _slate_ids(candidates):
        return candidate_id
    return decision.canonical_id


def _make_designating_triage_write(decode, attach=_attach_designated):
    """Build a triage_write that reads a designating verdict.

    *decode* maps a judge payload to ``(outcome, candidate_id)``, or None for
    "not the spelling this variant speaks" — in which case the payload is
    treated as main treats it, i.e. as a bare outcome word naming no candidate.

    *attach* maps ``(candidate_id, candidates, decision)`` to the id the write
    attaches to. Decoding a designation and HONOURING it are separate things,
    and a variant that decodes one and then attaches somewhere fixed is exactly
    the bug class the swap test exists to catch — so they are separate
    parameters rather than one entangled body.
    """
    async def triage_write(memory_service, *, content, project_id, counter,
                           judge=None, allow_near_duplicate=False,
                           caller_owns_attach_keys=False):
        if _forced(allow_near_duplicate, caller_owns_attach_keys):
            return BandDecision(OUTCOME_STORED, None, None, None, None)
        decision, candidates = await _band_and_candidates(
            memory_service, content, project_id, counter,
        )
        if decision.outcome != OUTCOME_JUDGE:
            return decision
        try:
            verdict = await (judge or _stub_judge)(
                memory_service=memory_service,
                content=content,
                project_id=project_id,
                decision=decision,
                candidates=candidates,
            )
        except Exception as exc:
            _record_fail_open(counter, project_id, exc, stage='judge')
            return BandDecision(
                OUTCOME_STORED, None, None, decision.t_high, decision.t_low,
            )
        designated = decode(verdict)
        outcome, candidate_id = designated if designated else (verdict, None)
        if not isinstance(outcome, str) or outcome not in TRIAGE_OUTCOMES:
            _record_fail_open(
                counter, project_id,
                ValueError('judge returned %r' % (verdict,)),
                stage='judge',
            )
            return BandDecision(
                OUTCOME_STORED, None, None, decision.t_high, decision.t_low,
            )
        if outcome == OUTCOME_STORED:
            canonical_id = None
        else:
            canonical_id = attach(candidate_id, candidates, decision)
        return BandDecision(
            outcome, canonical_id, decision.similarity,
            decision.t_high, decision.t_low,
        )

    return triage_write


#: The kwarg an announcing variant tells the judge its attach target through.
#: A NEW name, beyond main's five: an announcement the probe could read out of
#: `decision` would be one main already makes, so the branch would hold on a
#: codebase where nothing changed.
ANNOUNCE_KWARG = 'attach_target'


def _announced_target(candidates, decision):
    """The CALLER's own pick of attach target, under an option-(b) remedy.

    The first slate id that is not the band's canonical: announcing the band's
    own canonical would be announcing what main already attaches to, so it
    would carry no information about whether the announcement was honoured.
    """
    for ident in _slate_ids(candidates):
        if isinstance(ident, str) and ident and ident != decision.canonical_id:
            return ident
    return decision.canonical_id


def _honour_announcement(announced, candidates, decision):
    return announced


def _make_announcing_triage_write(attach=_honour_announcement):
    """Build a triage_write shaped like an option-(b) remedy.

    Under option (b) the CALLER picks the attach target and tells the judge
    which candidate it is reasoning about, rather than letting the judge name
    one back. The judge's verdict stays a bare outcome word, so no designation
    is decoded here at all.

    *attach* maps ``(announced, candidates, decision)`` to the id the write
    attaches to. Announcing a target and HONOURING it are separate things, and
    a module that names one candidate in the prompt and files the verdict
    against another is the option-(b)-shaped form of the very defect this item
    exists to detect -- so, as with the designating builder, they are separate
    parameters rather than one entangled body.
    """
    async def triage_write(memory_service, *, content, project_id, counter,
                           judge=None, allow_near_duplicate=False,
                           caller_owns_attach_keys=False):
        if _forced(allow_near_duplicate, caller_owns_attach_keys):
            return BandDecision(OUTCOME_STORED, None, None, None, None)
        decision, candidates = await _band_and_candidates(
            memory_service, content, project_id, counter,
        )
        if decision.outcome != OUTCOME_JUDGE:
            return decision
        announced = _announced_target(candidates, decision)
        try:
            verdict = await (judge or _stub_judge)(**{
                'memory_service': memory_service,
                'content': content,
                'project_id': project_id,
                'decision': decision,
                'candidates': candidates,
                ANNOUNCE_KWARG: announced,
            })
        except Exception as exc:
            _record_fail_open(counter, project_id, exc, stage='judge')
            return BandDecision(
                OUTCOME_STORED, None, None, decision.t_high, decision.t_low,
            )
        if not isinstance(verdict, str) or verdict not in TRIAGE_OUTCOMES:
            _record_fail_open(
                counter, project_id,
                ValueError('judge returned %r' % (verdict,)),
                stage='judge',
            )
            return BandDecision(
                OUTCOME_STORED, None, None, decision.t_high, decision.t_low,
            )
        if verdict == OUTCOME_STORED:
            canonical_id = None
        else:
            canonical_id = attach(announced, candidates, decision)
        return BandDecision(
            verdict, canonical_id, decision.similarity,
            decision.t_high, decision.t_low,
        )

    return triage_write
'''


#: main's shape, and the DEFECT this whole gate item exists to detect: the
#: judge's verdict is a bare outcome word, and the attach id is whatever the
#: BAND picked, so a verdict earned by candidate #3 is filed against the band's
#: top-1. The `canonical_id = None if verdict == OUTCOME_STORED else
#: decision.canonical_id` line below is main's write_triage.py:898 verbatim,
#: and so is the unguarded `verdict not in TRIAGE_OUTCOMES` above it — which
#: RAISES on an unhashable designating payload rather than falling open. Both
#: are reproduced rather than tidied: the probe has to be measured against what
#: main actually does.
_BAND_TOP1 = r'''

async def triage_write(memory_service, *, content, project_id, counter,
                       judge=None, allow_near_duplicate=False,
                       caller_owns_attach_keys=False):
    if _forced(allow_near_duplicate, caller_owns_attach_keys):
        return BandDecision(OUTCOME_STORED, None, None, None, None)
    decision, candidates = await _band_and_candidates(
        memory_service, content, project_id, counter,
    )
    if decision.outcome != OUTCOME_JUDGE:
        return decision
    try:
        verdict = await (judge or _stub_judge)(
            memory_service=memory_service,
            content=content,
            project_id=project_id,
            decision=decision,
            candidates=candidates,
        )
    except Exception as exc:
        _record_fail_open(counter, project_id, exc, stage='judge')
        return BandDecision(
            OUTCOME_STORED, None, None, decision.t_high, decision.t_low,
        )
    if verdict not in TRIAGE_OUTCOMES:
        _record_fail_open(
            counter, project_id,
            ValueError('judge returned %r' % (verdict,)),
            stage='judge',
        )
        return BandDecision(
            OUTCOME_STORED, None, None, decision.t_high, decision.t_low,
        )
    canonical_id = None if verdict == OUTCOME_STORED else decision.canonical_id
    return BandDecision(
        verdict, canonical_id, decision.similarity, decision.t_high, decision.t_low,
    )
'''


#: The three plausible option-(a) wire shapes for "the verdict names its own
#: candidate". Which one an eventual fix picks is unknown — option (a) has not
#: landed — so the probe may pin none of them, and the fixtures carry all three
#: so that discipline is testable rather than asserted. Each speaks exactly ONE
#: spelling and treats every other payload as main does: a bare outcome word
#: attaches to the band's canonical, and anything unrecognised falls open.
_CONSUMES_TUPLE = r'''

def _decode(verdict):
    """(outcome, candidate_id)."""
    if isinstance(verdict, tuple) and len(verdict) == 2:
        return verdict
    return None


triage_write = _make_designating_triage_write(_decode)
'''

_CONSUMES_DICT = r'''

def _decode(verdict):
    """{'outcome': ..., 'candidate_id': ...}."""
    if isinstance(verdict, dict):
        return verdict.get('outcome'), verdict.get('candidate_id')
    return None


triage_write = _make_designating_triage_write(_decode)
'''

_CONSUMES_OBJECT = r'''

def _decode(verdict):
    """An object exposing .outcome and .candidate_id."""
    outcome = getattr(verdict, 'outcome', None)
    candidate_id = getattr(verdict, 'candidate_id', None)
    if isinstance(outcome, str) and isinstance(candidate_id, str):
        return outcome, candidate_id
    return None


triage_write = _make_designating_triage_write(_decode)
'''


#: The POSITIONAL BUG, relocated to the consumption side. It decodes the
#: designation perfectly happily and then attaches to a fixed slot, so its
#: attach id is neither the band's canonical nor anything the judge said. A
#: single-shot "is the attach id different from the band canonical?" check
#: blesses it; only requiring the attach to TRACK the designation across two
#: different designations rejects it. Item 1 had to defeat the same class as
#: `candidates[0]`.
_HARDCODES_LAST = r'''

def _decode(verdict):
    if isinstance(verdict, tuple) and len(verdict) == 2:
        return verdict
    return None


def _attach_last(candidate_id, candidates, decision):
    ids = _slate_ids(candidates)
    return ids[-1] if ids else decision.canonical_id


triage_write = _make_designating_triage_write(_decode, _attach_last)
'''


#: main's ACTUAL response to a designating verdict today, and the catastrophic
#: false pass this whole item has to survive: the payload trips
#: `verdict not in TRIAGE_OUTCOMES`, a fail-open is recorded, and the write
#: returns BandDecision(stored, None, ...). That canonical_id of None is not the
#: band's top-1 either, so a naive "did the attach avoid the band canonical?"
#: check reads it as CONSUMED and authorises the production flag flip on a
#: codebase where nothing changed at all.
#:
#: isinstance-guarded rather than main's bare membership test, deliberately: an
#: unhashable payload must reach the FAIL-OPEN arm here instead of raising, or
#: this variant would measure the raise rather than the fail-open it exists to
#: isolate. `band_top1` keeps main's unguarded spelling.
_FAIL_OPENS_ON_DESIGNATION = r'''

async def triage_write(memory_service, *, content, project_id, counter,
                       judge=None, allow_near_duplicate=False,
                       caller_owns_attach_keys=False):
    if _forced(allow_near_duplicate, caller_owns_attach_keys):
        return BandDecision(OUTCOME_STORED, None, None, None, None)
    decision, candidates = await _band_and_candidates(
        memory_service, content, project_id, counter,
    )
    if decision.outcome != OUTCOME_JUDGE:
        return decision
    verdict = await (judge or _stub_judge)(
        memory_service=memory_service,
        content=content,
        project_id=project_id,
        decision=decision,
        candidates=candidates,
    )
    if not isinstance(verdict, str) or verdict not in TRIAGE_OUTCOMES:
        _record_fail_open(
            counter, project_id,
            ValueError('judge returned %r' % (verdict,)),
            stage='judge',
        )
        return BandDecision(
            OUTCOME_STORED, None, None, decision.t_high, decision.t_low,
        )
    canonical_id = None if verdict == OUTCOME_STORED else decision.canonical_id
    return BandDecision(
        verdict, canonical_id, decision.similarity, decision.t_high, decision.t_low,
    )
'''


#: An option-(b)-shaped remedy, and the reason item 5 may not require the
#: judge-side swap. Here the CALLER picks the attach target and announces it to
#: the judge through a kwarg beyond main's five; the judge never names a
#: candidate back, so a probe that only looked for a judge-side designation
#: would fail this -- re-blocking task 3169 against a correct fix, which is the
#: false-FAIL class this gate family was rewritten to remove.
_ANNOUNCES_TARGET = r"""

triage_write = _make_announcing_triage_write()
"""


#: The option-(b)-shaped form of the defect: the prompt names one candidate and
#: the write files the verdict against the band's top-1 anyway. It announces,
#: so the branch is reachable; it does not honour the announcement, so the
#: branch must not hold. Without this variant, "an announcement exists" would
#: be indistinguishable from "the announcement was consumed".
_ANNOUNCES_BUT_ATTACHES_ELSEWHERE = r"""

def _attach_band_canonical(announced, candidates, decision):
    return decision.canonical_id


triage_write = _make_announcing_triage_write(_attach_band_canonical)
"""


#: variant name -> the ``triage_write`` that defines it. Appended to
#: :data:`TRIAGE_PREAMBLE` by :func:`write_fake_triage`.
VARIANT_TAILS: dict[str, str] = {
    'band_top1': _BAND_TOP1,
    'consumes_designated_id': _CONSUMES_TUPLE,
    'consumes_designated_dict': _CONSUMES_DICT,
    'consumes_designated_object': _CONSUMES_OBJECT,
    'hardcodes_last_candidate': _HARDCODES_LAST,
    'fail_opens_on_designation': _FAIL_OPENS_ON_DESIGNATION,
    'announces_attach_target': _ANNOUNCES_TARGET,
    'announces_but_attaches_elsewhere': _ANNOUNCES_BUT_ATTACHES_ELSEWHERE,
}


def write_fake_triage(src_root: Path, *, variant: str) -> Path:
    """Lay down a standalone triage module at *src_root* and return *src_root*.

    *src_root* is what ``--src-root`` takes: the directory that CONTAINS the
    ``fused_memory`` package (i.e. the analogue of ``fused-memory/src``).

    ``variant='missing'`` writes no module at all — the fail-closed case where
    the ref carries nothing importable.
    """
    src_root.mkdir(parents=True, exist_ok=True)
    if variant == 'missing':
        return src_root
    server = src_root / 'fused_memory' / 'server'
    server.mkdir(parents=True, exist_ok=True)
    (src_root / 'fused_memory' / '__init__.py').write_text('')
    (server / '__init__.py').write_text('')
    (server / 'write_triage.py').write_text(TRIAGE_PREAMBLE + VARIANT_TAILS[variant])
    return src_root
