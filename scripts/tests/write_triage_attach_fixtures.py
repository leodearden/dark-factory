"""Standalone stand-ins for the two modules the flip gate's probes execute.

``fused_memory.server.write_triage`` first, and — below the triage variants —
``fused_memory.server.write_triage_judge``, because option (b) lives in the
judge module and item 5 has to be able to see it there.

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


#: A CORRECT option-(a) remedy — and the one shape a naive "the attach id
#: equals the designated id" test false-FAILs. ``_canonical_id_of``'s own
#: contract makes the hoist MANDATORY: attaching to a child would create a
#: grandchild that can never fold under the true canonical, which reads as
#: content loss. So a remedy that threads the designation must still hoist a
#: designated CHILD to its parent before attaching, and a probe that demanded
#: literal equality would tell its implementer to delete that hoist.
_CONSUMES_AND_HOISTS = r'''

def _decode(verdict):
    """(outcome, candidate_id)."""
    if isinstance(verdict, tuple) and len(verdict) == 2:
        return verdict
    return None


def _attach_hoisted(candidate_id, candidates, decision):
    """Honour the designation, then hoist it as _canonical_id_of mandates."""
    for candidate in candidates or ():
        if getattr(candidate, 'id', None) == candidate_id:
            return _canonical_id_of(candidate)
    return decision.canonical_id


triage_write = _make_designating_triage_write(_decode, _attach_hoisted)
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


#: The triage-side announcement channel, which nothing in this codebase uses:
#: here the CALLER picks the attach target and announces it to the judge
#: through a kwarg beyond main's five. Real option (b) announces it in the
#: JUDGE module instead (see the judge stand-ins below), so this models a
#: HYPOTHETICAL remedy rather than the one that landed. It is kept because the
#: invariant it asserts -- an announced target the write must honour -- is
#: sound and costs nothing, and because a branch with no fixture is a branch
#: nobody has run.
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


#: THE UNUSABLE REFS. Each is a way the probe can be pointed at a tree it
#: cannot decide the invariant on, and every one of them must land on
#: UNVERIFIABLE rather than on a verdict — an unverifiable invariant is not a
#: satisfied one, and this gate authorises a production flag flip.
_RAISES_ON_CALL = r"""

async def triage_write(memory_service, *, content, project_id, counter,
                       judge=None, allow_near_duplicate=False,
                       caller_owns_attach_keys=False):
    raise RuntimeError('triage_write is not usable in this ref')
"""


#: SystemExit is NOT an Exception, so an `except Exception` around the import
#: does not catch it: it propagates through the probe and out of the
#: interpreter, which exits 0 having printed nothing — and a gate that greps
#: for a FAIL marker reads silence as a PASS. This is a measured escape, not a
#: hypothetical one; the item-1 probe once exited 0 this way.
_EXITS_DURING_IMPORT = r"""

raise SystemExit(0)


async def triage_write(memory_service, *, content, project_id, counter,
                       judge=None, allow_near_duplicate=False,
                       caller_owns_attach_keys=False):
    return BandDecision(OUTCOME_STORED, None, None, None, None)
"""


#: A plain `def`: calling it returns a BandDecision rather than something to
#: await, so nothing the probe measures came from executing the write path.
_NOT_AWAITABLE = r"""

def triage_write(memory_service, *, content, project_id, counter,
                 judge=None, allow_near_duplicate=False,
                 caller_owns_attach_keys=False):
    return BandDecision(OUTCOME_RESTATED, 'm0', 0.6, None, None)
"""


#: Reaches the judge — so the probe's "was the judge slot reached?" check
#: passes — and then returns something with no `canonical_id` at all. Every
#: attach id the probe would read is then None, which is neither the band's
#: top-1 nor a designation: a probe that only compared ids would report NOT
#: CONSUMED and send an operator to fix a consumption defect that this run
#: never measured.
_RETURNS_NON_DECISION = r"""

async def triage_write(memory_service, *, content, project_id, counter,
                       judge=None, allow_near_duplicate=False,
                       caller_owns_attach_keys=False):
    decision, candidates = await _band_and_candidates(
        memory_service, content, project_id, counter,
    )
    if decision.outcome == OUTCOME_JUDGE:
        await (judge or _stub_judge)(
            memory_service=memory_service,
            content=content,
            project_id=project_id,
            decision=decision,
            candidates=candidates,
        )
    return OUTCOME_RESTATED
"""


#: A ref that CONSUMES the designation but whose fail-open counter class is
#: gone. The invariant is still decidable — the probe falls back to a counting
#: stand-in — so this must PASS, and must say out loud that it measured
#: fail-opens with a stand-in rather than with the ref's own accounting.
_COUNTER_CLASS_MISSING = _CONSUMES_TUPLE + r"""

del TriageFailOpenCounter
"""


#: variant name -> the ``triage_write`` that defines it. Appended to
#: :data:`TRIAGE_PREAMBLE` by :func:`write_fake_triage`.
VARIANT_TAILS: dict[str, str] = {
    'band_top1': _BAND_TOP1,
    'consumes_designated_id': _CONSUMES_TUPLE,
    'consumes_designated_dict': _CONSUMES_DICT,
    'consumes_designated_object': _CONSUMES_OBJECT,
    'consumes_designated_and_hoists': _CONSUMES_AND_HOISTS,
    'hardcodes_last_candidate': _HARDCODES_LAST,
    'fail_opens_on_designation': _FAIL_OPENS_ON_DESIGNATION,
    'announces_attach_target': _ANNOUNCES_TARGET,
    'announces_but_attaches_elsewhere': _ANNOUNCES_BUT_ATTACHES_ELSEWHERE,
    'raises_on_triage_write': _RAISES_ON_CALL,
    'exits_during_import': _EXITS_DURING_IMPORT,
    'not_awaitable': _NOT_AWAITABLE,
    'returns_non_decision': _RETURNS_NON_DECISION,
    'counter_class_missing': _COUNTER_CLASS_MISSING,
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


#: ============================= JUDGE STAND-INS ==============================
#:
#: Option (b) in THIS codebase lives in the JUDGE module, not in the triage
#: module: ``judge_write`` already holds the ``decision``, so it reads
#: ``decision.canonical_id`` and hands it to ``build_judge_prompt``, and
#: ``triage_write`` stays byte-identical to :data:`_BAND_TOP1`. A probe that
#: looked only at what ``triage_write`` tells its judge therefore cannot see
#: that remedy at all — so item 5 reads the judge module too, and these are the
#: stand-ins it is measured against.
#:
#: DELIBERATELY SEPARATE from the item-1 judge fixtures in
#: ``test_check_write_triage_flip_preconditions.py``. Those are shaped for
#: prompt-TEXT assertions — item 1 renders a prompt twice and diffs it — and
#: none of them defines ``judge_write`` at all, which is what keeps every
#: existing pairing's item-5 verdict unchanged. What item 5 needs is the
#: opposite half: a ``judge_write`` whose call to ``build_judge_prompt`` is the
#: thing under test, and a prompt body that matters to nobody.
JUDGE_PREAMBLE = r'''"""Standalone stand-in for fused_memory.server.write_triage_judge.

Dependency-free by construction: the probe imports this out of a bare directory
tree, so it may import nothing but the stdlib.

Only the two functions item 5 reads are defined. The prompt body is trivial on
purpose: what item 5 measures is whether judge_write TELLS the renderer which
candidate the attach will touch, never what the rendering then says about it.
"""
from __future__ import annotations

OUTCOME_RESTATED = 'restated'


def _render(content, candidates, marked=None):
    lines = ['NEW ENTRY:', str(content)]
    for candidate in candidates or ():
        ident = getattr(candidate, 'id', None)
        mark = '  <- attach target' if ident == marked else ''
        lines.append('- id: ' + str(ident) + mark)
    return '\n'.join(lines)


def _require_provider(memory_service):
    """main's unresolvable-provider raise, which fires before any model call."""
    config = getattr(memory_service, 'config', None)
    provider = getattr(getattr(config, 'llm', None), 'provider', None)
    if not isinstance(provider, str) or not provider:
        raise RuntimeError('no judge provider is configured')
    return provider
'''


#: THE REMEDY THAT HAS LANDED, end to end: the renderer can be told which
#: candidate the attach will touch, and ``judge_write`` feeds it the very id the
#: write will use. Consumption holds BY CONSTRUCTION here — announced target
#: and attach target are one expression — which is why item 5 may not demand a
#: measured swap of it.
_JUDGE_FEEDS_TARGET = r'''

def build_judge_prompt(content, candidates, *, attach_target_id=None):
    return _render(content, candidates, marked=attach_target_id)


async def judge_write(*, memory_service, content, project_id, decision,
                      candidates=()):
    attach_target_id = getattr(decision, 'canonical_id', None)
    build_judge_prompt(content, candidates, attach_target_id=attach_target_id)
    return OUTCOME_RESTATED
'''


#: The same remedy, reached only by READING the call. Its judge_write raises
#: before it renders — an unresolvable provider, which is what main's own judge
#: does on a deployment with no key — so the recorder never fires and a probe
#: with no static route would report a correct option (b) as absent.
_JUDGE_FEEDS_TARGET_AFTER_RAISING = r'''

def build_judge_prompt(content, candidates, *, attach_target_id=None):
    return _render(content, candidates, marked=attach_target_id)


async def judge_write(*, memory_service, content, project_id, decision,
                      candidates=()):
    attach_target_id = getattr(decision, 'canonical_id', None)
    _require_provider(memory_service)
    build_judge_prompt(content, candidates, attach_target_id=attach_target_id)
    return OUTCOME_RESTATED
'''


#: The shape this codebase's judge had BEFORE option (b): the renderer cannot be
#: told anything, so every candidate looks alike to the model and the attach
#: target is whatever the band picked.
_JUDGE_NO_TARGET_PARAMETER = r'''

def build_judge_prompt(content, candidates):
    return _render(content, candidates)


async def judge_write(*, memory_service, content, project_id, decision,
                      candidates=()):
    build_judge_prompt(content, candidates)
    return OUTCOME_RESTATED
'''


#: A WIDENED SIGNATURE, and nothing else. The parameter exists and is never
#: fed, so the prompt is byte-identical to the one above and the model is told
#: nothing. This is the whole difference between a signature and consumption —
#: and it is also the shape of all 17 target-carrying item-1 fixtures, none of
#: which defines judge_write, so it is what keeps their item-5 verdicts inert.
_JUDGE_TARGET_NEVER_FED = r'''

def build_judge_prompt(content, candidates, *, attach_target_id=None):
    return _render(content, candidates, marked=attach_target_id)


async def judge_write(*, memory_service, content, project_id, decision,
                      candidates=()):
    build_judge_prompt(content, candidates)
    return OUTCOME_RESTATED
'''


#: Fed, but fed the WRONG id: the slate's first entry rather than the one the
#: write will attach to. The model is then told to reason about a candidate the
#: verdict will not be filed against, which is item 1's harm with an extra step.
_JUDGE_FEEDS_A_DIFFERENT_ID = r'''

def build_judge_prompt(content, candidates, *, attach_target_id=None):
    return _render(content, candidates, marked=attach_target_id)


async def judge_write(*, memory_service, content, project_id, decision,
                      candidates=()):
    slate = list(candidates or ())
    elsewhere = getattr(slate[0], 'id', None) if slate else None
    build_judge_prompt(content, candidates, attach_target_id=elsewhere)
    return OUTCOME_RESTATED
'''


#: judge variant name -> the tail appended to :data:`JUDGE_PREAMBLE`.
JUDGE_VARIANT_TAILS: dict[str, str] = {
    'feeds_attach_target': _JUDGE_FEEDS_TARGET,
    'feeds_attach_target_after_raising': _JUDGE_FEEDS_TARGET_AFTER_RAISING,
    'no_target_parameter': _JUDGE_NO_TARGET_PARAMETER,
    'target_never_fed': _JUDGE_TARGET_NEVER_FED,
    'feeds_a_different_id': _JUDGE_FEEDS_A_DIFFERENT_ID,
}


def write_fake_judge(src_root: Path, *, variant: str) -> Path:
    """Lay a standalone judge module beside the triage one, and return *src_root*.

    The SAME ``<src_root>/fused_memory/server/`` tree the triage stand-in goes
    into, because that is how the gate ships them: one ``git archive`` of one
    source tree, read by both probe items.

    ``variant='missing'`` writes no module at all — the case every fixture
    written before this one is in, and the one that must leave their verdicts
    exactly as they were.
    """
    if variant == 'missing':
        return src_root
    server = src_root / 'fused_memory' / 'server'
    server.mkdir(parents=True, exist_ok=True)
    (src_root / 'fused_memory' / '__init__.py').write_text('')
    (server / '__init__.py').write_text('')
    (server / 'write_triage_judge.py').write_text(
        JUDGE_PREAMBLE + JUDGE_VARIANT_TAILS[variant],
    )
    return src_root
