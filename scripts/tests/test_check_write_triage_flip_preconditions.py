"""Tests for the write_triage flip gate's item 1 (task 4810).

Two units are under test, and they are deliberately separate files:

  * ``scripts/check_write_triage_attach_target.py`` — the PROBE. It imports a
    judge module out of a bare source tree and decides, by EXECUTING it,
    whether the judge path binds a verdict to a determinate candidate.
  * ``scripts/check_write_triage_flip_preconditions.sh`` — the GATE. Item 1
    delegates to the probe through the ``CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY``
    env seam (modelled on ``scripts/check_sandbox_soak.sh``'s
    ``CHECK_SANDBOX_SOAK_PY``).

WHY THE FIXTURES ARE STANDALONE. The probe is pointed at a ``--src-root`` and
imports ``fused_memory.server.write_triage_judge`` from it. A fixture module
that imported pydantic — or anything else the real judge module pulls in —
would make these tests depend on the fused-memory virtualenv. Every fixture
here is stdlib-only, so the suite runs under whatever interpreter collected it.

WHY THE SLATE IS BUILT BY THE FIXTURE'S OWN ``select_judge_candidates``. The
fixture mirrors main's selection EXACTLY, including the winner-rescue APPEND
(``selected = [*selected[: max(n - 1, 0)], winner]``). On a hoisted-parent
slate the rescued evidence child therefore lands LAST — measured on main
f474347580: ``select_judge_candidates(results, 3, canonical_id='parent-1')``
returned ``['m0', 'm1', 'child-1']``. An implementation that marks
``candidates[0]`` as the attach target is WRONG on exactly that slate, so the
probe must be driven on it or the gate would bless the bug it exists to catch.

Assertions are on EXIT CODES and on the probe's own structured stdout
vocabulary — never on the wording of a fixture's rendered prompt. Pinning
prompt prose is the meta-test class this repo deletes (task 3128 steps 23-25).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_SCRIPT = _REPO_ROOT / 'scripts' / 'check_write_triage_flip_preconditions.sh'
_PROBE = _REPO_ROOT / 'scripts' / 'check_write_triage_attach_target.py'

# The probe's stable stdout vocabulary. Kept as named constants so the tests
# and the probe cannot drift apart silently, and so it is obvious at a glance
# that no PROMPT wording is being pinned — only the probe's own verdict lines.
_INDETERMINATE = 'INDETERMINATE'
_UNVERIFIABLE = 'UNVERIFIABLE'
_RESCUE_PATH = 'select_judge_candidates'
_BYTE_IDENTICAL = 'byte-identical'
_MARKS_FIRST = 'distinguishes slate[0]'
_SWAP_HELD = 'rendering depends on which candidate is the attach target'
_OPTION_A_HELD = 'the verdict itself names its candidate'
_ECHOED = 'ECHOED into the prompt'
_ECHO_FORGIVEN = 'ACCEPTED ON AN ECHOED, TARGET-NAMED PARAMETER'
_BARE_STR = 'returns a bare str'
_OUTSIDE_SRC_ROOT = 'outside --src-root'


# ---------------------------------------------------------------------------
# Fixture judge modules
# ---------------------------------------------------------------------------

_JUDGE_PREAMBLE = r'''"""Standalone stand-in for fused_memory.server.write_triage_judge.

Dependency-free by construction: the probe imports this out of a bare
directory tree, so it may import nothing but the stdlib.

select_judge_candidates mirrors main's EXACTLY, including the winner-rescue
APPEND that puts a rescued hoisted-parent's evidence child LAST.
"""
from __future__ import annotations

import json

PARENT_ID_KEY = 'parent_id'
VERDICT_KEY = 'verdict'
JUDGE_VERDICTS = {
    'distinct': 'stored',
    'restates': 'restated',
    'amends': 'amended',
    'contests': 'contested',
}

_DEFAULT_JUDGE_CANDIDATE_COUNT = 5


def _cosine_of(result):
    value = (result.metadata or {}).get('store_score')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def select_judge_candidates(results, n, *, canonical_id):
    scored = []
    for result in results or ():
        cosine = _cosine_of(result)
        if cosine is not None:
            scored.append((cosine, result))
    if not scored:
        return []
    scored.sort(key=lambda pair: pair[0], reverse=True)
    ordered = [result for _, result in scored]
    if n <= 0:
        n = _DEFAULT_JUDGE_CANDIDATE_COUNT
    selected = ordered[:n]
    if canonical_id is not None and all(r.id != canonical_id for r in selected):
        winner = next(
            (r for r in ordered if r.id == canonical_id), None,
        ) or next(
            (
                r
                for r in ordered
                if (r.metadata or {}).get(PARENT_ID_KEY) == canonical_id
            ),
            None,
        )
        if winner is not None and winner not in selected:
            selected = [*selected[: max(n - 1, 0)], winner]
    return selected


def _render_candidate(candidate):
    return ['- id: ' + str(candidate.id), '  text: ' + str(candidate.content)]


def _parse_bare_str(raw):
    """main's shape: a bare verdict word, naming no candidate."""
    payload = json.loads(raw)
    word = payload.get(VERDICT_KEY)
    if not isinstance(word, str):
        raise ValueError('no string verdict key')
    verdict = JUDGE_VERDICTS.get(word.strip().lower())
    if verdict is None:
        raise ValueError('not a judge verdict word')
    return verdict
'''


_VARIANT_TAILS = {
    # main's shape exactly: an undifferentiated candidate list and a bare-str
    # verdict. Neither remedy present -> the attach target is indeterminate.
    'flat': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # Importable, but blows up when the probe renders. An unverifiable
    # invariant is not a satisfied one: this must FAIL, never pass.
    'raises': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    raise RuntimeError('fixture: build_judge_prompt blows up')


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # Takes an attach-target argument but labels candidates[0] regardless. On a
    # hoisted-parent slate that is the WRONG candidate — the rescued evidence
    # child is LAST — so this must FAIL. Its rendering is independent of the
    # argument, and slate[0]'s id is mentioned one extra time.
    'positional_target': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '']
    lines.append('ATTACH TARGET: ' + str(candidates[0].id))
    lines.append('')
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # The PROSE-ONLY fix: the signature grew an attach-target argument and the
    # prompt grew a sentence, but the rendering is byte-identical whichever
    # candidate is named. A source-text grep would pass this; the swap test
    # cannot.
    'ignores_target': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    lines.append('')
    lines.append('One of the candidates above is the one this write will attach to.')
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # A real option (b): whichever candidate matches the passed target id is
    # the one lifted out, wherever it sits in the slate.
    'by_id': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    target = [c for c in candidates if c.id == attach_target_id]
    others = [c for c in candidates if c.id != attach_target_id]
    lines = ['NEW ENTRY:', str(content), '', 'THE CANDIDATE THIS WRITE WILL ATTACH TO:']
    for candidate in target:
        lines.extend(_render_candidate(candidate))
    lines.append('')
    lines.append('OTHER CANDIDATES (CONTEXT ONLY):')
    for candidate in others:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # Byte-different heading text, identical id-level behaviour. Proves the
    # gate pins no prompt wording: reword freely, the check still passes.
    'by_id_reworded': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    target = [c for c in candidates if c.id == attach_target_id]
    others = [c for c in candidates if c.id != attach_target_id]
    lines = ['submitted:', str(content), '', '>>> the record this attaches to <<<']
    for candidate in target:
        lines.extend(_render_candidate(candidate))
    lines.append('')
    lines.append('~~~ background only, do not attach ~~~')
    for candidate in others:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # OPTION (a): the prompt marks nothing — main's two-argument
    # build_judge_prompt and its flat list — but the VERDICT carries the judged
    # candidate, so the model's answer names its own candidate and slate
    # position becomes irrelevant. The shape task 4762's own design decision
    # says option (a) produces: an (outcome, candidate_id) pair.
    # An id-less non-scalar return: the outcome grew a richer shape (a dict, a
    # dataclass, an (outcome, None) pair) for reasons having nothing to do with
    # candidate binding. Non-scalar is a SHAPE that could carry an id, never
    # evidence that it does — this must NOT satisfy option (a).
    'idless_non_scalar': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return {'outcome': _parse_bare_str(raw)}
''',
    # The same trap wearing a tuple: the ARITY of an (outcome, candidate_id)
    # pair without the id. A type test passes it; an evidence test cannot.
    'idless_pair': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return (_parse_bare_str(raw), None)
''',
    # A REAL option (b) that marks the target INLINE on its own line rather than
    # hoisting it. Arguably the most natural minimal fix. The target id keeps
    # its line INDEX and changes only its TEXT, so an index-only footprint
    # rejects it — which would re-block 3169 against a valid fix.
    'inline_marker': r'''

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if candidate.id == attach_target_id else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    # Renders differently every call for a reason unrelated to the attach
    # target, and does it by INSERTING lines — which shifts every candidate's
    # line index. Byte-inequality and a shifted footprint are both present, so
    # only the 'does any differing line mention a candidate' condition rejects
    # it.
    'spurious_inserted_lines': r'''
_CALLS = [0]

def build_judge_prompt(content, candidates, attach_target_id=None):
    _CALLS[0] += 1
    lines = ['NEW ENTRY:', str(content)]
    lines.extend('pad %d' % i for i in range(_CALLS[0]))
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
''',
    'option_a': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw, candidate_ids=None):
    payload = json.loads(raw)
    word = payload.get(VERDICT_KEY)
    verdict = JUDGE_VERDICTS.get(str(word).strip().lower())
    if verdict is None:
        raise ValueError('not a judge verdict word')
    return (verdict, payload.get('candidate_id'))
''',
    # An ECHO wearing option (a)'s clothes: the outcome grew a diagnostics /
    # logging field that hands the whole submitted payload back. It discards
    # the candidate entirely — nothing here READS candidate_id — yet the id
    # the probe supplied is recoverable from the result, and two payloads
    # naming different candidates parse to different values. Reviewer-reported
    # false pass: the pre-fix probe printed "option (a): satisfied" and exited
    # 0, authorising the production flip.
    'echoes_payload': r"""

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return {'outcome': _parse_bare_str(raw), 'raw_payload': json.loads(raw)}
""",
    # The same echo one level cruder: the RAW REQUEST TEXT carried alongside
    # the outcome. The id is a substring of that text, never a value the
    # parser extracted, so containment-style recovery (the old ``repr``
    # fallback) blessed it. Also a false pass before the fix.
    'echoes_raw_string': r"""

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return (_parse_bare_str(raw), raw)
""",
    # A REAL option (b) that identifies the target by OBJECT IDENTITY rather
    # than by id. Handed the candidate object it renders correctly; handed an
    # id string it matches nothing and renders every candidate the same way.
    # Reviewer-reported false FAIL: the old 'pick the first spelling that does
    # not raise' picked the id spelling (which never raises here), so the
    # object spelling was dead for exactly the implementations needing it.
    'object_identity_target': r"""

def build_judge_prompt(content, candidates, attach_target=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if candidate is attach_target else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
""",
    # A REAL option (b) whose target parameter is NOT the third one. The old
    # probe fed the id into ``verdict_words`` unconditionally, so the target
    # stayed None, both renderings came out identical, and a correct fix was
    # failed with the factually wrong diagnostic that the rendering "does not
    # depend on the argument at all".
    'target_in_fourth_position': r"""

def build_judge_prompt(content, candidates, *, verdict_words=None, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if candidate.id == attach_target_id else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
""",
    # A STRICTER option (a): the parser refuses a verdict that carries no
    # candidate id at all, and refuses one naming a candidate outside the
    # slate. It therefore RAISES on a bare {"verdict": ...} payload — which
    # must be read as inconclusive for the branch, not as a failure.
    'option_a_rejects_dangling': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw, candidate_ids=None):
    payload = json.loads(raw)
    word = payload.get(VERDICT_KEY)
    verdict = JUDGE_VERDICTS.get(str(word).strip().lower())
    if verdict is None:
        raise ValueError('not a judge verdict word')
    candidate_id = payload.get('candidate_id')
    if not isinstance(candidate_id, str):
        raise ValueError('verdict carries no candidate_id')
    if candidate_ids is not None and candidate_id not in candidate_ids:
        raise ValueError('verdict names a candidate outside the slate')
    return (verdict, candidate_id)
''',
    # The STRICTEST option (a), and the exact shape the gate's own old failure
    # text prescribed: "validate it against the slate in parse_judge_verdict".
    # The slate argument is REQUIRED POSITIONALLY, so a single-argument call
    # raises TypeError before the parser is ever exercised. Reviewer-reported
    # false FAIL: the probe called `parse(raw)` only, reported option (a)
    # "accepted no probe payload", fell through to option (b) -- which this
    # (correctly untouched) flat prompt cannot satisfy -- and exited 1. A
    # complete, correct option-(a) fix landing on main would have left the gate
    # failing closed forever and task 3169 permanently re-blocked.
    'option_a_validating': r'''

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw, candidates):
    payload = json.loads(raw)
    word = payload.get(VERDICT_KEY)
    verdict = JUDGE_VERDICTS.get(str(word).strip().lower())
    if verdict is None:
        raise ValueError('not a judge verdict word')
    candidate_id = payload.get('candidate_id')
    known = {getattr(c, 'id', c) for c in candidates}
    if candidate_id not in known:
        raise ValueError('verdict names a candidate outside the slate')
    return (verdict, candidate_id)
''',
    # An UNRELATED prompt-legibility change wearing a target-ish name. The band
    # canonical is interpolated into a header line; nothing binds any verdict to
    # any candidate, the candidate list stays flat and undifferentiated, and
    # parse_judge_verdict still returns a bare str. Reviewer-reported false
    # PASS: `canonical` was in _TARGET_NAME_RE, so the echo control was forgiven
    # and the probe printed "option (b): satisfied ... via 'canonical_id'" and
    # exited 0 -- authorising the production flip on a change that establishes
    # nothing.
    'unrelated_canonical_header': r"""

def build_judge_prompt(content, candidates, canonical_id=None):
    lines = ['NEW ENTRY:', str(content), '']
    lines.append('BAND CANONICAL: ' + str(canonical_id))
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
""",
    # A FALSE PASS in the option-(b) half, symmetric to `echoes_payload` in the
    # option-(a) half. The extra parameter is free-text diagnostics that is
    # merely INTERPOLATED into the prompt; nothing in the module binds a verdict
    # to a candidate, and the candidate list stays flat and undifferentiated.
    # Both swap conditions are nevertheless met when the probe feeds it a
    # candidate id — the NOTE line mentions an id, and each id's footprint
    # gains/loses that line — so an unguarded search blesses it and authorises
    # the production flip.
    'echoed_free_text': r"""

def build_judge_prompt(content, candidates, *, footer_note=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    if footer_note is not None:
        lines.append('NOTE: ' + str(footer_note))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
""",
    # The same echoing shape, but the module ALSO carries a real target-named
    # marker parameter. The search must reach the real one rather than stopping
    # at the first parameter whose swap test happens to hold.
    'echoed_free_text_plus_real_target': r"""

def build_judge_prompt(content, candidates, *, footer_note=None, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if candidate.id == attach_target_id else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    if footer_note is not None:
        lines.append('NOTE: ' + str(footer_note))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
""",
    # A REAL option (b) that names the target in a HEADER rather than marking it
    # in the list. Structurally this echoes its argument exactly like
    # `echoed_free_text` does — the ONLY thing separating them is that the
    # parameter is named for what it designates. It must still pass.
    'target_named_header': r"""

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '']
    lines.append('THE ATTACH TARGET IS: ' + str(attach_target_id))
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
""",
}


# A REAL option (a) that ALSO carries the parsed payload for diagnostics,
# paired with main's flat prompt so only option (a) can save it. It genuinely
# binds the verdict to the candidate -- element [1] is read from the contract
# -- but the echo control finds the decoy id inside the carried payload and
# reports 'ECHOED, not bound'. Reviewer-reported false FAIL: the control
# cannot tell 'reads the field' from 'reproduces the input' when BOTH are
# true. What separates them is that a binder surfaces the id one MORE time
# than it reproduces it.
_VARIANT_TAILS['option_a_with_payload'] = r"""

def build_judge_prompt(content, candidates):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    payload = json.loads(raw)
    return (_parse_bare_str(raw), payload.get('candidate_id'), payload)
"""


# A REAL option (b) whose target parameter is reachable only by ALSO supplying
# a REQUIRED parameter that sits between the slate and it. Reviewer-reported
# false FAIL (cycle 5): the probe called `fn(content, slate, **{target: value})`
# and supplied nothing for the intervening required parameter, so every attempt
# raised TypeError('missing 1 required positional argument'), the non-target
# parameter rendered instead, and a correct fix was reported INDETERMINATE --
# re-blocking task 3169 against a valid fix, one signature shape down from the
# defect this whole probe exists to remove.
_VARIANT_TAILS['required_intervening_target'] = r"""

def build_judge_prompt(content, candidates, verdict_words, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'VERDICT WORDS: ' + str(verdict_words)]
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if candidate.id == attach_target_id else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
"""


# The same defect one turn harder, on two axes: the required intervening
# parameter is KEYWORD-ONLY (so it is missing whatever the target's position
# is), and it is ITERATED rather than str()'d -- so the first placeholder the
# probe tries (None) raises TypeError of its own and the ladder has to advance
# to one that is iterable. A placeholder ladder that stopped at None would
# report this correct fix as INDETERMINATE too.
_VARIANT_TAILS['required_keyword_only_target'] = r"""

def build_judge_prompt(content, candidates, *, verdict_words, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'WORDS: ' + ', '.join(verdict_words)]
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if candidate.id == attach_target_id else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
"""


# A target-NAMED parameter that cannot be exercised at all, alongside a
# free-text one that renders happily. Nothing was ever MEASURED about the
# invariant here -- the only parameter that could have decided it never ran --
# so the honest verdict is UNVERIFIABLE (which still fails closed), not the
# factual assertion that the attach target is INDETERMINATE.
_VARIANT_TAILS['target_parameter_always_raises'] = r"""

def build_judge_prompt(content, candidates, footer_note=None, attach_target_id=None):
    if attach_target_id is not None:
        raise RuntimeError('fixture: the target parameter cannot be rendered')
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    if footer_note is not None:
        lines.append('NOTE: ' + str(footer_note))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
"""

# The winner-rescue APPEND is a MECHANISM, not the invariant. These two
# variants redefine select_judge_candidates to plain top-n, shadowing the
# preamble's, so the hoisted parent's evidence child never enters the slate.
# Nothing about "which candidate will the attach touch" has changed.
_NO_RESCUE_SELECTOR = r"""

def select_judge_candidates(results, n, *, canonical_id):
    # Plain top-n by cosine: a later refactor that reaches the hoisted parent
    # some other way, so no winner-rescue APPEND happens here.
    scored = [
        (_cosine_of(r), r)
        for r in results or ()
        if _cosine_of(r) is not None
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    limit = n if n > 0 else _DEFAULT_JUDGE_CANDIDATE_COUNT
    return [r for _, r in scored[:limit]]
"""

# A correct option (b) -- an inline id marker -- on a selector that does not
# rescue. The invariant HOLDS, so the gate must open.
_VARIANT_TAILS['no_rescue'] = _NO_RESCUE_SELECTOR + r"""

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if str(candidate.id) == str(attach_target_id) else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
"""

# The control for the above: the SAME non-rescuing selector, but the judge
# marks candidates[0] regardless. Tolerating a non-rescued slate must not cost
# the probe its ability to catch that.
_VARIANT_TAILS['no_rescue_positional'] = _NO_RESCUE_SELECTOR + r"""

def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '']
    lines.append('ATTACH TARGET: ' + str(candidates[0].id))
    lines.append('')
    lines.append('EXISTING CANDIDATES:')
    for candidate in candidates:
        lines.extend(_render_candidate(candidate))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
"""


# A judge whose code calls sys.exit() -- realistically a lazily-imported
# dependency's import guard, or an argparse-style bail. SystemExit is a
# BaseException and NOT an Exception, so it escapes _build_slate's
# `except Exception`, escapes _probe, escapes main()'s last-resort arm, and
# terminates the process with ITS OWN code. `raise SystemExit(0)` therefore
# exits 0 having printed NOTHING at all, and any caller trusting rc==0 alone
# reads that as PASS. Total fail-open on a gate whose EXIT-CODE CONTRACT says
# an invariant that could not be verified is not a satisfied one.
_VARIANT_TAILS['exits_during_select'] = r"""


def select_judge_candidates(results, n, *, canonical_id):
    raise SystemExit(0)


def build_judge_prompt(content, candidates, attach_target_id=None):
    return 'unreachable'


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
"""

# A CORRECT option (b) whose selector rescues the hoisted parent's evidence
# child by HOISTING IT TO THE FRONT rather than appending it. The append is a
# MECHANISM -- TestRescueAppendIsAMechanism already says so in as many words --
# but _build_slate took ids.index(_CHILD_ID) unconditionally, so the target
# became slate[0], _probe's `other is target` guard fired, and a valid fix was
# failed UNVERIFIABLE for a reason having nothing to do with the invariant.
# That is the mechanism-pin false FAIL task 4810 exists to remove, one level
# down from where it was removed.
_VARIANT_TAILS['front_hoisting_rescue'] = r"""


def select_judge_candidates(results, n, *, canonical_id):
    ordered = [r for r in (results or ()) if _cosine_of(r) is not None]
    ordered.sort(key=_cosine_of, reverse=True)
    if not ordered:
        return []
    if n <= 0:
        n = _DEFAULT_JUDGE_CANDIDATE_COUNT
    selected = ordered[:n]
    if canonical_id is not None and all(r.id != canonical_id for r in selected):
        winner = next(
            (
                r
                for r in ordered
                if (r.metadata or {}).get(PARENT_ID_KEY) == canonical_id
            ),
            None,
        )
        if winner is not None and winner not in selected:
            # PREPEND, where main appends. Same rescue, other end.
            selected = [winner, *selected[: max(n - 1, 0)]]
    return selected


def build_judge_prompt(content, candidates, attach_target_id=None):
    lines = ['NEW ENTRY:', str(content), '', 'EXISTING CANDIDATES:']
    for candidate in candidates:
        mark = '  <-- ATTACH TARGET' if candidate.id == attach_target_id else ''
        lines.append('- id: ' + str(candidate.id) + mark)
        lines.append('  text: ' + str(candidate.content))
    return '\n'.join(lines)


def parse_judge_verdict(raw):
    return _parse_bare_str(raw)
"""


def _write_fake_judge(src_root: Path, *, variant: str) -> Path:
    """Lay down a standalone judge module at *src_root* and return *src_root*.

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
    (server / 'write_triage_judge.py').write_text(
        _JUDGE_PREAMBLE + _VARIANT_TAILS[variant],
    )
    return src_root


def _run_probe(src_root: Path, *, env: dict[str, str] | None = None):
    """Drive the probe directly under this interpreter."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(_PROBE), '--src-root', str(src_root)],
        capture_output=True,
        text=True,
        timeout=120,
        env=full_env,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAttachTargetProbe:
    """The probe decides the attach-target invariant by EXECUTING the judge
    module it is pointed at, never by inspecting its source text."""

    def test_gate_script_exists_and_is_executable(self):
        # DeterministicRunner's _default_run_script executes the predicate
        # directly, not via `bash`, so the +x bit and the shebang are
        # load-bearing.
        assert _GATE_SCRIPT.exists(), f'gate script missing at {_GATE_SCRIPT}'
        assert os.access(_GATE_SCRIPT, os.X_OK), f'gate script not executable: {_GATE_SCRIPT}'

    def test_probe_exists_and_is_executable(self):
        assert _PROBE.exists(), f'probe missing at {_PROBE}'
        assert os.access(_PROBE, os.X_OK), f'probe not executable: {_PROBE}'

    def test_flat_judge_fails_with_indeterminate_target(self, tmp_path):
        """main's shape: no attach-target parameter and a bare-str verdict."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='flat')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed on main\'s shape:\n{proc.stdout}\n{proc.stderr}'
        assert _INDETERMINATE in proc.stdout, proc.stdout

    def test_missing_module_fails_closed(self, tmp_path):
        """Nothing importable at --src-root is UNVERIFIABLE, so it FAILS."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='missing')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed on an empty tree:\n{proc.stdout}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_raising_module_fails_closed(self, tmp_path):
        """A judge module that raises when rendered is UNVERIFIABLE, not OK."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='raises')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed on a raising module:\n{proc.stdout}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout


class TestSwapAndRescuePath:
    """The two properties that stop item 1 being satisfiable by prose, or by
    the measured ``candidates[0]`` bug.

    SWAP TEST — render the SAME slate twice against two different attach
    targets and require the two renderings to DIFFER. A sentence added to the
    prompt renders identically whichever candidate is the target, so a
    prose-only change cannot pass by construction. Nothing here matches
    heading text, so any rewording survives.

    RESCUE PATH — the fixture's ``select_judge_candidates`` appends a rescued
    hoisted-parent winner LAST (mirroring main). An implementation that labels
    ``candidates[0]`` therefore names the WRONG candidate on this slate, which
    is the same harm item 1 exists to prevent, merely relocated.
    """

    def test_marking_candidates_zero_fails_and_names_the_rescue_path(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='positional_target')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe blessed candidates[0]:\n{proc.stdout}'
        assert _MARKS_FIRST in proc.stdout, proc.stdout
        # The operator has to be told WHY slate[0] is not the attach target.
        assert _RESCUE_PATH in proc.stdout, proc.stdout

    def test_prose_only_change_fails_the_swap_test(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='ignores_target')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed a prose-only change:\n{proc.stdout}'
        assert _BYTE_IDENTICAL in proc.stdout, proc.stdout

    def test_marking_by_id_passes(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='by_id')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed a real fix:\n{proc.stdout}\n{proc.stderr}'
        assert _SWAP_HELD in proc.stdout, proc.stdout

    def test_rewording_the_headings_still_passes(self, tmp_path):
        """No prompt wording is pinned — only id-level behaviour."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='by_id_reworded')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe pinned prompt wording:\n{proc.stdout}\n{proc.stderr}'
        assert _SWAP_HELD in proc.stdout, proc.stdout


class TestEchoedArgumentIsNotATarget:
    """A parameter whose value is merely INTERPOLATED into the prompt is not an
    attach target, however well it satisfies the swap test.

    Reviewer-reported false pass (esc-4810-6), symmetric to the
    ``echoes_payload`` control in the option-(a) half: feed a candidate id into
    ANY free-text parameter that reaches the rendering and both swap conditions
    hold — a differing line mentions a candidate id, and each id's footprint
    gains or loses that line — while nothing in the module binds a verdict to a
    candidate. Exit 0 there would authorise the production ``write_triage``
    flip the whole gate exists to hold shut.

    The discriminator is a NONCE CONTROL: render against an id that is in no
    candidate's slate. A genuine marker matches it against the slate, marks
    nothing, and renders exactly as it would for any other unknown id; an
    echoing parameter emits the nonce verbatim. Because a target named in a
    HEADER echoes its argument too, an echo is forgiven only when the parameter
    is NAMED for what it designates — and the report says so out loud.
    """

    def test_echoed_free_text_parameter_does_not_satisfy_option_b(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='echoed_free_text')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe blessed an echoed free-text parameter:\n{proc.stdout}'
        assert _INDETERMINATE in proc.stdout, proc.stdout
        assert _ECHOED in proc.stdout, proc.stdout

    def test_search_reaches_the_real_target_past_the_echoing_parameter(self, tmp_path):
        src_root = _write_fake_judge(
            tmp_path / 'src', variant='echoed_free_text_plus_real_target',
        )
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed a real fix:\n{proc.stdout}\n{proc.stderr}'
        assert _SWAP_HELD in proc.stdout, proc.stdout
        assert "'attach_target_id'" in proc.stdout, proc.stdout

    def test_target_named_header_still_passes_but_is_flagged(self, tmp_path):
        """Echoing is forgiven for a target-NAMED parameter, and reported."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='target_named_header')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed a header-marking fix:\n{proc.stdout}\n{proc.stderr}'
        assert _SWAP_HELD in proc.stdout, proc.stdout
        assert _ECHO_FORGIVEN in proc.stdout, proc.stdout


class TestPayloadCarryingBinderIsNotAnEcho:
    """The echo control must not disqualify a parser that BOTH reads the
    candidate field and carries the payload it read it from.

    Reported as a review-cycle-4 suggestion. The control submits a decoy id
    under a field no contract reads and asks whether it comes back out; a
    parser carrying the payload for diagnostics answers yes even though it
    also, separately, binds the verdict to the candidate. Failing it is a
    false FAIL of the same class task 4810 exists to remove.
    """

    def test_binder_that_also_carries_the_payload_passes(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='option_a_with_payload')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            'a genuine option (a) was failed for carrying its own payload:\n'
            f'{proc.stdout}\n{proc.stderr}'
        )
        assert _OPTION_A_HELD in proc.stdout, proc.stdout

    def test_pure_echo_is_still_caught(self, tmp_path):
        """The CONTROL for the above: the discriminator must keep its teeth.

        echoes_payload discards the candidate entirely and surfaces the id
        only by reproducing the input -- exactly ONCE, where the binder above
        surfaces it twice.
        """
        src_root = _write_fake_judge(tmp_path / 'src', variant='echoes_payload')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'a pure echo passed:\n{proc.stdout}'
        assert _OPTION_A_HELD not in proc.stdout, proc.stdout


class TestRescueAppendIsAMechanism:
    """The probe must not pin HOW the attach target comes to sit somewhere
    other than slate[0] — only THAT it can, and that the judge tracks it.

    Reported as a review-cycle-4 suggestion: requiring the winner-rescue
    APPEND is a mechanism pin in the same false-FAIL direction task 4810
    exists to remove. A selector that reaches the hoisted parent another way
    would fail the gate closed forever, re-blocking task 3169 against a
    correct fix for a reason unrelated to the invariant.
    """

    def test_valid_fix_passes_without_the_rescue_append(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='no_rescue')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            'a correct option (b) was failed for its SELECTOR\'s shape:\n'
            f'{proc.stdout}\n{proc.stderr}'
        )
        assert _SWAP_HELD in proc.stdout, proc.stdout

    def test_candidates_zero_still_caught_without_the_rescue_append(self, tmp_path):
        """Tolerating a non-rescued slate must not cost the probe its teeth."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='no_rescue_positional')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, (
            f'a candidates[0]-marking judge passed:\n{proc.stdout}'
        )
        assert _MARKS_FIRST in proc.stdout, proc.stdout

    def test_valid_fix_passes_when_the_rescue_PREPENDS(self, tmp_path):
        """A selector that hoists the winner to the FRONT is still a mechanism.

        _build_slate took ids.index(_CHILD_ID) unconditionally, so a rescue
        that prepends made the attach target slate[0], tripped _probe's
        `other is target` guard, and failed a CORRECT option (b) as
        UNVERIFIABLE. That contradicts this class's own contract in the one
        direction no fixture covered: every existing variant moves the target
        AWAY from index 0, so nothing exercised a selector that reorders
        rather than appends.
        """
        src_root = _write_fake_judge(tmp_path / 'src', variant='front_hoisting_rescue')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            'a correct option (b) was failed because its selector PREPENDED '
            f'the rescued winner:\n{proc.stdout}\n{proc.stderr}'
        )
        assert _SWAP_HELD in proc.stdout, proc.stdout


class TestProbeCannotPassByExiting:
    """The probe must never exit 0 without having asserted the invariant.

    main()'s last-resort arm caught `Exception`, but the probe drives
    arbitrary code out of the ref's own tree, and SystemExit is a
    BaseException — the realistic carrier being a lazily-imported dependency's
    import guard calling sys.exit(). It escaped _build_slate's `except
    Exception`, escaped _probe, escaped main, and terminated the process with
    ITS code.

    Measured before the fix: a judge whose select_judge_candidates raises
    SystemExit(0) made the probe exit 0 having printed ZERO bytes, and the
    gate — which tested `probe_rc -eq 0` alone — printed `PASS item 1` and
    then `RESULT: all preconditions satisfied — the flip may proceed`. A total
    fail-open authorising the production write_triage.enabled flip, on a gate
    whose stated EXIT-CODE CONTRACT is that an invariant which could not be
    verified is not a satisfied one.

    _import_judge already caught BaseException for precisely this reason, so
    the convention existed and was simply not applied at the outer arm. Both
    halves are pinned here: the probe must fail closed, AND the gate must not
    take rc==0 on faith from a report that never claimed a PASS.
    """

    def test_system_exit_in_the_judge_fails_closed(self, tmp_path):
        src_root = _write_fake_judge(tmp_path / 'src', variant='exits_during_select')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, (
            'the probe exited 0 on a judge that called sys.exit():\n'
            f'stdout={proc.stdout!r} stderr={proc.stderr!r}'
        )
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_the_gate_refuses_a_silent_exit_zero_probe(self, tmp_path):
        """End to end: a judge that calls sys.exit() must not reach PASS."""
        repo = _make_gate_repo(tmp_path, judge='exits_during_select', eval_src='fixed')
        proc = _run_gate(
            repo / 'scripts' / 'check_write_triage_flip_preconditions.sh',
            ref=_FIXTURE_REF,
            probe_py=sys.executable,
        )
        assert proc.returncode != 0, (
            f'the gate authorised the flip on a mute probe:\n{proc.stdout}'
        )
        assert 'PASS  item 1' not in proc.stdout, proc.stdout
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_the_gate_alone_refuses_a_mute_exit_zero_probe(self, tmp_path):
        """The gate's SECOND lock, exercised without the probe's help.

        Once the probe fails closed, the end-to-end test above no longer
        reaches the shell's rc==0-without-a-PASS-marker arm — the probe exits 1
        first, and that arm becomes untested. So stub the probe out entirely
        with an interpreter that prints nothing and exits 0, which is exactly
        the observable behaviour the SystemExit fail-open produced. The gate
        must refuse it on its own, because the whole point of belt and braces
        is that neither lock depends on the other being right.
        """
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        mute = tmp_path / 'mute-probe-interpreter'
        mute.write_text('#!/bin/sh\nexit 0\n')
        mute.chmod(0o755)
        proc = _run_gate(
            repo / 'scripts' / 'check_write_triage_flip_preconditions.sh',
            ref=_FIXTURE_REF,
            probe_py=str(mute),
        )
        assert proc.returncode != 0, (
            f'the gate trusted rc==0 from a mute probe:\n{proc.stdout}'
        )
        assert 'PASS  item 1' not in proc.stdout, proc.stdout
        assert 'exited 0 without reporting a PASS' in proc.stdout, proc.stdout

    def test_the_gates_pass_marker_matches_the_probes_own_wording(self, tmp_path):
        """The two must not drift apart silently.

        The gate greps the probe's stdout for a literal PASS marker. If the
        probe reworded its PASS line, every real PASS would start being
        reported as 'exited 0 without reporting a PASS' — the gate would fail
        CLOSED forever, re-blocking task 3169 against a correct fix. Pin the
        shared string from both ends.
        """
        marker = None
        for line in _GATE_SCRIPT.read_text().splitlines():
            if line.startswith('PROBE_PASS_MARKER='):
                marker = line.split('=', 1)[1].strip().strip("'\"")
                break
        assert marker, 'the gate no longer defines PROBE_PASS_MARKER'

        src_root = _write_fake_judge(tmp_path / 'src', variant='by_id')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, proc.stdout
        assert marker in proc.stdout, (
            f'the gate greps for {marker!r}, which the probe no longer '
            f'emits on a PASS:\n{proc.stdout}'
        )


class TestDiagnosticsAndProvenance:
    """Two properties that do not change any verdict but decide whether the
    verdict can be TRUSTED and ACTED ON: the report has to name the real
    reason, and the module measured has to be the one the ref shipped.

    Both were reported as suggestions on review cycle 4.
    """

    def test_bare_str_gets_its_own_diagnostic(self, tmp_path):
        """main's actual shape must be reported as such, not as 'inconclusive'.

        A bare-str ``parse_judge_verdict`` sets the bare-str flag AND fails
        ``_binds_candidate``, so it also lands in the inconclusive bucket. If
        the inconclusive bucket is consulted first, the specific, actionable
        message ('the verdict names no candidate') is dead code and the single
        most common real-world case gets the least legible diagnostic --
        against a gate whose whole job is to tell an operator what to fix.
        """
        src_root = _write_fake_judge(tmp_path / 'src', variant='flat')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, proc.stdout
        assert _OPTION_A_HELD not in proc.stdout, proc.stdout
        assert _BARE_STR in proc.stdout, (
            'the bare-str case must name itself, not hide behind the generic '
            f'inconclusive bucket:\n{proc.stdout}'
        )

    def test_sibling_directory_extending_the_root_is_not_inside_it(self, tmp_path):
        """``--src-root <r>`` must measure a module UNDER <r>, not one whose
        path merely starts with <r>'s characters.

        A raw string-prefix guard accepts ``<r>-installed/...`` for
        ``--src-root <r>``, so the probe can report on a module the ref never
        shipped -- precisely the substitution the guard exists to catch. Here
        the ref's own tree carries NO judge at all while a path-extending
        sibling carries a passing one, so a prefix-matching guard exits 0 on
        evidence from the wrong tree.
        """
        root = tmp_path / 'src'
        root.mkdir()
        sibling = tmp_path / 'src-installed'
        _write_fake_judge(sibling, variant='by_id')
        proc = _run_probe(root, env={'PYTHONPATH': str(sibling)})
        assert proc.returncode != 0, (
            'the probe measured a module outside the ref it was pointed at:\n'
            f'{proc.stdout}'
        )
        assert _UNVERIFIABLE in proc.stdout, proc.stdout
        assert _OUTSIDE_SRC_ROOT in proc.stdout, proc.stdout


class TestOptionAAcceptance:
    """The gate is MECHANISM-AGNOSTIC: it asserts the invariant, not which fix
    landed.

    Task 4810's root defect is that item 1 hard-required option (a) while its
    only dependency (4762) implements option (b). Pinning option (b) instead
    would reproduce that defect mirrored — task 4798 item 7 still carries
    option (a) as the better long-term design, and an option-(b)-shaped gate
    would fail it and re-block 3169 all over again. So EITHER remedy closes
    the gate, and a later option-(a) landing needs no further gate edit.
    """

    def test_verdict_carrying_its_candidate_passes_with_an_unmarked_prompt(self, tmp_path):
        """Position is irrelevant once the answer names its own candidate."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='option_a')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed option (a):\n{proc.stdout}\n{proc.stderr}'
        assert _OPTION_A_HELD in proc.stdout, proc.stdout

    def test_stricter_parser_demanding_a_candidate_id_also_passes(self, tmp_path):
        """A parser that REJECTS a verdict carrying no candidate id — and one
        naming a candidate outside the slate — is a stronger option (a), not a
        failure. Raising on the probe's plainest payload must not be misread."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='option_a_rejects_dangling')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe failed a strict option (a):\n{proc.stdout}\n{proc.stderr}'
        assert _OPTION_A_HELD in proc.stdout, proc.stdout

    def test_slate_validating_parser_passes_option_a(self, tmp_path):
        """The gate must not fail the very remedy its own text prescribed.

        The old failure text said "validate it against the slate in
        parse_judge_verdict", which is the signature ``parse_judge_verdict(raw,
        candidates)`` — the slate REQUIRED, not defaulted. Calling it with one
        argument raises ``TypeError`` before the parser runs, so every probe
        payload was rejected, option (a) reported 'accepted no probe payload',
        and control fell through to an option (b) this (correctly untouched)
        flat prompt cannot satisfy. Measured before the fix: exit 1 with
        'the attach target is INDETERMINATE'. A complete option-(a) fix landing
        on main would have left the gate failing closed forever and task 3169
        permanently re-blocked."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='option_a_validating')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            f'probe failed a slate-validating option (a):\n{proc.stdout}\n{proc.stderr}'
        )
        assert _OPTION_A_HELD in proc.stdout, proc.stdout

    def test_unrelated_canonical_header_does_not_satisfy_option_b(self, tmp_path):
        """A target-ISH name is not a target. ``canonical_id`` interpolated into
        a header is an ordinary prompt-legibility change: it binds no verdict to
        any candidate, and ``parse_judge_verdict`` still returns a bare str.
        Reviewer-reported false PASS — ``canonical`` was in the forgiven-name
        set, so the echo control waved it through and the probe exited 0,
        authorising the production ``write_triage.enabled`` flip on a change
        that establishes nothing."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='unrelated_canonical_header')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, (
            f'an unrelated canonical header passed the gate:\n{proc.stdout}'
        )
        assert _INDETERMINATE in proc.stdout, proc.stdout

    def test_idless_non_scalar_return_does_not_satisfy_option_a(self, tmp_path):
        """A non-scalar return is a SHAPE that COULD carry a candidate id, never
        evidence that it does. Reviewer-reported false pass: a parser returning
        a plain ``{'outcome': ...}`` dict — no candidate anywhere in the
        contract — made the probe report option (a) satisfied and exit 0, in a
        gate whose exit 0 authorises a production write_triage.enabled flip."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='idless_non_scalar')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'id-less dict passed option (a):\n{proc.stdout}'
        assert _OPTION_A_HELD not in proc.stdout, proc.stdout
        assert _INDETERMINATE in proc.stdout, proc.stdout

    def test_idless_pair_does_not_satisfy_option_a(self, tmp_path):
        """The arity of an (outcome, candidate_id) pair without the id."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='idless_pair')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'(outcome, None) passed option (a):\n{proc.stdout}'
        assert _OPTION_A_HELD not in proc.stdout, proc.stdout

    def test_inline_marker_is_a_valid_option_b(self, tmp_path):
        """Marking the target inline on its own line is a correct option (b).

        The invariant is that the rendering DEPENDS on which candidate is
        named — not that the named candidate is RELOCATED. Rejecting the inline
        shape would assert a mechanism, the very defect task 4810 removes, and
        would re-block task 3169 against a valid fix."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='inline_marker')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, f'probe rejected inline option (b):\n{proc.stdout}\n{proc.stderr}'

    def test_difference_unrelated_to_the_candidates_still_fails(self, tmp_path):
        """Byte-inequality alone is not attribution. A prompt that inserts an
        unrelated line each call renders differently AND shifts every
        candidate's line index, yet names no attach target."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='spurious_inserted_lines')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'unrelated variance passed:\n{proc.stdout}'
        assert _INDETERMINATE in proc.stdout, proc.stdout

    def test_echoed_payload_does_not_satisfy_option_a(self, tmp_path):
        """Recoverability from an ECHO of the probe's own input is not a binding.

        Reviewer-reported false pass: a parser that carries the submitted
        payload alongside its outcome — a plainly plausible diagnostics
        refactor that discards the candidate entirely — round-tripped the id
        the probe handed in, so the probe reported option (a) satisfied and
        exited 0. Exit 0 here authorises the production write_triage.enabled
        flip, so this is the same class as the ``{'outcome': ...}`` false pass
        the id-less guard already covers; the fix was incomplete."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='echoes_payload')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'an echoing parser passed option (a):\n{proc.stdout}'
        assert _OPTION_A_HELD not in proc.stdout, proc.stdout
        assert _INDETERMINATE in proc.stdout, proc.stdout

    def test_echoed_raw_request_text_does_not_satisfy_option_a(self, tmp_path):
        """The id being a SUBSTRING of a returned blob is not the id being read."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='echoes_raw_string')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'a raw-text echo passed option (a):\n{proc.stdout}'
        assert _OPTION_A_HELD not in proc.stdout, proc.stdout
        assert _INDETERMINATE in proc.stdout, proc.stdout

    def test_object_identity_target_is_a_valid_option_b(self, tmp_path):
        """Naming the target by OBJECT rather than by id is a correct fix.

        Reviewer-reported false FAIL: the probe chose a spelling on 'did not
        raise', and the id spelling never raises for an identity-style
        implementation, so the object spelling was unreachable for exactly the
        implementations that need it. Failing this would re-block task 3169
        against a valid fix — the very defect task 4810 removes."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='object_identity_target')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            f'probe rejected an object-identity option (b):\n{proc.stdout}\n{proc.stderr}'
        )
        assert _SWAP_HELD in proc.stdout, proc.stdout

    def test_target_parameter_need_not_be_the_third(self, tmp_path):
        """Parameter POSITION is a mechanism; the invariant does not depend on it.

        Reviewer-reported false FAIL: the probe fed the id into whatever sat
        third, so ``(content, candidates, *, verdict_words=None,
        attach_target_id=None)`` never had its target set and was failed with
        a diagnostic that was factually wrong about why."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='target_in_fourth_position')
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            f'probe pinned the target parameter to position 3:\n{proc.stdout}\n{proc.stderr}'
        )
        assert _SWAP_HELD in proc.stdout, proc.stdout

    def test_todays_main_shape_still_fails(self, tmp_path):
        """Regression guard: the option-(a) branch must not accidentally pass
        a bare-str verdict paired with an unmarked prompt, i.e. main today."""
        src_root = _write_fake_judge(tmp_path / 'src', variant='flat')
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'option (a) branch passed main:\n{proc.stdout}'
        assert _INDETERMINATE in proc.stdout, proc.stdout


class TestRequiredInterveningParameters:
    """Reaching the target parameter must not depend on every OTHER parameter
    being optional.

    Reviewer-reported false FAIL (cycle 5). ``_target_parameters``' own
    docstring says the target's POSITION is a mechanism the invariant does not
    depend on, but the call site only delivered that for defaulted or
    keyword-only intervening parameters: it passed the first two arguments and
    the target, and nothing else. A correct option (b) spelled
    ``build_judge_prompt(content, candidates, verdict_words,
    attach_target_id=None)`` therefore raised ``TypeError`` on every attempt,
    fell through to the non-target parameter, and was reported INDETERMINATE.
    """

    def test_required_positional_intervening_parameter_still_passes(self, tmp_path):
        src_root = _write_fake_judge(
            tmp_path / 'src', variant='required_intervening_target',
        )
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            f'probe failed a valid fix whose target sits behind a REQUIRED '
            f'parameter:\n{proc.stdout}\n{proc.stderr}'
        )
        assert _SWAP_HELD in proc.stdout, proc.stdout
        assert "'attach_target_id'" in proc.stdout, proc.stdout

    def test_required_keyword_only_parameter_forces_the_placeholder_ladder(self, tmp_path):
        """The first placeholder is rejected by the fixture, so a one-value
        ladder is not enough."""
        src_root = _write_fake_judge(
            tmp_path / 'src', variant='required_keyword_only_target',
        )
        proc = _run_probe(src_root)
        assert proc.returncode == 0, (
            f'probe failed a valid fix whose required intervening parameter is '
            f'iterated:\n{proc.stdout}\n{proc.stderr}'
        )
        assert _SWAP_HELD in proc.stdout, proc.stdout

    def test_unreachable_target_parameter_is_unverifiable_not_indeterminate(self, tmp_path):
        """Fails closed either way — but says what it actually measured.

        A target-NAMED parameter that never rendered decided nothing. Asserting
        INDETERMINATE there states a fact the run did not establish, and points
        the reader at the wrong remedy.
        """
        src_root = _write_fake_judge(
            tmp_path / 'src', variant='target_parameter_always_raises',
        )
        proc = _run_probe(src_root)
        assert proc.returncode != 0, f'probe passed an unrenderable target:\n{proc.stdout}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout
        assert _INDETERMINATE not in proc.stdout, proc.stdout
        assert "'attach_target_id'" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------
# The gate script, end to end against hermetic fixture repos
# ---------------------------------------------------------------------------

# Committed identity so a fixture commit does not depend on a global git
# identity being configured in the test environment (the idiom from
# fused-memory/tests/test_predicate_contradiction.py).
_GIT_ENV_ARGS = [
    '-c',
    'user.email=write-triage-gate-test@example.com',
    '-c',
    'user.name=write-triage-gate-test',
    '-c',
    'commit.gpgsign=false',
]

#: The fixture repo's branch, pinned so the test does not depend on whatever
#: `init.defaultBranch` the host git is configured with.
_FIXTURE_REF = 'fixture-main'

# Item 2 greps for the frozenset being iterated directly; item 4 for the
# self-overwriting markdown sibling. These two fixtures straddle both.
_EVAL_FAILING = '''\
"""Fixture stand-in for fused-memory/scripts/eval_write_triage_judge.py."""
CONFUSION_COLUMNS = list(TRIAGE_OUTCOMES)


def publish(report_path):
    return report_path.with_suffix('.md')
'''

_EVAL_FIXED = '''\
"""Fixture stand-in with both cycle-2 findings fixed."""
EVAL_OUTCOMES = tuple(sorted(TRIAGE_OUTCOMES))
CONFUSION_COLUMNS = EVAL_OUTCOMES


def publish(report_path):
    sibling = report_path.parent / (report_path.stem + '.md')
    assert sibling != report_path
    return sibling
'''

_CONFIG_YAML = 'write_triage:\n  enabled: false\n  judge_enabled: true\n'


def _make_gate_repo(tmp_path: Path, *, judge: str = 'flat', eval_src: str = 'failing') -> Path:
    """A throwaway git repo the gate script can be run from.

    The script derives ``REPO`` from ``BASH_SOURCE/..`` with no env override,
    so BOTH scripts are copied into ``<repo>/scripts/`` — that is what makes
    the fixture repo, rather than the real checkout, the thing item 1's
    ``git archive`` reads.
    """
    repo = tmp_path / 'gate-repo'
    (repo / 'scripts').mkdir(parents=True)
    for script in (_GATE_SCRIPT, _PROBE):
        dest = repo / 'scripts' / script.name
        dest.write_bytes(script.read_bytes())
        dest.chmod(0o755)

    _write_fake_judge(repo / 'fused-memory' / 'src', variant=judge)
    (repo / 'fused-memory' / 'scripts').mkdir(parents=True)
    (repo / 'fused-memory' / 'scripts' / 'eval_write_triage_judge.py').write_text(
        _EVAL_FIXED if eval_src == 'fixed' else _EVAL_FAILING,
    )
    (repo / 'fused-memory' / 'config').mkdir(parents=True)
    (repo / 'fused-memory' / 'config' / 'config.yaml').write_text(_CONFIG_YAML)

    subprocess.run(
        ['git', 'init', '-q', '-b', _FIXTURE_REF],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(['git', 'add', '-A', '-f'], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ['git', *_GIT_ENV_ARGS, 'commit', '-q', '-m', 'fixture'],
        cwd=repo, check=True, capture_output=True,
    )
    return repo


def _run_gate(script: Path, *, ref: str, probe_py: str | None = None):
    """Execute the gate script DIRECTLY, as DeterministicRunner does."""
    full_env = dict(os.environ)
    full_env['WRITE_TRIAGE_GATE_REF'] = ref
    full_env['CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY'] = probe_py or sys.executable
    return subprocess.run(
        [str(script)],
        capture_output=True,
        text=True,
        timeout=240,
        env=full_env,
    )


def _assert_no_bare_grep_verdict(stdout: str) -> None:
    """Item 1 must no longer report a source-text grep as its verdict.

    A grep asserts which MECHANISM landed, not whether the invariant holds: it
    fails a correct option-(b) fix and passes prose that changes no behaviour.
    """
    assert 'candidate_id ABSENT' not in stdout, stdout
    assert 'candidate_id present' not in stdout, stdout


class TestFlipPreconditionsScript:
    """The gate script end to end. Item 1 delegates to the probe; items 2 and
    4 keep their existing eval-source patterns, unchanged."""

    def test_flat_judge_fails_item_1(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='flat')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert 'FAIL  item 1' in proc.stdout, proc.stdout
        _assert_no_bare_grep_verdict(proc.stdout)

    def test_by_id_judge_with_fixed_eval_passes_everything(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert 'PASS  item 1' in proc.stdout, proc.stdout
        _assert_no_bare_grep_verdict(proc.stdout)

    def test_marking_candidates_zero_does_not_satisfy_the_gate(self, tmp_path):
        """The gate must not bless candidates[0]-as-attach-target."""
        repo = _make_gate_repo(tmp_path, judge='positional_target', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert 'FAIL  item 1' in proc.stdout, proc.stdout
        assert _RESCUE_PATH in proc.stdout, proc.stdout

    def test_unreadable_ref_fails_closed(self):
        """NEGATIVE CONTROL, mandated by the script's own comment.

        A `$(...)` command substitution runs in a SUBSHELL, so a `fail=1`
        assigned inside one is discarded — the bug caught 2026-08-27, where an
        unreadable ref skipped its whole check block and the gate exited 0 on
        unverifiable input. Run against the REAL checkout, read-only.
        """
        proc = _run_gate(_GATE_SCRIPT, ref='no-such-ref')
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_unresolvable_probe_interpreter_fails_closed(self, tmp_path):
        """A judge that WOULD pass, but no interpreter to prove it with.

        An unverifiable invariant is not a satisfied one — this must never be
        the difference between exit 1 and exit 0.
        """
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        proc = _run_gate(
            repo / 'scripts' / _GATE_SCRIPT.name,
            ref=_FIXTURE_REF,
            probe_py=str(tmp_path / 'no-such-python3'),
        )
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_option_a_judge_passes_item_1_through_the_script(self, tmp_path):
        """Mechanism-agnostic END TO END, not just in the probe's own tests.

        Every other PASS-item-1 case here is an option (b). If the shell ever
        grew a step that presumed a marked PROMPT, the probe suite would not
        notice: it never runs the shell.
        """
        repo = _make_gate_repo(tmp_path, judge='option_a', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        assert 'PASS  item 1' in proc.stdout, proc.stdout
        assert _OPTION_A_HELD in proc.stdout, proc.stdout
        _assert_no_bare_grep_verdict(proc.stdout)

    def test_missing_probe_fails_closed(self, tmp_path):
        """A distinct record_fail site: the probe is simply not there.

        It is the one branch that cannot report anything from the probe, so a
        regression that dropped its record_fail would silently PASS item 1 on
        no evidence whatsoever.
        """
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        (repo / 'scripts' / _PROBE.name).unlink()
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, (
            f'the gate passed with no probe on disk:\n{proc.stdout}\n{proc.stderr}'
        )
        assert 'FAIL  item 1' in proc.stdout, proc.stdout
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_probe_timeout_fails_closed(self, tmp_path):
        """exit 124, the branch the bounded `timeout` exists to produce.

        Stands in for the probe hanging: a judge whose import never returns.
        The interpreter test covers exit 127 (cannot run at all); this covers
        ran-but-never-finished, which reaches a different record_fail site.
        """
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        # Shrink the gate's own `timeout 90` so the test does not take 90s.
        gate = repo / 'scripts' / _GATE_SCRIPT.name
        gate.write_text(gate.read_text().replace("timeout 90", "timeout 2"))
        judge = (
            repo / 'fused-memory' / 'src' / 'fused_memory' / 'server'
            / 'write_triage_judge.py'
        )
        judge.write_text('import time\ntime.sleep(600)\n')
        subprocess.run(['git', 'add', '-A', '-f'], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ['git', *_GIT_ENV_ARGS, 'commit', '-q', '-m', 'hang'],
            cwd=repo, check=True, capture_output=True,
        )
        proc = _run_gate(gate, ref=_FIXTURE_REF)
        assert proc.returncode == 1, (
            f'the gate passed on a probe that never finished:\n{proc.stdout}'
        )
        assert 'FAIL  item 1' in proc.stdout, proc.stdout
        assert _UNVERIFIABLE in proc.stdout, proc.stdout

    def test_items_2_and_4_still_fail_on_their_patterns(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='failing')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        assert 'FAIL  item 2' in proc.stdout, proc.stdout
        assert 'FAIL  item 4' in proc.stdout, proc.stdout

    def test_items_2_and_4_still_pass_on_the_fixed_patterns(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='flat', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert 'PASS  item 2' in proc.stdout, proc.stdout
        assert 'PASS  item 4' in proc.stdout, proc.stdout


#: DeterministicRunner._default_run_script folds stderr into stdout and returns
#: ``decode()[-2000:]``, and _run_predicate feeds exactly that into the
#: milestone_check_failed escalation's detail. The all-FAIL report is several
#: times that, and item 1 is emitted FIRST — so item 1's rewritten guidance,
#: the corrected spec an implementer is meant to read, is precisely what gets
#: truncated away from what an operator actually sees.
_ESCALATION_DETAIL_CHARS = 2000


class TestReportSurvivesTruncation:
    """What reaches the operator is the report's TAIL, not its head."""

    def test_all_fail_tail_still_names_every_failing_item(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='flat', eval_src='failing')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 1, f'{proc.stdout}\n{proc.stderr}'
        # Vacuous otherwise: a report that fits inside the window proves nothing.
        assert len(proc.stdout) > _ESCALATION_DETAIL_CHARS, len(proc.stdout)
        tail = proc.stdout[-_ESCALATION_DETAIL_CHARS:]
        # Item NUMBERS, not guidance prose — the summary is deliberately a
        # summary, so this does not become a wording pin.
        assert 'FAILING ITEMS: 1 2 4' in tail, tail

    def test_all_pass_tail_reports_no_failing_items(self, tmp_path):
        repo = _make_gate_repo(tmp_path, judge='by_id', eval_src='fixed')
        proc = _run_gate(repo / 'scripts' / _GATE_SCRIPT.name, ref=_FIXTURE_REF)
        assert proc.returncode == 0, f'{proc.stdout}\n{proc.stderr}'
        tail = proc.stdout[-_ESCALATION_DETAIL_CHARS:]
        assert 'FAILING ITEMS: none' in tail, tail
