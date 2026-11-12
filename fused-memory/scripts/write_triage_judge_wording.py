"""The judge system-prompt wordings an eval may run under.

plans/write-triage-flip-readiness-prd.md §11 D16 makes the judge's wording an
eval axis, {shipped, pre-ψ}, measured on the production call (task 6151).
:func:`judge_wording` swaps ``write_triage_judge.JUDGE_SYSTEM_PROMPT``, the
module global every provider arm of the shipped ``_call_llm`` reads at call
time, so shipped code is untouched and the request path stays the real one.

The override is PROCESS-GLOBAL: a caller runs one wording at a time, and the
context manager refuses to nest rather than let two wordings interleave.

:data:`PROMPTS_WHILE_PSI_SHIPPED` freezes what each wording sent while ψ was
shipped, the era the committed π and π2 populations measured.
"""
from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator, Mapping
from types import MappingProxyType

from fused_memory.server import write_triage_judge
from fused_memory.server.write_triage_judge import (
    JUDGE_EXEMPLARS,
    JUDGE_REPLY_SHAPE,
    render_judge_exemplars,
)

WORDING_SHIPPED = 'shipped'
WORDING_PRE_PSI = 'pre-psi'
WORDINGS = (WORDING_SHIPPED, WORDING_PRE_PSI)

#: The shipped prompt as imported, before any override can be in force.
_SHIPPED = write_triage_judge.JUDGE_SYSTEM_PROMPT

#: ``JUDGE_SYSTEM_PROMPT`` as of 16794cdd1d^1, the commit before ψ. Only the
#: definition text is spelled here; the exemplars and the reply shape are the
#: public renders ψ left byte-identical, so nothing else is re-spelled.
PRE_PSI_JUDGE_SYSTEM_PROMPT = f"""\
You classify the RELATIONSHIP between a new memory entry and a small set of \
existing entries retrieved as its closest matches. You do not decide which \
entry is correct, and you do not merge, rewrite or rank them.

Answer with exactly one of these four words:

- "distinct" — no candidate makes the same core claim as the new entry. \
Shared wording alone neither makes a match nor rules one out.
- "restates" — the new entry asserts what a candidate already asserts, adding \
nothing new. A paraphrase restates.
- "amends" — the new entry asserts what a candidate asserts AND adds \
something the candidate does not have: a detail, a scope, a later \
observation, a correction of degree.
- "contests" — the new entry asserts something that CANNOT be true at the \
same time as a candidate. Use this only for a genuine incompatibility, not \
for a difference in emphasis, scope, or point in time — two entries \
describing different situations, or the same situation at different times, \
are not in conflict. You are DETECTING a contradiction so a human or a \
downstream gate can adjudicate it; you are NOT deciding which side is true, \
and nothing you say here deletes or edits anything.

Find the candidate whose core claim the new entry shares (or, for \
"contests", contradicts), answer about THAT candidate, and name it by its \
id: the verdict is filed against the candidate you name and no other. \
Answer "distinct", naming none, only when no candidate qualifies. Between \
"amends" and "contests", prefer "amends" — a genuine incompatibility is a \
last resort, not a default reading.

Worked examples:

{render_judge_exemplars(JUDGE_EXEMPLARS)}

Reply with a bare JSON object and nothing else:

{JUDGE_REPLY_SHAPE}\
"""

#: ``JUDGE_SYSTEM_PROMPT`` as of 16794cdd1d (ψ, ruling C1''), the prompt the
#: committed π and π2 populations' shipped arms measured. As for pre-ψ, only
#: the definition text is spelled here.
PSI_JUDGE_SYSTEM_PROMPT = f"""\
You classify the RELATIONSHIP between a new memory entry and a small set of \
existing entries retrieved as its closest matches. You do not decide which \
entry is correct, and you do not merge, rewrite or rank them.

Answer with exactly one of these four words:

- "distinct" — no candidate makes the same core claim as the new entry. \
Shared wording alone neither makes a match nor rules one out.
- "restates" — the new entry asserts what a candidate already asserts, adding \
nothing new. A paraphrase restates.
- "amends" — the new entry asserts what a candidate asserts AND adds \
something the candidate does not have: a detail, a scope, a later \
observation — without saying the candidate is wrong.
- "contests" — the new entry addresses the same claim or subject as a \
candidate and says it is wrong, outdated or different, explicitly or \
implicitly — not when it merely restates it, and not when it agrees and \
only adds to it. You are DETECTING a contradiction so a human or a \
downstream gate can adjudicate it; you are NOT deciding which side is true, \
and nothing you say here deletes or edits anything.

Find the candidate whose core claim the new entry shares (or, for \
"contests", contradicts), answer about THAT candidate, and name it by its \
id: the verdict is filed against the candidate you name and no other. \
Answer "distinct", naming none, only when no candidate qualifies.

Worked examples:

{render_judge_exemplars(JUDGE_EXEMPLARS)}

Reply with a bare JSON object and nothing else:

{JUDGE_REPLY_SHAPE}\
"""

#: The prompt each wording sent while ψ was the shipped one: what
#: ``fused-memory/calibration/write_triage_population.json`` and
#: ``write_triage_population_write_time.json`` measured. A committed artifact
#: is checked against the prompts of its own era, never the live one.
PROMPTS_WHILE_PSI_SHIPPED: Mapping[str, str] = MappingProxyType({
    WORDING_SHIPPED: PSI_JUDGE_SYSTEM_PROMPT,
    WORDING_PRE_PSI: PRE_PSI_JUDGE_SYSTEM_PROMPT,
})

_PROMPTS = {
    WORDING_SHIPPED: _SHIPPED,
    WORDING_PRE_PSI: PRE_PSI_JUDGE_SYSTEM_PROMPT,
}

#: The wording :func:`judge_wording` currently holds in force, if any.
_in_force: str | None = None


def system_prompt(wording: str) -> str:
    """The judge system prompt *wording* names."""
    try:
        return _PROMPTS[wording]
    except KeyError:
        raise ValueError(
            f'unknown judge wording {wording!r}; known wordings are {list(WORDINGS)}',
        ) from None


@contextlib.contextmanager
def judge_wording(wording: str) -> Iterator[str]:
    """Hold *wording* in force on the shipped judge; yield its prompt's sha256.

    Refuses (``RuntimeError``) while any override is already active, whether
    this one or another's, and always restores the shipped prompt on exit.
    """
    global _in_force
    prompt = system_prompt(wording)
    if _in_force is not None or write_triage_judge.JUDGE_SYSTEM_PROMPT is not _SHIPPED:
        active = _in_force or 'an unrecognised override'
        raise RuntimeError(
            f'judge wording {active!r} is already in force; refusing to enter '
            f'{wording!r}, since two wordings must never interleave in one process',
        )
    _in_force = wording
    write_triage_judge.JUDGE_SYSTEM_PROMPT = prompt
    try:
        yield hashlib.sha256(prompt.encode()).hexdigest()
    finally:
        write_triage_judge.JUDGE_SYSTEM_PROMPT = _SHIPPED
        _in_force = None
