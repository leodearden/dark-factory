"""Anchor test for `PASTED_TEXT_PYTHON_LITERAL_GUIDANCE` (task 5964).

Held to the preamble-tail contract in `_role_splice_contract.py`; only this
block's own remedy prose lives here. Every assertion is an existence,
containment, count or index check against a NAMED constant — never a prose
pin, never a regex over wording, never a byte-size figure.
"""

from __future__ import annotations

from _role_splice_contract import PreambleTailBlock, PreambleTailContractTests

from orchestrator.agents.python_literal_guidance import PASTED_TEXT_PYTHON_LITERAL_GUIDANCE


class TestPastedTextPythonLiteralSplice(PreambleTailContractTests):
    block = PreambleTailBlock(
        constant_name='PASTED_TEXT_PYTHON_LITERAL_GUIDANCE',
        constant=PASTED_TEXT_PYTHON_LITERAL_GUIDANCE,
        restore_remedy=(
            'Restore the task-5964 guidance: without it every Bash-capable role '
            'pastes foreign source into a plain Python literal and either loses '
            'a turn to a unicodeescape SyntaxError or silently edits with '
            'decoded escapes.'
        ),
        brace_remedy=(
            'Describe the brace-form `\\u` escape of Rust and JavaScript in '
            'words; never spell it out.'
        ),
        excluded_roles_remedy=(
            'Remove the splice from judge or reviewer_comprehensive rather than '
            'widening the set: a `Bash(git:*)` grant cannot run python3, so the '
            'block is dead weight there.'
        ),
        carrier_note=(
            'architect and deep_reviewer belong in it although they lack `Edit`: '
            'the raw-literal rule applies to any python3 script, and the block '
            'names `Edit` only conditionally.'
        ),
    )
