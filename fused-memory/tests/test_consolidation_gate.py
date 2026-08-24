"""Tests for the consolidation-gate brief + closure predicate (task 3112).

Covers the two defects the task owns: the gate-filing instruction that
prescribed no end-state shape (Defect 1 — :func:`render_end_state_brief` /
:func:`render_consolidation_gate_section`), and the absence of any mechanical
refusal to close a gate task over a malformed cluster (Defect 2 —
:func:`evaluate_closure`).

Assertions are pinned to runtime return values (the verdict dataclass, the
builder's returned dict) and to stable load-bearing substrings within the
rendered prose — NOT verbatim prompt-text equality — mirroring the
``test_recon_self_model.py`` / ``test_predicate_contradiction.py`` convention.
"""

from __future__ import annotations
