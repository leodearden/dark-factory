"""Prompt/code drift guard for the recon stage prompts (task 2231, W5-xi).

PRD plans/recon-reliability-prd.md Sec 8.4 states the drift invariant this
module pins: each stage prompt is an f-string that interpolates
`{render_X_section()}` from `recon_self_model` (task 2220, W5-beta) at import
time, so the prose describing recon's control-plane mechanics and the code
that actually implements them cannot silently diverge. A hand-transcribed
paraphrase substituted back in for the rendered call would break the
containment assertion below rather than drift unnoticed; editing
recon_self_model's render_*() bodies changes what they render, and the
prompt recomputes to match automatically on next import.

Step 5 (RED) covers only the stage1 half; step 6 (GREEN) cuts stage1.py
over to interpolate these rendered sections. Steps 7/8 extend this module
with the analogous stage2 half.

Scope note: these are pure wiring guards — each asserts only that the
prompt f-string *interpolates* the render_*() call rather than
hand-transcribing a copy of its prose. They deliberately do NOT pin any
specific wording from inside a render_*() body; the behavioral content of
each rendered section (its load-bearing invariant tokens) is cross-checked
against live code in test_recon_self_model.py, where a content deletion
inside a render body is caught without locking cosmetic prompt wording in
place here.
"""

from __future__ import annotations

from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.recon_self_model import (
    render_cycle_summary_section,
    render_execution_class_section,
    render_marker_lifecycle_section,
    render_suppression_schema_section,
)

# ---------------------------------------------------------------------------
# Stage 1: suppression schema + marker lifecycle
# ---------------------------------------------------------------------------


class TestStage1PromptCutover:
    """STAGE1_SYSTEM_PROMPT must interpolate recon_self_model's rendered
    sections verbatim, not a hand-transcribed paraphrase of them."""

    def test_stage1_prompt_contains_rendered_suppression_schema_section(self):
        """render_suppression_schema_section()'s output must appear verbatim
        in STAGE1_SYSTEM_PROMPT. This containment relation IS the drift
        guard: it holds only if the prompt interpolates the render call
        rather than hand-transcribing its own copy of the suppression
        schema, so an edit to recon_self_model that isn't matched by a
        prompt re-render (impossible once interpolated — recomputed at
        import) or that reverts to a hand copy fails this assertion."""
        assert render_suppression_schema_section() in STAGE1_SYSTEM_PROMPT, (
            'STAGE1_SYSTEM_PROMPT must interpolate '
            'render_suppression_schema_section() verbatim (task 2231 '
            'W5-xi) — a hand-transcribed paraphrase of the suppression '
            'schema is prompt/code drift by definition.'
        )

    def test_stage1_prompt_contains_rendered_marker_lifecycle_section(self):
        """render_marker_lifecycle_section()'s output must appear verbatim
        in STAGE1_SYSTEM_PROMPT — the same drift guard as above, for the
        marker-kinds / run_id-fresh-per-cycle mechanics (Flag Deduplication
        + Stage 2 Flag Relay)."""
        assert render_marker_lifecycle_section() in STAGE1_SYSTEM_PROMPT, (
            'STAGE1_SYSTEM_PROMPT must interpolate '
            'render_marker_lifecycle_section() verbatim (task 2231 W5-xi) '
            '— a hand-transcribed paraphrase of the marker lifecycle / '
            'run_id-fresh-per-cycle mechanics is prompt/code drift by '
            'definition.'
        )


# ---------------------------------------------------------------------------
# Stage 2: cycle-summary convention + execution-class contract
# ---------------------------------------------------------------------------


class TestStage2PromptCutover:
    """STAGE2_SYSTEM_PROMPT must interpolate recon_self_model's rendered
    sections verbatim, not a hand-transcribed paraphrase of them."""

    def test_stage2_prompt_contains_rendered_cycle_summary_section(self):
        """render_cycle_summary_section()'s output must appear verbatim in
        STAGE2_SYSTEM_PROMPT. The prompt's own "Re-Verify Reconstruction
        Writes" section already references "the canonical cycle_summary
        metadata convention" by name — this containment relation is what
        makes that reference resolve to real, rendered text instead of a
        dangling pointer to a hand-transcribed (and driftable) copy
        elsewhere in the prompt."""
        assert render_cycle_summary_section() in STAGE2_SYSTEM_PROMPT, (
            'STAGE2_SYSTEM_PROMPT must interpolate '
            'render_cycle_summary_section() verbatim (task 2231 W5-xi) — '
            'a hand-transcribed paraphrase of the cycle_summary metadata '
            'convention is prompt/code drift by definition.'
        )

    def test_stage2_prompt_contains_rendered_execution_class_section(self):
        """render_execution_class_section()'s output must appear verbatim
        in STAGE2_SYSTEM_PROMPT. execution_class has no prior prompt
        precedent (task 2225/W5-eta added only the middleware-side
        declare-or-reject enforcement); this is net-new prompt content
        threaded into the "## Creating Tasks" submit_task guidance so
        recon agents are told what they must declare, rather than
        discovering the requirement only via a rejected submit_task call."""
        assert render_execution_class_section() in STAGE2_SYSTEM_PROMPT, (
            'STAGE2_SYSTEM_PROMPT must interpolate '
            'render_execution_class_section() verbatim (task 2231 W5-xi) '
            '— the execution_class declaration/rejection contract must be '
            "stated in the prompt, not only enforced silently at the "
            'submit_task middleware boundary.'
        )
