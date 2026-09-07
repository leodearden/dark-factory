"""Task-4216 drift-guard: pin the Stage-2 `memory_hints` write guidance.

The Stage-2 system prompt is not documentation — it is the runtime input that
drives the Stage-2 LLM's writes, so its text IS the program's behaviour. This
file follows the established convention of
``test_standing_decision_prompt_drift.py`` / ``test_recon_report_guidance_drift.py``:
import the assembled prompt constant and assert on load-bearing claims.

WHY A GUARD. The prompt used to carry a falsified premise — that Stage 2's
additive attach "silently discards" legacy list-format hints under old-wins
semantics — and prescribed a `get_task` -> local convert ->
``metadata_mode='replace'`` round trip as the remedy. The premise is false: the
additive branch runs ``apply_migrations`` over ``memory_hints`` on BOTH sides
before merging (``sqlite_task_backend.py::_merge_metadata``), so a plain
``append=True`` attach converts and unions a legacy row by itself. The
prescribed remedy was also the one mode that BYPASSES the corrupt-blob guard,
i.e. the only mode able to destroy a corrupt-but-recoverable metadata row. That
falsified claim had propagated into three independent copies and steered real
Stage-2 writes, so it is pinned absent here.

Three statements must SURVIVE every future rewrite of this region, and are
pinned POSITIVELY (a negative-only guard cannot catch a collateral revert):

  (i)   a bare ``append=False`` is rejected outright (task-2180 metadata-wipe);
  (ii)  ``metadata_mode='replace'`` with a COMPLETE read-modify-write payload
        remains sanctioned for a genuinely intended whole-blob overwrite, e.g.
        repairing a corrupt row;
  (iii) ``append=True`` must NOT be combined with ``metadata_mode='merge'`` —
        the backend REJECTS that pair (task-3581) — so the additive merge is
        requested as ``append=True`` alone or explicit ``metadata_mode='additive'``.

Assertions are deliberately scoped to those load-bearing claims rather than
pinning the corrected prose byte-for-byte, so ordinary rewording does not
produce a false failure.
"""
from __future__ import annotations

from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT

# The legacy hint shape, as the prompt renders it (the f-string's doubled
# braces collapse to single braces in the assembled string).
_LEGACY_SHAPE_TOKEN = '[{entity, query}, ...]'


def _window_after(needle: str, size: int) -> str:
    """The *size*-char slice of the prompt starting at *needle*."""
    idx = STAGE2_SYSTEM_PROMPT.find(needle)
    assert idx != -1, f'anchor vanished from the Stage-2 prompt: {needle!r}'
    return STAGE2_SYSTEM_PROMPT[idx:idx + size]


def _window_around(needle: str, size: int) -> str:
    """The +/- *size*-char slice of the prompt centred on *needle*."""
    idx = STAGE2_SYSTEM_PROMPT.find(needle)
    assert idx != -1, f'anchor vanished from the Stage-2 prompt: {needle!r}'
    return STAGE2_SYSTEM_PROMPT[max(0, idx - size):idx + size]


class TestFalsifiedReshapePremiseAbsent:
    """The obsolete legacy-hint reshape guidance must stay gone."""

    def test_silently_discards_claim_absent(self):
        """The falsified old-wins-discard claim, in any copy."""
        assert 'silently discards' not in STAGE2_SYSTEM_PROMPT, (
            "the additive attach does NOT discard legacy list-format hints - "
            "_merge_metadata migrates both sides before merging"
        )

    def test_distinct_reshape_case_framing_absent(self):
        """Legacy hints are not a distinct case needing distinct handling."""
        assert 'RESHAPE case' not in STAGE2_SYSTEM_PROMPT
        assert 'The additive union above is ONLY for the ATTACH case' not in (
            STAGE2_SYSTEM_PROMPT
        ), 'the additive union covers legacy rows too; it is not ATTACH-only'

    def test_reshape_via_replace_round_trip_absent(self):
        """The obsolete get_task -> local convert -> replace directive."""
        assert 'convert and merge the reshaped hints' not in STAGE2_SYSTEM_PROMPT
        assert (
            "write the COMPLETE metadata blob back with `metadata_mode='replace'`"
            not in STAGE2_SYSTEM_PROMPT
        )

    def test_legacy_shape_is_discussed_as_an_additive_attach(self):
        """Wherever the prompt names the legacy list shape, the remedy it
        prescribes nearby is the ADDITIVE attach, not a replace round trip.

        Catches a paraphrased reintroduction that the literal-fragment
        assertions above would miss.
        """
        window = _window_after(_LEGACY_SHAPE_TOKEN, 900)
        assert '`append=True`' in window, (
            'the legacy hint shape must be answered with the additive attach; '
            f'window was: {window!r}'
        )
        assert '`metadata_mode` OMITTED' in window, (
            'the additive attach for a legacy row omits metadata_mode; '
            f'window was: {window!r}'
        )


class TestCorrectedGuidancePresent:
    """The corrected positive guidance the LLM must actually follow."""

    def test_replace_is_warned_against_for_hints(self):
        assert 'BYPASSES the corrupt-blob guard' in STAGE2_SYSTEM_PROMPT, (
            "the prompt must say why metadata_mode='replace' is dangerous for "
            'hints: it is the one mode that can destroy a corrupt-but-'
            'recoverable metadata row'
        )


class TestMustSurviveBareAppendFalseRejection:
    """(i) task-2180: a bare ``append=False`` whole-blob overwrite is rejected."""

    def test_rejection_and_incident_detail_present(self):
        assert '`append=False`' in STAGE2_SYSTEM_PROMPT
        assert 'task-2180' in STAGE2_SYSTEM_PROMPT
        window = _window_around('task-2180', 700)
        assert 'REJECTED' in window
        assert 'substrate_confirmed' in window, (
            'the metadata-wipe incident detail is what makes the rule stick'
        )


class TestMustSurviveSanctionedReplace:
    """(ii) ``metadata_mode='replace'`` stays sanctioned for a genuine
    whole-blob overwrite, given a COMPLETE read-modify-write payload."""

    def test_sanctioned_whole_blob_overwrite_present(self):
        assert "`metadata_mode='replace'`" in STAGE2_SYSTEM_PROMPT
        assert 'read-modify-write' in STAGE2_SYSTEM_PROMPT
        window = _window_around('read-modify-write', 800)
        assert "`metadata_mode='replace'`" in window
        assert 'corrupt' in window, (
            'the sanctioned use is a genuinely intended whole-blob overwrite '
            'such as repairing a corrupt row'
        )


class TestMustSurviveTask3581Contradiction:
    """(iii) task-3581: ``append=True`` + ``metadata_mode='merge'`` is a
    contradiction the backend rejects.

    Pinned POSITIVELY so that a rewrite of the adjacent hint guidance cannot
    silently revert task 3581's landed prompt fix.
    """

    def test_merge_plus_append_contradiction_present(self):
        assert 'Do NOT combine' in STAGE2_SYSTEM_PROMPT
        window = _window_around('Do NOT combine', 800)
        assert '`append=True`' in window
        assert "`metadata_mode='merge'`" in window
        assert 'TASKMASTER_TOOL_ERROR' in window, (
            'the backend REJECTS the pair; the prompt must say so'
        )

    def test_additive_requested_as_append_alone_or_explicit_additive(self):
        assert "`metadata_mode='additive'`" in STAGE2_SYSTEM_PROMPT, (
            'the sanctioned additive spellings are append=True ALONE or the '
            "explicit metadata_mode='additive'"
        )


class TestMustSurviveDetailsAppendCaveat:
    """``append`` is not scoped to metadata — the details caveat added by this
    same task must not be lost to a later rewrite of the block it lives in."""

    def test_combined_details_plus_metadata_append_warned_against(self):
        assert 'NEVER combine a `details` rewrite with a metadata append' in (
            STAGE2_SYSTEM_PROMPT
        )
        window = _window_around('NEVER combine a `details` rewrite', 1200)
        assert 'silently DUPLICATED' in window, (
            'the hazard is that the metadata half succeeds while details is '
            'duplicated, so the response reads as a clean success'
        )
