"""Gate-owned finding phrasing — Stage 1 norm plus deterministic code gate (task 4814).

THE DEFECT. Stage 1 (``stages/memory_consolidator.py``) sometimes writes a
finding's ``suggested_action`` in words that read as authorizing Stage 2 to
DECIDE — the observed offenders being "Stage 2 or operator should decide",
"enumerate" and "extend" — when the finding's cited task is a HUMAN GATE
(``metadata.operational_mode == 'gate'`` and/or
``metadata.always_escalates == true``). Stage 2 holds
``update_task``/``set_task_status``, so that phrasing invites a stage to close
a question that only a human may answer.

EVIDENCE. autopilot_video run 8b2d3371, Stage 1 findings 640d4ceb (cited task
645, an ``operational_mode='gate'`` task) and 66af9601 (cited task 652). Stage
2 caught both that cycle. The recurring cost is that Stage 2 must catch it
EVERY cycle.

THE DESIGN — two halves sharing ONE canonical sentence:

- Half A, probabilistic: a Stage-1 prompt norm
  (:func:`render_gate_owned_action_norm`) telling the model how to phrase
  ``suggested_action`` when a finding's cited task is gate-owned.
- Half B, deterministic: a pure post-processor
  (:func:`normalize_gate_owned_suggested_actions`) in Stage 1's existing
  filter chain that PREPENDS the canonical sentence to any finding whose
  cited task is gate-owned. Gate-ownership is read as a structured metadata
  fact off ``filtered_task_tree.active_tasks`` — never guessed from the LLM's
  prose.

Both halves quote :data:`CANONICAL_HUMAN_GATE_ACTION`, owned here (INV-5:
one owner for a rule enforced in two places). If the prompt text and the
enforcement text could drift, an operator reading a corrected finding would
see wording the stage was never given. This follows the established
``consolidation_gate.render_consolidation_gate_section`` shape, which
``prompts/stage1.py`` already imports from a reconciliation module rather
than from ``prompts/__init__.py``.

A prompt directive alone would only lower the rate; the repo's established
shape for exactly this situation is a prompt directive PLUS a code-side
authoritative backstop (the Flag Suppression Check's prompt text says "the
code gate is the authoritative enforcement point", and
``filter_stale_count_snapshot_corrections`` backstops the "do not emit
off-by-N corrections" prompt section).

THE ONE CARVE-OUT. ``curator_gate_resolution_sweep.build_gate_resolution_flag``
deliberately emits a ``suggested_action`` telling Stage 2 to "set the task to
its resolved status" for an ``operational_mode='gate'`` task. That is NOT this
defect: its evidence is a ruling a human curator ALREADY recorded in Mem0, so
Stage 2 is transcribing a human decision rather than making one, and the flag
carries its own dismiss branch for the merely-curated case. Both halves carve
it out by ``flag_type == GATE_RESOLUTION_FLAG_TYPE`` (imported, never
re-spelled), so this module does not ship a rule that contradicts a live,
deliberately-designed sibling flag on its very first cycle.

Pure throughout: no I/O, no side effects, no task writes.
"""

from __future__ import annotations

from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    GATE_RESOLUTION_FLAG_TYPE,
)

#: The ONE sentence both halves of task 4814 quote (INV-5).
#:
#: The Stage-1 prompt norm tells the stage to write a gate-owned finding's
#: ``suggested_action`` in these terms; :func:`normalize_gate_owned_suggested_actions`
#: PREPENDS this exact text when the stage did not.  Because both halves read
#: this one constant, the correction an operator sees can never be wording the
#: stage was never given.
#:
#: CONTAINS NO BRACE, in any form, and this is load-bearing rather than
#: stylistic: the constant is interpolated into ``STAGE1_SYSTEM_PROMPT``, which
#: is an f-string — a bare brace fails at import — while
#: ``tests/test_stage1_consolidation_guidance.py`` separately asserts that
#: ``{{``/``}}`` never survive into the rendered text, so escaping is not an
#: escape hatch either.  Carrying no brace at all satisfies both.
CANONICAL_HUMAN_GATE_ACTION = (
    'HUMAN DECISION GATE: the cited task is a human decision gate '
    "(metadata.operational_mode = 'gate' and/or metadata.always_escalates = "
    'true), so it is awaiting human-operator sign-off only. No reconciliation '
    'stage may decide this question, and no stage may record a decision on the '
    'task (update_task / set_task_status) that a human has not made. '
    'Everything that follows is evidence assembled FOR the human operator, not '
    'an instruction to a stage.'
)

#: The prompt section's title and heading, exported so a rename moves the
#: prompt and the wiring pins in
#: ``tests/test_recon_gate_owned_action_phrasing_guidance.py`` together.
#: Follows the ``prompts/stage1.py::EXECUTING_A_CLUSTER_FOLD_HEADING``
#: precedent: tests locate the section by this constant rather than by pinning
#: its prose, so a pure rename with a byte-identical body cannot turn them red
#: for no behavioural reason.
GATE_OWNED_ACTION_NORM_TITLE = 'Gate-Owned Finding Phrasing'
GATE_OWNED_ACTION_NORM_HEADING = f'## {GATE_OWNED_ACTION_NORM_TITLE}'


def render_gate_owned_action_norm() -> str:
    """Render the Stage-1 gate-owned ``suggested_action`` phrasing norm.

    Half A of task 4814 — the probabilistic half.  It embeds
    :data:`CANONICAL_HUMAN_GATE_ACTION` VERBATIM exactly once (never a
    reworded second copy) and states the Stage-1-facing rule: when a finding's
    cited task is gate-owned, phrase ``suggested_action`` as awaiting
    human-operator sign-off only, and never in a form that reads as
    authorizing a later stage to decide.

    It names the ONE carve-out by the imported
    :data:`~fused_memory.reconciliation.curator_gate_resolution_sweep.GATE_RESOLUTION_FLAG_TYPE`
    rather than re-spelling it, and it tells the stage that the same rule is
    enforced deterministically in code after the stage runs — the
    prompt-directive-plus-authoritative-code-gate shape the Flag Suppression
    Check and ``filter_stale_count_snapshot_corrections`` already use.

    Stage 1 ONLY.  It is deliberately not given to Stage 2 or Stage 3: Stage 2
    is the stage that legitimately holds ``set_task_status``/``submit_task``,
    and a section telling IT that gate-owned findings are awaiting human
    sign-off only sits one reword away from contradicting both
    ``build_gate_resolution_flag``'s instruction to transcribe a recorded
    ruling and ``render_source_completion_section``'s instruction to FILE the
    gate.  ``tests/test_recon_gate_owned_action_phrasing_guidance.py`` pins
    that absence.

    HARD CONSTRAINTS on this text, each enforced by a live suite (see that
    module, and ``TestCanonicalHumanGateAction`` in
    ``tests/reconciliation/test_gate_owned_finding_phrasing.py``): no
    ``escalate_blocker``; no brace in any form; no ``source_finding_id`` or
    ``related_memory_ids`` (they would break the derived-count invariant in
    ``test_finding_provenance_prompt_guidance.py``); no
    ``as_submit_task_kwargs``; no ``write_entity_standing_decision``; and NO
    recon-report tool-call example at all, since
    ``test_recon_report_guidance_drift.py``'s balanced-paren scan requires
    ``run_id`` inside every ``add_finding``-shaped example and a phrasing rule
    needs no call example.

    Pure: no I/O, no side effects.
    """
    return (
        GATE_OWNED_ACTION_NORM_HEADING + '\n'
        'Before you write a finding, check whether the task it cites is a human '
        'decision gate — the structured markers are '
        "metadata.operational_mode = 'gate' and metadata.always_escalates = "
        'true, and EITHER one is sufficient. When it is, the finding is '
        'evidence for a human, not work for a stage.\n\n'
        'PHRASE suggested_action ACCORDINGLY. Write it as awaiting '
        'human-operator sign-off only. Do NOT write it in any form that reads '
        'as authorizing a later stage to settle the question — the observed '
        'offenders are "Stage 2 or operator should decide", "enumerate" and '
        '"extend". Stage 2 holds update_task and set_task_status, so a '
        'suggested_action phrased as a decision to be made is an invitation to '
        'close a human decision that has not been made. Assemble the evidence, '
        'name what the human would need in order to rule, and stop there.\n\n'
        'THE SENTENCE TO WRITE, verbatim, as the opening of such a '
        'suggested_action:\n' + CANONICAL_HUMAN_GATE_ACTION + '\n\n'
        'ONE CARVE-OUT. A finding whose flag_type is '
        + GATE_RESOLUTION_FLAG_TYPE + ' is exempt. That flag is emitted by the '
        'deterministic curator-gate-resolution sweep, and its evidence is a '
        'ruling a human curator ALREADY recorded in Mem0 — so it legitimately '
        'asks Stage 2 to TRANSCRIBE a human decision rather than to make one, '
        'and it carries its own dismiss branch for the case where the memories '
        'merely curate the gate without ruling on it. Leave that flag\'s '
        'suggested_action alone.\n\n'
        'THIS RULE IS ALSO ENFORCED IN CODE. After this stage returns, a pure '
        'post-processor reads the same two metadata markers off the live task '
        'tree and prepends the sentence above to any finding citing a '
        'gate-owned task that does not already carry it, preserving your text '
        'after it. So the phrasing is corrected whether or not you comply, and '
        'the correction keys on the structured task fact rather than on your '
        'wording — you cannot evade it by rephrasing, and you gain nothing by '
        'hedging. Write the finding\'s evidence plainly.'
    )


def extract_human_gated_task_ids(tasks):
    """Return sorted, deduped str ids of human-gate-owned tasks.

    A task qualifies when ALL hold:

    - it is a ``dict`` (a non-dict element is skipped, never fed to ``.get``);
    - ``task['metadata']`` is a ``dict`` (absent/``None``/str/list metadata is
      skipped);
    - ``metadata['operational_mode'] == 'gate'`` OR
      ``metadata['always_escalates'] is True``;
    - ``task['id']`` is not ``None`` (coerced to ``str``).

    Callers pass ``filtered_task_tree.active_tasks`` (the dataclass field on
    ``task_filter.FilteredTaskTree``), which excludes done/cancelled tasks for
    free while still including ``status == 'blocked'`` — the state a filed
    human gate sits in.

    Structure, dict/metadata guards, ``str(id)`` coercion and the sorted,
    deduped return mirror the sibling selector
    ``curator_gate_resolution_sweep.extract_open_gate_task_ids`` verbatim.
    THREE things about that relationship are deliberate and worth recording:

    1. WHY THE SIBLING IS NOT REUSED. ``extract_open_gate_task_ids`` filters
       on ``operational_mode`` ALONE, and its docstring gives the reason: its
       evidence is a ``curator_gate_{id}`` Mem0 key that only exists for
       mode-gated tasks, so widening it would spend one Qdrant count per
       non-gate task to learn nothing. Widening it in place would break that
       documented contract. It was CONFIRMED by execution that it returns
       only ``['645']`` for a ``[gate-mode 645, always_escalates 652]``
       input, so it cannot serve this task's population. Task 4814 needs the
       broader one because the operational-routing coercion can produce a
       pure human gate carrying ``always_escalates=true`` while leaving
       ``operational_mode='llm'`` in place (documented in
       ``stage1_stall_detector``'s rationale for ignoring
       ``operational_mode``), and the task explicitly names "and/or".

    2. WHY ``TaskInterceptor._is_gate_metadata`` IS NOT IMPORTED.
       ``fused-memory/src/fused_memory/middleware/task_interceptor.py::_is_gate_metadata``
       (task 3446) is the canonical gate predicate and already implements
       both clauses above — but it carries a THIRD,
       ``execution_class == 'operational'``, which answers True on its own.
       Task 4814 asks only for the two gate markers, so importing it would
       rewrite the ``suggested_action`` of findings citing operational-class
       tasks that are not human gates: over-selection, and the opposite of
       the fail-safe direction chosen here (a strict predicate can only
       UNDER-select, so a non-gate finding is never rewritten). The sibling
       selector in this very package faced the same choice and resolved it
       the same way — citing ``_is_gate_metadata`` in its docstring rather
       than importing a private staticmethod on a middleware class whose
       contract is free to change for middleware reasons. This citation is
       what keeps the duplicated boolean honest.

    3. WHY ``always_escalates`` IS MATCHED WITH ``is True``, NOT TRUTHILY.
       Quoting ``_is_gate_metadata``'s own rationale: ``bool('false')`` is
       True, so a loose read would silently accept the opposite of the
       caller's intent. The string ``'true'``, the string ``'false'`` and the
       int ``1`` are all non-matches, as is ``operational_mode == 'GATE'``.

    ``metadata`` is read as a dict ONLY, with no JSON-string parse path,
    exactly as the shipped sibling does over the same ``active_tasks`` field.
    Diverging would be a behaviour change to an unrelated contract, and
    under-selection is the fail-safe direction here.

    Pure: no I/O, no side effects. Empty input returns ``[]``.
    """
    seen = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        metadata = task.get('metadata')
        if not isinstance(metadata, dict):
            continue
        if not (
            metadata.get('operational_mode') == 'gate'
            or metadata.get('always_escalates') is True
        ):
            continue
        raw_tid = task.get('id')
        if raw_tid is None:
            continue
        seen.add(str(raw_tid))
    return sorted(seen)
