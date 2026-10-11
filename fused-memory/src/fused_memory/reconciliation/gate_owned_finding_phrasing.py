"""Gate-owned finding phrasing — Stage 1 norm plus deterministic code gate (task 4814).

THE DEFECT. Stage 1 (``stages/memory_consolidator.py``) sometimes writes a
finding's ``suggested_action`` in words that read as authorizing Stage 2 to
DECIDE — the observed offenders being "Stage 2 or operator should decide",
"enumerate" and "extend" — when the finding's cited task is a HUMAN GATE
(``metadata.operational_mode == 'gate'`` and/or
``metadata.always_escalates == true``).  Stage 2 holds
``update_task``/``set_task_status``, so that phrasing invites a stage to close
a question only a human may answer.  Observed in autopilot_video run 8b2d3371,
findings 640d4ceb (cited task 645) and 66af9601 (cited task 652).

THE DESIGN — two halves sharing ONE canonical sentence.  Stated here once and
cited from the rest of this package rather than restated:

- Half A, probabilistic: :func:`render_gate_owned_action_norm` renders the
  Stage-1 prompt section telling the model how to phrase such a finding.
- Half B, deterministic: :func:`normalize_gate_owned_suggested_actions`
  prepends :data:`CANONICAL_HUMAN_GATE_ACTION` to any finding citing a
  gate-owned task.  Gate-ownership is read as a structured metadata fact off
  ``filtered_task_tree.active_tasks``, never guessed from the model's prose.

Both halves quote the one constant (INV-5), so a corrected finding can never
show an operator wording the stage was never given.  A prompt directive alone
would only lower the rate; prompt-plus-authoritative-code-gate is this repo's
established shape for that (the Flag Suppression Check, and
``flag_dedup.filter_stale_count_snapshot_corrections``).

WHICH CHANNEL HALF B CORRECTS, and the one it does not.  It rewrites
``report.items_flagged``, which reaches Stage 2 through
``stages/task_knowledge_sync.py::_format_flagged`` (via ``assemble_payload``'s
``combined_flags``).  ``assemble_payload`` builds that same list from a SECOND
channel as well — surviving ``mem0_active_query`` markers rendered from the
LLM's raw Mem0 ``content`` — and Stage 1's "## Stage 2 Flag Relay (FIX B)"
prompt section requires the model to write each flag to Mem0 too.  Nothing
here touches that copy, so an ambiguous phrasing can still reach Stage 2 by
that path.  That is the known residual and the first place to look if the
pattern recurs.

THE CARVE-OUT. ``curator_gate_resolution_sweep.build_gate_resolution_flag``
deliberately tells Stage 2 to set a gate task's status — legitimately, because
its evidence is a ruling a human curator ALREADY recorded in Mem0, so Stage 2
transcribes a human decision rather than making one.  Exempting it takes
PROVENANCE, not a string: ``GATE_RESOLUTION_FLAG_TYPE`` is the free-form value
``'task_completed_not_reflected'``, which an LLM-authored finding may pick for
itself.  So the exemption requires that flag_type AND
:data:`CURATOR_GATE_SWEEP_FLAG_KEY`, set only by
:func:`stamp_curator_gate_sweep_provenance` at the single call site that
appends the sweep's output.

Pure throughout: no I/O, no side effects, no task writes.
"""

from __future__ import annotations

from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    GATE_RESOLUTION_FLAG_TYPE,
)

#: Leading marker of :data:`CANONICAL_HUMAN_GATE_ACTION`, and the short
#: sentinel :func:`normalize_gate_owned_suggested_actions` tests compliance
#: against.  Matching the whole ~470-character sentence would demand verbatim
#: reproduction of 70 words — the least likely model outcome — so a compliant
#: paraphrase would be answered by prepending a near-duplicate of itself, at
#: ~470 chars of the shared ``_FLAGGED_ITEMS_CHAR_BUDGET`` that
#: ``task_knowledge_sync.py::_format_flagged`` drops later findings against.
CANONICAL_HUMAN_GATE_ACTION_MARKER = 'HUMAN DECISION GATE:'

#: The ONE sentence both halves of task 4814 quote (INV-5).
#:
#: CONTAINS NO BRACE, in any form, and this is load-bearing rather than
#: stylistic: the constant is interpolated into ``STAGE1_SYSTEM_PROMPT``, which
#: is an f-string — a bare brace fails at import — while
#: ``tests/test_stage1_consolidation_guidance.py`` separately asserts that
#: ``{{``/``}}`` never survive into the rendered text, so escaping is not an
#: escape hatch either.  Carrying no brace at all satisfies both.
CANONICAL_HUMAN_GATE_ACTION = (
    CANONICAL_HUMAN_GATE_ACTION_MARKER + ' the cited task is a human decision '
    "gate (metadata.operational_mode = 'gate' and/or "
    'metadata.always_escalates = true), so it is awaiting human-operator '
    'sign-off only. No reconciliation stage may decide this question, and no '
    'stage may record a decision on the task (update_task / set_task_status) '
    'that a human has not made. Everything that follows is evidence assembled '
    'FOR the human operator, not an instruction to a stage.'
)

#: Flag key stamped by :func:`stamp_curator_gate_sweep_provenance`, and the
#: provenance half of the carve-out (see the module docstring).
CURATOR_GATE_SWEEP_FLAG_KEY = 'curator_gate_sweep_emitted'

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
    """Render the Stage-1 gate-owned ``suggested_action`` phrasing norm (Half A).

    Embeds :data:`CANONICAL_HUMAN_GATE_ACTION` verbatim exactly once, and names
    the carve-out by the imported
    :data:`~fused_memory.reconciliation.curator_gate_resolution_sweep.GATE_RESOLUTION_FLAG_TYPE`
    rather than re-spelling it.

    Stage 1 ONLY.  Deliberately not given to Stage 2: Stage 2 legitimately
    holds ``set_task_status``/``submit_task``, and a section telling IT that
    gate-owned findings await human sign-off sits one reword away from
    contradicting both ``build_gate_resolution_flag``'s instruction to
    transcribe a recorded ruling and
    ``render_source_completion_section``'s instruction to FILE the gate.

    HARD CONSTRAINTS on this text, each enforced by a live suite (see
    ``TestCanonicalHumanGateAction`` in
    ``tests/reconciliation/test_gate_owned_finding_phrasing.py``, which names
    the banning suite per token): no ``escalate_blocker``; no brace in any
    form; no ``source_finding_id`` or ``related_memory_ids``; no
    ``as_submit_task_kwargs``; no ``write_entity_standing_decision``; and no
    recon-report tool-call example at all.

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
        'ONE CARVE-OUT, and it is not one you can claim. A finding whose '
        'flag_type is ' + GATE_RESOLUTION_FLAG_TYPE + ' is exempt ONLY when '
        'the deterministic curator-gate-resolution sweep emitted it in code '
        'after you ran: its evidence is a ruling a human curator ALREADY '
        'recorded in Mem0, so it legitimately asks Stage 2 to TRANSCRIBE a '
        'human decision rather than to make one, and it carries its own '
        'dismiss branch for the case where the memories merely curate the gate '
        'without ruling on it. Writing that flag_type on a finding of your own '
        'exempts nothing — the code gate keys on the sweep provenance, not on '
        'the string.\n\n'
        'THIS RULE IS ALSO ENFORCED IN CODE. After this stage returns, a pure '
        'post-processor reads the same two metadata markers off the live task '
        'tree and prepends the sentence above to any finding citing a '
        'gate-owned task whose suggested_action does not already open with it, '
        'preserving your text after it. So the phrasing is corrected whether '
        'or not you comply, and the correction keys on the structured task '
        'fact rather than on your wording — you cannot evade it by rephrasing, '
        'and you gain nothing by hedging. Write the finding\'s evidence '
        'plainly.'
    )


def extract_human_gated_task_ids(tasks):
    """Return sorted, deduped str ids of human-gate-owned tasks.

    A task qualifies when it is a ``dict`` with a ``dict`` ``metadata``
    carrying ``operational_mode == 'gate'`` OR ``always_escalates is True``,
    and a non-``None`` ``id`` (coerced to ``str``).  Malformed shapes are
    skipped rather than raised on.  Callers pass
    ``task_filter.FilteredTaskTree.active_tasks``, which excludes done and
    cancelled tasks while keeping ``status == 'blocked'`` — the state a filed
    human gate sits in.

    Three facts about this predicate that the code cannot show:

    1. WHY THE SIBLING IS NOT REUSED.
       ``curator_gate_resolution_sweep.extract_open_gate_task_ids`` filters on
       ``operational_mode`` ALONE because its evidence is a
       ``curator_gate_{id}`` Mem0 key that only exists for mode-gated tasks —
       widening it in place would spend one Qdrant count per non-gate task to
       learn nothing, breaking its documented contract.  Task 4814 needs the
       broader population: the operational-routing coercion can produce a pure
       human gate carrying ``always_escalates=true`` while leaving
       ``operational_mode='llm'`` in place (see ``stage1_stall_detector``'s
       rationale for ignoring ``operational_mode``), and the task names
       "and/or" explicitly.

    2. WHY ``TaskInterceptor._is_gate_metadata`` IS NOT IMPORTED.
       ``middleware/task_interceptor.py::_is_gate_metadata`` (task 3446) is the
       canonical gate predicate and implements both clauses above — but it
       carries a THIRD, ``execution_class == 'operational'``, which answers
       True on its own.  Importing it would rewrite findings citing
       operational-class tasks that are not human gates.  The sibling selector
       in this package resolved the same choice the same way, citing
       ``_is_gate_metadata`` rather than importing a private staticmethod on a
       middleware class whose contract is free to change for middleware
       reasons.  That citation is what keeps the duplicated boolean honest.

    3. WHY ``always_escalates`` IS MATCHED WITH ``is True``, NOT TRUTHILY.
       Quoting ``_is_gate_metadata``'s own rationale: ``bool('false')`` is
       True, so a loose read would silently accept the opposite of the
       caller's intent.  ``'true'``, ``'false'``, ``1`` and
       ``operational_mode == 'GATE'`` are all non-matches.

    This strictness bounds SELECTION, not the whole correction: a task that is
    not selected is never rewritten.  The one residual ambiguity is on the
    other side — a flag's top-level ``task_id`` channel carries no project, so
    see :func:`_flag_cited_task_ids` for what that costs.

    Pure: no I/O, no side effects.  Empty input returns ``[]``.
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


def stamp_curator_gate_sweep_provenance(flags):
    """Return copies of *flags* marked as curator-gate-resolution-sweep output.

    Call this on ``sweep_resolved_curator_gates``' flags at the point they are
    appended to ``report.items_flagged``: that call site is the only place that
    KNOWS the flags came from the sweep, and
    :func:`normalize_gate_owned_suggested_actions`' carve-out needs that
    provenance rather than the free-form ``flag_type`` string an LLM-authored
    finding may also pick (module docstring, THE CARVE-OUT).

    Copies rather than mutates, matching this module's other pass.

    Pure: no I/O, no side effects.
    """
    stamped = []
    for flag in flags:
        if not isinstance(flag, dict):
            stamped.append(flag)
            continue
        marked = dict(flag)
        marked[CURATOR_GATE_SWEEP_FLAG_KEY] = True
        stamped.append(marked)
    return stamped


def _flag_cited_task_ids(flag, project_id=None):
    """Return the set of str task ids *flag* cites, from both id channels.

    Reads the top-level ``task_id`` — split on ``','`` and stripped, because
    ``FINDING_ITEM_SCHEMA`` documents a comma-joined multi-id shape — plus
    every ``cited_tasks[i]['task_id']``.  ``cited_tasks`` is the AUTHORITATIVE
    dedup key per that schema, and a finding may carry its task ONLY there (the
    shape observed in autopilot_video run 8b2d3371), so reading ``task_id``
    alone would miss the very population this module exists for.

    PROJECT SCOPING. The gate ids this set is matched against come from the
    LOCAL project's task tree, while a ``cited_tasks`` entry carries its own
    required ``project_id`` and foreign citations are routine — see
    ``flag_dedup._resolve_live_cross_project_fix_task``, whose docstring
    records a live flag citing ``know_live:598`` alongside two ``dark_factory``
    tasks.  Task ids are small per-project integers, so a foreign citation
    colliding with a local gate id is a matter of time.  When *project_id* is
    given, an entry naming a DIFFERENT project is therefore skipped; an entry
    with no ``project_id`` is kept, matching the top-level ``task_id`` channel,
    which carries no project at all and so remains the one residual ambiguity.
    Omitting *project_id* keeps the unscoped behaviour.

    None and blank ids contribute nothing.  Malformed shapes (a non-list
    ``cited_tasks``, a non-dict entry) are skipped rather than raised on.

    Pure: no I/O, no side effects.
    """
    ids = set()

    raw_tid = flag.get('task_id')
    if raw_tid is not None:
        for part in str(raw_tid).split(','):
            part = part.strip()
            if part:
                ids.add(part)

    local_project = None if project_id is None else str(project_id)
    cited = flag.get('cited_tasks')
    if isinstance(cited, list):
        for entry in cited:
            if not isinstance(entry, dict):
                continue
            entry_project = entry.get('project_id')
            if (
                local_project is not None
                and entry_project is not None
                and str(entry_project) != local_project
            ):
                continue
            raw = entry.get('task_id')
            if raw is None:
                continue
            cited_id = str(raw).strip()
            if cited_id:
                ids.add(cited_id)

    return ids


def normalize_gate_owned_suggested_actions(flags, gate_task_ids, *, project_id=None):
    """Prepend :data:`CANONICAL_HUMAN_GATE_ACTION` to gate-owned findings (Half B).

    Selection keys on the STRUCTURED gate fact — the caller passes ids from
    :func:`extract_human_gated_task_ids` — never on the model's prose: "is this
    phrasing ambiguous?" has no reliable regex, so a wording detector would
    catch only the three offenders task 4814 enumerates, while a fact-keyed
    correction fires on phrasings nobody has seen yet.

    A flag citing a gate id (see :func:`_flag_cited_task_ids`) is corrected
    unless it is the sweep's own flag (``flag_type == GATE_RESOLUTION_FLAG_TYPE``
    AND :data:`CURATOR_GATE_SWEEP_FLAG_KEY` — module docstring, THE CARVE-OUT)
    or its ``suggested_action`` already opens with
    :data:`CANONICAL_HUMAN_GATE_ACTION_MARKER`.  That second test is what makes
    the pass idempotent across cycles, findings being persistent.

    A corrected flag is a shallow ``dict`` COPY carrying
    ``gate_owned_action_normalized = True``, whose ``suggested_action`` is the
    canonical sentence followed by the model's original text (or the sentence
    alone when the original is missing/``None``/blank, so no dangling separator
    is emitted).  Copying is load-bearing: ``report.items_flagged``'s
    ``_pre_filter_flags`` snapshot is a shallow list copy that ALIASES the
    caller's dicts, so an in-place rewrite would retroactively alter the
    pre-filter snapshot.  The model's own text is preserved rather than
    replaced because it is the finding's evidence.

    ``task_id``, ``flag_type`` and ``cited_tasks`` are never touched — those
    three ARE ``flag_dedup.compute_flag_signature``'s key, so cross-cycle
    dedup, suppression and the ``stage1_flag_markers_acknowledged`` diff are
    provably unaffected.  Never drops and never reorders, unlike every other
    member of the Stage-1 post-processor chain it sits in.

    Args:
        flags: Stage-1 finding dicts (``report.items_flagged``).
        gate_task_ids: Str/int ids of human-gate-owned tasks, ``str``-coerced
            here so an int-typed caller cannot silently match nothing.
        project_id: The running project, scoping ``cited_tasks`` matches.

    Returns:
        ``(flags, count)`` — a NEW list of the same length and order, and the
        number of findings whose ``suggested_action`` was rewritten.

    Pure and sync: no I/O, no side effects.  Deliberately NOT try/except
    wrapped, matching its pure sibling
    ``flag_dedup.filter_stale_count_snapshot_corrections`` at the same call
    site.
    """
    gate_ids = {str(tid) for tid in gate_task_ids}
    if not gate_ids:
        return list(flags), 0

    normalized = []
    count = 0

    for flag in flags:
        if not isinstance(flag, dict):
            normalized.append(flag)
            continue

        if (
            flag.get('flag_type') == GATE_RESOLUTION_FLAG_TYPE
            and flag.get(CURATOR_GATE_SWEEP_FLAG_KEY) is True
        ):
            normalized.append(flag)
            continue

        if not (_flag_cited_task_ids(flag, project_id) & gate_ids):
            normalized.append(flag)
            continue

        existing = flag.get('suggested_action')
        existing_text = existing if isinstance(existing, str) else ''
        if existing_text.lstrip().startswith(CANONICAL_HUMAN_GATE_ACTION_MARKER):
            normalized.append(flag)
            continue

        corrected = dict(flag)
        corrected['suggested_action'] = (
            CANONICAL_HUMAN_GATE_ACTION + ' ' + existing_text.strip()
            if existing_text.strip()
            else CANONICAL_HUMAN_GATE_ACTION
        )
        corrected['gate_owned_action_normalized'] = True
        normalized.append(corrected)
        count += 1

    return normalized, count
