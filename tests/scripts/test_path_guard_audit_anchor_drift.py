"""Mirror contract: the watcher's path-guard audit branch tracks the producer's anchors.

Task 3465. ``fused-memory``'s path-scope guard files its rejection / advisory /
routing-override records under a **synthetic anchor** ``task_id`` — no such task
exists. ``skills/escalation-watcher-auto/SKILL.md`` carries an audit-only branch
that recognises those records by anchor and ``agent_role`` and closes them
``close_only``/``benign``. This guard pins the consumer's declared
discriminators to the producer's live constants.

WHY A MACHINE CHECK. The failure mode is SILENT. Rename ``_ANCHOR_TASK_ID`` in
the escalator and the branch simply stops matching: no test goes red, no log
line fires, and the records revert to sitting ``pending`` at level 1 forever.
That is not cosmetic — ``_watcher_has_actionable_l1``
(``orchestrator/src/orchestrator/harness.py:12263``) counts any pending
un-promoted L1 as actionable and respawns a watcher rotation for it, so the
regression presents as an unbounded rotation spin that resolves nothing. A
``stamp_triage`` does not stop it either: the precheck reads ``status``/``level``
only, never ``triaged_at``.

STDLIB + PYTEST ONLY, AND NOT BY PREFERENCE. ``tests/scripts/orchestrator.yaml``
runs this directory as ``uv run --project shared pytest tests/scripts/``, and the
``shared`` project does not have ``fused_memory``, ``escalation`` or
``orchestrator`` installed. Importing the producer is IMPOSSIBLE here, so every
producer constant is recovered by parsing the source with ``ast``.

STRUCTURE, NEVER WORDING. The consumer half is read from a bare
``<!-- path-guard-anchors:begin/end -->`` marker span holding backticked tokens
and nothing else. SKILL.md is an LLM system prompt read verbatim, so the span
carries no scaffolding prose; and pinning the surrounding paragraph's wording
would fire on a reword rather than on real drift.

VACUITY IS THE HAZARD. Every extractor raises a loud ``AssertionError`` naming
the marker and the file rather than returning an empty set — an extractor that
silently yields nothing turns the drift assertions green while pinning nothing,
which is strictly worse than no guard because it still reports success.

FORWARD-COMPATIBLE BY CONSTRUCTION. Anchors are matched by PREFIX, so task
3123's ``task-path-guard-override`` (not yet on main — measured:
``git grep OVERRIDE_ANCHOR HEAD -- fused-memory/`` is empty) is already covered
by the ``task-path-guard`` token and this guard cannot go red the day 3123
lands.

MEASURED RED at base HEAD ``265d4db1b8`` — the marker span does not exist yet,
so ``declared_discriminators()`` raises before either assertion is reached, and
BOTH tests fail identically. Verbatim, line-wrapped only for this docstring::

    E   AssertionError: expected exactly one '<!-- path-guard-anchors:begin
    E   -->' marker in skills/escalation-watcher-auto/SKILL.md, found 0 (task
    E   3465). That span declares the discriminator tokens the
    E   `scope_violation` audit-only branch matches path-guard records on. If
    E   it was deleted, restore it; if it was duplicated, one of the two copies
    E   is unpinned and free to drift.
    E   assert 0 == 1
    FAILED tests/scripts/test_path_guard_audit_anchor_drift.py::test_every_scope_violation_anchor_is_covered_by_a_declared_prefix
    FAILED tests/scripts/test_path_guard_audit_anchor_drift.py::test_declared_agent_role_matches_the_producer
    2 failed in 0.08s

MEASURED RED for the OPERATOR half, against the SKILL.md span already in place —
``OPERATIONS.md`` has no ``path-guard-esc-ids`` span yet, so Extractor C raises::

    E   AssertionError: expected exactly one '<!-- path-guard-esc-ids:begin
    E   -->' marker in OPERATIONS.md, found 0 (task 3465). That span declares
    E   the escalation-id prefixes an operator greps to identify path-guard
    E   synthetic-anchor audit records. If it was deleted, restore it; if it was
    E   duplicated, one of the two copies is unpinned and free to drift.
    E   assert 0 == 1
    FAILED ...::test_operations_declares_an_esc_id_prefix_for_every_scope_violation_anchor
    FAILED ...::test_no_declared_prefix_is_dead
    FAILED ...::test_the_blocking_adjudicator_record_is_not_swallowed
    3 failed, 2 passed in 0.17s

THREE, NOT TWO. ``test_the_blocking_adjudicator_record_is_not_swallowed`` is a
STANDING isolation guard, not part of that RED — it was expected to pass already
against the SKILL.md span. It does not, because it reads BOTH spans and so hits
the same missing-marker raise first. Recorded as measured rather than as
predicted, and not worked around: an isolation invariant that skipped the span it
cannot read would be exactly the vacuous pass this module's extractors exist to
prevent. It goes green with the other two in the same step, and from then on its
value is that it fails the day someone widens a prefix.

MEASURED RED for the MECHANISM-EXISTS half — the amendment's defect, and the one
a guard can actually stop recurring: a recipe that names a mechanism which does
not exist, and so silently no-ops while reporting success::

    E   AssertionError: the `scope_violation` recipe in
    E   skills/escalation-watcher-auto/SKILL.md never names 'granted_files'
    E   (task 3465). That parameter on `resolve_issue(action='resume')` is what
    E   ACTUALLY widens a task's scope — it is folded into plan.files /
    E   metadata.files / file-locks orchestrator-side. [...]
    E   AssertionError: the `scope_violation` recipe in
    E   skills/escalation-watcher-auto/SKILL.md mentions `update_task` alongside
    E   a ['"modules"'] key (task 3465). ``Scheduler._get_modules`` [...] derives
    E   locks from ``metadata.files`` and has never read ``metadata.modules``, so
    E   that write is a silent no-op. [...]
    FAILED ...::test_both_scope_violation_recipes_name_granted_files
    FAILED ...::test_no_scope_violation_recipe_writes_metadata_modules
    2 failed, 7 passed in 0.41s

The other two mechanism tests — ``granted_files`` is a real ``resolve_issue``
parameter, and ``_get_modules`` reads ``'files'`` and not ``'modules'`` — pass
already, and that is the point: they are standing PREMISE guards, verified true
BEFORE the recipes were rewritten to depend on them.

OUT OF SCOPE. ``fused-memory/src/fused_memory/middleware/scope_violation_escalator.py``
is READ-ONLY to this module — it is AST-parsed, never imported and never edited.
Re-gating the producer is explicitly not the remedy: the unconditional census is
deliberate (task 3123).
"""
from __future__ import annotations

import ast
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).parents[2]

ESCALATOR_SRC = (
    REPO_ROOT
    / "fused-memory/src/fused_memory/middleware/scope_violation_escalator.py"
)
SKILL_DOC = REPO_ROOT / "skills/escalation-watcher-auto/SKILL.md"
OPERATIONS_DOC = REPO_ROOT / "OPERATIONS.md"

ANCHORS_BEGIN = "<!-- path-guard-anchors:begin -->"
ANCHORS_END = "<!-- path-guard-anchors:end -->"

ESC_IDS_BEGIN = "<!-- path-guard-esc-ids:begin -->"
ESC_IDS_END = "<!-- path-guard-esc-ids:end -->"

# ``EscalationQueue.make_id`` mints ``esc-<task_id>-<n>``, and the producer's own
# module comment gives ``esc-task-path-guard-37`` as the worked example. So the id
# an operator greps is DERIVED from the anchor, never independently authored — the
# whole point of pinning the two.
ESC_ID_PREFIX = "esc-"

# The MECHANISM-EXISTS half (task 3465 amendment). Both recipes previously told an
# agent to widen scope by writing ``metadata.modules`` — a key ``Scheduler._get_modules``
# has never read. Every application of that instruction to a real task id was a
# silent no-op reporting success. These are the sources whose reality the rewritten
# recipes now depend on.
SERVER_SRC = REPO_ROOT / "escalation/src/escalation/server.py"
SCHEDULER_SRC = REPO_ROOT / "orchestrator/src/orchestrator/scheduler.py"

# Same file as SKILL_DOC, aliased at the name the mechanism tests read it under —
# one Path object, so the two halves of this module can never point at different
# copies of the auto skill.
AUTO_SKILL = SKILL_DOC
INTERACTIVE_SKILL = REPO_ROOT / "skills/escalation-watcher/SKILL.md"

RESOLVE_ISSUE = "resolve_issue"
GRANTED_FILES = "granted_files"
LOCK_DERIVATION_FN = "_get_modules"

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# A ``modules`` key inside a metadata mapping, in the spellings a recipe could
# plausibly carry. NOT a bare ``modules`` word match: both recipes legitimately
# use the English word ("needs modules beyond its assigned scope"), and firing on
# that would pin prose instead of the defect.
_METADATA_MODULES = (
    '"modules"',
    "'modules'",
    "metadata.modules",
)

# The budget-misconfig record shares this producer module but NOTHING else: it is
# category 'adjudicator_config_defect', agent_role 'fused-memory/path-scope-adjudicator'
# and severity 'blocking', and the producer's own comment calls it "deliberately
# distinct from the scope_violation family so operators can immediately tell
# these apart". It genuinely needs a human, so it must never be swept into the
# benign auto-close family.
BUDGET_MISCONFIG_ANCHOR_CONST = "_BUDGET_MISCONFIG_ANCHOR_TASK_ID"

# Recovered by name so a refactor that moves any of them into a class body, an
# f-string or a computed expression fails HERE — loudly, naming the constant —
# instead of silently emptying every check downstream.
REQUIRED_PRODUCER_CONSTANTS = (
    "_ANCHOR_TASK_ID",
    "_AGENT_ROLE",
    "_CATEGORY",
    BUDGET_MISCONFIG_ANCHOR_CONST,
    "_BUDGET_MISCONFIG_AGENT_ROLE",
)

_BACKTICKED = re.compile(r"`([^`\n]+)`")


def producer_constants() -> dict[str, str]:
    """Module-level ``NAME = 'literal'`` string constants of the escalator.

    Handles ``AnnAssign`` as well as ``Assign``: the producer writes
    ``_ANCHOR_TASK_ID: str = 'task-path-guard'``, so an ``Assign``-only walk
    returns an EMPTY dict against this file and every downstream check passes
    vacuously. Measured, not assumed — all five required names are annotated.

    Raises rather than returning a partial mapping when any required constant is
    missing.
    """
    tree = ast.parse(ESCALATOR_SRC.read_text(encoding="utf-8"))

    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = value.value

    missing = [name for name in REQUIRED_PRODUCER_CONSTANTS if name not in found]
    assert not missing, (
        f"could not recover {missing} as module-level string constants from "
        f"{ESCALATOR_SRC.relative_to(REPO_ROOT)} (task 3465). This guard AST-parses "
        f"that module because it cannot be imported under `--project shared`; if a "
        f"constant was renamed, moved into a class, or computed from an f-string, "
        f"update the names in REQUIRED_PRODUCER_CONSTANTS here AND the "
        f"discriminator tokens declared in the `path-guard-anchors` span of "
        f"{SKILL_DOC.relative_to(REPO_ROOT)}. Recovered: {sorted(found)}"
    )
    return found


def scope_violation_anchors() -> set[str]:
    """Every synthetic anchor the ``scope_violation`` family files under.

    Any ``*_ANCHOR_TASK_ID`` constant EXCEPT the budget-misconfig one, which is a
    different category, agent_role and severity — see
    ``BUDGET_MISCONFIG_ANCHOR_CONST`` above. Picking these up by name suffix
    rather than by a hard-coded list is what makes a NEW scope_violation anchor
    (e.g. task 3123's override anchor) arrive as a red test rather than as a
    silently-unhandled record class.
    """
    constants = producer_constants()
    anchors = {
        value
        for name, value in constants.items()
        if name.endswith("_ANCHOR_TASK_ID") and name != BUDGET_MISCONFIG_ANCHOR_CONST
    }
    assert anchors, (
        f"no scope_violation anchor constants found in "
        f"{ESCALATOR_SRC.relative_to(REPO_ROOT)} (task 3465) — every downstream "
        f"coverage assertion would pass vacuously. Constants recovered: "
        f"{sorted(constants)}"
    )
    return anchors


def marked_tokens(text: str, begin: str, end: str, doc_label: str, purpose: str) -> set[str]:
    """The backtick-quoted tokens inside the ``begin``/``end`` marker span.

    Every failure is a loud ``AssertionError`` naming the marker literal and the
    document — never an empty set. That is the vacuity contract this whole module
    rests on, and it is the same one ``_documented_lint_command`` in
    ``test_contributing_lint_command_drift.py`` states: a missing span must be
    RED, never a silent pass.

    Inverted markers produce a negative slice and therefore an empty token set,
    which the final assertion catches with the same remedy text.
    """
    begin_count = text.count(begin)
    assert begin_count == 1, (
        f"expected exactly one {begin!r} marker in {doc_label}, found "
        f"{begin_count} (task 3465). That span {purpose} If it was deleted, "
        f"restore it; if it was duplicated, one of the two copies is unpinned "
        f"and free to drift."
    )
    end_count = text.count(end)
    assert end_count == 1, (
        f"expected exactly one {end!r} marker to close {begin!r} in {doc_label}, "
        f"found {end_count} (task 3465) — restore the closing marker."
    )

    span = text[text.index(begin) + len(begin):text.index(end)]
    tokens = {match.strip() for match in _BACKTICKED.findall(span)}
    tokens.discard("")
    assert tokens, (
        f"the span between {begin!r} and {end!r} in {doc_label} declares no "
        f"backtick-quoted tokens (task 3465) — every assertion keyed on it would "
        f"pass vacuously. That span {purpose}"
    )
    return tokens


def declared_discriminators() -> set[str]:
    """Discriminator tokens the auto-watcher's audit-only branch declares."""
    return marked_tokens(
        SKILL_DOC.read_text(encoding="utf-8"),
        ANCHORS_BEGIN,
        ANCHORS_END,
        str(SKILL_DOC.relative_to(REPO_ROOT)),
        "declares the discriminator tokens the `scope_violation` audit-only "
        "branch matches path-guard records on.",
    )


def declared_escalation_id_prefixes() -> set[str]:
    """Escalation-id prefixes OPERATIONS.md tells an operator to grep for.

    A trailing ``*`` is stripped: ``esc-task-path-guard*`` is how an operator
    actually types the glob, and the doc should read the way the command is run.
    Everything else is returned verbatim — normalising further would silently
    canonicalise away a real difference.
    """
    tokens = marked_tokens(
        OPERATIONS_DOC.read_text(encoding="utf-8"),
        ESC_IDS_BEGIN,
        ESC_IDS_END,
        str(OPERATIONS_DOC.relative_to(REPO_ROOT)),
        "declares the escalation-id prefixes an operator greps to identify "
        "path-guard synthetic-anchor audit records.",
    )
    return {token.rstrip("*") for token in tokens} - {""}


def declared_task_id_prefixes() -> set[str]:
    """The SKILL.md span's tokens MINUS the producer's ``agent_role`` value.

    The branch's discriminator is a disjunction — ``agent_role`` equality OR an
    id prefix — so the one span legitimately holds tokens of two different kinds.
    Partitioning by "is this the live ``_AGENT_ROLE`` value?" rather than by a
    hard-coded list keeps both halves honest at once: if the producer renames the
    role, ``test_declared_agent_role_matches_the_producer`` goes red AND the now
    unclassifiable stale token falls into this set, where
    ``test_no_declared_prefix_is_dead`` reports it as dead. Neither rename can
    hide behind the other.
    """
    declared = declared_discriminators()
    agent_role = producer_constants()["_AGENT_ROLE"]
    prefixes = declared - {agent_role}
    assert prefixes, (
        f"the `path-guard-anchors` span in {SKILL_DOC.relative_to(REPO_ROOT)} "
        f"declares only the agent_role token {agent_role!r} and no anchor prefix "
        f"at all (task 3465) — the id-prefix half of the branch's discriminator "
        f"would be pinned by nothing. Declared: {sorted(declared)}"
    )
    return prefixes


def test_every_scope_violation_anchor_is_covered_by_a_declared_prefix():
    """Every live synthetic anchor is recognised by the watcher's branch.

    PREFIX, not equality: ``task-path-guard`` must keep covering the
    ``task-path-guard-override`` anchor task 3123 adds, so this cannot go red on
    the day that lands. Under-coverage is the failure this exists to stop — an
    anchor no declared token matches produces records the branch never fires on,
    which is exactly the pending-L1 rotation spin.
    """
    declared = declared_discriminators()
    anchors = scope_violation_anchors()

    uncovered = [
        anchor
        for anchor in sorted(anchors)
        if not any(anchor.startswith(token) for token in declared)
    ]
    assert not uncovered, (
        f"the `scope_violation` audit-only branch in "
        f"{SKILL_DOC.relative_to(REPO_ROOT)} declares no prefix covering "
        f"{uncovered} (task 3465).\n"
        f"  live anchors ({ESCALATOR_SRC.relative_to(REPO_ROOT)}): {sorted(anchors)}\n"
        f"  declared tokens (`path-guard-anchors` span): {sorted(declared)}\n"
        f"An uncovered anchor means the branch never fires on those records: they "
        f"stay pending at level 1, and _watcher_has_actionable_l1 keeps respawning "
        f"watcher rotations that resolve nothing. Add the anchor (or a covering "
        f"prefix of it) to the marker span."
    )


def test_declared_agent_role_matches_the_producer():
    """The branch's ``agent_role`` conjunct is the value the producer stamps.

    The branch matches on ``agent_role`` OR the escalation-id prefix, so this and
    the anchor test pin the two halves of one disjunction independently — a
    rename of either alone must not leave the other silently carrying the match.
    """
    declared = declared_discriminators()
    agent_role = producer_constants()["_AGENT_ROLE"]

    assert agent_role in declared, (
        f"the producer stamps agent_role={agent_role!r} "
        f"({ESCALATOR_SRC.relative_to(REPO_ROOT)}), but the `path-guard-anchors` "
        f"span in {SKILL_DOC.relative_to(REPO_ROOT)} declares "
        f"{sorted(declared)} (task 3465). The audit-only branch keys on that "
        f"exact string; if the producer's role was renamed, update the token in "
        f"the marker span to match."
    )


def test_operations_declares_an_esc_id_prefix_for_every_scope_violation_anchor():
    """The id an operator greps is DERIVED from the producer's anchor.

    ``EscalationQueue.make_id`` mints ``esc-<task_id>-<n>``, so ``esc-`` +
    anchor is not a convention this guard invents — it is the id the queue
    actually produces. Pinning it stops OPERATIONS.md from carrying a hand-typed
    prefix that quietly stops matching the day the anchor is renamed, which is
    precisely when an operator most needs the grep to work.
    """
    declared = declared_escalation_id_prefixes()
    anchors = scope_violation_anchors()

    uncovered = [
        ESC_ID_PREFIX + anchor
        for anchor in sorted(anchors)
        if not any(
            (ESC_ID_PREFIX + anchor).startswith(token) for token in declared
        )
    ]
    assert not uncovered, (
        f"{OPERATIONS_DOC.relative_to(REPO_ROOT)} declares no escalation-id "
        f"prefix covering {uncovered} (task 3465).\n"
        f"  live anchors ({ESCALATOR_SRC.relative_to(REPO_ROOT)}): {sorted(anchors)}\n"
        f"  declared prefixes (`path-guard-esc-ids` span, trailing `*` stripped): "
        f"{sorted(declared)}\n"
        f"An operator following the troubleshooting row would grep a prefix that "
        f"matches none of the records causing the symptom. Add the missing "
        f"prefix to the marker span."
    )


def test_no_declared_prefix_is_dead():
    """A prefix declared in either doc must still match something live.

    This is the half that catches a producer RENAME. The coverage tests above
    only notice an anchor nothing covers; a STALE token left behind after a
    rename is invisible to them — and stale is the dangerous direction, because
    the branch then silently stops firing while both docs still read as if it
    does. Here the orphaned token has nothing to match and goes red by name.
    """
    anchors = scope_violation_anchors()
    live_esc_ids = {ESC_ID_PREFIX + anchor for anchor in anchors}

    dead_task_prefixes = [
        token
        for token in sorted(declared_task_id_prefixes())
        if not any(anchor.startswith(token) for anchor in anchors)
    ]
    assert not dead_task_prefixes, (
        f"the `path-guard-anchors` span in {SKILL_DOC.relative_to(REPO_ROOT)} "
        f"declares {dead_task_prefixes}, which prefixes no live scope_violation "
        f"anchor (task 3465).\n"
        f"  live anchors ({ESCALATOR_SRC.relative_to(REPO_ROOT)}): {sorted(anchors)}\n"
        f"A dead prefix means the audit-only branch no longer fires on anything: "
        f"the records go back to sitting pending at level 1 and respawning "
        f"watcher rotations. If the producer's anchor was renamed, rename the "
        f"token here to match it."
    )

    dead_esc_prefixes = [
        token
        for token in sorted(declared_escalation_id_prefixes())
        if not any(esc_id.startswith(token) for esc_id in live_esc_ids)
    ]
    assert not dead_esc_prefixes, (
        f"{OPERATIONS_DOC.relative_to(REPO_ROOT)} declares escalation-id prefixes "
        f"{dead_esc_prefixes}, which cover no live `esc-<anchor>` (task 3465).\n"
        f"  live esc-id stems: {sorted(live_esc_ids)}\n"
        f"The documented grep would return nothing for the symptom the "
        f"troubleshooting row describes. Update the token in the "
        f"`path-guard-esc-ids` span to match the renamed anchor."
    )


def test_the_blocking_adjudicator_record_is_not_swallowed():
    """No declared token may reach the budget-misconfig record.

    That record is ``category='adjudicator_config_defect'``,
    ``agent_role='fused-memory/path-scope-adjudicator'``, ``severity='blocking'``
    and genuinely needs an operator; the producer's own comment calls it
    "deliberately distinct from the scope_violation family so operators can
    immediately tell these apart". It is not hypothetical collateral — it lives
    in the SAME producer module as the anchors this guard pins, so an over-broad
    future prefix is one edit away from routing a blocking operator record into
    a benign auto-close, silently.

    This is a STANDING guard, not the RED of the step that added it: it passes
    against today's tokens, and its whole value is that it fails the day someone
    widens a prefix.
    """
    constants = producer_constants()
    budget_anchor = constants[BUDGET_MISCONFIG_ANCHOR_CONST]
    budget_esc_id = ESC_ID_PREFIX + budget_anchor
    budget_role = constants["_BUDGET_MISCONFIG_AGENT_ROLE"]

    declared = {
        str(SKILL_DOC.relative_to(REPO_ROOT)): declared_discriminators(),
        str(OPERATIONS_DOC.relative_to(REPO_ROOT)): declared_escalation_id_prefixes(),
    }

    for doc_label, tokens in declared.items():
        swallowing = sorted(
            token
            for token in tokens
            if budget_anchor.startswith(token)
            or budget_esc_id.startswith(token)
            or token == budget_role
        )
        assert not swallowing, (
            f"{doc_label} declares {swallowing}, which reaches the "
            f"budget-misconfig record (anchor {budget_anchor!r}, esc-id stem "
            f"{budget_esc_id!r}, agent_role {budget_role!r}) — task 3465.\n"
            f"That record is category='adjudicator_config_defect', "
            f"severity='blocking' and needs a human. A discriminator that "
            f"matches it would auto-close it as benign, silently. Narrow the "
            f"token so it covers only the scope_violation family."
        )


def function_params(src: pathlib.Path, name: str) -> set[str]:
    """Every parameter name of the function *name* defined in *src*.

    ``ast.walk`` rather than a module-body scan, so a method inside a class body
    is found — ``resolve_issue`` is registered inside the MCP server factory and
    ``_get_modules`` is a ``Scheduler`` method, neither at module level.

    Positional-only, positional, keyword-only, ``*args`` and ``**kwargs`` are all
    included: the question this answers is "can a caller pass this name?", and a
    parameter's calling convention is not the guard's business.
    """
    tree = ast.parse(src.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, (
        f"expected exactly one function named {name!r} in "
        f"{src.relative_to(REPO_ROOT)}, found {len(matches)} (task 3465). This "
        f"guard AST-parses that module because it cannot be imported under "
        f"`--project shared`; a rename must fail HERE, naming the function, "
        f"rather than emptying the check."
    )

    args = matches[0].args
    params = {
        arg.arg
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    }
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            params.add(extra.arg)
    return params


def function_body_constants(src: pathlib.Path, name: str) -> set[str]:
    """String constants in the body of *name*, EXCLUDING its docstring.

    Walking the AST rather than the text is what makes this honest: comments are
    gone by construction, and the docstring is dropped explicitly. Both matter
    here — ``_get_modules``'s own docstring says "metadata.files" and "modules"
    in prose, so a text scan would report whatever the prose happens to claim
    rather than what the code reads.
    """
    tree = ast.parse(src.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, (
        f"expected exactly one function named {name!r} in "
        f"{src.relative_to(REPO_ROOT)}, found {len(matches)} (task 3465)"
    )

    func = matches[0]
    body = func.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    constants: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                constants.add(node.value)
    return constants


def scope_violation_recipe(doc: pathlib.Path) -> str:
    """The ``scope_violation`` recipe section of *doc*, heading included.

    Matched on the ``scope_violation`` TOKEN in a heading line, not on an exact
    heading string: the auto skill spells it ``#### `scope_violation``` and the
    interactive one ``### `scope_violation` (info or blocking)``, and a title
    tweak must not silently empty this slice. The section runs to the next
    heading of the SAME-OR-HIGHER level, so subsections the recipe owns (the
    audit-only branch, the synthetic-anchor safety net) are inside it and the
    next sibling category is not.
    """
    lines = doc.read_text(encoding="utf-8").split("\n")

    start = None
    level = 0
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match and "scope_violation" in match.group(2):
            start, level = index, len(match.group(1))
            break
    assert start is not None, (
        f"no heading naming `scope_violation` found in "
        f"{doc.relative_to(REPO_ROOT)} (task 3465) — every assertion over this "
        f"recipe would pass vacuously. If the section was renamed, it must still "
        f"carry the category token so both the guard and a reader can find it."
    )

    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = _HEADING.match(lines[index])
        if match and len(match.group(1)) <= level:
            end = index
            break

    section = "\n".join(lines[start:end])
    assert section.strip() != lines[start].strip(), (
        f"the `scope_violation` section of {doc.relative_to(REPO_ROOT)} is a bare "
        f"heading with no body (task 3465) — the recipe assertions would pass "
        f"vacuously"
    )
    return section


def test_granted_files_is_a_real_resolve_issue_parameter():
    """The scope-expansion mechanism both recipes name actually exists.

    THE point of this guard, and the root defect class of task 3465: a recipe
    that names a mechanism which does not exist silently no-ops while reporting
    success. That is exactly what ``metadata.modules`` did — a key
    ``Scheduler._get_modules`` never read — for every application of the old
    recipe to a real task id.

    If ``granted_files`` is ever renamed or dropped, the failure surfaces HERE,
    at merge time, rather than as a rejected tool call inside an autonomous
    watcher rotation with nobody watching.
    """
    params = function_params(SERVER_SRC, RESOLVE_ISSUE)

    assert GRANTED_FILES in params, (
        f"{RESOLVE_ISSUE}() in {SERVER_SRC.relative_to(REPO_ROOT)} has no "
        f"{GRANTED_FILES!r} parameter (task 3465). Both `scope_violation` "
        f"recipes instruct an agent to pass it as the scope-expansion grant — "
        f"without it those instructions are unexecutable, and an agent following "
        f"them would have its call rejected mid-rotation. Parameters found: "
        f"{sorted(params)}"
    )


def test_both_scope_violation_recipes_name_granted_files():
    """Both tiers must name the real mechanism, not just the auto one.

    The amendment's point 3 exists because the clone was missed once already:
    the interactive skill carried the same inert-mechanism instruction as the
    auto skill and was not in the original plan's file list at all. Pinning both
    is what stops the fix from being applied to one tier and quietly lost in the
    other.
    """
    for doc in (AUTO_SKILL, INTERACTIVE_SKILL):
        recipe = scope_violation_recipe(doc)
        assert GRANTED_FILES in recipe, (
            f"the `scope_violation` recipe in {doc.relative_to(REPO_ROOT)} never "
            f"names {GRANTED_FILES!r} (task 3465). That parameter on "
            f"`resolve_issue(action='resume')` is what ACTUALLY widens a task's "
            f"scope — it is folded into plan.files / metadata.files / file-locks "
            f"orchestrator-side. A recipe that omits it leaves the grant as prose "
            f"only, and the resumed agent's briefing never reflects the expanded "
            f"scope."
        )


def test_no_scope_violation_recipe_writes_metadata_modules():
    """Neither recipe may instruct a metadata-scoped ``modules`` write.

    ``Scheduler._get_modules`` derives locks from ``metadata.files`` and has
    never read ``metadata.modules`` (pinned by the test below). A recipe that
    writes that key produces a silent no-op reporting success — the defect this
    task exists to remove, not relocate.

    TWO SHAPES, AND ONLY THESE. A quoted ``modules`` key co-occurring with an
    ``update_task`` mention, and the bare ``metadata.modules`` attribute
    spelling. Deliberately NOT a bare ``modules`` word match: both recipes
    legitimately use the English word, and firing on it would pin prose.

    WHAT THIS DOES NOT CATCH, stated so nobody mistakes it for full coverage:
    the interactive skill's defect was PROSE ("Extend the required modules in
    task metadata via `update_task`") with no quoted key, so this test was never
    red for that file. The positive requirement above — name ``granted_files`` —
    is what covers the prose-form defect. Negative and positive halves together,
    neither alone.
    """
    for doc in (AUTO_SKILL, INTERACTIVE_SKILL):
        recipe = scope_violation_recipe(doc)
        label = doc.relative_to(REPO_ROOT)

        if "update_task" in recipe:
            quoted = [token for token in _METADATA_MODULES[:2] if token in recipe]
            assert not quoted, (
                f"the `scope_violation` recipe in {label} mentions `update_task` "
                f"alongside a {quoted} key (task 3465). ``Scheduler._get_modules`` "
                f"(orchestrator/src/orchestrator/scheduler.py) derives locks from "
                f"``metadata.files`` and has never read ``metadata.modules``, so "
                f"that write is a silent no-op. Pass the paths as "
                f"``{GRANTED_FILES}`` on `resolve_issue(action='resume')` instead."
            )

        assert _METADATA_MODULES[2] not in recipe, (
            f"the `scope_violation` recipe in {label} names "
            f"``{_METADATA_MODULES[2]}`` (task 3465) — a key nothing reads. Scope "
            f"is widened by ``{GRANTED_FILES}`` on `resolve_issue`, which the "
            f"orchestrator folds into plan.files / metadata.files / locks."
        )


def test_lock_derivation_reads_files_not_modules():
    """The PREMISE that makes the ``granted_files`` rewrite correct.

    Not a restatement of the test above — that one pins what the DOCS say, this
    one pins what the CODE does. If someone ever teaches ``_get_modules`` to read
    ``metadata.modules`` again, the recipes' rationale stops holding and both
    should be revisited; that must arrive as a red test, not as a rediscovery.

    Asserted against ``Constant`` nodes with the docstring dropped, so the
    docstring's own prose mention of ``metadata.files`` cannot make this pass
    vacuously.
    """
    constants = function_body_constants(SCHEDULER_SRC, LOCK_DERIVATION_FN)

    assert "files" in constants, (
        f"{LOCK_DERIVATION_FN}() in {SCHEDULER_SRC.relative_to(REPO_ROOT)} no "
        f"longer reads a 'files' key (task 3465). Both `scope_violation` recipes "
        f"rest on locks being derived from ``metadata.files`` — if that changed, "
        f"revisit them. String constants in its body: {sorted(constants)}"
    )
    assert "modules" not in constants, (
        f"{LOCK_DERIVATION_FN}() in {SCHEDULER_SRC.relative_to(REPO_ROOT)} now "
        f"reads a 'modules' key (task 3465). Task 3465 deleted the "
        f"``metadata.modules`` write from both `scope_violation` recipes BECAUSE "
        f"nothing read it. If that premise no longer holds, revisit both recipes "
        f"rather than silently leaving them inconsistent with the scheduler."
    )
