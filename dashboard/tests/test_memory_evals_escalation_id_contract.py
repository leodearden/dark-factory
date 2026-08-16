"""The MEMORY_EVALS ↔ ESCALATIONS id-space contract (task 3471).

`tab_memory_evals.jsx` navigates from a memory-eval row to the escalations tab
by id — `onNavigate('esc', escalation.id)` — and `tab_escalations.jsx`'s
`findEscalationRow` resolves that id with a strict `row.id === id`.  So every
escalation id MEMORY_EVALS emits must appear, with the SAME TYPE, among the
rows ESCALATIONS ships.  Nothing pins that today: the two payloads are built by
two producers that never reference each other

    MEMORY_EVALS  — `memory_evals._escalation_projection` → `record.get('id')`
    ESCALATIONS   — `redux_api.shape_escalations`         → `{**esc, ...}`

and no test in this suite has ever built both from one fixture directory (the
`build_escalation_queues` importers and the `build_memory_evals` importers were
disjoint sets).  A link that resolves is therefore a property of two
independently-evolving readers happening to agree, not a checked contract.

WHY THE MUTATION TEST BELOW EXISTS.  The contract HOLDS today: both producers
read `config.reconciliation_escalations_dir` and both pass the id through
without coercion.  A test that merely asserts "no violations" is therefore
green on arrival and would stay green if `collect_escalation_id_violations`
returned `[]` unconditionally — it would pin nothing at all.
`test_the_id_space_check_catches_a_divergent_projection` is what gives the rest
of this module its teeth: it monkeypatches the producer into emitting a
divergent id and demands the checker NOTICE.  Every other assertion here is
only as good as that one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from _dashboard_helpers import build_dual_escalation_tree

from dashboard.data.escalations import build_escalation_queues
from dashboard.data.redux_api import shape_escalations

# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------


def _iter_memory_eval_escalations(
    memory_evals_payload: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(payload_path, escalation)`` for every escalation MEMORY_EVALS reaches.

    THREE reach paths, which is the whole set the JSX can navigate from:

    * ``evals[i].metrics[j].escalation``  — a metric row's per-metric link
      (`memory_evals.py` `_build_eval`, via `_escalation_projection`)
    * ``storm_escape.escalation``         — the run-scoped aggregate filing
    * ``unmatched_escalations[k]``        — open escalations no metric row
      claims (via `_unmatched_projection`, which delegates to the same
      projection)

    The path string is what a violation NAMES, so a failure says which row of
    which payload carries the unresolvable id rather than only that one exists.
    """
    for ev in memory_evals_payload.get('evals') or []:
        if not isinstance(ev, dict):
            continue
        eval_id = ev.get('eval_id')
        for metric in ev.get('metrics') or []:
            if not isinstance(metric, dict):
                continue
            escalation = metric.get('escalation')
            if isinstance(escalation, dict):
                yield (
                    f'evals[{eval_id!r}].metrics[{metric.get("metric_id")!r}].escalation',
                    escalation,
                )

    storm = memory_evals_payload.get('storm_escape')
    if isinstance(storm, dict):
        escalation = storm.get('escalation')
        if isinstance(escalation, dict):
            yield ('storm_escape.escalation', escalation)

    for index, escalation in enumerate(memory_evals_payload.get('unmatched_escalations') or []):
        if isinstance(escalation, dict):
            yield (f'unmatched_escalations[{index}]', escalation)


def collect_escalation_id_violations(
    memory_evals_payload: dict[str, Any],
    escalations_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Every MEMORY_EVALS escalation id that ESCALATIONS cannot resolve.

    Args:
        memory_evals_payload: what ``build_memory_evals`` returns (the
            MEMORY_EVALS block itself, unwrapped).
        escalations_payload: the ``ESCALATIONS`` block —
            ``shape_escalations(build_escalation_queues(config), {})['ESCALATIONS']``.
            Rows live under ``subsections[k]['escalations']``; the key is
            ``escalations``, NOT ``rows`` (``rows`` is only ``shape_escalations``'
            local name for the list it is accumulating).

    Returns a list of ``{path, id, kind, detail}`` records, empty when every id
    resolves.  ``kind`` is:

    * ``absent``        — no row carries this id at all.  The link dead-ends.
    * ``type_mismatch`` — a row carries the same id VALUE but at a different
      Python type (``4242`` vs ``'4242'``).  Distinguished from ``absent``
      because it is the failure a `==` check cannot see and the browser
      cannot survive: `findEscalationRow` uses `row.id === id` with no
      `String()` coercion, so a str/int drift renders "escalation not found"
      while both payloads look perfectly populated.  The sibling ``task_id``
      IS `str()`-wrapped in `shape_escalations`; `id` deliberately is not, so
      this is a live shape, not a hypothetical one.

    Directionality is deliberate and asserted separately by
    `test_the_id_space_subset_direction_is_asymmetric_by_design`: this checks
    MEMORY_EVALS ⊆ ESCALATIONS only.  The reverse is FALSE BY DESIGN —
    `_index_escalations` drops non-`eval_regression` categories, closed
    statuses and duplicate fingerprints while `shape_escalations` filters
    nothing.
    """
    subsections = escalations_payload.get('subsections')
    if subsections is None and 'ESCALATIONS' in escalations_payload:
        raise TypeError(
            'collect_escalation_id_violations takes the ESCALATIONS BLOCK, not the '
            "shape_escalations() return value — pass shape_escalations(...)['ESCALATIONS']. "
            'Unwrapping silently here would let a caller pass the wrapper, find no '
            'subsections, and get an empty (vacuously clean) violation list.'
        )

    # str-keyed so a same-value/different-type row is found and classified
    # rather than reported as absent.  The list holds every type seen under
    # that key, since two subsections may legitimately carry the same id.
    rows_by_str_id: dict[str, list[Any]] = {}
    for subsection in subsections or []:
        for row in subsection.get('escalations') or []:
            if isinstance(row, dict):
                rows_by_str_id.setdefault(str(row.get('id')), []).append(row.get('id'))

    violations: list[dict[str, Any]] = []
    for path, escalation in _iter_memory_eval_escalations(memory_evals_payload):
        emitted = escalation.get('id')
        candidates = rows_by_str_id.get(str(emitted))
        if candidates is None:
            violations.append({
                'path': path,
                'id': emitted,
                'kind': 'absent',
                'detail': (
                    f'MEMORY_EVALS {path} links escalation id {emitted!r}, which appears '
                    'in NO ESCALATIONS row — tab_escalations.jsx findEscalationRow would '
                    'resolve nothing and the tab would render "not found".'
                ),
            })
        elif not any(type(candidate) is type(emitted) for candidate in candidates):
            violations.append({
                'path': path,
                'id': emitted,
                'kind': 'type_mismatch',
                'detail': (
                    f'MEMORY_EVALS {path} links escalation id {emitted!r} '
                    f'({type(emitted).__name__}), but every ESCALATIONS row with that '
                    f'value carries it as {sorted({type(c).__name__ for c in candidates})} — '
                    'findEscalationRow compares with `===`, so the link dead-ends.'
                ),
            })
    return violations


# ---------------------------------------------------------------------------
# The mutation harness — what makes every other assertion here non-vacuous
# ---------------------------------------------------------------------------


def _build_memory_evals(config) -> dict[str, Any]:
    from dashboard.data.memory_evals import build_memory_evals

    return build_memory_evals(
        config.memory_evals_dir, config.reconciliation_escalations_dir
    )


def _build_escalations(config) -> dict[str, Any]:
    """The ESCALATIONS block, built from the SAME config the payload above used."""
    return shape_escalations(build_escalation_queues(config), {})['ESCALATIONS']


def _reach_kind(path: str) -> str:
    """Bucket a walker path string into one of the three reach paths.

    Closed vocabulary on purpose: a path the walker learns to emit but this
    function does not know is an AssertionError, not a silently-uncounted
    fourth reach path that the coverage precondition below would then never
    notice was untested.
    """
    if path.startswith('evals['):
        return 'metric_row'
    if path.startswith('storm_escape'):
        return 'storm_escape'
    if path.startswith('unmatched_escalations['):
        return 'unmatched'
    raise AssertionError(
        f'_iter_memory_eval_escalations emitted an unclassified reach path {path!r} — '
        'add it here AND to the coverage precondition, or it goes unchecked.'
    )


def test_the_id_space_check_catches_a_divergent_projection(tmp_path: Path, monkeypatch) -> None:
    """A projection that emits a divergent id MUST produce a structured violation.

    This is the anti-vacuity proof for the whole module.  The contract holds
    today, so `collect_escalation_id_violations` returns `[]` on the real
    producers — and would keep returning `[]` if it were broken, if it walked
    the wrong keys, or if the fixture reached no escalation at all.  Here the
    producer is mutated into exactly the two ways the id space can drift and
    the checker is required to SEE both:

    (1) VALUE divergence — the emitted id is not the on-disk one.  This is what
        any future re-keying (a prefix, a slug, a per-payload surrogate) looks
        like from the consumer's side.

    (2) TYPE divergence — the emitted id is `str()`-coerced while the row keeps
        the raw JSON value.  Not hypothetical: `shape_escalations` ALREADY
        wraps the sibling `task_id` in `str()` (redux_api.py), so the coercion
        this simulates is one line away from being real, and a `==`-based
        checker would call it clean.

    Both arms monkeypatch `_escalation_projection` rather than editing the
    payload after the fact, so the mutation enters through the real code path
    that all three reach paths share.
    """
    from dashboard.data import memory_evals

    tree = build_dual_escalation_tree(tmp_path)
    config = tree.config
    escalations = _build_escalations(config)
    real_projection = memory_evals._escalation_projection

    # Control.  Both halves matter: the checker must be SILENT on the real
    # producers, and it must have had something to look at — a fixture that
    # reached no escalation would make every arm below meaningless.
    baseline = _build_memory_evals(config)
    reached = list(_iter_memory_eval_escalations(baseline))
    assert reached, (
        'the fixture tree yielded NO escalation on any MEMORY_EVALS reach path, so '
        'the mutation arms below would pass vacuously. build_dual_escalation_tree '
        'must produce at least one linked escalation.'
    )
    assert collect_escalation_id_violations(baseline, escalations) == [], (
        'the UNMUTATED producers already disagree on the id space — fix that before '
        'reading anything else in this module.'
    )

    # (1) VALUE divergence: every emitted id gains a suffix the queue never had.
    with monkeypatch.context() as mp:
        mp.setattr(
            memory_evals,
            '_escalation_projection',
            lambda record: {**real_projection(record), 'id': f'{record.get("id")}-x'},
        )
        value_violations = collect_escalation_id_violations(
            _build_memory_evals(config), escalations
        )
    assert len(value_violations) == len(reached), (
        'a projection emitting a re-keyed id must be caught on EVERY reach path, '
        f'but {len(value_violations)} of {len(reached)} were reported: '
        f'{value_violations}'
    )
    assert {v['kind'] for v in value_violations} == {'absent'}, (
        f'a re-keyed id resolves to no row at all, so every violation must be '
        f'kind="absent": {value_violations}'
    )
    assert all(v['path'] for v in value_violations), (
        'every violation must NAME the payload path it came from — a bare count '
        'leaves the operator to grep three reach paths by hand.'
    )

    # (2) TYPE divergence.  Needs an escalation whose on-disk id is a JSON
    #     NUMBER, so that `str()`-coercing the projection produces the same
    #     VALUE at a different TYPE.  Written here rather than baked into the
    #     shared tree: the mutation harness owns its own mutation.
    numeric_id = 4242
    esc_dir = config.reconciliation_escalations_dir
    esc_dir.mkdir(parents=True, exist_ok=True)
    (esc_dir / f'{numeric_id}.json').write_text(json.dumps({
        'id': numeric_id,
        'task_id': 'memory-eval-e1',
        'agent_role': 'memory-eval-runner',
        'severity': 'blocking',
        'category': 'eval_regression',
        'summary': 'numeric-id escalation',
        'detail': '',
        'timestamp': '2026-07-30T03:15:00+00:00',
        'status': 'pending',
        'level': 1,
        'dedupe_fingerprint': 'eval:no-such-eval|metric:no-such-metric',
    }, indent=2) + '\n')

    escalations_with_numeric = _build_escalations(config)
    assert any(
        row.get('id') == numeric_id
        for sub in escalations_with_numeric['subsections']
        for row in sub['escalations']
    ), (
        'ESCALATIONS did not carry the numeric-id record — it must reach a row '
        'un-coerced for the type arm below to mean anything.'
    )

    with monkeypatch.context() as mp:
        mp.setattr(
            memory_evals,
            '_escalation_projection',
            lambda record: {**real_projection(record), 'id': str(record.get('id'))},
        )
        type_violations = collect_escalation_id_violations(
            _build_memory_evals(config), escalations_with_numeric
        )
    assert [v['kind'] for v in type_violations] == ['type_mismatch'], (
        'str()-coercing ONLY the memory-evals side of a numeric id must be reported '
        'as exactly one type_mismatch (the string-id records are unaffected by '
        f'str()): {type_violations}'
    )
    assert type_violations[0]['id'] == str(numeric_id), (
        f'the violation must carry the EMITTED id, not the on-disk one: {type_violations}'
    )


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------

# The three reach paths, and WHY it takes two trees to exercise them.
#
# MEASURED against the real producer, not assumed: a TRIGGERED storm escape
# suppresses every per-metric link program-wide.  `_build_eval` only consults
# the escalation index when `storm is None`; otherwise the fingerprint goes to
# `storm_suppressed` and the row renders `parity='storm_collapsed'` with
# `escalation: None`.  So `evals[i].metrics[j].escalation` and
# `storm_escape.escalation` can never BOTH be populated in one payload — they
# are alternatives by design, since the storm exists precisely to collapse the
# per-metric links into one aggregate filing.
#
# The coverage precondition below is therefore taken over the UNION of two
# trees rather than weakened to "some path contributed".  Each tree is still
# ONE directory feeding BOTH producers, which is the property under test; what
# two trees buy is the two mutually-exclusive storm states.
_REACH_PATHS = ('metric_row', 'storm_escape', 'unmatched')


def test_every_memory_eval_escalation_id_resolves_in_the_escalations_payload(
    tmp_path: Path,
) -> None:
    """Every escalation id MEMORY_EVALS emits resolves to an ESCALATIONS row.

    The deliverable contract.  `tab_memory_evals.jsx` hands `escalation.id` to
    `onNavigate('esc', ...)` and `tab_escalations.jsx` resolves it with
    `row.id === id`; if the id spaces diverge the link dead-ends with both
    payloads looking fully populated and the whole suite green.

    Membership is the live exposure, not type.  `_index_escalations` filters
    hard — non-`eval_regression` categories, closed statuses and duplicate
    fingerprints are all dropped — while `shape_escalations` filters nothing,
    so MEMORY_EVALS ids are a strict SUBSET of ESCALATIONS rows and that
    containment is exactly why every link resolves today.  A filter added to
    `shape_escalations`, or a category/status widening in `_index_escalations`,
    breaks every link silently.  This is the test that would notice.

    ANTI-VACUITY.  The assertion is `violations == []`, which an empty payload,
    a walker aimed at the wrong keys or a fixture that reached no escalation
    would all satisfy.  So the reach paths are counted and every one of the
    three is required to contribute, and the `reconciliation` subsection is
    required to be non-empty — a MEMORY_EVALS id can only resolve against rows
    that are actually there.  Type agreement rides along inside the checker,
    which compares `type()` rather than `==`: both sides read the same JSON
    today, so only a `type()` check can see a future `str()` coercion on one
    side (`shape_escalations` already applies one to the sibling `task_id`).
    """
    trees = (
        ('no-storm', build_dual_escalation_tree(tmp_path / 'no-storm')),
        ('storm', build_dual_escalation_tree(tmp_path / 'storm', storm=True)),
    )

    reached: dict[str, list[str]] = {kind: [] for kind in _REACH_PATHS}
    for label, tree in trees:
        memory_evals = _build_memory_evals(tree.config)
        escalations = _build_escalations(tree.config)

        reconciliation = [s for s in escalations['subsections'] if s['id'] == 'reconciliation']
        assert len(reconciliation) == 1, (
            f'[{label}] expected exactly one reconciliation subsection, got '
            f'{[s["id"] for s in escalations["subsections"]]} — the memory-eval '
            'escalations live in that one, so anything else means the consumer '
            'side is not being built from this fixture at all.'
        )
        assert reconciliation[0]['escalations'], (
            f'[{label}] the reconciliation subsection carries NO rows, so every id '
            'below would have nothing to resolve against and the contract assertion '
            'would pass by being asked nothing.'
        )

        for path, _escalation in _iter_memory_eval_escalations(memory_evals):
            reached[_reach_kind(path)].append(f'{label}:{path}')

        assert collect_escalation_id_violations(memory_evals, escalations) == [], (
            f'[{label}] MEMORY_EVALS emits escalation ids that ESCALATIONS cannot '
            'resolve. Each violation names the payload path, the id and whether the '
            'row is missing outright or present at a different type:\n'
            + '\n'.join(
                f'  - {v["detail"]}'
                for v in collect_escalation_id_violations(memory_evals, escalations)
            )
        )

    for kind in _REACH_PATHS:
        assert reached[kind], (
            f'no escalation reached MEMORY_EVALS via the {kind!r} path across either '
            f'tree, so the contract above was never checked there. Reached: {reached}. '
            'Extend build_dual_escalation_tree rather than dropping this precondition '
            '— an unexercised reach path is exactly where the id space drifts unseen.'
        )


def test_the_id_space_subset_direction_is_asymmetric_by_design(tmp_path: Path) -> None:
    """MEMORY_EVALS ⊂ ESCALATIONS — strictly, in one direction, on purpose.

    The containment is not a coincidence the previous test happens to observe;
    it is the reason every link resolves, and it is one-way by construction:

    * `_index_escalations` filters HARD — it drops every record whose
      `category` is not `eval_regression`, every closed status
      (resolved/dismissed), and every duplicate fingerprint.
    * `shape_escalations` filters NOTHING — the reconciliation subsection ships
      the whole queue, `{**esc, ...}`.

    So ESCALATIONS is the superset by design, and asserting the reverse
    direction would be WRONG: it would fail on the first resolved escalation an
    operator closes, which is an ordinary Tuesday and not a defect.

    This test exists so that a future move toward symmetry — a filter added to
    `shape_escalations`, a category or status widening in `_index_escalations`
    — is a DELIBERATE decision that has to come here and change an assertion,
    rather than something that silently redefines what the previous test means.
    Both halves are asserted: the non-linkable records really do reach the
    consumer payload (so this is a live asymmetry, not an empty claim), and
    they really are unreachable from the producer side.
    """
    tree = build_dual_escalation_tree(tmp_path)
    memory_evals = _build_memory_evals(tree.config)
    escalations = _build_escalations(tree.config)

    row_ids = {
        row.get('id')
        for subsection in escalations['subsections']
        for row in subsection['escalations']
    }
    reached_ids = {
        escalation.get('id')
        for _path, escalation in _iter_memory_eval_escalations(memory_evals)
    }

    # Anti-vacuity: the filter must be discarding SELECTIVELY, not discarding
    # everything.  Without this, a totally broken `_index_escalations` that
    # dropped every record would satisfy every assertion below.
    assert tree.linked_id in reached_ids, (
        f'the linked record {tree.linked_id!r} is not reachable from MEMORY_EVALS, so '
        'the "never reachable" assertions below prove nothing — they would hold for a '
        'producer that dropped every escalation it was given.'
    )

    for what, esc_id in (
        ('a CLOSED (resolved) eval_regression', tree.resolved_id),
        ('an OPEN record of another category', tree.other_category_id),
    ):
        assert esc_id in row_ids, (
            f'{what} escalation {esc_id!r} is missing from ESCALATIONS. '
            '`shape_escalations` filters nothing, so it must ship the whole queue — '
            'if it has started filtering, the containment this module relies on is '
            'no longer guaranteed and the id space can drift.'
        )
        assert esc_id not in reached_ids, (
            f'{what} escalation {esc_id!r} IS reachable from MEMORY_EVALS. '
            '`_index_escalations` is supposed to drop it: a closed alarm must not '
            "render as open, and a non-eval_regression record is not this view's "
            'subject matter.'
        )

    assert reached_ids < row_ids, (
        'MEMORY_EVALS escalation ids must be a STRICT subset of ESCALATIONS row ids.\n'
        f'  only in MEMORY_EVALS: {sorted(map(str, reached_ids - row_ids))}\n'
        f'  only in ESCALATIONS:  {sorted(map(str, row_ids - reached_ids))}\n'
        'A non-empty first line is the contract break the previous test names. An '
        'EMPTY second line means the two id spaces have become equal — which is not '
        'a break, but it does mean this test is no longer observing the asymmetry it '
        'was written to pin, and the filters above should be re-read before it is '
        'relaxed into an ordinary subset check.'
    )

    assert collect_escalation_id_violations(memory_evals, escalations) == [], (
        'the checker must enforce MEMORY_EVALS ⊆ ESCALATIONS only. Rows that exist '
        'solely on the consumer side are the DESIGNED state, and reporting them '
        'would make the contract test fire on every escalation an operator closes.'
    )
