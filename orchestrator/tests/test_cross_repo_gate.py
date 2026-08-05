"""Tests for orchestrator.cross_repo_gate — dispatch-time cross-repo admission gate.

Covers: carries_cross_repo_signal (dispatch predicate), classify_cross_repo
(marker / files / scope-mismatch legs + degenerate metadata), and the
harness-level _run_cross_repo_gate / _block_and_escalate_cross_repo /
_run_slot wiring.

Task 3121 — block + escalate a foreign-owned task BEFORE agent spin-up,
beside the D4 substrate gate.  Incident shape: reify-5638 (a task whose
declared files are all owned by another project reaches the architect,
burns an agent, and lands an L2).

Scaffolding (make_*/_make_harness/_make_slot_harness/_make_mock_workflow) is
copied from test_substrate_gate.py rather than imported, matching this repo's
per-test-file helper convention.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Sentinel: distinguishes "metadata key absent" from "metadata is None".
_UNSET = object()


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------


def make_cross_repo_task(
    *,
    task_id: str = '3121',
    title: str = 'Cross-repo candidate',
    metadata: Any = _UNSET,
    metadata_as_json_string: bool = False,
    **meta_keys: Any,
) -> dict:
    """Return a task dict for cross-repo gate tests.

    Mirrors ``make_probe_task`` (test_substrate_gate.py:45).

    Args:
        task_id: value for ``task['id']``.
        title: value for ``task['title']``.
        metadata: used VERBATIM as ``task['metadata']`` when supplied — pass
            ``None``, a list, an int, or a raw string here to exercise the
            degenerate shapes.  Left at ``_UNSET`` (the default) the metadata
            dict is assembled from ``**meta_keys``; passing NO ``meta_keys``
            either then yields a task with NO ``metadata`` key at all.
        metadata_as_json_string: when True the assembled metadata dict is
            ``json.dumps``-ed, reproducing the fused-memory wire format where
            metadata arrives as a JSON string.
        **meta_keys: keys merged into the assembled metadata dict.
    """
    task: dict[str, Any] = {'id': task_id, 'title': title}

    if metadata is not _UNSET:
        task['metadata'] = metadata
        return task

    if not meta_keys:
        # No metadata key at all — the "absent metadata" shape.
        return task

    task['metadata'] = json.dumps(meta_keys) if metadata_as_json_string else dict(meta_keys)
    return task


# ---------------------------------------------------------------------------
# step-1 RED: carries_cross_repo_signal — the dispatch predicate
# ---------------------------------------------------------------------------


class TestCarriesCrossRepoSignal:
    """The dispatch predicate is pure KEY-presence, not value validation.

    Deliberate, and the lesson task 2121 already paid for on the substrate
    gate: a malformed marker must ENTER the gate and fail closed there, not
    skip the gate entirely.  See ``substrate_gate.carries_substrate_probe``.
    """

    # ---- True: a signal is present -------------------------------------

    def test_true_for_truthy_cross_repo_marker(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        assert carries_cross_repo_signal(make_cross_repo_task(cross_repo=True)) is True

    def test_true_for_string_cross_repo_marker(self):
        """The observed real-world spelling is an untyped, caller-authored string."""
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        assert carries_cross_repo_signal(
            make_cross_repo_task(cross_repo='dark-factory')
        ) is True

    def test_true_for_falsy_cross_repo_marker(self):
        """KEY-presence, not truth: a malformed marker must reach the gate."""
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        for falsy in (False, 0, '', None):
            task = make_cross_repo_task(cross_repo=falsy)
            assert carries_cross_repo_signal(task) is True, (
                f'cross_repo={falsy!r} must still enter the gate (key-presence), '
                f'so a malformed marker fails CLOSED inside classify rather than '
                f'skipping the gate entirely'
            )

    def test_true_for_possible_scope_mismatch_key(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task(
            possible_scope_mismatch={'source': 'prose', 'matched_paths': ['a/b.py']}
        )
        assert carries_cross_repo_signal(task) is True

    def test_true_for_falsy_possible_scope_mismatch_key(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        assert carries_cross_repo_signal(
            make_cross_repo_task(possible_scope_mismatch=None)
        ) is True

    def test_true_for_non_empty_files(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task(files=['/home/leo/src/other/foo.py'])
        assert carries_cross_repo_signal(task) is True

    def test_true_for_json_string_metadata_carrying_marker(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task(cross_repo=True, metadata_as_json_string=True)
        assert isinstance(task['metadata'], str), 'builder must produce a JSON string here'
        assert carries_cross_repo_signal(task) is True

    def test_true_for_json_string_metadata_carrying_files(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task(
            files=['/home/leo/src/other/foo.py'], metadata_as_json_string=True
        )
        assert carries_cross_repo_signal(task) is True

    def test_true_for_json_string_metadata_carrying_scope_mismatch(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task(
            possible_scope_mismatch={'suggested_project': 'dark_factory'},
            metadata_as_json_string=True,
        )
        assert carries_cross_repo_signal(task) is True

    # ---- False: no signal to act on -------------------------------------

    def test_false_for_absent_metadata(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task()
        assert 'metadata' not in task, 'builder must omit the key entirely here'
        assert carries_cross_repo_signal(task) is False

    def test_false_for_none_metadata(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        assert carries_cross_repo_signal(make_cross_repo_task(metadata=None)) is False

    def test_false_for_empty_metadata(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        assert carries_cross_repo_signal(make_cross_repo_task(metadata={})) is False

    def test_false_for_unrelated_keys_only(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task(task_kind='deterministic', milestone='2026-09-01')
        assert carries_cross_repo_signal(task) is False

    def test_false_for_empty_files_list(self):
        """An empty files list carries no path evidence — nothing for the gate to weigh."""
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        assert carries_cross_repo_signal(make_cross_repo_task(files=[])) is False

    def test_false_for_files_non_list_empty_shapes(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        for empty in ('', None, {}, ()):
            assert carries_cross_repo_signal(make_cross_repo_task(files=empty)) is False, (
                f'files={empty!r} carries no path evidence'
            )

    def test_false_for_non_dict_metadata(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        for bad in ([], ['cross_repo'], 42, 3.5, True):
            assert carries_cross_repo_signal(make_cross_repo_task(metadata=bad)) is False, (
                f'metadata={bad!r} exposes no readable keys'
            )

    def test_false_for_unparseable_json_string_metadata(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        task = make_cross_repo_task(metadata='{not valid json')
        assert carries_cross_repo_signal(task) is False

    def test_false_for_json_string_decoding_to_non_dict(self):
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        for payload in ('[1, 2, 3]', '"cross_repo"', '42', 'null'):
            task = make_cross_repo_task(metadata=payload)
            assert carries_cross_repo_signal(task) is False, (
                f'metadata={payload!r} decodes to a non-dict — no readable keys'
            )

    def test_never_raises_on_degenerate_input(self):
        """The predicate runs on every dispatch — it must never take down a slot."""
        from orchestrator.cross_repo_gate import carries_cross_repo_signal

        for bad in (None, [], 0, '', '{}', '[]', {'metadata': object()}):
            carries_cross_repo_signal(make_cross_repo_task(metadata=bad))


# ---------------------------------------------------------------------------
# step-3 RED: classify_cross_repo leg (A) — the metadata.cross_repo marker
# ---------------------------------------------------------------------------


class TestClassifyMarkerLeg:
    """Leg (A): a truthy ``metadata.cross_repo`` marker blocks, and names an owner.

    The markers observed in the wild are caller-authored and unvalidated, so
    every spelling below is a real shape this leg must handle, and the owner
    resolution must never INVENT a project name it cannot source.
    """

    # ---- the marker blocks, in every observed spelling -------------------

    def test_boolean_marker_blocks(self, tmp_path):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(cross_repo=True), project_root=tmp_path
        )
        assert verdict.verdict == BLOCK
        assert 'cross_repo_marker' in verdict.signals

    def test_bare_project_name_marker_blocks(self, tmp_path):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(cross_repo='dark-factory'), project_root=tmp_path
        )
        assert verdict.verdict == BLOCK
        assert 'cross_repo_marker' in verdict.signals

    def test_project_colon_path_marker_blocks(self, tmp_path):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                cross_repo='dark-factory:orchestrator/src/orchestrator/offline_lane.py'
            ),
            project_root=tmp_path,
        )
        assert verdict.verdict == BLOCK
        assert 'cross_repo_marker' in verdict.signals

    def test_marker_blocks_through_json_string_metadata(self, tmp_path):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(cross_repo=True, metadata_as_json_string=True),
            project_root=tmp_path,
        )
        assert verdict.verdict == BLOCK

    # ---- owner resolution precedence ------------------------------------

    def test_cross_repo_project_wins_over_string_marker(self, tmp_path):
        """The typed companion is authoritative when both are present."""
        from orchestrator.cross_repo_gate import classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                cross_repo='dark-factory', cross_repo_project='dark_factory'
            ),
            project_root=tmp_path,
        )
        assert verdict.owner_project == 'dark_factory'

    def test_string_marker_used_when_no_companion(self, tmp_path):
        from orchestrator.cross_repo_gate import classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(cross_repo='dark-factory'), project_root=tmp_path
        )
        assert verdict.owner_project == 'dark-factory'

    def test_string_marker_takes_pre_colon_field(self, tmp_path):
        """'project:path' spellings must yield the project, not the whole string."""
        from orchestrator.cross_repo_gate import classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                cross_repo='dark-factory:orchestrator/src/orchestrator/offline_lane.py'
            ),
            project_root=tmp_path,
        )
        assert verdict.owner_project == 'dark-factory'

    def test_scope_mismatch_suggested_project_is_last_fallback(self, tmp_path):
        from orchestrator.cross_repo_gate import classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                cross_repo=True,
                possible_scope_mismatch={'suggested_project': 'dark_factory'},
            ),
            project_root=tmp_path,
        )
        assert verdict.owner_project == 'dark_factory'

    def test_companion_wins_over_scope_mismatch_suggestion(self, tmp_path):
        from orchestrator.cross_repo_gate import classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                cross_repo=True,
                cross_repo_project='reify',
                possible_scope_mismatch={'suggested_project': 'dark_factory'},
            ),
            project_root=tmp_path,
        )
        assert verdict.owner_project == 'reify'

    def test_boolean_marker_alone_leaves_owner_unresolved(self, tmp_path):
        """Must NOT invent a name — an unresolved owner is reported as None."""
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(cross_repo=True), project_root=tmp_path
        )
        assert verdict.verdict == BLOCK, 'still blocks — the marker is the evidence'
        assert verdict.owner_project is None, (
            f'owner must be None when nothing names it; got {verdict.owner_project!r}'
        )

    def test_empty_and_non_string_companions_do_not_resolve_an_owner(self, tmp_path):
        from orchestrator.cross_repo_gate import classify_cross_repo

        for companion in ('', '   ', 42, [], {}, None):
            verdict = classify_cross_repo(
                task=make_cross_repo_task(cross_repo=True, cross_repo_project=companion),
                project_root=tmp_path,
            )
            assert verdict.owner_project is None, (
                f'cross_repo_project={companion!r} names no project; got '
                f'{verdict.owner_project!r} — a placeholder owner is worse than None'
            )

    # ---- a falsy marker is not evidence ---------------------------------

    def test_falsy_marker_does_not_block_on_leg_a_alone(self, tmp_path):
        """Key-presence gets it INTO the gate; only a truthy value blocks."""
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        for falsy in (False, 0, '', None):
            verdict = classify_cross_repo(
                task=make_cross_repo_task(cross_repo=falsy), project_root=tmp_path
            )
            assert verdict.verdict != BLOCK, (
                f'cross_repo={falsy!r} is not evidence of foreign ownership'
            )
            assert 'cross_repo_marker' not in verdict.signals

    # ---- verdict shape ---------------------------------------------------

    def test_verdict_is_frozen(self, tmp_path):
        import dataclasses

        from orchestrator.cross_repo_gate import CrossRepoVerdict, classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(cross_repo=True), project_root=tmp_path
        )
        assert isinstance(verdict, CrossRepoVerdict)
        assert dataclasses.is_dataclass(verdict)
        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.verdict = 'allow'  # type: ignore[misc]

    def test_blocked_property_tracks_verdict(self):
        from orchestrator.cross_repo_gate import ALLOW, BLOCK, SKIP, CrossRepoVerdict

        for value, expected in ((BLOCK, True), (ALLOW, False), (SKIP, False)):
            verdict = CrossRepoVerdict(
                verdict=value,
                owner_project=None,
                signals=(),
                foreign_paths=(),
                reason='',
            )
            assert verdict.blocked is expected, f'{value!r}.blocked should be {expected}'

    def test_allow_when_no_marker_and_no_other_evidence(self, tmp_path):
        from orchestrator.cross_repo_gate import ALLOW, classify_cross_repo

        verdict = classify_cross_repo(
            task=make_cross_repo_task(task_kind='deterministic'), project_root=tmp_path
        )
        assert verdict.verdict == ALLOW
        assert verdict.signals == ()
        assert verdict.blocked is False


# ---------------------------------------------------------------------------
# step-5 RED: classify_cross_repo leg (B) — all-foreign metadata.files
# ---------------------------------------------------------------------------


@pytest.fixture
def two_roots(tmp_path):
    """Return (project_root, foreign_root) as real, sibling directories.

    Both are created under ``tmp_path`` so no test depends on the machine's
    real project layout, and so ``Path.resolve()`` behaves identically for
    both sides of the containment comparison.
    """
    project_root = tmp_path / 'reify'
    foreign_root = tmp_path / 'dark-factory'
    project_root.mkdir()
    foreign_root.mkdir()
    return project_root, foreign_root


class TestClassifyFilesLeg:
    """Leg (B): every declared ``metadata.files`` entry resolves outside project_root.

    This is the reify-5638 shape — the one the orchestrator CAN classify
    unaided, by path containment, with no cross-project registry.  Delegated to
    ``merge_gates.is_cross_repo_task`` so dispatch time and merge time share one
    definition of "foreign"; the conservatism pinned below (empty → no block,
    any relative entry → no block, any in-tree entry → no block) is that
    function's documented contract.
    """

    def test_all_foreign_absolute_files_block(self, two_roots):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, foreign_root = two_roots
        files = [
            str(foreign_root / 'orchestrator/src/orchestrator/harness.py'),
            str(foreign_root / 'orchestrator/tests/test_harness.py'),
        ]
        verdict = classify_cross_repo(
            task=make_cross_repo_task(files=files), project_root=project_root
        )

        assert verdict.verdict == BLOCK
        assert 'all_files_foreign' in verdict.signals
        assert tuple(verdict.foreign_paths) == tuple(files), (
            f'foreign_paths must echo the offending entries so the L1 is '
            f'actionable; got {verdict.foreign_paths!r}'
        )

    def test_files_leg_alone_leaves_owner_unresolved(self, two_roots):
        """Path containment proves foreignness; it does NOT name the owner."""
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(files=[str(foreign_root / 'a.py')]),
            project_root=project_root,
        )

        assert verdict.verdict == BLOCK
        assert verdict.owner_project is None, (
            'the orchestrator has no cross-project registry — it must not guess '
            f'a project name from a path; got {verdict.owner_project!r}'
        )

    def test_relative_in_tree_files_allow(self, two_roots):
        """A still-undelivered NEW local file is exactly this shape."""
        from orchestrator.cross_repo_gate import ALLOW, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                files=['orchestrator/src/orchestrator/cross_repo_gate.py']
            ),
            project_root=project_root,
        )
        assert verdict.verdict == ALLOW
        assert verdict.blocked is False

    def test_mixed_foreign_and_relative_files_allow(self, two_roots):
        """A MIX is not an all-foreign deliverable — conservative by design."""
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                files=[str(foreign_root / 'a.py'), 'orchestrator/src/local.py']
            ),
            project_root=project_root,
        )
        assert verdict.blocked is False

    def test_mixed_foreign_and_in_tree_absolute_files_allow(self, two_roots):
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                files=[str(foreign_root / 'a.py'), str(project_root / 'b.py')]
            ),
            project_root=project_root,
        )
        assert verdict.blocked is False

    def test_absolute_in_tree_files_allow(self, two_roots):
        from orchestrator.cross_repo_gate import ALLOW, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                files=[str(project_root / 'orchestrator/src/orchestrator/harness.py')]
            ),
            project_root=project_root,
        )
        assert verdict.verdict == ALLOW

    def test_empty_files_allow(self, two_roots):
        from orchestrator.cross_repo_gate import ALLOW, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(files=[]), project_root=project_root
        )
        assert verdict.verdict == ALLOW

    def test_absent_files_allow(self, two_roots):
        from orchestrator.cross_repo_gate import ALLOW, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(task_kind='deterministic'),
            project_root=project_root,
        )
        assert verdict.verdict == ALLOW

    def test_marker_still_blocks_when_files_empty(self, two_roots):
        """Leg B's empty-list conservatism must not swallow leg A's marker.

        ``is_cross_repo_task`` returns False for an empty ``plan_files`` BEFORE
        it ever reads the marker (merge_gates.py:997), which is exactly why
        leg A reads ``metadata.cross_repo`` directly instead of delegating.
        """
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(cross_repo=True, files=[]),
            project_root=project_root,
        )
        assert verdict.verdict == BLOCK
        assert 'cross_repo_marker' in verdict.signals

    def test_both_legs_fire_signals_are_unioned(self, two_roots):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                cross_repo=True,
                cross_repo_project='dark_factory',
                files=[str(foreign_root / 'a.py')],
            ),
            project_root=project_root,
        )
        assert verdict.verdict == BLOCK
        assert 'cross_repo_marker' in verdict.signals
        assert 'all_files_foreign' in verdict.signals
        assert verdict.owner_project == 'dark_factory'

    def test_files_leg_blocks_through_json_string_metadata(self, two_roots):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                files=[str(foreign_root / 'a.py')], metadata_as_json_string=True
            ),
            project_root=project_root,
        )
        assert verdict.verdict == BLOCK

    def test_project_root_accepts_a_string(self, two_roots):
        """Callers may pass either a Path or a str — coerced at the boundary."""
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(files=[str(foreign_root / 'a.py')]),
            project_root=str(project_root),
        )
        assert verdict.verdict == BLOCK

    def test_non_string_and_empty_file_entries_are_ignored(self, two_roots):
        """Junk entries must not be mistaken for local paths that veto the block."""
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                files=[str(foreign_root / 'a.py'), '', None, 42]
            ),
            project_root=project_root,
        )
        assert verdict.verdict == BLOCK
        assert tuple(verdict.foreign_paths) == (str(foreign_root / 'a.py'),)

    def test_files_not_a_list_allows(self, two_roots):
        """A malformed files value carries no usable path evidence."""
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, foreign_root = two_roots
        for bad in (str(foreign_root / 'a.py'), {'a': 1}, 42):
            verdict = classify_cross_repo(
                task=make_cross_repo_task(files=bad), project_root=project_root
            )
            assert verdict.blocked is False, f'files={bad!r} should not block leg B'


# ---------------------------------------------------------------------------
# step-7 RED: leg (C) — a CONTAINMENT-CONFIRMED possible_scope_mismatch stamp
# ---------------------------------------------------------------------------


class TestClassifyScopeMismatchLeg:
    """Leg (C): the advisory stamp blocks ONLY when path evidence confirms it.

    ``possible_scope_mismatch`` is written by the fused-memory prose matcher
    (``'source': 'prose'``) and is documented to over-fire — task 3120 landed a
    right-boundary fix for exactly that and records a KNOWN FAIL-OPEN residue.
    Blocking dispatch on unconfirmed prose would convert a false-positive
    advisory into a stalled task PLUS a spurious L1: strictly worse than today.
    Requiring containment confirmation makes leg C a genuine second evidence
    source rather than a restatement of the heuristic.
    """

    def test_confirmed_stamp_blocks(self, two_roots):
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                possible_scope_mismatch={
                    'source': 'prose',
                    'suggested_project': 'dark_factory',
                    'matched_paths': [
                        str(foreign_root / 'orchestrator/src/orchestrator/offline_lane.py'),
                    ],
                }
            ),
            project_root=project_root,
        )

        assert verdict.verdict == BLOCK
        assert 'scope_mismatch_confirmed' in verdict.signals
        assert verdict.owner_project == 'dark_factory'

    def test_confirmed_stamp_reports_the_matched_paths(self, two_roots):
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, foreign_root = two_roots
        paths = [str(foreign_root / 'a.py'), str(foreign_root / 'b.py')]
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                possible_scope_mismatch={'matched_paths': paths}
            ),
            project_root=project_root,
        )
        assert verdict.blocked is True
        for path in paths:
            assert path in verdict.foreign_paths

    def test_prose_shaped_relative_matched_paths_do_not_block(self, two_roots):
        """The over-firing case: an UNCONFIRMED prose advisory must not block."""
        from orchestrator.cross_repo_gate import ALLOW, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                possible_scope_mismatch={
                    'source': 'prose',
                    'suggested_project': 'dark_factory',
                    'matched_paths': ['orchestrator/src/orchestrator/offline_lane.py'],
                }
            ),
            project_root=project_root,
        )
        assert verdict.verdict == ALLOW, (
            'an unconfirmed prose advisory must not strand a legitimate task'
        )
        assert verdict.blocked is False

    def test_prose_noise_matched_paths_do_not_block(self, two_roots):
        """The literal task-3120 over-fire strings must not be read as evidence."""
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                possible_scope_mismatch={
                    'matched_paths': ['tools/call', 'archive/pause', 'and/or'],
                }
            ),
            project_root=project_root,
        )
        assert verdict.blocked is False

    def test_in_tree_matched_paths_do_not_block(self, two_roots):
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                possible_scope_mismatch={'matched_paths': [str(project_root / 'a.py')]}
            ),
            project_root=project_root,
        )
        assert verdict.blocked is False

    def test_mixed_matched_paths_do_not_block(self, two_roots):
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, foreign_root = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                possible_scope_mismatch={
                    'matched_paths': [str(foreign_root / 'a.py'), 'local/b.py'],
                }
            ),
            project_root=project_root,
        )
        assert verdict.blocked is False

    def test_non_dict_stamp_does_not_block_or_crash(self, two_roots):
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, _ = two_roots
        for stamp in (None, True, 'dark_factory', ['a'], 42):
            verdict = classify_cross_repo(
                task=make_cross_repo_task(possible_scope_mismatch=stamp),
                project_root=project_root,
            )
            assert verdict.blocked is False, f'stamp={stamp!r} is not confirmable'

    def test_stamp_without_matched_paths_does_not_block(self, two_roots):
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, _ = two_roots
        for stamp in ({}, {'source': 'prose'}, {'matched_paths': []},
                      {'matched_paths': None}, {'matched_paths': 'a/b.py'}):
            verdict = classify_cross_repo(
                task=make_cross_repo_task(possible_scope_mismatch=stamp),
                project_root=project_root,
            )
            assert verdict.blocked is False, f'stamp={stamp!r} carries no path evidence'

    def test_unconfirmed_stamp_does_not_veto_the_marker(self, two_roots):
        """Leg C being unconfirmed must not suppress an independent leg-A block."""
        from orchestrator.cross_repo_gate import BLOCK, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(
                cross_repo='dark-factory',
                possible_scope_mismatch={'matched_paths': ['tools/call']},
            ),
            project_root=project_root,
        )
        assert verdict.verdict == BLOCK
        assert 'cross_repo_marker' in verdict.signals
        assert 'scope_mismatch_confirmed' not in verdict.signals


# ---------------------------------------------------------------------------
# step-7 RED: degenerate metadata is a LOUD SKIP, never a silent ALLOW
# ---------------------------------------------------------------------------


class TestClassifyDegenerateMetadata:
    """Unreadable metadata yields SKIP + a WARNING.

    SKIP means "no evidence readable", NEVER "verified clean".  Collapsing it
    into a silent ALLOW is the no-silent-fail-soft violation this gate exists
    to avoid — a task whose markers could not be parsed would be waved through
    with no trace that anything was skipped.
    """

    @pytest.mark.parametrize('bad', [[], ['cross_repo'], 42, 3.5, True, '{not valid json', '[1,2]', '"x"', 'null'])
    def test_degenerate_metadata_is_skip(self, bad, two_roots):
        from orchestrator.cross_repo_gate import SKIP, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(metadata=bad), project_root=project_root
        )
        assert verdict.verdict == SKIP, (
            f'metadata={bad!r} is unreadable — SKIP, never a silent ALLOW'
        )
        assert verdict.blocked is False, 'SKIP must not block dispatch'

    @pytest.mark.parametrize('bad', [[], 42, '{not valid json', '[1,2]'])
    def test_degenerate_metadata_warns(self, bad, two_roots, caplog):
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, _ = two_roots
        caplog.set_level(logging.WARNING, logger='orchestrator.cross_repo_gate')
        classify_cross_repo(
            task=make_cross_repo_task(task_id='9001', metadata=bad),
            project_root=project_root,
        )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, f'metadata={bad!r} must warn, not degrade silently'
        message = ' '.join(r.getMessage() for r in warnings)
        assert '9001' in message, f'warning must name the task id; got {message!r}'
        assert type(bad).__name__ in message, (
            f'warning must name the discarded type; got {message!r}'
        )

    def test_absent_metadata_is_skip(self, two_roots):
        """Unreachable in production — carries_cross_repo_signal gates this out.

        Pinned so the behaviour is deliberate rather than accidental: with no
        metadata there is genuinely nothing to read, which is SKIP's meaning.
        """
        from orchestrator.cross_repo_gate import SKIP, classify_cross_repo

        project_root, _ = two_roots
        verdict = classify_cross_repo(
            task=make_cross_repo_task(), project_root=project_root
        )
        assert verdict.verdict == SKIP
        assert verdict.blocked is False

    def test_never_raises_on_any_metadata_shape(self, two_roots):
        """classify runs on the dispatch path — it must never take down a slot."""
        from orchestrator.cross_repo_gate import classify_cross_repo

        project_root, foreign_root = two_roots
        shapes: list[Any] = [
            None, [], {}, 42, '', '{}', '[]', 'null',
            {'cross_repo': object()},
            {'files': object()},
            {'files': [object()]},
            {'possible_scope_mismatch': {'matched_paths': [object()]}},
            {'cross_repo': True, 'cross_repo_project': object()},
            {'files': [str(foreign_root / 'a.py')], 'possible_scope_mismatch': object()},
        ]
        for shape in shapes:
            verdict = classify_cross_repo(
                task=make_cross_repo_task(metadata=shape), project_root=project_root
            )
            assert verdict.verdict in ('block', 'allow', 'skip')


# ---------------------------------------------------------------------------
# step-9 RED: Harness._block_and_escalate_cross_repo
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path, project_root: Path | None = None):
    """Build a bare Harness with mocked internals for cross-repo gate tests.

    Copied from test_substrate_gate.py:777 (per-test-file helper convention),
    with ``project_root`` made overridable: the containment leg is meaningless
    unless the harness's project_root and the task's declared paths can be
    placed in a deliberate relationship.
    """
    from orchestrator.config import OrchestratorConfig
    from orchestrator.harness import Harness

    config = OrchestratorConfig(project_root=project_root or tmp_path, max_per_module=1)
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(config)

    h.scheduler = MagicMock()
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.is_deterministic = MagicMock(return_value=False)

    h.git_ops = MagicMock()
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' * 5)
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.git_ops.project_root = tmp_path
    h.git_ops.prune_worktrees = AsyncMock(return_value=None)

    # No escalation queue by default — tests that need one attach it explicitly.
    h._escalation_queue = None

    return h


def _block_verdict(owner: str | None = 'dark_factory', paths: tuple[str, ...] = ()):
    from orchestrator.cross_repo_gate import BLOCK, CrossRepoVerdict

    return CrossRepoVerdict(
        verdict=BLOCK,
        owner_project=owner,
        signals=('cross_repo_marker', 'all_files_foreign'),
        foreign_paths=paths or ('/home/leo/src/dark-factory/orchestrator/src/x.py',),
        reason='all 1 declared metadata.files entries resolve outside project_root',
    )


class TestBlockAndEscalateCrossRepo:
    """Unit tests for ``Harness._block_and_escalate_cross_repo``.

    Modelled on TestBlockAndEscalateSubstrateFlip (test_substrate_gate.py:786).
    """

    @pytest.mark.asyncio
    async def test_sets_task_blocked(self, tmp_path: Path):
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')

        await h._block_and_escalate_cross_repo('99', verdict=_block_verdict())

        h.scheduler.set_task_status.assert_awaited_once_with('99', 'blocked')

    @pytest.mark.asyncio
    async def test_files_one_l1_scope_violation(self, tmp_path: Path):
        """scope_violation, not design_concern: the work belongs to another project."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        await h._block_and_escalate_cross_repo('77', verdict=_block_verdict())

        l1s = [e for e in esc_queue.get_pending() if e.task_id == '77' and e.level == 1]
        assert len(l1s) == 1, f'Expected exactly 1 L1 for task 77; got {l1s!r}'
        esc = l1s[0]
        assert esc.category == 'scope_violation'
        assert esc.severity == 'blocking'
        assert esc.level == 1

    @pytest.mark.asyncio
    async def test_summary_names_the_owning_project(self, tmp_path: Path):
        """An L1 naming the owner is directly actionable; one that doesn't costs a triage round trip."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        await h._block_and_escalate_cross_repo(
            '66', verdict=_block_verdict(owner='dark_factory')
        )

        esc = [e for e in esc_queue.get_pending() if e.task_id == '66'][0]
        assert 'dark_factory' in esc.summary, (
            f'summary must name the owning project; got {esc.summary!r}'
        )
        assert len(esc.summary) <= 200, 'summary must be truncated to 200 chars'

    @pytest.mark.asyncio
    async def test_detail_carries_reason_signals_and_paths(self, tmp_path: Path):
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        verdict = _block_verdict(paths=('/home/leo/src/dark-factory/orchestrator/a.py',))
        await h._block_and_escalate_cross_repo('65', verdict=verdict)

        esc = [e for e in esc_queue.get_pending() if e.task_id == '65'][0]
        assert verdict.reason in esc.detail
        assert '/home/leo/src/dark-factory/orchestrator/a.py' in esc.detail
        for signal in verdict.signals:
            assert signal in esc.detail, f'detail must name the signal {signal!r}'
        assert 'refile' in esc.detail.lower(), (
            'detail must state the fix — refile the task under the owning project'
        )

    @pytest.mark.asyncio
    async def test_unresolved_owner_says_so_explicitly(self, tmp_path: Path):
        """No placeholder, no empty name — say the owner could not be resolved."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        await h._block_and_escalate_cross_repo('64', verdict=_block_verdict(owner=None))

        esc = [e for e in esc_queue.get_pending() if e.task_id == '64'][0]
        assert 'None' not in esc.summary, (
            f'must not leak a Python None into the L1 summary; got {esc.summary!r}'
        )
        assert 'unresolved' in esc.summary.lower(), (
            f'summary must say the owner is unresolved; got {esc.summary!r}'
        )
        assert 'unresolved' in esc.detail.lower()

    @pytest.mark.asyncio
    async def test_deduped_by_has_open_l1(self, tmp_path: Path):
        """Repeated dispatch after the requeue cooldown must not stack duplicates."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        await h._block_and_escalate_cross_repo('55', verdict=_block_verdict())
        await h._block_and_escalate_cross_repo('55', verdict=_block_verdict())

        l1s = [e for e in esc_queue.get_pending() if e.task_id == '55' and e.level == 1]
        assert len(l1s) == 1, f'Expected exactly 1 L1 after dedup; got {l1s!r}'

    @pytest.mark.asyncio
    async def test_noop_when_no_escalation_queue(self, tmp_path: Path):
        """Blocking the task is unconditional even with no queue to file into."""
        h = _make_harness(tmp_path)
        h._escalation_queue = None

        await h._block_and_escalate_cross_repo('33', verdict=_block_verdict())

        h.scheduler.set_task_status.assert_awaited_once_with('33', 'blocked')

    @pytest.mark.asyncio
    async def test_files_l1_even_when_set_task_status_raises(self, tmp_path: Path):
        """A transient status-write failure must not swallow the escalation."""
        from escalation.queue import EscalationQueue

        h = _make_harness(tmp_path)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue
        h.scheduler.set_task_status = AsyncMock(side_effect=RuntimeError('memory down'))

        await h._block_and_escalate_cross_repo('22', verdict=_block_verdict())

        l1s = [e for e in esc_queue.get_pending() if e.task_id == '22' and e.level == 1]
        assert len(l1s) == 1, (
            f'L1 must still be filed when set_task_status raises; got {l1s!r}'
        )


# ---------------------------------------------------------------------------
# step-11 RED: Harness._run_cross_repo_gate
# ---------------------------------------------------------------------------


def _make_assignment(task_id: str = '42', **meta_keys):
    """Build a minimal TaskAssignment-like object carrying cross-repo metadata."""
    task = make_cross_repo_task(task_id=task_id, **meta_keys)
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = task
    return assignment


def _allow_verdict():
    from orchestrator.cross_repo_gate import ALLOW, CrossRepoVerdict

    return CrossRepoVerdict(
        verdict=ALLOW, owner_project=None, signals=(), foreign_paths=(),
        reason='no cross-repo evidence in task metadata',
    )


def _skip_verdict():
    from orchestrator.cross_repo_gate import SKIP, CrossRepoVerdict

    return CrossRepoVerdict(
        verdict=SKIP, owner_project=None, signals=(), foreign_paths=(),
        reason='task metadata is not a readable dict (type=list)',
    )


class TestRunCrossRepoGate:
    """Unit tests for ``Harness._run_cross_repo_gate``.

    Unlike ``_run_substrate_gate`` this gate is PURE: no worktree, no
    subprocess, no thread offload.  Several tests below assert that property
    directly, so a later refactor cannot quietly make dispatch pay for it.
    """

    @pytest.mark.asyncio
    async def test_block_returns_false_and_escalates(self, tmp_path: Path):
        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()
        verdict = _block_verdict()

        with patch('orchestrator.cross_repo_gate.classify_cross_repo', return_value=verdict):
            allowed = await h._run_cross_repo_gate(_make_assignment(cross_repo=True))

        assert allowed is False
        h._block_and_escalate_cross_repo.assert_awaited_once()
        assert h._block_and_escalate_cross_repo.await_args.kwargs['verdict'] is verdict

    @pytest.mark.asyncio
    async def test_allow_returns_true_without_escalating(self, tmp_path: Path):
        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()

        with patch('orchestrator.cross_repo_gate.classify_cross_repo',
                   return_value=_allow_verdict()):
            allowed = await h._run_cross_repo_gate(_make_assignment(cross_repo=False))

        assert allowed is True
        h._block_and_escalate_cross_repo.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skip_returns_true_without_escalating(self, tmp_path: Path):
        """SKIP is not evidence of a violation — it must not block or escalate."""
        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()

        with patch('orchestrator.cross_repo_gate.classify_cross_repo',
                   return_value=_skip_verdict()):
            allowed = await h._run_cross_repo_gate(_make_assignment(cross_repo=True))

        assert allowed is True
        h._block_and_escalate_cross_repo.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_classify_raising_fails_closed(self, tmp_path: Path):
        """Mirrors _run_substrate_gate's except branch: unverifiable → block."""
        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()

        with patch('orchestrator.cross_repo_gate.classify_cross_repo',
                   side_effect=RuntimeError('registry exploded')):
            allowed = await h._run_cross_repo_gate(_make_assignment(cross_repo=True))

        assert allowed is False, 'an unverifiable classification must fail CLOSED'
        h._block_and_escalate_cross_repo.assert_awaited_once()
        verdict = h._block_and_escalate_cross_repo.await_args.kwargs['verdict']
        assert verdict.blocked is True
        assert 'registry exploded' in verdict.reason, (
            f'the synthesized verdict must name the exception; got {verdict.reason!r}'
        )

    @pytest.mark.asyncio
    async def test_passes_config_project_root(self, tmp_path: Path):
        """A wrong root would make every task look foreign, or none of them."""
        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()

        with patch('orchestrator.cross_repo_gate.classify_cross_repo',
                   return_value=_allow_verdict()) as classify:
            await h._run_cross_repo_gate(_make_assignment(cross_repo=True))

        classify.assert_called_once()
        assert classify.call_args.kwargs['project_root'] == h.config.project_root

    @pytest.mark.asyncio
    async def test_passes_the_assignment_task(self, tmp_path: Path):
        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()
        assignment = _make_assignment(cross_repo=True)

        with patch('orchestrator.cross_repo_gate.classify_cross_repo',
                   return_value=_allow_verdict()) as classify:
            await h._run_cross_repo_gate(assignment)

        assert classify.call_args.kwargs['task'] is assignment.task

    @pytest.mark.asyncio
    async def test_no_subprocess_and_no_worktree(self, tmp_path: Path):
        """The gate is pure and in-process — it must not build a worktree."""
        import asyncio as _asyncio

        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()

        with (
            patch.object(_asyncio, 'create_subprocess_exec') as spawn,
            patch('orchestrator.cross_repo_gate.classify_cross_repo',
                  return_value=_block_verdict()),
        ):
            await h._run_cross_repo_gate(_make_assignment(cross_repo=True))

        spawn.assert_not_called()
        h.git_ops.resolve_branch_sha.assert_not_awaited()
        assert not (tmp_path / '.worktrees').exists(), (
            'the cross-repo gate must not create any worktree'
        )

    @pytest.mark.asyncio
    async def test_logs_the_verdict_even_when_allowing(self, tmp_path: Path, caplog):
        """The gate must be observable in logs on every path, not just on block."""
        h = _make_harness(tmp_path)
        h._block_and_escalate_cross_repo = AsyncMock()
        caplog.set_level(logging.INFO, logger='orchestrator.harness')

        with patch('orchestrator.cross_repo_gate.classify_cross_repo',
                   return_value=_allow_verdict()):
            await h._run_cross_repo_gate(_make_assignment(task_id='4242', cross_repo=False))

        message = ' '.join(r.getMessage() for r in caplog.records)
        assert '4242' in message, f'gate must log the task id; got {message!r}'
        assert 'allow' in message.lower(), f'gate must log the verdict; got {message!r}'


# ---------------------------------------------------------------------------
# step-13 RED: dispatch-level wiring through _run_slot
# ---------------------------------------------------------------------------


def _make_slot_harness(tmp_path: Path, project_root: Path | None = None):
    """Build a Harness whose _run_slot can be called directly in tests.

    Copied from test_substrate_gate.py:857.
    """
    h = _make_harness(tmp_path, project_root=project_root)
    h.scheduler.release = MagicMock()
    h.scheduler._dispatched = set()
    h._run_id = None
    h.event_store = None
    return h


def _make_mock_workflow():
    """Return a minimal AsyncMock workflow whose run() is awaitable."""
    from orchestrator.workflow import TerminalReport, WorkflowOutcome, WorkflowState

    wf = AsyncMock()
    wf.run = AsyncMock(return_value=TerminalReport(
        outcome=WorkflowOutcome.DONE, reason='', phase=WorkflowState.DONE,
        detail='', category=None,
    ))
    wf.metrics = MagicMock(
        total_cost_usd=0.0,
        total_duration_ms=0,
        agent_invocations=0,
        execute_iterations=0,
        verify_attempts=0,
        review_cycles=0,
    )
    wf._steward = None
    return wf


_PROBE = {'checker': ['python', '-m', 'checker'], 'probe_set': 'probes/p.json'}


class TestRunSlotCrossRepoGateWiring:
    """_run_slot wiring for the cross-repo gate.

    (a) BLOCK  → build_workflow never called, cooldown armed, BLOCKED report.
    (b) ALLOW  → build_workflow IS called and workflow.run awaited.
    (c) no signal → gate never invoked; dispatch unaffected.
    (d) ORDERING → the cheap terminal gate runs BEFORE the substrate gate.
    (e) probe-only task still reaches the substrate gate (no regression).
    """

    @pytest.mark.asyncio
    async def test_block_prevents_workflow_construction(self, tmp_path: Path):
        """(a) No agent spins up: build_workflow is never called."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(cross_repo=True)
        h._run_cross_repo_gate = AsyncMock(return_value=False)
        h._block_and_escalate_cross_repo = AsyncMock()

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            MockWorkflow.return_value = _make_mock_workflow()
            report = await h._run_slot(assignment, sem)

        assert not MockWorkflow.called, (
            'TaskWorkflow must NOT be constructed for a foreign-owned task — '
            'no agent, no worktree, no run row'
        )
        h._run_cross_repo_gate.assert_awaited_once()

        assert report is not None
        from orchestrator.workflow import WorkflowOutcome

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.block_reason == 'cross_repo_misfile'

    @pytest.mark.asyncio
    async def test_block_arms_requeue_cooldown(self, tmp_path: Path):
        """(a) scheduler.release(requeued=True) so the task is not re-dispatched at once."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(cross_repo=True)
        h._run_cross_repo_gate = AsyncMock(return_value=False)
        h._block_and_escalate_cross_repo = AsyncMock()

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            MockWorkflow.return_value = _make_mock_workflow()
            await h._run_slot(assignment, sem)

        h.scheduler.release.assert_called_once()
        requeued = h.scheduler.release.call_args.kwargs.get('requeued', None)
        assert requeued is True, (
            f'scheduler.release must be called with requeued=True; got '
            f'{h.scheduler.release.call_args!r}'
        )

    @pytest.mark.asyncio
    async def test_allow_permits_workflow_construction(self, tmp_path: Path):
        """(b) A gate that allows must not disturb normal dispatch."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(cross_repo=True)
        h._run_cross_repo_gate = AsyncMock(return_value=True)

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = _make_mock_workflow()
            MockWorkflow.return_value = mock_wf
            await h._run_slot(assignment, sem)

        assert MockWorkflow.called
        mock_wf.run.assert_awaited_once()
        h._run_cross_repo_gate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_signal_skips_the_gate(self, tmp_path: Path):
        """(c) A task with no cross-repo signal pays nothing."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(task_kind='implementation')
        gate = AsyncMock(return_value=True)
        h._run_cross_repo_gate = gate

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = _make_mock_workflow()
            MockWorkflow.return_value = mock_wf
            await h._run_slot(assignment, sem)

        gate.assert_not_awaited()
        mock_wf.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cross_repo_gate_runs_before_substrate_gate(self, tmp_path: Path):
        """(d) The cheap terminal gate runs FIRST — no worktree for a doomed task."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(cross_repo=True, substrate_probe=_PROBE)
        h._run_cross_repo_gate = AsyncMock(return_value=False)
        h._block_and_escalate_cross_repo = AsyncMock()
        substrate_gate_mock = AsyncMock(return_value=True)
        h._run_substrate_gate = substrate_gate_mock

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            MockWorkflow.return_value = _make_mock_workflow()
            report = await h._run_slot(assignment, sem)

        # The substrate gate builds an ephemeral worktree + runs a checker
        # subprocess; a task the cross-repo gate already blocked must never
        # pay for it.
        substrate_gate_mock.assert_not_awaited()
        assert report is not None
        assert report.block_reason == 'cross_repo_misfile'
        assert not MockWorkflow.called

    @pytest.mark.asyncio
    async def test_probe_only_task_still_reaches_substrate_gate(self, tmp_path: Path):
        """(e) The insertion must not break the existing D4 gate."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(substrate_probe=_PROBE)
        cross_repo_gate_mock = AsyncMock(return_value=True)
        h._run_cross_repo_gate = cross_repo_gate_mock
        substrate_gate_mock = AsyncMock(return_value=False)
        h._run_substrate_gate = substrate_gate_mock
        h._block_and_escalate_substrate_flip = AsyncMock()

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            MockWorkflow.return_value = _make_mock_workflow()
            report = await h._run_slot(assignment, sem)

        cross_repo_gate_mock.assert_not_awaited()
        substrate_gate_mock.assert_awaited_once()
        assert report is not None
        assert report.block_reason == 'substrate_flip'

    @pytest.mark.asyncio
    async def test_both_gates_run_when_neither_blocks(self, tmp_path: Path):
        """A task carrying both signals passes through both gates, in order."""
        h = _make_slot_harness(tmp_path)
        assignment = _make_assignment(cross_repo=False, substrate_probe=_PROBE)
        order: list[str] = []
        h._run_cross_repo_gate = AsyncMock(
            side_effect=lambda _a: order.append('cross_repo') or True
        )
        h._run_substrate_gate = AsyncMock(
            side_effect=lambda _a: order.append('substrate') or True
        )

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = _make_mock_workflow()
            MockWorkflow.return_value = mock_wf
            await h._run_slot(assignment, sem)

        assert order == ['cross_repo', 'substrate'], (
            f'cross-repo gate must run first; got {order!r}'
        )
        mock_wf.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# step-17: the USER-OBSERVABLE SIGNAL, end to end through the real path
# ---------------------------------------------------------------------------


class TestCrossRepoAdmissionEndToEnd:
    """The task's stated user-observable signal, with nothing stubbed but build_workflow.

    Real ``carries_cross_repo_signal``, real ``classify_cross_repo``, real
    ``_run_cross_repo_gate``, real ``_block_and_escalate_cross_repo``, real
    ``EscalationQueue`` — only ``build_workflow`` is a spy, and only because it
    IS the assertion: no workflow means no agent, no worktree, and no run row,
    so "no architect invocation is recorded" is exactly "build_workflow was
    never called".
    """

    @pytest.mark.asyncio
    async def test_foreign_owned_task_is_blocked_before_any_agent(self, tmp_path: Path):
        from escalation.queue import EscalationQueue

        from orchestrator.workflow import WorkflowOutcome

        # The reify-5638 shape: this orchestrator serves 'reify', the task
        # declares only files under the sibling 'dark-factory' checkout.
        project_root = tmp_path / 'reify'
        foreign_root = tmp_path / 'dark-factory'
        project_root.mkdir()
        foreign_root.mkdir()

        h = _make_slot_harness(tmp_path, project_root=project_root)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        assignment = _make_assignment(
            task_id='5638',
            files=[
                str(foreign_root / 'orchestrator/src/orchestrator/offline_lane.py'),
                str(foreign_root / 'orchestrator/tests/test_offline_lane.py'),
            ],
            cross_repo_project='dark_factory',
        )

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            MockWorkflow.return_value = _make_mock_workflow()
            report = await h._run_slot(assignment, sem)

        # (1) No architect invocation: no workflow was ever constructed, so no
        #     agent spun up, no worktree was built, and no runs.db row exists.
        assert not MockWorkflow.called, (
            'build_workflow must never be called for a foreign-owned task'
        )

        # (2) The task reaches `blocked`.
        h.scheduler.set_task_status.assert_awaited_once_with('5638', 'blocked')

        # (3) Exactly one L1, naming the owning project.
        l1s = [e for e in esc_queue.get_pending() if e.task_id == '5638' and e.level == 1]
        assert len(l1s) == 1, f'expected exactly one L1; got {l1s!r}'
        esc = l1s[0]
        assert esc.category == 'scope_violation'
        assert 'dark_factory' in esc.summary, (
            f'the L1 summary must name the owning project; got {esc.summary!r}'
        )
        assert str(foreign_root / 'orchestrator/src/orchestrator/offline_lane.py') in esc.detail

        # (4) The report names the cross-repo cause.
        assert report is not None
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.block_reason == 'cross_repo_misfile'

    @pytest.mark.asyncio
    async def test_negative_control_local_task_runs_normally(self, tmp_path: Path):
        """The same shape with relative in-tree paths and no marker must dispatch."""
        from escalation.queue import EscalationQueue

        project_root = tmp_path / 'reify'
        project_root.mkdir()

        h = _make_slot_harness(tmp_path, project_root=project_root)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        assignment = _make_assignment(
            task_id='5639',
            files=[
                'orchestrator/src/orchestrator/offline_lane.py',
                'orchestrator/tests/test_offline_lane.py',
            ],
        )

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            mock_wf = _make_mock_workflow()
            MockWorkflow.return_value = mock_wf
            report = await h._run_slot(assignment, sem)

        assert MockWorkflow.called, 'a local task must dispatch normally'
        mock_wf.run.assert_awaited_once()
        h.scheduler.set_task_status.assert_not_awaited()
        assert not [e for e in esc_queue.get_pending() if e.task_id == '5639'], (
            'a local task must file no escalation'
        )
        assert report is not None
        assert report.block_reason != 'cross_repo_misfile'

    @pytest.mark.asyncio
    async def test_marker_stamped_after_creation_is_still_honoured(self, tmp_path: Path):
        """The files_tagged_at shape a submit-time gate structurally cannot see.

        This gate reads the task AS IT STANDS AT DISPATCH, so a marker (or a
        files list) stamped after the task was created is caught all the same.
        """
        from escalation.queue import EscalationQueue

        project_root = tmp_path / 'reify'
        project_root.mkdir()

        h = _make_slot_harness(tmp_path, project_root=project_root)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        h._escalation_queue = esc_queue

        assignment = _make_assignment(
            task_id='5308',
            # Relative foreign paths — only the submit-written marker can see
            # these, which is exactly why leg A reads it.
            files=['orchestrator/src/orchestrator/offline_lane.py'],
            cross_repo=True,
            cross_repo_project='dark_factory',
            files_tagged_at='2026-08-05T12:00:00+00:00',
        )

        sem = MagicMock()
        sem.release = MagicMock()

        with patch('orchestrator.harness.build_workflow') as MockWorkflow:
            MockWorkflow.return_value = _make_mock_workflow()
            report = await h._run_slot(assignment, sem)

        assert not MockWorkflow.called
        assert report is not None
        assert report.block_reason == 'cross_repo_misfile'
        l1s = [e for e in esc_queue.get_pending() if e.task_id == '5308']
        assert len(l1s) == 1
        assert 'dark_factory' in l1s[0].summary
