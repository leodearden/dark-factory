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
from typing import Any

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
