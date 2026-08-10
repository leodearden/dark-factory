"""Unit tests for the completion-claim verification gate (task 3142, PRD leaf pi).

The gate is the code-level enforcement of the "Terminal-State Pre-Check
Discipline" that until now existed only as prompt text (reconciliation/prompts/
stage1.py). It extracts completion claims that NAME something concrete (a task
id, a commit sha, a tkt_ id), checks each against its live authority, and — on
mismatch OR unresolvable — has the episode ingested TAGGED rather than
rejected.

These tests are pure: extraction is textual, verification runs against injected
tri-state probes, so no Taskmaster, no ticket DB and no git are needed (the one
exception is the make_commit_probe suite, which builds a throwaway repo).
"""

from __future__ import annotations

import pytest

from fused_memory.services.completion_claim_gate import (
    UNRESOLVABLE,
    UNVERIFIED_CLAIM_TAG,
    CompletionClaim,
    build_unverified_flag,
    extract_completion_claims,
    make_commit_probe,
    verify_claims,
)

_KNOWN = frozenset({'reify', 'dark_factory'})


def _extract(text: str, default_project_id: str = 'reify') -> list[CompletionClaim]:
    return extract_completion_claims(
        text,
        default_project_id=default_project_id,
        known_project_ids=_KNOWN,
    )


class TestAppliedWorkExtraction:
    """`applied_work` phrasing (applied / landed / merged / shipped / patched)
    anchored to an explicit task reference in the same clause."""

    def test_has_been_applied_yields_one_task_claim(self):
        text = "task 5422's de-flake fix has been applied"
        claims = _extract(text)

        assert len(claims) == 1
        claim = claims[0]
        assert claim.kind == 'applied_work'
        assert claim.subject == 'task'
        assert claim.ref == '5422'
        assert claim.project_id == 'reify'
        # The span points back into the original text so the flag can quote it.
        start, end = claim.span
        assert 0 <= start < end <= len(text)
        assert text[start:end].strip()

    @pytest.mark.parametrize(
        'text',
        [
            "task 5422's de-flake fix has been applied",
            'the fix for task 5422 landed',
            'df 5422 was merged',
            'task 5422 has shipped',
            "task 5422's flake was patched",
        ],
    )
    def test_applied_work_family_each_yields_one_claim(self, text):
        claims = _extract(text)

        assert len(claims) == 1, f'{text!r} -> {claims!r}'
        assert claims[0].kind == 'applied_work'
        assert claims[0].subject == 'task'
        assert claims[0].ref == '5422'

    @pytest.mark.parametrize(
        'text',
        [
            # Negated terminal outcome — a consistent NON-completion statement.
            "task 5422's fix has NOT yet landed",
            "task 5422's fix has not been applied",
            # Future/aspirational framing — describes work that has NOT completed.
            "task 5422's fix will land tomorrow",
            'task 5422 is going to be merged once review clears',
        ],
    )
    def test_negated_and_aspirational_framing_yields_nothing(self, text):
        assert _extract(text) == []

    def test_claim_without_a_named_ref_yields_nothing(self):
        """Volume control: an unanchored claim is not actionable and not tagged."""
        assert _extract('the de-flake fix has been applied') == []

    def test_ref_without_completion_phrasing_yields_nothing(self):
        assert _extract('task 5422 is pending review') == []

    def test_clause_scoping_keeps_ref_and_phrasing_together(self):
        """A ref in one clause and phrasing in another is not a claim."""
        assert _extract('task 5422 is under review. the other fix has been applied') == []


# The verbatim text from esc-3085-1 instance (2): a reify-authored claim that
# a task was re-filed into ANOTHER project's tree as a ticket that did not
# exist. Neither the phrasing family nor the ticket subject was covered before.
_INSTANCE_2 = (
    'reify task 5638 was reported unactionable and re-filed into '
    "dark_factory's task tree as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376"
)


class TestFilingDispatchExtraction:
    """The esc-3085-1 scope extension: filing/dispatch phrasing, and the
    ticket / commit subjects alongside tasks."""

    def test_instance_2_extracts_as_a_ticket_filing_claim(self):
        claims = _extract(_INSTANCE_2)

        assert len(claims) == 1, claims
        claim = claims[0]
        assert claim.kind == 'filing_dispatch'
        # Ticket beats task: the tkt_ id is the more specific authority, and it
        # is the one that was actually false in the incident.
        assert claim.subject == 'ticket'
        assert claim.ref == 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'

    @pytest.mark.parametrize(
        'phrasing',
        [
            'was filed as',
            'was re-filed as',
            'was refiled as',
            'was submitted as',
            'was queued as',
            'was dispatched as',
            'was cancelled as',
            'was closed as duplicate of',
        ],
    )
    def test_filing_dispatch_family_each_yields_a_ticket_claim(self, phrasing):
        text = f'the follow-up {phrasing} ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376'
        claims = _extract(text)

        assert len(claims) == 1, f'{text!r} -> {claims!r}'
        assert claims[0].kind == 'filing_dispatch'
        assert claims[0].subject == 'ticket'
        assert claims[0].ref == 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'

    def test_commit_sha_claim_resolves_to_the_commit_subject(self):
        claims = _extract('the de-flake fix landed in commit 7bbcd5d815')

        assert len(claims) == 1, claims
        assert claims[0].kind == 'applied_work'
        assert claims[0].subject == 'commit'
        assert claims[0].ref == '7bbcd5d815'

    def test_commit_beats_task_but_ticket_beats_commit(self):
        """Subject precedence is ticket > commit > task, per clause."""
        task_and_commit = _extract('task 5422 was merged as commit 7bbcd5d815')
        assert [(c.subject, c.ref) for c in task_and_commit] == [('commit', '7bbcd5d815')]

    @pytest.mark.parametrize(
        'text',
        [
            # Hex-looking word with no commit/sha/merge cue anchoring it.
            'the deadbeef fixture was applied',
            # A `tkt_` prefix with no body is not a ticket id.
            'the follow-up was filed as ticket tkt_',
            # Filing phrasing with no ref at all.
            'the follow-up was filed as a ticket',
        ],
    )
    def test_non_refs_yield_nothing(self, text):
        assert _extract(text) == []

    @pytest.mark.parametrize(
        'text',
        [
            'the follow-up will be filed as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376',
            'the follow-up is supposed to be filed as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376',
            'the follow-up has not been filed as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376',
        ],
    )
    def test_filing_negation_and_aspiration_yields_nothing(self, text):
        """The imported strippers have no arm for the filing vocabulary, so the
        supplementary ones must cover it — otherwise the negation hole the
        strippers exist to close reopens for exactly the new family."""
        assert _extract(text) == []


class TestCrossProjectRefResolution:
    """Which project's registry adjudicates a task ref.

    esc-3085-1: the incident claim was written by a reify agent ABOUT a
    dark_factory artefact, so resolving every ref against the writer's project
    would have produced a false verdict in the other direction.
    """

    @pytest.mark.parametrize(
        'text',
        [
            'dark_factory task 3142 has landed',
            'dark_factory:3142 was merged',
            "dark_factory's task 3142 has landed",
        ],
    )
    def test_recognised_qualifier_overrides_the_writers_project(self, text):
        claims = _extract(text, default_project_id='reify')

        assert len(claims) == 1, f'{text!r} -> {claims!r}'
        assert claims[0].subject == 'task'
        assert claims[0].ref == '3142'
        assert claims[0].project_id == 'dark_factory'

    def test_unqualified_ref_inherits_the_writers_project(self):
        claims = _extract('task 3142 has landed', default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].project_id == 'reify'

    def test_arbitrary_preceding_word_is_not_a_qualifier(self):
        """'the merge task 3142' must not make 'merge' a project name."""
        claims = _extract('the merge task 3142 has landed', default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].project_id == 'reify'

    def test_unknown_project_qualifier_falls_back_to_the_writer(self):
        claims = _extract('someproject task 3142 has landed', default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].project_id == 'reify'

    def test_ticket_claim_carries_no_project(self):
        """A tkt_ id is a globally unique PK — it needs no project to resolve,
        and pinning one would reintroduce the instance-(2) false verdict."""
        claims = _extract(_INSTANCE_2, default_project_id='reify')

        assert len(claims) == 1
        assert claims[0].subject == 'ticket'
        assert claims[0].project_id is None


def _verify(claims, *, task=None, ticket=None, commit=None):
    """Run verify_claims with probes that fail loudly if an unexpected one is
    consulted — the short-circuit contract is part of what is under test."""

    def _unexpected(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError(f'probe called unexpectedly: {args} {kwargs}')

    return verify_claims(
        claims,
        task_status_probe=task or _unexpected,
        ticket_probe=ticket or _unexpected,
        commit_probe=commit or _unexpected,
    )


class TestVerifyClaims:
    """Tri-state probes -> {verified, mismatch, unverifiable}.

    The fail direction is INVERTED relative to _premature_completion_block and
    make_source_and_history_probe: those REJECT or DROP, so an unresolvable
    authority must fail OPEN there. This gate only LABELS, so an unresolvable
    authority must land on 'unverifiable' and get TAGGED.
    """

    def _task_claim(self, project_id='reify'):
        return CompletionClaim(
            kind='applied_work', subject='task', ref='5422',
            project_id=project_id, span=(0, 40),
        )

    def test_live_task_status_contradicting_the_claim_is_a_mismatch(self):
        verdicts = _verify([self._task_claim()], task=lambda ref, project: 'in-progress')

        assert len(verdicts) == 1
        assert verdicts[0].status == 'mismatch'
        assert verdicts[0].observed == 'in-progress'

    def test_terminal_task_status_verifies_the_claim(self):
        verdicts = _verify([self._task_claim()], task=lambda ref, project: 'done')

        assert verdicts[0].status == 'verified'
        assert verdicts[0].observed == 'done'

    @pytest.mark.parametrize('probed', [None, 'unknown'])
    def test_unresolvable_task_status_is_unverifiable_not_verified(self, probed):
        verdicts = _verify([self._task_claim()], task=lambda ref, project: probed)

        assert verdicts[0].status == 'unverifiable'
        assert verdicts[0].observed

    def test_task_probe_receives_the_claims_resolved_project(self):
        seen = []

        def probe(ref, project):
            seen.append((ref, project))
            return 'done'

        _verify([self._task_claim(project_id='dark_factory')], task=probe)
        assert seen == [('5422', 'dark_factory')]

    def _ticket_claim(self):
        return CompletionClaim(
            kind='filing_dispatch', subject='ticket',
            ref='tkt_0RRRC5AASJ9Z630VP4PCN9H376', project_id=None, span=(0, 120),
        )

    def test_absent_ticket_is_a_mismatch_naming_the_absent_id(self):
        """esc-3085-1 instance (2): the claimed ticket did not exist."""
        verdicts = _verify([self._ticket_claim()], ticket=lambda ref: None)

        assert verdicts[0].status == 'mismatch'
        assert 'tkt_0RRRC5AASJ9Z630VP4PCN9H376' in verdicts[0].observed

    def test_present_ticket_verifies_and_surfaces_its_owning_project(self):
        row = {'ticket_id': 'tkt_0RRRC5AASJ9Z630VP4PCN9H376',
               'project_id': 'dark_factory', 'status': 'resolved'}
        verdicts = _verify([self._ticket_claim()], ticket=lambda ref: row)

        assert verdicts[0].status == 'verified'
        assert 'dark_factory' in verdicts[0].observed

    def test_unreachable_ticket_registry_is_unverifiable(self):
        verdicts = _verify([self._ticket_claim()], ticket=lambda ref: UNRESOLVABLE)

        assert verdicts[0].status == 'unverifiable'
        assert verdicts[0].observed

    def _commit_claim(self):
        return CompletionClaim(
            kind='applied_work', subject='commit', ref='7bbcd5d815',
            project_id='dark_factory', span=(0, 44),
        )

    @pytest.mark.parametrize(
        ('probed', 'expected'),
        [(True, 'verified'), (False, 'mismatch'), (None, 'unverifiable')],
    )
    def test_commit_probe_is_tri_state(self, probed, expected):
        verdicts = _verify([self._commit_claim()], commit=lambda ref, project: probed)

        assert verdicts[0].status == expected

    def test_no_claims_consults_no_probe_at_all(self):
        """The cheap textual extractor short-circuits before any authority read."""
        assert _verify([]) == []


class TestBuildUnverifiedFlag:
    _TEXT = (
        "task 5422's de-flake fix has been applied. "
        'the follow-up was filed as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376'
    )

    def _mixed_verdicts(self):
        claims = extract_completion_claims(
            self._TEXT, default_project_id='reify', known_project_ids=_KNOWN,
        )
        assert len(claims) == 2, claims
        return verify_claims(
            claims,
            task_status_probe=lambda ref, project: 'in-progress',
            ticket_probe=lambda ref: None,
            commit_probe=lambda ref, project: None,
        )

    def test_flag_reports_every_non_verified_claim_with_its_observed_state(self):
        flag = build_unverified_flag(self._mixed_verdicts(), text=self._TEXT)

        assert flag is not None
        assert flag['tag'] == UNVERIFIED_CLAIM_TAG
        assert len(flag['claims']) == 2

        by_ref = {entry['ref']: entry for entry in flag['claims']}
        task_entry = by_ref['5422']
        assert task_entry['subject'] == 'task'
        assert task_entry['project_id'] == 'reify'
        assert task_entry['status'] == 'mismatch'
        # INV-2: the flag records what was actually OBSERVED, not just a verdict.
        assert task_entry['observed'] == 'in-progress'
        # The claim quotes itself so a reader never has to re-derive the span.
        assert 'has been applied' in task_entry['text']
        assert task_entry['span'] == [
            *extract_completion_claims(
                self._TEXT, default_project_id='reify', known_project_ids=_KNOWN,
            )[0].span
        ]

        ticket_entry = by_ref['tkt_0RRRC5AASJ9Z630VP4PCN9H376']
        assert ticket_entry['subject'] == 'ticket'
        assert ticket_entry['project_id'] is None
        assert ticket_entry['status'] == 'mismatch'

    def test_verified_verdicts_are_omitted_from_the_flag(self):
        claims = extract_completion_claims(
            "task 5422's fix has been applied",
            default_project_id='reify', known_project_ids=_KNOWN,
        )
        verdicts = verify_claims(
            claims,
            task_status_probe=lambda ref, project: 'done',
            ticket_probe=lambda ref: None,
            commit_probe=lambda ref, project: None,
        )

        assert build_unverified_flag(verdicts, text='x') is None

    def test_no_verdicts_yields_no_flag(self):
        assert build_unverified_flag([], text='') is None


class TestMakeCommitProbe:
    """The one impure export. Tri-state, bounded, and never raises: an infra
    failure must read as UNRESOLVABLE (None), never as a clean absence (False)
    — a False here would be reported as "the writer claimed a commit that does
    not exist", which is a serious accusation to make on a git hiccup.
    """

    @pytest.fixture
    def repo(self, tmp_path):
        import subprocess

        root = tmp_path / 'repo'
        root.mkdir()
        subprocess.run(['git', 'init', '-q', '.'], cwd=root, check=True)
        subprocess.run(
            ['git', '-c', 'user.email=a@b', '-c', 'user.name=a',
             'commit', '-q', '--allow-empty', '-m', 'x'],
            cwd=root, check=True,
        )
        sha = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return root, sha

    def test_existing_commit_probes_true_full_and_abbreviated(self, repo):
        root, sha = repo
        probe = make_commit_probe(root)

        assert probe(sha) is True
        assert probe(sha[:8]) is True

    def test_absent_but_well_formed_sha_probes_false(self, repo):
        root, _sha = repo
        probe = make_commit_probe(root)

        assert probe('0000000000000000000000000000000000000001') is False

    def test_non_repository_root_is_unresolvable_not_absent(self, tmp_path):
        """`git cat-file` exits 128 for BOTH a missing object and a missing
        repository — conflating them would turn 'git was unusable' into 'the
        writer lied'."""
        outside = tmp_path / 'not-a-repo'
        outside.mkdir()

        assert make_commit_probe(outside)('0' * 40) is None

    def test_missing_git_binary_is_unresolvable(self, repo, monkeypatch):
        root, sha = repo
        probe = make_commit_probe(root)
        monkeypatch.setattr(
            'fused_memory.services.completion_claim_gate.subprocess.run',
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('git')),
        )

        assert probe(sha) is None

    def test_subprocess_timeout_is_unresolvable(self, repo, monkeypatch):
        import subprocess

        root, sha = repo
        probe = make_commit_probe(root)

        def _timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd='git', timeout=1.0)

        monkeypatch.setattr(
            'fused_memory.services.completion_claim_gate.subprocess.run', _timeout,
        )

        assert probe(sha) is None

    def test_probe_never_raises(self, tmp_path):
        """Including a repo_root that does not exist at all."""
        probe = make_commit_probe(tmp_path / 'does-not-exist')

        assert probe('0' * 40) is None


class TestEmitUnverifiedClaimEscalation:
    """The operator-visible half (INV-4). The tag labels the corpus, but nothing
    reads the corpus looking for tags — an unverified claim also has to reach a
    queue a human or the auto-watcher actually opens.

    Copied shape-for-shape from markup_tripwire.emit_markup_storm_escalation,
    and for the same reason it exists there rather than going through
    recon_lifecycle_filer: the recon_report channel silently DROPS findings when
    no Stage-2 run is active, and an episode arrives at arbitrary times.
    """

    @staticmethod
    def _flag(ref: str = '5422') -> dict:
        return {
            'tag': UNVERIFIED_CLAIM_TAG,
            'claims': [
                {
                    'kind': 'applied_work',
                    'subject': 'task',
                    'ref': ref,
                    'project_id': 'dark_factory',
                    'span': [0, 41],
                    'text': f"task {ref}'s de-flake fix has been applied",
                    'status': 'mismatch',
                    'observed': 'in-progress',
                }
            ],
        }

    def test_files_one_escalation_carrying_the_flag_payload(self, tmp_path):
        import json

        from fused_memory.services import completion_claim_gate as gate_mod

        esc_id = gate_mod.emit_unverified_claim_escalation(str(tmp_path), self._flag())
        if not gate_mod.HAS_ESCALATION:
            assert esc_id is None
            return

        assert isinstance(esc_id, str)
        files = list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))
        assert len(files) == 1, f'expected exactly one escalation file, found: {files}'
        payload = json.loads(files[0].read_text())
        assert payload['id'] == esc_id
        assert payload['category'] == 'unverified_completion_claim'

        # The record must be actionable without opening the code or the corpus:
        # it names WHAT was claimed and WHAT was observed. Asserted as labelled
        # substrings — a bare `'5422' in detail` would also be satisfied by a
        # digit in the interpolated tmp_path.
        detail = payload['detail']
        assert "ref='5422'" in detail, f'must name the ref: {detail!r}'
        assert "observed='in-progress'" in detail, f'must state what was seen: {detail!r}'
        assert "subject='task'" in detail, f'must name the authority: {detail!r}'
        assert 'has been applied' in detail, f'must quote the claim: {detail!r}'

    def test_dedupes_per_project_and_ref(self, tmp_path):
        """A repeated claim about the SAME ref collapses onto the open record;
        a claim about a DIFFERENT ref is its own finding and files separately.
        """
        from fused_memory.services import completion_claim_gate as gate_mod

        first = gate_mod.emit_unverified_claim_escalation(str(tmp_path), self._flag())
        if not gate_mod.HAS_ESCALATION:
            assert first is None
            return

        again = gate_mod.emit_unverified_claim_escalation(str(tmp_path), self._flag())
        assert again == first, (
            f'the same (project_root, ref) must dedup onto {first!r}; got {again!r}'
        )
        assert len(list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))) == 1

        other = gate_mod.emit_unverified_claim_escalation(
            str(tmp_path), self._flag(ref='9999'),
        )
        assert other != first, 'a different ref is a different finding'
        assert len(list((tmp_path / 'data' / 'escalations').glob('esc-*.json'))) == 2

    def test_none_project_root_is_a_quiet_no_op(self):
        from fused_memory.services import completion_claim_gate as gate_mod

        assert gate_mod.emit_unverified_claim_escalation(None, self._flag()) is None

    def test_missing_escalation_package_is_a_quiet_no_op(self, tmp_path, monkeypatch):
        from fused_memory.services import completion_claim_gate as gate_mod

        monkeypatch.setattr(gate_mod, 'HAS_ESCALATION', False)
        assert gate_mod.emit_unverified_claim_escalation(str(tmp_path), self._flag()) is None
        assert not (tmp_path / 'data').exists()

    def test_queue_open_failure_returns_none_without_raising(self, tmp_path, monkeypatch):
        """Escalation is purely ADDITIVE — the episode is already ingested and
        tagged by the time this runs. Every failure mode degrades to None plus a
        log line, never to an exception on the write path.
        """
        from fused_memory.services import completion_claim_gate as gate_mod

        if not gate_mod.HAS_ESCALATION:
            pytest.skip('escalation package unavailable')
        monkeypatch.setattr(
            gate_mod,
            'EscalationQueue',
            lambda *a, **k: (_ for _ in ()).throw(OSError('queue dir unwritable')),
        )
        assert gate_mod.emit_unverified_claim_escalation(str(tmp_path), self._flag()) is None

    def test_empty_flag_files_nothing(self, tmp_path):
        from fused_memory.services import completion_claim_gate as gate_mod

        assert gate_mod.emit_unverified_claim_escalation(str(tmp_path), {'claims': []}) is None
