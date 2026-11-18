"""The checked-in capability-manifest corpus: every committed sidecar loads, sits where its PRD derives, and names a runnable script target.

These guards read the committed plans/ and docs/prds/ trees and the git index,
which no shared cache key covers, so they live in this never-cached module.
The schema's unit tests stay in shared/tests/test_capability_manifest.py; the
corpus walker they share is shared/tests/cross_suite_helpers/capability_manifest_corpus.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml
from capability_manifest_corpus import (
    MANIFEST_SUFFIX,
    REPO_ROOT,
    AmbiguousIndexEntry,
    GitUnavailable,
    check_manifest,
    check_script_target,
    committed_file_mode,
    discover_manifests,
    iter_script_checks,
    missing_from_worktree_message,
)
from pydantic import ValidationError
from shared.capability_manifest import (
    ManifestCapability,
    load_capability_manifest,
    sidecar_path,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


#: The one committed exemplar sidecar TestLoader validates directly, and the
#: anchor TestManifestCorpusDiscovery asserts is present in the full corpus
#: sweep — hoisted once (code-review amendment, task 3362) so a rename of
#: this PRD or its sidecar breaks exactly one place instead of two.
_EXEMPLAR_SIDECAR_REL = 'plans/capability-delivered-checks-prd.capability-manifest.yaml'


class TestLoader:
    """The committed exemplar sidecar, as a CI fixture."""

    def test_committed_exemplar_sidecar_validates(self):
        # CI fixture: this PRD's own committed exemplar sidecar (PRD
        # §Contract signal for task α — "this PRD's own committed exemplar
        # sidecar parses and validates in a CI test").
        sidecar_path = REPO_ROOT / _EXEMPLAR_SIDECAR_REL
        doc = load_capability_manifest(sidecar_path)
        assert doc.prd == 'plans/capability-delivered-checks-prd.md'
        assert doc.schema_version == 1
        labels = [task.label for task in doc.tasks]
        assert labels == ['α', 'β', 'γ', 'δ', 'ε', 'ζ']
        alpha = doc.tasks[0]
        assert alpha.task_id == 2574
        first_cap = alpha.capabilities[0]
        assert first_cap.delivered_check is not None
        assert first_cap.delivered_check.kind == 'grep'
        assert first_cap.delivered_check.expect == 'present'
        manual_caps = [
            cap
            for cap in alpha.capabilities
            if cap.delivered_check is not None and cap.delivered_check.kind == 'manual'
        ]
        assert len(manual_caps) == 1
        manual_check = manual_caps[0].delivered_check
        assert manual_check is not None
        assert manual_check.reason


class TestManifestCorpusDiscovery:
    """discover_manifests() — git-ls-files-based corpus discovery.

    Not Path.rglob: see capability_manifest_corpus.py's module docstring for
    the measurement (30 tracked sidecars via git ls-files, 0 of them under
    .worktrees/, vs 3793 extra sidecars an rglob from the main checkout would
    sweep in from other in-flight tasks' gitignored worktree checkouts).
    """

    def test_returns_nonempty_list_for_a_git_checkout(self):
        paths = discover_manifests()
        assert paths is not None
        assert paths != []

    def test_every_path_is_a_valid_sidecar_not_nested_in_worktrees(self):
        # The three properties that are real assertions on discover_manifests()'s
        # behavior, folded into one test (code-review amendment, task 3362:
        # the individual is_absolute() and sorted-with-no-duplicates cases
        # dropped as guaranteed by construction — repo_root / rel with an
        # absolute repo_root; sorted(set(...)) — three lines inside
        # discover_manifests itself, so they can't fail without a rewrite
        # that would also break this test and added no regression signal of
        # their own).
        paths = discover_manifests()
        assert paths is not None
        for path in paths:
            assert path.name.endswith(MANIFEST_SUFFIX)
            assert path.exists()
            # Load-bearing anti-rglob assertion — see class docstring. Checked
            # relative to REPO_ROOT, not as an absolute-path substring: when
            # this suite itself runs from inside a worktree, REPO_ROOT (this
            # file's own parents[2]) legitimately resolves to
            # .../.worktrees/<n>, so the absolute path always contains that
            # segment. What must never happen is a *nested* .worktrees
            # segment relative to REPO_ROOT — that would mean discovery
            # walked into another task's checkout.
            assert '.worktrees' not in path.relative_to(REPO_ROOT).parts

    def test_known_stable_anchor_is_present(self):
        # Shares _EXEMPLAR_SIDECAR_REL with
        # TestLoader.test_committed_exemplar_sidecar_validates (code-review
        # amendment, task 3362) so a rename of that PRD/sidecar breaks
        # exactly one place instead of two independently-hardcoded ones.
        # Deliberately does NOT also assert that both docs/prds/ and plans/
        # are represented (a dropped sibling test) — that pinned a corpus
        # fact (today's directory layout), not a discover_manifests()
        # invariant, and would go red on a harmless doc reorganization.
        paths = discover_manifests()
        assert paths is not None
        rels = {str(path.relative_to(REPO_ROOT)) for path in paths}
        assert _EXEMPLAR_SIDECAR_REL in rels


class TestCommittedFileMode:
    """committed_file_mode(rel) — the INDEX mode, not the working-tree bit.

    os.access is greened by a local `chmod +x` that is never staged, which
    would leave main exactly as broken while the guard reported green. This
    checker is what proves a mode fix actually LANDS.

    Its three outcomes are asserted separately (code-review amendment, task
    3649), because collapsing them is what makes a red lie: a mode string, a
    None meaning genuinely-untracked, and a raised GitUnavailable meaning git
    never answered.
    """

    def test_known_executable_script_reports_100755(self):
        # scripts/check_method_param_wiring.py is committed 100755 (task 3364).
        assert committed_file_mode('scripts/check_method_param_wiring.py') == '100755'

    def test_untracked_path_returns_none(self):
        # None means exactly one thing: git answered, and there is no index
        # record. NOT "something went wrong" — that raises instead.
        assert committed_file_mode('scripts/no-such-file-3649.py') is None

    def test_glob_metacharacters_are_not_expanded(self):
        # git ls-files takes a PATHSPEC. Without :(literal) magic this pattern
        # glob-expands and reports scripts/check_method_param_wiring.py's mode
        # (measured) — a mode for a file that is not the named target. The
        # literal pathspec must instead match nothing.
        assert committed_file_mode('scripts/check_*_param_wiring.py') is None

    def test_directory_path_raises_ambiguous_index_entry(self):
        # Leading-path matching survives :(literal), so a directory matches
        # every tracked file beneath it (measured: 60 records). Taking
        # records[0] would report an unrelated file's mode as green — and the
        # is_file() gate that would catch this lives in check_script_target,
        # i.e. a different test method entirely.
        with pytest.raises(AmbiguousIndexEntry):
            committed_file_mode('scripts/tests')


_MANIFEST_PATHS = discover_manifests() or []
_MANIFEST_IDS = [str(p.relative_to(REPO_ROOT)) for p in _MANIFEST_PATHS]

#: Derived from _MANIFEST_PATHS, NOT a second corpus walk — script-check
#: discovery inherits discover_manifests' anti-rglob, worktree-excluding,
#: non-raising git ls-files behaviour verbatim. iter_script_checks is
#: likewise non-raising (see its docstring): this runs at module import
#: time, so anything escaping here takes down collection of the whole module.
_SCRIPT_CHECKS = [ref for path in _MANIFEST_PATHS for ref in iter_script_checks(path)]
_SCRIPT_CHECK_IDS = [
    f'{ref.manifest.relative_to(REPO_ROOT)}::{ref.capability}' for ref in _SCRIPT_CHECKS
]

#: Independent, loader-free way to count the `kind: script` checks a sidecar
#: DECLARES — deliberately a regex over raw text, so it still counts a check
#: inside a sidecar that iter_script_checks dropped (schema-invalid, or
#: tracked-but-absent from the working tree). Matches block style
#: (`kind: script`) and the flow style the corpus also uses
#: (`{kind: script, script: ...}`), quoted or bare. Feeds the count guard in
#: TestCheckedInScriptCheckTargets (code-review amendment, task 3649).
_SCRIPT_KIND_RE = re.compile(r'\bkind:\s*[\'"]?script\b')


class OpenVerdictRef(NamedTuple):
    """One ``verdict: OPEN`` capability row located in a checked-in sidecar."""

    manifest: Path
    label: str
    capability: ManifestCapability


def _discover_open_verdict_rows() -> list[OpenVerdictRef]:
    """Every ``verdict: OPEN`` capability row in the checked-in corpus.

    Derived from _MANIFEST_PATHS, NOT a second corpus walk — so it inherits
    discover_manifests' anti-rglob, worktree-excluding, non-raising
    ``git ls-files`` behaviour verbatim. The load is wrapped because this
    runs at module import time and anything escaping here takes down
    collection of the whole module; an unloadable or schema-invalid sidecar
    is TestCheckedInManifestCorpus's failure to report (with a
    file-attributed message), not this sweep's. The three caught types are
    exactly the ones load_capability_manifest documents as propagating.
    What the swallow drops is caught by
    test_open_row_sweep_covers_every_declared_row.
    """
    rows: list[OpenVerdictRef] = []
    for path in _MANIFEST_PATHS:
        try:
            doc = load_capability_manifest(path)
        except (OSError, yaml.YAMLError, ValidationError):
            continue
        for task in doc.tasks:
            rows.extend(
                OpenVerdictRef(manifest=path, label=task.label, capability=cap)
                for cap in task.capabilities
                if cap.verdict == 'OPEN'
            )
    return rows


_OPEN_ROWS = _discover_open_verdict_rows()
_OPEN_ROW_IDS = [
    f'{ref.manifest.relative_to(REPO_ROOT)}::{ref.label}::{ref.capability.name}'
    for ref in _OPEN_ROWS
]

#: Independent, loader-free way to count the `verdict: OPEN` rows a sidecar
#: DECLARES — same regex-over-raw-text discipline as _SCRIPT_KIND_RE above,
#: so it still counts a row inside a sidecar the loader could not parse.
#: Matches block style (`verdict: OPEN`) and the flow style the corpus also
#: uses (`{verdict: OPEN, ...}`), quoted or bare.
_OPEN_VERDICT_RE = re.compile(r'\bverdict:\s*[\'"]?OPEN\b')


def _open_row_evidence_gap(ref: OpenVerdictRef) -> str | None:
    """Why an OPEN row carries no evidence obligation, or ``None`` if it does.

    An OPEN row asserts that the binding decision is the leaf's own work
    product. That is only an honest record if the leaf owes something
    checkable for it: a mechanical `delivered_check` the δ gate can run, or
    a `kind: manual` check whose `reason` says what a human must confirm.
    A bare OPEN row with neither is indistinguishable from an unmeasured
    one — see TestCheckedInOpenVerdictEvidence.
    """
    check = ref.capability.delivered_check
    if check is None:
        return 'no delivered_check at all'
    if check.kind == 'manual' and not (check.reason or '').strip():
        return 'kind: manual with no reason'
    return None


class TestCheckedInManifestCorpus:
    """Every checked-in capability-manifest sidecar validates against the schema.

    This is the guard the task was filed for: TestLoader above covers exactly
    one committed exemplar; this class sweeps all of them (measured at
    planning time: 30 files under docs/prds/ and plans/, 447 capabilities,
    0 failures — see task 3362's plan.json for the full measurement). It is
    GREEN on arrival, which is inherent to a preventative guard, not a
    reason to trust it unverified — so the RED signal was proven by mutation
    instead of relying on a naturally-failing first run:

        1. Inserted an unknown key (`verdcit: PASS`) into one real capability
           entry of plans/capability-delivered-checks-prd.capability-manifest.yaml
           (the "sidecar-loader-models-exist" capability under task α).
        2. Ran this class: both parametrized cases keyed to that one file
           failed (test_checked_in_manifest_validates and
           _prd_field_matches_its_own_filename — each loads the same mutated
           file), while every other file's cases across both methods stayed
           green. The test_checked_in_manifest_validates failure carried the
           exact assertion message: "plans/capability-delivered-checks-prd.
           capability-manifest.yaml: schema validation failed: 1 validation
           error for CapabilityManifestDoc tasks.0.capabilities.1.verdcit
           Extra inputs are not permitted [...]" — naming both the mutated
           file and the offending `verdcit` key, exactly as check_manifest's
           docstring promises.
        3. Reverted the sidecar (`git checkout --`) and reran: full class
           green again, all 30 files (150 tests across the whole module —
           re-measured after the code-review amendments, task 3362, that
           dropped the third, tautological parametrization and trimmed
           TestManifestCorpusDiscovery).

    Does NOT assert delivered_check `script:` targets exist on disk — that
    property now has its own sweep, TestCheckedInScriptCheckTargets below.
    The exclusion this docstring used to record ("out of scope: n=1 in the
    corpus at planning time") was retired by task 3649: n reached 2, and the
    second one (plans/agent-transcript-archival-prd.capability-manifest.yaml
    -> scripts/gc_agent_transcripts.py) was committed at mode 100644 and had
    been ERRORing forever. This class stays scoped to manifest SHAPE; the
    sibling class covers script LIFECYCLE.

    Does NOT separately re-validate each capability via
    `ManifestCapability.model_validate(cap.model_dump()) == cap`
    (code-review amendment, task 3362: dropped as tautological) —
    `load_capability_manifest` already constructs every `ManifestCapability`
    through that same `extra='forbid'` model with plain str/list/int fields,
    so the round trip could only fail when `load_capability_manifest` itself
    already raised, which `test_checked_in_manifest_validates` already
    reports with a better, file-attributed message.
    """

    def test_corpus_discovery_is_not_vacuous(self):
        # An empty parametrize list below would collect ZERO test cases and
        # pass silently — exactly the fail-soft this guard exists to
        # prevent. Skip (not fail) when the tree genuinely isn't a git
        # checkout, mirroring test_lock_charter_guard.py's own split.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        assert _MANIFEST_PATHS, 'discover_manifests() returned an empty corpus'

    @pytest.mark.parametrize('manifest_path', _MANIFEST_PATHS, ids=_MANIFEST_IDS)
    def test_checked_in_manifest_validates(self, manifest_path):
        try:
            message = check_manifest(manifest_path)
        except FileNotFoundError:
            pytest.fail(missing_from_worktree_message(manifest_path))
        assert message is None, message

    @pytest.mark.parametrize('manifest_path', _MANIFEST_PATHS, ids=_MANIFEST_IDS)
    def test_checked_in_manifest_prd_field_matches_its_own_filename(self, manifest_path):
        # manifest_stamping.py derives the sidecar path as
        # re.sub(r'\.md$', '', prd_path) + MANIFEST_SUFFIX, so a doc.prd that
        # disagrees with its own sidecar's filename means commit_planning
        # looks for a sidecar that isn't there and silently no-ops (its
        # documented "no sidecar on disk is a complete no-op" contract) —
        # the drift is invisible at exactly the moment it matters.
        try:
            doc = load_capability_manifest(manifest_path)
        except FileNotFoundError:
            pytest.fail(missing_from_worktree_message(manifest_path))
        rel = str(manifest_path.relative_to(REPO_ROOT))
        expected_prd = rel[: -len(MANIFEST_SUFFIX)] + '.md'
        assert doc.prd == expected_prd
        assert (REPO_ROOT / doc.prd).is_file()


class TestSidecarPath:
    """Every checked-in sidecar resolves back from its own PRD key (task 6582)."""

    @pytest.mark.parametrize('manifest_path', _MANIFEST_PATHS, ids=_MANIFEST_IDS)
    def test_every_checked_in_sidecar_lives_where_its_prd_derives(self, manifest_path):
        # The property the stamper and the dashboard reader both rely on:
        # resolving a sidecar from its PRD key lands on this very file.
        try:
            doc = load_capability_manifest(manifest_path)
        except FileNotFoundError:
            pytest.fail(missing_from_worktree_message(manifest_path))
        rel = str(manifest_path.relative_to(REPO_ROOT))
        derived = sidecar_path(doc.prd)
        assert derived == rel, (
            f'{rel}: prd {doc.prd!r} derives sidecar {derived!r}, not this file {rel!r}'
        )


class TestCheckedInOpenVerdictEvidence:
    """Every checked-in `verdict: OPEN` row carries an evidence obligation.

    `verdict` has NO programmatic consumer — grep the tree: nothing reads
    `ManifestCapability.verdict` outside its own Literal. So "PASS required
    to queue", "FAIL blocks queueing" and "OPEN must never be read as a
    green G3 binding" are prose-only invariants, and widening the vocabulary
    to three values (task 4471) enlarged the surface on which a non-green
    row can queue silently. This class is the one mechanical backstop under
    OPEN: it cannot check that the deferral was JUSTIFIED, but it can check
    that the leaf owes something checkable for it, so an OPEN row is never a
    bare assertion with nothing behind it (code-review amendment, task 4471).

    Green on arrival — inherent to a preventative guard, and not a reason to
    trust it unverified. The RED signal was proven by mutation instead, in
    the same discipline TestCheckedInManifestCorpus records:

        1. Deleted the `delivered_check:` block from the γ1
           `list-typed-evidence-recovery-pinned` row of
           plans/toolcall-markup-containment-prd.capability-manifest.yaml,
           and separately blanked the `reason:` on the γ2 `kind: manual`
           row. Each mutation turned exactly its own parametrized case red
           ("no delivered_check at all" / "kind: manual with no reason"),
           naming the manifest, label and capability, with every other case
           and every other class staying green.
        2. Reverted with `git checkout --` and confirmed the sidecar was
           byte-identical again and the module fully green.

    Deliberately asserts NO non-vacuity floor, unlike
    test_corpus_discovery_is_not_vacuous and
    test_script_check_corpus_is_not_vacuous. An OPEN row exists precisely
    because its owning task is in flight, so the corpus legitimately reaches
    ZERO OPEN rows the day the last one lands — a floor here would be a
    landmine that goes red on someone else's success. The count cross-check
    below is the anti-shrinkage guard instead, and it holds at zero.
    """

    def test_open_row_sweep_covers_every_declared_row(self):
        # _discover_open_verdict_rows swallows sidecars the loader cannot
        # parse (it must — it runs at import time). Without this cross-check
        # a schema-invalid sidecar would drop ALL of its OPEN rows out of the
        # sweep below and every remaining case would stay green, which is the
        # precise fail-soft this class exists to remove. Re-derives the
        # expected count by regex over raw TEXT, bypassing the loader, and
        # asserts EQUALITY rather than a hand-bumped floor so nothing has to
        # be remembered later. Mirrors
        # test_sweep_covers_every_declared_script_check.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        declared = 0
        for path in _MANIFEST_PATHS:
            try:
                text = path.read_text(encoding='utf-8')
            except OSError:
                pytest.fail(missing_from_worktree_message(path))
            declared += len(_OPEN_VERDICT_RE.findall(text))
        assert len(_OPEN_ROWS) == declared, (
            f'{declared} `verdict: OPEN` rows are declared across the checked-in '
            f'sidecars but only {len(_OPEN_ROWS)} reached the sweep — '
            '_discover_open_verdict_rows dropped a sidecar (unloadable or '
            'schema-invalid; see TestCheckedInManifestCorpus for the attributed '
            'error), so its OPEN rows are silently unguarded'
        )

    @pytest.mark.parametrize('ref', _OPEN_ROWS, ids=_OPEN_ROW_IDS)
    def test_open_row_carries_an_evidence_obligation(self, ref):
        gap = _open_row_evidence_gap(ref)
        assert gap is None, (
            f'{ref.manifest.relative_to(REPO_ROOT)}: capability '
            f'{ref.capability.name!r} under label {ref.label!r} is '
            f'verdict: OPEN but has {gap}. OPEN records that the binding '
            'decision is THIS leaf\'s own work product, so the leaf must owe '
            'something checkable for it — a mechanical delivered_check the '
            'gate can run, or a kind: manual check whose reason states what a '
            'human must confirm. Without either, OPEN is indistinguishable '
            'from an unmeasured binding and queues with nothing behind it.'
        )


class TestCheckedInScriptCheckTargets:
    """Every checked-in `kind: script` delivered_check names a runnable target.

    The generalization of the per-capability assertion task 3364 left behind in
    scripts/tests/test_check_method_param_wiring.py::TestManifestScriptCoherence
    (its own NOTE earmarked this ticket, tkt_0RS3PJJZFSJRJERYB11FNNKF5W /
    esc-3364-2, for both the generalization and the retirement of that copy).

    Why it matters: `_run_script_check` builds
    argv = [str(project_root / meta.script), *args] and execs the target
    DIRECTLY, so a missing or non-executable script raises OSError, which
    run_delivered_check's catch-all maps to DeliveredCheckResult.ERRORED. The
    check never FAILs — it ERRORs forever, so the capability it was authored to
    gate is silently un-gated. That is not hypothetical: this guard arrived
    NATURALLY RED (task 3649), unlike TestCheckedInManifestCorpus above, which
    was green on arrival and needed a mutation to prove its RED signal.

    Provenance: arrived RED on scripts/gc_agent_transcripts.py (working tree
    -rw-rw-r--, committed 100644, both halves failing) and was greened by a
    `chmod +x` with no logic change — task 3649. Corpus sizes and parametrize
    ids are deliberately NOT quoted here: unlike the assertions below, prose
    counts rot silently the first time a sidecar or script check is added.

    Both halves are asserted, as two separate methods rather than one combined
    assertion, because either alone is a hole: os.access is the exact property
    _run_script_check depends on (it execs against the WORKING checkout), but it
    is greened by a local `chmod +x` that is never staged — which would leave
    main exactly as broken while this guard reported green. The committed-mode
    assertion is what proves the fix LANDS. Two methods means the red tells you
    WHICH is wrong (patch the tree vs stage the mode).
    """

    def test_script_check_corpus_is_not_vacuous(self):
        # @pytest.mark.parametrize over an empty list collects ZERO cases and
        # reports success — the precise silent-pass this guard exists to
        # remove. Skip (not fail) on a non-checkout, mirroring
        # test_corpus_discovery_is_not_vacuous.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        assert _SCRIPT_CHECKS, 'no kind: script delivered_check found in any checked-in sidecar'

    def test_both_known_script_targets_are_present(self):
        # Acceptance criterion 3 as a machine-checked assertion: dropping
        # either real target from coverage goes red rather than quietly
        # shrinking the sweep. Mirrors test_known_stable_anchor_is_present.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        assert {ref.script for ref in _SCRIPT_CHECKS} >= {
            'scripts/gc_agent_transcripts.py',
            'scripts/check_method_param_wiring.py',
        }

    def test_sweep_covers_every_declared_script_check(self):
        # Neither guard above detects coverage SHRINKAGE for a script check
        # added after task 3649: non-vacuity only asserts non-empty, and the
        # anchor test is a superset check pinned to today's two names. So a
        # future third target whose sidecar is tracked but absent from the
        # working tree, or temporarily schema-invalid, would drop out of
        # iter_script_checks' swallow with both staying green (code-review
        # amendment, task 3649).
        #
        # The cross-check re-derives the expected count by regex over raw
        # sidecar TEXT — bypassing the loader entirely, so it still counts a
        # declaration the loader could not parse — and asserts EQUALITY rather
        # than a hand-bumped floor, so nothing has to be remembered later.
        if discover_manifests() is None:
            pytest.skip('not a git checkout (git ls-files failed)')
        declared = 0
        for path in _MANIFEST_PATHS:
            try:
                text = path.read_text(encoding='utf-8')
            except OSError:
                # Tracked but unreadable: the very case the swallow hides.
                pytest.fail(missing_from_worktree_message(path))
            declared += len(_SCRIPT_KIND_RE.findall(text))
        assert len(_SCRIPT_CHECKS) == declared, (
            f'{declared} `kind: script` checks are declared across the checked-in '
            f'sidecars but only {len(_SCRIPT_CHECKS)} reached the sweep — '
            'iter_script_checks dropped one (unloadable or schema-invalid sidecar; '
            'see TestCheckedInManifestCorpus for the attributed error), so it is '
            'silently unguarded'
        )

    @pytest.mark.parametrize('ref', _SCRIPT_CHECKS, ids=_SCRIPT_CHECK_IDS)
    def test_target_exists_and_is_executable(self, ref):
        # The WORKING-tree property _run_script_check actually execs.
        message = check_script_target(ref)
        assert message is None, message

    @pytest.mark.parametrize('ref', _SCRIPT_CHECKS, ids=_SCRIPT_CHECK_IDS)
    def test_target_is_committed_executable(self, ref):
        # The INDEX mode — so a local-only `chmod +x` that is never staged
        # cannot green this guard while main stays broken.
        prefix = (
            f'{ref.manifest.relative_to(REPO_ROOT)}: capability {ref.capability!r}: {ref.script}'
        )
        try:
            mode = committed_file_mode(ref.script)
        except AmbiguousIndexEntry as exc:
            # Not skippable: either a conflicted index or a `script:` naming a
            # directory. Both mean this method cannot answer, and the second is
            # a real defect.
            pytest.fail(f'{prefix}: {exc}')
        except GitUnavailable as exc:
            # Pure infra — a missing binary, or an index.lock wedge against the
            # merge worker on this same checkout. Skip, matching the
            # non-vacuity tests' skip-on-non-checkout convention, rather than
            # rendering a flake as a staged-mode defect and sending the reader
            # to `git update-index --chmod` (code-review amendment, task 3649).
            pytest.skip(f'git could not answer: {exc}')
        assert mode is not None, (
            f'{prefix} is not tracked by git — it exists in the working tree but was '
            'never `git add`ed, so main has no script for _run_script_check to exec'
        )
        assert mode == '100755', (
            f'{prefix} is committed at mode {mode}, not 100755 — a working-tree '
            'chmod that is never staged leaves main ERRORing'
        )
