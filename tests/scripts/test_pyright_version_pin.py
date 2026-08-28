"""The fleet TYPE gate's pyright version must be PINNED, from one declaration.

Task 4538. ``dark-factory-orchestrator.yaml``'s ``type_check_command`` is seven
bare ``npx pyright`` clauses. Until this guard landed the repo carried NO
``package.json`` at all, so every one of those clauses resolved whatever the
executing host's npx happened to reach for — a warm ``~/.npm/_npx`` entry, else
a fresh registry fetch of ``pyright@latest``. Two consequences, neither of which
any check could see:

  * a green main turns red on nothing but an npm publish, and
  * two verify hosts on the same commit can disagree, because a pyright version
    skew IS a verdict skew.

MEASURED, 2026-08-28, on this workstation, from a directory whose parent
declared ``"pyright": "1.1.408"`` in ``package.json`` but had no
``node_modules/``:

    npx pyright --version                       -> pyright 1.1.413
    npx --offline pyright --version             -> npm error code ENOTCACHED
                                                   request to
                                                   https://registry.npmjs.org/pyright

1.1.413 is the floating latest. Meanwhile ``uv.lock`` pins pyright-python at
1.1.408 and the pre-commit hook (``hooks/project-checks``, ``cd $dir && uv run
pyright``) plus all nine ``<pkg>/orchestrator.yaml`` module configs resolve THAT
— so the fleet chain was type-checking the repo with a DIFFERENT checker than
every other type gate in it. That is the skew gate 3417's steward addendum
recorded as a correction (1.1.408 vs 1.1.411, 2026-08-06) still live, two
releases further apart.

The second measurement is the load-bearing one: it proves a root
``package.json`` with a pinned devDependency is NOT sufficient on its own. npx
does not read the dependency SPEC to decide what to fetch — it looks for an
INSTALLED binary, and a merge-verify worktree is fresh with ``node_modules/``
gitignored. So the pin needs an install step, which is why
``verify_cold_preprovision_command`` carries the ``npm ci`` clause this file
asserts.

WHY THE PIN IS NOT SPELLED AT THE INVOCATION. ``npx pyright@1.1.408`` would
also pin, but it would put the version in SEVEN places inside one yaml string,
and the yaml's own ~45-line comment above ``type_check_command`` (task 3367 /
esc-3359-1, task 3397, task 3022) requires every clause of that chain to stay a
bare ``cd <dir>`` or a bare ``npx pyright``: ``verify._cd_clause_target``
accepts only an exact two-token ``cd <dir>``, and both
``verify._scope_fallback_tool_to_subproject`` and
``test_fallback_verify_config.py``'s chain walkers stall cwd tracking on
anything else. The chain is therefore left BYTE-IDENTICAL by task 4538 and the
pin lives entirely outside it.

WHAT "EXACTLY ONE PLACE" MEANS HERE, precisely, because the phrase is easy to
overclaim: exactly one AUTHORED declaration —
``package.json``'s ``devDependencies.pyright``. ``package-lock.json`` is
npm-generated and is checked for AGREEMENT, not treated as a second source;
``uv.lock``'s pyright-python version is a different ecosystem's lane and is
checked for AGREEMENT too, so the two type-check lanes cannot resolve different
checkers. The fleet chain itself is asserted to carry no inline version, which
is what keeps a second, drifting declaration from appearing there later.

PLACEMENT IS LOAD-BEARING, same reason ``test_contributing_lint_command_drift``
states: ``tests/scripts/`` carries its own module config, so this guard runs
under FULL_SUITE and under merge-role ``merge_verify_breadth: full``.

OUT OF SCOPE. The nine ``uv run … pyright`` module configs (tasks 3842, 4358)
already resolve a pinned pyright-python and are guarded by
``test_scripts_module_config.py::test_type_gates_resolve_pyright_without_npx``;
this file adds only the cross-lane agreement assertion between them and the npm
lane. It makes no claim about the pyright-python wheel's own bundled-dist
resolution — that is recorded in ``tests/scripts/orchestrator.yaml``'s comment
block and is unaffected here.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import tomllib

import pytest
import yaml

from orchestrator import verify_cmd

REPO_ROOT = pathlib.Path(__file__).parents[2]

DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"
PACKAGE_JSON_PATH = REPO_ROOT / "package.json"
PACKAGE_LOCK_PATH = REPO_ROOT / "package-lock.json"
UV_LOCK_PATH = REPO_ROOT / "uv.lock"

# An EXACT npm version: three dot-separated numeric components and nothing else.
# Deliberately rejects every range operator (`^`, `~`, `>=`, `*`, `x`), every
# dist-tag (`latest`, `next`) and every URL/`file:` spec — each of which floats,
# which is the whole defect. Prereleases (`1.1.408-dev`) are rejected too: the
# fleet gate should never run one.
_EXACT_NPM_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def _fleet_config() -> dict:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))


def _package_json() -> dict:
    """The committed root ``package.json``, or a loud failure naming the remedy.

    Never returns ``{}`` for a missing file. That is the vacuity hazard this
    whole file exists downstream of: an accessor that shrugs at an absent
    declaration turns every assertion below green while pinning nothing, which
    is strictly worse than no guard because it reports success.
    """
    assert PACKAGE_JSON_PATH.is_file(), (
        f"no {PACKAGE_JSON_PATH} (task 4538). dark-factory-orchestrator.yaml's "
        f"type_check_command is seven bare `npx pyright` clauses; with no root "
        f"package.json declaring a pinned pyright devDependency there is nothing "
        f"for the cold-worktree `npm ci` to install, so each clause resolves "
        f"whatever the host's npx cache holds or fetches pyright@latest. Restore "
        f"package.json + package-lock.json at the repo root."
    )
    return json.loads(PACKAGE_JSON_PATH.read_text(encoding="utf-8"))


def _package_lock() -> dict:
    assert PACKAGE_LOCK_PATH.is_file(), (
        f"no {PACKAGE_LOCK_PATH} (task 4538). `npm ci` REFUSES to run without a "
        f"lockfile, so the cold-verify pre-provision would fail open and leave "
        f"`npx pyright` unpinned again. Regenerate with `npm install "
        f"--package-lock-only` and commit it."
    )
    return json.loads(PACKAGE_LOCK_PATH.read_text(encoding="utf-8"))


def _declared_npm_pyright_pin() -> str:
    """The single authored npm pyright pin: ``devDependencies.pyright``."""
    dev_deps = _package_json().get("devDependencies", {})
    pin = dev_deps.get("pyright")
    assert pin, (
        f"root package.json declares no devDependencies.pyright (task 4538) — "
        f"the fleet type_check_command's `npx pyright` clauses have nothing to "
        f"resolve against; devDependencies: {dev_deps!r}"
    )
    return pin


def _uv_lock_pyright_version() -> str:
    """The pyright-python version ``uv.lock`` pins for the uv-resolved lane."""
    lock = tomllib.loads(UV_LOCK_PATH.read_text(encoding="utf-8"))
    versions = [p["version"] for p in lock.get("package", []) if p.get("name") == "pyright"]
    assert len(versions) == 1, (
        f"expected exactly one `pyright` package entry in uv.lock, found "
        f"{versions!r} (task 4538) — the cross-lane agreement assertion cannot "
        f"be evaluated"
    )
    return versions[0]


def _fleet_pyright_clauses() -> list[str]:
    """Every ``&&``-clause of the fleet ``type_check_command`` that runs pyright.

    Split with the PRODUCTION quote-aware splitter, matching
    ``test_contributing_lint_command_drift._ruff_segment`` and
    ``test_fallback_verify_config._pyright_clause_cwds`` — a naive
    ``str.split('&&')`` would read a quoted ``&&`` as a clause boundary.
    """
    cmd = _fleet_config()["type_check_command"]
    clauses = [c.strip() for c in verify_cmd.split_top_level_and(cmd)]
    pyright_clauses = [c for c in clauses if "pyright" in c]
    assert pyright_clauses, (
        f"no pyright clause in dark-factory-orchestrator.yaml type_check_command "
        f"(task 4538) — every assertion about how the fleet TYPE gate resolves "
        f"pyright would pass vacuously; command: {cmd!r}"
    )
    return pyright_clauses


def test_the_npm_pyright_pin_is_exact() -> None:
    """``devDependencies.pyright`` must be a bare ``X.Y.Z``, not a range.

    A caret range is the easy mistake and it reintroduces the whole defect at
    one remove: ``npm ci`` resolves a caret from the LOCKFILE, so it stays
    reproducible, but the next ``npm install`` silently re-resolves it upward
    and the "pinned" version moves without anybody editing a version anywhere.
    """
    pin = _declared_npm_pyright_pin()
    assert _EXACT_NPM_VERSION.match(pin), (
        f"root package.json pins pyright as {pin!r}, which is not an exact "
        f"X.Y.Z version (task 4538). A range (`^`/`~`/`>=`), a dist-tag "
        f"(`latest`), or a prerelease all float — the fleet type_check_command's "
        f"seven `npx pyright` clauses would go back to resolving a version "
        f"nobody chose, which is the defect this pin exists to close."
    )


def test_the_pin_is_declared_in_exactly_one_place() -> None:
    """Exactly one AUTHORED declaration of the npm pyright version.

    Two halves, and both are needed:

    (a) exactly one git-tracked ``package.json`` in the whole repo, so a second
        one cannot appear under a subdirectory and quietly become the
        ``localPrefix`` npx walks up to from a ``cd``-ed clause; and

    (b) no clause of the fleet chain carries an inline ``pyright@<version>``.
        That is the competing fix — ``npx pyright@1.1.408`` — and it is
        rejected here on purpose: it would spell the version SEVEN times in one
        string, and it would break the bare-``npx pyright`` invariant the yaml's
        own comment block and ``verify._cd_clause_target`` depend on.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*package.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert tracked == ["package.json"], (
        f"expected exactly one git-tracked package.json, at the repo root, found "
        f"{tracked!r} (task 4538). npx resolves its local bin directory by "
        f"walking UP from the clause's cwd to the nearest directory holding a "
        f"package.json or node_modules — so a second package.json under, say, "
        f"fused-memory/ would capture that walk and the root pin would stop "
        f"applying to that clause, silently and only for that clause."
    )

    for clause in _fleet_pyright_clauses():
        assert "pyright@" not in clause, (
            f"fleet type_check_command clause {clause!r} spells the pyright "
            f"version inline (task 4538). The pin has exactly one authored home, "
            f"root package.json's devDependencies.pyright; an inline `@version` "
            f"here is a second declaration free to drift from it, and it breaks "
            f"the bare-`npx pyright` clause shape that "
            f"verify._cd_clause_target and "
            f"test_fallback_verify_config.TestRootTypeCheckCommandPyrightInterpreterPinned "
            f"both walk."
        )


def test_the_fleet_chain_stays_bare_npx_pyright() -> None:
    """Every pyright clause is exactly the two tokens ``npx pyright``.

    This is what makes the package.json pin the ONLY thing standing between the
    fleet TYPE gate and a floating version — so it is asserted here rather than
    left as prose. It is also the invariant task 3367's yaml comment protects
    for an unrelated reason (cwd tracking through the ``cd`` clauses), which is
    why the pin had to be expressed outside the command string at all.
    """
    for clause in _fleet_pyright_clauses():
        assert shlex.split(clause) == ["npx", "pyright"], (
            f"fleet type_check_command pyright clause is {clause!r}, not a bare "
            f"`npx pyright` (task 4538 / task 3367). Every clause of this chain "
            f"must stay a bare `cd <dir>` or a bare `npx pyright`: "
            f"verify._cd_clause_target accepts only an exact two-token `cd "
            f"<dir>`, and a wider clause stalls cwd tracking at the previous "
            f"member for verify._scope_fallback_tool_to_subproject and for the "
            f"chain walkers in test_fallback_verify_config.py, mis-attributing "
            f"later clauses to the wrong directory."
        )


def test_the_committed_lockfile_agrees_with_the_pin_and_is_integrity_pinned() -> None:
    """``package-lock.json`` must resolve the pinned version, by integrity hash.

    ``npm ci`` installs strictly from the lockfile, so the lockfile — not
    package.json — is what the cold worktree actually gets. If the two disagree
    ``npm ci`` ABORTS, the pre-provision fails open with a warning, and the type
    leg silently reverts to an unpinned ``npx pyright``: a red-to-green-looking
    degradation with no other detector. The ``integrity`` assertion is the
    reproducibility half — without it a same-version republish would install
    different bytes.
    """
    pin = _declared_npm_pyright_pin()
    lock = _package_lock()

    root_dev_deps = lock.get("packages", {}).get("", {}).get("devDependencies", {})
    assert root_dev_deps.get("pyright") == pin, (
        f"package-lock.json's root devDependencies records pyright as "
        f"{root_dev_deps.get('pyright')!r} but package.json pins {pin!r} (task "
        f"4538) — `npm ci` refuses to run on a lockfile out of sync with "
        f"package.json, so the cold-verify pre-provision would fail open and "
        f"leave `npx pyright` unpinned. Re-run `npm install --package-lock-only` "
        f"and commit the result."
    )

    entry = lock.get("packages", {}).get("node_modules/pyright")
    assert entry is not None, (
        "package-lock.json has no `node_modules/pyright` entry (task 4538) — "
        "`npm ci` would install no pyright at all and the type leg would fall "
        "back to a registry fetch"
    )
    assert entry.get("version") == pin, (
        f"package-lock.json installs pyright {entry.get('version')!r}, but "
        f"package.json pins {pin!r} (task 4538). The lockfile is what `npm ci` "
        f"obeys, so THIS is the version the fleet type gate would actually run."
    )
    integrity = entry.get("integrity", "")
    assert integrity.startswith("sha512-"), (
        f"package-lock.json's pyright entry carries no sha512 integrity hash "
        f"(got {integrity!r}, task 4538) — the install is then reproducible only "
        f"in version, not in bytes"
    )
    assert entry.get("resolved", "").endswith(f"pyright-{pin}.tgz"), (
        f"package-lock.json's pyright entry resolves {entry.get('resolved')!r}, "
        f"which does not name the pinned {pin} tarball (task 4538)"
    )


def test_the_cold_worktree_install_step_runs_before_the_type_leg() -> None:
    """``verify_cold_preprovision_command`` must ``npm ci``, after the uv sync.

    This is the clause that makes the pin real on the path that matters. A
    merge verify runs in a FRESH throwaway worktree where ``node_modules/`` does
    not exist (it is gitignored) — so without an install step npx has no local
    binary to find and falls back to the registry, and the committed
    package.json pins nothing at all. ``verify._preprovision_shared_venv`` runs
    this command synchronously, in the worktree root, BEFORE the
    test/lint/type ``asyncio.gather``, which is exactly the slot the install
    needs.

    ORDER IS ASSERTED, not incidental. The clauses are ``&&``-chained, so a
    failing head skips the tail: with ``uv sync`` first, a broken npm install is
    REPORTED (rc != 0 -> loud fail-open warning) while the pre-existing
    cold-venv race fix still ran; reversed, a transient npm failure would
    silently take the venv sync down with it and reintroduce esc-2913-3.
    """
    cmd = _fleet_config()["verify_cold_preprovision_command"]
    clauses = [c.strip() for c in verify_cmd.split_top_level_and(cmd)]

    npm_ci = [i for i, c in enumerate(clauses) if shlex.split(c)[:2] == ["npm", "ci"]]
    assert len(npm_ci) == 1, (
        f"expected exactly one `npm ci` clause in "
        f"verify_cold_preprovision_command, found {len(npm_ci)} (task 4538); "
        f"command: {cmd!r}. Without it a cold merge-verify worktree has no "
        f"node_modules/, so every `npx pyright` clause of type_check_command "
        f"resolves from the host's npx cache or fetches pyright@latest — the "
        f"committed package.json pin never applies."
    )

    uv_sync = [i for i, c in enumerate(clauses) if shlex.split(c)[:2] == ["uv", "sync"]]
    assert uv_sync, (
        f"verify_cold_preprovision_command no longer runs `uv sync` (task 4538 / "
        f"task 2997, esc-2913-3): {cmd!r}. That clause is what populates the "
        f"shared .venv before the concurrent gather; the npm clause was added "
        f"beside it, not in place of it."
    )
    assert uv_sync[0] < npm_ci[0], (
        f"`npm ci` runs BEFORE `uv sync` in verify_cold_preprovision_command "
        f"(task 4538): {cmd!r}. The chain is `&&`-joined, so a transient npm "
        f"failure would then skip the venv sync entirely and reintroduce the "
        f"cold-venv race of esc-2913-3. Put `uv sync --all-packages` first."
    )


def test_both_type_check_lanes_resolve_the_same_pyright_version() -> None:
    """The npm pin and uv.lock's pyright-python must name the same version.

    Two lanes type-check this repo and they are NOT interchangeable by
    construction:

      * the fleet chain — ``npx pyright`` — now pinned by package.json, and
      * every ``<pkg>/orchestrator.yaml`` module config plus the pre-commit
        hook — ``uv run pyright`` — pinned by uv.lock's pyright-python wheel.

    pyright-python's version tracks the pyright release it bundles exactly, so
    equality here is meaningful rather than coincidental. Gate 3417's steward
    addendum recorded this pair MEASURED APART (venv 1.1.408 vs npx 1.1.411,
    2026-08-06) and correcting that was a memory entry, not a check. This is
    the check: a lane skew means the merge gate and the pre-commit hook can
    return different verdicts on the same tree, which is the same class of
    unreproducibility as the floating version itself.

    If ``uv lock --upgrade`` bumps pyright-python, this fails and the remedy is
    a matching ``npm install --save-exact pyright@<new>`` — deliberately a
    human decision, because it changes what every type gate in the repo runs.
    """
    npm_pin = _declared_npm_pyright_pin()
    uv_pin = _uv_lock_pyright_version()
    assert npm_pin == uv_pin, (
        f"pyright version skew between the repo's two type-check lanes (task "
        f"4538): package.json pins the npm package at {npm_pin}, uv.lock pins "
        f"pyright-python at {uv_pin}. The fleet type_check_command (`npx "
        f"pyright`) runs the first; every <pkg>/orchestrator.yaml module config "
        f"and hooks/project-checks (`uv run pyright`) run the second — so the "
        f"merge gate and the pre-commit hook can disagree about the same tree. "
        f"Bring them together with `npm install --save-exact pyright=={uv_pin}` "
        f"(note pyright-python's version tracks the pyright release it bundles)."
    )


# ---------------------------------------------------------------------------
# The load-bearing half: what a CLEAN worktree with a COLD npx cache actually
# resolves. Everything above is a paper contract between committed files; this
# executes the real npm/npx resolution the fleet chain depends on.
# ---------------------------------------------------------------------------


def _npx_pyright_version(subdir: pathlib.Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run ``npx --offline pyright --version`` from *subdir*, capturing rc+output.

    ``--offline`` is what makes this a PROOF rather than an observation: with an
    empty ``npm_config_cache`` the ONLY way this can succeed is a local
    ``node_modules/.bin/pyright`` found by walking up from *subdir*. Any
    registry or npx-cache path fails with ENOTCACHED instead of quietly
    returning some other version — so a green result cannot come from a warm
    cache the way the real hosts' shared ``~/.npm/_npx`` allows.

    *subdir* mirrors the fleet chain's first clause, ``cd fused-memory``: the
    walk-up from a SUBDIRECTORY is the part that could plausibly not work, and
    a check run at the root would not exercise it.
    """
    return subprocess.run(
        ["npx", "--yes", "--offline", "pyright", "--version"],
        cwd=subdir, env=env, capture_output=True, text=True, timeout=180,
    )


@pytest.fixture(scope="module")
def clean_worktree_resolution(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Reproduce a clean worktree + cold npx cache, and record BOTH resolutions.

    Copies only the two committed pin files into a throwaway tree, points npm at
    an EMPTY cache dir and an empty HOME (so no ``~/.npmrc`` and, decisively, no
    warm ``_npx`` entry), and measures the same invocation twice:

      ``before`` — no ``node_modules/``, i.e. exactly what a fresh merge-verify
                   worktree looks like. This is the negative control, and it
                   needs no network.
      ``after``  — following the same ``npm ci`` the cold-verify pre-provision
                   runs. This is the claim.

    Both are taken in ONE fixture so the two tests below cannot be reordered
    into disagreement, and so the (network-dependent) install happens once.
    """
    root = tmp_path_factory.mktemp("pyright-pin")
    # Read through the asserting accessors, never `if src.is_file()`. A fixture
    # that shrugs at a missing pin file would build a tree with nothing to
    # install, `npm ci` would fail, and the positive test below would SKIP —
    # reporting "not verified" for what is actually the pin being absent. The
    # negative control's whole claim also depends on package.json being PRESENT
    # and still not sufficient.
    _package_json()
    _package_lock()
    for name in ("package.json", "package-lock.json"):
        (root / name).write_bytes((REPO_ROOT / name).read_bytes())
    # Mirrors the fleet chain's first `cd` target; deliberately holds no
    # package.json or node_modules of its own, so npx must walk up to `root`.
    subdir = root / "fused-memory"
    subdir.mkdir()
    cache = root / "npm-cache"
    home = root / "home"
    home.mkdir()
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("npm_config_")},
        "HOME": str(home),
        "npm_config_cache": str(cache),
        "npm_config_update_notifier": "false",
    }

    before = _npx_pyright_version(subdir, env)

    install = subprocess.run(
        ["npm", "ci", "--no-audit", "--no-fund"],
        cwd=root, env=env, capture_output=True, text=True, timeout=900,
    )
    after = _npx_pyright_version(subdir, env) if install.returncode == 0 else None

    return {"root": root, "before": before, "install": install, "after": after}


def test_npx_in_a_fresh_worktree_does_not_resolve_the_pin_on_its_own(
    clean_worktree_resolution: dict[str, object],
) -> None:
    """NEGATIVE CONTROL: package.json alone pins nothing.

    Without this, the positive test below could pass for the wrong reason —
    "both hosts happened to share a cached version" is the vacuous green the
    whole task warns about. Here the committed package.json IS present and the
    pinned devDependency IS declared, and the invocation still cannot resolve:
    npx reads an INSTALLED binary, never the dependency spec. That is the
    measured reason a lockfile alone is not the fix, and the reason
    ``verify_cold_preprovision_command`` carries an install step.

    Needs no network — an offline npx with an empty cache fails locally.
    """
    before = clean_worktree_resolution["before"]
    assert isinstance(before, subprocess.CompletedProcess)
    assert before.returncode != 0, (
        f"`npx --offline pyright --version` SUCCEEDED in a tree with no "
        f"node_modules/ (task 4538), printing {before.stdout.strip()!r}. Either "
        f"the cache isolation in this fixture stopped working — in which case "
        f"the positive test below is now vacuous, because it could be reading "
        f"the same warm cache — or npx changed to honour the package.json spec "
        f"directly, which would make the install step unnecessary. Investigate "
        f"before touching either test."
    )


def test_npx_in_a_clean_worktree_resolves_the_pinned_version(
    clean_worktree_resolution: dict[str, object],
) -> None:
    """THE SIGNAL: after the pre-provision's ``npm ci``, npx resolves the pin.

    Run from a subdirectory, offline, against an empty npm cache — the three
    conditions that together rule out every non-pinned resolution path. What is
    left is the one the fleet chain relies on: ``npm ci`` materialises the
    lockfile's pyright into ``<worktree>/node_modules/.bin``, and npx's walk up
    from the ``cd``-ed member directory finds it.

    The ``npm ci`` step is the only part that reaches the network. A failure
    there SKIPS rather than fails, and says so — an unreachable registry is not
    evidence about the pin. The assertion itself never skips.
    """
    install = clean_worktree_resolution["install"]
    assert isinstance(install, subprocess.CompletedProcess)
    if install.returncode != 0:
        pytest.skip(
            "could not `npm ci` the committed lockfile into a throwaway tree "
            f"(rc={install.returncode}), so the clean-worktree resolution of "
            "`npx pyright` was NOT verified on this host. This is an "
            "infrastructure skip, not evidence: the npm registry is the only "
            f"network dependency here. stderr tail: {install.stderr[-500:]!r}"
        )

    after = clean_worktree_resolution["after"]
    assert isinstance(after, subprocess.CompletedProcess)
    pin = _declared_npm_pyright_pin()
    assert after.returncode == 0, (
        f"`npx --offline pyright --version` failed (rc={after.returncode}) from a "
        f"subdirectory of a tree where `npm ci` had just installed the pinned "
        f"pyright (task 4538) — so the fleet type_check_command's `cd "
        f"fused-memory && npx pyright` would NOT pick up the local install "
        f"either. stdout: {after.stdout!r} stderr: {after.stderr[-500:]!r}"
    )
    assert after.stdout.strip() == f"pyright {pin}", (
        f"`npx pyright --version` resolved {after.stdout.strip()!r} in a clean "
        f"worktree, but package.json pins {pin} (task 4538). The fleet TYPE gate "
        f"is running a version nobody declared, which is exactly the "
        f"host-to-host and publish-to-publish drift this pin exists to remove."
    )
