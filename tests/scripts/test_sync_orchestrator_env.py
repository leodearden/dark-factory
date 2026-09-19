"""Guards on scripts/sync-orchestrator-env.sh — now the venv's ONLY repair path.

Until task 5553 the orchestrator units carried ``--frozen`` and, believing that
stopped a start from touching the shared venv, treated this script as one
repair path among several.  It was never that: ``uv run`` is an INEXACT sync
that INSTALLS a member's missing closure at process start (measured on uv
0.11.6), and ``--frozen`` — a LOCKFILE option — did not stop it.  So a
half-synced venv healed itself on the next unit start, and the script's own
drift was invisible.

With ``--no-sync`` on every unit that self-repair is gone, deliberately: a unit
now fails with ModuleNotFoundError rather than mutating the environment its
siblings are running out of.  That makes this script exactly what its header
has always claimed to be — the ONLY sanctioned way to (re)materialize
dark-factory/.venv — and makes two pre-existing defects in it load-bearing.
Each is one test below.

Both assert on the PARSED invocation rather than on a substring of the file,
because a substring check cannot tell an argument from a word in a comment, and
this file discusses the very commands it runs.
"""
import pathlib
import shlex

from systemd_unit_invariants import ALL_ORCHESTRATOR_SERVICE_FILES

SYNC_SCRIPT = pathlib.Path(__file__).parents[2] / "scripts" / "sync-orchestrator-env.sh"

# The one scripts/orchestrator-*.service file the script must NOT carry in
# SERVICES, asserted explicitly below so it is a decision rather than an
# accident of whichever names happened to be typed.  The watchdog is the PROBE,
# not a supervised orchestrator: it runs a bare Python script (no uv, no shared
# venv), and the script already stops and starts its TIMER separately — first
# and last respectively, because a 60s probe would otherwise revive a unit
# mid-sync.  Folding it into the services loop would stop the timer's unit
# instead of the timer and break that ordering.
_WATCHDOG_UNIT = "orchestrator-watchdog.service"


def _uv_sync_invocations(script: str) -> list[list[str]]:
    """Every ``uv sync ...`` command line in *script*, tokenised.

    Tolerates the ``"$UV"`` variable spelling the script actually uses (and a
    bare ``uv`` or an absolute path, so a later respelling does not silently
    drop the invocation out of the guard): a token is the uv executable if its
    final path segment is ``uv`` or it is a shell parameter expansion naming a
    variable ending in ``UV``.

    Comment lines are dropped BEFORE tokenising.  This script explains uv's
    sync semantics in prose, so a scan that did not would match the explanation
    and report on text that never runs — which is the same class of error as
    the substring check this helper exists to replace.

    ``shlex`` rather than a regex because the assertions below are about
    ARGUMENTS: ``--all-packages`` present, ``--project``/``--package`` absent.
    A regex over the raw line would have to re-derive quoting and word
    splitting to answer that, and would answer it differently from the shell.
    """
    invocations: list[list[str]] = []
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            # An unbalanced quote means the line is a fragment of something
            # larger; it cannot be a whole `uv sync` command on its own.
            continue
        for i, token in enumerate(tokens[:-1]):
            is_uv = token.rsplit("/", 1)[-1] == "uv" or (
                token.startswith("$") and token.strip("${}").endswith("UV")
            )
            if is_uv and tokens[i + 1] == "sync":
                invocations.append(tokens[i:])
    return invocations


def test_sync_uses_all_packages() -> None:
    """The repair must sync the WHOLE workspace, never a single member.

    Measured on uv 0.11.6 against a throwaway two-member workspace: ``uv sync
    --project <member>`` is EXACT by default and uninstalls everything no
    selected member needs.  It printed "Uninstalled 2 packages" and removed the
    SIBLING member and its dependency from the shared root .venv.  So the script
    that exists to repair the shared venv was pruning it — and that is the real
    sibling-uninstall vector, not the ``uv run --project <member>`` the
    2026-05-29 note blamed (measured inexact: installs, never prunes).

    ``uv sync --inexact --project <member>`` does not prune, but it also cannot
    INSTALL the other members' dependencies, so it cannot repair the venv the
    seven orchestrators, the dashboard, the load sampler and fused-memory all
    share.  ``--all-packages`` is the only form that does the job.

    SETUP.md:549-554 already records this norm from the other direction — always
    ``--all-packages``, never a bare ``uv sync``, because a bare one "exits 0
    while pruning the other members' console scripts — including the
    ``orchestrator`` entry point this host exists to expose (task 4539)".  That
    was observed at console-script granularity; the measurement above reproduces
    it at package granularity.  Cited rather than restated so the repair path
    and the onboarding doc give one answer.
    """
    invocations = _uv_sync_invocations(SYNC_SCRIPT.read_text(encoding="utf-8"))

    assert len(invocations) == 1, (
        f"expected exactly one `uv sync` invocation in {SYNC_SCRIPT.name}, found "
        f"{len(invocations)}: {invocations}. The count is pinned so a second one "
        "added later cannot slip past a first-match read — this guard would keep "
        "reporting on the original call while the new one did whatever it liked "
        "to the shared venv."
    )
    (invocation,) = invocations

    assert "--all-packages" in invocation, (
        f"{SYNC_SCRIPT.name} runs {' '.join(invocation)!r}, without "
        "`--all-packages`. Measured on uv 0.11.6: a member-scoped `uv sync "
        "--project <member>` is EXACT and UNINSTALLS every package no selected "
        "member needs — it removed a sibling workspace member and its dependency "
        "from the shared root .venv ('Uninstalled 2 packages'). With `--no-sync` "
        "on every unit, a start no longer repairs that, so this script is the "
        "whole safety net. SETUP.md records the same norm from task 4539."
    )

    member_scoping = [t for t in invocation if t.partition("=")[0] in ("--project", "--package")]
    assert not member_scoping, (
        f"{SYNC_SCRIPT.name} runs {' '.join(invocation)!r}, which scopes the sync "
        f"to a subset via {member_scoping}. Every workspace member resolves to the "
        "ONE root .venv, so scoping the repair means repairing part of a shared "
        "environment and — because a plain `uv sync` is exact — pruning the rest "
        "of it. `--all-packages` must select the whole workspace."
    )


def _services_array(script: str) -> list[str]:
    """The unit names in the script's ``SERVICES=( ... )`` array, tokenised.

    Tokenised rather than regexed off one line, because the array spans several
    physical lines and its entries are shell words: a line-oriented read would
    have to re-derive where the array ends, and would answer differently from
    the shell the moment an entry moved or a comment appeared inside it.  The
    body between the parentheses is handed to shlex, which is what the script's
    own expansion effectively does.
    """
    _, _, after = script.partition("SERVICES=(")
    assert after != "", (
        f"{SYNC_SCRIPT.name} has no `SERVICES=(` array. It is the single source "
        "of the list of units to stop and restart around the sync; if it was "
        "renamed or inlined, this guard must follow it rather than silently "
        "stop checking which units are covered."
    )
    body, closed, _ = after.partition(")")
    assert closed == ")", f"{SYNC_SCRIPT.name}'s SERVICES=( array is unterminated"
    return shlex.split(body, comments=True)


def test_sync_stops_every_committed_orchestrator_unit() -> None:
    """SERVICES must name every committed orchestrator unit but the watchdog.

    A stale list was TOLERABLE while a unit start could repair itself; with
    ``--no-sync`` it is not, for two compounding reasons the script's own header
    already argues. A unit left RUNNING through the sync is bound to an
    interpreter being rebuilt underneath it. A unit never RESTARTED afterwards
    keeps whatever it had — and can no longer pick the new environment up by
    re-syncing at its next start, because that is precisely what was removed.

    The expected set is DERIVED from ALL_ORCHESTRATOR_SERVICE_FILES (the glob
    over scripts/orchestrator-*.service, itself pinned against a known-basename
    set by test_orchestrator_service_files.py:570-582) rather than hand-listed.
    Hand-listing is what produced the drift being fixed: the script named three
    units while seven existed, and nothing could notice. Derived, an eighth
    orchestrator unit added next month turns this RED on its own.
    """
    expected = {p.name for p in ALL_ORCHESTRATOR_SERVICE_FILES} - {_WATCHDOG_UNIT}
    assert expected, (
        "ALL_ORCHESTRATOR_SERVICE_FILES yielded no units, so this guard would "
        "compare two empty sets and pass vacuously. The glob is anchored at the "
        "repo's scripts/ directory — check it resolved."
    )

    services = _services_array(SYNC_SCRIPT.read_text(encoding="utf-8"))

    missing = expected - set(services)
    assert not missing, (
        f"{SYNC_SCRIPT.name}'s SERVICES array does not name {sorted(missing)}. "
        "Every committed orchestrator unit runs out of the ONE shared .venv this "
        "script rebuilds, so one left running through the sync is bound to an "
        "interpreter being replaced underneath it, and one never restarted "
        "afterwards keeps stale bytecode — which `--no-sync` means it can no "
        "longer fix by re-syncing at its next start. Derive the list from the "
        "committed units rather than extending it by hand."
    )

    extra = set(services) - expected
    assert not extra, (
        f"{SYNC_SCRIPT.name}'s SERVICES array names {sorted(extra)}, which is not "
        "a committed scripts/orchestrator-*.service unit. A name with no "
        "committed unit behind it is stopped with `|| true` and so fails "
        "silently, leaving the array reading as if it covered something it does "
        "not."
    )

    assert _WATCHDOG_UNIT not in services, (
        f"{SYNC_SCRIPT.name}'s SERVICES array names {_WATCHDOG_UNIT}. The "
        "watchdog is the PROBE, not a supervised orchestrator: it runs a bare "
        "Python script that never touches the shared venv, and the script "
        "already stops its TIMER first and starts it last precisely so a 60s "
        "probe cannot revive a unit mid-sync. Stopping the service here would "
        "not stop the timer and would break that ordering."
    )
