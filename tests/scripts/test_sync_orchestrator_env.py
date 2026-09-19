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

SYNC_SCRIPT = pathlib.Path(__file__).parents[2] / "scripts" / "sync-orchestrator-env.sh"


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
