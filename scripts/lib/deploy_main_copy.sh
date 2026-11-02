#!/usr/bin/env bash
# shellcheck shell=bash
# scripts/lib/deploy_main_copy.sh -- the preamble of a deploy script that
# lands an edit to a config file through its checkout's own merge queue
# (scripts/merge-deep-set-cap.sh, scripts/merge-pytest-n-ab-switch.sh).
# Sourced, never executed: it defines one function and sets no shell options.
#
# deploy_main_copy <config_path> <fail>
#   Resolves the checkout holding <config_path> and copies main's committed
#   copy of that file into a scratch directory, so the caller edits the copy
#   and the checkout itself is never written: only the queue's landing syncs
#   it. On success it sets, in the caller's shell:
#     REPO       the checkout's root
#     REL        <config_path> relative to REPO
#     BASE       main's commit, which the edit is computed against
#     SCRATCH    the scratch directory, removed at exit
#     MAIN_COPY  main's copy of the file, in SCRATCH
#     EDITED     a copy of MAIN_COPY for the caller's editor to rewrite
#   On failure it calls `<fail> <tag> <detail>`, which must not return. The
#   tag is one of config_not_in_git, no_main_branch, no_scratch_dir and
#   config_not_on_main.
#
# It also installs the caller's EXIT, TERM and INT traps, and their order is
# load-bearing. Bash runs a TERM or INT trap only once the foreground command
# returns, so a killpg lets the landing step cancel its request and tear down
# its deploy branch first; the trap's `exit` then runs the EXIT trap, which
# removes SCRATCH.
deploy_main_copy() {
    local config="$1" fail="$2"
    REPO="$(git -C "$(dirname "$config")" rev-parse --show-toplevel)" \
        || "$fail" config_not_in_git "config is not inside a git checkout: $config"
    REL="$(realpath --relative-to="$REPO" "$config")"
    BASE="$(git -C "$REPO" rev-parse --verify 'main^{commit}')" \
        || "$fail" no_main_branch "no main branch in $REPO"
    SCRATCH="$(mktemp -d)" || "$fail" no_scratch_dir "mktemp -d could not make a scratch directory"
    trap 'rm -rf "$SCRATCH"' EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    MAIN_COPY="$SCRATCH/main.yaml"
    EDITED="$SCRATCH/edited.yaml"
    git -C "$REPO" show "$BASE:$REL" > "$MAIN_COPY" \
        || "$fail" config_not_on_main "$REL is not tracked on main ($BASE)"
    cp "$MAIN_COPY" "$EDITED"
}
