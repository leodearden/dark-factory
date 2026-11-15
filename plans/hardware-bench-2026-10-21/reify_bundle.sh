#!/usr/bin/env bash
# reify_bundle.sh <out.tar>
#
# Packs a small reify repository holding the exact commits the reify gate
# bench needs, without shipping reify's 37 GiB history: a shallow fetch from
# the local checkout of REIFY_SHA (the gated merge) and COLD_SHA (the merge the
# laptop gated just before it), with SHALLOW_SINCE days of history behind them
# so first-parent diffs, merge-base and `git log -- <path>` see real ancestors.
# git bundles cannot carry a shallow history, so the result is a tar of a
# non-bare repo's .git (top dir `reify/`, no working tree): main points at
# REIFY_SHA^1 (main as the laptop saw it during the gate) and HEAD is detached
# at REIFY_SHA. Every commit and tree is checked against the source repo.
set -euo pipefail

REIFY_ROOT=${REIFY_ROOT:-/home/leo/src/reify}
REIFY_SHA=${REIFY_SHA:-57be3b937f8cddfb77408603f46a798710644d89}
COLD_SHA=${COLD_SHA:-fad2d981951d606291196fd8c9f8781f6cfbf350}
SHALLOW_SINCE_DAYS=${SHALLOW_SINCE_DAYS:-30}
out=${1:?usage: reify_bundle.sh <out.tar>}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
log() { echo "[reify-bundle $(date -u +%T)] $*"; }

since=$(date -u -d "@$(($(git -C "$REIFY_ROOT" log -1 --format=%ct "$REIFY_SHA") - SHALLOW_SINCE_DAYS * 86400))" +%F)
log "fetching $REIFY_SHA and $COLD_SHA (history since $since)"
git init -q "$work/reify"
git -C "$work/reify" fetch -q --shallow-since="$since" \
	--upload-pack='git -c uploadpack.allowAnySHA1InWant=true upload-pack' \
	"file://$REIFY_ROOT" "$REIFY_SHA" "$COLD_SHA"

first_parent=$(git -C "$REIFY_ROOT" rev-parse "$REIFY_SHA^1")
git -C "$work/reify" update-ref refs/heads/main "$first_parent"
git -C "$work/reify" update-ref refs/hwbench/gate "$REIFY_SHA"
git -C "$work/reify" update-ref refs/hwbench/cold "$COLD_SHA"
git -C "$work/reify" update-ref --no-deref HEAD "$REIFY_SHA"
git -C "$work/reify" gc -q --prune=now

for sha in "$REIFY_SHA" "$COLD_SHA" "$first_parent"; do
	for rev in "$sha" "$sha^{tree}" "$sha^1" "$sha^1^{tree}"; do
		want=$(git -C "$REIFY_ROOT" rev-parse "$rev")
		got=$(git -C "$work/reify" rev-parse "$rev")
		[[ $want == "$got" ]] || { log "MISMATCH at $rev: source $want, packed $got"; exit 1; }
	done
done
git -C "$work/reify" fsck --no-progress --connectivity-only
commits=$(git -C "$work/reify" rev-list --count "$REIFY_SHA")

tar -C "$work" -cf "$out.tmp" reify
mv "$out.tmp" "$out"
log "wrote $out ($(du -h "$out" | cut -f1), $commits commits behind REIFY_SHA; trees verified)"
