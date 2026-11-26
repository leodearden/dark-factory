#!/usr/bin/env bash
# First step on the BENCH pod (bench.sh --from-volume): extract what the volume
# holds onto local disk, so the run itself touches /workspace for nothing.
#   $STAGE/home.tar             -> $HOME (must be the HOME stage.sh ran with)
#   $STAGE/apt-archives.tar     -> apt's local archive dir; the DF apt list is then
#                                  installed from it (apt-get update still runs, but
#                                  the versions it finds equal the cached .debs)
#   $STAGE/reify-deps.tar       -> /opt (top-level dir reify-deps/), if present
#   $STAGE/src/df.git           -> $BENCH/df.git
#   $STAGE/src/reify-<sha>.tar  -> $BENCH/reify
# Every step is a stream of one big file; timings and rates are logged.
#   env: DF_SHA REIFY_SHA STATUS_DIR [STAGE=/workspace/hwbench]
set -euo pipefail

: "${DF_SHA:?}" "${REIFY_SHA:?}" "${STATUS_DIR:?}"
STAGE=${STAGE:-/workspace/hwbench}
BENCH=$HOME/df-hwbench
APT_LOCAL_ARCHIVES=/var/cache/apt/archives
log() { echo "[localize $(date -u +%T)] $*"; }
. "$(dirname "$0")/lib.sh"
flag_exit_status "$STATUS_DIR"

# extract <tar> <dest dir> [root]: untar onto local disk (as root when asked)
# and log the rate.
extract() {
	local tarball=$1 dest=$2 how=${3:-user} t0=$SECONDS
	if [[ $how == root ]]; then
		as_root tar -C "$dest" -xf "$tarball"
	else
		tar -C "$dest" -xf "$tarball"
	fi
	log "$(basename "$tarball"): $(($(stat -c %s "$tarball") / 1000000)) MB in $((SECONDS - t0)) s"
}

# extract_staged <artifact> <dest dir> [root]: extract, after checking the size
# against staged.json's, so a truncated tar fails here and not later.
extract_staged() {
	local name=$1 want
	want=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"][sys.argv[2]]["bytes"])' "$STAGE/staged.json" "$name")
	[[ $(stat -c %s "$STAGE/$name.tar") == "$want" ]] || { log "$name.tar is not the staged $want bytes"; exit 1; }
	extract "$STAGE/$name.tar" "${@:2}"
}

install_apt_from_cache() {
	# the image's apt config deletes archives/*.deb after every update: extract after it
	apt_noninteractive update -qq
	extract_staged apt-archives "$APT_LOCAL_ARCHIVES" root
	local out
	out=$(apt_noninteractive install -y -o APT::Keep-Downloaded-Packages=true -o Dpkg::Options::=--force-unsafe-io "${DF_APT_PACKAGES[@]}")
	grep -E '^(Need to get|[0-9]+ upgraded)' <<<"$out" || true
}

[[ -s $STAGE/staged.json ]] || { log "$STAGE/staged.json is missing: stage first"; exit 1; }
log "volume holds $(cat "$STAGE/staged.json")"
python3 -I - "$STAGE/staged.json" "$DF_SHA" "$REIFY_SHA" "$HOME" <<'PY' || { log "the staged SHAs or HOME differ from this run's"; exit 1; }
import json, sys
staged = json.load(open(sys.argv[1]))
sys.exit(0 if (staged['df_sha'], staged['reify_sha'], staged['home']) == tuple(sys.argv[2:5]) else 1)
PY
test "$(git --git-dir="$STAGE/src/df.git" rev-parse refs/heads/main)" = "$DF_SHA" || { log "volume df.git main is not $DF_SHA"; exit 1; }

timed "home" extract_staged home "$HOME"
chmod 700 "$HOME"
if [[ -s $STAGE/reify-deps.tar ]]; then
	as_root mkdir -p /opt
	timed "reify-deps" extract "$STAGE/reify-deps.tar" /opt root
else
	log "no $STAGE/reify-deps.tar on the volume; skipping"
fi
rm -rf "$BENCH/df.git" "$BENCH/reify"
# A plain copy: preserving even the mode makes cp read the volume's ACLs, which
# the pod's local filesystem refuses (rc 1, "Operation not supported").
timed "df.git" cp -r "$STAGE/src/df.git" "$BENCH/df.git"
mkdir -p "$BENCH/reify"
timed "reify" extract "$STAGE/src/reify-$REIFY_SHA.tar" "$BENCH/reify"
echo "$REIFY_SHA" >"$BENCH/reify/.bench-sha"
timed "apt (install from the cached .debs)" install_apt_from_cache
log "done"
