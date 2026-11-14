#!/usr/bin/env bash
# The network-bound prefetch, run on the cheap STAGING pod (bench.sh --stage).
# Everything is built on the pod's LOCAL disk, as HOME=/root, and published to
# the network volume as a few large tar files: the volume is fast for streaming
# one big file and slow for many small ones. bench.sh pushed the sources first.
#   $STAGE/home.tar          toolchains and caches under $HOME (uv, node, rustup/cargo
#                            + registry, uv/npm/pyright caches, uv-managed Pythons,
#                            dummy ssh key, ~/.gitconfig); extracts to the same $HOME
#   $STAGE/apt-archives.tar  the .debs of DF_APT_PACKAGES
#   $STAGE/staged.json       SHAs, toolchain versions, size + sha256 of each tar (written last)
#   $STAGE/src/df.git        (pushed by bench.sh) bare repo, main = DF_SHA
#   $STAGE/src/reify-<sha>.tar  (streamed by bench.sh) git archive of REIFY_SHA
# Each tar is written under a .tmp name and renamed, so a half-written file is
# never mistaken for a complete one. Idempotent: a re-run finds the caches warm.
#   env: DF_SHA REIFY_SHA STATUS_DIR [STAGE=/workspace/hwbench]
set -euo pipefail

: "${DF_SHA:?}" "${REIFY_SHA:?}" "${STATUS_DIR:?}"
STAGE=${STAGE:-/workspace/hwbench}
HERE=$(cd "$(dirname "$0")" && pwd)
log() { echo "[stage $(date -u +%T)] $*"; }
. "$HERE/lib.sh"
flag_exit_status "$STATUS_DIR"

export PATH=$HOME/.local/bin:$HOME/.local/node/bin:$HOME/.cargo/bin:$PATH
unset VIRTUAL_ENV
scratch=$(mktemp -d /tmp/hwbench-stage.XXXXXX)
apt_dir=$HOME/hwbench-apt-archives
cleanup_on_exit() { rm -rf "$scratch"; }
reify_tar=$STAGE/src/reify-$REIFY_SHA.tar
shas=()

test "$(git --git-dir="$STAGE/src/df.git" rev-parse refs/heads/main)" = "$DF_SHA" || { log "df.git main is not $DF_SHA"; exit 1; }
test -s "$reify_tar" || { log "$reify_tar is missing"; exit 1; }
rm -f "$STAGE"/*.tmp

# publish_tar <name> <tar args...>: stream a tar to $STAGE/<name>.tar.tmp, rename
# it, and remember its sha256 (computed on the stream, not by re-reading).
publish_tar() {
	local name=$1 sha
	shift
	sha=$(tar -cf - "$@" | tee "$STAGE/$name.tar.tmp" | sha256sum | cut -d' ' -f1)
	mv -f "$STAGE/$name.tar.tmp" "$STAGE/$name.tar"
	shas+=("$name=$sha")
	log "published $name.tar: $(stat -c %s "$STAGE/$name.tar") bytes"
}

# Download first, so the .debs are kept whatever the image's apt config says
# about keeping them; then install, because building the uv and npm caches
# may compile sdists and native addons.
stage_apt() {
	local args=(-o "Dir::Cache::archives=$apt_dir" -o APT::Sandbox::User=root -o APT::Keep-Downloaded-Packages=true)
	as_root mkdir -p "$apt_dir/partial"
	apt_noninteractive update -qq
	apt_noninteractive "${args[@]}" install -y -qq --download-only "${DF_APT_PACKAGES[@]}" >/dev/null
	apt_noninteractive "${args[@]}" -o Dpkg::Options::=--force-unsafe-io install -y -qq "${DF_APT_PACKAGES[@]}" >/dev/null
	publish_tar apt-archives -C "$apt_dir" --files-from <(cd "$apt_dir" && ls -- *.deb)
}

warm_caches() {
	git clone -q "$STAGE/src/df.git" "$scratch/df"
	git -C "$scratch/df" checkout -q --detach "$DF_SHA"
	(
		cd "$scratch/df"
		uv sync --all-packages --frozen -q
		npm ci --no-audit --no-fund --silent
		.venv/bin/pyright --version >/dev/null # the wrapper fetches its npm package on first use
	)
	mkdir "$scratch/reify"
	tar -C "$scratch/reify" -xf "$reify_tar"
	(cd "$scratch/reify" && cargo fetch --locked -q)
	toolchain=$(python3 -I "$HERE/toolchain_versions.py" "$scratch/df/.venv/bin/python")
}

publish_home() {
	local paths=() p
	for p in .local .cargo .rustup .cache .npm .ssh/id_runpod .ssh/id_runpod.pub .gitconfig; do
		[[ -e $HOME/$p ]] && paths+=("$p")
	done
	publish_tar home -C "$HOME" "${paths[@]}"
}

timed "apt: download, install, publish" stage_apt
timed "toolchains" install_user_toolchains
timed "uv/npm/cargo caches" warm_caches
timed "home.tar" publish_home

python3 -I - "$DF_SHA" "$REIFY_SHA" "$HOME" "$STAGE" "$toolchain" "${shas[@]}" >"$STAGE/staged.json.tmp" <<'PY'
import json, os, sys, time
df_sha, reify_sha, home, stage, toolchain, *shas = sys.argv[1:]
artifacts = {}
for item in shas:
    name, sha = item.split('=')
    artifacts[name] = {'bytes': os.path.getsize(f'{stage}/{name}.tar'), 'sha256': sha}
print(json.dumps({
    'df_sha': df_sha, 'reify_sha': reify_sha, 'home': home, 'toolchain': json.loads(toolchain),
    'artifacts': artifacts, 'staged_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}))
PY
mv -f "$STAGE/staged.json.tmp" "$STAGE/staged.json"
du -sh "$STAGE"/* | sed 's/^/[stage] /'
log "done"
