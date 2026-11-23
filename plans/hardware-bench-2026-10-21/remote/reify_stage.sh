#!/usr/bin/env bash
# reify_stage.sh: the staging-pod half of the reify gate bench. Network-bound
# work only (the staging pod has 2 vCPU): nothing is compiled here.
#
#   STAGE         network volume root (default /workspace/hwbench)
#   REIFY_REPO    reify-repo.tar from reify_bundle.sh (default $STAGE/src/reify-repo.tar)
#   REIFY_SHA     the gated merge (default 57be3b937f...); its lockfiles drive the fetches
#   COLD_SHA      the cold run's commit (default fad2d98195...); its lockfiles are fetched too
#   WORK          local-disk scratch (default /root/reify-stage-work); deleted at the end
#
# Produces, writing each to a temp name then renaming:
#   $STAGE/apt-archives/*.deb   reify's apt closure, OCCT 7.8 from the FreeCAD PPA
#   $STAGE/reify-stage.tar      one tar, top dir reify-stage/:
#       bin/          cargo-nextest, sccache, tree-sitter (prebuilt, laptop versions)
#       cargo-home/   CARGO_HOME for both commits' Cargo.lock (registry + git);
#                     the gate's spec sets CARGO_HOME=/tmp/reify-agent-cargo-home
#       npm-cache/    npm cache for gui, gui/sidecar and tree-sitter-reify `npm ci`
#       apt/          the PPA's deb822 source and signing key, and packages.txt
#                     (the package list), for the bench pod's install
#       MANIFEST.json versions and checksums
# Idempotent: a finished tar whose MANIFEST matches the pinned versions is kept.
set -euo pipefail

STAGE=${STAGE:-/workspace/hwbench}
REIFY_REPO=${REIFY_REPO:-$STAGE/src/reify-repo.tar}
REIFY_SHA=${REIFY_SHA:-57be3b937f8cddfb77408603f46a798710644d89}
COLD_SHA=${COLD_SHA:-fad2d981951d606291196fd8c9f8781f6cfbf350}
WORK=${WORK:-/root/reify-stage-work}

# The laptop's versions (ssh leo-laptop: cargo-nextest/sccache/tree-sitter --version,
# dpkg -s libocct-foundation-dev; rust-toolchain.toml pins rustc).
NEXTEST_VERSION=0.9.136
SCCACHE_VERSION=0.14.0
TREE_SITTER_VERSION=0.26.8
RUST_TOOLCHAIN=1.96.0
NODE_VERSION=22.23.3
OCCT_VERSION=1:7.8.1+dfsg1-3~ubuntu24.04.1
OCCT_PPA_FINGERPRINT=9F0E375806520BD65BC61B18A728AE60A790F0EB

# setup-dev.sh's apt list, OCCT pinned to the laptop's build, plus the tools the
# gate's infra tests probe that the laptop has and the runpod image lacks.
APT_PACKAGES=(
	build-essential pkg-config clang cmake libslvs1-dev
	libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev librsvg2-dev
	libxdo-dev libayatana-appindicator3-dev libssl-dev
	"libocct-foundation-dev=$OCCT_VERSION" "libocct-modeling-algorithms-dev=$OCCT_VERSION"
	"libocct-modeling-data-dev=$OCCT_VERSION" "libocct-data-exchange-dev=$OCCT_VERSION"
	jq sqlite3 bc xxd psmisc pigz time file python3-yaml python3-pytest xz-utils
)

log() { echo "[reify-stage $(date -u +%T)] $*"; }
SUDO=$([[ $(id -u) == 0 ]] || echo sudo)
OUT=$WORK/out/reify-stage
TAR=$STAGE/reify-stage.tar

manifest_json() {
	python3 -I - "$OUT" "$REIFY_SHA" "$COLD_SHA" "$NEXTEST_VERSION" "$SCCACHE_VERSION" "$TREE_SITTER_VERSION" "$OCCT_VERSION" <<-'PY'
		import hashlib, json, pathlib, sys
		out, reify_sha, cold_sha, nextest, sccache, ts, occt = sys.argv[1:]
		bins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path(out, 'bin').iterdir())}
		print(json.dumps({'reify_sha': reify_sha, 'cold_sha': cold_sha, 'cargo_nextest': nextest,
		                  'sccache': sccache, 'tree_sitter': ts, 'occt': occt, 'bin_sha256': bins}, indent=1))
	PY
}

already_staged() {
	[[ -f $TAR ]] && tar -xOf "$TAR" reify-stage/MANIFEST.json 2>/dev/null | python3 -I -c '
import json, sys
m = json.load(sys.stdin)
want = dict(zip(["reify_sha", "cold_sha", "cargo_nextest", "sccache", "tree_sitter", "occt"], sys.argv[1:]))
sys.exit(any(m.get(k) != v for k, v in want.items()))' \
		"$REIFY_SHA" "$COLD_SHA" "$NEXTEST_VERSION" "$SCCACHE_VERSION" "$TREE_SITTER_VERSION" "$OCCT_VERSION"
}

ppa_source() {
	log "OCCT PPA key $OCCT_PPA_FINGERPRINT"
	mkdir -p "$OUT/apt"
	curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&search=0x$OCCT_PPA_FINGERPRINT" \
		>"$OUT/apt/occt-releases.asc"
	grep -q 'BEGIN PGP PUBLIC KEY BLOCK' "$OUT/apt/occt-releases.asc"
	cat >"$OUT/apt/occt-releases.sources" <<-EOF
		Types: deb
		URIs: https://ppa.launchpadcontent.net/freecad-maintainers/occt-releases/ubuntu/
		Suites: noble
		Components: main
		Signed-By: /etc/apt/keyrings/occt-releases.asc
	EOF
	printf '%s\n' "${APT_PACKAGES[@]}" >"$OUT/apt/packages.txt"
}

install_ppa_source() { # <dir holding occt-releases.{asc,sources}>
	$SUDO install -D -m644 "$1/occt-releases.asc" /etc/apt/keyrings/occt-releases.asc
	$SUDO install -D -m644 "$1/occt-releases.sources" /etc/apt/sources.list.d/occt-releases.sources
}

download_debs() {
	log "apt .debs into $STAGE/apt-archives"
	install_ppa_source "$OUT/apt"
	$SUDO mkdir -p "$STAGE/apt-archives/partial"
	$SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
	$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --download-only -o APT::Sandbox::User=root \
		-o Dir::Cache::archives="$STAGE/apt-archives" "${APT_PACKAGES[@]}" >/dev/null
	log "$(find "$STAGE/apt-archives" -maxdepth 1 -name '*.deb' | wc -l) .debs staged"
}

fetch_binaries() {
	local dl=$WORK/dl
	mkdir -p "$dl" "$OUT/bin"
	log "cargo-nextest $NEXTEST_VERSION, sccache $SCCACHE_VERSION, tree-sitter $TREE_SITTER_VERSION (prebuilt)"
	curl -fsSL "https://get.nexte.st/$NEXTEST_VERSION/linux" | tar -xz -C "$OUT/bin" cargo-nextest
	curl -fsSL -o "$dl/sccache.tgz" \
		"https://github.com/mozilla/sccache/releases/download/v$SCCACHE_VERSION/sccache-v$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz"
	echo "$(curl -fsSL "https://github.com/mozilla/sccache/releases/download/v$SCCACHE_VERSION/sccache-v$SCCACHE_VERSION-x86_64-unknown-linux-musl.tar.gz.sha256" | cut -c1-64)  $dl/sccache.tgz" |
		sha256sum -c --quiet
	tar -xzf "$dl/sccache.tgz" -C "$dl"
	install -m755 "$dl/sccache-v$SCCACHE_VERSION-x86_64-unknown-linux-musl/sccache" "$OUT/bin/sccache"
	curl -fsSL "https://github.com/tree-sitter/tree-sitter/releases/download/v$TREE_SITTER_VERSION/tree-sitter-linux-x64.gz" |
		gunzip >"$OUT/bin/tree-sitter"
	chmod 755 "$OUT/bin/tree-sitter" "$OUT/bin/cargo-nextest"
	"$OUT/bin/cargo-nextest" --version | grep -q "^cargo-nextest $NEXTEST_VERSION "
	"$OUT/bin/sccache" --version | grep -qx "sccache $SCCACHE_VERSION"
	"$OUT/bin/tree-sitter" --version | grep -q "^tree-sitter $TREE_SITTER_VERSION"
}

# A cargo and a node to drive the fetches, under $WORK only: never shipped.
fetch_tools() {
	export RUSTUP_HOME=$WORK/rustup CARGO_HOME_TOOLS=$WORK/cargo-tools
	if [[ ! -x $CARGO_HOME_TOOLS/bin/rustup ]]; then
		log "rustup + $RUST_TOOLCHAIN (fetch only)"
		curl -fsSL https://sh.rustup.rs | CARGO_HOME=$CARGO_HOME_TOOLS sh -s -- -y -q --profile minimal \
			--default-toolchain "$RUST_TOOLCHAIN" --no-modify-path >/dev/null
	fi
	if [[ ! -x $WORK/node/bin/node ]]; then
		log "node $NODE_VERSION (fetch only)"
		mkdir -p "$WORK/node"
		curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-x64.tar.xz" |
			tar -xJ -C "$WORK/node" --strip-components=1
	fi
	export PATH=$CARGO_HOME_TOOLS/bin:$WORK/node/bin:$PATH RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN
}

fetch_sources() {
	local repo=$WORK/repo sha
	rm -rf "$repo"
	mkdir -p "$repo"
	tar --no-same-owner -xf "$REIFY_REPO" -C "$repo"
	for sha in "$COLD_SHA" "$REIFY_SHA"; do
		git -C "$repo/reify" checkout -q -f --detach "$sha"
		log "cargo fetch at ${sha:0:10}"
		(cd "$repo/reify" && CARGO_HOME=$OUT/cargo-home cargo fetch --locked -q)
		for dir in gui gui/sidecar tree-sitter-reify; do
			[[ -f $repo/reify/$dir/package-lock.json ]] || continue
			log "npm cache for $dir at ${sha:0:10}"
			(cd "$repo/reify/$dir" && npm ci --ignore-scripts --no-audit --no-fund --silent --cache "$OUT/npm-cache")
			rm -rf "$repo/reify/$dir/node_modules"
		done
	done
}

if already_staged; then
	log "$TAR already holds these versions; only refreshing the .debs"
	mkdir -p "$OUT"
	tar -xf "$TAR" -C "$WORK/out" reify-stage/apt
	download_debs
	exit 0
fi

rm -rf "$WORK"
mkdir -p "$OUT"
ppa_source
download_debs
fetch_binaries
fetch_tools
fetch_sources
rm -rf "$OUT/cargo-home/.package-cache" "$OUT/npm-cache/_logs"
manifest_json >"$OUT/MANIFEST.json"
log "writing $TAR"
tar -C "$WORK/out" -cf "$TAR.tmp" reify-stage
mv "$TAR.tmp" "$TAR"
log "staged: $(du -h "$TAR" | cut -f1) tar, $(du -sh "$STAGE/apt-archives" | cut -f1) of .debs"
rm -rf "$WORK"
