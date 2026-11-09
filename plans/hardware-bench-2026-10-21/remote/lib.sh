# Shared by provision.sh, stage.sh and localize.sh (sourced, never run).
# Callers define log() first.

# Everything the DF probes, the DF gates and the reify compile probes need
# from apt. One list: provision.sh installs it, stage.sh caches its .debs,
# localize.sh installs it from that cache.
DF_APT_PACKAGES=(
	git curl ca-certificates build-essential pkg-config cmake clang
	bubblewrap python3 xz-utils time procps rsync
)

# Every toolchain version is pinned here, so a box provisioned from the
# internet (provision.sh) and one restored from the volume (stage.sh +
# localize.sh) carry identical tools. host.json records what was installed.
UV_VERSION=0.11.6
NODE_VERSION=v22.23.3
RUST_TOOLCHAIN=1.96.0

# timed <label> <command...>: run it and log how long it took.
timed() {
	local label=$1 t0=$SECONDS
	shift
	"$@"
	log "$label: $((SECONDS - t0)) s"
}

# as_root <command...>: run as root whether we are root (a container) or a
# sudoer (a VM).
as_root() {
	if ((EUID == 0)); then
		"$@"
	elif command -v sudo >/dev/null; then
		sudo "$@"
	else
		echo "need root for: $*" >&2
		return 1
	fi
}

apt_noninteractive() { as_root env DEBIAN_FRONTEND=noninteractive apt-get "$@"; }

install_uv() {
	[[ $(~/.local/bin/uv --version 2>/dev/null) == "uv $UV_VERSION"* ]] && return
	log "uv $UV_VERSION"
	curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh >/dev/null
}

# Always the pinned node under ~/.local/node, whatever the image ships.
install_node() {
	[[ $(~/.local/node/bin/node --version 2>/dev/null) == "$NODE_VERSION" ]] && return
	local base=https://nodejs.org/dist/$NODE_VERSION tarball=node-$NODE_VERSION-linux-x64.tar.xz scratch
	log "node $NODE_VERSION"
	scratch=$(mktemp -d)
	curl -fsSL "$base/$tarball" -o "$scratch/$tarball"
	(cd "$scratch" && curl -fsSL "$base/SHASUMS256.txt" | grep " $tarball\$" | sha256sum -c --quiet -)
	rm -rf ~/.local/node
	mkdir -p ~/.local/node
	tar -xJf "$scratch/$tarball" -C ~/.local/node --strip-components=1
	rm -rf "$scratch"
}

install_rust() {
	if [[ ! -x ~/.cargo/bin/rustup ]]; then
		log "rustup"
		curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path >/dev/null
	fi
	~/.cargo/bin/rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal -c rustfmt -c clippy >/dev/null 2>&1
	~/.cargo/bin/rustup default "$RUST_TOOLCHAIN" >/dev/null 2>&1
}

# scripts/run_vllm_eval.py reads this key at import, and the orchestrator
# tests import it: without the file, collection errors stop that module early.
install_dummy_key_and_git_identity() {
	mkdir -p ~/.ssh
	[[ -e ~/.ssh/id_runpod.pub ]] || ssh-keygen -q -t ed25519 -N '' -C hwbench -f ~/.ssh/id_runpod
	git config --global user.name >/dev/null || git config --global user.name "hwbench"
	git config --global user.email >/dev/null || git config --global user.email "hwbench@localhost"
}

install_user_toolchains() {
	install_uv
	install_node
	install_rust
	install_dummy_key_and_git_identity
}

# flag_exit_status <dir>: leave <dir>/DONE or <dir>/FAILED when the script
# ends, for bench.sh's poll loop. The script may define cleanup_on_exit.
flag_exit_status() {
	FLAG_DIR=$1
	mkdir -p "$FLAG_DIR"
	rm -f "$FLAG_DIR/DONE" "$FLAG_DIR/FAILED"
	trap 'rc=$?; declare -F cleanup_on_exit >/dev/null && cleanup_on_exit; ((rc == 0)) && touch "$FLAG_DIR/DONE" || echo "rc=$rc at line $LINENO" >"$FLAG_DIR/FAILED"' EXIT
}
