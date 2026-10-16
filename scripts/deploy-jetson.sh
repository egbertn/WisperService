#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 == -* ]]; then
    echo "Usage: $0 user@jetson" >&2
    echo "Optional environment: JETSON_JOBS (default: 2)" >&2
    exit 1
fi

printf -v remote_command '%q ' bash -s -- "${JETSON_JOBS:-2}"
ssh -o BatchMode=yes "$1" "$remote_command" <<'REMOTE'
set -euo pipefail

prefix="$HOME/apps/whisper-service"
jobs=$1
service=whisper_transcribe.service

if [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo "JETSON_JOBS must be positive." >&2
    exit 1
fi
for tool in git cmake ctest ninja pkg-config flock sudo systemctl; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
if [[ ! -x "$prefix/bin/whisper_transcribe" ]]; then
    echo "Install the service first; no executable found in $prefix/bin." >&2
    exit 1
fi
for dependency in "$prefix/deps/whisper" "$prefix/deps/ffmpeg/lib/pkgconfig"; do
    [[ -d $dependency ]] || { echo "Missing private dependency directory: $dependency" >&2; exit 1; }
done
sudo -n true
sudo -n systemctl cat "$service" >/dev/null

exec 9>"$prefix/.deploy-jetson.lock"
flock -n 9 || { echo "Another deployment to this install prefix is already running." >&2; exit 1; }
workspace=$(mktemp -d "${TMPDIR:-/tmp}/whisper-deploy.XXXXXXXX")
replacement=
cleanup() {
    if [[ -n $replacement ]]; then
        rm -f -- "$replacement"
    fi
    rm -rf -- "$workspace"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
repository="$workspace/source"
build="$workspace/build"
stage="$workspace/stage"
echo "Temporary workspace: $workspace"
echo "Install prefix: $prefix"
git init --quiet "$repository"
cd "$repository"
git remote add origin https://github.com/egbertn/WisperService.git
git fetch --no-tags -- origin main
commit=$(git rev-parse --verify 'FETCH_HEAD^{commit}')
git checkout --detach "$commit"
echo "Building revision $commit"

export PKG_CONFIG_PATH="$prefix/deps/ffmpeg/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
cmake -S "$repository" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DWHISPER_SERVICE_CUDA=ON \
    -DCMAKE_PREFIX_PATH="$prefix/deps/whisper" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_BINDIR=bin
cmake --build "$build" --parallel "$jobs"
ctest --test-dir "$build" --output-on-failure --no-tests=error

DESTDIR="$stage" cmake --install "$build"

# Rename on the same filesystem; do not overwrite the running executable in place.
replacement=$(mktemp "$prefix/bin/.whisper_transcribe.XXXXXX")
install -m 755 "$stage$prefix/bin/whisper_transcribe" "$replacement"
mv -f -- "$replacement" "$prefix/bin/whisper_transcribe"
replacement=

sudo -n systemctl restart "$service"
sudo -n systemctl is-active --quiet "$service"
echo "Deployed $commit; $service is active."
REMOTE
