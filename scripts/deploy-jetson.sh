#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || $1 == -* ]]; then
    echo "Usage: $0 user@jetson /absolute/remote/repository [git-ref (default: main)]" >&2
    echo "Optional environment: JETSON_PREFIX, JETSON_JOBS (default: 2)" >&2
    exit 1
fi

printf -v remote_command '%q ' bash -s -- "$2" "${3:-main}" \
    "${JETSON_PREFIX:-}" "${JETSON_JOBS:-2}"
ssh -o BatchMode=yes "$1" "$remote_command" <<'REMOTE'
set -euo pipefail

repository=$1
revision=$2
prefix=${3:-"$HOME/apps/whisper-service"}
jobs=$4
service=whisper_transcribe.service

if [[ $repository != /* || $prefix != /* || ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo "Repository and prefix must be absolute paths; JETSON_JOBS must be positive." >&2
    exit 1
fi
for tool in git cmake ctest ninja pkg-config flock sudo systemctl; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
cd "$repository"
if [[ $(git rev-parse --show-toplevel) != "$(pwd -P)" ]]; then
    echo "The repository argument must point to the Git repository root." >&2
    exit 1
fi

exec 9>"$(git rev-parse --git-path deploy-jetson.lock)"
flock -n 9 || { echo "Another deployment is already running." >&2; exit 1; }
if [[ -n $(git status --porcelain --untracked-files=normal) ]]; then
    echo "Remote repository has local changes; refusing to overwrite them." >&2
    exit 1
fi
if [[ ! -x "$prefix/bin/whisper_transcribe" ]]; then
    echo "Install the service first; no executable found in $prefix/bin." >&2
    exit 1
fi
for dependency in "$prefix/deps/whisper" "$prefix/deps/ffmpeg/lib/pkgconfig"; do
    [[ -d $dependency ]] || { echo "Missing private dependency directory: $dependency" >&2; exit 1; }
done
sudo -n true
sudo -n systemctl cat "$service" >/dev/null

git fetch --no-tags -- origin "$revision"
commit=$(git rev-parse --verify 'FETCH_HEAD^{commit}')
git checkout --detach "$commit"
echo "Building revision $commit"

build="$repository/build/deploy-jetson"
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

stage=$(mktemp -d)
replacement=
cleanup() {
    if [[ -n $replacement ]]; then
        rm -f -- "$replacement"
    fi
    rm -f -- "$stage$prefix/bin/whisper_transcribe" \
        "$stage$prefix/share/whisper-transcriber/whisper_transcribe.service"
    # Remove only empty staging directories, never follow or recursively delete paths.
    find "$stage" -depth -type d -empty -delete
}
trap cleanup EXIT
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
