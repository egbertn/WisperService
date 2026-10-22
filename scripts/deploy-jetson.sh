#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 == -* ]]; then
    echo "Usage: $0 user@jetson" >&2
    echo "Optional environment: JETSON_JOBS (default: 6)" >&2
    exit 1
fi

scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$scripts/prepare-jetson-deps.sh" "$1"
printf -v remote_command '%q ' bash -s -- "${JETSON_JOBS:-6}" "${JETSON_HEALTH_URL:-http://${1#*@}:8007/health}"
ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" "$remote_command" <<'REMOTE'
set -euo pipefail

prefix="$HOME/apps/whisper-service"
jobs=$1
health_url=$2
service=whisper_transcribe.service
ffmpeg="$HOME/apps/shared/ffmpeg/8.0.1-nvidia1-l4t39.2.1-rtsp-local1"
whisper="$HOME/apps/shared/whisper/0c7be696-cuda"

if [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo "JETSON_JOBS must be positive." >&2
    exit 1
fi
for tool in git cmake ctest ninja pkg-config flock sudo systemctl curl ldd; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
if [[ ! -x "$prefix/bin/whisper_transcribe" ]]; then
    echo "Install the service first; no executable found in $prefix/bin." >&2
    exit 1
fi
for dependency in "$whisper/.ready" "$ffmpeg/.ready"; do
    [[ -s $dependency ]] || { echo "Missing verified shared dependency: $dependency" >&2; exit 1; }
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

export PKG_CONFIG_PATH="$ffmpeg/lib/pkgconfig"
export PKG_CONFIG_LIBDIR="$ffmpeg/lib/pkgconfig"
export LD_LIBRARY_PATH="$whisper/lib:$ffmpeg/lib"
export PATH="$ffmpeg/bin:$PATH"
cmake -S "$repository" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DWHISPER_SERVICE_CUDA=ON \
    -DCMAKE_PREFIX_PATH="$whisper" \
    -DWHISPER_SERVICE_WHISPER_ROOT="$whisper" \
    -DWHISPER_SERVICE_FFMPEG_ROOT="$ffmpeg" \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_INSTALL_BINDIR=bin
cmake --build "$build" --parallel "$jobs"
ctest --test-dir "$build" --output-on-failure --no-tests=error

DESTDIR="$stage" cmake --install "$build"

libraries=$(ldd "$stage$prefix/bin/whisper_transcribe")
if grep -q 'not found' <<< "$libraries"; then
    echo "Staged service has missing libraries: $libraries" >&2
    exit 1
fi
for library in libavcodec libavformat libavutil libswresample libwhisper libggml libggml-base libggml-cpu libggml-cuda; do
    case "$library" in
        libav*|libsw*) expected="$ffmpeg/lib" ;;
        *) expected="$whisper/lib" ;;
    esac
    if ! grep -F "$library.so" <<< "$libraries" | grep -Fq "=> $expected/"; then
        echo "Staged service does not resolve $library from $expected." >&2
        exit 1
    fi
done
while read -r library arrow resolved rest; do
    case "$library" in
        libwhisper.so*|libggml*.so*) expected="$whisper/lib/" ;;
        libav*.so*|libsw*.so*) expected="$ffmpeg/lib/" ;;
        *) continue ;;
    esac
    if [[ $resolved != "$expected"* ]]; then
        echo "Unexpected dependency: $library => $resolved" >&2
        exit 1
    fi
done <<< "$libraries"

# Preserve the existing unit, API key, model, bind address and service user.
mkdir -p "$stage/unit"
printf '[Service]\nEnvironment="LD_LIBRARY_PATH=%s/lib:%s/lib"\n' "$whisper" "$ffmpeg" \
    > "$stage/unit/20-shared-libraries.conf"
sudo -n install -d -m 755 "/etc/systemd/system/$service.d"
sudo -n install -m 644 "$stage/unit/20-shared-libraries.conf" \
    "/etc/systemd/system/$service.d/20-shared-libraries.conf"
sudo -n systemctl daemon-reload

# Rename on the same filesystem; do not overwrite the running executable in place.
replacement=$(mktemp "$prefix/bin/.whisper_transcribe.XXXXXX")
install -m 755 "$stage$prefix/bin/whisper_transcribe" "$replacement"
mv -f -- "$replacement" "$prefix/bin/whisper_transcribe"
replacement=

sudo -n systemctl restart "$service"
sudo -n systemctl is-active --quiet "$service"
health=$(curl --fail --silent --show-error --retry 20 --retry-connrefused --retry-all-errors \
    --retry-delay 2 --retry-max-time 90 --max-time 5 "$health_url")
if ! grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"' <<< "$health"; then
    echo "Service health endpoint did not report ready: $health" >&2
    exit 1
fi
pid=$(sudo -n systemctl show "$service" -p MainPID --value)
[[ $pid =~ ^[1-9][0-9]*$ ]] || { echo "Service has no running PID." >&2; exit 1; }
maps=$(sudo -n cat "/proc/$pid/maps")
for library in libavcodec libavformat libavutil libswresample libwhisper libggml libggml-base libggml-cpu libggml-cuda; do
    case "$library" in
        libav*|libsw*) expected="$ffmpeg/lib" ;;
        *) expected="$whisper/lib" ;;
    esac
    if ! grep -Fq "$expected/$library.so" <<< "$maps"; then
        echo "Running service does not use the expected $library from $expected." >&2
        exit 1
    fi
done
if grep -E '/lib(whisper|ggml[^/]*|avcodec|avformat|avutil|swresample)\.so' <<< "$maps" |
    grep -Fv -e "$whisper/lib/" -e "$ffmpeg/lib/"; then
    echo "Running service loaded libraries outside the shared installations." >&2
    exit 1
fi
echo "Deployed $commit; $service is active."
REMOTE
