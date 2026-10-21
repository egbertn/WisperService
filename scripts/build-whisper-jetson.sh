#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || $1 == -* || ( $# -eq 2 && $2 != --install-shared ) ]]; then
    echo "Usage: $0 user@jetson [--install-shared]" >&2
    echo "Optional environment: JETSON_JOBS (default: 6), JETSON_MODEL (absolute remote model path)" >&2
    exit 1
fi

printf -v remote_command '%q ' bash -s -- "${JETSON_JOBS:-6}" "${JETSON_MODEL:-}" "${2:-}"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$remote_command" <<'REMOTE'
set -euo pipefail

fail() { echo "Error: $*" >&2; exit 1; }
jobs=$1
install_shared=${3:-}
cuda=/usr/local/cuda
revision=v1.9.5
shared="$HOME/apps/shared"
base="$shared/models/whisper/ggml-base.bin"
model=${2:-"$base"}
prefix="$shared/whisper/0c7be696-cuda"
commit=0c7be696371069e0baa2515ed7aa08d5a5473aa6
[[ $jobs =~ ^[1-9][0-9]*$ ]] || fail "JETSON_JOBS must be positive."
if [[ -n ${2:-} ]]; then
    [[ $model == /* && -s $model ]] || fail "Model not found; set JETSON_MODEL to an existing absolute remote model path: $model"
fi

prepare_base() (
    set -euo pipefail
    [[ ! -s $base ]] || exit 0
    mkdir -p "${base%/*}"
    temporary=$(mktemp "${base}.XXXXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    if [[ -s "$HOME/whisper.cpp/models/ggml-base.bin" ]]; then
        echo "Reusing existing base model for $base"
        cp "$HOME/whisper.cpp/models/ggml-base.bin" "$temporary"
    else
        command -v curl >/dev/null || fail "Missing required tool: curl"
        echo "Downloading Whisper base model to $base"
        curl --fail --location --retry 2 --connect-timeout 30 --max-time 3600 \
            --speed-limit 1024 --speed-time 60 --output "$temporary" \
            https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
    fi
    [[ -s $temporary ]] || fail "Downloaded base model is empty."
    chmod 644 "$temporary"
    mv -- "$temporary" "$base"
)

verify_install() {
    local input libraries
    for input in include/whisper.h include/ggml.h lib/cmake/whisper/whisper-config.cmake \
        lib/cmake/ggml/ggml-config.cmake lib/libwhisper.so lib/libggml.so lib/libggml-base.so \
        lib/libggml-cpu.so lib/libggml-cuda.so bin/whisper-cli; do
        [[ -s "$prefix/$input" ]] || fail "Incomplete shared Whisper SDK: $prefix/$input"
    done
    libraries=$(LD_LIBRARY_PATH="$prefix/lib" ldd "$prefix/bin/whisper-cli")
    if grep -q 'not found' <<< "$libraries"; then
        fail "Shared Whisper SDK has unresolved libraries: $libraries"
    fi
}
if [[ -n $install_shared ]]; then
    revision=$commit
    mkdir -p "$shared"
    exec 9>"$shared/.whisper-build.lock"
    flock -n 9 || fail "Another shared Whisper build is running."
    if [[ -f "$prefix/.ready" ]]; then
        verify_install
        [[ -s "$shared/test-data/whisper/jfk.wav" ]] || fail "Shared Whisper test sample is missing."
        prepare_base
        echo "Verified shared Whisper at $prefix; skipping build."
        exit 0
    fi
    [[ ! -e $prefix ]] || fail "Incomplete Whisper prefix exists: $prefix; inspect and move it aside before retrying."
fi

for tool in git cmake ctest ninja g++ uname mktemp grep tee; do
    command -v "$tool" >/dev/null || fail "Missing required tool: $tool"
done
[[ $(uname -s) == Linux && $(uname -m) == aarch64 ]] || fail "Run this build on an ARM64 Linux Jetson."
[[ -r /proc/device-tree/model ]] || fail "Jetson device-tree model is unavailable."
board=$(tr -d '\0' < /proc/device-tree/model)
[[ $board == *Jetson*Orin* ]] || fail "Architecture 87 requires a Jetson Orin; detected: $board"
[[ -x $cuda/bin/nvcc ]] || fail "CUDA toolkit not found at $cuda."
if [[ -n $install_shared ]]; then
    prepare_base
fi
[[ $model == /* && -s $model ]] || fail "Model not found; run prepare-jetson-deps.sh first or set JETSON_MODEL to an existing absolute remote model path: $model"
"$cuda/bin/nvcc" --version
echo "Jetson: $board; CUDA architecture: 87; build jobs: $jobs"

workspace=$(mktemp -d "$HOME/whisper-jetson.XXXXXXXX")
trap 'status=$?; if (( status != 0 )); then echo "Build/verification failed; diagnostic files retained in: $workspace" >&2; fi' EXIT
source="$workspace/source"
build="$workspace/build"
echo "Build workspace (retained): $workspace"
if [[ -n $install_shared ]]; then
    git init --quiet "$source"
    git -C "$source" remote add origin https://github.com/ggml-org/whisper.cpp.git
    git -C "$source" fetch --depth 1 origin "$commit"
    git -C "$source" checkout --detach FETCH_HEAD
    [[ $(git -C "$source" rev-parse HEAD) == "$commit" ]] || fail "Unexpected Whisper revision."
else
    git clone --depth 1 --branch "$revision" \
        https://github.com/ggml-org/whisper.cpp.git "$source"
fi
git -C "$source" rev-parse HEAD | tee "$workspace/commit.txt"

cmake -S "$source" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_CUDA_COMPILER="$cuda/bin/nvcc" \
    -DCUDAToolkit_ROOT="$cuda" \
    -DWHISPER_BUILD_TESTS=ON \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DWHISPER_BUILD_SERVER=OFF \
    -DWHISPER_CURL=OFF \
    2>&1 | tee "$workspace/configure.log"
cmake --build "$build" --parallel "$jobs" 2>&1 | tee "$workspace/build.log"

grep -Eq '^GGML_CUDA:BOOL=ON$' "$build/CMakeCache.txt" || fail "CUDA was not enabled."
grep -Eq '^BUILD_SHARED_LIBS:BOOL=ON$' "$build/CMakeCache.txt" || fail "Shared libraries were not enabled."
grep -Eq '^CMAKE_CUDA_ARCHITECTURES:[^=]+=87$' "$build/CMakeCache.txt" || fail "Unexpected CUDA architecture."

# The full upstream suite requires multiple separately downloaded test models.
ctest --test-dir "$build" --output-on-failure --no-tests=error \
    -R '^(test-common-utf8|test-whisper-buffer-loader)$' \
    2>&1 | tee "$workspace/tests.log"
"$build/bin/whisper-cli" -m "$model" -f "$source/samples/jfk.wav" \
    -l en -t "$jobs" --output-txt --output-file "$workspace/jfk" \
    2>&1 | tee "$workspace/inference.log"
grep -Eq 'using CUDA[0-9]+ backend' "$workspace/inference.log" || fail "Inference did not select a CUDA backend."
grep -Eq 'CUDA[0-9]+ (compute buffer size|total size)[[:space:]]*=[[:space:]]*[1-9][0-9]*[.]' \
    "$workspace/inference.log" || fail "No nonzero CUDA allocation found; CPU fallback is not accepted."
if grep -Eq 'failed to initialize.*backend' "$workspace/inference.log"; then
    fail "Backend initialization failed."
fi
grep -Eiq 'ask not what your country can do for you' "$workspace/jfk.txt" || fail "JFK transcription did not match the expected English speech."

echo "Verified $revision: Release shared libraries, Orin CUDA (87), unit tests and GPU transcription."
echo "Build retained at: $build"
if [[ -n $install_shared ]]; then
    cmake --install "$build" --prefix "$prefix" 2>&1 | tee "$workspace/install.log"
    verify_install
    LD_LIBRARY_PATH="$prefix/lib" "$prefix/bin/whisper-cli" -m "$model" -f "$source/samples/jfk.wav" \
        -l en -t "$jobs" --output-txt --output-file "$workspace/installed-jfk" \
        2>&1 | tee "$workspace/installed-inference.log"
    grep -Eq 'using CUDA[0-9]+ backend' "$workspace/installed-inference.log" || fail "Installed SDK did not select CUDA."
    grep -Eq 'CUDA[0-9]+ (compute buffer size|total size)[[:space:]]*=[[:space:]]*[1-9][0-9]*[.]' \
        "$workspace/installed-inference.log" || fail "Installed SDK did not allocate CUDA memory."
    if grep -Eq 'failed to initialize.*backend' "$workspace/installed-inference.log"; then
        fail "Installed SDK backend initialization failed."
    fi
    grep -Eiq 'ask not what your country can do for you' "$workspace/installed-jfk.txt" || fail "Installed SDK transcription failed."
    mkdir -p "$shared/models/whisper" "$shared/test-data/whisper"
    install -m 644 "$source/samples/jfk.wav" "$shared/test-data/whisper/jfk.wav"
    cp "$workspace/commit.txt" "$prefix/commit.txt"
    cp "$build/CMakeCache.txt" "$prefix/build-config.txt"
    printf '%s\n' "$commit; Release; GGML_CUDA=ON; architecture=87" > "$prefix/.ready"
    echo "Shared Whisper SDK installed at $prefix; no services restarted."
else
    echo "Nothing installed; existing Whisper, private dependencies and running services are unchanged."
fi
REMOTE
