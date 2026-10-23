#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 == -* ]]; then
    echo "Usage: $0 user@jetson" >&2
    echo "Optional environment: JETSON_JOBS (default: 6)" >&2
    exit 1
fi
printf -v remote_command '%q ' bash -s -- "${JETSON_JOBS:-6}"
ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" "$remote_command" <<'REMOTE'
set -euo pipefail
fail() { echo "Error: $*" >&2; exit 1; }
jobs=$1
[[ $jobs =~ ^[1-9][0-9]*$ ]] || fail "JETSON_JOBS must be positive."
shared="$HOME/apps/shared"
prefix="$shared/ffmpeg/8.0.1-nvidia1-l4t39.2.1-rtsp-local1"
# RTSP selects HTTP internally, even without HTTP tunneling.
protocols=file,pipe,http,tcp,udp,rtp
demuxers=rtsp,rtp,sdp,mpegts,h264,hevc,mov,matroska,wav,mp3,aac,ogg,flac,aiff,asf
muxers=mp4,mov,ipod,wav,mp3,adts,matroska,webm,ogg,flac,null
mkdir -p "$shared"
exec 9>"$shared/.ffmpeg-build.lock"
flock -n 9 || fail "Another shared FFmpeg build is running."

verify() {
    [[ -x "$prefix/bin/ffmpeg" && -x "$prefix/bin/ffprobe" ]] || fail "Incomplete FFmpeg executables at $prefix"
    local library libraries arrow resolved rest
    for library in libavformat libavcodec libavutil libswscale libswresample; do
        [[ -s "$prefix/lib/$library.so" ]] || fail "Missing $library at $prefix"
        [[ $(PKG_CONFIG_PATH="$prefix/lib/pkgconfig" PKG_CONFIG_LIBDIR="$prefix/lib/pkgconfig" pkg-config --variable=prefix "$library") == "$prefix" ]] ||
            fail "Incorrect pkg-config prefix for $library"
    done
    libraries=$(LD_LIBRARY_PATH="$prefix/lib" ldd "$prefix/bin/ffmpeg")
    if grep -q 'not found' <<< "$libraries"; then
        fail "Shared FFmpeg has unresolved libraries: $libraries"
    fi
    while read -r library arrow resolved rest; do
        case "$library" in
            libav*.so*|libsw*.so*)
                [[ $resolved == "$prefix/lib/"* ]] || fail "Unexpected FFmpeg dependency: $library => $resolved" ;;
        esac
    done <<< "$libraries"
    LD_LIBRARY_PATH="$prefix/lib" "$prefix/bin/ffmpeg" -hide_banner -decoders > "$shared/.ffmpeg-decoders"
    grep -q h264_nvv4l2dec "$shared/.ffmpeg-decoders" || fail "Missing Jetson H.264 decoder"
    grep -q hevc_nvv4l2dec "$shared/.ffmpeg-decoders" || fail "Missing Jetson HEVC decoder"
    LD_LIBRARY_PATH="$prefix/lib" "$prefix/bin/ffmpeg" -hide_banner -encoders > "$shared/.ffmpeg-encoders"
    grep -q libx264 "$shared/.ffmpeg-encoders" || fail "Missing x264 encoder"
    grep -q libx265 "$shared/.ffmpeg-encoders" || fail "Missing x265 encoder"
    grep -qw aac "$shared/.ffmpeg-encoders" || fail "Missing AAC encoder"
    rm -f -- "$shared/.ffmpeg-decoders" "$shared/.ffmpeg-encoders"
    local listing component
    listing=$(LD_LIBRARY_PATH="$prefix/lib" "$prefix/bin/ffmpeg" -hide_banner -protocols)
    listing=$(awk '/^(Input|Output):/ { protocols = 1; next } protocols && NF == 1 { print $1 }' <<< "$listing")
    for component in file pipe http tcp udp rtp; do
        grep -qx "$component" <<< "$listing" || fail "Missing protocol: $component"
    done
    for component in $listing; do
        [[ ,$protocols, == *",$component,"* ]] || fail "Unexpected protocol: $component"
    done
    # RTP selects RM; MOV and Matroska expose additional format aliases.
    verify_formats demuxers "$demuxers,rm,mp4,m4a,3gp,3g2,mj2,webm" \
        rtsp rtp sdp mpegts h264 hevc mov wav mp3 aac matroska ogg flac aiff asf
    verify_formats muxers "$muxers" mp4 mov ipod wav mp3 adts matroska webm ogg flac null
}
verify_formats() {
    local kind=$1 allowed=$2 listing component
    shift 2
    listing=$(LD_LIBRARY_PATH="$prefix/lib" "$prefix/bin/ffmpeg" -hide_banner "-$kind")
    # Match the fixed-width D/E flags, excluding device rows (third flag: d).
    listing=$(awk '/^ [D ][E ] / { print $2 }' <<< "$listing" | tr ',' '\n')
    for component in "$@"; do
        grep -qx "$component" <<< "$listing" || fail "Missing $kind component: $component"
    done
    for component in $listing; do
        [[ ,$allowed, == *",$component,"* ]] || fail "Unexpected $kind component: $component"
    done
}
if [[ -f "$prefix/.ready" ]]; then
    verify
    echo "Verified shared FFmpeg at $prefix; skipping build."
    exit 0
fi
[[ ! -e "$prefix" ]] || fail "Incomplete FFmpeg prefix exists: $prefix; inspect and move it aside before retrying."
[[ $(uname -m) == aarch64 ]] || fail "Native AARCH64 Jetson required."
[[ $(dpkg-query -W -f='${Version}' nvidia-l4t-core) == 39.2.1* ]] || fail "Jetson L4T 39.2.1 required."
for tool in curl tar sha256sum make pkg-config; do
    command -v "$tool" >/dev/null || fail "Missing required tool: $tool"
done
work=$(mktemp -d "$shared/ffmpeg-build.XXXXXXXX")
trap 'status=$?; if (( status != 0 )); then echo "FFmpeg build failed; diagnostics retained at $work" >&2; fi' EXIT
archive="$work/ffmpeg_8.0.1-nvidia1.tar.xz"
if [[ -f "$HOME/camera-build-deps/ffmpeg_8.0.1-nvidia1.tar.xz" ]]; then
    cp "$HOME/camera-build-deps/ffmpeg_8.0.1-nvidia1.tar.xz" "$archive"
else
    curl -fL --retry 2 -o "$archive" \
        https://repo.download.nvidia.com/jetson/ffmpeg/pool/main/f/ffmpeg/ffmpeg_8.0.1-nvidia1.tar.xz
fi
echo "d8ab33a31c685398112b2039f784ab71b2379e3adb6bcebfc9324f678e2d2922  $archive" | sha256sum -c -
mkdir "$work/source"
tar -xf "$archive" -C "$work/source" --strip-components=1
cd "$work/source"
echo "Building shared Jetson FFmpeg; logs: $work"
./configure --prefix="$prefix" --libdir="$prefix/lib" \
    --enable-shared --disable-static --enable-gpl \
    --enable-libx264 --enable-libx265 --enable-libv4l2 --enable-nvv4l2dec \
    --disable-autodetect --disable-doc --disable-debug \
    --disable-protocols --enable-protocol="$protocols" \
    --disable-demuxers --enable-demuxer="$demuxers" \
    --disable-muxers --enable-muxer="$muxers" \
    --disable-nvenc --disable-nvdec --disable-cuvid --disable-cuda-nvcc \
    --enable-rpath \
    --extra-cflags="-mcpu=native -I/usr/src/jetson_multimedia_api/include" \
    --extra-ldflags="-L/usr/lib/aarch64-linux-gnu/nvidia -Wl,-rpath,$prefix/lib -Wl,-rpath,/usr/lib/aarch64-linux-gnu/nvidia" \
    > "$work/configure.log" 2>&1
make -j"$jobs" > "$work/build.log" 2>&1
make install > "$work/install.log" 2>&1
verify
printf '%s\n' 'FFmpeg 8.0.1-nvidia1; L4T 39.2.1; rtsp-local1; GPL; x264/x265/AAC; NVIDIA V4L2' > "$prefix/.ready"
echo "Shared FFmpeg installed at $prefix; source and logs retained at $work"
REMOTE
