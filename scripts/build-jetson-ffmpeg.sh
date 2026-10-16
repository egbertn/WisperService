#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 /path/ffmpeg_8.0.1-nvidia1.tar.xz /absolute/private/prefix" >&2
    exit 1
fi
archive=$(realpath "$1")
prefix=$(realpath -m "$2")
work="${prefix}-build"
if [[ -e "$prefix" || -e "$work" ]]; then
    echo "Refusing to overwrite existing FFmpeg prefix or build: $prefix, $work" >&2
    exit 1
fi
mkdir -p "$work/source" "$work/build"
tar -xf "$archive" -C "$work/source" --strip-components=1
cd "$work/build"
"$work/source/configure" \
    --prefix="$prefix" --libdir="$prefix/lib" \
    --enable-shared --disable-static --enable-rpath \
    --disable-autodetect --disable-everything --disable-network \
    --disable-doc --disable-debug --disable-avdevice --disable-swscale \
    --disable-gpl --disable-nonfree --disable-version3 \
    --enable-avcodec --enable-avformat --enable-avfilter --enable-swresample \
    --enable-ffmpeg --enable-ffprobe \
    --enable-protocol=file,pipe \
    --enable-demuxer=wav,mp3,aac,mov,matroska,ogg,flac,aiff,asf \
    --enable-decoder=pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,pcm_f32le,pcm_f32be,pcm_f64le,pcm_f64be,pcm_u8,pcm_alaw,pcm_mulaw,mp3,mp3float,mp2,mp2float,aac,aac_fixed,opus,vorbis,flac,alac,wmav1,wmav2 \
    --enable-parser=mpegaudio,aac,aac_latm,opus,vorbis,flac \
    --enable-encoder=pcm_s16le,pcm_s32le,pcm_f32le,aac,flac,alac \
    --enable-muxer=wav,adts,mp4,ipod,flac,matroska \
    --enable-filter=aresample,aformat,anull
make -j2
make install
"$prefix/bin/ffmpeg" -version
