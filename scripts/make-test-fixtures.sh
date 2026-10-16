#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 /path/to/whisper.cpp/samples/jfk.wav /path/to/fixture-directory" >&2
    exit 1
fi
input=$(realpath "$1")
mkdir -p "$2"
output=$(realpath "$2")
ffmpeg -nostdin -v error -y -i "$input" -c:a pcm_s16le "$output/speech.wav"
ffmpeg -nostdin -v error -y -i "$input" -c:a pcm_s32le "$output/speech32.wav"
ffmpeg -nostdin -v error -y -i "$input" -c:a pcm_f32le "$output/speech-float.wav"
ffmpeg -nostdin -v error -y -i "$input" -ar 48000 -ac 2 -c:a pcm_s16le "$output/speech-stereo.wav"
ffmpeg -nostdin -v error -y -i "$input" -c:a libmp3lame "$output/speech.mp3"
ffmpeg -nostdin -v error -y -i "$input" -c:a aac "$output/speech.m4a"
ffmpeg -nostdin -v error -y -i "$input" -c:a libopus "$output/speech.webm"
ffmpeg -nostdin -v error -y -i "$input" -c:a libvorbis "$output/speech.ogg"
ffmpeg -nostdin -v error -y -i "$input" -c:a flac "$output/speech.flac"
ffmpeg -nostdin -v error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 601 -c:a flac "$output/too-long.flac"
