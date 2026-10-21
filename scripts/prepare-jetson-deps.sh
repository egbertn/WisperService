#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 == -* ]]; then
    echo "Usage: $0 user@jetson" >&2
    exit 1
fi
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$scripts/build-jetson-ffmpeg.sh" "$1"
bash "$scripts/build-whisper-jetson.sh" "$1" --install-shared
