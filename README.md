# Whisper Transcriber Service

Author: Egbert Nierop (Nierop Computer Vision, Netherlands).

A C++17 HTTP service using Whisper.cpp for transcription and FFmpeg for audio
decoding. Every upload is converted to mono, 16 kHz float PCM before inference.
Whisper is loaded on the first valid transcription request, reused while active,
and unloaded after 10 minutes without a completed transcription. The lightweight
HTTP service remains running; the next request reloads Whisper and may take
longer. Requests do not share previous transcription context or prompts.

## HTTP API

Send **raw file bytes**, not multipart/form-data:

```sh
curl --fail-with-body --data-binary @recording.m4a \
  -H "Authorization: Bearer $WHISPER_API_KEY" \
  -H 'Content-Type: application/octet-stream' \
  'https://api.adccure.nl/?lang=nl&translate=false'
```

JSON is the default response. This is a breaking change from the original
plain-text API; add `format=text` to retain a plain-text transcript.

```json
{
  "text": " Hallo wereld.",
  "language": "nl",
  "task": "transcribe",
  "model": "small",
  "processing_seconds": 0.8,
  "segments": [
    {
      "start": 0.0,
      "end": 1.5,
      "text": " Hallo wereld.",
      "no_speech_probability": 0.02
    }
  ],
  "audio": {
    "container": "mov,mp4,m4a,3gp,3g2,mj2",
    "codec": "aac",
    "duration_seconds": 1.5,
    "original_sample_rate": 48000,
    "original_channels": 2,
    "sample_rate": 16000,
    "channels": 1
  }
}
```

All durations and timestamps are in **seconds**. Duration is calculated from
decoded, resampled samples, not untrusted container tags. Segment endpoints are
bounded by the audio duration. `language` describes the source language, including
when translating to English. `no_speech_probability` is Whisper's segment
no-speech estimate, **not** a general transcription confidence score.
Processing time includes decoding and waiting for the model, but not uploading.

| Query parameter | Default | Meaning |
|---|---|---|
| `lang` | `auto` | Whisper language code, e.g. `nl`, `en`, or automatic detection |
| `translate` | `false` | `true` translates speech to English; only `true`/`false` accepted |
| `hint` | empty | Initial prompt, at most 4096 bytes; URL-encode its contents |
| `format` | `json` | `json` or `text` |

`GET /health` returns readiness, model and selected device. The HTTP listener
starts without loading Whisper; the model loads when the first valid transcription
request arrives.

### Limits and errors

- At most **32 MiB per upload** and **600 seconds of decoded audio**.
- One inference at a time; at most four admitted requests, including the active
  request. Overload returns `503` with `Retry-After: 5`.
- HTTP workers and the underlying connection queue are also bounded. Connections
  beyond that lower-level queue can be closed without an HTTP response.
- Read/write timeouts are 30 seconds; these are socket I/O timeouts, not a
  transcription deadline.
- Empty uploads/invalid parameters: `400`; oversized/overlong audio: `413`;
  multipart uploads: `415`; unsupported/corrupt audio: `422`; internal failures:
  `500`. Errors use JSON even when `format=text`.

```json
{"error":{"status":422,"message":"Cannot open audio: Invalid data found when processing input"}}
```

Only the first selected audio stream is transcribed. External media references
are blocked. Changing sample formats within a single stream is rejected.
Supported formats depend on the linked FFmpeg decoders and demuxers, not the
filename extension. The private build below enables WAV/PCM, MP3, AAC/M4A,
WebM/Opus, Ogg/Vorbis, FLAC, ALAC, AIFF and common WMA variants.

Every transcription request requires `Authorization: Bearer <key>`. The key is
read from `WHISPER_API_KEY` at startup; the service will not start if it is
missing or shorter than 32 characters. Keep it in a root-readable server-side
environment file, never in source control, a URL, or browser code. The service
does not provide TLS; use HTTPS at a reverse proxy and keep its upstream private.
`GET /health` remains unauthenticated for readiness checks.

### Public HTTPS proxy

The Pi's Nginx terminates TLS for `api.adccure.nl` and forwards only
`POST /` (plus `GET /health`) to the private Jetson listener at
`192.168.1.190:8007`. Install
[`deploy/nginx/api.adccure.nl.conf`](./deploy/nginx/api.adccure.nl.conf) after
Certbot has issued `/etc/letsencrypt/live/api.adccure.nl/`. The service key is
for server-to-server calls only; never expose it to browser JavaScript. A
per-person demo key and its quotas belong in the later C# backend, not in this
single service credential.

## Dependencies and build

Linux is the supported deployment platform. Requirements:

- CMake 3.18+, C++17 compiler, pkg-config, threads.
- Whisper.cpp with installed shared libraries and its CMake package. Validated
  with a local 1.7.6 CPU build and Jetson 1.9.5 CUDA build.
- FFmpeg development libraries: libavformat/libavcodec 59+,
  libavutil 57+, libswresample 4+. Validated with FFmpeg 8.
- libcurl 7.85+ development files and nlohmann-json 3.2+.
- CUDA toolkit and a CUDA-enabled Whisper backend for GPU inference.

Ubuntu development packages:

```sh
sudo apt-get install cmake ninja-build g++ pkg-config \
  libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
  libcurl4-openssl-dev nlohmann-json3-dev
```

Use private prefixes on a Jetson with an existing camera pipeline. Do not replace
the system FFmpeg or reuse the camera application's install directory.

### Isolated NVIDIA FFmpeg build

Obtain the FFmpeg source archive from the NVIDIA Jetson repository matching the
installed L4T release. The deployment used the existing
`~/camera-build-deps/ffmpeg_8.0.1-nvidia1.tar.xz` archive, whose Debian changelog
identifies `7:8.0.1-nvidia1`. The camera install and its source trees are untouched.

```sh
bash scripts/build-jetson-ffmpeg.sh \
  ~/camera-build-deps/ffmpeg_8.0.1-nvidia1.tar.xz \
  ~/apps/whisper-service/deps/ffmpeg
```

This extracts a fresh source tree, builds with two jobs and installs shared
libraries with private runtime paths. It refuses to overwrite an existing prefix
or build directory. Use a fresh prefix for subsequent builds. NVIDIA video/CUDA
codecs are unnecessary for audio decoding; CUDA acceleration belongs to Whisper.
The script keeps GPL/nonfree components disabled and disables FFmpeg networking.

### Existing Jetson Whisper build

Use the existing CUDA build from `~/whisper.cpp`; installing it below does not
rebuild it or replace its original artifacts:

```sh
cmake --install ~/whisper.cpp/build \
  --prefix ~/apps/whisper-service/deps/whisper
```

The validated Jetson build uses `GGML_CUDA=ON`, CUDA 13.2 and
`CMAKE_CUDA_ARCHITECTURES=87` for Orin. If rebuilding Whisper yourself, select the
architecture matching your GPU and give it a separate build directory.

### Build this service

```sh
PKG_CONFIG_PATH="$HOME/apps/whisper-service/deps/ffmpeg/lib/pkgconfig" \
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/apps/whisper-service/deps/whisper" \
  -DCMAKE_INSTALL_PREFIX="$HOME/apps/whisper-service"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
cmake --install build
```

Private runtime library paths are embedded in Linux executables, including
transitive dependencies, so the camera application's libraries are not selected.
Moving private dependency directories requires reconfiguration and reinstallation.

### Deploy over SSH to the Jetson

After the initial installation below, run this from your development machine:

```sh
bash scripts/deploy-jetson.sh egbert@192.168.1.190
```

The installation directory is fixed at `~/apps/whisper-service` on the Jetson
and must already contain `bin/whisper_transcribe` and `deps/`.
No existing Git checkout is needed:
the script fetches `https://github.com/egbertn/WisperService.git` into a unique
temporary workspace on the Jetson and builds there. Source, build and staging
files are removed on success or failure. Existing source/build directories in
the installation are left untouched. The Jetson needs HTTPS access to GitHub.

The repository and branch (`main`) are fixed; push local commits to `main`
first. The script checks out the fetched commit detached and refuses
concurrent deployments to the same installation directory.

The remote build uses Ninja, Release mode, two build jobs, CUDA detection and the
private dependencies under `~/apps/whisper-service/deps`. Override the job count
when necessary:

```sh
JETSON_JOBS=2 \
  bash scripts/deploy-jetson.sh egbert@192.168.1.190
```

All registered CTest tests must pass (an empty test suite is an error) before
installation or restart. A failed build or test leaves the running service and
installed executable untouched. After success the executable is staged and
atomically replaced, then `whisper_transcribe.service` is restarted and checked
for active status. This runs the unit suite, not the optional HTTP/GPU integration
suite. A restart failure is reported but does not automatically roll back.

The existing systemd unit and API-key environment file are preserved. The remote
user needs write access to the install prefix, SSH key authentication and
non-interactive sudo access for the service commands (`sudo -n`). Dependencies
must already be installed; the script does not rebuild Whisper or FFmpeg.

For a CPU-only build, use a CPU-built Whisper package and configure
`-DWHISPER_SERVICE_CUDA=OFF`. Merely disabling service CUDA detection does not
remove dependencies from an already CUDA-built Whisper library.

## Running

```sh
./build/whisper_transcribe -p 8007 -m small --device cuda --host 192.168.1.190
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `-p`, `--port` | `8007` | Port 1-65535 |
| `-m`, `--model` | `small` | Model name, e.g. `base`, `small`, `large-v3-turbo` |
| `--model-file` | empty | Use an existing model file instead of the model cache |
| `--host` | `127.0.0.1` | Bind address |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

`cuda` fails at startup if the service cannot find both a CUDA device and a
Whisper CUDA backend. `auto` explicitly logs CPU fallback.

Models are cached in `$HOME/.cache/whisper/ggml-<model>.bin`. Missing models are
downloaded over HTTPS from the Whisper.cpp Hugging Face repository (tinydiarize
uses its own repository). Downloads use unique temporary files, check transfer
success and advertised size, and are atomically renamed. Interrupted download
files do not become cached models. Downloads have a one-hour total timeout and
low-speed/connect timeouts. Existing models are validated by Whisper at startup;
cryptographic model checksum verification is not implemented.

## Systemd installation

Installation **does not automatically enable/start a service**. Configure the
generated unit explicitly before installing it:

```sh
cmake -S . -B build \
  -DSERVICE_USER=egbert \
  -DSERVICE_HOME=/var/lib/whisper \
  -DSERVICE_MODEL=small \
  -DSERVICE_HOST=192.168.1.190 \
  -DSERVICE_DEVICE=cuda
cmake --build build --parallel 2
cmake --install build
sudo install -m 644 \
  ~/apps/whisper-service/share/whisper-transcriber/whisper_transcribe.service \
  /etc/systemd/system/whisper_transcribe.service
sudo sh -c 'umask 077; printf "WHISPER_API_KEY=%s\n" "$(openssl rand -hex 32)" > /etc/whisper-transcribe.env'
sudo chown root:root /etc/whisper-transcribe.env
sudo chmod 600 /etc/whisper-transcribe.env
sudo systemctl daemon-reload
sudo systemctl enable --now whisper_transcribe.service
```

The unit runs unprivileged, creates `/var/lib/whisper` using `StateDirectory`,
sets `HOME` explicitly, mounts home directories read-only and permits writes to
its state directory. On a new host, create the configured service user first.
The required API key is loaded from `/etc/whisper-transcribe.env`, which must
exist before starting the service and should be readable only by root.
For a dedicated `whisper` user, install application/dependencies outside a
private home directory, for example under `/opt/whisper-service`.

The Jetson deployment deliberately uses `egbert` so it can read its private
application prefix, with the model cache in `/var/lib/whisper/.cache/whisper`.
The service restarts on failure. SIGINT/SIGTERM stop the listener and request
Whisper inference cancellation before model cleanup.

```sh
curl --fail http://192.168.1.190:8007/health
systemctl status whisper_transcribe.service
journalctl -u whisper_transcribe.service --no-pager -n 50
```

## Tests

Unit tests require neither a model nor a GPU. They cover 16/32-bit PCM and float
WAV, extra header chunks, stereo resampling and tail flushing, corrupt/empty
uploads, size/duration boundaries, non-finite samples, recovery after errors,
query validation, JSON escaping/metadata units and safe model paths.

```sh
ctest --test-dir build --output-on-failure
```

The SSH deployment script has separate tests (Node.js 18+, no Jetson required)
using mocked remote commands to check failure handling and restart ordering:

```sh
node --test tests/deploy_jetson_tests.mjs
```

End-to-end tests require Node.js 18+, a running service and the Whisper.cpp JFK
sample. Generate fixtures using a separate full FFmpeg installation with
libmp3lame/libopus/libvorbis encoders; the private service build is decode-focused.
Generated audio remains in the ignored build directory:

```sh
bash scripts/make-test-fixtures.sh ../whisper.cpp/samples/jfk.wav build/http-fixtures
node tests/http_integration.mjs http://192.168.1.190:8007 build/http-fixtures
```

Set `WHISPER_API_KEY` in the test process environment before running the
integration suite. It checks missing and invalid keys before exercising
transcription and response behavior.

The integration suite checks transcription and metadata for nine encodings,
automatic language detection, translation, legacy text output, HTTP errors,
upload/duration limits, overload and subsequent request recovery.

Test
```sh
curl --fail-with-body \
  --data-binary @recording.m4a \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/octet-stream" \
  "https://api.adccure.nl/?lang=nl&translate=false"
```
## License

MIT; see [LICENSE](LICENSE). Third-party dependencies retain their own licenses.
