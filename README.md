# Whisper Transcriber Service

Author: Egbert Nierop.

A C++17 HTTP service that transcribes audio with Whisper.cpp and FFmpeg.

## HTTP API

Send raw audio bytes to `POST /` with `Content-Type: application/octet-stream`
and an `Authorization` header:

```sh
curl --fail-with-body --data-binary @recording.m4a \
  -H "Authorization: $WHISPER_API_KEY" \
  -H "Content-Type: application/octet-stream" \
  "http://localhost:8007/?lang=nl"
```

The default response is JSON. Add `format=text` for a plain-text transcript.
Optional query parameters are `lang`, `translate` (`true` or `false`), `hint`,
and `format` (`json` or `text`).

`GET /health` reports whether the service is ready. Transcription requests
require an API key supplied through the `WHISPER_API_KEY` environment variable.
Keep the key private. The service does not provide TLS; use a trusted HTTPS
proxy when access beyond the local machine is required.

Uploads are limited to 32 MiB and 600 seconds of decoded audio. Supported
formats depend on the FFmpeg build.

## Build and test

Requirements: Linux, CMake 3.18+, a C++17 compiler, FFmpeg development
libraries, libcurl, nlohmann-json 3.12+, and Whisper.cpp with its CMake package.
CUDA is optional.

Configure, build, and run unit tests:

```sh
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run locally:

```sh
export WHISPER_API_KEY="replace-with-a-private-key-of-at-least-32-characters"
./build/whisper_transcribe
```

The service listens only on the local machine by default. See `--help` for
command-line options.

## License

MIT; see [LICENSE](LICENSE). Third-party dependencies retain their own licenses.
