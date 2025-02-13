# Whisper Transcriber Service
## Author Egbert Nierop (Nierop Computer Vision, Netherlands)

## Overview
The **Whisper Transcriber Service** is a C++-based server application that provides speech-to-text transcription using **OpenAI's Whisper model**. Unlike the official Whisper implementation, this project:

- Uses a **newer version of FFmpeg**, allowing better support for modern audio formats.
- **Automatically downloads models** using recent lib curl  if not available locally in .cache/whisper using the CURL api
- Runs as a **system service** on Linux (or as a Windows service).

This project is primarily developed and tested on **Linux** but is compatible with Windows.

## Features
- **Real-time transcription** of audio via an HTTP API.
- **Supports multiple audio formats** via FFmpeg.
- **Optimized for GPU acceleration** (CUDA support available).
- **Supports multiple concurrent transcription requests**.
- **Configurable language detection and translation** (auto-detect, translate to English, or custom language prompts).

## Requirements
### Dependencies
Ensure the following dependencies are installed:

- **CMake** (version 3.10 or later)
- **C++17 or later**
- **Whisper.cpp** (precompiled with shared libraries)
   assuming you cloned whisper.cpp and compiled and installed it succesfully 
- **FFmpeg** (latest version, not limited to 4.4)
  make sure if you compile this yourselves, you include the flags --enable-shared --disable-static
  e.g. 
  ./configure --prefix=/usr/local --enable-gpl --enable-nonfree --enable-libx264 --enable-libx265 --enable-libvpx --enable-libopus --enable-libfdk-aac --enable-libmp3lame --enable-libass --enable-libfreetype --enable-libvorbis --enable-zlib --enable-shared --disable-static --enable-cuda-nvcc
- **CUDA** (optional)
  if cuda not available you can test but slow

- **CURL** (for model downloading)
- **Threads** (multi-threading support)

On **Ubuntu**, install dependencies with:
```sh
sudo apt update
sudo apt install cmake g++  libavformat-dev libavcodec-dev libavutil-dev libswresample-dev libcurl4-openssl-dev
```

## Installation
### Clone and Build
```sh
git clone https://github.com/egbertn/WisperService.git
cd WhisperService
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### Install as a System Service (Linux)
After installation, enable and start the service:
```sh
sudo systemctl enable whisper_transcribe.service
sudo systemctl start whisper_transcribe.service
```

To check the service status:
```sh
sudo systemctl status whisper_transcribe.service
```

### Windows Installation
On Windows, the service is installed automatically:
```sh
sc create WhisperService binPath= C:\whisper_service\whisper_transcribe.exe start= auto
sc start WhisperService
```

## Running the Transcription Service
You can manually start the server with:
```sh
./whisper_transcribe -p 8007 -m small
```
This runs the service on **port 8007** using the **small Whisper model**.

## API Usage
### Transcription Request
Send a POST request with an audio file:
```sh
curl -X POST --data-binary @test.webm \
     -H "Content-Type: application/octet-stream" \
     "http://127.0.0.1:8007/?lang=nl&translate=false"
```

### Query Parameters
| Parameter  | Description |
|------------|------------|
| `lang`     | Language code (e.g., `nl`, `en`, `auto`) |
| `translate` | Translate output (`true` for English translation) |
| `hint`      | Initial prompt for model guidance |

## Model Management
### Downloading a Whisper Model
Models are **are downloaded automatically**. Models will be downloaded automatically from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp) and placed  in `~/.cache/whisper/`.

If you want to download yourself, do so like this e.g.
```sh
mkdir -p ~/.cache/whisper
wget -O ~/.cache/whisper/ggml-small.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

## Development Notes
- **FFmpeg** integration allows better support for audio formats.
- **CMake build system** ensures cross-platform compatibility.
- **Multi-threaded execution** allows efficient transcription handling.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

## License
This project is licensed under the MIT License.

