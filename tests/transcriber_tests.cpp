#include "voice_transcribe.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>

using namespace transcriber;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<class F>
void expect_http(int status, F operation) {
    try {
        operation();
    } catch (const HttpError& error) {
        require(error.status == status, "Incorrect HTTP status");
        return;
    }
    throw std::runtime_error("Expected HttpError");
}

void little_endian(std::string& bytes, uint32_t value, int count) {
    for (int i = 0; i < count; ++i) {
        bytes.push_back(static_cast<char>((value >> (8 * i)) & 255));
    }
}

std::string wav(int rate, int channels, int bits, bool floating = false, bool extra_chunk = false,
                int frames = -1) {
    if (frames < 0) {
        frames = rate;
    }
    uint32_t data_size = frames * channels * (bits / 8);
    std::string bytes = "RIFF";
    little_endian(bytes, 36 + data_size + (extra_chunk ? 12 : 0), 4);
    bytes += "WAVEfmt ";
    little_endian(bytes, 16, 4);
    little_endian(bytes, floating ? 3 : 1, 2);
    little_endian(bytes, channels, 2);
    little_endian(bytes, rate, 4);
    little_endian(bytes, rate * channels * (bits / 8), 4);
    little_endian(bytes, channels * (bits / 8), 2);
    little_endian(bytes, bits, 2);
    if (extra_chunk) {
        bytes += "JUNK";
        little_endian(bytes, 4, 4);
        bytes += "test";
    }
    bytes += "data";
    little_endian(bytes, data_size, 4);
    for (int i = 0; i < frames; ++i) {
        float sample = 0.25f * std::sin(2.0 * 3.141592653589793 * 440 * i / rate);
        for (int channel = 0; channel < channels; ++channel) {
            if (floating) {
                uint32_t raw = 0;
                std::memcpy(&raw, &sample, sizeof(raw));
                little_endian(bytes, raw, 4);
            } else if (bits == 16) {
                little_endian(bytes, static_cast<uint16_t>(static_cast<int16_t>(sample * 32768)), 2);
            } else {
                little_endian(bytes, static_cast<uint32_t>(static_cast<int32_t>(sample * 2147483648.0)), 4);
            }
        }
    }
    return bytes;
}

} // namespace

int main() {
    int failures = 0;
    auto test = [&](const char* name, const std::function<void()>& operation) {
        try {
            operation();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    test("16-bit mono WAV and duration", [] {
        auto audio = decode_audio(wav(16000, 1, 16));
        require(audio.samples.size() == 16000, "Wrong sample count");
        require(audio.codec == "pcm_s16le", "Wrong codec");
        require(audio.original_sample_rate == 16000 && audio.original_channels == 1, "Wrong source properties");
        require(std::abs(audio.samples[9] - 0.25f) < 0.002f, "Incorrect PCM scaling");
    });
    test("32-bit integer PCM WAV", [] {
        auto audio = decode_audio(wav(16000, 1, 32));
        require(audio.samples.size() == 16000, "32-bit PCM rejected or truncated");
        require(std::abs(audio.samples[9] - 0.25f) < 0.002f, "Incorrect 32-bit scaling");
    });
    test("32-bit float PCM WAV", [] {
        auto audio = decode_audio(wav(16000, 1, 32, true));
        require(audio.samples.size() == 16000, "Float PCM rejected or truncated");
        require(audio.codec == "pcm_f32le", "Wrong float PCM codec");
    });
    test("48 kHz stereo resampling and decoder tail", [] {
        auto audio = decode_audio(wav(48000, 2, 16, false, false, 48003));
        require(audio.original_sample_rate == 48000 && audio.original_channels == 2, "Source properties lost");
        require(audio.samples.size() >= 16001 && audio.samples.size() <= 16002, "Resampler tail lost");
        for (float sample : audio.samples) {
            require(std::isfinite(sample), "Invalid output sample");
        }
    });
    test("WAV with nonstandard header chunks", [] {
        require(decode_audio(wav(16000, 1, 16, false, true)).samples.size() == 16000, "Extra chunk mishandled");
    });
    test("empty and invalid uploads", [] {
        expect_http(400, [] { decode_audio(""); });
        expect_http(422, [] { decode_audio("not an audio file"); });
        expect_http(422, [] { decode_audio(wav(16000, 1, 16, false, false, 0)); });
    });
    test("duration and upload size limits", [] {
        require(decode_audio(wav(16000, 1, 16), 1).samples.size() == 16000, "Exact duration limit rejected");
        expect_http(413, [] { decode_audio(wav(16000, 1, 16, false, false, 32000), 1); });
        std::string oversized(max_upload_bytes + 1, '\0');
        expect_http(413, [&] { decode_audio(oversized); });
        oversized.resize(max_upload_bytes);
        expect_http(422, [&] { decode_audio(oversized); });
    });
    test("non-finite float samples rejected", [] {
        auto bytes = wav(16000, 1, 32, true);
        uint32_t nan = 0x7fc00000;
        std::memcpy(bytes.data() + 44, &nan, sizeof(nan));
        expect_http(422, [&] { decode_audio(bytes); });
    });
    test("malformed audio does not poison later requests", [] {
        for (int i = 0; i < 20; ++i) {
            expect_http(422, [] { decode_audio("broken"); });
        }
        require(decode_audio(wav(16000, 1, 16)).samples.size() == 16000, "Decoder failed after invalid uploads");
    });
    test("query defaults and text compatibility", [] {
        auto defaults = parse_options("", "", "", "");
        require(defaults.language == "auto" && !defaults.translate && !defaults.text_response, "Wrong defaults");
        auto options = parse_options("nl", "true", "names", "text");
        require(options.language == "nl" && options.translate && options.text_response && options.prompt == "names",
                "Options ignored");
    });
    test("invalid query parameters", [] {
        expect_http(400, [] { parse_options("not-a-language", "", "", ""); });
        expect_http(400, [] { parse_options("", "yes", "", ""); });
        expect_http(400, [] { parse_options("", "", "", "xml"); });
        expect_http(400, [] { parse_options("", "", std::string(4097, 'x'), ""); });
    });
    test("metadata JSON shape, seconds and escaping", [] {
        Audio audio;
        audio.samples.resize(24000);
        audio.container = "wav";
        audio.codec = "pcm_s16le";
        audio.original_sample_rate = 48000;
        audio.original_channels = 2;
        Transcript transcript{"A \"quote\"\nand backslash \\", "nl",
            {{0.1, 1.2, "A \"quote\"\nand backslash \\", 0.05f}}};
        auto output = metadata(audio, transcript, Options{}, "small", 0.75);
        auto json = nlohmann::json::parse(output.dump());
        require(json["text"] == transcript.text && json["language"] == "nl", "Text not JSON escaped");
        require(json["audio"]["duration_seconds"] == 1.5, "Duration not in seconds");
        require(json["audio"]["sample_rate"] == 16000 && json["audio"]["channels"] == 1, "Wrong normalized properties");
        require(json["audio"]["original_channels"] == 2, "Original channels lost");
        require(json["segments"][0]["start"] == 0.1 && json["segments"][0]["end"] == 1.2, "Wrong timestamps");
        require(json["model"] == "small" && json["processing_seconds"] == 0.75, "Model/timing missing");
        Options translated;
        translated.translate = true;
        require(metadata(audio, transcript, translated, "small", 1)["task"] == "translate", "Wrong translation task");
    });
    test("safe model cache paths", [] {
        require(model_path("large-v3-turbo-q5_0", "/tmp").empty() == false, "Quantized model name rejected");
    });
    test("model path validation", [] {
        require(model_path("small.en", "/tmp") == "/tmp/.cache/whisper/ggml-small.en.bin", "Wrong cache path");
        for (const auto& name : {"../small", "/tmp/model", "", ".hidden"}) {
            bool rejected = false;
            try {
                model_path(name, "/tmp");
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "Unsafe model path accepted");
        }
        bool missing_home = false;
        try {
            model_path("small", "");
        } catch (const std::invalid_argument&) {
            missing_home = true;
        }
        require(missing_home, "Missing HOME accepted");
    });
    return failures == 0 ? 0 : 1;
}
