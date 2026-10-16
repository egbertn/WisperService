#ifndef VOICE_TRANSCRIBE_H
#define VOICE_TRANSCRIBE_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>

namespace transcriber {

constexpr int sample_rate = 16000;
constexpr std::size_t max_upload_bytes = 32 * 1024 * 1024;
constexpr int max_audio_seconds = 600;

class HttpError : public std::runtime_error {
public:
    HttpError(int status, const std::string& message) : std::runtime_error(message), status(status) {}
    int status;
};

struct Audio {
    std::vector<float> samples;
    std::string container;
    std::string codec;
    int original_sample_rate = 0;
    int original_channels = 0;
};

struct Options {
    std::string language = "auto";
    std::string prompt;
    bool translate = false;
    bool text_response = false;
};

struct Segment {
    double start;
    double end;
    std::string text;
    float no_speech_probability;
};

struct Transcript {
    std::string text;
    std::string language;
    std::vector<Segment> segments;
};

Audio decode_audio(std::string_view input, int duration_limit = max_audio_seconds);
Options parse_options(const std::string& language, const std::string& translate,
                      const std::string& prompt, const std::string& format);
nlohmann::json metadata(const Audio& audio, const Transcript& transcript,
                        const Options& options, const std::string& model, double processing_seconds);
std::string model_path(const std::string& model, const std::string& home);
void download_model(const std::string& model, const std::string& path);

} // namespace transcriber

#endif
