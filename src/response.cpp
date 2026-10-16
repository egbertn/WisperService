#include "voice_transcribe.h"
#include <whisper.h>

namespace transcriber {

Options parse_options(const std::string& language, const std::string& translate,
                      const std::string& prompt, const std::string& format) {
    Options options;
    if (!language.empty()) {
        if (language != "auto" && whisper_lang_id(language.c_str()) < 0) {
            throw HttpError(400, "Unsupported language");
        }
        options.language = language;
    }
    if (!translate.empty() && translate != "true" && translate != "false") {
        throw HttpError(400, "translate must be true or false");
    }
    if (!format.empty() && format != "json" && format != "text") {
        throw HttpError(400, "format must be json or text");
    }
    if (prompt.size() > 4096) {
        throw HttpError(400, "hint exceeds 4096 bytes");
    }
    options.prompt = prompt;
    options.translate = translate == "true";
    options.text_response = format == "text";
    return options;
}

nlohmann::json metadata(const Audio& audio, const Transcript& transcript,
                        const Options& options, const std::string& model, double processing_seconds) {
    auto segments = nlohmann::json::array();
    for (const auto& segment : transcript.segments) {
        segments.push_back({{"start", segment.start}, {"end", segment.end}, {"text", segment.text},
                            {"no_speech_probability", segment.no_speech_probability}});
    }
    return {
        {"text", transcript.text}, {"language", transcript.language},
        {"task", options.translate ? "translate" : "transcribe"}, {"model", model},
        {"processing_seconds", processing_seconds}, {"segments", segments},
        {"audio", {
            {"container", audio.container}, {"codec", audio.codec},
            {"duration_seconds", static_cast<double>(audio.samples.size()) / sample_rate},
            {"original_sample_rate", audio.original_sample_rate},
            {"original_channels", audio.original_channels},
            {"sample_rate", sample_rate}, {"channels", 1}
        }}
    };
}

} // namespace transcriber
