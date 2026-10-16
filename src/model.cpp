#include "voice_transcribe.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <unistd.h>
#include <curl/curl.h>

namespace transcriber {
namespace {

void validate_model(const std::string& model) {
    if (model.empty() || model.size() > 80 || model.front() == '.' ||
        !std::all_of(model.begin(), model.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '-' || c == '.' || c == '_';
        })) {
        throw std::invalid_argument("Model must be a name such as small or large-v3-turbo, not a path");
    }
}

struct CurlGlobal {
    CurlGlobal() {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("Cannot initialize CURL");
        }
    }
    ~CurlGlobal() { curl_global_cleanup(); }
};

size_t write_data(char* data, size_t size, size_t count, void* stream) {
    if (size != 0 && count > std::numeric_limits<size_t>::max() / size) {
        return 0;
    }
    return std::fwrite(data, 1, size * count, static_cast<FILE*>(stream));
}

struct TemporaryFile {
    std::string path;
    FILE* stream = nullptr;
    ~TemporaryFile() {
        if (stream) {
            std::fclose(stream);
        }
        if (!path.empty()) {
            std::error_code error;
            std::filesystem::remove(path, error);
            if (error) {
                std::cerr << "Cannot remove partial model " << path << ": " << error.message() << '\n';
            }
        }
    }
};

} // namespace

std::string model_path(const std::string& model, const std::string& home) {
    validate_model(model);
    if (home.empty()) {
        throw std::invalid_argument("HOME is not set; specify --model-file");
    }
    return (std::filesystem::path(home) / ".cache/whisper" / ("ggml-" + model + ".bin")).string();
}

void download_model(const std::string& model, const std::string& path) {
    validate_model(model);
    static CurlGlobal global;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    TemporaryFile file;
    std::string pattern = path + ".part-XXXXXX";
    int fd = mkstemp(pattern.data());
    if (fd < 0) {
        throw std::runtime_error("Cannot create temporary model file");
    }
    file.path = pattern;
    file.stream = fdopen(fd, "wb");
    if (!file.stream) {
        close(fd);
        throw std::runtime_error("Cannot open temporary model file");
    }
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
    if (!curl) {
        throw std::runtime_error("Cannot create CURL handle");
    }
    const std::string repository = model.find("tdrz") != std::string::npos
        ? "akashmjn/tinydiarize-whisper.cpp" : "ggerganov/whisper.cpp";
    const std::string url = "https://huggingface.co/" + repository + "/resolve/main/ggml-" + model + ".bin";
    char error[CURL_ERROR_SIZE]{};
    auto set = [&](CURLoption option, auto value) {
        CURLcode result = curl_easy_setopt(curl.get(), option, value);
        if (result != CURLE_OK) {
            throw std::runtime_error(std::string("Cannot configure download: ") + curl_easy_strerror(result));
        }
    };
    set(CURLOPT_URL, url.c_str());
    set(CURLOPT_ERRORBUFFER, error);
    set(CURLOPT_WRITEFUNCTION, write_data);
    set(CURLOPT_WRITEDATA, file.stream);
    set(CURLOPT_FOLLOWLOCATION, 1L);
    set(CURLOPT_FAILONERROR, 1L);
    set(CURLOPT_CONNECTTIMEOUT, 30L);
    set(CURLOPT_TIMEOUT, 3600L);
    set(CURLOPT_LOW_SPEED_LIMIT, 1024L);
    set(CURLOPT_LOW_SPEED_TIME, 60L);
    set(CURLOPT_NOSIGNAL, 1L);
    set(CURLOPT_PROTOCOLS_STR, "https");
    set(CURLOPT_REDIR_PROTOCOLS_STR, "https");
    CURLcode result = curl_easy_perform(curl.get());
    if (result != CURLE_OK) {
        throw std::runtime_error(std::string("Model download failed: ") +
                                 (error[0] ? error : curl_easy_strerror(result)));
    }
    curl_off_t expected = -1;
    result = curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &expected);
    if (result != CURLE_OK) {
        throw std::runtime_error("Cannot read model download length");
    }
    FILE* completed = file.stream;
    file.stream = nullptr;
    if (std::fclose(completed) != 0) {
        throw std::runtime_error("Cannot flush downloaded model");
    }
    auto actual = std::filesystem::file_size(file.path);
    if (actual == 0 || (expected >= 0 && actual != static_cast<std::uintmax_t>(expected))) {
        throw std::runtime_error("Downloaded model is incomplete");
    }
    std::filesystem::rename(file.path, path);
    file.path.clear();
}

} // namespace transcriber
