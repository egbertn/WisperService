#include "voice_transcribe.h"
#include "httplib.h"
#include <whisper.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#ifdef WHISPER_SERVICE_HAS_CUDA
#include <cuda_runtime_api.h>
#include <ggml-backend.h>
#endif

namespace {
using namespace transcriber;
volatile std::sig_atomic_t interrupted = 0;

void handle_signal(int) { interrupted = 1; }

struct Config {
    std::string host = "0.0.0.0";
    int port = 8007;
    std::string model = "small";
    std::string model_file;
    std::string device = "auto";
};

Config arguments(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "Usage: whisper_transcribe [-p 8007] [-m small] [--host 127.0.0.1]\n"
                         "       [--model-file /path/ggml-model.bin] [--device auto|cpu|cuda]\n";
            std::exit(EXIT_SUCCESS);
        }
        if (++i == argc) {
            throw std::invalid_argument("Missing value for " + key);
        }
        std::string value = argv[i];
        if (key == "-p" || key == "--port") {
            std::size_t end = 0;
            config.port = std::stoi(value, &end);
            if (end != value.size() || config.port < 1 || config.port > 65535) {
                throw std::invalid_argument("Port must be between 1 and 65535");
            }
        } else if (key == "-m" || key == "--model") {
            config.model = value;
        } else if (key == "--model-file") {
            config.model_file = value;
        } else if (key == "--host") {
            config.host = value;
        } else if (key == "--device") {
            config.device = value;
        } else {
            throw std::invalid_argument("Unknown argument: " + key);
        }
    }
    if (config.device != "auto" && config.device != "cpu" && config.device != "cuda") {
        throw std::invalid_argument("Device must be auto, cpu or cuda");
    }
    return config;
}

bool use_cuda(const std::string& device) {
    if (device == "cpu") {
        return false;
    }
#ifdef WHISPER_SERVICE_HAS_CUDA
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result == cudaSuccess && count > 0) {
        ggml_backend_load_all();
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto backend = ggml_backend_dev_get(i);
            if (std::string(ggml_backend_dev_name(backend)).find("CUDA") == 0) {
                return true;
            }
        }
        std::cerr << "CUDA device exists, but Whisper has no CUDA backend\n";
    } else {
        std::cerr << "CUDA is unavailable: " << cudaGetErrorString(result) << "; devices=" << count << '\n';
    }
#endif
    if (device == "cuda") {
        throw std::runtime_error("CUDA was requested but is not available");
    }
    std::cerr << "Using CPU inference\n";
    return false;
}

class Transcriber {
public:
    Transcriber(const std::string& path, bool gpu) {
        auto parameters = whisper_context_default_params();
        parameters.use_gpu = gpu;
        parameters.gpu_device = 0;
        context_.reset(whisper_init_from_file_with_params(path.c_str(), parameters));
        if (!context_) {
            throw std::runtime_error("Cannot load Whisper model: " + path);
        }
    }

    Transcript run(const Audio& audio, const Options& options) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto parameters = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        parameters.n_threads = static_cast<int>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
        parameters.language = options.language.c_str();
        parameters.translate = options.translate;
        parameters.initial_prompt = options.prompt.c_str();
        parameters.print_realtime = false;
        parameters.print_progress = false;
        parameters.print_timestamps = false;
        parameters.print_special = false;
        parameters.no_context = true;
        parameters.abort_callback = [](void*) { return interrupted != 0; };
        if (whisper_full(context_.get(), parameters, audio.samples.data(),
                         static_cast<int>(audio.samples.size())) != 0) {
            throw std::runtime_error("Whisper transcription failed");
        }
        Transcript transcript;
        int language = whisper_full_lang_id(context_.get());
        const char* code = whisper_lang_str(language);
        if (!code) {
            throw std::runtime_error("Whisper returned an invalid language");
        }
        transcript.language = code;
        double duration = static_cast<double>(audio.samples.size()) / sample_rate;
        for (int i = 0; i < whisper_full_n_segments(context_.get()); ++i) {
            std::string text = whisper_full_get_segment_text(context_.get(), i);
            transcript.text += text;
            double start = std::clamp(whisper_full_get_segment_t0(context_.get(), i) / 100.0, 0.0, duration);
            double end = std::clamp(whisper_full_get_segment_t1(context_.get(), i) / 100.0, start, duration);
            transcript.segments.push_back({start, end, text,
                whisper_full_get_segment_no_speech_prob(context_.get(), i)});
        }
        return transcript;
    }

private:
    std::unique_ptr<whisper_context, decltype(&whisper_free)> context_{nullptr, whisper_free};
    std::mutex mutex_;
};

void error_response(httplib::Response& response, int status, const std::string& message) {
    response.status = status;
    response.set_content(nlohmann::json({{"error", {{"status", status}, {"message", message}}}}).dump(),
                         "application/json");
}

bool constant_time_equal(std::string_view left, std::string_view right) {
    const std::size_t length = std::max(left.size(), right.size());
    std::size_t difference = left.size() ^ right.size();
    for (std::size_t i = 0; i < length; ++i) {
        const auto lhs = i < left.size() ? static_cast<unsigned char>(left[i]) : 0;
        const auto rhs = i < right.size() ? static_cast<unsigned char>(right[i]) : 0;
        difference |= lhs ^ rhs;
    }
    return difference == 0;
}

bool authorized(const httplib::Request& request, std::string_view api_key) {
    const std::string header = request.get_header_value("Authorization");
    constexpr std::string_view prefix = "Bearer ";
    const std::string_view value(header);
    if (value.substr(0, prefix.size()) != prefix) {
        return false;
    }
    return constant_time_equal(value.substr(prefix.size()), api_key);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto config = arguments(argc, argv);
        const char* configured_api_key = std::getenv("WHISPER_API_KEY");
        if (configured_api_key == nullptr || std::string_view(configured_api_key).size() < 32) {
            throw std::runtime_error("WHISPER_API_KEY must be set to a secret of at least 32 characters");
        }
        const std::string api_key(configured_api_key);
        std::string path = config.model_file;
        if (path.empty()) {
            const char* home = std::getenv("HOME");
            path = model_path(config.model, home ? home : "");
            if (!std::filesystem::exists(path)) {
                std::cerr << "Downloading model to " << path << '\n';
                download_model(config.model, path);
            }
        }
        bool gpu = use_cuda(config.device);
        Transcriber transcriber(path, gpu);
        const std::string model = config.model_file.empty() ? config.model : std::filesystem::path(path).filename().string();
        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(8, 16); };
        server.set_payload_max_length(max_upload_bytes);
        server.set_read_timeout(30);
        server.set_write_timeout(30);
        std::atomic<unsigned int> admitted{0};
        server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
            response.set_content(nlohmann::json({{"status", "ready"}, {"model", model},
                {"device", gpu ? "cuda" : "cpu"}}).dump(), "application/json");
        });
        server.Post("/", [&](const httplib::Request& request, httplib::Response& response) {
            if (!authorized(request, api_key)) {
                response.set_header("WWW-Authenticate", "Bearer");
                error_response(response, 401, "A valid Bearer API key is required");
                return;
            }
            if (admitted.fetch_add(1) >= 4) {
                admitted.fetch_sub(1);
                response.set_header("Retry-After", "5");
                error_response(response, 503, "Transcription queue is full");
                return;
            }
            struct Release {
                std::atomic<unsigned int>& count;
                ~Release() { count.fetch_sub(1); }
            } release{admitted};
            try {
                auto start = std::chrono::steady_clock::now();
                if (request.is_multipart_form_data()) {
                    throw HttpError(415, "Send raw audio bytes, not multipart/form-data");
                }
                auto options = parse_options(request.get_param_value("lang"), request.get_param_value("translate"),
                                             request.get_param_value("hint"), request.get_param_value("format"));
                auto audio = decode_audio(request.body);
                auto transcript = transcriber.run(audio, options);
                double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                if (options.text_response) {
                    response.set_content(transcript.text, "text/plain; charset=utf-8");
                } else {
                    response.set_content(metadata(audio, transcript, options, model, seconds).dump(), "application/json");
                }
                std::cerr << "Transcribed " << audio.samples.size() << " samples in " << seconds << " seconds\n";
            } catch (const HttpError& error) {
                std::cerr << "Request rejected (" << error.status << "): " << error.what() << '\n';
                error_response(response, error.status, error.what());
            } catch (const std::exception& error) {
                std::cerr << "Transcription failed: " << error.what() << '\n';
                error_response(response, 500, "Transcription failed; see service log");
            }
        });
        server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
            if (response.body.empty()) {
                error_response(response, response.status,
                    response.status == 413 ? "Audio upload exceeds 32 MiB" : "HTTP request failed");
            }
        });
        server.set_exception_handler([](const httplib::Request&, httplib::Response& response, std::exception_ptr error) {
            try {
                if (error) {
                    std::rethrow_exception(error);
                }
            } catch (const std::exception& exception) {
                std::cerr << "HTTP handler failed: " << exception.what() << '\n';
            } catch (...) {
                std::cerr << "HTTP handler failed with a non-standard exception\n";
            }
            error_response(response, 500, "Internal server error; see service log");
        });
        if (!server.bind_to_port(config.host, config.port)) {
            throw std::runtime_error("Cannot bind to " + config.host + ":" + std::to_string(config.port));
        }
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::atomic<bool> finished{false};
        std::thread shutdown([&] {
            while (!finished.load()) {
                if (interrupted && server.is_running()) {
                    server.stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        std::cerr << "Listening on http://" << config.host << ':' << config.port
                  << " (one " << (gpu ? "CUDA" : "CPU") << " model, at most four admitted requests)\n";
        bool success = false;
        try {
            success = server.listen_after_bind();
        } catch (...) {
            finished = true;
            shutdown.join();
            throw;
        }
        finished = true;
        shutdown.join();
        if (!success) {
            throw std::runtime_error("HTTP listener failed");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
