/**
 * @file voice_transcribe.cpp
 * @author Egbert Nierop (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-08-16
 *
 * @copyright Copyright (c) 2024 Nierop Computer Vision
 * compile with
 
 * git clone https://github.com/ggerganov/whisper.cpp.git --depth 1
 * cd whisper.cpp
 * mkdir build
 * cd build
 * #-DWHISPER_FFMPEG=ON
 * this is why we leave FFMPEG off for WHISPER
 * it seems that whisper likes ffmpeg up to 4.4 which is already quite old
 * cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DGGML_OPENMP=ON -DGGML_NATIVE=ON
 * g++ -std=c++17 voice_transcribe.cpp -o voice_transcribe `pkg-config --cflags whisper --libs whisper` `pkg-config --cflags opencv4` -lopencv_core -lavformat -lavcodec -lavutil -lswresample -lcurl -pthread -O2
  *
 * Intel Laptop, assuming you have openmp already inplace
 # cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_OPENMP=ON -DGGML_NATIVE=ON
  *
 * sudo make install
 *
 * result should be something like below
 * -- Install configuration: "Release"
-- Installing: /usr/local/lib/libwhisper.so.1.6.2
-- Installing: /usr/local/lib/libwhisper.so.1
-- Set runtime path of "/usr/local/lib/libwhisper.so.1.6.2" to ""
-- Installing: /usr/local/lib/libwhisper.so
-- Installing: /usr/local/include/whisper.h
-- Installing: /usr/local/lib/cmake/whisper/whisper-config.cmake
-- Installing: /usr/local/lib/cmake/whisper/whisper-version.cmake
-- Installing: /usr/local/lib/pkgconfig/whisper.pc
-- Installing: /usr/local/lib/libggml.so
-- Installing: /usr/local/include/ggml.h
-- Installing: /usr/local/include/ggml-alloc.h
-- Installing: /usr/local/include/ggml-backend.h
-- Installing: /usr/local/include/ggml-blas.h
-- Installing: /usr/local/include/ggml-cann.h
-- Installing: /usr/local/include/ggml-cuda.h
-- Up-to-date: /usr/local/include/ggml.h
-- Installing: /usr/local/include/ggml-kompute.h
-- Installing: /usr/local/include/ggml-metal.h
-- Installing: /usr/local/include/ggml-rpc.h
-- Installing: /usr/local/include/ggml-sycl.h
-- Installing: /usr/local/include/ggml-vulkan.h
-- Up-to-date: /usr/local/lib/libggml.so
-- Up-to-date: /usr/local/include/ggml.h
-- Up-to-date: /usr/local/include/ggml-alloc.h
-- Up-to-date: /usr/local/include/ggml-backend.h
-- Up-to-date: /usr/local/include/ggml-blas.h
-- Up-to-date: /usr/local/include/ggml-cann.h
-- Up-to-date: /usr/local/include/ggml-cuda.h
-- Up-to-date: /usr/local/include/ggml.h
-- Up-to-date: /usr/local/include/ggml-kompute.h
-- Up-to-date: /usr/local/include/ggml-metal.h
-- Up-to-date: /usr/local/include/ggml-rpc.h
-- Up-to-date: /usr/local/include/ggml-sycl.h
-- Up-to-date: /usr/local/include/ggml-vulkan.h
 */

#include "voice_transcribe.h"
#include "httplib.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <signal.h>
#include <thread>
#include <getopt.h> // Voor argument parsing

#include <iostream>


#include <filesystem>

extern "C" {
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavcodec/avcodec.h>
#include "curl/curl.h"
}
/**
 test with 
 curl -X POST --data-binary @test.webm \
     -H "Content-Type: application/octet-stream" \
     "http://127.0.0.1:8007/?lang=nl&translate=false"
*/

std::vector<WhisperContextWrapper> whisper_contexts;
std::vector<std::unique_ptr<std::mutex>> whisper_mutexes;
std::mutex whisper_mutexes_mutex;
struct mg_context* g_ctx = nullptr;

std::atomic<bool> g_running(true);
bool use_gpu = false;
std::string global_model_path;
std::condition_variable context_available_cv;

#define WAVE_SAMPLE_RATE 16000  // cd quality


#if __has_include(<cuda_runtime.h>)
    #include <cuda_runtime.h>
    #define CUDA_AVAILABLE 1
    
#else
    #define CUDA_AVAILABLE 0
#endif

CpuInfo get_number_of_gpus() {
    int device_count = 0;
    #ifdef CUDA_AVAILABLE
    
    cudaError_t err = cudaGetDeviceCount(&device_count);
     if (err != cudaSuccess ) {
        device_count = 0;
     }
    #endif
    return {
      (unsigned int)  device_count,
        std::min(4u, (uint)std::thread::hardware_concurrency())
    };
}

void unload_inactive_contexts() {
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::minutes(1));
        auto now = std::chrono::steady_clock::now();

        for (auto& wrapper : whisper_contexts) {
            std::lock_guard<std::mutex> lock(*wrapper.context_mutex);
            if (wrapper.loaded.load() &&
                std::chrono::duration_cast<std::chrono::minutes>(now - wrapper.last_used).count() >= 10 &&
                !wrapper.in_use.load()) {
                whisper_free(wrapper.context);
                wrapper.context = nullptr;
                wrapper.loaded = false;
                std::cout << "Whisper context vrijgegeven.\n";
            }
        }
    }
}

AvBuffer::AvBuffer(size_t size) : size_(size) {
    buffer_ = static_cast<unsigned char*>(av_malloc(size_));
    if (!buffer_) {
        throw std::runtime_error("Failed to allocate memory with av_malloc");
    }
}

unsigned char* AvBuffer::data() const {
    return buffer_;
}

size_t AvBuffer::size() const {
    return size_;
}
/**
 * makes sure ffmpeg will own this memory
 */
void AvBuffer::Detach() {
    do_not_free = true;
}
AvBuffer::~AvBuffer()
{
    if (buffer_ != nullptr && do_not_free == false){
        av_freep(&buffer_);
    }
}

/* returns true if audio format is PCM and bitsPerSample =32 bit*/
bool is_wav_format(const AvBuffer& audio_data) {
    if (audio_data.size() < sizeof(WAVHeader)) {
        return false;
    }

    WAVHeader header;
    std::memcpy(&header, audio_data.data(), sizeof(WAVHeader));

    // Controleer of het begint met "RIFF" en "WAVE"
    return std::strncmp(header.riff, "RIFF", 4) == 0 && std::strncmp(header.wave, "WAVE", 4) == 0 && header.audioFormat == 1 && header.bitsPerSample == 32;
}

// Functie om PCM-gegevens uit een WAV-bestand te lezen
std::vector<float> read_wav_file(const AvBuffer& wav_data) {

    // Controleer of de WAV-header aanwezig is
    if (wav_data.size() < 44) {
        throw std::runtime_error("Invalid WAV file: header too short");
    }

    // WAV-header
    const unsigned char* header = wav_data.data();
    uint16_t num_channels = header[22] | (header[23] << 8);
    uint32_t sample_rate = header[24] | (header[25] << 8) | (header[26] << 16) | (header[27] << 24);
    uint16_t bits_per_sample = header[34] | (header[35] << 8);

    // Only 16-bit PCM
    if (bits_per_sample != 16) {
        throw std::runtime_error("Unsupported WAV file: only 16-bit PCM is supported");
    }

    // Start van de audio data (44 bytes header overslaan)
    const unsigned char* audio_data = wav_data.data() + 44;
    size_t audio_data_size = wav_data.size() - 44;

    size_t num_samples = audio_data_size / 2; // 2 bytes per sample voor 16-bit PCM
    std::vector<float> pcm_floats;
    pcm_floats.reserve(num_samples);
    // Audio data converteren naar floats
    for (size_t i = 0; i < audio_data_size; i += 2) {
        // 16-bit PCM is signed, dus converteren naar float en normaliseren naar het bereik [-1, 1]
        int16_t sample = audio_data[i] | (audio_data[i + 1] << 8);
        pcm_floats.push_back(static_cast<float>(sample) / 32768.0f);
    }

    return pcm_floats;
}

std::string get_model_path(const std::string& model_name) {

    const char* home_dir = std::getenv("HOME");
    if (!home_dir) {
        throw std::runtime_error("HOME environment variable is not set, do you run as a 'user'?");
    }

    std::string model_path = std::string(home_dir) + "/.cache/whisper/ggml-" + model_name + ".bin";
    return model_path;
}

// Functie om het model te laden en in cache te houden

whisper_context* load_whisper_model(int index) {
    whisper_context_params init_params = whisper_context_default_params();
    init_params.use_gpu = use_gpu;
    init_params.dtw_aheads_preset = WHISPER_AHEADS_NONE;
    
    if (use_gpu) {
        init_params.gpu_device = index;
    }

    whisper_context* ctx = whisper_init_from_file_with_params(global_model_path.c_str(), init_params);
    if (!ctx) {
        throw std::runtime_error("Failed to load Whisper model");
    }

    return ctx;
}

whisper_context* get_whisper_context() {
    while (true) {
        for (auto& wrapper : whisper_contexts) {
            bool expected = false;
            if (wrapper.in_use.compare_exchange_strong(expected, true)) {
                std::lock_guard<std::mutex> lock(*wrapper.context_mutex);

                if (!wrapper.loaded.load()) {
                    wrapper.context = load_whisper_model(use_gpu ? 0 : 0);
                    wrapper.loaded.store(true);
                    wrapper.last_used = std::chrono::steady_clock::now();
                    std::cout << "Whisper context geladen.\n";
                }

                wrapper.last_used = std::chrono::steady_clock::now();
                return wrapper.context;
            }
        }

        std::unique_lock<std::mutex> lock(whisper_mutexes_mutex);
        context_available_cv.wait(lock);
    }
}
void release_context(whisper_context* ctx) {
    for (auto& wrapper : whisper_contexts) {
        if (wrapper.context == ctx) {
            wrapper.in_use.store(false);

            // Notificeer dat een context beschikbaar is
            context_available_cv.notify_one();
            break;
        }
    }
}
/**
 * @brief
 * using ffmpeg to normalize audio data to pcm float format
 * this should deal with mp3, ogg, m4a etc
 * note that normally, whisper can deal with it, itself,
 * however, we use a recent version of ffmpeg so compilation will fail
 * @param audio_data
 * @return std::vector<float>
 */
std::vector<float> decode_audio_with_ffmpeg(const AvBuffer& audio_data) {
    std::vector<float> pcm_samples;

    AVFormatContext* format_ctx = avformat_alloc_context();
    if (!format_ctx) {
        throw std::runtime_error("Failed to allocate AVFormatContext");
    }
    // Use a custom IO context to read directly from memory
    size_t input_buffer_size = audio_data.size();

    // Use the av_malloc'ed buffer for the AVIOContext    
    auto avio_ctx = avio_alloc_context(audio_data.data(), input_buffer_size, 0, nullptr, nullptr, nullptr, nullptr);
    format_ctx->pb = avio_ctx;

    if (avformat_open_input(&format_ctx, nullptr, nullptr, nullptr) < 0) {
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to open audio with FFmpeg");
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to find stream info");
    }

    AVCodecContext* codec_ctx = nullptr;
    const AVCodec* codec = nullptr;
    int stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);

    if (stream_index < 0 || !codec) {
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to find audio codec");
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to allocate codec context");
    }

    if (avcodec_parameters_to_context(codec_ctx, format_ctx->streams[stream_index]->codecpar) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to copy codec parameters to codec context");
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to open codec");
    }

    AVPacket packet = {};

    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to allocate frame");
    }

    SwrContext* swr_ctx = nullptr;
    AVChannelLayout chLaout = AV_CHANNEL_LAYOUT_MONO;
    int ret = swr_alloc_set_opts2(&swr_ctx,                // we're allocating a new context
                                  &chLaout,                // out_ch_layout
                                  AV_SAMPLE_FMT_FLT,       // out_sample_fmt
                                  WAVE_SAMPLE_RATE,        // out_sample_rate
                                  &codec_ctx->ch_layout,   // in_ch_layout
                                  codec_ctx->sample_fmt,   // in_sample_fmt
                                  codec_ctx->sample_rate,  // in_sample_rate
                                  0,                       // log_offset
                                  nullptr);                // log_ctx

    if (ret < 0) {
        av_frame_free(&frame);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to allocate SwrContext");
    }

    if (swr_init(swr_ctx) < 0) {
        swr_free(&swr_ctx);
        av_frame_free(&frame);
        avcodec_free_context(&codec_ctx);
        avformat_free_context(format_ctx);
        throw std::runtime_error("Failed to initialize SwrContext");
    }

    std::vector<float> output_samples;
    while (av_read_frame(format_ctx, &packet) >= 0)
    {
        if (packet.stream_index == stream_index) {
            if (avcodec_send_packet(codec_ctx, &packet) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    int nb_channels = codec_ctx->ch_layout.nb_channels;
                    int out_samples_count = swr_get_out_samples(swr_ctx, frame->nb_samples);

                    if (output_samples.size() < out_samples_count * nb_channels) {
                        output_samples.resize(out_samples_count * nb_channels);  // Resize buffer indien nodig
                    }

                    float* out_samples = output_samples.data();
                    int converted_samples = swr_convert(swr_ctx, (uint8_t**)&out_samples, out_samples_count,
                                                        (const uint8_t**)frame->data, frame->nb_samples);

                    if (converted_samples > 0) {
                        pcm_samples.insert(pcm_samples.end(), out_samples, out_samples + converted_samples);
                    }
                }
            }
        }
        av_packet_unref(&packet);
    }

    // Cleanup
    swr_free(&swr_ctx);
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    avformat_free_context(format_ctx);

    return pcm_samples;
}

/**
 * @brief
 * Functie om audio te transcriberen met Whisper
 * @param ctx assigned whisper_context
 * @param audio_data PCM (32bit) compatible floats
 * @param lang assume this spoken language
 * @param translate if true, will translate to language specified
 * @return std::string the full translatino
 */
std::string transcribe_audio_with_whisper(const std::vector<float>& audio_data, const std::string& lang, const bool translate, const std::string initial_prompt) {
    whisper_context* ctx = get_whisper_context();

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = get_number_of_gpus().cpus;
    params.language = lang.empty() ? "auto" : lang.c_str();
    params.detect_language = false;
    params.translate = translate;
    params.initial_prompt = initial_prompt.c_str();
    int n_samples = audio_data.size();
    if (whisper_full(ctx, params, audio_data.data(), n_samples) != 0) {
        throw std::runtime_error("Failed to transcribe audio with Whisper");
    }

    int n_segments = whisper_full_n_segments(ctx);

    std::string stringBuffer;
    stringBuffer.reserve(n_segments * 30);

    for (int i = 0; i < n_segments; ++i) {
        stringBuffer.append(whisper_full_get_segment_text(ctx, i));
    }
    release_context(ctx);

    return stringBuffer;
}

// Signal handler om netjes af te sluiten
void signal_handler(int signal) {
    if (signal == SIGTERM || signal == SIGINT) {
        g_running = false;
    }
}

void parse_arguments(int argc, char* argv[], std::string &port, std::string &model_name) {
    int opt;
    while ((opt = getopt(argc, argv, "p:m:")) != -1) {
        switch (opt) {
            case 'p':
                port = optarg;
                break;
            case 'm':
                model_name = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -p <port> -m <model path>" << std::endl;
                exit(EXIT_FAILURE);
        }
    }
}

void initialize_whisper_contexts() {
    auto cpu_info = get_number_of_gpus();
    int max_contexts = cpu_info.cpus;

    for (int i = 0; i < max_contexts; ++i) {
        WhisperContextWrapper wrapper;
        wrapper.context_mutex = std::make_unique<std::mutex>();
        whisper_contexts.push_back(std::move(wrapper));
    }
}

void uninitialize_whisper_contexts() {
    for (auto& wrapper : whisper_contexts) {
        if (wrapper.context) {
            whisper_free(wrapper.context);
        }
    }
    whisper_contexts.clear();
}


size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

bool download_model(const std::string& model_url, const std::string& output_path) {
    CURL *curl;
    FILE *fp;
    CURLcode res;

    curl = curl_easy_init();
    if (curl) {
        fp = fopen(output_path.c_str(), "wb");
        if (!fp) {
            std::cerr << "Failed to open file for writing: " << output_path << std::endl;
            return false;
        }

        curl_easy_setopt(curl, CURLOPT_URL, model_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

        // Opties om problemen met incomplete downloads te voorkomen
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Volg redirects
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L); // Fout als HTTP code >= 400
        curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 102400L); // Buffer grootte instellen

        // Optioneel: instellen van timeouts
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L); // Geen timeout
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L); // 30 seconden om verbinding te maken

        // Voor content-length monitoring (optioneel)
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L); // Toon voortgang in terminal
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, nullptr); // Je kunt hier een voortgangsfunctie toevoegen
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, nullptr);

        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);

        if (res != CURLE_OK) {
            std::cerr << "Failed to download model: " << curl_easy_strerror(res) << std::endl;
            remove(output_path.c_str()); // Verwijder het onvolledige bestand
            return false;
        }

        // Controleer de bestandsgrootte als laatste check (optioneel)
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        if (response_code == 200) {
            double cl;
            res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
            if (res == CURLE_OK && cl > 0.0) {
                std::ifstream in(output_path, std::ifstream::ate | std::ifstream::binary);
                if (static_cast<double>(in.tellg()) < cl) {
                    std::cerr << "Downloaded file size is smaller than expected." << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

std::string get_model_url(const std::string& model_name) {
    std::string baseUri = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
    if (model_name.find("tdrz") != std::string::npos) {
        baseUri = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main/";
    }
    return baseUri + "ggml-" + model_name + ".bin";
}


void setup_routes(httplib::Server& svr) {
    svr.Post("/", [](const httplib::Request& req, httplib::Response& res) {
        std::cout << "Received request with size: " << req.body.size() << " bytes" << std::endl;
        std::cout << "Query parameters: lang=" << req.get_param_value("lang") 
                  << ", translate=" << req.get_param_value("translate") << std::endl;

        if (req.body.empty()) {
            res.status = 400;
            res.set_content("Fout: Geen audio ontvangen!", "text/plain");
            return;
        }

        try {
            // Audio verwerken
            AvBuffer audio_buffer(req.body.size());
            std::memcpy(audio_buffer.data(), req.body.data(), req.body.size());
            std::string language = req.get_param_value("lang");
            if (language.empty())
            {
                language = "auto";
            }
            auto translate = req.get_param_value("translate") == "true";
            std::string initial_prompt = req.get_param_value("hint");
            bool isPcm32Wav = is_wav_format(audio_buffer);
            if (!isPcm32Wav) {
                audio_buffer.Detach();
            }

            std::vector<float> audio_float_data = isPcm32Wav
                ? read_wav_file(audio_buffer)
                : decode_audio_with_ffmpeg(audio_buffer);

            std::string transcript = transcribe_audio_with_whisper(audio_float_data, language, translate, initial_prompt);
            std::cout << "✅ Transcriptie voltooid. Lengte: " << transcript.size() << " tekens.\n";
            
            res.set_content(transcript, "text/plain");
        }
        catch (const std::exception& ex) {
            res.status = 500;
            res.set_content(std::string("Server error: ") + ex.what(), "text/plain");
        }
    });
}

int main(int argc, char* argv[]) {
    std::string port;
    std::string model_name;

    // Argumenten parsen
    parse_arguments(argc, argv, port, model_name);
    if (port.empty() || model_name.empty()) {
        std::cerr << "Usage: " << argv[0] << " -p <port> -m <model path>" << std::endl;
        return EXIT_FAILURE;
    }
    global_model_path = get_model_path(model_name);

    // Controleer of het model al gedownload is
    if (!std::filesystem::exists(global_model_path)) {
        std::cout << "Model niet gevonden op " << global_model_path << ". Downloaden..." << std::endl;
        std::string model_url = get_model_url(model_name);
        if (!download_model(model_url, global_model_path)) {
            std::cerr << "Het downloaden van het model is mislukt." << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Model succesvol gedownload naar " << global_model_path << std::endl;
    } else {
        std::cout << "Model gevonden op " << global_model_path << std::endl;
    }

    use_gpu = get_number_of_gpus().gpus > 0;
    initialize_whisper_contexts();
    try {        
        httplib::Server svr;
        setup_routes(svr);
        std::cout << "Server gestart op http://0.0.0.0:" << port << std::endl;
        svr.listen("0.0.0.0", std::stoi(port));
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        uninitialize_whisper_contexts();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}