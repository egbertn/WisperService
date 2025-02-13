#ifndef VOICE_TRANSCRIBE_H
#define VOICE_TRANSCRIBE_H


#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>  // Ensure memory header is included for std::unique_ptr
#include "whisper.h"
#include <condition_variable>

// WAV header structure
struct WAVHeader {
    char riff[4];
    uint32_t chunkSize;
    char wave[4];
    char fmt[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4];
    uint32_t dataSize;
};

/**
 * @brief wrapper around binary presentation of media file
 * so if file is 102K this buffer reserves 102Kb.
 * We own this buffer when native pcm 32 bit audio stream
 * we do not own this buffer when non pcm 32 bit. Ffmpeg will own it and fiddle with it.
 * FFMpeg will free it with avformat_free_context(format_ctx);
 * see also comments for buffer at avio_alloc_context
 */
class AvBuffer {
public:
    AvBuffer(size_t size);
    unsigned char* data() const;
    size_t size() const;
    void Detach();
    ~AvBuffer();
private:
    unsigned char* buffer_;
    bool do_not_free;
    int size_;
};
struct CpuInfo {
public:
    unsigned int gpus;
     unsigned int cpus;
};

// Wrapper for a Whisper context with last-used timestamp and mutex
// since mutexes don't like to be moved in memory
// we wrap it with unique_ptr
struct WhisperContextWrapper {
    whisper_context* context = nullptr;
    std::unique_ptr<std::mutex> context_mutex;
    std::chrono::steady_clock::time_point last_used;
    std::atomic<bool> loaded{false};
    std::condition_variable cv;
    std::atomic<bool> in_use{false};
    WhisperContextWrapper() = default;

    // Verplaatsconstructor
    WhisperContextWrapper(WhisperContextWrapper&& other) noexcept
        : context(other.context),
          context_mutex(std::move(other.context_mutex)),
          last_used(other.last_used),
          loaded(other.loaded.load()),
          in_use(other.in_use.load()),
          cv() {}

    // Verplaatsoperator
    WhisperContextWrapper& operator=(WhisperContextWrapper&& other) noexcept {
        if (this != &other) {
            context = other.context;
            context_mutex = std::move(other.context_mutex);
            last_used = other.last_used;
            loaded.store(other.loaded.load());
            in_use.store(other.in_use.load());
            // std::condition_variable kan niet worden verplaatst, dus we initialiseren deze opnieuw
        }
        return *this;
    }

    // Kopieerconstructors en kopieeroperators verwijderen
    WhisperContextWrapper(const WhisperContextWrapper&) = delete;
    WhisperContextWrapper& operator=(const WhisperContextWrapper&) = delete;
};

// Function declarations
CpuInfo get_number_of_gpus();
bool is_wav_format(const AvBuffer& audio_data);
std::vector<float> read_wav_file(const AvBuffer& wav_data);
std::vector<float> decode_audio_with_ffmpeg(const AvBuffer& audio_data);
std::string get_model_path(const std::string& model_name);
whisper_context* load_whisper_model(int index);
whisper_context* get_whisper_context();
void release_context(whisper_context* ctx);

void initialize_whisper_contexts();
void uninitialize_whisper_contexts();
std::string transcribe_audio_with_whisper(const std::vector<float>& audio_data, const std::string& lang, bool translate);
void unload_inactive_contexts();
void parse_arguments(int argc, char* argv[], std::string &port, std::string &model_path);
bool download_model(const std::string& model_url, const std::string& output_path);
std::string get_model_url(const std::string& model_name);
void signal_handler(int signal);

#endif // VOICE_TRANSCRIBE_H
