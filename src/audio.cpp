#include "voice_transcribe.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libswresample/swresample.h>
}

namespace transcriber {
namespace {

std::string av_error(int code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(code, buffer, sizeof(buffer));
    return buffer;
}

void check(int result, const char* operation) {
    if (result < 0) {
        throw HttpError(422, std::string(operation) + ": " + av_error(result));
    }
}

struct Input {
    std::string_view bytes;
    std::size_t position = 0;
};

int read_packet(void* opaque, uint8_t* buffer, int size) {
    auto& input = *static_cast<Input*>(opaque);
    auto count = std::min(input.bytes.size() - input.position, static_cast<std::size_t>(size));
    if (count == 0) {
        return AVERROR_EOF;
    }
    std::memcpy(buffer, input.bytes.data() + input.position, count);
    input.position += count;
    return static_cast<int>(count);
}

int64_t seek(void* opaque, int64_t offset, int whence) {
    auto& input = *static_cast<Input*>(opaque);
    if (whence == AVSEEK_SIZE) {
        return static_cast<int64_t>(input.bytes.size());
    }
    whence &= ~AVSEEK_FORCE;
    int64_t base = 0;
    if (whence == SEEK_CUR) {
        base = static_cast<int64_t>(input.position);
    } else if (whence == SEEK_END) {
        base = static_cast<int64_t>(input.bytes.size());
    } else if (whence != SEEK_SET) {
        return AVERROR(EINVAL);
    }
    if (offset < -base || offset > static_cast<int64_t>(input.bytes.size()) - base) {
        return AVERROR(EINVAL);
    }
    input.position = static_cast<std::size_t>(base + offset);
    return static_cast<int64_t>(input.position);
}

struct Format {
    AVFormatContext* context = nullptr;
    AVIOContext* io = nullptr;
    ~Format() {
        avformat_close_input(&context);
        if (io) {
            av_freep(&io->buffer);
            avio_context_free(&io);
        }
    }
};

struct CodecDeleter {
    void operator()(AVCodecContext* value) const { avcodec_free_context(&value); }
};
struct FrameDeleter {
    void operator()(AVFrame* value) const { av_frame_free(&value); }
};
struct PacketDeleter {
    void operator()(AVPacket* value) const { av_packet_free(&value); }
};
struct ResamplerDeleter {
    void operator()(SwrContext* value) const { swr_free(&value); }
};

} // namespace

Audio decode_audio(std::string_view input, int duration_limit) {
    if (input.empty()) {
        throw HttpError(400, "No audio received");
    }
    if (input.size() > max_upload_bytes) {
        throw HttpError(413, "Audio upload exceeds 32 MiB");
    }
    if (duration_limit <= 0 || duration_limit > max_audio_seconds) {
        throw std::invalid_argument("Invalid audio duration limit");
    }

    Input reader{input};
    Format format;
    auto* buffer = static_cast<unsigned char*>(av_malloc(32768));
    if (!buffer) {
        throw std::bad_alloc();
    }
    format.io = avio_alloc_context(buffer, 32768, 0, &reader, read_packet, nullptr, seek);
    if (!format.io) {
        av_free(buffer);
        throw std::bad_alloc();
    }
    format.context = avformat_alloc_context();
    if (!format.context) {
        throw std::bad_alloc();
    }
    format.context->pb = format.io;
    format.context->flags |= AVFMT_FLAG_CUSTOM_IO;
    format.context->io_open = [](AVFormatContext*, AVIOContext**, const char*, int, AVDictionary**) {
        return AVERROR(EACCES);
    };
    format.context->probesize = 1024 * 1024;
    format.context->max_analyze_duration = 5 * AV_TIME_BASE;
    check(avformat_open_input(&format.context, nullptr, nullptr, nullptr), "Cannot open audio");
    check(avformat_find_stream_info(format.context, nullptr), "Cannot read audio stream information");

    const AVCodec* decoder = nullptr;
    int index = av_find_best_stream(format.context, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
    check(index, "No supported audio stream");
    std::unique_ptr<AVCodecContext, CodecDeleter> codec(avcodec_alloc_context3(decoder));
    if (!codec) {
        throw std::bad_alloc();
    }
    auto* stream = format.context->streams[index];
    check(avcodec_parameters_to_context(codec.get(), stream->codecpar), "Cannot configure decoder");
    codec->thread_count = 1;
    codec->err_recognition = AV_EF_CRCCHECK | AV_EF_EXPLODE;
    check(avcodec_open2(codec.get(), decoder, nullptr), "Cannot open decoder");
    if (codec->sample_rate <= 0 || codec->ch_layout.nb_channels <= 0) {
        throw HttpError(422, "Audio has invalid sample rate or channel layout");
    }
    Audio audio;
    audio.container = format.context->iformat->name;
    audio.codec = avcodec_get_name(codec->codec_id);
    audio.original_sample_rate = codec->sample_rate;
    audio.original_channels = codec->ch_layout.nb_channels;
    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    SwrContext* raw_resampler = nullptr;
    int allocation = swr_alloc_set_opts2(&raw_resampler, &mono, AV_SAMPLE_FMT_FLT, sample_rate,
                                       &codec->ch_layout, codec->sample_fmt, codec->sample_rate, 0, nullptr);
    std::unique_ptr<SwrContext, ResamplerDeleter> resampler(raw_resampler);
    check(allocation, "Cannot allocate resampler");
    check(swr_init(resampler.get()), "Cannot initialize resampler");
    std::unique_ptr<AVFrame, FrameDeleter> frame(av_frame_alloc());
    std::unique_ptr<AVPacket, PacketDeleter> packet(av_packet_alloc());
    if (!frame || !packet) {
        throw std::bad_alloc();
    }

    const auto max_samples = static_cast<std::size_t>(duration_limit) * sample_rate;
    std::vector<float> converted;
    auto convert = [&](const uint8_t** data, int count) {
        int capacity = swr_get_out_samples(resampler.get(), count);
        check(capacity, "Cannot size resampler output");
        if (static_cast<std::size_t>(capacity) > max_samples + sample_rate) {
            throw HttpError(413, "Audio frame exceeds duration limit");
        }
        converted.resize(std::max(capacity, 1));
        auto* output = reinterpret_cast<uint8_t*>(converted.data());
        int produced = swr_convert(resampler.get(), &output, static_cast<int>(converted.size()), data, count);
        check(produced, "Cannot resample audio");
        if (static_cast<std::size_t>(produced) > max_samples - audio.samples.size()) {
            throw HttpError(413, "Audio duration exceeds limit");
        }
        for (int i = 0; i < produced; ++i) {
            if (!std::isfinite(converted[i])) {
                throw HttpError(422, "Audio contains non-finite samples");
            }
        }
        audio.samples.insert(audio.samples.end(), converted.begin(), converted.begin() + produced);
        return produced;
    };
    auto receive = [&] {
        while (true) {
            int result = avcodec_receive_frame(codec.get(), frame.get());
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
                return;
            }
            check(result, "Cannot decode audio frame");
            if (frame->sample_rate != audio.original_sample_rate ||
                av_channel_layout_compare(&frame->ch_layout, &codec->ch_layout) != 0 ||
                frame->format != codec->sample_fmt) {
                throw HttpError(422, "Changing audio format within a stream is not supported");
            }
            convert(const_cast<const uint8_t**>(frame->extended_data), frame->nb_samples);
            av_frame_unref(frame.get());
        }
    };

    int result;
    while ((result = av_read_frame(format.context, packet.get())) >= 0) {
        if (packet->stream_index == index) {
            if (packet->flags & AV_PKT_FLAG_CORRUPT) {
                throw HttpError(422, "Audio contains a corrupt packet");
            }
            check(avcodec_send_packet(codec.get(), packet.get()), "Cannot submit audio packet");
            receive();
        }
        av_packet_unref(packet.get());
    }
    if (result != AVERROR_EOF) {
        check(result, "Cannot read audio packet");
    }
    check(avcodec_send_packet(codec.get(), nullptr), "Cannot flush audio decoder");
    receive();
    while (convert(nullptr, 0) > 0) {}
    if (audio.samples.empty()) {
        throw HttpError(422, "Audio contains no decodable samples");
    }
    return audio;
}

} // namespace transcriber
