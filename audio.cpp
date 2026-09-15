#include "audio.h"
#include "logger.h"
#include "timer.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

using namespace std;

AudioRecorder::AudioRecorder(int chunk_seconds)
    : chunk_seconds_(chunk_seconds) {
    paths_[0] = "/tmp/voice_chunk_a.wav";
    paths_[1] = "/tmp/voice_chunk_b.wav";
}

AudioRecorder::~AudioRecorder() {
    shutdown();
}

bool AudioRecorder::start() {
    if (worker_.joinable()) return true;
    worker_ = thread([this] { worker(); });
    Logger::log("RECORD", "Background recorder started ("
                + to_string(chunk_seconds_) + "s chunks)");
    return true;
}

void AudioRecorder::shutdown() {
    {
        unique_lock<mutex> lock(mtx_);
        stop_ = true;
    }
    cv_free_.notify_all();
    cv_ready_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    Logger::log("RECORD", "Recorder stopped");
}

string AudioRecorder::next() {
    unique_lock<mutex> lock(mtx_);
    cv_ready_.wait(lock, [this] { return stop_ || ready_[0] || ready_[1]; });
    if (stop_) return "";

    // Pick the oldest ready chunk to preserve order.
    int slot = -1;
    uint64_t oldest = ~0ULL;
    for (int i = 0; i < 2; i++) {
        if (ready_[i] && seq_[i] < oldest) {
            oldest = seq_[i];
            slot = i;
        }
    }
    if (slot < 0) return "";

    ready_[slot] = false;
    consumed_ = slot;
    last_ok_ = ok_[slot];
    return paths_[slot];
}

void AudioRecorder::release() {
    unique_lock<mutex> lock(mtx_);
    if (consumed_ >= 0) {
        free_[consumed_] = true;
        consumed_ = -1;
        cv_free_.notify_one();
    }
}

bool AudioRecorder::record_to(const string& path) {
    Timer timer("RECORD");

    ostringstream cmd;
    cmd << "arecord -f S16_LE -r 16000 -c 1 -d " << chunk_seconds_
        << " \"" << path << "\" 2>/dev/null";

    if (system(cmd.str().c_str()) == 0) {
        return true;
    }

    Logger::log("RECORD", "arecord failed, trying ffmpeg...");
    ostringstream cmd2;
    cmd2 << "ffmpeg -f alsa -i default -ar 16000 -ac 1 -t " << chunk_seconds_
         << " \"" << path << "\" -y 2>/dev/null";
    return system(cmd2.str().c_str()) == 0;
}

void AudioRecorder::worker() {
    unique_lock<mutex> lock(mtx_);

    while (true) {
        cv_free_.wait(lock, [this] { return stop_ || free_[0] || free_[1]; });
        if (stop_) break;

        int slot = free_[0] ? 0 : 1;
        free_[slot] = false;
        string path = paths_[slot];

        lock.unlock();

        bool ok = record_to(path);

        lock.lock();
        ok_[slot] = ok;
        seq_[slot] = next_seq_++;
        ready_[slot] = true;
        cv_ready_.notify_one();
    }
}

AudioBuffer load_wav(const string& path) {
    AudioBuffer audio;
    ifstream file(path, ios::binary);
    if (!file) {
        Logger::log("WAV", "ERROR: Cannot open file: " + path);
        return audio;
    }
    auto read_u32 = [&file]() -> uint32_t {
        uint8_t buf[4];
        file.read(reinterpret_cast<char*>(buf), 4);
        return buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24);
    };
    auto read_u16 = [&file]() -> uint16_t {
        uint8_t buf[2];
        file.read(reinterpret_cast<char*>(buf), 2);
        return buf[0] | (buf[1] << 8);
    };
    char riff[4];
    file.read(riff, 4);
    if (strncmp(riff, "RIFF", 4) != 0) {
        Logger::log("WAV", "ERROR: Not a RIFF file");
        return audio;
    }
    read_u32();
    char wave[4];
    file.read(wave, 4);
    if (strncmp(wave, "WAVE", 4) != 0) {
        Logger::log("WAV", "ERROR: Not a WAVE file");
        return audio;
    }
    while (file) {
        char chunk_id[4];
        file.read(chunk_id, 4);
        if (!file) break;
        uint32_t chunk_size = read_u32();
        if (strncmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t audio_format = read_u16();
            uint16_t num_channels = read_u16();
            uint32_t sample_rate = read_u32();
            read_u32();
            read_u16();
            uint16_t bits_per_sample = read_u16();
            audio.sample_rate = static_cast<int>(sample_rate);
            Logger::log("WAV", "Format: " + to_string(audio_format)
                      + ", Channels: " + to_string(num_channels)
                      + ", Rate: " + to_string(sample_rate)
                      + ", Bits: " + to_string(bits_per_sample));
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, ios::cur);
            }
        } else if (strncmp(chunk_id, "data", 4) == 0) {
            size_t num_samples = chunk_size / 2;
            vector<int16_t> pcm(num_samples);
            file.read(reinterpret_cast<char*>(pcm.data()), chunk_size);
            audio.samples.reserve(num_samples);
            for (int16_t s : pcm) {
                audio.samples.push_back(s / 32768.0f);
            }
            Logger::log("WAV", "Loaded " + to_string(num_samples)
                      + " samples from " + path);
            break;
        } else {
            file.seekg(chunk_size, ios::cur);
        }
    }
    return audio;
}