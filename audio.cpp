#include "audio.h"
#include "logger.h"
#include "timer.h"
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

using namespace std;

AudioRecorder::AudioRecorder(int sample_rate, int max_seconds)
    : sample_rate_(sample_rate), max_samples_((size_t)max_seconds * sample_rate) {
}

AudioRecorder::~AudioRecorder() {
    shutdown();
}

bool AudioRecorder::start() {
    {
        lock_guard<mutex> lock(mtx_);
        if (worker_.joinable()) return true;
        // Reset state from any previous recording session so each start()
        // begins a fresh timeline and an empty ring.
        samples_.clear();
        total_ = 0;
        stop_ = false;
        ok_ = false;
    }
    worker_ = thread([this] { worker(); });
    Logger::log("RECORD", "Continuous recorder started ("
                + to_string(sample_rate_) + " Hz, ring "
                + to_string(max_samples_ / sample_rate_) + "s)");
    return true;
}

void AudioRecorder::shutdown() {
    {
        lock_guard<mutex> lock(mtx_);
        stop_ = true;
    }
    if (pipe_) {
        fclose(pipe_); // closes our read end; child gets SIGPIPE on next write
        pipe_ = nullptr;
    }
    if (child_pid_ > 0) {
        kill(child_pid_, SIGTERM);
        waitpid(child_pid_, nullptr, 0);
        child_pid_ = -1;
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    Logger::log("RECORD", "Recorder stopped");
}

uint64_t AudioRecorder::sample_count() const {
    lock_guard<mutex> lock(mtx_);
    return total_;
}

AudioBuffer AudioRecorder::slice(uint64_t from_sample, uint64_t to_sample) const {
    AudioBuffer audio;
    audio.sample_rate = sample_rate_;
    lock_guard<mutex> lock(mtx_);
    // The ring only holds the most recent max_samples_ samples.
    uint64_t avail_from = total_ > samples_.size() ? total_ - samples_.size() : 0;
    uint64_t s = from_sample > avail_from ? from_sample : avail_from;
    uint64_t e = to_sample < total_ ? to_sample : total_;
    if (e <= s) return audio;
    size_t off = static_cast<size_t>(s - avail_from);
    audio.samples.assign(samples_.begin() + off,
                         samples_.begin() + off + static_cast<size_t>(e - s));
    return audio;
}

// Spawn the command with stdout connected to a pipe. Returns false on failure.
bool AudioRecorder::open_pipe(const string& cmd) {
    int fds[2];
    if (pipe(fds) != 0) return false;

    pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        // Child: replace stdout with the write end of the pipe.
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)nullptr);
        _exit(127);
    }

    close(fds[1]);
    pipe_ = fdopen(fds[0], "r");
    child_pid_ = (int)pid;
    return pipe_ != nullptr;
}

void AudioRecorder::worker() {
    string arecord = "arecord -q -f S16_LE -r " + to_string(sample_rate_)
                     + " -c 1 -t raw";
    string ffmpeg = "ffmpeg -loglevel error -f alsa -i default -ac 1 -ar "
                    + to_string(sample_rate_) + " -f s16le pipe:1";

    bool fallback_tried = false;

    {
        lock_guard<mutex> lock(mtx_);
        ok_ = open_pipe(arecord);
    }
    if (!ok_) {
        Logger::log("RECORD", "arecord unavailable, trying ffmpeg...");
        lock_guard<mutex> lock(mtx_);
        ok_ = open_pipe(ffmpeg);
        fallback_tried = true;
    }
    if (!ok_) {
        Logger::log("RECORD", "ERROR: failed to open audio input (arecord/ffmpeg)");
        return;
    }

    vector<short> chunk(4096);
    while (true) {
        {
            lock_guard<mutex> lock(mtx_);
            if (stop_) break;
        }
        size_t got = fread(chunk.data(), sizeof(short), chunk.size(), pipe_);
        if (got == 0) {
            bool stopping = false;
            {
                lock_guard<mutex> lock(mtx_);
                stopping = stop_;
            }
            if (stopping) break;
            // Capture ended (or arecord is missing): try ffmpeg once.
            fclose(pipe_);
            pipe_ = nullptr;
            if (child_pid_ > 0) {
                kill(child_pid_, SIGTERM);
                waitpid(child_pid_, nullptr, 0);
                child_pid_ = -1;
            }
            if (fallback_tried) {
                Logger::log("RECORD", "Audio input closed unexpectedly");
                break;
            }
            Logger::log("RECORD", "arecord stopped, switching to ffmpeg...");
            lock_guard<mutex> lock(mtx_);
            ok_ = open_pipe(ffmpeg);
            fallback_tried = true;
            if (!ok_) break;
            continue;
        }
        lock_guard<mutex> lock(mtx_);
        for (size_t i = 0; i < got; i++) {
            samples_.push_back(chunk[i] / 32768.0f);
        }
        total_ += got;
        // Trim the ring to the last max_samples_.
        if (samples_.size() > max_samples_) {
            samples_.erase(samples_.begin(),
                           samples_.begin() + (samples_.size() - max_samples_));
        }
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