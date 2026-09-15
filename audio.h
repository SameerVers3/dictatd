#pragma once
#include <string>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;

struct AudioBuffer {
    vector<float> samples;
    int sample_rate = 0;
};

// Continuous recorder: captures audio from the microphone into an in-memory
// ring of the last max_seconds seconds. Consumers slice out arbitrary spans
// with slice(), so the decode region can start at the committed frontier and
// only grow on the tail.
class AudioRecorder {
public:
    // sample_rate is fixed (16000 Hz), max_seconds bounds memory usage.
    explicit AudioRecorder(int sample_rate, int max_seconds);
    ~AudioRecorder();

    bool start();    // spawn the capture thread (arecord, fallback ffmpeg)
    void shutdown(); // stop the child and join the thread

    // Total audio samples captured so far.
    uint64_t sample_count() const;
    // Copy of audio in [from_sample, to_sample) (clamped to what exists).
    AudioBuffer slice(uint64_t from_sample, uint64_t to_sample) const;

private:
    void worker();
    // Spawn `cmd` with its stdout connected to a pipe we read.
    bool open_pipe(const string& cmd);

    int sample_rate_;
    size_t max_samples_;

    vector<float> samples_; // ring: last max_samples_ samples
    uint64_t total_ = 0;

    bool ok_ = false;
    bool stop_ = false;
    thread worker_;
    FILE* pipe_ = nullptr;
    int child_pid_ = -1;
    mutable mutex mtx_;
};

AudioBuffer load_wav(const string& path);