#pragma once
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

struct AudioBuffer {
    vector<float> samples;
    int sample_rate = 0;
};

// Background recorder: continuously records fixed-length chunks into two
// rotating WAV files so that recording of the next chunk overlaps with the
// consumer processing the current one.
class AudioRecorder {
public:
    explicit AudioRecorder(int chunk_seconds);
    ~AudioRecorder();

    bool start();    // spawn the background recorder thread
    void shutdown(); // stop the thread and join it

    // Blocks until the next chunk has finished recording and returns its path
    // ("" when stopped).
    string next();
    // Marks the most recently consumed chunk as reusable.
    void release();
    // Whether the most recent chunk recorded successfully.
    bool last_ok() const { return last_ok_; }

private:
    void worker();
    bool record_to(const string& path);

    int chunk_seconds_;
    string paths_[2];
    bool free_[2] = { true, true };
    bool ready_[2] = { false, false };
    bool ok_[2] = { false, false };
    uint64_t seq_[2] = { 0, 0 };
    uint64_t next_seq_ = 1;

    mutex mtx_;
    condition_variable cv_free_;
    condition_variable cv_ready_;
    bool stop_ = false;
    thread worker_;

    int consumed_ = -1;
    bool last_ok_ = false;
};

AudioBuffer load_wav(const string& path);