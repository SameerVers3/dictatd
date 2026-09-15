#include <iostream>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "logger.h"
#include "timer.h"
#include "audio.h"
#include "transcribe.h"
#include "grammar.h"

using namespace std;

atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

static const int kSampleRate = 16000;

// Applies grammar correction on a background thread so decoding of the next
// audio region is never blocked by the corrector.
class CorrectionWorker {
public:
    explicit CorrectionWorker(LlamaCorrector* llama) : llama_(llama) {}

    ~CorrectionWorker() { stop(); }

    void start() {
        th_ = thread([this] { run(); });
    }

    void submit(string text) {
        {
            lock_guard<mutex> lock(mtx_);
            q_.push(std::move(text));
        }
        cv_.notify_one();
    }

    void stop() {
        {
            lock_guard<mutex> lock(mtx_);
            done_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

private:
    void run() {
        while (true) {
            string text;
            {
                unique_lock<mutex> lock(mtx_);
                cv_.wait(lock, [this] { return done_ || !q_.empty(); });
                if (q_.empty()) {
                    if (done_) break;
                    continue;
                }
                text = std::move(q_.front());
                q_.pop();
            }

            string corrected = llama_->correct(text);
            if (corrected.empty()) corrected = text;

            Logger::log("RESULT", "RAW:        \"" + text + "\"");
            Logger::log("RESULT", "CORRECTED:  \"" + corrected + "\"");
        }
    }

    LlamaCorrector* llama_;
    thread th_;
    mutex mtx_;
    condition_variable cv_;
    queue<string> q_;
    bool done_ = false;
};

// Rolling tail of committed text, fed back to whisper so contiguous windows
// stay coherent across the commit frontier.
static string prompt_context(const string& all, size_t max_chars) {
    if (all.size() <= max_chars) return all;
    return all.substr(all.size() - max_chars);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <whisper_model> <llama_model> "
             << "[min_region_seconds] [cadence_seconds] [settle_seconds]\n\n"
             << "Streams the microphone with a growing decode region.\n"
             << "The decode region always starts at the committed frontier and\n"
             << "grows on the tail as new audio arrives. Each pass commits every\n"
             << "segment that has settled (ended before now - settle_seconds) and\n"
             << "re-decodes only the unsettled tail with more future context, so\n"
             << "output streams at ~cadence speed while every word is verified.\n\n"
             << "Arguments:\n"
             << "  whisper_model   Path to whisper.cpp model (tiny.en for ~realtime)\n"
             << "  llama_model     Path to llama.cpp GGUF model for grammar correction\n"
             << "  min_region      Min audio before the first decode (default: 3)\n"
             << "  cadence         Min seconds between decode passes (default: 2)\n"
             << "  settle          Tail margin held back for re-decoding (default: 0.4)\n";
        return 1;
    }
    string whisper_model = argv[1];
    string llama_model = argv[2];
    double min_region_s = (argc > 3) ? atof(argv[3]) : 3.0;
    double cadence_s = (argc > 4) ? atof(argv[4]) : 2.0;
    double settle_s = (argc > 5) ? atof(argv[5]) : 0.4;

    if (min_region_s < 1.0) min_region_s = 1.0;
    if (cadence_s < 0.5) cadence_s = 0.5;
    if (settle_s < 0.2) settle_s = 0.2;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    Logger::log("MAIN", "========================================");
    Logger::log("MAIN", "Voice Pipeline Starting (streaming)");
    Logger::log("MAIN", "Whisper model: " + whisper_model);
    Logger::log("MAIN", "LLama model:   " + llama_model);
    Logger::log("MAIN", "min region: " + to_string(min_region_s) + "s, "
                        "cadence: " + to_string(cadence_s) + "s, "
                        "settle: " + to_string(settle_s) + "s");
    Logger::log("MAIN", "Press Ctrl+C to stop");
    Logger::log("MAIN", "========================================");

    WhisperEngine whisper;
    if (!whisper.init(whisper_model)) {
        Logger::log("MAIN", "FATAL: failed to load whisper model");
        return 1;
    }

    LlamaCorrector llama;
    if (!llama.init(llama_model)) {
        Logger::log("MAIN", "FATAL: failed to load llama model");
        return 1;
    }

    // Ring keeps enough history for the region to grow between commits.
    AudioRecorder recorder(kSampleRate, 20);
    if (!recorder.start()) {
        Logger::log("MAIN", "FATAL: failed to start recorder");
        return 1;
    }

    CorrectionWorker corrector(&llama);
    corrector.start();

    double from_s = 0.0;        // committed frontier, seconds
    double last_decode_s = -1;  // last time we ran a decode pass
    string committed_all;       // rolling text of everything committed so far
    string pending;             // committed but not yet sent to the corrector
    int quiet_ticks = 0;
    uint64_t n_ticks = 0;

    while (g_running) {
        double now_s = (double)recorder.sample_count() / kSampleRate;
        double region = now_s - from_s;

        // Need some audio, and (except startup) respect the cadence.
        if (region < min_region_s || (last_decode_s >= 0 && now_s - last_decode_s < cadence_s)) {
            this_thread::sleep_for(chrono::milliseconds(50));
            continue;
        }
        last_decode_s = now_s;
        n_ticks++;

        Logger::log("MAIN", "===== TICK #" + to_string(n_ticks)
                    + " decode [" + to_string((long long)from_s)
                    + "s, " + to_string((long long)now_s) + "s) =====");

        AudioBuffer clip = recorder.slice((uint64_t)(from_s * kSampleRate),
                                          (uint64_t)(now_s * kSampleRate));
        auto segs = whisper.transcribe(clip.samples, clip.sample_rate,
                                       prompt_context(committed_all, 300));

        double threshold = now_s - settle_s;
        string new_text;
        double new_from = from_s;

        for (const auto& sg : segs) {
            double b = from_s + sg.t0;
            double e = from_s + sg.t1;

            // Only settled segments are committed; the tail is re-decoded next
            // pass once more future audio exists.
            if (e > threshold) continue;

            if (!new_text.empty() && sg.text[0] != '.' && sg.text[0] != ','
                && sg.text[0] != '!' && sg.text[0] != '?' && sg.text[0] != ';') {
                new_text += ' ';
            }
            new_text += sg.text;
            new_from = max(new_from, e);
        }

        // Skip dead air so the region doesn't grow unbounded during silence.
        if (new_text.empty() && segs.empty() && region > 8.0) {
            new_from = now_s;
        }
        // During continuous speech with no paused segments, bound the region
        // so decode cost (and latency) can't grow without limit.
        if (new_text.empty() && !segs.empty() && region > 12.0) {
            new_from = now_s - settle_s;
        }
        from_s = new_from;

        if (!new_text.empty()) {
            quiet_ticks = 0;
            pending += new_text;
            committed_all += " " + new_text;
            Logger::log("STREAM", "committed: \"" + new_text + "\"");
            Logger::log("STREAM", "pending:   \"" + pending + "\"");
        } else {
            quiet_ticks++;
        }

        // A sentence is done when it ends in punctuation, or when audio has
        // gone quiet for a couple of ticks. Flush it to the corrector.
        bool punctuated = !pending.empty() &&
            (pending.back() == '.' || pending.back() == '?' ||
             pending.back() == '!' || pending.back() == ';');
        if (!pending.empty() && (punctuated || quiet_ticks >= 2)) {
            Logger::log("MAIN", "Sending sentence to grammar corrector");
            corrector.submit(pending);
            pending.clear();
        }
    }

    if (!pending.empty()) {
        corrector.submit(pending);
    }

    corrector.stop();
    recorder.shutdown();

    cout << "\n";
    Logger::log("MAIN", "Shutdown signal received. Exiting...");
    Logger::log("MAIN", "Total ticks processed: " + to_string(n_ticks));
    Logger::log("MAIN", "Goodbye!");
    return 0;
}