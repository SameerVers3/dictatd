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
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/input.h>
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

// Polls all /dev/input/event* keyboards for the Win+Space hotkey.
class HotkeyListener {
public:
    ~HotkeyListener() { close(); }

    bool open() {
        DIR* dir = opendir("/dev/input");
        if (!dir) {
            perror("opendir /dev/input");
            return false;
        }
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (strncmp(entry->d_name, "event", 5) != 0) continue;
            string path = "/dev/input/" + string(entry->d_name);
            int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd < 0) continue;
            if (is_keyboard(fd)) {
                fds_.push_back(fd);
                Logger::log("HOTKEY", "Monitoring " + path);
            } else {
                ::close(fd);
            }
        }
        closedir(dir);
        if (fds_.empty()) {
            Logger::log("HOTKEY", "ERROR: no keyboards found");
            return false;
        }
        return true;
    }

    // Polls hotkey state. Returns 1 when the hotkey goes down, 2 when it
    // comes back up, or 0 when nothing changed.
    int poll() {
        int result = 0;
        for (int fd : fds_) {
            struct input_event ev;
            while (read(fd, &ev, sizeof(ev)) == sizeof(ev)) {
                if (ev.type != EV_KEY) continue;
                if (ev.code == KEY_LEFTMETA || ev.code == KEY_RIGHTMETA) {
                    if (down_ && ev.value == 0 && result == 0) {
                        // Super lifted while we were recording.
                        down_ = false;
                        result = 2;
                    }
                    super_down_ = (ev.value == 1 || ev.value == 2);
                }
                if (ev.code == KEY_SPACE) {
                    if (ev.value == 0) {
                        if (down_ && result == 0) {
                            down_ = false;
                            result = 2;
                        }
                    } else if (super_down_ && !down_ && result == 0) {
                        down_ = true;
                        result = 1;
                    }
                }
            }
        }
        return result;
    }

    void close() {
        for (int fd : fds_) ::close(fd);
        fds_.clear();
    }

private:
    static bool is_keyboard(int fd) {
        unsigned long evbit = 0;
        if (ioctl(fd, EVIOCGBIT(0, sizeof(evbit)), &evbit) < 0) return false;
        return (evbit & (1UL << EV_KEY)) != 0;
    }

    vector<int> fds_;
    bool super_down_ = false;
    bool down_ = false;
};

// Applies grammar correction on a background thread so decoding of the next
// audio region is never blocked by the corrector.
class CorrectionWorker {
public:
    explicit CorrectionWorker(LlamaCorrector* llama) : llama_(llama) {}

    ~CorrectionWorker() { stop(); }

    void start() {
        {
            lock_guard<mutex> lock(mtx_);
            done_ = false;
        }
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

            // Type the corrected text into whichever field has focus.
            FILE* wt = popen("wtype -", "w");
            if (wt) {
                fwrite(corrected.c_str(), 1, corrected.size(), wt);
                pclose(wt);
            }
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
        cerr << "Usage: " << argv[0] << " <whisper_model> <llama_model>\n\n"
             << "Models are loaded once at startup. Hold Win+Space to dictate;\n"
             << "audio is transcribed (and grammar-corrected) while held, and\n"
             << "the utterance is finalized when you release the keys.\n\n"
             << "Arguments:\n"
             << "  whisper_model   Path to whisper.cpp model\n"
             << "  llama_model     Path to llama.cpp GGUF model for grammar correction\n";
        return 1;
    }
    string whisper_model = argv[1];
    string llama_model = argv[2];

    // Dictation is processed in fixed-length audio chunks. Each chunk is
    // transcribed on the main thread while the recorder thread keeps
    // capturing the next chunk in the ring, so listening never stops.
    const double kChunkSeconds = 5.0;

    // Pauses/silence below this RMS skip transcription entirely. Feeding a
    // silent chunk (with the prompt context) makes whisper-basemodels
    // hallucinate repeats of the last sentence on the empty audio.
    const double kSilenceRms = 0.005;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    Logger::log("MAIN", "========================================");
    Logger::log("MAIN", "Dictation Daemon (hotkey toggled)");
    Logger::log("MAIN", "Whisper model: " + whisper_model);
    Logger::log("MAIN", "LLama model:   " + llama_model);
    Logger::log("MAIN", "========================================");

    // Load both models up front, before the hotkey loop starts.
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

    HotkeyListener hotkey;
    if (!hotkey.open()) {
        Logger::log("MAIN", "FATAL: no keyboard device available");
        return 1;
    }

    Logger::log("MAIN", "Models loaded. Hold Win+Space to dictate; "
                        "release to stop.");

    AudioRecorder recorder(kSampleRate, 20);
    CorrectionWorker corrector(&llama);

    bool active = false;
    double from_s = 0.0;        // committed frontier, seconds
    string committed_all;       // rolling text of everything committed so far
    string pending;             // committed but not yet sent to the corrector
    int quiet_ticks = 0;
    uint64_t n_ticks = 0;

    // Transcribe one fixed-length chunk [from_s, from_s + kChunkSeconds).
    // The recorder thread keeps capturing the next chunk in the ring while
    // whisper runs, so a chunk is always waiting when this finishes. With
    // final=true (on hotkey release) whatever audio remains is transcribed.
    auto run_decode = [&](bool final) {
        double now_s = (double)recorder.sample_count() / kSampleRate;
        double end = final ? now_s : min(from_s + kChunkSeconds, now_s);
        if (end - from_s < 0.25) return;
        n_ticks++;

        Logger::log("MAIN", "===== TICK #" + to_string(n_ticks)
                    + " decode [" + to_string((long long)from_s)
                    + "s, " + to_string((long long)end) + "s) =====");

        AudioBuffer clip = recorder.slice((uint64_t)(from_s * kSampleRate),
                                          (uint64_t)(end * kSampleRate));

        // Gate on silence so empty chunks don't reach whisper (it would
        // hallucinate repeats of the prompt context). Advance the frontier
        // and wait for the next chunk instead.
        double rms = 0.0;
        for (float v : clip.samples) rms += v * v;
        rms = clip.samples.empty() ? 0.0 : sqrt(rms / clip.samples.size());
        if (rms < kSilenceRms) {
            Logger::log("MAIN", "chunk [" + to_string((long long)from_s)
                        + "s, " + to_string((long long)end)
                        + "s) is silent (rms=" + to_string(rms)
                        + "), skipping");
            from_s = end;
            quiet_ticks++;
            return;
        }

        auto segs = whisper.transcribe(clip.samples, clip.sample_rate,
                                       prompt_context(committed_all, 300));

        string new_text;
        for (const auto& sg : segs) {
            if (!new_text.empty() && sg.text[0] != '.' && sg.text[0] != ','
                && sg.text[0] != '!' && sg.text[0] != '?' && sg.text[0] != ';') {
                new_text += ' ';
            }
            new_text += sg.text;
        }

        from_s = end;

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
    };

    while (g_running) {
        int hot = hotkey.poll();
        if (hot == 1 && !active) {
            // Start recording as soon as the hotkey is pressed.
            active = true;
            from_s = 0.0;
            committed_all.clear();
            pending.clear();
            quiet_ticks = 0;
            n_ticks = 0;
            recorder.start();
            corrector.start();
            Logger::log("MAIN", "===== RECORDING (release Win+Space to stop) =====");
        } else if (hot == 2 && active) {
            // Release finalizes the utterance: transcribe whatever tail of the
            // current chunk remains, then drain the corrector.
            Logger::log("MAIN", "===== STOPPING =====");
            double now_s = (double)recorder.sample_count() / kSampleRate;
            if (now_s - from_s >= 0.5) run_decode(true);
            if (!pending.empty()) {
                Logger::log("MAIN", "Sending final text to grammar corrector");
                corrector.submit(pending);
                pending.clear();
            }
            corrector.stop();
            recorder.shutdown();
            active = false;
            Logger::log("MAIN", "===== DICTATION DONE =====");
        }

        if (active) {
            double now_s = (double)recorder.sample_count() / kSampleRate;

            // A full chunk is ready: transcribe it (recorder keeps capturing
            // the next chunk concurrently). Otherwise wait for more audio.
            if (now_s - from_s >= kChunkSeconds) {
                run_decode(false);
            } else {
                this_thread::sleep_for(chrono::milliseconds(50));
            }
        } else {
            this_thread::sleep_for(chrono::milliseconds(5));
        }
    }

    // Shutdown on signal: tear down any active session.
    if (active) {
        double now_s = (double)recorder.sample_count() / kSampleRate;
        if (now_s - from_s >= 0.5) run_decode(true);
        if (!pending.empty()) corrector.submit(pending);
        corrector.stop();
        recorder.shutdown();
    }

    cout << "\n";
    Logger::log("MAIN", "Shutdown signal received. Exiting...");
    Logger::log("MAIN", "Total ticks processed: " + to_string(n_ticks));
    Logger::log("MAIN", "Goodbye!");
    return 0;
}