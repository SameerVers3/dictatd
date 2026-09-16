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
#include <poll.h>
#include <sys/ioctl.h>
#include <linux/input.h>
#include <linux/uinput.h>
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

// Grabs the real keyboards and re-emits every key through a uinput virtual
// keyboard, swallowing the Left Shift + S dictation combo so it never types
// into the focused app. Held keys are released on shutdown so nothing sticks.
//
// The grab/forward loop runs on its own thread so re-emission never stalls
// while the main thread is busy transcribing/correcting audio. If the main
// thread handled reads itself, key-up events would pile up in the evdev
// buffer during a slow decode and get dropped, leaving keys stuck down.
class HotkeyListener {
public:
    HotkeyListener() : key_down_(KEY_MAX, 0) {}

    ~HotkeyListener() { close(); }

    bool open() {
        // Grab real keyboards first, so the uinput device created below is
        // not picked up by the scan (that would cause an echo loop).
        if (!open_keyboards()) {
            Logger::log("HOTKEY", "ERROR: no keyboards found");
            return false;
        }
        if (!open_uinput()) {
            for (int fd : fds_) ioctl(fd, EVIOCGRAB, 0);
            Logger::log("HOTKEY", "ERROR: cannot create virtual keyboard "
                        "(need write access to /dev/uinput)");
            return false;
        }
        running_ = true;
        th_ = thread([this] { poll_loop(); });
        return true;
    }

    // Virtual keyboard used for re-emission; the corrector reuses it to paste.
    int uinput_fd() const { return uinput_fd_; }
    // Serializes writes to the virtual keyboard: the forwarder thread and the
    // corrector thread (Ctrl+V pasting) never interleave event streams.
    mutex& uinput_mtx() const { return out_mtx_; }
    // Current physical Left Shift state (set on the grab thread). The
    // corrector reads it so its Ctrl+V is never sent as Ctrl+Shift+V while the
    // user is still holding the dictation combo.
    bool physical_shift_down() const { return physical_shift_.load(); }

    // Returns 1 when the combo goes down, 2 when it comes back up, or 0 when
    // nothing changed. Called from the main thread; the grab thread just
    // records transitions into pending_ for us to drain.
    int poll() {
        lock_guard<mutex> lock(state_mtx_);
        if (pending_ & 1) {
            pending_ &= ~1;
            return 1;
        }
        if (pending_ & 2) {
            pending_ &= ~2;
            return 2;
        }
        return 0;
    }

    void close() {
        {
            lock_guard<mutex> lock(state_mtx_);
            running_ = false;
        }
        if (th_.joinable()) th_.join();
        if (uinput_fd_ >= 0) {
            struct input_event rpt;
            memset(&rpt, 0, sizeof(rpt));
            rpt.type = EV_KEY;
            for (int c = 0; c < KEY_MAX; c++) {
                if (key_down_[c]) {
                    rpt.code = c;
                    rpt.value = 0;
                    lock_guard<mutex> lock(out_mtx_);
                    write(uinput_fd_, &rpt, sizeof(rpt));
                }
            }
            rpt.type = EV_SYN;
            rpt.code = SYN_REPORT;
            write(uinput_fd_, &rpt, sizeof(rpt));
            ::close(uinput_fd_);
            uinput_fd_ = -1;
        }
        for (int fd : fds_) {
            ioctl(fd, EVIOCGRAB, 0);
            ::close(fd);
        }
        fds_.clear();
    }

private:
    // Dedicated event loop: reads grabbed keyboards, re-emits every key via
    // the virtual keyboard, and records Left Shift+S transitions.
    void poll_loop() {
        vector<pollfd> pfds;
        pfds.reserve(fds_.size());
        for (int fd : fds_) pfds.push_back({fd, POLLIN, 0});
        while (true) {
            {
                lock_guard<mutex> lock(state_mtx_);
                if (!running_) break;
            }
            int rc = ::poll(pfds.data(), pfds.size(), 100);
            if (rc <= 0) continue;

            bool new_press = false;
            bool new_release = false;
            for (auto& p : pfds) {
                if ((p.revents & POLLIN) == 0) continue;
                struct input_event ev;
                while (read(p.fd, &ev, sizeof(ev)) == sizeof(ev)) {
                    bool prev = left_shift_ && s_in_combo_;
                    if (ev.type == EV_KEY) handle_key(ev);
                    if (!swallow_ && (ev.type == EV_KEY || ev.type == EV_SYN)) {
                        forward(ev);
                    }
                    bool now = left_shift_ && s_in_combo_;
                    if (now && !prev) new_press = true;
                    if (!now && prev) new_release = true;
                }
                p.revents = 0;
            }
            if (new_press || new_release) {
                lock_guard<mutex> lock(state_mtx_);
                if (new_press) pending_ |= 1;
                if (new_release) pending_ |= 2;
            }
        }
    }

    bool open_keyboards() {
        DIR* dir = opendir("/dev/input");
        if (!dir) return false;
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (strncmp(entry->d_name, "event", 5) != 0) continue;
            string path = "/dev/input/" + string(entry->d_name);
            int fd = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd < 0) continue;
            if (is_keyboard(fd)) {
                if (ioctl(fd, EVIOCGRAB, 1) == 0) {
                    Logger::log("HOTKEY", "Grabbed " + path);
                } else {
                    Logger::log("HOTKEY", "WARN: cannot grab " + path
                                + ", combo will type through");
                }
                fds_.push_back(fd);
            } else {
                ::close(fd);
            }
        }
        closedir(dir);
        return !fds_.empty();
    }

    bool open_uinput() {
        uinput_fd_ = ::open("/dev/uinput", O_WRONLY | O_NONBLOCK);
        if (uinput_fd_ < 0) return false;
        struct uinput_setup setup;
        memset(&setup, 0, sizeof(setup));
        setup.id.bustype = BUS_VIRTUAL;
        setup.id.vendor = 0x7370;  // "sp"
        setup.id.product = 0x7669; // "vi"
        snprintf(setup.name, sizeof(setup.name),
                 "voice_pipeline virtual keyboard");
        ioctl(uinput_fd_, UI_SET_EVBIT, EV_KEY);
        ioctl(uinput_fd_, UI_SET_EVBIT, EV_SYN);
        for (int c = 0; c < KEY_MAX; c++) {
            ioctl(uinput_fd_, UI_SET_KEYBIT, c);
        }
        if (ioctl(uinput_fd_, UI_DEV_SETUP, &setup) < 0) return false;
        if (ioctl(uinput_fd_, UI_DEV_CREATE) < 0) return false;
        return true;
    }

    void handle_key(const struct input_event& ev) {
        swallow_ = false;
        if (ev.code == KEY_LEFTSHIFT) {
            physical_shift_ = (ev.value != 0);
            left_shift_ = (ev.value != 0);
            // Shift itself always forwards; the 'S' is what we swallow.
        } else if (ev.code == KEY_S) {
            if (left_shift_ && ev.value == 1 && !s_in_combo_) {
                s_in_combo_ = true;
                swallow_ = true; // start of the combo: eat it
            } else if (s_in_combo_) {
                swallow_ = true; // repeat + release stay swallowed (no orphan key-up)
                if (ev.value == 0) s_in_combo_ = false;
            }
            // S without left shift forwards as a normal key.
        }
    }

    void forward(const struct input_event& ev) {
        if (uinput_fd_ < 0) return;
        if (ev.type == EV_KEY) {
            if (ev.code >= KEY_MAX) return;
            if (ev.value == 1) key_down_[ev.code] = 1;
            else if (ev.value == 0) key_down_[ev.code] = 0;
        }
        struct input_event out = ev;
        lock_guard<mutex> lock(out_mtx_);
        write(uinput_fd_, &out, sizeof(out));
    }

    static bool is_keyboard(int fd) {
        unsigned long evbit = 0;
        if (ioctl(fd, EVIOCGBIT(0, sizeof(evbit)), &evbit) < 0) return false;
        if ((evbit & (1UL << EV_KEY)) == 0) return false;
        // Require typical letter keys — mice/touchpads only have BTN_* codes,
        // never KEY_A..KEY_Z, so this filters them out.
        unsigned char keybit[KEY_MAX / 8 + 1];
        memset(keybit, 0, sizeof(keybit));
        if (ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(keybit)), keybit) < 0) return false;
        static const int kLetters[] = {
            KEY_Q, KEY_W, KEY_E, KEY_R, KEY_T, KEY_Y, KEY_U, KEY_I, KEY_O, KEY_P,
            KEY_A, KEY_S, KEY_D, KEY_F, KEY_G, KEY_H, KEY_J, KEY_K, KEY_L,
            KEY_Z, KEY_X, KEY_C, KEY_V, KEY_B, KEY_N, KEY_M,
        };
        for (int c : kLetters) {
            if (keybit[c / 8] & (1U << (c % 8))) return true;
        }
        return false;
    }

    vector<int> fds_;
    int uinput_fd_ = -1;
    mutable mutex out_mtx_;
    atomic<bool> physical_shift_{false};
    bool left_shift_ = false;
    bool s_in_combo_ = false;
    bool swallow_ = false;
    vector<unsigned char> key_down_;

    bool running_ = false;
    thread th_;
    mutex state_mtx_;
    unsigned char pending_ = 0; // bit 0 = combo pressed, bit 1 = released
};

// Applies grammar correction on a background thread so decoding of the next
// audio region is never blocked by the corrector. Corrected text is typed
// directly into the focused field through the uinput virtual keyboard.
class CorrectionWorker {
public:
    explicit CorrectionWorker(LlamaCorrector* llama, HotkeyListener* hotkey,
                              mutex& uinput_mtx)
        : llama_(llama), hotkey_(hotkey), uinput_mtx_(uinput_mtx) {}

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

    // Every corrected sentence is pasted immediately (streaming), so the user
    // sees text appear while dictating instead of only at the end.
    void flush() {}

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
            if (corrected.empty()) continue;

            // Whisper commits chunks without trailing spaces, so sentences
            // pasted one after the other would run together. Join them.
            {
                lock_guard<mutex> lock(mtx_);
                if (have_prev_ && corrected[0] != ' ' && !prev_ends_space_) {
                    corrected = " " + corrected;
                }
            }
            paste_text(corrected);
            {
                lock_guard<mutex> lock(mtx_);
                have_prev_ = true;
                prev_ends_space_ = corrected.back() == ' ';
            }
        }
    }

    // Emit one key press+release through the virtual keyboard.
    void press_key(unsigned short code, int value) {
        if (hotkey_ == nullptr) return;
        struct input_event ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = EV_KEY;
        ev.code = code;
        ev.value = value;
        write(hotkey_->uinput_fd(), &ev, sizeof(ev));
        ev.type = EV_SYN;
        ev.code = SYN_REPORT;
        ev.value = 0;
        write(hotkey_->uinput_fd(), &ev, sizeof(ev));
    }

    // Map an ASCII char to a keycode + whether it needs Left Shift.
    static bool char_to_key(char c, unsigned short& code, bool& shifted) {
        shifted = false;
        if (c >= 'a' && c <= 'z') { code = KEY_A + (c - 'a'); return true; }
        if (c >= 'A' && c <= 'Z') { code = KEY_A + (c - 'A'); shifted = true; return true; }
        if (c >= '1' && c <= '9') { code = KEY_1 + (c - '1'); return true; }
        switch (c) {
            case '0': code = KEY_0; return true;
            case ' ': code = KEY_SPACE; return true;
            case '.': code = KEY_DOT; return true;
            case ',': code = KEY_COMMA; return true;
            case '-': code = KEY_MINUS; return true;
            case '_': code = KEY_MINUS; shifted = true; return true;
            case '=': code = KEY_EQUAL; return true;
            case '+': code = KEY_EQUAL; shifted = true; return true;
            case '/': code = KEY_SLASH; return true;
            case '?': code = KEY_SLASH; shifted = true; return true;
            case '\\': code = KEY_BACKSLASH; return true;
            case '|': code = KEY_BACKSLASH; shifted = true; return true;
            case '\'': code = KEY_APOSTROPHE; return true;
            case '"': code = KEY_APOSTROPHE; shifted = true; return true;
            case ';': code = KEY_SEMICOLON; return true;
            case ':': code = KEY_SEMICOLON; shifted = true; return true;
            case '[': code = KEY_LEFTBRACE; return true;
            case '{': code = KEY_LEFTBRACE; shifted = true; return true;
            case ']': code = KEY_RIGHTBRACE; return true;
            case '}': code = KEY_RIGHTBRACE; shifted = true; return true;
            case '`': code = KEY_GRAVE; return true;
            case '~': code = KEY_GRAVE; shifted = true; return true;
            case '!': code = KEY_1; shifted = true; return true;
            case '@': code = KEY_2; shifted = true; return true;
            case '#': code = KEY_3; shifted = true; return true;
            case '$': code = KEY_4; shifted = true; return true;
            case '%': code = KEY_5; shifted = true; return true;
            case '^': code = KEY_6; shifted = true; return true;
            case '&': code = KEY_7; shifted = true; return true;
            case '*': code = KEY_8; shifted = true; return true;
            case '(': code = KEY_9; shifted = true; return true;
            case ')': code = KEY_0; shifted = true; return true;
            case '<': code = KEY_COMMA; shifted = true; return true;
            case '>': code = KEY_DOT; shifted = true; return true;
            case '\n': code = KEY_ENTER; return true;
            case '\t': code = KEY_TAB; return true;
            default:  return false;
        }
    }

    // Decode a UTF-8 codepoint at s[i], advancing i past it. Invalid/partial
    // sequences advance one byte and return -1.
    static uint32_t utf8_codepoint(const string& s, size_t& i) {
        unsigned char b = (unsigned char)s[i];
        if (b < 0x80) { uint32_t cp = b; i++; return cp; }
        int n, mask;
        if ((b & 0xE0) == 0xC0)        { n = 1; mask = 0x1F; }
        else if ((b & 0xF0) == 0xE0)   { n = 2; mask = 0x0F; }
        else if ((b & 0xF8) == 0xF0)   { n = 3; mask = 0x07; }
        else { i++; return 0xFFFFFFFF; }
        if (i + (size_t)n >= s.size()) { i++; return 0xFFFFFFFF; }
        uint32_t cp = b & mask;
        for (int k = 0; k < n; k++) {
            unsigned char cb = (unsigned char)s[i + 1 + k];
            if ((cb & 0xC0) != 0x80) { i++; return 0xFFFFFFFF; }
            cp = (cp << 6) | (cb & 0x3F);
        }
        i += 1 + n;
        return cp;
    }

    // Type text at the current cursor by emitting per-character keycodes
    // through the uinput virtual keyboard, managing Left Shift as needed.
    // Fallback used only when wl-copy/clipboard paste is unavailable.
    void inject_text(const string& text) {
        if (hotkey_ == nullptr || text.empty()) return;
        lock_guard<mutex> lock(uinput_mtx_);
        bool shift_down = false;
        auto set_shift = [&](bool on) {
            if (on == shift_down) return;
            press_key(KEY_LEFTSHIFT, on ? 1 : 0);
            shift_down = on;
        };
        for (size_t i = 0; i < text.size();) {
            uint32_t cp = utf8_codepoint(text, i);
            if (cp == 0xFFFFFFFF) continue;
            // Common punctuation produced by the models -> ASCII equivalents.
            switch (cp) {
                case 0x2018: case 0x2019: cp = '\''; break;
                case 0x201C: case 0x201D: cp = '"'; break;
                case 0x2013: case 0x2014: cp = '-'; break;
                case 0x00A0: cp = ' '; break;
                case 0x2026: cp = '.'; break;
                default: break;
            }
            unsigned short code;
            bool shifted;
            if (cp > 0x7F || !char_to_key((char)cp, code, shifted)) continue;
            set_shift(shifted);
            press_key(code, 1);
            press_key(code, 0);
        }
        set_shift(false);
    }

    // Insert text into the focused field via the Wayland clipboard + Ctrl+V.
    // This is far more robust than per-key injection (no layout/shift/keystate
    // dependence), so pastes can never come out as garbage. While the user is
    // still holding Left Shift to dictate, lift shift around the paste so the
    // app receives Ctrl+V and not Ctrl+Shift+V.
    void paste_text(const string& text) {
        if (text.empty()) return;
        Logger::log("RESULT", "INSERTING:  \"" + text + "\"");
        FILE* wc = popen("wl-copy", "w");
        if (!wc) {
            Logger::log("RESULT", "wl-copy unavailable, falling back to typing");
            inject_text(text);
            return;
        }
        fwrite(text.c_str(), 1, text.size(), wc);
        pclose(wc);
        // Let the wl-copy daemon take ownership of the selection before pasting.
        usleep(200 * 1000);
        lock_guard<mutex> lock(uinput_mtx_);
        bool shift_state = hotkey_ && hotkey_->physical_shift_down();
        if (shift_state) press_key(KEY_LEFTSHIFT, 0);
        press_key(KEY_LEFTCTRL, 1);
        press_key(KEY_V, 1);
        press_key(KEY_V, 0);
        press_key(KEY_LEFTCTRL, 0);
        if (shift_state) press_key(KEY_LEFTSHIFT, 1);
        Logger::log("RESULT", "PASTED:     \"" + text + "\"");
    }

    LlamaCorrector* llama_;
    HotkeyListener* hotkey_;
    mutex& uinput_mtx_;
    thread th_;
    mutex mtx_;
    condition_variable cv_;
    queue<string> q_;
    bool have_prev_ = false;    // a previous sentence was pasted this session
    bool prev_ends_space_ = false;
    bool done_ = false;
};

// Rolling tail of committed text, fed back to whisper so contiguous windows
// stay coherent across the commit frontier.
static string prompt_context(const string& all, size_t max_chars) {
    if (all.size() <= max_chars) return all;
    return all.substr(all.size() - max_chars);
}

int main(int argc, char** argv) {
    int hw = thread::hardware_concurrency();
    int wthreads = max(2, min(8, hw - max(2, min(4, hw / 4))));
    int lthreads = max(2, min(4, hw / 4));
    double kChunkSeconds = 3.0;

    // Parse optional tuning flags before the two model paths.
    vector<string> positional;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto need_int = [&](const char* name) -> int {
            if (i + 1 >= argc) {
                cerr << "Option " << name << " requires a number\n";
                exit(1);
            }
            return atoi(argv[++i]);
        };
        if (a == "--whisper-threads") wthreads = max(1, need_int("--whisper-threads"));
        else if (a == "--llama-threads") lthreads = max(1, need_int("--llama-threads"));
        else if (a == "--chunk-seconds") kChunkSeconds = max(1.0, atof(argv[++i]));
        else positional.push_back(a);
    }
    if (positional.size() < 2) {
        cerr << "Usage: " << argv[0] << " <whisper_model> <llama_model>\n\n"
             << "Models are loaded once at startup. Hold Left Shift+S to dictate;\n"
             << "audio is transcribed (and grammar-corrected) while held, and each\n"
             << "corrected sentence is pasted at the cursor the moment it is ready.\n\n"
             << "Arguments:\n"
             << "  whisper_model        Path to whisper.cpp model\n"
             << "  llama_model          Path to llama.cpp GGUF model for grammar correction\n\n"
             << "Options (all optional):\n"
             << "  --whisper-threads N  parallel threads for whisper (default "
             << wthreads << ")\n"
             << "  --llama-threads N    parallel threads for llama grammar (default "
             << lthreads << ")\n"
             << "  --chunk-seconds S    transcription chunk length (default "
             << kChunkSeconds << ")  [lower = lower first-word latency]\n";
        return 1;
    }
    string whisper_model = positional[0];
    string llama_model = positional[1];

    // Dictation is processed in fixed-length audio chunks. Each chunk is
    // transcribed on the main thread while the recorder thread keeps
    // capturing the next chunk in the ring, so listening never stops.
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
    Logger::log("MAIN", "Threads:       whisper=" + to_string(wthreads)
                + ", llama=" + to_string(lthreads));
    Logger::log("MAIN", "Chunk length:  " + to_string(kChunkSeconds) + "s");
    Logger::log("MAIN", "========================================");

    // Load both models up front, before the hotkey loop starts.
    WhisperEngine whisper;
    whisper.set_threads(wthreads);
    if (!whisper.init(whisper_model)) {
        Logger::log("MAIN", "FATAL: failed to load whisper model");
        return 1;
    }

    LlamaCorrector llama;
    llama.set_threads(lthreads);
    if (!llama.init(llama_model)) {
        Logger::log("MAIN", "FATAL: failed to load llama model");
        return 1;
    }

    HotkeyListener hotkey;
    if (!hotkey.open()) {
        Logger::log("MAIN", "FATAL: no keyboard device available");
        return 1;
    }

    Logger::log("MAIN", "Models loaded. Hold Left Shift+S to dictate; "
                        "release to stop.");

    AudioRecorder recorder(kSampleRate, 20);
    CorrectionWorker corrector(&llama, &hotkey, hotkey.uinput_mtx());

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
        if (end - from_s < 0.2) return;
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
            // Whisper chunks rarely end in a space; join the committed text so
            // sentences don't run together when pasted one after another.
            if (!pending.empty() && new_text[0] != ' ' && new_text[0] != '.'
                && new_text[0] != ',' && new_text[0] != '!' && new_text[0] != '?'
                && new_text[0] != ';') {
                pending += ' ';
            }
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
            Logger::log("MAIN", "===== RECORDING (release Left Shift+S to stop) =====");
        } else if (hot == 2 && active) {
            // Release finalizes the utterance: transcribe whatever tail of the
            // current chunk remains, then drain the corrector.
            Logger::log("MAIN", "===== STOPPING =====");
            double now_s = (double)recorder.sample_count() / kSampleRate;
            if (now_s - from_s >= 0.2) run_decode(true);
            if (!pending.empty()) {
                Logger::log("MAIN", "Sending final text to grammar corrector");
                corrector.submit(pending);
                pending.clear();
            }
            corrector.stop();
            corrector.flush();
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
        if (now_s - from_s >= 0.2) run_decode(true);
        if (!pending.empty()) corrector.submit(pending);
        corrector.stop();
        corrector.flush();
        recorder.shutdown();
    }

    cout << "\n";
    Logger::log("MAIN", "Shutdown signal received. Exiting...");
    Logger::log("MAIN", "Total ticks processed: " + to_string(n_ticks));
    Logger::log("MAIN", "Goodbye!");
    return 0;
}