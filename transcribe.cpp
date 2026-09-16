#include "transcribe.h"
#include "timer.h"
#include "logger.h"
#include "whisper.h"
#include <thread>
#include <algorithm>
#include <chrono>

using namespace std;

WhisperEngine::~WhisperEngine() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

bool WhisperEngine::init(const string& model_path) {
    Logger::log("WHISPER", "Loading model (once): " + model_path);

    whisper_context_params ctx_params = whisper_context_default_params();
    ctx_ = whisper_init_from_file_with_params(model_path.c_str(), ctx_params);

    if (!ctx_) {
        Logger::log("WHISPER", "ERROR: Failed to load whisper model");
        return false;
    }

    n_threads_ = 3;
    Logger::log("WHISPER", "Model loaded. Inference will use "
                + to_string(n_threads_) + " threads.");
    return true;
}

vector<WhisperSegment> WhisperEngine::transcribe(const vector<float>& samples,
                                                 int sample_rate,
                                                 const string& context_text) {
    Timer timer("WHISPER");

    if (samples.empty()) {
        Logger::log("WHISPER", "ERROR: No audio samples provided");
        return {};
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.translate = false;
    wparams.language = "en";
    wparams.n_threads = n_threads_;
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.suppress_blank = true;
    wparams.suppress_nst = true;
    wparams.no_context = false; // feed past text (prompt_tokens) as context

    vector<whisper_token> prompt;
    if (!context_text.empty()) {
        int cap = min(whisper_n_text_ctx(ctx_), 2048);
        prompt.resize(cap);
        int n = whisper_tokenize(ctx_, context_text.c_str(), prompt.data(), cap);
        if (n < 0) {
            prompt.clear();
        } else {
            prompt.resize(n);
            wparams.prompt_tokens = prompt.data();
            wparams.prompt_n_tokens = static_cast<int>(prompt.size());
        }
    }

    auto t0 = chrono::high_resolution_clock::now();

    if (whisper_full(ctx_, wparams, samples.data(),
                     static_cast<int>(samples.size())) != 0) {
        Logger::log("WHISPER", "ERROR: whisper_full() failed");
        return {};
    }

    auto t1 = chrono::high_resolution_clock::now();
    Logger::log("WHISPER", "Inference took "
                + Timer::format_time(chrono::duration<double>(t1 - t0).count()));

    vector<WhisperSegment> segs;
    int n_segments = whisper_full_n_segments(ctx_);

    for (int i = 0; i < n_segments; i++) {
        const char* segment_text = whisper_full_get_segment_text(ctx_, i);
        if (!segment_text) continue;

        WhisperSegment seg;
        seg.text = segment_text;
        // t0/t1 are centiseconds; convert to seconds relative to window start.
        seg.t0 = whisper_full_get_segment_t0(ctx_, i) / 100.0;
        seg.t1 = whisper_full_get_segment_t1(ctx_, i) / 100.0;

        size_t b = seg.text.find_first_not_of(" \t\n\r");
        size_t e = seg.text.find_last_not_of(" \t\n\r");
        if (b == string::npos || e == string::npos) continue;
        seg.text = seg.text.substr(b, e - b + 1);

        segs.push_back(seg);
    }

    return segs;
}