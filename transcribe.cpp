#include "transcribe.h"
#include "audio.h"
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

    n_threads_ = max(1, static_cast<int>(thread::hardware_concurrency()));
    Logger::log("WHISPER", "Model loaded. Inference will use "
                + to_string(n_threads_) + " threads.");
    return true;
}

string WhisperEngine::transcribe(const string& audio_path) {
    Timer timer("WHISPER");

    AudioBuffer audio = load_wav(audio_path);

    if (audio.samples.empty()) {
        Logger::log("WHISPER", "ERROR: No audio data loaded: " + audio_path);
        return "";
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

    auto t0 = chrono::high_resolution_clock::now();

    if (whisper_full(ctx_, wparams, audio.samples.data(),
                     static_cast<int>(audio.samples.size())) != 0) {
        Logger::log("WHISPER", "ERROR: whisper_full() failed");
        return "";
    }

    auto t1 = chrono::high_resolution_clock::now();
    Logger::log("WHISPER", "Inference took "
                + Timer::format_time(chrono::duration<double>(t1 - t0).count()));

    string text;
    int n_segments = whisper_full_n_segments(ctx_);

    for (int i = 0; i < n_segments; i++) {
        const char* segment_text = whisper_full_get_segment_text(ctx_, i);
        if (segment_text) {
            text += segment_text;
        }
    }

    text.erase(0, text.find_first_not_of(" \t\n\r"));
    text.erase(text.find_last_not_of(" \t\n\r") + 1);

    Logger::log("WHISPER", "Transcription: \"" + text + "\"");
    return text;
}