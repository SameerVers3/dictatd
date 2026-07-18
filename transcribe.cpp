#include "transcribe.h"
#include "audio.h"
#include "timer.h"
#include "logger.h"
#include "whisper.h"
#include <thread>
#include <algorithm>

using namespace std;

string transcribe_with_whisper(const string& audio_path, const string& model_path) {
    Timer timer("WHISPER");
    
    Logger::log("WHISPER", "Loading model: " + model_path);
    
    auto t0 = chrono::high_resolution_clock::now();
    
    whisper_context_params ctx_params = whisper_context_default_params();
    
    struct whisper_context* ctx = whisper_init_from_file_with_params(
        model_path.c_str(), ctx_params);
    
    if (!ctx) {
        Logger::log("WHISPER", "ERROR: Failed to load whisper model");
        return "";
    }
    
    auto t1 = chrono::high_resolution_clock::now();
    
    Logger::log("WHISPER", "Model loaded in "
                + Timer::format_time(chrono::duration<double>(t1 - t0).count()));
    Logger::log("WHISPER", "Loading audio: " + audio_path);
    
    AudioBuffer audio = load_wav(audio_path);
    
    if (audio.samples.empty()) {
        Logger::log("WHISPER", "ERROR: No audio data loaded");
        whisper_free(ctx);
        return "";
    }
    
    Logger::log("WHISPER", "Audio: " + to_string(audio.samples.size())
                + " samples @ " + to_string(audio.sample_rate) + " Hz");
    
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.translate = false;
    wparams.language = "en";
    wparams.n_threads = max(1, static_cast<int>(thread::hardware_concurrency()));
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.suppress_blank = true;
    wparams.suppress_nst = true;
    
    Logger::log("WHISPER", "Running inference with "
                + to_string(wparams.n_threads) + " threads...");
    
    auto t2 = chrono::high_resolution_clock::now();
    
    if (whisper_full(ctx, wparams, audio.samples.data(),
                     static_cast<int>(audio.samples.size())) != 0) {
        Logger::log("WHISPER", "ERROR: whisper_full() failed");
        whisper_free(ctx);
        return "";
    }
    
    auto t3 = chrono::high_resolution_clock::now();
    
    Logger::log("WHISPER", "Inference completed in "
                + Timer::format_time(chrono::duration<double>(t3 - t2).count()));
    
    string text;
    
    int n_segments = whisper_full_n_segments(ctx);
    
    for (int i = 0; i < n_segments; i++) {
        const char* segment_text = whisper_full_get_segment_text(ctx, i);
        if (segment_text) {
            text += segment_text;
        }
    }
    
    text.erase(0, text.find_first_not_of(" \t\n\r"));
    text.erase(text.find_last_not_of(" \t\n\r") + 1);
    
    Logger::log("WHISPER", "Transcription: \"" + text + "\"");
    Logger::log("WHISPER", "Segments: " + to_string(n_segments));
    
    whisper_free(ctx);
    
    return text;
}
