#include <iostream>
#include <csignal>
#include <atomic>
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

void print_banner() {
    cout << R"(
    Voice pipeline experimenting
    )";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <whisper_model> <llama_model> [chunk_seconds]\n\n"
             << "Arguments:\n"
             << "  whisper_model   Path to whisper.cpp model file (ggml/gguf)\n"
             << "  llama_model     Path to llama.cpp model file (gguf)\n"
             << "  chunk_seconds   Duration of each recording chunk (default: 5)\n\n"
             << "Example:\n"
             << "  " << argv[0] << " ../exploration/whisper.cpp/models/ggml-base.en.bin ../exploration/llama.cpp/models/Qwen2.5-0.5B-Instruct.Q2_K.gguf 5\n";
        return 1;
    }
    string whisper_model = argv[1];
    string llama_model = argv[2];
    int chunk_seconds = (argc > 3) ? atoi(argv[3]) : 5;
    if (chunk_seconds < 1) chunk_seconds = 1;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    print_banner();
    Logger::log("MAIN", "========================================");
    Logger::log("MAIN", "Voice Pipeline Starting");
    Logger::log("MAIN", "Whisper model: " + whisper_model);
    Logger::log("MAIN", "LLama model:   " + llama_model);
    Logger::log("MAIN", "Chunk duration: " + to_string(chunk_seconds) + "s");
    Logger::log("MAIN", "Models are loaded once at startup; recording overlaps with transcription.");
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

    AudioRecorder recorder(chunk_seconds);
    if (!recorder.start()) {
        Logger::log("MAIN", "FATAL: failed to start recorder");
        return 1;
    }

    int chunk_num = 0;

    while (g_running) {
        chunk_num++;
        cout << "\n";

        Logger::log("MAIN", "========== CHUNK #" + to_string(chunk_num) + " ==========");

        string wav_path;
        {
            Timer record_timer("RECORD");
            Logger::log("RECORD", "Waiting for the next " + to_string(chunk_seconds)
                        + "s chunk (recording overlaps with processing of the previous one)...");
            wav_path = recorder.next();
        }

        if (wav_path.empty()) {
            Logger::log("RECORD", "ERROR: Recorder stopped unexpectedly");
            break;
        }

        if (!recorder.last_ok()) {
            Logger::log("RECORD", "ERROR: Recording failed. Make sure 'arecord' or 'ffmpeg' is installed.");
            Logger::log("MAIN", "Hint: sudo apt install alsa-utils   # for arecord");
            Logger::log("MAIN", "      sudo apt install ffmpeg       # for ffmpeg");
            recorder.release();
            break;
        }

        string raw_text = whisper.transcribe(wav_path);

        // The WAV file has been read; free the buffer slot now so the next
        // chunk can be recorded while grammar correction is still running.
        recorder.release();

        string corrected_text = raw_text;

        if (!raw_text.empty()) {
            corrected_text = llama.correct(raw_text);
        } else {
            Logger::log("MAIN", "Transcription empty, skipping grammar correction");
        }

        if (corrected_text.empty()) {
            Logger::log("MAIN", "Grammar correction failed, using raw transcription");
            corrected_text = raw_text;
        }

        cout << "\n";

        Logger::log("RESULT", "RAW:        \"" + raw_text + "\"");
        Logger::log("RESULT", "CORRECTED:  \"" + corrected_text + "\"");

        cout << "\n";

        Logger::log("MAIN", "========== END CHUNK #" + to_string(chunk_num) + " ==========");
    }

    recorder.shutdown();

    cout << "\n";

    Logger::log("MAIN", "Shutdown signal received. Exiting...");
    Logger::log("MAIN", "Total chunks processed: " + to_string(chunk_num));
    Logger::log("MAIN", "Goodbye!");

    return 0;
}