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
    Logger::log("MAIN", "Press Ctrl+C to stop");
    Logger::log("MAIN", "========================================");
    
    int chunk_num = 0;
    
    while (g_running) {
        chunk_num++;
        cout << "\n";
        
        Logger::log("MAIN", "========== CHUNK #" + to_string(chunk_num) + " ==========");
        
        string wav_path = "/tmp/voice_chunk_" + to_string(chunk_num) + ".wav";
        
        bool record_ok = false;
        
        {
            Timer record_timer("RECORD");
            Logger::log("RECORD", "Recording " + to_string(chunk_seconds)
                        + " second audio chunk...");
            record_ok = record_audio_chunk(wav_path, chunk_seconds);
        }
        
        if (!record_ok) {
            Logger::log("RECORD", "ERROR: Recording failed. Make sure 'arecord' or 'ffmpeg' is installed.");
            Logger::log("MAIN", "Hint: sudo apt install alsa-utils   # for arecord");
            Logger::log("MAIN", "      sudo apt install ffmpeg       # for ffmpeg");
            break;
        }
        
        string raw_text = transcribe_with_whisper(wav_path, whisper_model);
        
        if (raw_text.empty()) {
            Logger::log("MAIN", "Transcription failed, skipping grammar correction");
            continue;
        }

        
        string corrected_text = correct_grammar_with_llama(raw_text, llama_model);
        
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

    cout << "\n";
    
    Logger::log("MAIN", "Shutdown signal received. Exiting...");
    Logger::log("MAIN", "Total chunks processed: " + to_string(chunk_num));
    Logger::log("MAIN", "Goodbye!");
    
    return 0;
}
