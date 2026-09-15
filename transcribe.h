#pragma once
#include <string>

using namespace std;

// Persistent Whisper engine: loads the model once and reuses it across chunks.
class WhisperEngine {
public:
    ~WhisperEngine();
    bool init(const string& model_path);
    bool initialized() const { return ctx_ != nullptr; }
    string transcribe(const string& audio_path);

private:
    struct whisper_context* ctx_ = nullptr;
    int n_threads_ = 1;
};