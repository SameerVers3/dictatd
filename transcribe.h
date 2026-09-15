#pragma once
#include <string>
#include <vector>

using namespace std;

// A transcribed text span, times relative to the start of the audio window.
struct WhisperSegment {
    string text;
    double t0;
    double t1;
};

// Persistent Whisper engine: loads the model once and reuses it across chunks.
class WhisperEngine {
public:
    ~WhisperEngine();
    bool init(const string& model_path);
    bool initialized() const { return ctx_ != nullptr; }
    // Transcribe raw 16 kHz audio. context_text carries previously committed
    // words into the decoder so consecutive sliding windows stay coherent.
    vector<WhisperSegment> transcribe(const vector<float>& samples,
                                      int sample_rate,
                                      const string& context_text);

private:
    struct whisper_context* ctx_ = nullptr;
    int n_threads_ = 1;
};