#pragma once
#include <string>

using namespace std;

// Persistent Llama corrector: loads the model/context once and reuses them.
class LlamaCorrector {
public:
    ~LlamaCorrector();
    bool init(const string& model_path);
    bool initialized() const { return model_ != nullptr && ctx_ != nullptr; }
    string correct(const string& raw_text);

private:
    struct llama_model* model_ = nullptr;
    struct llama_context* ctx_ = nullptr;
    const struct llama_vocab* vocab_ = nullptr;
    struct llama_sampler* sampler_ = nullptr;
    bool backend_inited_ = false;
    int n_threads_ = 1;
};