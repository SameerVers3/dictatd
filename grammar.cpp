#include "grammar.h"
#include "timer.h"
#include "logger.h"
#include "llama.h"
#include <thread>
#include <algorithm>
#include <vector>
#include <chrono>

using namespace std;

LlamaCorrector::~LlamaCorrector() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    if (backend_inited_) {
        llama_backend_free();
        backend_inited_ = false;
    }
}

bool LlamaCorrector::init(const string& model_path) {
    Logger::log("LLAMA", "Initializing backend...");
    llama_backend_init();
    backend_inited_ = true;

    Logger::log("LLAMA", "Loading model (once): " + model_path);

    llama_model_params model_params = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!model_) {
        Logger::log("LLAMA", "ERROR: Failed to load llama model");
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    n_threads_ = max(1, static_cast<int>(thread::hardware_concurrency()));
    ctx_params.n_threads = n_threads_;
    ctx_params.n_threads_batch = n_threads_;

    Logger::log("LLAMA", "Creating context (n_ctx=4096, n_threads="
                + to_string(n_threads_) + ")...");

    ctx_ = llama_init_from_model(model_, ctx_params);

    if (!ctx_) {
        Logger::log("LLAMA", "ERROR: Failed to create context");
        return false;
    }

    vocab_ = llama_model_get_vocab(model_);

    sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(0.95f, 1));
    // Penalize repeated tokens (applied after top-k/top-p so it stays fast).
    // Small low-quant models fall into "Hello. Hello. Hello. ..." echo loops;
    // a moderate repeat penalty forces them to move on and hit EOG instead.
    llama_sampler_chain_add(sampler_, llama_sampler_init_penalties(64, 1.30f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(0.3f));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(12345));

    Logger::log("LLAMA", "Model ready.");
    return true;
}

string LlamaCorrector::correct(const string& raw_text) {
    Timer timer("LLAMA");

    string prompt =
        "Correct the grammar and spelling of the following text. "
        "Only output the corrected text, nothing else.\n\n"
        "Text: " + raw_text + "\n\n"
        "Corrected: ";

    int n_prompt_est = static_cast<int>(prompt.length()) + 16;

    vector<llama_token> prompt_tokens(n_prompt_est);

    int n_prompt = llama_tokenize(vocab_, prompt.c_str(),
                                  static_cast<int>(prompt.length()),
                                  prompt_tokens.data(), n_prompt_est,
                                  true, false);
    if (n_prompt < 0) {
        Logger::log("LLAMA", "ERROR: Tokenization failed");
        return "";
    }
    if (n_prompt > n_prompt_est) {
        prompt_tokens.resize(n_prompt);
        n_prompt = llama_tokenize(vocab_, prompt.c_str(),
                                  static_cast<int>(prompt.length()),
                                  prompt_tokens.data(), n_prompt,
                                  true, false);
    }
    prompt_tokens.resize(n_prompt);

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);

    for (int i = 0; i < n_prompt; i++) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }

    batch.logits[n_prompt - 1] = 1;
    batch.n_tokens = n_prompt;

    auto t0 = chrono::high_resolution_clock::now();

    if (llama_decode(ctx_, batch) != 0) {
        Logger::log("LLAMA", "ERROR: llama_decode() failed on prompt");
        llama_batch_free(batch);
        return "";
    }

    auto t1 = chrono::high_resolution_clock::now();
    Logger::log("LLAMA", "Prompt evaluated (" + to_string(n_prompt)
                + " tokens) in "
                + Timer::format_time(chrono::duration<double>(t1 - t0).count()));

    llama_sampler_reset(sampler_);

    string response;
    const int max_tokens = 128;
    int n_gen = 0;
    llama_pos n_pos = static_cast<llama_pos>(n_prompt);

    auto t2 = chrono::high_resolution_clock::now();

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(sampler_, ctx_, -1);

        if (llama_vocab_is_eog(vocab_, new_token)) {
            Logger::log("LLAMA", "End-of-generation token reached");
            break;
        }

        char piece[256];
        int n = llama_token_to_piece(vocab_, new_token, piece, sizeof(piece), 0, true);

        if (n > 0) {
            response.append(piece, n);
        }

        // Corrections are a single line: stop at the first newline.
        if (response.find('\n') != string::npos) {
            Logger::log("LLAMA", "Stopping at newline");
            break;
        }

        // Backstop against echo loops: if the tail of the response already
        // appeared earlier, the model is stuck repeating itself.
        if ((int) response.size() >= 48) {
            const size_t win = 24;
            size_t prev =
                response.rfind(response.substr(response.size() - win),
                               response.size() - win - 1);
            if (prev != string::npos) {
                Logger::log("LLAMA", "Stopping at repeated text");
                break;
            }
        }

        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(ctx_, batch) != 0) {
            Logger::log("LLAMA", "ERROR: llama_decode() failed during generation");
            break;
        }

        n_pos++;
        n_gen++;
    }

    auto t3 = chrono::high_resolution_clock::now();
    Logger::log("LLAMA", "Generated " + to_string(n_gen) + " tokens in "
                + Timer::format_time(chrono::duration<double>(t3 - t2).count()));

    llama_batch_free(batch);

    response.erase(0, response.find_first_not_of(" \t\n\r"));
    response.erase(response.find_last_not_of(" \t\n\r") + 1);

    Logger::log("LLAMA", "Corrected text: \"" + response + "\"");
    return response;
}