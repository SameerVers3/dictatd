#include "grammar.h"
#include "timer.h"
#include "logger.h"
#include "llama.h"
#include <thread>
#include <algorithm>
#include <vector>

using namespace std;

string correct_grammar_with_llama(const string& raw_text, const string& model_path) {
    Timer timer("LLAMA");
    Logger::log("LLAMA", "Initializing backend...");
    llama_backend_init();
    Logger::log("LLAMA", "Loading model: " + model_path);
    
    auto t0 = chrono::high_resolution_clock::now();
    
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    
    if (!model) {
        Logger::log("LLAMA", "ERROR: Failed to load llama model");
        llama_backend_free();
        return "";
    }
    
    auto t1 = chrono::high_resolution_clock::now();
    
    Logger::log("LLAMA", "Model loaded in "
                + Timer::format_time(chrono::duration<double>(t1 - t0).count()));
    
    llama_context_params ctx_params = llama_context_default_params();
    
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = max(1, static_cast<int>(thread::hardware_concurrency()));
    ctx_params.n_threads_batch = ctx_params.n_threads;
    
    Logger::log("LLAMA", "Creating context (n_ctx=" + to_string(ctx_params.n_ctx)
                + ", n_threads=" + to_string(ctx_params.n_threads) + ")...");
    
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    
    if (!ctx) {
        Logger::log("LLAMA", "ERROR: Failed to create context");
        llama_model_free(model);
        llama_backend_free();
        return "";
    }
    

    const llama_vocab* vocab = llama_model_get_vocab(model);
    
    string prompt =
        "Correct the grammar and spelling of the following text. "
        "Only output the corrected text, nothing else.\n\n"
        "Text: " + raw_text + "\n\n"
        "Corrected: ";
    
    Logger::log("LLAMA", "Prompt length: " + to_string(prompt.length()) + " chars");
    
    int n_prompt_est = static_cast<int>(prompt.length()) + 16;
    
    vector<llama_token> prompt_tokens(n_prompt_est);
    
    int n_prompt = llama_tokenize(vocab, prompt.c_str(),
                                  static_cast<int>(prompt.length()),
                                  prompt_tokens.data(), n_prompt_est,
                                  true, false);
    if (n_prompt < 0) {
        Logger::log("LLAMA", "ERROR: Tokenization failed (first pass)");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return "";
    }
    if (n_prompt > n_prompt_est) {
        prompt_tokens.resize(n_prompt);
        n_prompt = llama_tokenize(vocab, prompt.c_str(),
                                  static_cast<int>(prompt.length()),
                                  prompt_tokens.data(), n_prompt,
                                  true, false);
    }
    prompt_tokens.resize(n_prompt);
    
    Logger::log("LLAMA", "Prompt tokens: " + to_string(n_prompt));
    
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.3f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(12345));
    
    llama_batch batch = llama_batch_init(static_cast<int>(prompt_tokens.size()), 0, 1);
    
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    
    batch.logits[prompt_tokens.size() - 1] = 1;
    
    batch.n_tokens = static_cast<int>(prompt_tokens.size());
    
    Logger::log("LLAMA", "Evaluating prompt...");
    
    auto t2 = chrono::high_resolution_clock::now();
    
    if (llama_decode(ctx, batch) != 0) {
        Logger::log("LLAMA", "ERROR: llama_decode() failed on prompt");
        llama_batch_free(batch);
        llama_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return "";
    }
    
    auto t3 = chrono::high_resolution_clock::now();
    
    Logger::log("LLAMA", "Prompt evaluated in "
                + Timer::format_time(chrono::duration<double>(t3 - t2).count()));
    Logger::log("LLAMA", "Generating corrected text...");
    
    auto t4 = chrono::high_resolution_clock::now();
    
    string response;
    
    const int max_tokens = 256;
    
    int n_gen = 0;
    
    llama_pos n_pos = static_cast<llama_pos>(prompt_tokens.size());
    
    for (int i = 0; i < max_tokens; i++) {
        
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        

        if (llama_vocab_is_eog(vocab, new_token)) {
            Logger::log("LLAMA", "End-of-generation token reached");
            break;
        }
        
        char piece[256];
        
        int n = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
        
        if (n > 0) {
            response.append(piece, n);
        }
        
        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = n_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        
        if (llama_decode(ctx, batch) != 0) {
            Logger::log("LLAMA", "ERROR: llama_decode() failed during generation");
            break;
        }
        
        n_pos++;
        n_gen++;

        if (n_gen > 10 && response.find('\n') != string::npos) {
            Logger::log("LLAMA", "Stopping at newline");
            break;
        }
    }
    
    auto t5 = chrono::high_resolution_clock::now();
    
    Logger::log("LLAMA", "Generated " + to_string(n_gen) + " tokens in "
                + Timer::format_time(chrono::duration<double>(t5 - t4).count()));
    
    response.erase(0, response.find_first_not_of(" \t\n\r"));
    response.erase(response.find_last_not_of(" \t\n\r") + 1);
    
    Logger::log("LLAMA", "Corrected text: \"" + response + "\"");
    
    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    
    llama_backend_free();
    
    return response;
}
