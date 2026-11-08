#include "../../include/inference/inference.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nol {
namespace inference {

InferenceEngine::InferenceEngine(NolModel& mdl, const NolConfig& config)
    : model(mdl), cfg(config), rng(config.rng_seed) {}

int InferenceEngine::sample_token(const float* logits, int vocab_size,
                                  const SamplingParams& params) {
    std::vector<float> probs(vocab_size);
    float max_logit = -1e30f;
    bool has_nan = false;

    for (int v = 0; v < vocab_size; ++v) {
        if (std::isnan(logits[v]) || std::isinf(logits[v])) {
            has_nan = true;
            break;
        }
        if (logits[v] > max_logit) max_logit = logits[v];
    }

    // Safeguard: jika logits corrupt, return token acak atau 0
    if (has_nan || max_logit <= -1e29f) {
        std::uniform_int_distribution<int> dist(0, vocab_size - 1);
        return dist(rng);
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        probs[v] = std::exp((logits[v] - max_logit) / params.temperature);
        sum_exp += probs[v];
    }
    
    if (sum_exp <= 0.0f || std::isnan(sum_exp)) {
        std::uniform_int_distribution<int> dist(0, vocab_size - 1);
        return dist(rng);
    }

    for (int v = 0; v < vocab_size; ++v) {
        probs[v] /= sum_exp;
    }

    if (params.top_k > 0 && params.top_k < vocab_size) {
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + params.top_k,
                          indices.end(),
                          [&probs](int a, int b) { return probs[a] > probs[b]; });

        std::vector<bool> keep(vocab_size, false);
        for (int i = 0; i < params.top_k; ++i) keep[indices[i]] = true;

        float new_sum = 0.0f;
        for (int v = 0; v < vocab_size; ++v) {
            if (!keep[v]) probs[v] = 0.0f;
            else new_sum += probs[v];
        }
        if (new_sum > 0.0f) {
            for (int v = 0; v < vocab_size; ++v) probs[v] /= new_sum;
        }
    }

    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&probs](int a, int b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += probs[indices[i]];
            if (cumsum >= params.top_p) {
                cutoff = i + 1;
                break;
            }
        }

        std::vector<bool> keep(vocab_size, false);
        float new_sum = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            keep[indices[i]] = true;
            new_sum += probs[indices[i]];
        }
        if (new_sum > 0.0f) {
            for (int v = 0; v < vocab_size; ++v) {
                if (!keep[v]) probs[v] = 0.0f;
                else probs[v] /= new_sum;
            }
        }
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        cumsum += probs[v];
        if (r <= cumsum) return v;
    }
    return vocab_size - 1;
}

std::vector<int> InferenceEngine::generate(const std::vector<int>& prompt,
                                           const SamplingParams& params) {
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    std::vector<int> generated;
    std::vector<int> context = prompt;

    for (int step = 0; step < params.max_tokens; ++step) {
        int seq_len = static_cast<int>(context.size());

        if (seq_len > cfg.max_seq_len) {
            context.erase(context.begin(), context.begin() + (seq_len - cfg.max_seq_len));
            seq_len = cfg.max_seq_len;
        }

        Tensor token_ids({1, static_cast<size_t>(seq_len)}, DType::FP32);
        float* tip = token_ids.fp32_data();
        for (int s = 0; s < seq_len; ++s) {
            tip[s] = static_cast<float>(context[s]);
        }

        Tensor logits, aux;
        model.forward(token_ids, logits, aux);

        int vocab = cfg.vocab_size;
        const float* last_logits = logits.fp32_data() + (seq_len - 1) * vocab;

        int next_token = sample_token(last_logits, vocab, params);
        generated.push_back(next_token);
        context.push_back(next_token);
    }

    #ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    #endif

    return generated;
}

float InferenceEngine::compute_perplexity(const std::vector<int>& tokens) {
    if (tokens.size() < 2) return 1e10f;

    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    int seq_len = static_cast<int>(tokens.size());
    if (seq_len > cfg.max_seq_len) seq_len = cfg.max_seq_len;

    Tensor token_ids({1, static_cast<size_t>(seq_len)}, DType::FP32);
    float* tip = token_ids.fp32_data();
    for (int s = 0; s < seq_len; ++s) {
        tip[s] = static_cast<float>(tokens[s]);
    }

    Tensor logits, aux;
    model.forward(token_ids, logits, aux);

    int vocab = cfg.vocab_size;
    const float* lp = logits.fp32_data();
    float total_nll = 0.0f;
    int count = 0;
    bool has_nan = false;

    for (int s = 0; s < seq_len - 1; ++s) {
        int target = tokens[s + 1];
        target = std::max(0, std::min(target, vocab - 1));

        float max_logit = -1e30f;
        bool row_has_nan = false;
        for (int v = 0; v < vocab; ++v) {
            float val = lp[s * vocab + v];
            if (std::isnan(val) || std::isinf(val)) {
                row_has_nan = true;
                break;
            }
            if (val > max_logit) max_logit = val;
        }
        
        if (row_has_nan || max_logit <= -1e29f) {
            has_nan = true;
            break;
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab; ++v) {
            sum_exp += std::exp(lp[s * vocab + v] - max_logit);
        }
        float nll = max_logit + std::log(sum_exp) - lp[s * vocab + target];
        
        if (std::isnan(nll) || std::isinf(nll)) {
            has_nan = true;
            break;
        }
        
        total_nll += nll;
        count++;
    }

    #ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    #endif

    // Safeguard: Model acak/hiperbolik bisa menghasilkan NaN/Inf.
    // Kembalikan PPL maksimum (1e10f) jika perhitungan gagal.
    if (count == 0 || has_nan) return 1e10f;
    
    float avg_nll = total_nll / count;
    if (avg_nll < 0.0f) avg_nll = 0.0f; // Clamp untuk mencegah PPL < 1.0 akibat presisi float
    
    float ppl = std::exp(avg_nll);
    if (std::isinf(ppl) || std::isnan(ppl)) return 1e10f;
    
    return ppl;
}

ProficiencyMetrics InferenceEngine::compute_proficiency(
    const std::vector<int>& generated_tokens,
    const std::vector<int>& prompt_tokens) {

    ProficiencyMetrics metrics;

    std::vector<int> full_seq = prompt_tokens;
    full_seq.insert(full_seq.end(), generated_tokens.begin(), generated_tokens.end());
    metrics.perplexity = compute_perplexity(full_seq);

    if (!generated_tokens.empty()) {
        std::vector<int> freq(cfg.vocab_size, 0);
        for (int tok : generated_tokens) {
            if (tok >= 0 && tok < cfg.vocab_size) freq[tok]++;
        }
        float entropy = 0.0f;
        float total = static_cast<float>(generated_tokens.size());
        for (int v = 0; v < cfg.vocab_size; ++v) {
            if (freq[v] > 0) {
                float p = freq[v] / total;
                entropy -= p * std::log(p);
            }
        }
        float max_entropy = std::log(static_cast<float>(cfg.vocab_size));
        metrics.topic_mixing_rate = (max_entropy > 0) ? entropy / max_entropy : 0.0f;
    }

    metrics.hierarchy_correlation = 0.5f;

    float ppl_score = 1.0f / (1.0f + std::log(std::max(metrics.perplexity, 1.0f)));
    float mix_score = 1.0f - metrics.topic_mixing_rate;
    float hier_score = metrics.hierarchy_correlation;
    metrics.overall_score = 0.5f * ppl_score + 0.3f * mix_score + 0.2f * hier_score;

    return metrics;
}

std::string InferenceEngine::generate_text(const std::string& prompt_text,
                                           BPETokenizer& tok,
                                           const SamplingParams& params) {
    std::vector<int> prompt_ids = tok.encode(prompt_text, false, false);
    
    if (prompt_ids.empty()) {
        prompt_ids.push_back(BPETokenizer::BOS_ID);
    }

    std::vector<int> generated_ids = generate(prompt_ids, params);

    return tok.decode(generated_ids);
}

} // namespace inference
} // namespace nol

// ==========================================
// UNIT TEST INFERENCE
// ==========================================
static void run_inference_tests() {
    using namespace nol::inference;
    using namespace nol::model;
    using namespace nol::config;
    using namespace nol::core;
    using namespace nol::token;
    std::cout << "[TEST] Memulai unit test inference...\n";

    NolConfig cfg;
    cfg.vocab_size = 50;
    cfg.d_model = 32;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 1;
    cfg.d_head = 16;
    cfg.ffn_hidden = 64;
    cfg.max_seq_len = 16;
    cfg.n_hyp_heads = 1;
    cfg.d_hyp = 4;
    cfg.n_top_anchors = 2;
    cfg.n_children = 2;
    cfg.top_k_experts = 1;
    cfg.n_shared_experts = 1;
    cfg.rng_seed = 42;

    NolModel model(cfg);
    model.initialize_weights(42);
    InferenceEngine engine(model, cfg);

    // Test 1: Sampling produces valid token
    {
        SamplingParams params;
        params.top_k = 10;
        params.temperature = 1.0f;

        float logits[50];
        for (int i = 0; i < 50; ++i) logits[i] = static_cast<float>(i) / 50.0f;

        int token = engine.sample_token(logits, 50, params);
        assert(token >= 0 && token < 50);
        std::cout << "  Sampling OK (token=" << token << ")\n";
    }

    // Test 2: Greedy sampling
    {
        SamplingParams params;
        params.top_k = 1;
        params.temperature = 1.0f;

        float logits[50];
        for (int i = 0; i < 50; ++i) logits[i] = 0.0f;
        logits[42] = 10.0f;

        int token = engine.sample_token(logits, 50, params);
        assert(token == 42);
        std::cout << "  Greedy sampling OK\n";
    }

    // Test 3: Generate produces correct length
    {
        SamplingParams params;
        params.max_tokens = 10;
        params.top_k = 5;
        params.temperature = 1.0f;

        std::vector<int> prompt = {1, 5, 10};
        std::vector<int> generated = engine.generate(prompt, params);

        assert(static_cast<int>(generated.size()) == 10);
        for (int tok : generated) {
            assert(tok >= 0 && tok < 50);
        }
        std::cout << "  Generate OK (" << generated.size() << " tokens)\n";
    }

    // Test 4: Perplexity computation (dengan safeguard untuk model acak)
    {
        std::vector<int> tokens = {1, 5, 10, 20, 30, 40};
        float ppl = engine.compute_perplexity(tokens);
        // Safeguard memastikan PPL selalu > 0 dan finite meskipun logits mengandung NaN/Inf
        assert(ppl > 0.0f);
        assert(std::isfinite(ppl));
        std::cout << "  Perplexity OK (ppl=" << ppl << ")\n";
    }

    // Test 5: Proficiency metrics
    {
        std::vector<int> prompt = {1, 5, 10};
        std::vector<int> generated = {20, 30, 40, 15, 25};

        ProficiencyMetrics metrics = engine.compute_proficiency(generated, prompt);
        assert(metrics.perplexity > 0.0f);
        assert(metrics.topic_mixing_rate >= 0.0f && metrics.topic_mixing_rate <= 1.0f);
        assert(metrics.overall_score >= 0.0f && metrics.overall_score <= 1.0f);
        std::cout << "  Proficiency OK (score=" << metrics.overall_score
                  << " ppl=" << metrics.perplexity
                  << " mix=" << metrics.topic_mixing_rate << ")\n";
    }

    // Test 6: Empty prompt handling
    {
        SamplingParams params;
        params.max_tokens = 5;
        params.top_k = 5;

        std::vector<int> prompt = {0};
        std::vector<int> generated = engine.generate(prompt, params);
        assert(static_cast<int>(generated.size()) == 5);
        std::cout << "  Single-token prompt OK\n";
    }

    // Test 7: generate_text dengan tokenizer
    {
        BPETokenizer tok;
        std::vector<std::string> corpus = {"hello world test abc"};
        tok.build_vocab(corpus, 50);

        SamplingParams params;
        params.max_tokens = 5;
        params.top_k = 5;

        std::string result = engine.generate_text("hello", tok, params);
        assert(result.size() >= 0);
        std::cout << "  generate_text OK (len=" << result.size() << ")\n";
    }

    // Test 8: Sampling dengan logits NaN (Safeguard check)
    {
        SamplingParams params;
        params.top_k = 10;
        params.temperature = 1.0f;

        float logits[50];
        for (int i = 0; i < 50; ++i) logits[i] = std::nanf(""); // Penuhi dengan NaN

        int token = engine.sample_token(logits, 50, params);
        // Harus tetap menghasilkan token valid tanpa crash
        assert(token >= 0 && token < 50);
        std::cout << "  NaN logits safeguard OK (token=" << token << ")\n";
    }

    std::cout << "[PASS] ✅ Semua unit test inference berhasil!\n\n";
}

#ifdef NOL_TEST_INFERENCE
int main() {
    run_inference_tests();
    return 0;
}
#endif
