#pragma once

#include "../config/config.hpp"
#include "../core/tensor.hpp"
#include "../model/nol_model.hpp"
#include "../token/tokenizer.hpp"
#include <string>
#include <vector>
#include <random>

namespace nol {
namespace inference {

using namespace nol::core;
using namespace nol::config;
using namespace nol::model;
using namespace nol::token;

/**
 * @brief Sampling parameters untuk text generation.
 */
struct SamplingParams {
    int top_k = 50;
    float top_p = 0.9f;
    float temperature = 1.0f;
    int max_tokens = 256;
    uint64_t seed = 42;
};

/**
 * @brief Context Proficiency metrics.
 */
struct ProficiencyMetrics {
    float perplexity = 0.0f;
    float topic_mixing_rate = 0.0f;
    float hierarchy_correlation = 0.0f;
    float overall_score = 0.0f;
};

/**
 * @brief Inference Engine untuk Model Nol.
 */
class InferenceEngine {
public:
    NolModel& model;
    NolConfig cfg;
    std::mt19937 rng;

    InferenceEngine(NolModel& mdl, const NolConfig& config);

    std::vector<int> generate(const std::vector<int>& prompt,
                              const SamplingParams& params);

    int sample_token(const float* logits, int vocab_size,
                     const SamplingParams& params);

    ProficiencyMetrics compute_proficiency(const std::vector<int>& generated_tokens,
                                           const std::vector<int>& prompt_tokens);

    float compute_perplexity(const std::vector<int>& tokens);

    /**
     * @brief Generate text dari string prompt (encode → generate → decode).
     * @param prompt_text Input text
     * @param tok Tokenizer
     * @param params Sampling parameters
     * @return Generated text string
     */
    std::string generate_text(const std::string& prompt_text,
                              BPETokenizer& tok,
                              const SamplingParams& params);
};

} // namespace inference
} // namespace nol
