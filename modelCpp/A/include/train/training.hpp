#pragma once

#include "../config/config.hpp"
#include "../core/tensor.hpp"
#include "../model/nol_model.hpp"
#include "../io/io.hpp"
#include "../token/tokenizer.hpp"
#include <string>
#include <vector>
#include <functional>

namespace nol {
namespace train {

using namespace nol::core;
using namespace nol::config;
using namespace nol::model;
using namespace nol::io;
using namespace nol::token;

/**
 * @brief Status training yang ditampilkan di Playground Lobby.
 */
struct TrainingStatus {
    int remaining_epochs = 0;
    int completed_steps = 0;
    float last_loss = 0.0f;
    float last_acc = 0.0f;
    float last_ppl = 0.0f;
    float context_proficiency = 0.0f;
};

/**
 * @brief Dataset untuk training/testing.
 * Setiap sample adalah sequence token IDs dengan panjang tetap.
 */
struct Dataset {
    std::vector<std::vector<int>> samples; // [num_samples][seq_len]
    std::vector<std::string> raw_texts;    // Teks asli (opsional, untuk debugging)
    int seq_len = 0;

    void generate_random(int num_samples, int seq_length, int vocab_size, uint64_t seed);
    
    /**
     * @brief Load dataset dari file teks.
     * Setiap baris = satu sample. Di-tokenize via BPETokenizer.
     * @param path Path ke file teks
     * @param tok Tokenizer yang sudah di-build vocab-nya
     * @param max_seq_len Panjang maksimum sequence (potong/pad)
     * @return Jumlah sample yang berhasil di-load
     */
    int load_from_text(const std::string& path, BPETokenizer& tok, int max_seq_len);

    /**
     * @brief Load dataset dari vector strings.
     * @param texts Vector strings
     * @param tok Tokenizer
     * @param max_seq_len Panjang maksimum sequence
     * @return Jumlah sample
     */
    int load_from_texts(const std::vector<std::string>& texts, BPETokenizer& tok, int max_seq_len);

    size_t size() const { return samples.size(); }
};

/**
 * @brief Trainer utama Model Nol.
 */
class Trainer {
public:
    NolModel& model;
    NolConfig cfg;
    TrainingStatus status;

    std::vector<float> adam_m;
    std::vector<float> adam_v;
    int adam_t = 0;

    Trainer(NolModel& mdl, const NolConfig& config);

    void init_optimizer();
    float train_epoch(Dataset& dataset);
    void train(Dataset& dataset, const std::string& model_name, int start_epoch = 0);

    float compute_loss(const Tensor& logits, const Tensor& targets,
                       const Tensor& aux_loss, int step);

    void compute_gradients(Dataset& dataset, int batch_start, int batch_end,
                           std::vector<float>& grads);

    void optimizer_step(const std::vector<float>& grads);
    float get_lr(int step) const;
    void self_evolution(Dataset& dataset);
    void evaluate(Dataset& dataset, float& acc, float& ppl);
    void flatten_params(std::vector<float>& out) const;
    void unflatten_params(const std::vector<float>& in);
    size_t total_params() const;
};

} // namespace train
} // namespace nol
