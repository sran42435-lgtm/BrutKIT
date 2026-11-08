#include "../../include/train/training.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nol {
namespace train {

// ==========================================
// DATASET
// ==========================================

void Dataset::generate_random(int num_samples, int seq_length, int vocab_size, uint64_t seed) {
    seq_len = seq_length;
    samples.resize(num_samples);
    raw_texts.clear();
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> dist(0, vocab_size - 1);

    for (int i = 0; i < num_samples; ++i) {
        samples[i].resize(seq_length);
        for (int j = 0; j < seq_length; ++j) {
            samples[i][j] = dist(rng);
        }
    }
}

int Dataset::load_from_text(const std::string& path, BPETokenizer& tok, int max_seq_len) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Dataset] Cannot open file: " << path << "\n";
        return 0;
    }

    std::vector<std::string> texts;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            texts.push_back(line);
        }
    }
    file.close();

    if (texts.empty()) {
        std::cerr << "[Dataset] File empty: " << path << "\n";
        return 0;
    }

    return load_from_texts(texts, tok, max_seq_len);
}

int Dataset::load_from_texts(const std::vector<std::string>& texts, BPETokenizer& tok, int max_seq_len) {
    seq_len = max_seq_len;
    samples.clear();
    raw_texts.clear();

    for (const auto& text : texts) {
        std::vector<int> ids = tok.encode(text, false, false);
        if (ids.empty()) continue;

        // Pad atau truncate ke max_seq_len
        std::vector<int> sample(max_seq_len, BPETokenizer::PAD_ID);
        int copy_len = std::min(static_cast<int>(ids.size()), max_seq_len);
        for (int i = 0; i < copy_len; ++i) {
            sample[i] = ids[i];
        }

        samples.push_back(sample);
        raw_texts.push_back(text);
    }

    std::cout << "[Dataset] Loaded " << samples.size() << " samples (seq_len=" << max_seq_len << ")\n";
    return static_cast<int>(samples.size());
}

// ==========================================
// TRAINER CONSTRUCTOR
// ==========================================

Trainer::Trainer(NolModel& mdl, const NolConfig& config)
    : model(mdl), cfg(config) {
    status.remaining_epochs = cfg.epochs;
    status.completed_steps = 0;
    status.last_loss = 0.0f;
    status.last_acc = 0.0f;
    status.last_ppl = 0.0f;
    status.context_proficiency = 0.0f;
}

size_t Trainer::total_params() const {
    return model.param_count();
}

void Trainer::flatten_params(std::vector<float>& out) const {
    out.resize(total_params());
    size_t offset = 0;

    auto copy_tensor = [&](const Tensor& t) {
        const float* p = t.fp32_data();
        std::memcpy(out.data() + offset, p, t.numel * sizeof(float));
        offset += t.numel;
    };

    copy_tensor(model.embed_E);
    copy_tensor(model.embed_T);

    for (const auto& bp : model.blocks) {
        copy_tensor(bp.rmsnorm1_gamma);
        copy_tensor(bp.W_q); copy_tensor(bp.W_k);
        copy_tensor(bp.W_v); copy_tensor(bp.W_o);
        copy_tensor(bp.W_q_hyp); copy_tensor(bp.W_k_hyp); copy_tensor(bp.W_v_hyp);
        copy_tensor(bp.W_g); copy_tensor(bp.W_u);
        copy_tensor(bp.W_z); copy_tensor(bp.W_c);
        copy_tensor(bp.anchors_l1); copy_tensor(bp.anchors_l2);
        copy_tensor(bp.W_r);
        copy_tensor(bp.rmsnorm2_gamma);
        copy_tensor(bp.W1_shared); copy_tensor(bp.W2_shared); copy_tensor(bp.W3_shared);
        for (int e = 0; e < cfg.n_top_anchors; ++e) {
            copy_tensor(bp.W1_expert[e]);
            copy_tensor(bp.W2_expert[e]);
            copy_tensor(bp.W3_expert[e]);
        }
    }

    copy_tensor(model.rmsnorm_final_gamma);
    copy_tensor(model.H_v);

    assert(offset == total_params());
}

void Trainer::unflatten_params(const std::vector<float>& in) {
    size_t offset = 0;

    auto paste_tensor = [&](Tensor& t) {
        float* p = t.fp32_data();
        std::memcpy(p, in.data() + offset, t.numel * sizeof(float));
        offset += t.numel;
    };

    paste_tensor(model.embed_E);
    paste_tensor(model.embed_T);

    for (auto& bp : model.blocks) {
        paste_tensor(bp.rmsnorm1_gamma);
        paste_tensor(bp.W_q); paste_tensor(bp.W_k);
        paste_tensor(bp.W_v); paste_tensor(bp.W_o);
        paste_tensor(bp.W_q_hyp); paste_tensor(bp.W_k_hyp); paste_tensor(bp.W_v_hyp);
        paste_tensor(bp.W_g); paste_tensor(bp.W_u);
        paste_tensor(bp.W_z); paste_tensor(bp.W_c);
        paste_tensor(bp.anchors_l1); paste_tensor(bp.anchors_l2);
        paste_tensor(bp.W_r);
        paste_tensor(bp.rmsnorm2_gamma);
        paste_tensor(bp.W1_shared); paste_tensor(bp.W2_shared); paste_tensor(bp.W3_shared);
        for (int e = 0; e < cfg.n_top_anchors; ++e) {
            paste_tensor(bp.W1_expert[e]);
            paste_tensor(bp.W2_expert[e]);
            paste_tensor(bp.W3_expert[e]);
        }
    }

    paste_tensor(model.rmsnorm_final_gamma);
    paste_tensor(model.H_v);

    model.rebuild_tied_embedding();
}

// ==========================================
// OPTIMIZER
// ==========================================

void Trainer::init_optimizer() {
    size_t n = total_params();
    adam_m.assign(n, 0.0f);
    adam_v.assign(n, 0.0f);
    adam_t = 0;
}

float Trainer::get_lr(int step) const {
    float base_lr = cfg.lr;
    int warmup = cfg.warmup_steps;

    if (step < warmup) {
        return base_lr * static_cast<float>(step) / static_cast<float>(warmup);
    }

    int total = cfg.epochs * 1000;
    int progress = step - warmup;
    int remaining = std::max(1, total - warmup);
    float cosine = 0.5f * (1.0f + std::cos(M_PI * static_cast<float>(progress) / static_cast<float>(remaining)));
    return base_lr * cosine;
}

void Trainer::optimizer_step(const std::vector<float>& grads) {
    adam_t++;
    float lr = get_lr(adam_t);
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;

    float bias_corr1 = 1.0f - std::pow(beta1, adam_t);
    float bias_corr2 = 1.0f - std::pow(beta2, adam_t);

    size_t n = total_params();

    std::vector<float> params;
    flatten_params(params);

    for (size_t i = 0; i < n; ++i) {
        float g = grads[i];

        if (g > cfg.grad_clip) g = cfg.grad_clip;
        if (g < -cfg.grad_clip) g = -cfg.grad_clip;

        adam_m[i] = beta1 * adam_m[i] + (1.0f - beta1) * g;
        adam_v[i] = beta2 * adam_v[i] + (1.0f - beta2) * g * g;

        float m_hat = adam_m[i] / bias_corr1;
        float v_hat = adam_v[i] / bias_corr2;

        params[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * params[i]);
    }

    unflatten_params(params);
}

// ==========================================
// LOSS COMPUTATION
// ==========================================

float Trainer::compute_loss(const Tensor& logits, const Tensor& targets,
                            const Tensor& aux_loss, int step) {
    int total = static_cast<int>(logits.shape[0]);
    int vocab = static_cast<int>(logits.shape[1]);
    const float* lp = logits.fp32_data();
    const float* tp = targets.fp32_data();

    float lm_loss = 0.0f;
    for (int i = 0; i < total; ++i) {
        int target = static_cast<int>(tp[i]);
        target = std::max(0, std::min(target, vocab - 1));

        float max_logit = -1e30f;
        for (int v = 0; v < vocab; ++v) {
            if (lp[i * vocab + v] > max_logit) max_logit = lp[i * vocab + v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab; ++v) {
            sum_exp += std::exp(lp[i * vocab + v] - max_logit);
        }
        float log_sum_exp = max_logit + std::log(sum_exp);
        lm_loss += log_sum_exp - lp[i * vocab + target];
    }
    lm_loss /= static_cast<float>(total);

    int curriculum_steps = static_cast<int>(cfg.curriculum_ratio * cfg.epochs * 
                                             (cfg.checkpoint_steps > 0 ? cfg.checkpoint_steps : 100));
    
    float total_loss = cfg.w_lm * lm_loss;

    if (step >= curriculum_steps) {
        const float* alp = aux_loss.fp32_data();
        total_loss += cfg.w_iso * alp[0];
        total_loss += cfg.w_bal * alp[1];
    } else {
        const float* alp = aux_loss.fp32_data();
        total_loss += cfg.w_bal * alp[1];
    }

    return total_loss;
}

// ==========================================
// GRADIENT COMPUTATION (Placeholder)
// ==========================================

void Trainer::compute_gradients(Dataset& dataset, int batch_start, int batch_end,
                                std::vector<float>& grads) {
    size_t n = total_params();
    grads.assign(n, 0.0f);
}

// ==========================================
// EVALUATE
// ==========================================

void Trainer::evaluate(Dataset& dataset, float& acc, float& ppl) {
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    int num_correct = 0;
    int total_tokens = 0;
    float total_nll = 0.0f;
    int seq_len = dataset.seq_len;

    int eval_batch = std::min(4, static_cast<int>(dataset.size()));
    int num_batches = static_cast<int>(dataset.size()) / eval_batch;

    for (int b = 0; b < num_batches && b < 10; ++b) {
        Tensor token_ids({static_cast<size_t>(eval_batch), static_cast<size_t>(seq_len)}, DType::FP32);
        Tensor targets({static_cast<size_t>(eval_batch * seq_len)}, DType::FP32);

        float* tip = token_ids.fp32_data();
        float* tp = targets.fp32_data();
        for (int i = 0; i < eval_batch; ++i) {
            int idx = b * eval_batch + i;
            if (idx >= static_cast<int>(dataset.size())) break;
            for (int s = 0; s < seq_len; ++s) {
                tip[i * seq_len + s] = static_cast<float>(dataset.samples[idx][s]);
                int next_s = (s + 1) % seq_len;
                tp[i * seq_len + s] = static_cast<float>(dataset.samples[idx][next_s]);
            }
        }

        Tensor logits, aux;
        model.forward(token_ids, logits, aux);

        int vocab = cfg.vocab_size;
        const float* lp = logits.fp32_data();
        for (int i = 0; i < eval_batch * seq_len; ++i) {
            int target = static_cast<int>(tp[i]);
            target = std::max(0, std::min(target, vocab - 1));

            int pred = 0;
            float max_val = lp[i * vocab];
            for (int v = 1; v < vocab; ++v) {
                if (lp[i * vocab + v] > max_val) {
                    max_val = lp[i * vocab + v];
                    pred = v;
                }
            }
            if (pred == target) num_correct++;

            float max_logit = -1e30f;
            for (int v = 0; v < vocab; ++v) {
                if (lp[i * vocab + v] > max_logit) max_logit = lp[i * vocab + v];
            }
            float sum_exp = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                sum_exp += std::exp(lp[i * vocab + v] - max_logit);
            }
            float nll = max_logit + std::log(sum_exp) - lp[i * vocab + target];
            total_nll += nll;
            total_tokens++;
        }
    }

    acc = (total_tokens > 0) ? static_cast<float>(num_correct) / total_tokens : 0.0f;
    ppl = (total_tokens > 0) ? std::exp(total_nll / total_tokens) : 1e10f;

    #ifdef _OPENMP
    omp_set_num_threads(omp_get_max_threads());
    #endif
}

// ==========================================
// SELF-EVOLUTION
// ==========================================

void Trainer::self_evolution(Dataset& dataset) {
    float c = std::log(1.0f + std::exp(model.curvature)) + cfg.c_min;

    std::mt19937 rng(static_cast<uint32_t>(status.completed_steps));
    std::normal_distribution<float> normal(0.0f, 0.1f);

    for (auto& bp : model.blocks) {
        int dh1 = cfg.d_hyp + 1;
        float* a1p = bp.anchors_l1.fp32_data();

        for (int k = 0; k < cfg.n_top_anchors; ++k) {
            std::vector<float> v(cfg.d_hyp);
            for (int d = 0; d < cfg.d_hyp; ++d) v[d] = normal(rng);
            exp0(v.data(), a1p + k * dh1, dh1, c);
        }
    }

    std::cout << "  [Self-Evolution] Anchors updated at step " << status.completed_steps << "\n";
}

// ==========================================
// TRAIN EPOCH
// ==========================================

float Trainer::train_epoch(Dataset& dataset) {
    int num_samples = static_cast<int>(dataset.size());
    int bs = cfg.batch_size;
    int num_batches = (num_samples + bs - 1) / bs;
    float epoch_loss = 0.0f;

    for (int b = 0; b < num_batches; ++b) {
        int batch_start = b * bs;
        int batch_end = std::min(batch_start + bs, num_samples);

        std::vector<float> grads;
        compute_gradients(dataset, batch_start, batch_end, grads);

        optimizer_step(grads);

        status.completed_steps++;

        int actual_bs = batch_end - batch_start;
        int seq_len = dataset.seq_len;
        Tensor token_ids({static_cast<size_t>(actual_bs), static_cast<size_t>(seq_len)}, DType::FP32);
        Tensor targets({static_cast<size_t>(actual_bs * seq_len)}, DType::FP32);

        float* tip = token_ids.fp32_data();
        float* tp = targets.fp32_data();
        for (int i = 0; i < actual_bs; ++i) {
            for (int s = 0; s < seq_len; ++s) {
                tip[i * seq_len + s] = static_cast<float>(dataset.samples[batch_start + i][s]);
                int next_s = (s + 1) % seq_len;
                tp[i * seq_len + s] = static_cast<float>(dataset.samples[batch_start + i][next_s]);
            }
        }

        Tensor logits, aux;
        model.forward(token_ids, logits, aux);
        float loss = compute_loss(logits, targets, aux, status.completed_steps);
        epoch_loss += loss;

        if (cfg.self_evo_steps > 0 && status.completed_steps % cfg.self_evo_steps == 0) {
            self_evolution(dataset);
        }
    }

    return epoch_loss / static_cast<float>(num_batches);
}

// ==========================================
// MAIN TRAINING LOOP
// ==========================================

void Trainer::train(Dataset& dataset, const std::string& model_name, int start_epoch) {
    init_optimizer();

    std::cout << "\n\033[1;32m═══ TRAINING START ═══\033[0m\n";
    std::cout << "  Epochs: " << cfg.epochs << " | Batch: " << cfg.batch_size
              << " | LR: " << std::scientific << cfg.lr << "\n";
    std::cout << "  Params: " << total_params() << " | Steps/epoch: "
              << (dataset.size() + cfg.batch_size - 1) / cfg.batch_size << "\n\n";

    for (int epoch = start_epoch; epoch < cfg.epochs; ++epoch) {
        float avg_loss = train_epoch(dataset);

        float acc = 0.0f, ppl = 0.0f;
        evaluate(dataset, acc, ppl);

        status.remaining_epochs = cfg.epochs - epoch - 1;
        status.last_loss = avg_loss;
        status.last_acc = acc;
        status.last_ppl = ppl;

        std::cout << "  Epoch " << (epoch + 1) << "/" << cfg.epochs
                  << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | ACC: " << std::setprecision(4) << acc
                  << " | PPL: " << std::setprecision(2) << ppl
                  << " | Remaining: " << status.remaining_epochs << "\n";

        if (cfg.checkpoint_epochs > 0 && (epoch + 1) % cfg.checkpoint_epochs == 0) {
            save_safetensors(model, model_name + ".safetensors");

            CheckpointMeta meta;
            meta.remaining_epochs = status.remaining_epochs;
            meta.completed_steps = status.completed_steps;
            meta.last_loss = status.last_loss;
            meta.last_acc = status.last_acc;
            meta.last_ppl = status.last_ppl;
            meta.context_proficiency = status.context_proficiency;
            save_checkpoint_meta(meta, model_name + ".log");

            std::cout << "  💾 Checkpoint saved\n";
        }
    }

    save_safetensors(model, model_name + ".safetensors");
    CheckpointMeta meta;
    meta.remaining_epochs = 0;
    meta.completed_steps = status.completed_steps;
    meta.last_loss = status.last_loss;
    meta.last_acc = status.last_acc;
    meta.last_ppl = status.last_ppl;
    meta.context_proficiency = status.context_proficiency;
    save_checkpoint_meta(meta, model_name + ".log");

    std::cout << "\n\033[1;32m═══ TRAINING COMPLETE ═══\033[0m\n";
    std::cout << "  Final Loss: " << std::fixed << std::setprecision(4) << status.last_loss
              << " | ACC: " << std::setprecision(4) << status.last_acc
              << " | PPL: " << std::setprecision(2) << status.last_ppl << "\n\n";
}

} // namespace train
} // namespace nol

// ==========================================
// UNIT TEST TRAINING
// ==========================================
static void run_training_tests() {
    using namespace nol::train;
    using namespace nol::model;
    using namespace nol::config;
    using namespace nol::core;
    using namespace nol::token;
    std::cout << "[TEST] Memulai unit test training...\n";

    NolConfig cfg;
    cfg.vocab_size = 50;
    cfg.d_model = 32;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 1;
    cfg.d_head = 16;
    cfg.ffn_hidden = 64;
    cfg.max_seq_len = 8;
    cfg.n_hyp_heads = 1;
    cfg.d_hyp = 4;
    cfg.n_top_anchors = 2;
    cfg.n_children = 2;
    cfg.top_k_experts = 1;
    cfg.n_shared_experts = 1;
    cfg.rng_seed = 42;
    cfg.epochs = 2;
    cfg.batch_size = 4;
    cfg.lr = 1e-3f;
    cfg.warmup_steps = 5;
    cfg.checkpoint_epochs = 1;
    cfg.self_evo_steps = 10;

    // Test 1: Dataset generation
    {
        Dataset ds;
        ds.generate_random(20, 8, 50, 42);
        assert(ds.size() == 20);
        assert(ds.seq_len == 8);
        for (const auto& sample : ds.samples) {
            assert(static_cast<int>(sample.size()) == 8);
            for (int tok : sample) {
                assert(tok >= 0 && tok < 50);
            }
        }
        std::cout << "  Dataset generation OK\n";
    }

    // Test 2: Flatten/unflatten roundtrip
    {
        NolModel model(cfg);
        model.initialize_weights(42);

        Trainer trainer(model, cfg);
        std::vector<float> flat;
        trainer.flatten_params(flat);
        assert(flat.size() == trainer.total_params());

        flat[0] = 999.0f;
        trainer.unflatten_params(flat);
        assert(model.embed_E.fp32_data()[0] == 999.0f);

        model.initialize_weights(42);
        std::cout << "  Flatten/unflatten OK (" << flat.size() << " params)\n";
    }

    // Test 3: Loss computation
    {
        NolModel model(cfg);
        model.initialize_weights(42);
        Trainer trainer(model, cfg);

        Tensor logits({4, 50}, DType::FP32);
        logits.fill(0.0f);
        Tensor targets({4}, DType::FP32);
        targets.fill(0.0f);
        Tensor aux({2}, DType::FP32);
        aux.fill(0.0f);

        float loss = trainer.compute_loss(logits, targets, aux, 100);
        assert(std::abs(loss - std::log(50.0f)) < 0.1f);
        std::cout << "  Loss computation OK (loss=" << loss << ")\n";
    }

    // Test 4: Learning rate schedule
    {
        NolModel model(cfg);
        model.initialize_weights(42);
        Trainer trainer(model, cfg);

        float lr0 = trainer.get_lr(0);
        float lr2 = trainer.get_lr(2);
        float lr5 = trainer.get_lr(5);
        assert(lr0 < lr2);
        assert(lr2 < lr5);
        assert(std::abs(lr5 - cfg.lr) < 1e-6f);

        float lr10 = trainer.get_lr(10);
        assert(lr10 < lr5);
        std::cout << "  LR schedule OK (warmup:" << lr0 << "->" << lr5 << " decay:" << lr10 << ")\n";
    }

    // Test 5: Short training run
    {
        NolModel model(cfg);
        model.initialize_weights(42);
        Trainer trainer(model, cfg);

        Dataset ds;
        ds.generate_random(16, 8, 50, 42);

        trainer.train(ds, "/tmp/test_train_model", 0);

        assert(trainer.status.completed_steps > 0);
        assert(trainer.status.last_loss > 0.0f);

        std::ifstream f1("/tmp/test_train_model.safetensors");
        std::ifstream f2("/tmp/test_train_model.log");
        assert(f1.good());
        assert(f2.good());
        f1.close();
        f2.close();

        std::cout << "  Short training run OK\n";
    }

    // Test 6: Resume from checkpoint
    {
        CheckpointMeta meta;
        bool ok = load_checkpoint_meta(meta, "/tmp/test_train_model.log");
        assert(ok);
        assert(meta.remaining_epochs == 0);
        assert(meta.completed_steps > 0);
        std::cout << "  Resume checkpoint OK (steps=" << meta.completed_steps << ")\n";
    }

    // Test 7: Dataset load_from_texts dengan tokenizer
    {
        BPETokenizer tok;
        std::vector<std::string> corpus = {
            "hello world",
            "the quick brown fox",
            "testing dataset loader"
        };
        tok.build_vocab(corpus, 50);

        Dataset ds;
        int count = ds.load_from_texts(corpus, tok, 8);
        assert(count == 3);
        assert(ds.seq_len == 8);
        for (const auto& sample : ds.samples) {
            assert(static_cast<int>(sample.size()) == 8);
        }
        std::cout << "  Dataset load_from_texts OK (" << count << " samples)\n";
    }

    // Test 8: Dataset load_from_text file
    {
        // Buat file test
        {
            std::ofstream f("/tmp/test_corpus.txt");
            f << "hello world\n";
            f << "foo bar baz\n";
            f << "testing one two three\n";
            f.close();
        }

        BPETokenizer tok;
        std::vector<std::string> corpus = {"hello world foo bar baz testing one two three"};
        tok.build_vocab(corpus, 50);

        Dataset ds;
        int count = ds.load_from_text("/tmp/test_corpus.txt", tok, 8);
        assert(count == 3);
        std::cout << "  Dataset load_from_text file OK\n";

        std::remove("/tmp/test_corpus.txt");
    }

    // Cleanup
    std::remove("/tmp/test_train_model.safetensors");
    std::remove("/tmp/test_train_model.log");

    std::cout << "[PASS] ✅ Semua unit test training berhasil!\n\n";
}

#ifdef NOL_TEST_TRAINING
int main() {
    run_training_tests();
    return 0;
}
#endif
