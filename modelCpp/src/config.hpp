#ifndef PROYEK_CONFIG_HPP
#define PROYEK_CONFIG_HPP

#include <cstdint>
#include <string>

struct ModelConfig {
    int vocab = 32000;
    int d_model = 768;
    int n_layers = 12;
    int n_heads = 12;
    int n_kv_heads = 4;
    int d_head = 64;
    int ffn_hidden = 2048;
    int max_seq = 2048;

    bool tied_embeddings = true;
    bool use_moe = true;
    bool use_rope = true;
    bool use_gqa = true;
    bool use_swiglu = true;
};

struct HyperbolicConfig {
    bool enable_t1 = true; // Hyperbolic Geodesic Attention
    bool enable_t2 = true; // Gated Fusion
    bool enable_t3 = true; // Core-Gravity Layer
    bool enable_t4 = true; // Topic Isolation Gate
    bool enable_t5 = true; // Hyperbolic Output Head

    int n_hyp_heads = 4;
    int d_hyp = 32;

    float c_init = 1.0f;
    float c_min = 1e-3f;

    float tau_att = 1.0f;
    float tau_core = 1.0f;
    float tau_route = 1.0f;
    float tau_head = 1.0f;
    float tau_hcl = 1.0f;
};

struct RoutingConfig {
    int n_top_anchors = 16;
    int n_children = 16;
    int top_k_experts = 2;

    float iso_delta = 2.0f;
    float route_beta = 1.0f;

    int anchor_evolve_steps = 0;
};

struct LossConfig {
    float w_hcl = 0.1f;
    float w_hier = 0.05f;
    float w_iso = 0.02f;
    float w_bal = 0.01f;

    float label_smoothing = 0.1f;
    float hier_margin = 0.2f;
};

struct TrainConfig {
    int epochs = 10;
    int batch_size = 8;
    int seq_len = 2048;

    float learning_rate = 3e-4f;
    float min_lr = 3e-5f;
    float weight_decay = 0.1f;

    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float adam_eps = 1e-8f;

    float warmup_ratio = 0.03f;
    float grad_clip = 1.0f;

    int save_every_steps = 500;
    int save_every_epochs = 1;

    float stop_loss_below = 0.0f;
    float stop_acc_above = 0.0f;
    int early_patience = 0;

    int curriculum_warmup_percent = 10;

    bool shuffle = true;
    bool pack_sequences = true;

    bool use_ema = true;
    float ema_decay = 0.999f;

    int eval_every_steps = 0;
};

struct RuntimeConfig {
    std::string modelName;
    std::string modelDir = "models";
    std::string configPath;
    std::string tokenizerPath;
    std::string dataPath;
    std::string valDataPath;
    std::string logPath;

    uint64_t seed = 1234;

    bool verbose = false;
    bool dryRun = false;
    bool yes = false;
    bool overwrite = false;
};

struct ResolvedConfig {
    std::string profile = "default";

    ModelConfig model;
    HyperbolicConfig hyper;
    RoutingConfig routing;
    LossConfig loss;
    TrainConfig train;
    RuntimeConfig runtime;
};

ResolvedConfig defaultConfig();

void normalizeConfig(ResolvedConfig& cfg);

bool validateConfig(const ResolvedConfig& cfg, std::string& err);

ResolvedConfig presetConfig(const std::string& name, std::string& err);

bool parseConfigText(const std::string& text, ResolvedConfig& cfg, std::string& err);

bool loadConfigFile(const std::string& path, ResolvedConfig& cfg, std::string& err);

std::string configToJson(const ResolvedConfig& cfg);

bool saveConfigFile(const std::string& path, const ResolvedConfig& cfg, std::string& err);

bool applyOverride(
    ResolvedConfig& cfg,
    const std::string& key,
    const std::string& value,
    std::string& err
);

long long estimateParameters(const ResolvedConfig& cfg);

long long estimateTrainingMemoryBytes(const ResolvedConfig& cfg);

#endif // PROYEK_CONFIG_HPP
