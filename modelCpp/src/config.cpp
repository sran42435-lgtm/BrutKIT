#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }

    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }

    return s.substr(start, end - start);
}

std::string lower(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

std::string unquote(const std::string& v) {
    if (v.size() >= 2) {
        char a = v.front();
        char b = v.back();
        if ((a == '"' && b == '"') || (a == '\'' && b == '\'')) {
            return v.substr(1, v.size() - 2);
        }
    }
    return v;
}

bool parseBool(const std::string& value, bool& out, std::string& err) {
    std::string v = lower(trim(value));

    if (v == "true" || v == "1" || v == "yes" || v == "on") {
        out = true;
        return true;
    }

    if (v == "false" || v == "0" || v == "no" || v == "off") {
        out = false;
        return true;
    }

    err = "invalid boolean value: '" + value + "'";
    return false;
}

bool parseInt(const std::string& value, long long& out, std::string& err) {
    std::string v = trim(value);

    if (v.empty()) {
        err = "empty integer value";
        return false;
    }

    try {
        std::size_t pos = 0;
        out = std::stoll(v, &pos);

        if (pos != v.size()) {
            err = "trailing characters in integer value: '" + value + "'";
            return false;
        }

        return true;
    } catch (...) {
        err = "invalid integer value: '" + value + "'";
        return false;
    }
}

bool setInt(int& field, const std::string& value, std::string& err, const std::string& name) {
    long long tmp = 0;

    if (!parseInt(value, tmp, err)) {
        if (err.empty()) {
            err = name + ": invalid integer";
        } else {
            err = name + ": " + err;
        }
        return false;
    }

    if (tmp < std::numeric_limits<int>::min() || tmp > std::numeric_limits<int>::max()) {
        err = name + ": integer out of range";
        return false;
    }

    field = static_cast<int>(tmp);
    return true;
}

bool setUint64(uint64_t& field, const std::string& value, std::string& err, const std::string& name) {
    std::string v = trim(value);

    if (v.empty()) {
        err = name + ": empty unsigned integer value";
        return false;
    }

    if (v[0] == '-') {
        err = name + ": negative value not allowed for unsigned integer";
        return false;
    }

    try {
        std::size_t pos = 0;
        unsigned long long tmp = std::stoull(v, &pos);

        if (pos != v.size()) {
            err = name + ": trailing characters in unsigned integer value";
            return false;
        }

        field = static_cast<uint64_t>(tmp);
        return true;
    } catch (...) {
        err = name + ": invalid unsigned integer value";
        return false;
    }
}

bool setFloat(float& field, const std::string& value, std::string& err, const std::string& name) {
    std::string v = trim(value);

    if (v.empty()) {
        err = name + ": empty float value";
        return false;
    }

    try {
        std::size_t pos = 0;
        float tmp = std::stof(v, &pos);

        if (pos != v.size()) {
            err = name + ": trailing characters in float value";
            return false;
        }

        if (!std::isfinite(static_cast<double>(tmp))) {
            err = name + ": non-finite float value";
            return false;
        }

        field = tmp;
        return true;
    } catch (...) {
        err = name + ": invalid float value";
        return false;
    }
}

bool setBool(bool& field, const std::string& value, std::string& err, const std::string& name) {
    bool tmp = false;

    if (!parseBool(value, tmp, err)) {
        err = name + ": " + err;
        return false;
    }

    field = tmp;
    return true;
}

bool setString(std::string& field, const std::string& value) {
    field = value;
    return true;
}

bool applyKeyValue(
    ResolvedConfig& cfg,
    const std::string& section,
    const std::string& key,
    const std::string& value,
    std::string& err
) {
    std::string sec = lower(section);
    std::string k = lower(key);
    std::string val = unquote(trim(value));
    std::string fullName = sec + "." + k;

    auto failUnknown = [&]() -> bool {
        err = "unknown config key: " + fullName;
        return false;
    };

    auto setI = [&](int& field) -> bool {
        return setInt(field, val, err, fullName);
    };

    auto setU = [&](uint64_t& field) -> bool {
        return setUint64(field, val, err, fullName);
    };

    auto setF = [&](float& field) -> bool {
        return setFloat(field, val, err, fullName);
    };

    auto setB = [&](bool& field) -> bool {
        return setBool(field, val, err, fullName);
    };

    auto setS = [&](std::string& field) -> bool {
        return setString(field, val);
    };

    if (sec == "model") {
        if (k == "vocab") return setI(cfg.model.vocab);
        if (k == "d_model" || k == "hidden_size") return setI(cfg.model.d_model);
        if (k == "n_layers" || k == "num_layers" || k == "layers") return setI(cfg.model.n_layers);
        if (k == "n_heads" || k == "num_heads" || k == "heads") return setI(cfg.model.n_heads);
        if (k == "n_kv_heads" || k == "num_kv_heads" || k == "kv_heads") return setI(cfg.model.n_kv_heads);
        if (k == "d_head" || k == "head_dim") return setI(cfg.model.d_head);
        if (k == "ffn_hidden" || k == "intermediate_size") return setI(cfg.model.ffn_hidden);
        if (k == "max_seq" || k == "max_position_embeddings" || k == "context_length") return setI(cfg.model.max_seq);

        if (k == "tied_embeddings" || k == "tie_word_embeddings") return setB(cfg.model.tied_embeddings);
        if (k == "use_moe" || k == "moe") return setB(cfg.model.use_moe);
        if (k == "use_rope" || k == "rope") return setB(cfg.model.use_rope);
        if (k == "use_gqa" || k == "gqa") return setB(cfg.model.use_gqa);
        if (k == "use_swiglu" || k == "swiglu") return setB(cfg.model.use_swiglu);

        return failUnknown();
    }

    if (sec == "hyperbolic" || sec == "hyper") {
        if (k == "enable_t1") return setB(cfg.hyper.enable_t1);
        if (k == "enable_t2") return setB(cfg.hyper.enable_t2);
        if (k == "enable_t3") return setB(cfg.hyper.enable_t3);
        if (k == "enable_t4") return setB(cfg.hyper.enable_t4);
        if (k == "enable_t5") return setB(cfg.hyper.enable_t5);

        if (k == "n_hyp_heads" || k == "hyp_heads" || k == "hyper_heads") return setI(cfg.hyper.n_hyp_heads);
        if (k == "d_hyp" || k == "hyp_dim" || k == "hyper_dim") return setI(cfg.hyper.d_hyp);

        if (k == "c_init" || k == "curvature_init") return setF(cfg.hyper.c_init);
        if (k == "c_min" || k == "curvature_min") return setF(cfg.hyper.c_min);

        if (k == "tau_att") return setF(cfg.hyper.tau_att);
        if (k == "tau_core") return setF(cfg.hyper.tau_core);
        if (k == "tau_route") return setF(cfg.hyper.tau_route);
        if (k == "tau_head") return setF(cfg.hyper.tau_head);
        if (k == "tau_hcl") return setF(cfg.hyper.tau_hcl);

        return failUnknown();
    }

    if (sec == "routing" || sec == "moe") {
        if (k == "n_top_anchors" || k == "top_anchors" || k == "num_top_anchors") return setI(cfg.routing.n_top_anchors);
        if (k == "n_children" || k == "children_per_anchor") return setI(cfg.routing.n_children);
        if (k == "top_k_experts" || k == "top_k") return setI(cfg.routing.top_k_experts);

        if (k == "iso_delta" || k == "isolation_delta") return setF(cfg.routing.iso_delta);
        if (k == "route_beta" || k == "beta") return setF(cfg.routing.route_beta);

        if (k == "anchor_evolve_steps") return setI(cfg.routing.anchor_evolve_steps);

        return failUnknown();
    }

    if (sec == "loss") {
        if (k == "w_hcl" || k == "alpha") return setF(cfg.loss.w_hcl);
        if (k == "w_hier" || k == "beta") return setF(cfg.loss.w_hier);
        if (k == "w_iso" || k == "gamma") return setF(cfg.loss.w_iso);
        if (k == "w_bal" || k == "delta") return setF(cfg.loss.w_bal);

        if (k == "label_smoothing") return setF(cfg.loss.label_smoothing);
        if (k == "hier_margin") return setF(cfg.loss.hier_margin);

        return failUnknown();
    }

    if (sec == "train" || sec == "training") {
        if (k == "epochs") return setI(cfg.train.epochs);
        if (k == "batch_size" || k == "batch") return setI(cfg.train.batch_size);
        if (k == "seq_len" || k == "seq") return setI(cfg.train.seq_len);

        if (k == "learning_rate" || k == "lr") return setF(cfg.train.learning_rate);
        if (k == "min_lr") return setF(cfg.train.min_lr);
        if (k == "weight_decay" || k == "wd") return setF(cfg.train.weight_decay);

        if (k == "beta1") return setF(cfg.train.beta1);
        if (k == "beta2") return setF(cfg.train.beta2);
        if (k == "adam_eps" || k == "eps") return setF(cfg.train.adam_eps);

        if (k == "warmup_ratio" || k == "warmup") return setF(cfg.train.warmup_ratio);
        if (k == "grad_clip" || k == "clip") return setF(cfg.train.grad_clip);

        if (k == "save_every_steps" || k == "save_steps") return setI(cfg.train.save_every_steps);
        if (k == "save_every_epochs") return setI(cfg.train.save_every_epochs);

        if (k == "stop_loss_below") return setF(cfg.train.stop_loss_below);
        if (k == "stop_acc_above") return setF(cfg.train.stop_acc_above);
        if (k == "early_patience" || k == "patience") return setI(cfg.train.early_patience);

        if (k == "curriculum_warmup_percent" || k == "curriculum_warmup") return setI(cfg.train.curriculum_warmup_percent);

        if (k == "shuffle") return setB(cfg.train.shuffle);
        if (k == "pack_sequences" || k == "pack") return setB(cfg.train.pack_sequences);

        if (k == "use_ema" || k == "ema") return setB(cfg.train.use_ema);
        if (k == "ema_decay") return setF(cfg.train.ema_decay);

        if (k == "eval_every_steps" || k == "eval_steps") return setI(cfg.train.eval_every_steps);

        return failUnknown();
    }

    if (sec == "runtime" || sec == "cli") {
        if (k == "model_name" || k == "name") return setS(cfg.runtime.modelName);
        if (k == "model_dir" || k == "dir") return setS(cfg.runtime.modelDir);
        if (k == "config_path") return setS(cfg.runtime.configPath);
        if (k == "tokenizer_path") return setS(cfg.runtime.tokenizerPath);

        if (k == "data_path" || k == "data") return setS(cfg.runtime.dataPath);
        if (k == "val_data_path" || k == "val_data") return setS(cfg.runtime.valDataPath);
        if (k == "log_path") return setS(cfg.runtime.logPath);

        if (k == "seed") return setU(cfg.runtime.seed);

        if (k == "verbose") return setB(cfg.runtime.verbose);
        if (k == "dry_run") return setB(cfg.runtime.dryRun);
        if (k == "yes") return setB(cfg.runtime.yes);
        if (k == "overwrite") return setB(cfg.runtime.overwrite);

        return failUnknown();
    }

    err = "unknown config section: " + sec;
    return false;
}

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);

    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0x0f];
                    out += hex[c & 0x0f];
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }

    return out;
}

void appendJsonString(std::ostream& os, const std::string& s) {
    os << '"' << jsonEscape(s) << '"';
}

const char* boolToStr(bool x) {
    return x ? "true" : "false";
}

} // namespace

ResolvedConfig defaultConfig() {
    ResolvedConfig cfg;
    normalizeConfig(cfg);
    return cfg;
}

void normalizeConfig(ResolvedConfig& cfg) {
    if (cfg.model.n_heads > 0 && cfg.model.d_model > 0 &&
        cfg.model.d_model % cfg.model.n_heads == 0) {
        cfg.model.d_head = cfg.model.d_model / cfg.model.n_heads;
    } else {
        cfg.model.d_head = 0;
    }

    if (cfg.train.seq_len <= 0) {
        cfg.train.seq_len = cfg.model.max_seq;
    }

    if (cfg.runtime.modelDir.empty()) {
        cfg.runtime.modelDir = "models";
    }

    if (!cfg.model.use_moe) {
        cfg.hyper.enable_t4 = false;
    }

    if (cfg.routing.top_k_experts <= 0) {
        cfg.routing.top_k_experts = 1;
    }

    if (cfg.hyper.n_hyp_heads < 0) {
        cfg.hyper.n_hyp_heads = 0;
    }
}

bool validateConfig(const ResolvedConfig& cfg, std::string& err) {
    auto fail = [&](const std::string& msg) -> bool {
        err = msg;
        return false;
    };

    auto finite = [](float x) -> bool {
        return std::isfinite(static_cast<double>(x));
    };

    if (cfg.model.vocab <= 0) return fail("model.vocab must be > 0");
    if (cfg.model.d_model <= 0) return fail("model.d_model must be > 0");
    if (cfg.model.n_layers <= 0) return fail("model.n_layers must be > 0");
    if (cfg.model.n_heads <= 0) return fail("model.n_heads must be > 0");
    if (cfg.model.ffn_hidden <= 0) return fail("model.ffn_hidden must be > 0");
    if (cfg.model.max_seq <= 0) return fail("model.max_seq must be > 0");

    if (cfg.model.d_model % cfg.model.n_heads != 0) {
        return fail("model.d_model must be divisible by model.n_heads");
    }

    if (cfg.model.d_head <= 0) {
        return fail("model.d_head must be > 0 after normalization");
    }

    if (cfg.model.d_head != cfg.model.d_model / cfg.model.n_heads) {
        return fail("model.d_head must equal model.d_model / model.n_heads");
    }

    if (cfg.model.n_kv_heads <= 0) return fail("model.n_kv_heads must be > 0");
    if (cfg.model.n_kv_heads > cfg.model.n_heads) {
        return fail("model.n_kv_heads must be <= model.n_heads");
    }

    if (cfg.model.n_heads % cfg.model.n_kv_heads != 0) {
        return fail("model.n_heads must be divisible by model.n_kv_heads for GQA");
    }

    if (cfg.hyper.enable_t1) {
        if (cfg.hyper.n_hyp_heads <= 0) {
            return fail("hyperbolic.n_hyp_heads must be > 0 when T1 is enabled");
        }

        if (cfg.hyper.n_hyp_heads > cfg.model.n_heads) {
            return fail("hyperbolic.n_hyp_heads must be <= model.n_heads");
        }

        if (cfg.hyper.d_hyp <= 0) {
            return fail("hyperbolic.d_hyp must be > 0 when T1 is enabled");
        }
    }

    if (cfg.hyper.enable_t3 || cfg.hyper.enable_t5) {
        if (cfg.hyper.d_hyp <= 0) {
            return fail("hyperbolic.d_hyp must be > 0 when T3 or T5 is enabled");
        }
    }

    if (!finite(cfg.hyper.c_init) || cfg.hyper.c_init <= 0.0f) {
        return fail("hyperbolic.c_init must be finite and > 0");
    }

    if (!finite(cfg.hyper.c_min) || cfg.hyper.c_min <= 0.0f) {
        return fail("hyperbolic.c_min must be finite and > 0");
    }

    if (!finite(cfg.hyper.tau_att) || cfg.hyper.tau_att <= 0.0f) {
        return fail("hyperbolic.tau_att must be finite and > 0");
    }

    if (!finite(cfg.hyper.tau_core) || cfg.hyper.tau_core <= 0.0f) {
        return fail("hyperbolic.tau_core must be finite and > 0");
    }

    if (!finite(cfg.hyper.tau_route) || cfg.hyper.tau_route <= 0.0f) {
        return fail("hyperbolic.tau_route must be finite and > 0");
    }

    if (!finite(cfg.hyper.tau_head) || cfg.hyper.tau_head <= 0.0f) {
        return fail("hyperbolic.tau_head must be finite and > 0");
    }

    if (!finite(cfg.hyper.tau_hcl) || cfg.hyper.tau_hcl <= 0.0f) {
        return fail("hyperbolic.tau_hcl must be finite and > 0");
    }

    if (cfg.hyper.enable_t4 && !cfg.model.use_moe) {
        return fail("hyperbolic.enable_t4 requires model.use_moe=true");
    }

    if (cfg.routing.n_top_anchors <= 0) {
        return fail("routing.n_top_anchors must be > 0");
    }

    if (cfg.routing.n_children <= 0) {
        return fail("routing.n_children must be > 0");
    }

    if (cfg.routing.top_k_experts <= 0) {
        return fail("routing.top_k_experts must be > 0");
    }

    if (cfg.routing.top_k_experts > cfg.routing.n_top_anchors) {
        return fail("routing.top_k_experts must be <= routing.n_top_anchors");
    }

    if (!finite(cfg.routing.iso_delta) || cfg.routing.iso_delta <= 0.0f) {
        return fail("routing.iso_delta must be finite and > 0");
    }

    if (!finite(cfg.routing.route_beta) || cfg.routing.route_beta < 0.0f) {
        return fail("routing.route_beta must be finite and >= 0");
    }

    if (cfg.routing.anchor_evolve_steps < 0) {
        return fail("routing.anchor_evolve_steps must be >= 0");
    }

    if (!finite(cfg.loss.w_hcl) || cfg.loss.w_hcl < 0.0f) {
        return fail("loss.w_hcl must be finite and >= 0");
    }

    if (!finite(cfg.loss.w_hier) || cfg.loss.w_hier < 0.0f) {
        return fail("loss.w_hier must be finite and >= 0");
    }

    if (!finite(cfg.loss.w_iso) || cfg.loss.w_iso < 0.0f) {
        return fail("loss.w_iso must be finite and >= 0");
    }

    if (!finite(cfg.loss.w_bal) || cfg.loss.w_bal < 0.0f) {
        return fail("loss.w_bal must be finite and >= 0");
    }

    if (!finite(cfg.loss.label_smoothing) ||
        cfg.loss.label_smoothing < 0.0f ||
        cfg.loss.label_smoothing >= 1.0f) {
        return fail("loss.label_smoothing must be in [0, 1)");
    }

    if (!finite(cfg.loss.hier_margin) || cfg.loss.hier_margin < 0.0f) {
        return fail("loss.hier_margin must be finite and >= 0");
    }

    if (cfg.train.epochs <= 0) return fail("train.epochs must be > 0");
    if (cfg.train.batch_size <= 0) return fail("train.batch_size must be > 0");
    if (cfg.train.seq_len <= 0) return fail("train.seq_len must be > 0");

    if (cfg.train.seq_len > cfg.model.max_seq) {
        return fail("train.seq_len must be <= model.max_seq");
    }

    if (!finite(cfg.train.learning_rate) || cfg.train.learning_rate <= 0.0f) {
        return fail("train.learning_rate must be finite and > 0");
    }

    if (!finite(cfg.train.min_lr) || cfg.train.min_lr < 0.0f) {
        return fail("train.min_lr must be finite and >= 0");
    }

    if (cfg.train.min_lr > cfg.train.learning_rate) {
        return fail("train.min_lr must be <= train.learning_rate");
    }

    if (!finite(cfg.train.weight_decay) || cfg.train.weight_decay < 0.0f) {
        return fail("train.weight_decay must be finite and >= 0");
    }

    if (!finite(cfg.train.beta1) || cfg.train.beta1 <= 0.0f || cfg.train.beta1 >= 1.0f) {
        return fail("train.beta1 must be in (0, 1)");
    }

    if (!finite(cfg.train.beta2) || cfg.train.beta2 <= 0.0f || cfg.train.beta2 >= 1.0f) {
        return fail("train.beta2 must be in (0, 1)");
    }

    if (!finite(cfg.train.adam_eps) || cfg.train.adam_eps <= 0.0f) {
        return fail("train.adam_eps must be finite and > 0");
    }

    if (!finite(cfg.train.warmup_ratio) ||
        cfg.train.warmup_ratio < 0.0f ||
        cfg.train.warmup_ratio >= 1.0f) {
        return fail("train.warmup_ratio must be in [0, 1)");
    }

    if (!finite(cfg.train.grad_clip) || cfg.train.grad_clip <= 0.0f) {
        return fail("train.grad_clip must be finite and > 0");
    }

    if (cfg.train.save_every_steps < 0) {
        return fail("train.save_every_steps must be >= 0");
    }

    if (cfg.train.save_every_epochs < 0) {
        return fail("train.save_every_epochs must be >= 0");
    }

    if (!finite(cfg.train.stop_loss_below) || cfg.train.stop_loss_below < 0.0f) {
        return fail("train.stop_loss_below must be finite and >= 0");
    }

    if (!finite(cfg.train.stop_acc_above) || cfg.train.stop_acc_above < 0.0f) {
        return fail("train.stop_acc_above must be finite and >= 0");
    }

    if (cfg.train.early_patience < 0) {
        return fail("train.early_patience must be >= 0");
    }

    if (cfg.train.curriculum_warmup_percent < 0 ||
        cfg.train.curriculum_warmup_percent > 100) {
        return fail("train.curriculum_warmup_percent must be in [0, 100]");
    }

    if (!finite(cfg.train.ema_decay) ||
        cfg.train.ema_decay < 0.0f ||
        cfg.train.ema_decay >= 1.0f) {
        return fail("train.ema_decay must be in [0, 1)");
    }

    if (cfg.train.eval_every_steps < 0) {
        return fail("train.eval_every_steps must be >= 0");
    }

    err.clear();
    return true;
}

ResolvedConfig presetConfig(const std::string& name, std::string& err) {
    ResolvedConfig cfg = defaultConfig();
    std::string p = lower(trim(name));

    if (p.empty()) {
        p = "default";
    }

    cfg.profile = p;

    if (p == "default") {
        // already default
    } else if (p == "tiny") {
        cfg.model.vocab = 512;
        cfg.model.d_model = 128;
        cfg.model.n_layers = 2;
        cfg.model.n_heads = 4;
        cfg.model.n_kv_heads = 2;
        cfg.model.ffn_hidden = 256;
        cfg.model.max_seq = 256;

        cfg.model.tied_embeddings = true;
        cfg.model.use_moe = true;
        cfg.model.use_rope = true;
        cfg.model.use_gqa = true;
        cfg.model.use_swiglu = true;

        cfg.hyper.enable_t1 = true;
        cfg.hyper.enable_t2 = true;
        cfg.hyper.enable_t3 = true;
        cfg.hyper.enable_t4 = true;
        cfg.hyper.enable_t5 = true;

        cfg.hyper.n_hyp_heads = 2;
        cfg.hyper.d_hyp = 16;

        cfg.hyper.c_init = 1.0f;
        cfg.hyper.c_min = 1e-3f;

        cfg.hyper.tau_att = 1.0f;
        cfg.hyper.tau_core = 1.0f;
        cfg.hyper.tau_route = 1.0f;
        cfg.hyper.tau_head = 1.0f;
        cfg.hyper.tau_hcl = 1.0f;

        cfg.routing.n_top_anchors = 4;
        cfg.routing.n_children = 4;
        cfg.routing.top_k_experts = 2;
        cfg.routing.iso_delta = 1.0f;
        cfg.routing.route_beta = 1.0f;
        cfg.routing.anchor_evolve_steps = 0;

        cfg.loss.w_hcl = 0.1f;
        cfg.loss.w_hier = 0.05f;
        cfg.loss.w_iso = 0.02f;
        cfg.loss.w_bal = 0.01f;
        cfg.loss.label_smoothing = 0.1f;
        cfg.loss.hier_margin = 0.2f;

        cfg.train.epochs = 3;
        cfg.train.batch_size = 4;
        cfg.train.seq_len = 256;

        cfg.train.learning_rate = 3e-4f;
        cfg.train.min_lr = 3e-5f;
        cfg.train.weight_decay = 0.1f;

        cfg.train.beta1 = 0.9f;
        cfg.train.beta2 = 0.95f;
        cfg.train.adam_eps = 1e-8f;

        cfg.train.warmup_ratio = 0.03f;
        cfg.train.grad_clip = 1.0f;

        cfg.train.save_every_steps = 100;
        cfg.train.save_every_epochs = 1;

        cfg.train.stop_loss_below = 0.0f;
        cfg.train.stop_acc_above = 0.0f;
        cfg.train.early_patience = 0;

        cfg.train.curriculum_warmup_percent = 10;

        cfg.train.shuffle = true;
        cfg.train.pack_sequences = true;

        cfg.train.use_ema = true;
        cfg.train.ema_decay = 0.999f;

        cfg.train.eval_every_steps = 0;

        cfg.runtime.seed = 1234;
    } else if (p == "small") {
        cfg.model.vocab = 4096;
        cfg.model.d_model = 256;
        cfg.model.n_layers = 4;
        cfg.model.n_heads = 8;
        cfg.model.n_kv_heads = 2;
        cfg.model.ffn_hidden = 512;
        cfg.model.max_seq = 512;

        cfg.hyper.n_hyp_heads = 2;
        cfg.hyper.d_hyp = 16;

        cfg.routing.n_top_anchors = 8;
        cfg.routing.n_children = 8;
        cfg.routing.top_k_experts = 2;
        cfg.routing.iso_delta = 1.5f;

        cfg.train.epochs = 5;
        cfg.train.batch_size = 4;
        cfg.train.seq_len = 512;
        cfg.train.save_every_steps = 200;
        cfg.train.curriculum_warmup_percent = 10;
    } else if (p == "nol_s" || p == "nols" || p == "nol-s") {
        cfg.model.vocab = 32000;
        cfg.model.d_model = 768;
        cfg.model.n_layers = 12;
        cfg.model.n_heads = 12;
        cfg.model.n_kv_heads = 4;
        cfg.model.ffn_hidden = 2048;
        cfg.model.max_seq = 2048;

        cfg.hyper.n_hyp_heads = 4;
        cfg.hyper.d_hyp = 32;

        cfg.routing.n_top_anchors = 16;
        cfg.routing.n_children = 16;
        cfg.routing.top_k_experts = 2;
        cfg.routing.iso_delta = 2.0f;

        cfg.train.epochs = 10;
        cfg.train.batch_size = 8;
        cfg.train.seq_len = 2048;
        cfg.train.save_every_steps = 500;
        cfg.train.curriculum_warmup_percent = 10;
    } else {
        err = "unknown preset: " + name;
        return cfg;
    }

    normalizeConfig(cfg);

    std::string validationErr;
    if (!validateConfig(cfg, validationErr)) {
        err = "preset '" + name + "' produced invalid config: " + validationErr;
    } else {
        err.clear();
    }

    return cfg;
}

bool parseConfigText(const std::string& text, ResolvedConfig& cfg, std::string& err) {
    std::istringstream ss(text);
    std::string line;
    std::string section;
    bool hasSection = false;
    std::size_t lineNo = 0;

    while (std::getline(ss, line)) {
        ++lineNo;

        std::size_t cut = std::string::npos;

        std::size_t hash = line.find('#');
        if (hash != std::string::npos) {
            cut = hash;
        }

        std::size_t semi = line.find(';');
        if (semi != std::string::npos && (cut == std::string::npos || semi < cut)) {
            cut = semi;
        }

        if (cut != std::string::npos) {
            line = line.substr(0, cut);
        }

        line = trim(line);

        if (line.empty()) {
            continue;
        }

        if (line.front() == '[') {
            std::size_t end = line.find(']');

            if (end == std::string::npos) {
                err = "line " + std::to_string(lineNo) + ": unterminated section";
                return false;
            }

            section = lower(trim(line.substr(1, end - 1)));

            if (section.empty()) {
                err = "line " + std::to_string(lineNo) + ": empty section name";
                return false;
            }

            hasSection = true;
            continue;
        }

        if (!hasSection) {
            err = "line " + std::to_string(lineNo) + ": key outside any section";
            return false;
        }

        std::size_t eq = line.find('=');

        if (eq == std::string::npos) {
            err = "line " + std::to_string(lineNo) + ": missing '=' in key-value pair";
            return false;
        }

        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));

        if (key.empty()) {
            err = "line " + std::to_string(lineNo) + ": empty key";
            return false;
        }

        std::string subErr;

        if (!applyKeyValue(cfg, section, key, value, subErr)) {
            err = "line " + std::to_string(lineNo) + ": " + subErr;
            return false;
        }
    }

    normalizeConfig(cfg);
    err.clear();
    return true;
}

bool loadConfigFile(const std::string& path, ResolvedConfig& cfg, std::string& err) {
    std::ifstream f(path);

    if (!f.is_open()) {
        err = "cannot open config file: " + path;
        return false;
    }

    std::string content(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>()
    );

    cfg = defaultConfig();

    return parseConfigText(content, cfg, err);
}

std::string configToJson(const ResolvedConfig& cfg) {
    std::ostringstream os;
    os << std::setprecision(9);

    os << "{";

    os << "\"profile\":";
    appendJsonString(os, cfg.profile);

    os << ",\"model\":{";
    os << "\"vocab\":" << cfg.model.vocab;
    os << ",\"d_model\":" << cfg.model.d_model;
    os << ",\"n_layers\":" << cfg.model.n_layers;
    os << ",\"n_heads\":" << cfg.model.n_heads;
    os << ",\"n_kv_heads\":" << cfg.model.n_kv_heads;
    os << ",\"d_head\":" << cfg.model.d_head;
    os << ",\"ffn_hidden\":" << cfg.model.ffn_hidden;
    os << ",\"max_seq\":" << cfg.model.max_seq;
    os << ",\"tied_embeddings\":" << boolToStr(cfg.model.tied_embeddings);
    os << ",\"use_moe\":" << boolToStr(cfg.model.use_moe);
    os << ",\"use_rope\":" << boolToStr(cfg.model.use_rope);
    os << ",\"use_gqa\":" << boolToStr(cfg.model.use_gqa);
    os << ",\"use_swiglu\":" << boolToStr(cfg.model.use_swiglu);
    os << "}";

    os << ",\"hyperbolic\":{";
    os << "\"enable_t1\":" << boolToStr(cfg.hyper.enable_t1);
    os << ",\"enable_t2\":" << boolToStr(cfg.hyper.enable_t2);
    os << ",\"enable_t3\":" << boolToStr(cfg.hyper.enable_t3);
    os << ",\"enable_t4\":" << boolToStr(cfg.hyper.enable_t4);
    os << ",\"enable_t5\":" << boolToStr(cfg.hyper.enable_t5);
    os << ",\"n_hyp_heads\":" << cfg.hyper.n_hyp_heads;
    os << ",\"d_hyp\":" << cfg.hyper.d_hyp;
    os << ",\"c_init\":" << cfg.hyper.c_init;
    os << ",\"c_min\":" << cfg.hyper.c_min;
    os << ",\"tau_att\":" << cfg.hyper.tau_att;
    os << ",\"tau_core\":" << cfg.hyper.tau_core;
    os << ",\"tau_route\":" << cfg.hyper.tau_route;
    os << ",\"tau_head\":" << cfg.hyper.tau_head;
    os << ",\"tau_hcl\":" << cfg.hyper.tau_hcl;
    os << "}";

    os << ",\"routing\":{";
    os << "\"n_top_anchors\":" << cfg.routing.n_top_anchors;
    os << ",\"n_children\":" << cfg.routing.n_children;
    os << ",\"top_k_experts\":" << cfg.routing.top_k_experts;
    os << ",\"iso_delta\":" << cfg.routing.iso_delta;
    os << ",\"route_beta\":" << cfg.routing.route_beta;
    os << ",\"anchor_evolve_steps\":" << cfg.routing.anchor_evolve_steps;
    os << "}";

    os << ",\"loss\":{";
    os << "\"w_hcl\":" << cfg.loss.w_hcl;
    os << ",\"w_hier\":" << cfg.loss.w_hier;
    os << ",\"w_iso\":" << cfg.loss.w_iso;
    os << ",\"w_bal\":" << cfg.loss.w_bal;
    os << ",\"label_smoothing\":" << cfg.loss.label_smoothing;
    os << ",\"hier_margin\":" << cfg.loss.hier_margin;
    os << "}";

    os << ",\"train\":{";
    os << "\"epochs\":" << cfg.train.epochs;
    os << ",\"batch_size\":" << cfg.train.batch_size;
    os << ",\"seq_len\":" << cfg.train.seq_len;
    os << ",\"learning_rate\":" << cfg.train.learning_rate;
    os << ",\"min_lr\":" << cfg.train.min_lr;
    os << ",\"weight_decay\":" << cfg.train.weight_decay;
    os << ",\"beta1\":" << cfg.train.beta1;
    os << ",\"beta2\":" << cfg.train.beta2;
    os << ",\"adam_eps\":" << cfg.train.adam_eps;
    os << ",\"warmup_ratio\":" << cfg.train.warmup_ratio;
    os << ",\"grad_clip\":" << cfg.train.grad_clip;
    os << ",\"save_every_steps\":" << cfg.train.save_every_steps;
    os << ",\"save_every_epochs\":" << cfg.train.save_every_epochs;
    os << ",\"stop_loss_below\":" << cfg.train.stop_loss_below;
    os << ",\"stop_acc_above\":" << cfg.train.stop_acc_above;
    os << ",\"early_patience\":" << cfg.train.early_patience;
    os << ",\"curriculum_warmup_percent\":" << cfg.train.curriculum_warmup_percent;
    os << ",\"shuffle\":" << boolToStr(cfg.train.shuffle);
    os << ",\"pack_sequences\":" << boolToStr(cfg.train.pack_sequences);
    os << ",\"use_ema\":" << boolToStr(cfg.train.use_ema);
    os << ",\"ema_decay\":" << cfg.train.ema_decay;
    os << ",\"eval_every_steps\":" << cfg.train.eval_every_steps;
    os << "}";

    os << ",\"runtime\":{";
    os << "\"model_name\":";
    appendJsonString(os, cfg.runtime.modelName);
    os << ",\"model_dir\":";
    appendJsonString(os, cfg.runtime.modelDir);
    os << ",\"config_path\":";
    appendJsonString(os, cfg.runtime.configPath);
    os << ",\"tokenizer_path\":";
    appendJsonString(os, cfg.runtime.tokenizerPath);
    os << ",\"data_path\":";
    appendJsonString(os, cfg.runtime.dataPath);
    os << ",\"val_data_path\":";
    appendJsonString(os, cfg.runtime.valDataPath);
    os << ",\"log_path\":";
    appendJsonString(os, cfg.runtime.logPath);
    os << ",\"seed\":" << cfg.runtime.seed;
    os << ",\"verbose\":" << boolToStr(cfg.runtime.verbose);
    os << ",\"dry_run\":" << boolToStr(cfg.runtime.dryRun);
    os << ",\"yes\":" << boolToStr(cfg.runtime.yes);
    os << ",\"overwrite\":" << boolToStr(cfg.runtime.overwrite);
    os << "}";

    os << "}";

    return os.str();
}

bool saveConfigFile(const std::string& path, const ResolvedConfig& cfg, std::string& err) {
    std::ofstream out(path);

    if (!out.is_open()) {
        err = "cannot open config file for writing: " + path;
        return false;
    }

    out << configToJson(cfg);
    out.flush();

    if (!out.good()) {
        err = "failed to write config file: " + path;
        return false;
    }

    err.clear();
    return true;
}

bool applyOverride(
    ResolvedConfig& cfg,
    const std::string& key,
    const std::string& value,
    std::string& err
) {
    std::size_t dot = key.find('.');

    if (dot == std::string::npos || dot == 0 || dot + 1 >= key.size()) {
        err = "override key must be in section.field format, got: " + key;
        return false;
    }

    std::string section = key.substr(0, dot);
    std::string field = key.substr(dot + 1);

    if (!applyKeyValue(cfg, section, field, value, err)) {
        return false;
    }

    normalizeConfig(cfg);
    err.clear();
    return true;
}

long long estimateParameters(const ResolvedConfig& cfg) {
    ResolvedConfig c = cfg;
    normalizeConfig(c);

    long long V = c.model.vocab;
    long long d = c.model.d_model;
    long long L = c.model.n_layers;
    long long H = c.model.n_heads;
    long long KV = c.model.n_kv_heads;
    long long Dh = c.model.d_head;
    long long F = c.model.ffn_hidden;

    if (V <= 0 || d <= 0 || L <= 0 || H <= 0 || KV <= 0 || Dh <= 0 || F <= 0) {
        return 0;
    }

    long long params = 0;

    // Token embedding E [V x d]
    params += V * d;

    // Hyperbolic token table T [V x d_hyp]
    if (c.hyper.enable_t1 || c.hyper.enable_t3 || c.hyper.enable_t5) {
        params += V * static_cast<long long>(c.hyper.d_hyp);
    }

    for (long long layer = 0; layer < L; ++layer) {
        // Two RMSNorm gains per block
        params += 2 * d;

        // Traditional GQA/RoPE attention projections
        params += d * d;                         // Wq
        params += d * (KV * Dh);                 // Wk
        params += d * (KV * Dh);                 // Wv
        params += (H * Dh) * d;                  // Wo

        // T1: Hyperbolic Geodesic Attention projections
        if (c.hyper.enable_t1) {
            long long nh = c.hyper.n_hyp_heads;
            long long dh = c.hyper.d_hyp;
            long long lorentzDimPerHead = dh + 1;
            long long hypOutDim = nh * lorentzDimPerHead;

            params += d * hypOutDim;             // Wz
            params += d * hypOutDim;             // hyp Wq
            params += d * hypOutDim;             // hyp Wk
            params += d * hypOutDim;             // hyp Wv
            params += (nh * dh) * d;             // Wu maps tangent output to d_model
        }

        // T2: Gated Fusion gate
        if (c.hyper.enable_t2) {
            params += d * d;                     // Wg
        }

        // T3: Core-Gravity Layer anchors and projection
        if (c.hyper.enable_t3) {
            long long anchors = c.routing.n_top_anchors;
            long long children = c.routing.n_children;
            long long dh = c.hyper.d_hyp;

            params += anchors * dh;                          // level-1 anchors
            params += anchors * children * dh;               // level-2 children
            params += 1;                                     // lambda gate scalar
            params += dh * d;                                // Wc
            params += d;                                     // gamma vector
        }

        // T4: Topic Isolation Gate + MoE SwiGLU
        long long ffnShared = 3LL * d * F;

        if (c.hyper.enable_t4 && c.model.use_moe) {
            params += d * c.routing.n_top_anchors;           // routing logits
            params += ffnShared;                             // shared expert
            params += c.routing.n_top_anchors * ffnShared;   // topic experts
        } else {
            params += ffnShared;                             // dense/shared FFN only
        }
    }

    // Output head
    if (!c.model.tied_embeddings) {
        params += V * d;
    }

    // T5 hyperbolic head scalar lambda
    if (c.hyper.enable_t5) {
        params += 1;
    }

    return params;
}

long long estimateTrainingMemoryBytes(const ResolvedConfig& cfg) {
    long long p = estimateParameters(cfg);

    if (p <= 0) {
        return 0;
    }

    // Rough training memory:
    // weights fp32       = 4 bytes/param
    // gradients fp32     = 4 bytes/param
    // AdamW exp_avg      = 4 bytes/param
    // AdamW exp_avg_sq   = 4 bytes/param
    // total              = 16 bytes/param
    return p * 16LL;
}

#ifdef CONFIG_SELFTEST

int main() {
    int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "CHECK failed: " #cond                           \
                      << " at line " << __LINE__ << "\n";                 \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

#define CHECK_MSG(cond, msg)                                              \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "CHECK failed: " #msg                            \
                      << " at line " << __LINE__ << "\n";                 \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

    std::string err;

    // 1. Default config harus valid.
    {
        ResolvedConfig cfg = defaultConfig();
        CHECK(validateConfig(cfg, err));
        CHECK(cfg.model.vocab == 32000);
        CHECK(cfg.model.d_model == 768);
        CHECK(cfg.model.d_head == 64);
        CHECK(cfg.hyper.enable_t1 == true);
        CHECK(cfg.hyper.enable_t5 == true);
        CHECK(cfg.train.epochs == 10);
    }

    // 2. Preset tiny harus valid dan kecil.
    {
        err.clear();
        ResolvedConfig tiny = presetConfig("tiny", err);
        CHECK(err.empty());
        CHECK(validateConfig(tiny, err));
        CHECK(tiny.model.d_model == 128);
        CHECK(tiny.model.n_layers == 2);
        CHECK(tiny.model.vocab == 512);
        CHECK(tiny.hyper.d_hyp == 16);
        CHECK(tiny.routing.n_top_anchors == 4);
    }

    // 3. Preset tidak dikenal harus error.
    {
        err.clear();
        ResolvedConfig dummy = presetConfig("not-a-preset", err);
        (void)dummy;
        CHECK(!err.empty());
    }

    // 4. Parse INI-like config text.
    {
        std::string ini = R"(
# Ini adalah komentar
[model]
vocab = 512
d_model = 128
n_layers = 2
n_heads = 4
n_kv_heads = 2
ffn_hidden = 256
max_seq = 256
use_moe = true

[hyperbolic]
enable_t1 = true
enable_t2 = true
enable_t3 = false
enable_t4 = true
enable_t5 = true
n_hyp_heads = 2
d_hyp = 16
c_init = 1.0

[routing]
n_top_anchors = 4
n_children = 4
top_k_experts = 2
iso_delta = 1.0

[loss]
w_hcl = 0.1
w_hier = 0.05
w_iso = 0.02
w_bal = 0.01

[train]
epochs = 3
batch_size = 4
seq_len = 256
lr = 3e-4
save_every_steps = 100
curriculum_warmup_percent = 10

[runtime]
model_name = mawar
seed = 42
verbose = true
)";

        ResolvedConfig cfg = defaultConfig();
        CHECK(parseConfigText(ini, cfg, err));
        CHECK(validateConfig(cfg, err));

        CHECK(cfg.model.vocab == 512);
        CHECK(cfg.model.d_model == 128);
        CHECK(cfg.model.d_head == 32);
        CHECK(cfg.hyper.enable_t3 == false);
        CHECK(cfg.hyper.n_hyp_heads == 2);
        CHECK(cfg.routing.n_top_anchors == 4);
        CHECK(cfg.train.epochs == 3);
        CHECK(cfg.train.batch_size == 4);
        CHECK(std::fabs(cfg.train.learning_rate - 0.0003f) < 1e-6f);
        CHECK(cfg.runtime.modelName == "mawar");
        CHECK(cfg.runtime.seed == 42);
        CHECK(cfg.runtime.verbose == true);
    }

    // 5. Unknown key harus ditolak.
    {
        std::string bad = R"(
[model]
foo = 1
)";

        ResolvedConfig cfg = defaultConfig();
        err.clear();
        CHECK(!parseConfigText(bad, cfg, err));
        CHECK(!err.empty());
    }

    // 6. Override CLI-style harus bekerja dan menormalkan d_head.
    {
        ResolvedConfig cfg = presetConfig("tiny", err);
        CHECK(err.empty());

        CHECK(applyOverride(cfg, "model.d_model", "256", err));
        CHECK(cfg.model.d_model == 256);
        CHECK(cfg.model.d_head == 64);
        CHECK(validateConfig(cfg, err));

        err.clear();
        CHECK(!applyOverride(cfg, "model.foo", "1", err));
        CHECK(!err.empty());
    }

    // 7. JSON serialization harus mengandung field penting.
    {
        ResolvedConfig cfg = presetConfig("tiny", err);
        CHECK(err.empty());

        std::string json = configToJson(cfg);

        CHECK(json.find("\"model\"") != std::string::npos);
        CHECK(json.find("\"hyperbolic\"") != std::string::npos);
        CHECK(json.find("\"routing\"") != std::string::npos);
        CHECK(json.find("\"loss\"") != std::string::npos);
        CHECK(json.find("\"train\"") != std::string::npos);
        CHECK(json.find("\"runtime\"") != std::string::npos);

        CHECK(json.find("\"vocab\"") != std::string::npos);
        CHECK(json.find("512") != std::string::npos);
        CHECK(json.find("\"enable_t1\"") != std::string::npos);
        CHECK(json.find("true") != std::string::npos);
    }

    // 8. Validasi harus menangkap konfigurasi tidak konsisten.
    {
        ResolvedConfig bad = defaultConfig();
        bad.model.d_model = 100;
        bad.model.n_heads = 3;
        normalizeConfig(bad);

        err.clear();
        CHECK(!validateConfig(bad, err));
        CHECK(!err.empty());
    }

    // 9. Estimasi parameter dan memori harus positif untuk preset tiny.
    {
        ResolvedConfig tiny = presetConfig("tiny", err);
        CHECK(err.empty());

        long long params = estimateParameters(tiny);
        long long mem = estimateTrainingMemoryBytes(tiny);

        CHECK(params > 0);
        CHECK(mem > 0);
        CHECK(mem >= params * 4);
    }

    // 10. T4 tidak boleh aktif jika MoE dimatikan setelah normalisasi.
    {
        ResolvedConfig cfg = defaultConfig();
        cfg.model.use_moe = false;
        cfg.hyper.enable_t4 = true;
        normalizeConfig(cfg);

        CHECK(cfg.hyper.enable_t4 == false);
        CHECK(validateConfig(cfg, err));
    }

    if (failures == 0) {
        std::cout << "config unit tests passed\n";
        return 0;
    }

    std::cerr << failures << " config unit test(s) failed\n";
    return 1;
}

#endif // CONFIG_SELFTEST
