#include "trainer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string toLower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))
        );
    }

    return s;
}

std::string trim(const std::string& s) {
    std::size_t a = 0;

    while (a < s.size() &&
           std::isspace(static_cast<unsigned char>(s[a]))) {
        ++a;
    }

    std::size_t b = s.size();

    while (b > a &&
           std::isspace(static_cast<unsigned char>(s[b - 1]))) {
        --b;
    }

    return s.substr(a, b - a);
}

bool parseBoolValue(const std::string& value, bool fallback) {
    std::string v = toLower(trim(value));

    if (v == "1" || v == "true" || v == "yes" || v == "on") {
        return true;
    }

    if (v == "0" || v == "false" || v == "no" || v == "off") {
        return false;
    }

    return fallback;
}

int parseIntValue(const std::string& value, int fallback) {
    std::string v = trim(value);

    if (v.empty()) {
        return fallback;
    }

    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

float parseFloatValue(const std::string& value, float fallback) {
    std::string v = trim(value);

    if (v.empty()) {
        return fallback;
    }

    try {
        return std::stof(v);
    } catch (...) {
        return fallback;
    }
}

std::uint64_t parseUint64Value(
    const std::string& value,
    std::uint64_t fallback
) {
    std::string v = trim(value);

    if (v.empty()) {
        return fallback;
    }

    char* end = nullptr;

    unsigned long long x = std::strtoull(
        v.c_str(),
        &end,
        10
    );

    if (end == v.c_str() || *end != '\0') {
        return fallback;
    }

    return static_cast<std::uint64_t>(x);
}

bool hasFlag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == name) {
            return true;
        }
    }

    return false;
}

bool looksLikeFlag(const std::string& s) {
    return s.size() >= 2 && s[0] == '-' && s[1] == '-';
}

std::string optionValue(
    int argc,
    char** argv,
    const std::string& name,
    const std::string& fallback
) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == name) {
            if (i + 1 < argc) {
                std::string next = argv[i + 1];

                if (looksLikeFlag(next)) {
                    return fallback;
                }

                return next;
            }

            return fallback;
        }
    }

    return fallback;
}

int optionInt(
    int argc,
    char** argv,
    const std::string& name,
    int fallback
) {
    std::string v = optionValue(argc, argv, name, "");

    if (v.empty()) {
        return fallback;
    }

    return parseIntValue(v, fallback);
}

float optionFloat(
    int argc,
    char** argv,
    const std::string& name,
    float fallback
) {
    std::string v = optionValue(argc, argv, name, "");

    if (v.empty()) {
        return fallback;
    }

    return parseFloatValue(v, fallback);
}

std::uint64_t optionUint64(
    int argc,
    char** argv,
    const std::string& name,
    std::uint64_t fallback
) {
    std::string v = optionValue(argc, argv, name, "");

    if (v.empty()) {
        return fallback;
    }

    return parseUint64Value(v, fallback);
}

std::string dimsPath(const std::string& modelDir) {
    return modelDir + "/nol_dims.txt";
}

bool writeDimsFile(
    const std::string& modelDir,
    const arch::ArchitectureDims& d,
    std::string& err
) {
    err.clear();

    if (modelDir.empty()) {
        return true;
    }

    std::ostringstream oss;

    oss << "# nol architecture dims\n";

    oss << "vocab=" << d.vocab << "\n";
    oss << "d_model=" << d.d_model << "\n";
    oss << "n_layers=" << d.n_layers << "\n";

    oss << "n_heads=" << d.n_heads << "\n";
    oss << "n_kv_heads=" << d.n_kv_heads << "\n";
    oss << "d_head=" << d.d_head << "\n";

    oss << "ffn_hidden=" << d.ffn_hidden << "\n";

    oss << "n_hyp_heads=" << d.n_hyp_heads << "\n";
    oss << "d_hyp=" << d.d_hyp << "\n";

    oss << "n_top_anchors=" << d.n_top_anchors << "\n";
    oss << "n_children=" << d.n_children << "\n";

    oss << "enable_t1=" << (d.enableT1 ? 1 : 0) << "\n";
    oss << "enable_t2=" << (d.enableT2 ? 1 : 0) << "\n";
    oss << "enable_t3=" << (d.enableT3 ? 1 : 0) << "\n";
    oss << "enable_t4=" << (d.enableT4 ? 1 : 0) << "\n";
    oss << "enable_t5=" << (d.enableT5 ? 1 : 0) << "\n";

    oss << "use_moe=" << (d.useMoe ? 1 : 0) << "\n";
    oss << "tied_embeddings=" << (d.tiedEmbeddings ? 1 : 0) << "\n";

    oss << "c_init=" << d.cInit << "\n";
    oss << "tau_att=" << d.tauAtt << "\n";
    oss << "tau_core=" << d.tauCore << "\n";
    oss << "tau_route=" << d.tauRoute << "\n";
    oss << "tau_head=" << d.tauHead << "\n";

    oss << "iso_delta=" << d.isoDelta << "\n";
    oss << "route_beta=" << d.routeBeta << "\n";
    oss << "rope_base=" << d.ropeBase << "\n";

    oss << "seed=" << d.seed << "\n";

    return writeFileString(dimsPath(modelDir), oss.str(), err);
}

bool readDimsFile(
    const std::string& modelDir,
    arch::ArchitectureDims& d,
    std::string& err
) {
    err.clear();

    if (modelDir.empty()) {
        err = "model dir is empty";
        return false;
    }

    std::string text;

    if (!readFileString(dimsPath(modelDir), text, err)) {
        return false;
    }

    std::istringstream iss(text);
    std::string line;

    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        line = trim(line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::size_t eq = line.find('=');

        if (eq == std::string::npos) {
            continue;
        }

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (key == "vocab") {
            d.vocab = parseIntValue(val, d.vocab);
        } else if (key == "d_model") {
            d.d_model = parseIntValue(val, d.d_model);
        } else if (key == "n_layers") {
            d.n_layers = parseIntValue(val, d.n_layers);
        } else if (key == "n_heads") {
            d.n_heads = parseIntValue(val, d.n_heads);
        } else if (key == "n_kv_heads") {
            d.n_kv_heads = parseIntValue(val, d.n_kv_heads);
        } else if (key == "d_head") {
            d.d_head = parseIntValue(val, d.d_head);
        } else if (key == "ffn_hidden") {
            d.ffn_hidden = parseIntValue(val, d.ffn_hidden);
        } else if (key == "n_hyp_heads") {
            d.n_hyp_heads = parseIntValue(val, d.n_hyp_heads);
        } else if (key == "d_hyp") {
            d.d_hyp = parseIntValue(val, d.d_hyp);
        } else if (key == "n_top_anchors") {
            d.n_top_anchors = parseIntValue(val, d.n_top_anchors);
        } else if (key == "n_children") {
            d.n_children = parseIntValue(val, d.n_children);
        } else if (key == "enable_t1") {
            d.enableT1 = parseBoolValue(val, d.enableT1);
        } else if (key == "enable_t2") {
            d.enableT2 = parseBoolValue(val, d.enableT2);
        } else if (key == "enable_t3") {
            d.enableT3 = parseBoolValue(val, d.enableT3);
        } else if (key == "enable_t4") {
            d.enableT4 = parseBoolValue(val, d.enableT4);
        } else if (key == "enable_t5") {
            d.enableT5 = parseBoolValue(val, d.enableT5);
        } else if (key == "use_moe") {
            d.useMoe = parseBoolValue(val, d.useMoe);
        } else if (key == "tied_embeddings") {
            d.tiedEmbeddings = parseBoolValue(val, d.tiedEmbeddings);
        } else if (key == "c_init") {
            d.cInit = parseFloatValue(val, d.cInit);
        } else if (key == "tau_att") {
            d.tauAtt = parseFloatValue(val, d.tauAtt);
        } else if (key == "tau_core") {
            d.tauCore = parseFloatValue(val, d.tauCore);
        } else if (key == "tau_route") {
            d.tauRoute = parseFloatValue(val, d.tauRoute);
        } else if (key == "tau_head") {
            d.tauHead = parseFloatValue(val, d.tauHead);
        } else if (key == "iso_delta") {
            d.isoDelta = parseFloatValue(val, d.isoDelta);
        } else if (key == "route_beta") {
            d.routeBeta = parseFloatValue(val, d.routeBeta);
        } else if (key == "rope_base") {
            d.ropeBase = parseFloatValue(val, d.ropeBase);
        } else if (key == "seed") {
            d.seed = parseUint64Value(val, d.seed);
        }
    }

    err.clear();
    return true;
}

void printUsage() {
    std::cout <<
        "nol_cli - training and inference runner\n"
        "\n"
        "Basic training:\n"
        "  ./build/nol_cli --train data.txt --model-dir out --epochs 1\n"
        "\n"
        "Generate:\n"
        "  ./build/nol_cli --model-dir out --load --generate \"prompt\" --max-new-tokens 32\n"
        "\n"
        "Resume:\n"
        "  ./build/nol_cli --train data.txt --model-dir out --resume --epochs 4\n"
        "\n"
        "Common options:\n"
        "  --train FILE              training text file, one document per line\n"
        "  --model-dir DIR           checkpoint directory\n"
        "  --load                    load checkpoint before generate/evaluate\n"
        "  --resume                  alias for --load, intended for continuing training\n"
        "  --generate TEXT           generate from prompt\n"
        "  --max-new-tokens N        maximum generated tokens\n"
        "  --temperature F           sampling temperature, 0 means greedy\n"
        "  --top-k N                 top-k sampling\n"
        "  --top-p F                 nucleus sampling\n"
        "\n"
        "Training options:\n"
        "  --seq-len N               sequence length\n"
        "  --epochs N                number of epochs\n"
        "  --total-steps N           total steps for cosine schedule\n"
        "  --warmup-steps N          warmup steps\n"
        "  --lr F                    learning rate\n"
        "  --min-lr F                minimum learning rate\n"
        "  --weight-decay F          AdamW weight decay\n"
        "  --beta1 F                 Adam beta1\n"
        "  --beta2 F                 Adam beta2\n"
        "  --eps F                   Adam epsilon\n"
        "  --max-grad-norm F         gradient clipping norm\n"
        "  --seed N                  random seed\n"
        "  --save-every N            save checkpoint every N steps\n"
        "  --schedule TYPE           cosine or constant\n"
        "\n"
        "Architecture options:\n"
        "  --vocab N                 vocabulary size\n"
        "  --d-model N               model dimension\n"
        "  --n-layers N              number of blocks\n"
        "  --n-heads N               attention heads\n"
        "  --n-kv-heads N            KV heads for GQA\n"
        "  --d-head N                head dimension\n"
        "  --ffn-hidden N            FFN hidden size\n"
        "  --n-hyp-heads N           hyperbolic attention heads\n"
        "  --d-hyp N                 hyperbolic dimension\n"
        "  --n-top-anchors N         top-level anchors / MoE experts\n"
        "  --n-children N            children per top anchor for CGL\n"
        "\n"
        "Hyperbolic / routing options:\n"
        "  --c-init F                curvature init\n"
        "  --tau-att F               HGA temperature\n"
        "  --tau-core F              CGL temperature\n"
        "  --tau-route F             TIG temperature\n"
        "  --tau-head F              hyperbolic head temperature\n"
        "  --iso-delta F             topic isolation radius\n"
        "  --route-beta F            routing distance strength\n"
        "  --rope-base F             RoPE base\n"
        "\n"
        "Ablation flags:\n"
        "  --no-t1                   disable HGA\n"
        "  --no-t2                   disable gated fusion\n"
        "  --no-t3                   disable core-gravity layer contribution\n"
        "  --no-t4                   disable topic isolation gating / MoE routing\n"
        "  --no-t5                   disable hyperbolic output head\n"
        "  --no-moe                  disable MoE experts\n"
        "  --no-tied                 untie embedding and output head\n"
        "  --no-checkpoint           do not write checkpoint files\n"
        "\n"
        "Utility:\n"
        "  --print-config            print resolved config and exit\n"
        "  --help                    show this help\n";
}

void printConfig(
    const arch::ArchitectureDims& d,
    const tr::Config& c,
    const std::string& modelDir,
    const std::string& trainPath,
    bool doGenerate,
    const std::string& prompt
) {
    std::cout << "runtime config\n";
    std::cout << "  model_dir=" << (modelDir.empty() ? "<none>" : modelDir) << "\n";
    std::cout << "  train_path=" << (trainPath.empty() ? "<none>" : trainPath) << "\n";
    std::cout << "  generate=" << (doGenerate ? "yes" : "no") << "\n";
    std::cout << "  prompt=\"" << prompt << "\"\n";

    std::cout << "\narchitecture\n";
    std::cout << "  vocab=" << d.vocab << "\n";
    std::cout << "  d_model=" << d.d_model << "\n";
    std::cout << "  n_layers=" << d.n_layers << "\n";
    std::cout << "  n_heads=" << d.n_heads << "\n";
    std::cout << "  n_kv_heads=" << d.n_kv_heads << "\n";
    std::cout << "  d_head=" << d.d_head << "\n";
    std::cout << "  ffn_hidden=" << d.ffn_hidden << "\n";
    std::cout << "  n_hyp_heads=" << d.n_hyp_heads << "\n";
    std::cout << "  d_hyp=" << d.d_hyp << "\n";
    std::cout << "  n_top_anchors=" << d.n_top_anchors << "\n";
    std::cout << "  n_children=" << d.n_children << "\n";

    std::cout << "  enable_t1=" << (d.enableT1 ? 1 : 0) << "\n";
    std::cout << "  enable_t2=" << (d.enableT2 ? 1 : 0) << "\n";
    std::cout << "  enable_t3=" << (d.enableT3 ? 1 : 0) << "\n";
    std::cout << "  enable_t4=" << (d.enableT4 ? 1 : 0) << "\n";
    std::cout << "  enable_t5=" << (d.enableT5 ? 1 : 0) << "\n";
    std::cout << "  use_moe=" << (d.useMoe ? 1 : 0) << "\n";
    std::cout << "  tied_embeddings=" << (d.tiedEmbeddings ? 1 : 0) << "\n";

    std::cout << "  c_init=" << d.cInit << "\n";
    std::cout << "  tau_att=" << d.tauAtt << "\n";
    std::cout << "  tau_core=" << d.tauCore << "\n";
    std::cout << "  tau_route=" << d.tauRoute << "\n";
    std::cout << "  tau_head=" << d.tauHead << "\n";
    std::cout << "  iso_delta=" << d.isoDelta << "\n";
    std::cout << "  route_beta=" << d.routeBeta << "\n";
    std::cout << "  rope_base=" << d.ropeBase << "\n";
    std::cout << "  seed=" << d.seed << "\n";

    std::cout << "\ntraining\n";
    std::cout << "  seq_len=" << c.seqLen << "\n";
    std::cout << "  epochs=" << c.epochs << "\n";
    std::cout << "  total_steps=" << c.totalSteps << "\n";
    std::cout << "  warmup_steps=" << c.warmupSteps << "\n";
    std::cout << "  lr=" << c.learningRate << "\n";
    std::cout << "  min_lr=" << c.minLr << "\n";
    std::cout << "  weight_decay=" << c.weightDecay << "\n";
    std::cout << "  beta1=" << c.beta1 << "\n";
    std::cout << "  beta2=" << c.beta2 << "\n";
    std::cout << "  eps=" << c.eps << "\n";
    std::cout << "  max_grad_norm=" << c.maxGradNorm << "\n";
    std::cout << "  save_every=" << c.saveEverySteps << "\n";
    std::cout << "  schedule=" << c.scheduleType << "\n";
    std::cout << "  vocab_size=" << c.vocabSize << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
            printUsage();
            return 0;
        }

        std::string err;

        std::string trainPath = optionValue(argc, argv, "--train", "");
        std::string modelDir = optionValue(argc, argv, "--model-dir", "model");
        std::string prompt = optionValue(argc, argv, "--generate", "");

        bool doGenerate = hasFlag(argc, argv, "--generate");
        bool doResume = hasFlag(argc, argv, "--resume");
        bool doLoad = hasFlag(argc, argv, "--load") || doResume;
        bool printCfg = hasFlag(argc, argv, "--print-config");

        if (hasFlag(argc, argv, "--no-checkpoint")) {
            modelDir.clear();
        }

        if (doLoad && modelDir.empty()) {
            std::cerr << "error: --load/--resume requires --model-dir\n";
            return 1;
        }

        arch::ArchitectureDims dims;
        bool dimsLoaded = false;

        if (doLoad) {
            dimsLoaded = readDimsFile(modelDir, dims, err);

            if (!dimsLoaded) {
                err.clear();
            }
        }

        dims.vocab = optionInt(argc, argv, "--vocab", dims.vocab);
        dims.d_model = optionInt(argc, argv, "--d-model", dims.d_model);
        dims.n_layers = optionInt(argc, argv, "--n-layers", dims.n_layers);

        dims.n_heads = optionInt(argc, argv, "--n-heads", dims.n_heads);
        dims.n_kv_heads = optionInt(argc, argv, "--n-kv-heads", dims.n_kv_heads);
        dims.d_head = optionInt(argc, argv, "--d-head", dims.d_head);

        dims.ffn_hidden = optionInt(argc, argv, "--ffn-hidden", dims.ffn_hidden);

        dims.n_hyp_heads = optionInt(argc, argv, "--n-hyp-heads", dims.n_hyp_heads);
        dims.d_hyp = optionInt(argc, argv, "--d-hyp", dims.d_hyp);

        dims.n_top_anchors = optionInt(argc, argv, "--n-top-anchors", dims.n_top_anchors);
        dims.n_children = optionInt(argc, argv, "--n-children", dims.n_children);

        dims.cInit = optionFloat(argc, argv, "--c-init", dims.cInit);
        dims.tauAtt = optionFloat(argc, argv, "--tau-att", dims.tauAtt);
        dims.tauCore = optionFloat(argc, argv, "--tau-core", dims.tauCore);
        dims.tauRoute = optionFloat(argc, argv, "--tau-route", dims.tauRoute);
        dims.tauHead = optionFloat(argc, argv, "--tau-head", dims.tauHead);

        dims.isoDelta = optionFloat(argc, argv, "--iso-delta", dims.isoDelta);
        dims.routeBeta = optionFloat(argc, argv, "--route-beta", dims.routeBeta);
        dims.ropeBase = optionFloat(argc, argv, "--rope-base", dims.ropeBase);

        dims.seed = optionUint64(argc, argv, "--seed", dims.seed);

        if (hasFlag(argc, argv, "--no-t1")) {
            dims.enableT1 = false;
        }

        if (hasFlag(argc, argv, "--no-t2")) {
            dims.enableT2 = false;
        }

        if (hasFlag(argc, argv, "--no-t3")) {
            dims.enableT3 = false;
        }

        if (hasFlag(argc, argv, "--no-t4")) {
            dims.enableT4 = false;
        }

        if (hasFlag(argc, argv, "--no-t5")) {
            dims.enableT5 = false;
        }

        if (hasFlag(argc, argv, "--no-moe")) {
            dims.useMoe = false;
        }

        if (hasFlag(argc, argv, "--no-tied")) {
            dims.tiedEmbeddings = false;
        }

        Tokenizer tok = Tokenizer::makeByte();

        bool vocabExplicit = dimsLoaded || hasFlag(argc, argv, "--vocab");

        if (!vocabExplicit || dims.vocab <= 0) {
            int maxId = 255;

            maxId = std::max(maxId, tok.bosId());
            maxId = std::max(maxId, tok.eosId());
            maxId = std::max(maxId, tok.padId());

            dims.vocab = maxId + 1;
        }

        tr::Config cfg;

        cfg.seqLen = optionInt(argc, argv, "--seq-len", cfg.seqLen);
        cfg.epochs = optionInt(argc, argv, "--epochs", cfg.epochs);

        cfg.totalSteps = optionInt(argc, argv, "--total-steps", cfg.totalSteps);
        cfg.warmupSteps = optionInt(argc, argv, "--warmup-steps", cfg.warmupSteps);

        cfg.learningRate = optionFloat(argc, argv, "--lr", cfg.learningRate);
        cfg.minLr = optionFloat(argc, argv, "--min-lr", cfg.minLr);

        cfg.weightDecay = optionFloat(argc, argv, "--weight-decay", cfg.weightDecay);
        cfg.beta1 = optionFloat(argc, argv, "--beta1", cfg.beta1);
        cfg.beta2 = optionFloat(argc, argv, "--beta2", cfg.beta2);
        cfg.eps = optionFloat(argc, argv, "--eps", cfg.eps);
        cfg.maxGradNorm = optionFloat(argc, argv, "--max-grad-norm", cfg.maxGradNorm);

        cfg.seed = optionUint64(argc, argv, "--seed", cfg.seed);
        cfg.saveEverySteps = optionInt(argc, argv, "--save-every", cfg.saveEverySteps);
        cfg.scheduleType = optionValue(argc, argv, "--schedule", cfg.scheduleType);

        if (printCfg) {
            printConfig(dims, cfg, modelDir, trainPath, doGenerate, prompt);
            return 0;
        }

        arch::NolModelParams model = arch::createRandomModel(dims);

        cfg.vocabSize = model.vocab;

        std::unique_ptr<tr::Trainer> trainer =
            std::make_unique<tr::Trainer>(
                std::move(model),
                std::move(tok),
                cfg
            );

        if (doLoad) {
            if (!trainer->loadCheckpoint(modelDir, err)) {
                std::cerr << "error loading checkpoint: " << err << "\n";
                return 1;
            }

            std::cout <<
                "checkpoint loaded: step=" << trainer->globalStep() <<
                " epoch=" << trainer->epoch() <<
                " loss=" << trainer->lastLoss() <<
                "\n";
        }

        if (!trainPath.empty()) {
            std::cout <<
                "training start: file=" << trainPath <<
                " model_dir=" << (modelDir.empty() ? "<none>" : modelDir) <<
                " epochs=" << cfg.epochs <<
                " seq_len=" << cfg.seqLen <<
                "\n";

            if (!trainer->trainFile(trainPath, modelDir, err)) {
                std::cerr << "training failed: " << err << "\n";
                return 1;
            }

            if (!modelDir.empty()) {
                if (!writeDimsFile(modelDir, dims, err)) {
                    std::cerr << "warning: failed writing dims file: " << err << "\n";
                    err.clear();
                }
            }

            std::cout <<
                "training finished: step=" << trainer->globalStep() <<
                " epoch=" << trainer->epoch() <<
                " loss=" << trainer->lastLoss() <<
                "\n";
        }

        if (doGenerate) {
            int maxNewTokens = optionInt(argc, argv, "--max-new-tokens", 32);
            float temperature = optionFloat(argc, argv, "--temperature", 0.7f);
            int topK = optionInt(argc, argv, "--top-k", 50);
            float topP = optionFloat(argc, argv, "--top-p", 0.95f);

            std::string out = trainer->generate(
                prompt,
                maxNewTokens,
                temperature,
                topK,
                topP,
                err
            );

            if (!err.empty()) {
                std::cerr << "generation warning: " << err << "\n";
                err.clear();
            }

            std::cout << out << "\n";
        }

        if (trainPath.empty() && !doGenerate) {
            printUsage();
            return 0;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "fatal error: " << ex.what() << "\n";
        return 1;
    }
}
