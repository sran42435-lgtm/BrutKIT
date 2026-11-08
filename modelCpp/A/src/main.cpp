#include "../include/config/config.hpp"
#include "../include/core/tensor.hpp"
#include "../include/model/nol_model.hpp"
#include "../include/io/io.hpp"
#include "../include/train/training.hpp"
#include "../include/inference/inference.hpp"
#include "../include/token/tokenizer.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <fstream>

using namespace nol::config;
using namespace nol::core;
using namespace nol::model;
using namespace nol::io;
using namespace nol::train;
using namespace nol::inference;
using namespace nol::token;

// ==========================================
// GLOBAL STATE
// ==========================================
static NolConfig g_cfg;
static NolModel* g_model = nullptr;
static TrainingStatus g_status;
static std::string g_current_model_name;
static volatile bool g_interrupted = false;

// Tambahan: Tokenizer dan Dataset global
static BPETokenizer* g_tokenizer = nullptr;
static Dataset* g_dataset = nullptr;
static bool g_vocab_ready = false;

void signal_handler(int sig) {
    g_interrupted = true;
    std::cout << "\n⚠️  Interrupt received. Saving checkpoint...\n";
    if (g_model && !g_current_model_name.empty()) {
        save_safetensors(*g_model, g_current_model_name + ".safetensors");
        CheckpointMeta meta;
        meta.remaining_epochs = g_status.remaining_epochs;
        meta.completed_steps = g_status.completed_steps;
        meta.last_loss = g_status.last_loss;
        meta.last_acc = g_status.last_acc;
        meta.last_ppl = g_status.last_ppl;
        meta.context_proficiency = g_status.context_proficiency;
        save_checkpoint_meta(meta, g_current_model_name + ".log");
        std::cout << "💾 Checkpoint saved.\n";
    }
}

// ==========================================
// CLI HELPERS
// ==========================================

void print_lobby_header() {
    std::cout << "\n\033[1;36m";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║          [" << std::setw(20) << std::left << g_current_model_name
              << "] PLAYGROUND LOBBY      ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n\033[0m";
    std::cout << "║  Epochs Remaining : " << std::setw(6) << std::right << g_status.remaining_epochs
              << "                        ║\n";
    std::cout << "║  Last Loss        : " << std::fixed << std::setprecision(4) << std::setw(8)
              << g_status.last_loss << "                    ║\n";
    std::cout << "║  Last ACC         : " << std::setprecision(4) << std::setw(8)
              << g_status.last_acc << "                    ║\n";
    std::cout << "║  Last PPL         : " << std::setprecision(2) << std::setw(8)
              << g_status.last_ppl << "                    ║\n";
    std::cout << "║  Context Prof.    : " << std::setprecision(4) << std::setw(8)
              << g_status.context_proficiency << "                    ║\n";
    std::cout << "║  Vocab Ready      : " << std::setw(8) << std::left
              << (g_vocab_ready ? "YES" : "NO") << "                        ║\n";
    std::cout << "║  Dataset Loaded   : " << std::setw(8) << std::left
              << (g_dataset ? std::to_string(g_dataset->size()) : "0")
              << " samples                  ║\n";
    std::cout << "\033[1;36m╠═══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Commands:                                                ║\n";
    std::cout << "║  /load <file.txt>                  Load dataset from file ║\n";
    std::cout << "║  /build-vocab <file.txt>           Build vocab from corpus║\n";
    std::cout << "║  /training [--epochs N] [--lr F]   Train model            ║\n";
    std::cout << "║  /prompt [--top-k N] [--temp F]    Generate text          ║\n";
    std::cout << "║  /switch --model \"name\"            Switch model           ║\n";
    std::cout << "║  /resume                           Resume training        ║\n";
    std::cout << "║  /exit                             Exit                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\033[0m\n";
}

std::vector<std::string> parse_args(const std::string& line) {
    std::vector<std::string> args;
    std::istringstream iss(line);
    std::string arg;
    while (iss >> arg) args.push_back(arg);
    return args;
}

float get_flag_float(const std::vector<std::string>& args, const std::string& flag, float def) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) {
            try { return std::stof(args[i + 1]); } catch (...) { return def; }
        }
    }
    return def;
}

int get_flag_int(const std::vector<std::string>& args, const std::string& flag, int def) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) {
            try { return std::stoi(args[i + 1]); } catch (...) { return def; }
        }
    }
    return def;
}

std::string get_flag_str(const std::vector<std::string>& args, const std::string& flag, const std::string& def) {
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == flag) return args[i + 1];
    }
    return def;
}

// ==========================================
// COMMAND HANDLERS
// ==========================================

void handle_build_vocab(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cout << "Usage: /build-vocab <corpus.txt>\n";
        return;
    }
    std::string corpus_path = args[1];

    // Baca corpus
    std::ifstream file(corpus_path);
    if (!file.is_open()) {
        std::cout << "Error: Cannot open file '" << corpus_path << "'\n";
        return;
    }

    std::vector<std::string> texts;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) texts.push_back(line);
    }
    file.close();

    if (texts.empty()) {
        std::cout << "Error: File is empty.\n";
        return;
    }

    // Build tokenizer
    if (!g_tokenizer) g_tokenizer = new BPETokenizer();
    g_tokenizer->build_vocab(texts, g_cfg.vocab_size);

    // Save vocab
    std::string vocab_path = g_current_model_name + ".vocab";
    g_tokenizer->save_vocab(vocab_path);

    g_vocab_ready = true;
    std::cout << "✅ Vocab built: " << g_tokenizer->vocab_size() << " tokens, saved to " << vocab_path << "\n";
}

void handle_load(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        std::cout << "Usage: /load <dataset.txt>\n";
        return;
    }
    std::string dataset_path = args[1];

    // Cek apakah vocab sudah siap
    if (!g_vocab_ready) {
        std::cout << "Vocab belum siap. Mencoba load dari " << g_current_model_name << ".vocab ...\n";
        if (!g_tokenizer) g_tokenizer = new BPETokenizer();
        std::string vocab_path = g_current_model_name + ".vocab";
        if (!g_tokenizer->load_vocab(vocab_path)) {
            std::cout << "Error: Vocab tidak ditemukan. Jalankan /build-vocab <corpus.txt> terlebih dahulu.\n";
            return;
        }
        g_vocab_ready = true;
    }

    // Load dataset
    if (!g_dataset) g_dataset = new Dataset();
    int count = g_dataset->load_from_text(dataset_path, *g_tokenizer, g_cfg.max_seq_len);

    if (count > 0) {
        std::cout << "✅ Dataset loaded: " << count << " samples dari " << dataset_path << "\n";
    } else {
        std::cout << "Error: Gagal load dataset.\n";
    }
}

void handle_training(const std::vector<std::string>& args) {
    NolConfig train_cfg = g_cfg;
    train_cfg.epochs = get_flag_int(args, "--epochs", g_cfg.epochs);
    train_cfg.lr = get_flag_float(args, "--lr", g_cfg.lr);
    train_cfg.batch_size = get_flag_int(args, "--batch-size", g_cfg.batch_size);

    Trainer trainer(*g_model, train_cfg);
    trainer.status = g_status;

    // Gunakan dataset yang sudah di-load jika ada, otherwise generate random
    Dataset ds;
    if (g_dataset && g_dataset->size() > 0) {
        // Copy dataset yang sudah di-load
        ds.samples = g_dataset->samples;
        ds.raw_texts = g_dataset->raw_texts;
        ds.seq_len = g_dataset->seq_len;
        std::cout << "  Using loaded dataset (" << ds.size() << " samples)\n";
    } else {
        ds.generate_random(100, train_cfg.max_seq_len, train_cfg.vocab_size, train_cfg.rng_seed);
        std::cout << "  No dataset loaded, using random data (" << ds.size() << " samples)\n";
    }

    trainer.train(ds, g_current_model_name, g_cfg.epochs - g_status.remaining_epochs);

    g_status = trainer.status;
}

void handle_prompt(const std::vector<std::string>& args) {
    SamplingParams params;
    params.top_k = get_flag_int(args, "--top-k", 50);
    params.top_p = get_flag_float(args, "--top-p", 0.9f);
    params.temperature = get_flag_float(args, "--temp", 1.0f);
    params.max_tokens = get_flag_int(args, "--max-tokens", 64);

    std::cout << "Enter prompt (text or token IDs separated by space):\n> ";
    std::string input;
    std::getline(std::cin, input);

    if (input.empty()) {
        std::cout << "Empty prompt.\n";
        return;
    }

    InferenceEngine engine(*g_model, g_cfg);

    // Cek apakah input berupa angka (token IDs) atau teks
    bool is_token_ids = true;
    std::istringstream check_iss(input);
    std::string word;
    while (check_iss >> word) {
        bool all_digits = true;
        for (char c : word) {
            if (!std::isdigit(c)) { all_digits = false; break; }
        }
        if (!all_digits) { is_token_ids = false; break; }
    }

    if (is_token_ids) {
        // Parse sebagai token IDs
        std::vector<int> prompt;
        std::istringstream iss(input);
        int tok;
        while (iss >> tok) prompt.push_back(tok);

        if (prompt.empty()) prompt.push_back(0);

        std::vector<int> generated = engine.generate(prompt, params);

        std::cout << "\nGenerated token IDs: ";
        for (int t : generated) std::cout << t << " ";
        std::cout << "\n";

        // Decode jika tokenizer tersedia
        if (g_vocab_ready && g_tokenizer) {
            std::string decoded = g_tokenizer->decode(generated);
            std::cout << "Decoded text: " << decoded << "\n";
        }

        ProficiencyMetrics metrics = engine.compute_proficiency(generated, prompt);
        g_status.context_proficiency = metrics.overall_score;
        std::cout << "Proficiency: " << std::fixed << std::setprecision(4) << metrics.overall_score
                  << " (PPL=" << std::setprecision(2) << metrics.perplexity
                  << " Mix=" << std::setprecision(4) << metrics.topic_mixing_rate << ")\n";
    } else {
        // Parse sebagai teks
        if (!g_vocab_ready || !g_tokenizer) {
            std::cout << "Error: Vocab belum siap. Jalankan /build-vocab atau /load terlebih dahulu.\n";
            return;
        }

        std::string generated_text = engine.generate_text(input, *g_tokenizer, params);
        std::cout << "\nGenerated text: " << generated_text << "\n";

        // Compute proficiency
        std::vector<int> prompt_ids = g_tokenizer->encode(input, false, false);
        std::vector<int> gen_ids = g_tokenizer->encode(generated_text, false, false);
        ProficiencyMetrics metrics = engine.compute_proficiency(gen_ids, prompt_ids);
        g_status.context_proficiency = metrics.overall_score;
        std::cout << "Proficiency: " << std::fixed << std::setprecision(4) << metrics.overall_score
                  << " (PPL=" << std::setprecision(2) << metrics.perplexity
                  << " Mix=" << std::setprecision(4) << metrics.topic_mixing_rate << ")\n";
    }
}

void handle_switch(const std::vector<std::string>& args) {
    std::string new_model = get_flag_str(args, "--model", "");
    if (new_model.empty()) {
        std::cout << "Usage: /switch --model \"model-name\"\n";
        return;
    }

    if (g_model) {
        save_safetensors(*g_model, g_current_model_name + ".safetensors");
        delete g_model;
        g_model = nullptr;
    }

    g_model = new NolModel(g_cfg);
    std::string path = new_model + ".safetensors";
    if (!load_safetensors(*g_model, path)) {
        std::cout << "Error: Model '" << new_model << "' not found.\n";
        delete g_model;
        g_model = nullptr;
        return;
    }

    g_current_model_name = new_model;

    CheckpointMeta meta;
    if (load_checkpoint_meta(meta, new_model + ".log")) {
        g_status.remaining_epochs = meta.remaining_epochs;
        g_status.completed_steps = meta.completed_steps;
        g_status.last_loss = meta.last_loss;
        g_status.last_acc = meta.last_acc;
        g_status.last_ppl = meta.last_ppl;
        g_status.context_proficiency = meta.context_proficiency;
    }

    // Coba load vocab untuk model baru
    if (g_tokenizer) {
        std::string vocab_path = new_model + ".vocab";
        if (g_tokenizer->load_vocab(vocab_path)) {
            g_vocab_ready = true;
        } else {
            g_vocab_ready = false;
        }
    }

    std::cout << "Switched to model: " << new_model << "\n";
}

void handle_resume() {
    if (g_status.remaining_epochs <= 0) {
        std::cout << "No remaining epochs to resume.\n";
        return;
    }

    handle_training({"/training", "--epochs", std::to_string(g_status.remaining_epochs)});
}

// ==========================================
// MAIN
// ==========================================

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    g_cfg = get_default_config();

    std::string model_name;
    std::string resume_model;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-name" && i + 1 < argc) {
            model_name = argv[++i];
        } else if (arg == "--resume" && i + 1 < argc) {
            resume_model = argv[++i];
        }
    }

    // Mode: Create new model
    if (!model_name.empty()) {
        print_config(g_cfg);
        std::cout << "Create model '" << model_name << "' with above config? (y/n): ";
        std::string confirm;
        std::getline(std::cin, confirm);

        if (confirm == "y" || confirm == "Y") {
            create_new_model(g_cfg, model_name);
            g_current_model_name = model_name;
            g_model = new NolModel(g_cfg);
            load_safetensors(*g_model, model_name + ".safetensors");

            CheckpointMeta meta;
            load_checkpoint_meta(meta, model_name + ".log");
            g_status.remaining_epochs = meta.remaining_epochs;
            g_status.completed_steps = meta.completed_steps;

            // Coba load vocab yang sudah ada
            g_tokenizer = new BPETokenizer();
            std::string vocab_path = model_name + ".vocab";
            if (g_tokenizer->load_vocab(vocab_path)) {
                g_vocab_ready = true;
                std::cout << "Vocab loaded: " << g_tokenizer->vocab_size() << " tokens\n";
            }
        } else {
            std::cout << "Cancelled.\n";
            return 0;
        }
    }
    // Mode: Resume
    else if (!resume_model.empty()) {
        g_current_model_name = resume_model;
        g_model = new NolModel(g_cfg);

        if (!load_safetensors(*g_model, resume_model + ".safetensors")) {
            std::cerr << "Error: Model '" << resume_model << "' not found.\n";
            delete g_model;
            return 1;
        }

        CheckpointMeta meta;
        if (load_checkpoint_meta(meta, resume_model + ".log")) {
            g_status.remaining_epochs = meta.remaining_epochs;
            g_status.completed_steps = meta.completed_steps;
            g_status.last_loss = meta.last_loss;
            g_status.last_acc = meta.last_acc;
            g_status.last_ppl = meta.last_ppl;
            g_status.context_proficiency = meta.context_proficiency;
        }

        // Load vocab
        g_tokenizer = new BPETokenizer();
        std::string vocab_path = resume_model + ".vocab";
        if (g_tokenizer->load_vocab(vocab_path)) {
            g_vocab_ready = true;
        }

        std::cout << "Resuming model: " << resume_model << "\n";
        handle_resume();
        print_lobby_header();
    }
    // Mode: No arguments
    else {
        std::cout << "Usage:\n";
        std::cout << "  ./proyek --model-name \"name\"   Create new model\n";
        std::cout << "  ./proyek --resume \"name\"       Resume training\n";
        return 0;
    }

    // Playground Lobby Loop
    while (!g_interrupted) {
        print_lobby_header();
        std::cout << "> ";
        std::string line;
        if (!std::getline(std::cin, line)) break;

        auto args = parse_args(line);
        if (args.empty()) continue;

        std::string cmd = args[0];

        if (cmd == "/load") {
            handle_load(args);
        } else if (cmd == "/build-vocab") {
            handle_build_vocab(args);
        } else if (cmd == "/training") {
            handle_training(args);
        } else if (cmd == "/prompt") {
            handle_prompt(args);
        } else if (cmd == "/switch") {
            handle_switch(args);
        } else if (cmd == "/resume") {
            handle_resume();
        } else if (cmd == "/exit") {
            std::cout << "Exiting...\n";
            break;
        } else {
            std::cout << "Unknown command: " << cmd << "\n";
        }
    }

    // Cleanup
    if (g_model) {
        save_safetensors(*g_model, g_current_model_name + ".safetensors");
        CheckpointMeta meta;
        meta.remaining_epochs = g_status.remaining_epochs;
        meta.completed_steps = g_status.completed_steps;
        meta.last_loss = g_status.last_loss;
        meta.last_acc = g_status.last_acc;
        meta.last_ppl = g_status.last_ppl;
        meta.context_proficiency = g_status.context_proficiency;
        save_checkpoint_meta(meta, g_current_model_name + ".log");
        delete g_model;
        g_model = nullptr;
    }

    if (g_tokenizer) { delete g_tokenizer; g_tokenizer = nullptr; }
    if (g_dataset) { delete g_dataset; g_dataset = nullptr; }

    return 0;
}
