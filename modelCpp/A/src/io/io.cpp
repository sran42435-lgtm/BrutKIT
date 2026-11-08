#include "../../include/io/io.hpp"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

namespace nol {
namespace io {

// ==========================================
// HELPER: Tensor registry untuk serialisasi
// ==========================================

struct TensorEntry {
    std::string name;
    const float* data;
    size_t numel;
    std::vector<size_t> shape;
};

static void collect_tensors(const NolModel& model, std::vector<TensorEntry>& entries) {
    entries.clear();

    entries.push_back({"embed_E", model.embed_E.fp32_data(), model.embed_E.numel, model.embed_E.shape});
    entries.push_back({"embed_T", model.embed_T.fp32_data(), model.embed_T.numel, model.embed_T.shape});

    for (int l = 0; l < model.cfg.n_layers; ++l) {
        const auto& bp = model.blocks[l];
        std::string prefix = "block" + std::to_string(l) + ".";

        entries.push_back({prefix + "rmsnorm1_gamma", bp.rmsnorm1_gamma.fp32_data(), bp.rmsnorm1_gamma.numel, bp.rmsnorm1_gamma.shape});
        entries.push_back({prefix + "W_q", bp.W_q.fp32_data(), bp.W_q.numel, bp.W_q.shape});
        entries.push_back({prefix + "W_k", bp.W_k.fp32_data(), bp.W_k.numel, bp.W_k.shape});
        entries.push_back({prefix + "W_v", bp.W_v.fp32_data(), bp.W_v.numel, bp.W_v.shape});
        entries.push_back({prefix + "W_o", bp.W_o.fp32_data(), bp.W_o.numel, bp.W_o.shape});
        entries.push_back({prefix + "W_q_hyp", bp.W_q_hyp.fp32_data(), bp.W_q_hyp.numel, bp.W_q_hyp.shape});
        entries.push_back({prefix + "W_k_hyp", bp.W_k_hyp.fp32_data(), bp.W_k_hyp.numel, bp.W_k_hyp.shape});
        entries.push_back({prefix + "W_v_hyp", bp.W_v_hyp.fp32_data(), bp.W_v_hyp.numel, bp.W_v_hyp.shape});
        entries.push_back({prefix + "W_g", bp.W_g.fp32_data(), bp.W_g.numel, bp.W_g.shape});
        entries.push_back({prefix + "W_u", bp.W_u.fp32_data(), bp.W_u.numel, bp.W_u.shape});
        entries.push_back({prefix + "W_z", bp.W_z.fp32_data(), bp.W_z.numel, bp.W_z.shape});
        entries.push_back({prefix + "W_c", bp.W_c.fp32_data(), bp.W_c.numel, bp.W_c.shape});
        entries.push_back({prefix + "anchors_l1", bp.anchors_l1.fp32_data(), bp.anchors_l1.numel, bp.anchors_l1.shape});
        entries.push_back({prefix + "anchors_l2", bp.anchors_l2.fp32_data(), bp.anchors_l2.numel, bp.anchors_l2.shape});
        entries.push_back({prefix + "W_r", bp.W_r.fp32_data(), bp.W_r.numel, bp.W_r.shape});
        entries.push_back({prefix + "rmsnorm2_gamma", bp.rmsnorm2_gamma.fp32_data(), bp.rmsnorm2_gamma.numel, bp.rmsnorm2_gamma.shape});
        entries.push_back({prefix + "W1_shared", bp.W1_shared.fp32_data(), bp.W1_shared.numel, bp.W1_shared.shape});
        entries.push_back({prefix + "W2_shared", bp.W2_shared.fp32_data(), bp.W2_shared.numel, bp.W2_shared.shape});
        entries.push_back({prefix + "W3_shared", bp.W3_shared.fp32_data(), bp.W3_shared.numel, bp.W3_shared.shape});

        for (int e = 0; e < model.cfg.n_top_anchors; ++e) {
            std::string ep = prefix + "expert" + std::to_string(e) + ".";
            entries.push_back({ep + "W1", bp.W1_expert[e].fp32_data(), bp.W1_expert[e].numel, bp.W1_expert[e].shape});
            entries.push_back({ep + "W2", bp.W2_expert[e].fp32_data(), bp.W2_expert[e].numel, bp.W2_expert[e].shape});
            entries.push_back({ep + "W3", bp.W3_expert[e].fp32_data(), bp.W3_expert[e].numel, bp.W3_expert[e].shape});
        }
    }

    entries.push_back({"rmsnorm_final_gamma", model.rmsnorm_final_gamma.fp32_data(), model.rmsnorm_final_gamma.numel, model.rmsnorm_final_gamma.shape});
    entries.push_back({"H_v", model.H_v.fp32_data(), model.H_v.numel, model.H_v.shape});

    // Scalar parameters disimpan sebagai tensor 1-elemen
    // curvature, lambda_h, gamma per block, lambda_tilde per block, beta per block
    // Disimpan terpisah di metadata JSON, bukan di binary data
}

// Build JSON header untuk safetensors
static std::string build_json_header(const std::vector<TensorEntry>& entries,
                                     const NolModel& model) {
    std::ostringstream json;
    json << "{";

    // Tensor entries
    size_t offset = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) json << ",";
        json << "\"" << entries[i].name << "\":{";
        json << "\"dtype\":\"F32\",";
        json << "\"shape\":[";
        for (size_t s = 0; s < entries[i].shape.size(); ++s) {
            if (s > 0) json << ",";
            json << entries[i].shape[s];
        }
        json << "],";
        json << "\"data_offsets\":[" << offset << "," << offset + entries[i].numel * 4 << "]";
        json << "}";
        offset += entries[i].numel * 4;
    }

    // Metadata scalars
    json << ",\"__metadata__\":{";
    json << "\"vocab_size\":\"" << model.cfg.vocab_size << "\"";
    json << ",\"d_model\":\"" << model.cfg.d_model << "\"";
    json << ",\"n_layers\":\"" << model.cfg.n_layers << "\"";
    json << ",\"curvature\":\"" << model.curvature << "\"";
    json << ",\"lambda_h\":\"" << model.lambda_h << "\"";

    for (int l = 0; l < model.cfg.n_layers; ++l) {
        std::string p = "block" + std::to_string(l) + "_";
        json << ",\"" << p << "gamma\":\"" << model.blocks[l].gamma << "\"";
        json << ",\"" << p << "lambda_tilde\":\"" << model.blocks[l].lambda_tilde << "\"";
        json << ",\"" << p << "beta\":\"" << model.blocks[l].beta << "\"";
    }

    json << "}}";
    return json.str();
}

// ==========================================
// SAVE / LOAD SAFETENSORS
// ==========================================

bool save_safetensors(const NolModel& model, const std::string& path) {
    std::vector<TensorEntry> entries;
    collect_tensors(model, entries);

    std::string json = build_json_header(entries, model);
    uint64_t header_size = json.size();

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[IO ERROR] Tidak bisa membuka file untuk write: " << path << "\n";
        return false;
    }

    // Write header size (8 bytes, little-endian)
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(uint64_t));

    // Write JSON header
    file.write(json.c_str(), header_size);

    // Write binary tensor data
    for (const auto& entry : entries) {
        file.write(reinterpret_cast<const char*>(entry.data), entry.numel * sizeof(float));
    }

    file.close();
    std::cout << "[IO] Saved safetensors: " << path << " (" << entries.size() << " tensors)\n";
    return true;
}

bool load_safetensors(NolModel& model, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[IO ERROR] Tidak bisa membuka file untuk read: " << path << "\n";
        return false;
    }

    // Read header size
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));
    if (header_size == 0 || header_size > 100 * 1024 * 1024) {
        std::cerr << "[IO ERROR] Header size tidak valid: " << header_size << "\n";
        return false;
    }

    // Read JSON header
    std::string json(header_size, '\0');
    file.read(&json[0], header_size);

    // Collect expected tensors
    std::vector<TensorEntry> entries;
    collect_tensors(model, entries);

    // Parse offsets dari JSON (simple parser untuk format known)
    // Format: "name":{"dtype":"F32","shape":[...],"data_offsets":[start,end]}
    size_t binary_start = sizeof(uint64_t) + header_size;

    for (const auto& entry : entries) {
        // Cari tensor name di JSON
        std::string search = "\"" + entry.name + "\":{\"dtype\":\"F32\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) {
            std::cerr << "[IO WARNING] Tensor tidak ditemukan di file: " << entry.name << "\n";
            continue;
        }

        // Cari data_offsets
        size_t off_pos = json.find("\"data_offsets\":[", pos);
        if (off_pos == std::string::npos) continue;

        off_pos += strlen("\"data_offsets\":[");
        size_t comma_pos = json.find(",", off_pos);
        size_t end_pos = json.find("]", comma_pos);

        std::string start_str = json.substr(off_pos, comma_pos - off_pos);
        std::string end_str = json.substr(comma_pos + 1, end_pos - comma_pos - 1);

        size_t data_start = std::stoull(start_str);
        size_t data_end = std::stoull(end_str);
        size_t expected_bytes = entry.numel * sizeof(float);

        if (data_end - data_start != expected_bytes) {
            std::cerr << "[IO WARNING] Size mismatch untuk " << entry.name
                      << ": expected " << expected_bytes << ", got " << (data_end - data_start) << "\n";
            continue;
        }

        // Seek dan baca data
        file.seekg(binary_start + data_start);
        file.read(reinterpret_cast<char*>(const_cast<float*>(entry.data)), expected_bytes);
    }

    // Load scalar metadata dari JSON
    auto parse_scalar = [&](const std::string& key) -> float {
        std::string search = "\"" + key + "\":\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return 0.0f;
        pos += search.size();
        size_t end = json.find("\"", pos);
        return std::stof(json.substr(pos, end - pos));
    };

    model.curvature = parse_scalar("curvature");
    model.lambda_h = parse_scalar("lambda_h");

    for (int l = 0; l < model.cfg.n_layers; ++l) {
        std::string p = "block" + std::to_string(l) + "_";
        model.blocks[l].gamma = parse_scalar(p + "gamma");
        model.blocks[l].lambda_tilde = parse_scalar(p + "lambda_tilde");
        model.blocks[l].beta = parse_scalar(p + "beta");
    }

    file.close();
    model.rebuild_tied_embedding();
    std::cout << "[IO] Loaded safetensors: " << path << "\n";
    return true;
}

// ==========================================
// CHECKPOINT META (LOG FILE)
// ==========================================

bool save_checkpoint_meta(const CheckpointMeta& meta, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[IO ERROR] Tidak bisa menulis checkpoint log: " << path << "\n";
        return false;
    }

    file << "remaining_epochs=" << meta.remaining_epochs << "\n";
    file << "completed_steps=" << meta.completed_steps << "\n";
    file << "last_loss=" << std::setprecision(8) << meta.last_loss << "\n";
    file << "last_acc=" << std::setprecision(8) << meta.last_acc << "\n";
    file << "last_ppl=" << std::setprecision(8) << meta.last_ppl << "\n";
    file << "context_proficiency=" << std::setprecision(8) << meta.context_proficiency << "\n";

    file.close();
    return true;
}

bool load_checkpoint_meta(CheckpointMeta& meta, const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[IO WARNING] Checkpoint log tidak ditemukan: " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        if (key == "remaining_epochs") meta.remaining_epochs = std::stoi(val);
        else if (key == "completed_steps") meta.completed_steps = std::stoi(val);
        else if (key == "last_loss") meta.last_loss = std::stof(val);
        else if (key == "last_acc") meta.last_acc = std::stof(val);
        else if (key == "last_ppl") meta.last_ppl = std::stof(val);
        else if (key == "context_proficiency") meta.context_proficiency = std::stof(val);
    }

    file.close();
    return true;
}

// ==========================================
// EXPORT MODEL DENGAN NAMA KUSTOM
// ==========================================

bool export_model(const NolModel& model, const std::string& export_name) {
    std::string path = export_name + ".safetensors";
    bool ok = save_safetensors(model, path);
    if (ok) {
        std::cout << "[IO] Model exported: " << path << "\n";
    }
    return ok;
}

// ==========================================
// CREATE NEW MODEL
// ==========================================

bool create_new_model(const NolConfig& cfg, const std::string& model_name) {
    NolModel model(cfg);
    model.initialize_weights(cfg.rng_seed);

    std::string path = model_name + ".safetensors";
    bool ok = save_safetensors(model, path);
    if (!ok) return false;

    // Buat checkpoint log awal
    CheckpointMeta meta;
    meta.remaining_epochs = cfg.epochs;
    meta.completed_steps = 0;
    meta.last_loss = 0.0f;
    meta.last_acc = 0.0f;
    meta.last_ppl = 0.0f;
    meta.context_proficiency = 0.0f;

    std::string log_path = model_name + ".log";
    save_checkpoint_meta(meta, log_path);

    std::cout << "[IO] New model created: " << path << "\n";
    return true;
}

} // namespace io
} // namespace nol

// ==========================================
// UNIT TEST IO
// ==========================================
static void run_io_tests() {
    using namespace nol::io;
    using namespace nol::model;
    using namespace nol::config;
    using namespace nol::core;
    std::cout << "[TEST] Memulai unit test io...\n";

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

    // Test 1: Create new model
    {
        bool ok = create_new_model(cfg, "/tmp/test_nol_model");
        assert(ok);
    }

    // Test 2: Load model dan verifikasi bobot identik
    {
        NolModel model1(cfg);
        model1.initialize_weights(42);

        NolModel model2(cfg);
        bool ok = load_safetensors(model2, "/tmp/test_nol_model.safetensors");
        assert(ok);

        // Verifikasi embedding identik
        const float* e1 = model1.embed_E.fp32_data();
        const float* e2 = model2.embed_E.fp32_data();
        for (size_t i = 0; i < model1.embed_E.numel; ++i) {
            assert(std::abs(e1[i] - e2[i]) < 1e-6f);
        }

        // Verifikasi scalar parameters
        assert(std::abs(model1.curvature - model2.curvature) < 1e-6f);
        assert(std::abs(model1.lambda_h - model2.lambda_h) < 1e-6f);
        assert(std::abs(model1.blocks[0].gamma - model2.blocks[0].gamma) < 1e-6f);

        std::cout << "  Save/Load roundtrip OK\n";
    }

    // Test 3: Forward pass setelah load menghasilkan output identik
    {
        NolModel model1(cfg);
        model1.initialize_weights(42);

        NolModel model2(cfg);
        load_safetensors(model2, "/tmp/test_nol_model.safetensors");

        Tensor token_ids({1, 4}, DType::FP32);
        float* tip = token_ids.fp32_data();
        tip[0] = 1; tip[1] = 5; tip[2] = 10; tip[3] = 20;

        Tensor logits1, aux1, logits2, aux2;
        model1.forward(token_ids, logits1, aux1);
        model2.forward(token_ids, logits2, aux2);

        const float* l1 = logits1.fp32_data();
        const float* l2 = logits2.fp32_data();
        for (size_t i = 0; i < logits1.numel; ++i) {
            assert(std::abs(l1[i] - l2[i]) < 1e-4f);
        }
        std::cout << "  Forward after load matches original OK\n";
    }

    // Test 4: Checkpoint meta save/load
    {
        CheckpointMeta meta_out;
        meta_out.remaining_epochs = 7;
        meta_out.completed_steps = 1500;
        meta_out.last_loss = 2.3456f;
        meta_out.last_acc = 0.789f;
        meta_out.last_ppl = 10.43f;
        meta_out.context_proficiency = 0.82f;

        bool ok = save_checkpoint_meta(meta_out, "/tmp/test_nol_checkpoint.log");
        assert(ok);

        CheckpointMeta meta_in;
        ok = load_checkpoint_meta(meta_in, "/tmp/test_nol_checkpoint.log");
        assert(ok);

        assert(meta_in.remaining_epochs == 7);
        assert(meta_in.completed_steps == 1500);
        assert(std::abs(meta_in.last_loss - 2.3456f) < 1e-4f);
        assert(std::abs(meta_in.last_acc - 0.789f) < 1e-4f);
        assert(std::abs(meta_in.last_ppl - 10.43f) < 1e-2f);
        assert(std::abs(meta_in.context_proficiency - 0.82f) < 1e-4f);

        std::cout << "  Checkpoint meta save/load OK\n";
    }

    // Test 5: Export dengan nama kustom
    {
        NolModel model(cfg);
        model.initialize_weights(42);

        bool ok = export_model(model, "/tmp/custom_export_name");
        assert(ok);

        // Verifikasi file ada dan bisa di-load
        NolModel model2(cfg);
        ok = load_safetensors(model2, "/tmp/custom_export_name.safetensors");
        assert(ok);
        std::cout << "  Export custom name OK\n";
    }

    // Test 6: Load file tidak exist → return false
    {
        NolModel model(cfg);
        bool ok = load_safetensors(model, "/tmp/nonexistent_file.safetensors");
        assert(!ok);
        std::cout << "  Load nonexistent file returns false OK\n";
    }

    // Cleanup
    std::remove("/tmp/test_nol_model.safetensors");
    std::remove("/tmp/test_nol_model.log");
    std::remove("/tmp/test_nol_checkpoint.log");
    std::remove("/tmp/custom_export_name.safetensors");

    std::cout << "[PASS] ✅ Semua unit test io berhasil!\n\n";
}

#ifdef NOL_TEST_IO
int main() {
    run_io_tests();
    return 0;
}
#endif
