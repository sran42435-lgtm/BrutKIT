#pragma once

#include "../config/config.hpp"
#include "../core/tensor.hpp"
#include "../model/nol_model.hpp"
#include <string>
#include <unordered_map>

namespace nol {
namespace io {

using namespace nol::core;
using namespace nol::config;
using namespace nol::model;

/**
 * @brief Metadata checkpoint untuk resume training.
 */
struct CheckpointMeta {
    int remaining_epochs = 0;
    int completed_steps = 0;
    float last_loss = 0.0f;
    float last_acc = 0.0f;
    float last_ppl = 0.0f;
    float context_proficiency = 0.0f;
};

/**
 * @brief Save semua bobot model ke file safetensors.
 * Format: [8-byte header_size][JSON header][binary tensor data]
 * @param model Model yang akan disimpan
 * @param path Path file output (.safetensors)
 * @return true jika berhasil
 */
bool save_safetensors(const NolModel& model, const std::string& path);

/**
 * @brief Load bobot model dari file safetensors.
 * @param model Model tujuan (harus sudah dikonstruksi dengan config yang sama)
 * @param path Path file input (.safetensors)
 * @return true jika berhasil
 */
bool load_safetensors(NolModel& model, const std::string& path);

/**
 * @brief Save metadata checkpoint ke file log terpisah.
 * @param meta Metadata checkpoint
 * @param path Path file log (.log)
 * @return true jika berhasil
 */
bool save_checkpoint_meta(const CheckpointMeta& meta, const std::string& path);

/**
 * @brief Load metadata checkpoint dari file log.
 * @param meta Output metadata
 * @param path Path file log (.log)
 * @return true jika berhasil
 */
bool load_checkpoint_meta(CheckpointMeta& meta, const std::string& path);

/**
 * @brief Export model dengan nama kustom pengguna.
 * Menyimpan bobot + config ke bundle safetensors.
 * @param model Model yang akan di-export
 * @param export_name Nama file output (tanpa ekstensi)
 * @return true jika berhasil
 */
bool export_model(const NolModel& model, const std::string& export_name);

/**
 * @brief Inisialisasi bobot acak dan simpan sebagai model baru.
 * @param cfg Konfigurasi model
 * @param model_name Nama model (tanpa ekstensi)
 * @return true jika berhasil
 */
bool create_new_model(const NolConfig& cfg, const std::string& model_name);

} // namespace io
} // namespace nol
