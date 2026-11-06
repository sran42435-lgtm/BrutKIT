#pragma once

#include <cstdint>
#include <string>

namespace nol {
namespace config {

/**
 * @brief Konfigurasi Arsitektur Model Nol (24 Lapisan Terintegrasi)
 * Hanya berisi variabel arsitektur & flag integrasi. 
 * Tidak ada threshold/trigger kompleks.
 */
struct NolConfig {
    // ==========================================
    // PILAR 1: FONDASI MEMORI & KOMPUTASI RENDAH
    // ==========================================
    bool use_bf16 = false;          // L1: FP32 default, BF16 opsional untuk stream Euclidean
    size_t arena_block_size = 1 << 20; // L2: Memory Arena block size (1MB default)
    uint64_t rng_seed = 42;         // L3: Seed untuk inisialisasi bobot acak
    int gemm_block_size = 64;       // L4: Block size untuk GEMM blocked SIMD
    bool enable_elementwise_simd = true; // L5: SIMD untuk elementwise ops

    // ==========================================
    // PILAR 3: TULANG PUNGGUNG TRANSFORMER MODERN
    // ==========================================
    int vocab_size = 32000;         // L10: Embedding & Tied Head
    int d_model = 768;              // L10: Dimensi model
    int n_layers = 12;              // Jumlah blok transformer
    int n_heads = 12;               // L8: GQA heads
    int n_kv_heads = 4;             // L8: GQA KV heads (n_kv_heads < n_heads)
    int d_head = 64;                // d_model / n_heads
    int ffn_hidden = 2048;          // L9: SwiGLU FFN hidden dim
    int max_seq_len = 2048;         // Konteks maksimum
    float rmsnorm_eps = 1e-5f;      // L6: RMSNorm epsilon
    float rope_base = 10000.0f;     // L7: RoPE base frequency
    bool tied_embedding = true;     // L10: W_v berbagi bobot dengan E

    // ==========================================
    // PILAR 2: MATEMATIKA GEOMETRI HIPERBOLIK
    // ==========================================
    float c_init = 1.0f;            // L12: Kelengkungan awal Lorentz (-c)
    float c_min = 0.01f;            // L12: Minimum curvature (c = softplus(c̃) + c_min)
    bool lorentz_fp32_only = true;  // L12: Bagian hiperbolik SELALU fp32

    // ==========================================
    // PILAR 4: CABANG HIPERBOLIK & LAPISAN KUSTOM (T1-T5)
    // ==========================================
    // T1: HGA
    int n_hyp_heads = 4;            // L13: 25% dari n_heads
    int d_hyp = 32;                 // L13: Dimensi ruang tangen (titik Lorentz = d_hyp+1)
    float tau_att = 1.0f;           // L13: Temperature HGA
    
    // T2: Gated Fusion
    float t2_gate_init_bias = -2.2f; // L14: Bias negatif agar σ ≈ 0.1 di awal
    
    // T3: CGL
    int n_top_anchors = 16;         // L15: Anchor level-1 (K1)
    int n_children = 16;            // L15: Anak per anchor (K2) = 256 total L2
    float tau_c = 1.0f;             // L15: Temperature CGL
    float t3_gamma_init = 0.0f;     // L15: γ diinisialisasi 0
    
    // T4: TIG
    int top_k_experts = 2;          // L16: Expert aktif per token
    float iso_delta = 2.0f;         // L16: Radius isolasi topik δ
    float tau_r = 1.0f;             // L16: Temperature routing TIG
    int n_shared_experts = 1;       // L16: Shared expert (selalu aktif)
    
    // T5: Hyperbolic Output Head
    float tau_head = 1.0f;          // L17: Temperature T5 (τ_o)
    float lambda_h_init = 0.1f;     // L17: λ_h inisialisasi

    // ==========================================
    // PILAR 5: TRAINING & MOE ROUTING
    // ==========================================
    float w_lm = 1.0f;              // L19: Loss weight L_LM
    float w_hcl = 0.1f;             // L19: α - L_hcl
    float w_hier = 0.05f;           // L19: β - L_hier
    float w_iso = 0.02f;            // L19: γ_i - L_iso
    float w_bal = 0.01f;            // L19: δ - L_bal
    float curriculum_ratio = 0.05f; // L19: 5% langkah awal hanya L_LM + L_bal
    float grad_clip = 1.0f;         // L19: Clip gradien
    float lr = 1e-4f;               // L19: Learning rate AdamW
    int warmup_steps = 1000;        // L19: Linear warmup
    int self_evo_steps = 1000;      // L19: Self-evolution anchor setiap N steps
    
    // Flag Integrasi (untuk verifikasi, bukan ablasi produksi)
    bool enable_hga = true;         // T1
    bool enable_gated_fusion = true;// T2
    bool enable_cgl = true;         // T3
    bool enable_tig = true;         // T4
    bool enable_hyp_head = true;    // T5
    bool enable_moe = true;         // L18: MoE Routing

    // Constructor default
    NolConfig() = default;
};

// Utilitas
NolConfig get_default_config();
void print_config(const NolConfig& cfg);
bool validate_config(const NolConfig& cfg);

} // namespace config
} // namespace nol
