#include "../../include/config/config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace nol {
namespace config {

NolConfig get_default_config() {
    return NolConfig();
}

void print_config(const NolConfig& cfg) {
    std::cout << "\n\033[1;36m";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║         KONFIGURASI MODEL NOL (24 LAPISAN)              ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n\033[0m";

    std::cout << "║ \033[1m[Pilar 1: Fondasi]\033[0m                                      ║\n";
    std::cout << "║  BF16: " << (cfg.use_bf16 ? "ON " : "OFF") 
              << "  Arena: " << cfg.arena_block_size 
              << "  GEMM Block: " << cfg.gemm_block_size << "        ║\n";

    std::cout << "║ \033[1m[Pilar 3: Transformer Modern]\033[0m                           ║\n";
    std::cout << "║  Vocab:" << std::setw(6) << cfg.vocab_size 
              << "  d_model:" << std::setw(5) << cfg.d_model 
              << "  Layers:" << std::setw(3) << cfg.n_layers << "           ║\n";
    std::cout << "║  Heads:" << std::setw(3) << cfg.n_heads 
              << "/" << cfg.n_kv_heads 
              << "  d_head:" << std::setw(4) << cfg.d_head 
              << "  FFN:" << std::setw(5) << cfg.ffn_hidden << "       ║\n";
    std::cout << "║  RMSNorm ε:" << std::scientific << std::setprecision(1) << cfg.rmsnorm_eps 
              << "  RoPE base:" << std::fixed << std::setprecision(0) << cfg.rope_base << "    ║\n";

    std::cout << "║ \033[1m[Pilar 2: Lorentz Geometry]\033[0m                             ║\n";
    std::cout << "║  c_init:" << std::setprecision(2) << cfg.c_init 
              << "  c_min:" << cfg.c_min 
              << "  FP32 Only: " << (cfg.lorentz_fp32_only ? "YES" : "NO ") << "      ║\n";

    std::cout << "║ \033[1m[Pilar 4: T1-T5 Hyperbolic]\033[0m                            ║\n";
    std::cout << "║  T1 HGA: " << cfg.n_hyp_heads << " heads, d_hyp=" << cfg.d_hyp 
              << ", τ=" << std::setprecision(1) << cfg.tau_att << "         ║\n";
    std::cout << "║  T2 Gate bias:" << std::setprecision(2) << cfg.t2_gate_init_bias 
              << "  T3 γ_init:" << cfg.t3_gamma_init << "          ║\n";
    std::cout << "║  T3 Anchors: " << cfg.n_top_anchors << "+" 
              << (cfg.n_top_anchors * cfg.n_children) 
              << "  τ_c=" << std::setprecision(1) << cfg.tau_c << "              ║\n";
    std::cout << "║  T4 Experts:" << std::setw(3) << cfg.top_k_experts 
              << "+" << cfg.n_shared_experts << "sh  δ=" << std::setprecision(1) << cfg.iso_delta 
              << "  τ_r=" << cfg.tau_r << "     ║\n";
    std::cout << "║  T5 τ_o=" << cfg.tau_head << "  λ_h=" << cfg.lambda_h_init << "                      ║\n";

    std::cout << "║ \033[1m[Pilar 5: Training & MoE]\033[0m                               ║\n";
    std::cout << "║  Loss: LM=" << std::setprecision(2) << cfg.w_lm 
              << " HCL=" << cfg.w_hcl << " HIER=" << cfg.w_hier 
              << " ISO=" << cfg.w_iso << " BAL=" << cfg.w_bal << " ║\n";
    std::cout << "║  Curriculum:" << std::setprecision(0) << (cfg.curriculum_ratio * 100) << "%  GradClip:" 
              << std::setprecision(1) << cfg.grad_clip << "  LR:" << std::scientific << cfg.lr << "  ║\n";
    std::cout << "║  Self-Evo every " << std::fixed << cfg.self_evo_steps << " steps                  ║\n";

    std::cout << "\033[1;36m╚═══════════════════════════════════════════════════════════╝\033[0m\n\n";
}

bool validate_config(const NolConfig& cfg) {
    // Pilar 1
    if (cfg.arena_block_size == 0 || cfg.gemm_block_size == 0) return false;

    // Pilar 3
    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.n_layers <= 0) return false;
    if (cfg.d_model % cfg.n_heads != 0) return false;
    if (cfg.d_head != cfg.d_model / cfg.n_heads) return false;
    if (cfg.n_kv_heads <= 0 || cfg.n_kv_heads > cfg.n_heads) return false;
    if (cfg.ffn_hidden <= 0 || cfg.max_seq_len <= 0) return false;
    if (cfg.rmsnorm_eps <= 0.0f || cfg.rope_base <= 0.0f) return false;

    // Pilar 2
    if (cfg.c_init <= 0.0f || cfg.c_min <= 0.0f) return false;

    // Pilar 4
    if (cfg.n_hyp_heads <= 0 || cfg.n_hyp_heads > cfg.n_heads) return false;
    if (cfg.d_hyp <= 0) return false;
    if (cfg.tau_att <= 0.0f || cfg.tau_c <= 0.0f || cfg.tau_r <= 0.0f || cfg.tau_head <= 0.0f) return false;
    if (cfg.n_top_anchors <= 0 || cfg.n_children <= 0) return false;
    if (cfg.top_k_experts <= 0 || cfg.top_k_experts > cfg.n_top_anchors) return false;
    if (cfg.iso_delta <= 0.0f) return false;

    // Pilar 5
    if (cfg.w_lm < 0.0f || cfg.w_hcl < 0.0f || cfg.w_hier < 0.0f || cfg.w_iso < 0.0f || cfg.w_bal < 0.0f) return false;
    if (cfg.curriculum_ratio < 0.0f || cfg.curriculum_ratio > 1.0f) return false;
    if (cfg.grad_clip <= 0.0f || cfg.lr <= 0.0f || cfg.warmup_steps < 0 || cfg.self_evo_steps <= 0) return false;

    return true;
}
// ==========================================
// UNIT TEST CONFIG
// ==========================================
static void run_config_tests() {
    std::cout << "[TEST] Memulai unit test config...\n";

    // Test 1: Default config valid
    NolConfig cfg = get_default_config();
    assert(validate_config(cfg));

    // Test 2: Konsistensi dimensi
    assert(cfg.d_head == cfg.d_model / cfg.n_heads);
    assert(cfg.n_kv_heads < cfg.n_heads);
    assert(cfg.n_hyp_heads <= cfg.n_heads);

    // Test 3: Inisialisasi khusus sesuai dokumentasi
    assert(cfg.t3_gamma_init == 0.0f);           // T3 γ = 0
    assert(cfg.t2_gate_init_bias < 0.0f);        // T2 bias negatif
    assert(cfg.lorentz_fp32_only == true);       // Hiperbolik selalu fp32
    assert(cfg.tied_embedding == true);          // W_v = E
    assert(cfg.n_shared_experts == 1);           // T4 shared expert

    // Test 4: Validasi menolak config invalid
    NolConfig bad = cfg;
    
    bad = cfg; bad.c_init = -1.0f;
    assert(!validate_config(bad));

    bad = cfg; bad.n_hyp_heads = 100;
    assert(!validate_config(bad));

    bad = cfg; bad.n_kv_heads = cfg.n_heads + 1;
    assert(!validate_config(bad));

    bad = cfg; bad.d_head = 999;
    assert(!validate_config(bad));

    bad = cfg; bad.curriculum_ratio = 1.5f;
    assert(!validate_config(bad));

    bad = cfg; bad.top_k_experts = cfg.n_top_anchors + 1;
    assert(!validate_config(bad));

    // Test 5: Print config tidak crash
    print_config(cfg);

    std::cout << "[PASS] ✅ Semua unit test config berhasil!\n\n";
}

} // namespace config
} // namespace nol

// main HARUS di luar namespace agar linker bisa menemukan entry point
#ifdef NOL_TEST_CONFIG
int main() {
    nol::config::run_config_tests();
    return 0;
}
#endif
