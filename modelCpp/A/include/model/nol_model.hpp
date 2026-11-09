#pragma once

#include "../config/config.hpp"
#include "../core/tensor.hpp"
#include "../core/gemm.hpp"
#include "../core/lorentz.hpp"
#include <vector>
#include <string>
#include <memory>

namespace nol {
namespace model {

using namespace nol::core;
using namespace nol::config;

/**
 * @brief Arsitektur Model Nol Lengkap (24 Lapisan Terintegrasi)
 * 
 * Pilar 3: RMSNorm, RoPE, GQA, SwiGLU, Embedding & Tied Head
 * Pilar 4: HGA(T1), GatedFusion(T2), CGL(T3), TIG(T4), HypHead(T5)
 * 
 * Forward pass persis Section 6 dokumentasi.
 */

// ==========================================
// FORWARD CACHE STRUCTURES (BARU)
// ==========================================

/**
 * @brief Cache intermediate tensors untuk satu block (dibutuhkan untuk backward).
 */
struct BlockCache {
    // Input/Output block
    Tensor h_in;            // Hidden state masuk ke block
    Tensor x;               // Setelah RMSNorm1

    // GQA intermediate
    Tensor q_proj;          // Q setelah projection (sebelum RoPE)
    Tensor k_proj;          // K setelah projection (sebelum RoPE)
    Tensor v_proj;          // V setelah projection
    Tensor q_rope;          // Q setelah RoPE
    Tensor k_rope;          // K setelah RoPE
    Tensor attn_scores;     // [B, n_heads, T, T] attention weights
    Tensor attn_out;        // Output attention (sebelum W_o)
    Tensor a;               // Output GQA lengkap (setelah W_o)

    // HGA intermediate (T1)
    Tensor z_proj;          // W_z * x (sebelum exp0)
    Tensor z;               // Lorentz point (setelah exp0)
    Tensor q_hyp_proj;      // W_q_hyp * x
    Tensor k_hyp_proj;      // W_k_hyp * x
    Tensor v_hyp_proj;      // W_v_hyp * x
    Tensor q_man;           // Q di manifold (setelah exp0 per head)
    Tensor k_man;           // K di manifold
    Tensor v_man;           // V di manifold
    Tensor hga_scores;      // [B, n_hyp_heads, T, T]
    Tensor u;               // Output HGA (tangent space)

    // T2: Gated Fusion
    Tensor gate;            // σ(W_g * x)
    Tensor Wu_u;            // W_u * u
    Tensor m;               // a + gate ⊙ Wu_u

    // T3: CGL intermediate
    Tensor h1;              // h + W_o * m (atau h + m jika fused)
    Tensor pi;              // [B*T, n_top_anchors] attention ke anchor
    Tensor m_vec;           // [B*T, d_hyp+1] tarikan anchor
    Tensor z_plus_m;        // z + m_vec
    Tensor z_tilde;         // Setelah normalisasi
    Tensor delta_core;      // log0(z_tilde) - log0(z)
    Tensor h2;              // h1 + γ ⊙ W_c * delta_core

    // T4: TIG routing
    Tensor x2;              // RMSNorm2(h2)
    Tensor Wr_x;            // W_r * x2
    Tensor route_weights;   // [B*T, n_experts + n_shared]
    Tensor tig_aux;         // Auxiliary loss dari TIG

    // SwiGLU intermediate
    Tensor h1_shared;       // W1_shared * x2
    Tensor h3_shared;       // W3_shared * x2
    Tensor h1_shared_silu;  // SiLU(h1_shared)
    Tensor gate_shared;     // h1_shared_silu ⊙ h3_shared
    Tensor ffn_shared_out;  // W2_shared * gate_shared

    // Per-expert SwiGLU cache (hanya untuk expert yang aktif)
    std::vector<Tensor> expert_h1;
    std::vector<Tensor> expert_h3;
    std::vector<Tensor> expert_h1_silu;
    std::vector<Tensor> expert_gate;
    std::vector<Tensor> expert_out;

    Tensor ffn_out;         // Total FFN output (shared + experts)
    Tensor h_out;           // Output block (h2 + ffn_out)
};

/**
 * @brief Cache untuk seluruh forward pass.
 */
struct ForwardCache {
    // Input asli
    Tensor token_ids;       // [B, T] integer token IDs

    // Embedding
    Tensor h0;              // [B*T, d_model] hasil embedding lookup
    Tensor T_lookup;        // [B*T, d_hyp] hyperbolic embedding

    // Per-block cache
    std::vector<BlockCache> blocks;

    // Final layers
    Tensor h_final;         // Output block terakhir
    Tensor xN;              // Setelah RMSNorm final
    Tensor zN_proj;         // W_z * xN
    Tensor zN;              // z untuk T5 Head
    Tensor logits;          // [B*T, vocab_size] final logits
    Tensor aux_loss;        // [2] = [L_iso, L_bal]

    // Metadata
    int batch_size = 0;
    int seq_len = 0;
    int total_tokens = 0;

    void resize(int n_layers) {
        blocks.resize(n_layers);
    }

    void clear() {
        blocks.clear();
    }
};

/**
 * @brief Akumulator gradient (flat vector sejajar dengan parameter).
 */
struct GradientAccumulator {
    std::vector<float> grads;      // Gradient flat
    std::vector<float> grad_norms; // Untuk tracking
    
    void resize(size_t n_params) {
        grads.assign(n_params, 0.0f);
    }
    
    void zero() {
        std::fill(grads.begin(), grads.end(), 0.0f);
    }
    
    float* data() { return grads.data(); }
    const float* data() const { return grads.data(); }
    size_t size() const { return grads.size(); }
};

// ==========================================
// MAIN MODEL CLASS
// ==========================================

class NolModel {
public:
    NolConfig cfg;

    // ==========================================
    // PILAR 3: EMBEDDING & TIED HEAD
    // ==========================================
    Tensor embed_E;       // [V x d_model] Euclidean embedding
    Tensor embed_T;       // [V x d_hyp] Hyperbolic table (tangent space)

    // ==========================================
    // PARAMETER PER BLOK
    // ==========================================
    struct BlockParams {
        Tensor rmsnorm1_gamma;  // [d_model]
        Tensor W_q;             // [d_model x d_model]
        Tensor W_k;             // [d_model x (n_kv_heads*d_head)]
        Tensor W_v;             // [d_model x (n_kv_heads*d_head)]
        Tensor W_o;             // [d_model x d_model]
        Tensor W_q_hyp;         // [d_model x (n_hyp_heads*(d_hyp+1))]
        Tensor W_k_hyp;         // [d_model x (n_hyp_heads*(d_hyp+1))]
        Tensor W_v_hyp;         // [d_model x (n_hyp_heads*(d_hyp+1))]
        Tensor W_g;             // [d_model x d_model] (bias negatif)
        Tensor W_u;             // [d_model x d_model]
        Tensor W_z;             // [d_model x (d_hyp+1)]
        Tensor W_c;             // [d_model x d_model]
        float gamma;            // Scalar γ, inisialisasi 0
        float lambda_tilde;     // Scalar λ̃ untuk σ(λ̃)
        Tensor anchors_l1;      // [n_top_anchors x (d_hyp+1)] tangent space
        Tensor anchors_l2;      // [n_top_anchors*n_children x (d_hyp+1)] tangent space
        std::vector<int> parent_map; // [n_top_anchors*n_children] → index L1
        Tensor W_r;             // [d_model x n_top_anchors]
        float beta;             // Scalar β
        Tensor rmsnorm2_gamma;  // [d_model]
        Tensor W1_shared;       // [d_model x ffn_hidden]
        Tensor W2_shared;       // [ffn_hidden x d_model]
        Tensor W3_shared;       // [d_model x ffn_hidden]
        std::vector<Tensor> W1_expert;
        std::vector<Tensor> W2_expert;
        std::vector<Tensor> W3_expert;
    };

    std::vector<BlockParams> blocks;

    // ==========================================
    // OUTPUT HEAD
    // ==========================================
    Tensor rmsnorm_final_gamma;
    Tensor W_v_head;            // Tied with embed_E
    Tensor H_v;                 // [V x (d_hyp+1)]
    float lambda_h;
    float curvature;            // c = softplus(c̃) + c_min

    // ==========================================
    // CONSTRUCTOR & INITIALIZATION
    // ==========================================
    NolModel() = default;
    explicit NolModel(const NolConfig& config);
    void initialize_weights(uint64_t seed);
    void rebuild_tied_embedding();

    // ==========================================
    // FORWARD PASS
    // ==========================================
    /**
     * @brief Forward pass standar (tanpa cache, untuk inference).
     */
    void forward(const Tensor& token_ids, Tensor& logits, Tensor& aux_loss);

    /**
     * @brief Forward pass dengan cache (untuk training dengan backward).
     * Menyimpan semua intermediate tensors untuk backward pass.
     */
    void forward_with_cache(const Tensor& token_ids, Tensor& logits, 
                           Tensor& aux_loss, ForwardCache& cache);

    // ==========================================
    // BACKWARD PASS (BARU)
    // ==========================================
    /**
     * @brief Backward pass: menghitung gradient untuk semua parameter.
     * @param grad_logits Gradient dari loss terhadap logits [B*T, vocab]
     * @param cache ForwardCache dari forward_with_cache()
     * @param grad_acc Output: gradient accumulator (flat, size = total_params)
     */
    void backward(const Tensor& grad_logits, const ForwardCache& cache,
                  GradientAccumulator& grad_acc);

    // ==========================================
    // COMPONENT FORWARD FUNCTIONS (STATIC)
    // ==========================================
    static void rmsnorm_forward(const Tensor& x, const Tensor& gamma,
                                Tensor& out, float eps);

    static void rope_apply(Tensor& q, Tensor& k, int seq_len,
                           int d_head, int n_heads, int n_kv_heads,
                           float base);

    static void gqa_forward(const Tensor& x, const Tensor& W_q,
                            const Tensor& W_k, const Tensor& W_v,
                            const Tensor& W_o, Tensor& out,
                            int batch, int seq_len, int n_heads,
                            int n_kv_heads, int d_head);

    static void hga_forward(const Tensor& x, const Tensor& W_q_hyp,
                            const Tensor& W_k_hyp, const Tensor& W_v_hyp,
                            Tensor& out, int batch, int seq_len,
                            int n_hyp_heads, int d_hyp, float c, float tau);

    static void gated_fusion_forward(const Tensor& a, const Tensor& u,
                                     const Tensor& x, const Tensor& W_g,
                                     const Tensor& W_u, Tensor& out,
                                     int batch, int seq_len, int d_model);

    static void cgl_forward(Tensor& h, const Tensor& z,
                            const Tensor& anchors_l1, const Tensor& anchors_l2,
                            const std::vector<int>& parent_map,
                            const Tensor& W_c, float gamma, float lambda_tilde,
                            float tau_c, float c, int batch, int seq_len,
                            int d_model, int d_hyp, int n_l1, int n_l2,
                            int n_children);

    static void tig_route(const Tensor& x, const Tensor& z,
                          const Tensor& anchors_l1, const Tensor& W_r,
                          float beta, float delta, float tau_r, float c,
                          int batch, int seq_len, int d_model, int d_hyp,
                          int n_experts, int top_k, int n_shared,
                          Tensor& route_weights, Tensor& aux_loss);

    static void swiglu_moe_forward(const Tensor& x,
                                   const Tensor& W1s, const Tensor& W2s, const Tensor& W3s,
                                   const std::vector<Tensor>& W1e,
                                   const std::vector<Tensor>& W2e,
                                   const std::vector<Tensor>& W3e,
                                   const Tensor& route_weights,
                                   Tensor& out, int batch, int seq_len,
                                   int d_model, int ffn_hidden,
                                   int n_experts, int top_k, int n_shared);

    static void head_forward(const Tensor& xN, const Tensor& z_N,
                             const Tensor& W_v, const Tensor& H_v,
                             float lambda_h, float c, float tau_o,
                             Tensor& logits, int batch, int seq_len,
                             int d_model, int d_hyp, int vocab_size);

    // ==========================================
    // COMPONENT BACKWARD FUNCTIONS (BARU - STATIC)
    // ==========================================

    /**
     * @brief Backward RMSNorm.
     * Input: y = x / sqrt(mean(x²) + eps) ⊙ gamma
     * Output: grad_x, grad_gamma
     */
    static void rmsnorm_backward(const Tensor& x, const Tensor& gamma,
                                 const Tensor& grad_out,
                                 Tensor& grad_x, Tensor& grad_gamma,
                                 float eps);

    /**
     * @brief Backward Linear (y = x @ W).
     * Output: grad_x = grad_y @ W^T, grad_W = x^T @ grad_y
     */
    static void linear_backward(const Tensor& x, const Tensor& grad_y,
                                Tensor& grad_x, Tensor& grad_W);

    /**
     * @brief Backward SiLU (y = x * sigmoid(x)).
     */
    static void silu_backward(const Tensor& x, const Tensor& grad_y, Tensor& grad_x);

    /**
     * @brief Backward Sigmoid (y = 1/(1+exp(-x))).
     */
    static void sigmoid_backward(const Tensor& y, const Tensor& grad_y, Tensor& grad_x);

    /**
     * @brief Backward GQA (full attention + projections).
     */
    static void gqa_backward(const Tensor& x, const Tensor& W_q, const Tensor& W_k,
                             const Tensor& W_v, const Tensor& W_o,
                             const Tensor& grad_out, const BlockCache& cache,
                             Tensor& grad_x,
                             Tensor& grad_W_q, Tensor& grad_W_k,
                             Tensor& grad_W_v, Tensor& grad_W_o,
                             int batch, int seq_len, int n_heads,
                             int n_kv_heads, int d_head);

    /**
     * @brief Backward HGA (T1).
     */
    static void hga_backward(const Tensor& x, const Tensor& W_q_hyp,
                             const Tensor& W_k_hyp, const Tensor& W_v_hyp,
                             const Tensor& grad_out, const BlockCache& cache,
                             Tensor& grad_x,
                             Tensor& grad_W_q_hyp, Tensor& grad_W_k_hyp,
                             Tensor& grad_W_v_hyp,
                             int batch, int seq_len, int n_hyp_heads,
                             int d_hyp, float c, float tau);

    /**
     * @brief Backward Gated Fusion (T2).
     * m = a + σ(W_g x) ⊙ (W_u u)
     */
    static void gated_fusion_backward(const Tensor& a, const Tensor& u,
                                      const Tensor& x, const Tensor& W_g,
                                      const Tensor& W_u, const Tensor& grad_m,
                                      const BlockCache& cache,
                                      Tensor& grad_a, Tensor& grad_u, Tensor& grad_x,
                                      Tensor& grad_W_g, Tensor& grad_W_u,
                                      int batch, int seq_len, int d_model);

    /**
     * @brief Backward CGL (T3).
     */
    static void cgl_backward(const Tensor& h, const Tensor& z,
                             const Tensor& grad_h2, const BlockCache& cache,
                             const Tensor& W_c, float gamma, float lambda_tilde,
                             float tau_c, float c,
                             Tensor& grad_h, Tensor& grad_z,
                             Tensor& grad_W_c, Tensor& grad_anchors_l1,
                             int batch, int seq_len, int d_model, int d_hyp,
                             int n_l1, int n_children);

    /**
     * @brief Backward TIG routing (T4).
     */
    static void tig_backward(const Tensor& x, const Tensor& z,
                             const Tensor& grad_ffn_in, const BlockCache& cache,
                             Tensor& grad_x, Tensor& grad_z,
                             Tensor& grad_W_r,
                             int batch, int seq_len, int d_model, int d_hyp,
                             int n_experts, float c, float tau_r, float beta);

    /**
     * @brief Backward SwiGLU MoE.
     */
    static void swiglu_moe_backward(const Tensor& x,
                                    const Tensor& W1s, const Tensor& W2s, const Tensor& W3s,
                                    const std::vector<Tensor>& W1e,
                                    const std::vector<Tensor>& W2e,
                                    const std::vector<Tensor>& W3e,
                                    const Tensor& route_weights,
                                    const Tensor& grad_out, const BlockCache& cache,
                                    Tensor& grad_x,
                                    Tensor& grad_W1s, Tensor& grad_W2s, Tensor& grad_W3s,
                                    std::vector<Tensor>& grad_W1e,
                                    std::vector<Tensor>& grad_W2e,
                                    std::vector<Tensor>& grad_W3e,
                                    int batch, int seq_len, int d_model, int ffn_hidden,
                                    int n_experts);

    /**
     * @brief Backward T5 Head.
     * logit_v = <W_v, xN> + λ_h * (2/c + 2<z_N, H_v>_L) / tau_o
     */
    static void head_backward(const Tensor& xN, const Tensor& z_N,
                              const Tensor& W_v, const Tensor& H_v,
                              const Tensor& grad_logits,
                              Tensor& grad_xN, Tensor& grad_zN,
                              Tensor& grad_W_v, Tensor& grad_H_v,
                              float lambda_h, float c, float tau_o,
                              int batch, int seq_len, int d_model,
                              int d_hyp, int vocab_size);

    // ==========================================
    // UTILITIES
    // ==========================================
    size_t param_count() const;
    void print_summary() const;

    /**
     * @brief Flatten gradient dari cache ke vector flat (sejajar dengan flatten_params Trainer).
     * Harus urutannya SAMA PERSIS dengan flatten_params di Trainer.
     */
    void accumulate_gradients(const ForwardCache& cache,
                              const std::vector<Tensor>& grad_embed_E,
                              const std::vector<Tensor>& grad_embed_T,
                              const std::vector<std::vector<Tensor>>& grad_blocks,
                              const Tensor& grad_rmsnorm_final,
                              const Tensor& grad_H_v,
                              GradientAccumulator& grad_acc) const;
};

} // namespace model
} // namespace nol
