#pragma once

#include "../config/config.hpp"
#include "../core/tensor.hpp"
#include "../core/gemm.hpp"
#include "../core/lorentz.hpp"
#include <vector>
#include <string>

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

    // ==========================================
    // FORWARD PASS
    // ==========================================
    void forward(const Tensor& token_ids, Tensor& logits, Tensor& aux_loss);

    // ==========================================
    // COMPONENT FORWARD FUNCTIONS
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
    // UTILITIES
    // ==========================================
    size_t param_count() const;
    void print_summary() const;
};

} // namespace model
} // namespace nol
