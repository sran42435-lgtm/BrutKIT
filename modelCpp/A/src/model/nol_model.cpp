#include "../../include/model/nol_model.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <cassert>

namespace nol {
namespace model {

// ==========================================
// KONSTRUKTOR & INISIALISASI BOBOT
// ==========================================

NolModel::NolModel(const NolConfig& config) : cfg(config) {
    curvature = std::log(std::exp(cfg.c_init - cfg.c_min) - 1.0f + 1e-8f);
    lambda_h = cfg.lambda_h_init;

    embed_E = Tensor({static_cast<size_t>(cfg.vocab_size),
                      static_cast<size_t>(cfg.d_model)}, DType::FP32);
    embed_T = Tensor({static_cast<size_t>(cfg.vocab_size),
                      static_cast<size_t>(cfg.d_hyp)}, DType::FP32);

    blocks.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& bp = blocks[l];
        int d = cfg.d_model;
        int dh1 = cfg.d_hyp + 1;
        int kv_dim = cfg.n_kv_heads * cfg.d_head;
        int hyp_dim = cfg.n_hyp_heads * dh1;

        bp.rmsnorm1_gamma = Tensor({static_cast<size_t>(d)}, DType::FP32);
        bp.W_q = Tensor({static_cast<size_t>(d), static_cast<size_t>(d)}, DType::FP32);
        bp.W_k = Tensor({static_cast<size_t>(d), static_cast<size_t>(kv_dim)}, DType::FP32);
        bp.W_v = Tensor({static_cast<size_t>(d), static_cast<size_t>(kv_dim)}, DType::FP32);
        bp.W_o = Tensor({static_cast<size_t>(d), static_cast<size_t>(d)}, DType::FP32);
        bp.W_q_hyp = Tensor({static_cast<size_t>(d), static_cast<size_t>(hyp_dim)}, DType::FP32);
        bp.W_k_hyp = Tensor({static_cast<size_t>(d), static_cast<size_t>(hyp_dim)}, DType::FP32);
        bp.W_v_hyp = Tensor({static_cast<size_t>(d), static_cast<size_t>(hyp_dim)}, DType::FP32);
        bp.W_g = Tensor({static_cast<size_t>(d), static_cast<size_t>(d)}, DType::FP32);
        bp.W_u = Tensor({static_cast<size_t>(d), static_cast<size_t>(d)}, DType::FP32);
        bp.W_z = Tensor({static_cast<size_t>(d), static_cast<size_t>(dh1)}, DType::FP32);
        bp.W_c = Tensor({static_cast<size_t>(d), static_cast<size_t>(d)}, DType::FP32);
        bp.gamma = cfg.t3_gamma_init;
        bp.lambda_tilde = 0.0f;

        int n_l1 = cfg.n_top_anchors;
        int n_l2 = cfg.n_top_anchors * cfg.n_children;
        bp.anchors_l1 = Tensor({static_cast<size_t>(n_l1), static_cast<size_t>(dh1)}, DType::FP32);
        bp.anchors_l2 = Tensor({static_cast<size_t>(n_l2), static_cast<size_t>(dh1)}, DType::FP32);
        bp.parent_map.resize(n_l2);
        for (int i = 0; i < n_l2; ++i) {
            bp.parent_map[i] = i / cfg.n_children;
        }

        bp.W_r = Tensor({static_cast<size_t>(d), static_cast<size_t>(n_l1)}, DType::FP32);
        bp.beta = 1.0f;
        bp.rmsnorm2_gamma = Tensor({static_cast<size_t>(d)}, DType::FP32);
        bp.W1_shared = Tensor({static_cast<size_t>(d), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bp.W2_shared = Tensor({static_cast<size_t>(cfg.ffn_hidden), static_cast<size_t>(d)}, DType::FP32);
        bp.W3_shared = Tensor({static_cast<size_t>(d), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);

        bp.W1_expert.resize(n_l1);
        bp.W2_expert.resize(n_l1);
        bp.W3_expert.resize(n_l1);
        for (int e = 0; e < n_l1; ++e) {
            bp.W1_expert[e] = Tensor({static_cast<size_t>(d), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
            bp.W2_expert[e] = Tensor({static_cast<size_t>(cfg.ffn_hidden), static_cast<size_t>(d)}, DType::FP32);
            bp.W3_expert[e] = Tensor({static_cast<size_t>(d), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        }
    }

    rmsnorm_final_gamma = Tensor({static_cast<size_t>(cfg.d_model)}, DType::FP32);
    H_v = Tensor({static_cast<size_t>(cfg.vocab_size),
                  static_cast<size_t>(cfg.d_hyp + 1)}, DType::FP32);

    // W_v_head TIDAK dialokasikan di sini — akan di-set sebagai alias di initialize_weights
    // atau direkonstruksi setelah load_safetensors
}

void NolModel::initialize_weights(uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> normal(0.0f, 0.02f);

    auto fill_normal = [&](Tensor& t) {
        float* p = t.fp32_data();
        for (size_t i = 0; i < t.numel; ++i) p[i] = normal(rng);
    };

    auto fill_const = [&](Tensor& t, float val) {
        t.fill(val);
    };

    fill_normal(embed_E);
    fill_normal(embed_T);

    for (auto& bp : blocks) {
        fill_const(bp.rmsnorm1_gamma, 1.0f);
        fill_normal(bp.W_q); fill_normal(bp.W_k);
        fill_normal(bp.W_v); fill_normal(bp.W_o);
        fill_normal(bp.W_q_hyp); fill_normal(bp.W_k_hyp); fill_normal(bp.W_v_hyp);

        fill_normal(bp.W_g);
        {
            float* wg = bp.W_g.fp32_data();
            int d = cfg.d_model;
            for (int i = 0; i < d; ++i) {
                wg[i * d + i] = cfg.t2_gate_init_bias;
            }
        }
        fill_normal(bp.W_u);
        fill_normal(bp.W_z);
        fill_normal(bp.W_c);
        fill_normal(bp.anchors_l1);
        fill_normal(bp.anchors_l2);
        fill_normal(bp.W_r);
        fill_const(bp.rmsnorm2_gamma, 1.0f);
        fill_normal(bp.W1_shared); fill_normal(bp.W2_shared); fill_normal(bp.W3_shared);

        for (int e = 0; e < cfg.n_top_anchors; ++e) {
            fill_normal(bp.W1_expert[e]);
            fill_normal(bp.W2_expert[e]);
            fill_normal(bp.W3_expert[e]);
        }
    }

    fill_const(rmsnorm_final_gamma, 1.0f);
    fill_normal(H_v);

    // Set tied embedding alias SETELAH embed_E dialokasikan dan diisi
    W_v_head.data = embed_E.data;
    W_v_head.shape = embed_E.shape;
    W_v_head.strides = embed_E.strides;
    W_v_head.numel = embed_E.numel;
    W_v_head.dtype = embed_E.dtype;
    W_v_head.owns_memory = false;
}

/**
 * @brief Rekonstruksi alias W_v_head → embed_E setelah load.
 * HARUS dipanggil setelah load_safetensors.
 */
void NolModel::rebuild_tied_embedding() {
    W_v_head.data = embed_E.data;
    W_v_head.shape = embed_E.shape;
    W_v_head.strides = embed_E.strides;
    W_v_head.numel = embed_E.numel;
    W_v_head.dtype = embed_E.dtype;
    W_v_head.owns_memory = false;
}

// ==========================================
// COMPONENT IMPLEMENTATIONS
// ==========================================

void NolModel::rmsnorm_forward(const Tensor& x, const Tensor& gamma,
                               Tensor& out, float eps) {
    assert(x.dtype == DType::FP32 && gamma.dtype == DType::FP32);
    size_t total = x.numel;
    size_t d = gamma.numel;
    size_t rows = total / d;
    const float* xp = x.fp32_data();
    const float* gp = gamma.fp32_data();
    float* op = out.fp32_data();

    for (size_t r = 0; r < rows; ++r) {
        const float* row = xp + r * d;
        float* orow = op + r * d;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) sum_sq += row[j] * row[j];
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(d) + eps);
        for (size_t j = 0; j < d; ++j) {
            orow[j] = row[j] * inv_rms * gp[j];
        }
    }
}

void NolModel::rope_apply(Tensor& q, Tensor& k, int seq_len,
                          int d_head, int n_heads, int n_kv_heads,
                          float base) {
    int half = d_head / 2;
    float* qp = q.fp32_data();
    float* kp = k.fp32_data();
    int batch = static_cast<int>(q.shape[0]);

    for (int b = 0; b < batch; ++b) {
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int h = 0; h < n_heads; ++h) {
                size_t offset = (b * seq_len + pos) * n_heads * d_head + h * d_head;
                for (int i = 0; i < half; ++i) {
                    float theta = static_cast<float>(pos) / std::pow(base, 2.0f * i / d_head);
                    float cos_t = std::cos(theta);
                    float sin_t = std::sin(theta);
                    float q0 = qp[offset + 2 * i];
                    float q1 = qp[offset + 2 * i + 1];
                    qp[offset + 2 * i] = q0 * cos_t - q1 * sin_t;
                    qp[offset + 2 * i + 1] = q0 * sin_t + q1 * cos_t;
                }
            }
            for (int h = 0; h < n_kv_heads; ++h) {
                size_t offset = (b * seq_len + pos) * n_kv_heads * d_head + h * d_head;
                for (int i = 0; i < half; ++i) {
                    float theta = static_cast<float>(pos) / std::pow(base, 2.0f * i / d_head);
                    float cos_t = std::cos(theta);
                    float sin_t = std::sin(theta);
                    float k0 = kp[offset + 2 * i];
                    float k1 = kp[offset + 2 * i + 1];
                    kp[offset + 2 * i] = k0 * cos_t - k1 * sin_t;
                    kp[offset + 2 * i + 1] = k0 * sin_t + k1 * cos_t;
                }
            }
        }
    }
}

void NolModel::gqa_forward(const Tensor& x, const Tensor& W_q,
                           const Tensor& W_k, const Tensor& W_v,
                           const Tensor& W_o, Tensor& out,
                           int batch, int seq_len, int n_heads,
                           int n_kv_heads, int d_head) {
    int d_model = n_heads * d_head;
    int kv_dim = n_kv_heads * d_head;
    int total_tokens = batch * seq_len;

    Tensor Q({static_cast<size_t>(total_tokens), static_cast<size_t>(d_model)}, DType::FP32);
    Tensor K({static_cast<size_t>(total_tokens), static_cast<size_t>(kv_dim)}, DType::FP32);
    Tensor V({static_cast<size_t>(total_tokens), static_cast<size_t>(kv_dim)}, DType::FP32);

    gemm(x.fp32_data(), W_q.fp32_data(), Q.fp32_data(), total_tokens, d_model, d_model);
    gemm(x.fp32_data(), W_k.fp32_data(), K.fp32_data(), total_tokens, kv_dim, d_model);
    gemm(x.fp32_data(), W_v.fp32_data(), V.fp32_data(), total_tokens, kv_dim, d_model);

    Q.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    K.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)});
    rope_apply(Q, K, seq_len, d_head, n_heads, n_kv_heads, 10000.0f);
    Q.reshape({static_cast<size_t>(total_tokens), static_cast<size_t>(d_model)});
    K.reshape({static_cast<size_t>(total_tokens), static_cast<size_t>(kv_dim)});

    Tensor attn_out({static_cast<size_t>(total_tokens), static_cast<size_t>(d_model)}, DType::FP32);
    attn_out.fill(0.0f);

    float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    int heads_per_kv = n_heads / n_kv_heads;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n_heads; ++h) {
            int kv_h = h / heads_per_kv;
            for (int i = 0; i < seq_len; ++i) {
                std::vector<float> scores(i + 1);
                float max_score = -1e30f;
                for (int j = 0; j <= i; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < d_head; ++d) {
                        size_t q_idx = (b * seq_len + i) * d_model + h * d_head + d;
                        size_t k_idx = (b * seq_len + j) * kv_dim + kv_h * d_head + d;
                        s += Q.fp32_data()[q_idx] * K.fp32_data()[k_idx];
                    }
                    s *= scale;
                    scores[j] = s;
                    if (s > max_score) max_score = s;
                }
                float sum_exp = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = 1.0f / sum_exp;
                for (int j = 0; j <= i; ++j) scores[j] *= inv_sum;

                for (int d = 0; d < d_head; ++d) {
                    float val = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        size_t v_idx = (b * seq_len + j) * kv_dim + kv_h * d_head + d;
                        val += scores[j] * V.fp32_data()[v_idx];
                    }
                    size_t out_idx = (b * seq_len + i) * d_model + h * d_head + d;
                    attn_out.fp32_data()[out_idx] = val;
                }
            }
        }
    }

    gemm(attn_out.fp32_data(), W_o.fp32_data(), out.fp32_data(),
         total_tokens, d_model, d_model);
}

void NolModel::hga_forward(const Tensor& x, const Tensor& W_q_hyp,
                           const Tensor& W_k_hyp, const Tensor& W_v_hyp,
                           Tensor& out, int batch, int seq_len,
                           int n_hyp_heads, int d_hyp, float c, float tau) {
    int dh1 = d_hyp + 1;
    int hyp_dim = n_hyp_heads * dh1;
    int total_tokens = batch * seq_len;
    int d_model = static_cast<int>(x.shape[1]);

    Tensor Q_tan({static_cast<size_t>(total_tokens), static_cast<size_t>(hyp_dim)}, DType::FP32);
    Tensor K_tan({static_cast<size_t>(total_tokens), static_cast<size_t>(hyp_dim)}, DType::FP32);
    Tensor V_tan({static_cast<size_t>(total_tokens), static_cast<size_t>(hyp_dim)}, DType::FP32);

    gemm(x.fp32_data(), W_q_hyp.fp32_data(), Q_tan.fp32_data(), total_tokens, hyp_dim, d_model);
    gemm(x.fp32_data(), W_k_hyp.fp32_data(), K_tan.fp32_data(), total_tokens, hyp_dim, d_model);
    gemm(x.fp32_data(), W_v_hyp.fp32_data(), V_tan.fp32_data(), total_tokens, hyp_dim, d_model);

    Tensor Q_man({static_cast<size_t>(total_tokens * n_hyp_heads), static_cast<size_t>(dh1)}, DType::FP32);
    Tensor K_man({static_cast<size_t>(total_tokens * n_hyp_heads), static_cast<size_t>(dh1)}, DType::FP32);
    Tensor V_man({static_cast<size_t>(total_tokens * n_hyp_heads), static_cast<size_t>(dh1)}, DType::FP32);

    for (int t = 0; t < total_tokens; ++t) {
        for (int h = 0; h < n_hyp_heads; ++h) {
            size_t tan_off = t * hyp_dim + h * dh1;
            size_t man_off = (t * n_hyp_heads + h) * dh1;
            exp0(Q_tan.fp32_data() + tan_off, Q_man.fp32_data() + man_off, dh1, c);
            exp0(K_tan.fp32_data() + tan_off, K_man.fp32_data() + man_off, dh1, c);
            exp0(V_tan.fp32_data() + tan_off, V_man.fp32_data() + man_off, dh1, c);
        }
    }

    out.fill(0.0f);
    float inv_tau = 1.0f / tau;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n_hyp_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                int ti = b * seq_len + i;
                size_t qi_off = (ti * n_hyp_heads + h) * dh1;

                std::vector<float> scores(i + 1);
                float max_s = -1e30f;
                for (int j = 0; j <= i; ++j) {
                    int tj = b * seq_len + j;
                    size_t kj_off = (tj * n_hyp_heads + h) * dh1;
                    float s = (2.0f / c + 2.0f * linner(
                        Q_man.fp32_data() + qi_off,
                        K_man.fp32_data() + kj_off, dh1)) * inv_tau;
                    scores[j] = s;
                    if (s > max_s) max_s = s;
                }

                float Z = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - max_s);
                    Z += scores[j];
                }
                float inv_Z = 1.0f / Z;
                for (int j = 0; j <= i; ++j) scores[j] *= inv_Z;

                std::vector<float> centroid(dh1, 0.0f);
                for (int j = 0; j <= i; ++j) {
                    int tj = b * seq_len + j;
                    size_t vj_off = (tj * n_hyp_heads + h) * dh1;
                    for (int d = 0; d < dh1; ++d) {
                        centroid[d] += scores[j] * V_man.fp32_data()[vj_off + d];
                    }
                }
                float zz = linner(centroid.data(), centroid.data(), dh1);
                float abs_zz = std::abs(zz);
                float inv_norm = 1.0f / (std::sqrt(c) * std::sqrt(std::max(abs_zz, 1e-8f)));
                for (int d = 0; d < dh1; ++d) centroid[d] *= inv_norm;
                if (centroid[0] < 0.0f) {
                    for (int d = 0; d < dh1; ++d) centroid[d] = -centroid[d];
                }

                std::vector<float> u_tan(d_hyp);
                log0(centroid.data(), u_tan.data(), dh1, c);

                size_t out_off = ti * (n_hyp_heads * d_hyp) + h * d_hyp;
                for (int d = 0; d < d_hyp; ++d) {
                    out.fp32_data()[out_off + d] = u_tan[d];
                }
            }
        }
    }
}

void NolModel::gated_fusion_forward(const Tensor& a, const Tensor& u,
                                    const Tensor& x, const Tensor& W_g,
                                    const Tensor& W_u, Tensor& out,
                                    int batch, int seq_len, int d_model) {
    int total = batch * seq_len;

    Tensor gate({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
    gemm(x.fp32_data(), W_g.fp32_data(), gate.fp32_data(), total, d_model, d_model);
    gate.sigmoid_inplace();

    Tensor Wu_u({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
    gemm(u.fp32_data(), W_u.fp32_data(), Wu_u.fp32_data(), total, d_model, d_model);

    const float* ap = a.fp32_data();
    const float* gp = gate.fp32_data();
    const float* wup = Wu_u.fp32_data();
    float* op = out.fp32_data();
    for (int i = 0; i < total * d_model; ++i) {
        op[i] = ap[i] + gp[i] * wup[i];
    }
}

void NolModel::cgl_forward(Tensor& h, const Tensor& z,
                           const Tensor& anchors_l1, const Tensor& anchors_l2,
                           const std::vector<int>& parent_map,
                           const Tensor& W_c, float gamma, float lambda_tilde,
                           float tau_c, float c, int batch, int seq_len,
                           int d_model, int d_hyp, int n_l1, int n_l2,
                           int n_children) {
    if (std::abs(gamma) < 1e-8f) return;

    int total = batch * seq_len;
    int dh1 = d_hyp + 1;
    float lambda = 1.0f / (1.0f + std::exp(-lambda_tilde));

    Tensor delta_core({static_cast<size_t>(total), static_cast<size_t>(d_hyp)}, DType::FP32);
    delta_core.fill(0.0f);

    const float* zp = z.fp32_data();
    const float* a1p = anchors_l1.fp32_data();

    for (int t = 0; t < total; ++t) {
        std::vector<float> pi(n_l1);
        float max_d = -1e30f;
        for (int k = 0; k < n_l1; ++k) {
            float d2 = dist2_lorentz(zp + t * dh1, a1p + k * dh1, dh1, c);
            pi[k] = -d2 / tau_c;
            if (pi[k] > max_d) max_d = pi[k];
        }
        float Z = 0.0f;
        for (int k = 0; k < n_l1; ++k) {
            pi[k] = std::exp(pi[k] - max_d);
            Z += pi[k];
        }
        for (int k = 0; k < n_l1; ++k) pi[k] /= Z;

        std::vector<float> m_vec(dh1, 0.0f);
        for (int k = 0; k < n_l1; ++k) {
            for (int d = 0; d < dh1; ++d) {
                m_vec[d] += pi[k] * lambda * a1p[k * dh1 + d];
            }
            for (int ch = 0; ch < n_children; ++ch) {
                int child_idx = k * n_children + ch;
                int parent_idx = parent_map[child_idx];
                for (int d = 0; d < dh1; ++d) {
                    m_vec[d] += pi[k] * lambda * lambda * a1p[parent_idx * dh1 + d]
                                / static_cast<float>(n_children);
                }
            }
        }

        std::vector<float> z_plus_m(dh1);
        for (int d = 0; d < dh1; ++d) {
            z_plus_m[d] = zp[t * dh1 + d] + m_vec[d];
        }
        float zz = linner(z_plus_m.data(), z_plus_m.data(), dh1);
        float inv_norm = 1.0f / (std::sqrt(c) * std::sqrt(std::max(std::abs(zz), 1e-8f)));
        for (int d = 0; d < dh1; ++d) z_plus_m[d] *= inv_norm;
        if (z_plus_m[0] < 0.0f) {
            for (int d = 0; d < dh1; ++d) z_plus_m[d] = -z_plus_m[d];
        }

        std::vector<float> log_ztilde(d_hyp), log_z(d_hyp);
        log0(z_plus_m.data(), log_ztilde.data(), dh1, c);
        log0(zp + t * dh1, log_z.data(), dh1, c);
        for (int d = 0; d < d_hyp; ++d) {
            delta_core.fp32_data()[t * d_hyp + d] = log_ztilde[d] - log_z[d];
        }
    }

    Tensor Wc_dc({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
    gemm(delta_core.fp32_data(), W_c.fp32_data(), Wc_dc.fp32_data(),
         total, d_model, d_hyp);

    float* hp = h.fp32_data();
    const float* wcdcp = Wc_dc.fp32_data();
    for (int i = 0; i < total * d_model; ++i) {
        hp[i] += gamma * wcdcp[i];
    }
}

void NolModel::tig_route(const Tensor& x, const Tensor& z,
                         const Tensor& anchors_l1, const Tensor& W_r,
                         float beta, float delta, float tau_r, float c,
                         int batch, int seq_len, int d_model, int d_hyp,
                         int n_experts, int top_k, int n_shared,
                         Tensor& route_weights, Tensor& aux_loss) {
    int total = batch * seq_len;
    int dh1 = d_hyp + 1;

    Tensor Wr_x({static_cast<size_t>(total), static_cast<size_t>(n_experts)}, DType::FP32);
    gemm(x.fp32_data(), W_r.fp32_data(), Wr_x.fp32_data(), total, n_experts, d_model);

    const float* zp = z.fp32_data();
    const float* ap = anchors_l1.fp32_data();
    float* rp = Wr_x.fp32_data();

    for (int t = 0; t < total; ++t) {
        for (int e = 0; e < n_experts; ++e) {
            float d2 = dist2_lorentz(zp + t * dh1, ap + e * dh1, dh1, c);
            rp[t * n_experts + e] += beta * (-d2) / tau_r;
        }
    }

    route_weights.fill(0.0f);
    float* rwp = route_weights.fp32_data();

    for (int t = 0; t < total; ++t) {
        int e_star = 0;
        float max_r = rp[t * n_experts];
        for (int e = 1; e < n_experts; ++e) {
            if (rp[t * n_experts + e] > max_r) {
                max_r = rp[t * n_experts + e];
                e_star = e;
            }
        }

        std::vector<int> neighbors;
        for (int e = 0; e < n_experts; ++e) {
            if (e == e_star) continue;
            float d2 = dist2_lorentz(ap + e * dh1, ap + e_star * dh1, dh1, c);
            float dist_val = std::sqrt(std::max(-d2, 0.0f));
            if (dist_val <= delta) neighbors.push_back(e);
        }

        int e2 = -1;
        if (!neighbors.empty()) {
            float max_r2 = -1e30f;
            for (int e : neighbors) {
                if (rp[t * n_experts + e] > max_r2) {
                    max_r2 = rp[t * n_experts + e];
                    e2 = e;
                }
            }
        }

        if (e2 >= 0) {
            float r1 = rp[t * n_experts + e_star];
            float r2 = rp[t * n_experts + e2];
            float m = std::max(r1, r2);
            float e1 = std::exp(r1 - m);
            float e2v = std::exp(r2 - m);
            float Z = e1 + e2v;
            rwp[t * (n_experts + n_shared) + e_star] = e1 / Z;
            rwp[t * (n_experts + n_shared) + e2] = e2v / Z;
        } else {
            rwp[t * (n_experts + n_shared) + e_star] = 1.0f;
        }
        rwp[t * (n_experts + n_shared) + n_experts] = 1.0f;
    }

    aux_loss.fill(0.0f);
}

void NolModel::swiglu_moe_forward(const Tensor& x,
                                  const Tensor& W1s, const Tensor& W2s, const Tensor& W3s,
                                  const std::vector<Tensor>& W1e,
                                  const std::vector<Tensor>& W2e,
                                  const std::vector<Tensor>& W3e,
                                  const Tensor& route_weights,
                                  Tensor& out, int batch, int seq_len,
                                  int d_model, int ffn_hidden,
                                  int n_experts, int top_k, int n_shared) {
    int total = batch * seq_len;
    out.fill(0.0f);

    {
        Tensor h1({static_cast<size_t>(total), static_cast<size_t>(ffn_hidden)}, DType::FP32);
        Tensor h3({static_cast<size_t>(total), static_cast<size_t>(ffn_hidden)}, DType::FP32);
        gemm(x.fp32_data(), W1s.fp32_data(), h1.fp32_data(), total, ffn_hidden, d_model);
        gemm(x.fp32_data(), W3s.fp32_data(), h3.fp32_data(), total, ffn_hidden, d_model);
        h1.silu_inplace();
        h1.mul(h3);
        Tensor ffn_out({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
        gemm(h1.fp32_data(), W2s.fp32_data(), ffn_out.fp32_data(), total, d_model, ffn_hidden);
        out.add(ffn_out);
    }

    const float* rwp = route_weights.fp32_data();
    for (int t = 0; t < total; ++t) {
        for (int e = 0; e < n_experts; ++e) {
            float w = rwp[t * (n_experts + n_shared) + e];
            if (w < 1e-6f) continue;

            const float* xt = x.fp32_data() + t * d_model;
            std::vector<float> h1(ffn_hidden), h3(ffn_hidden);

            const float* w1p = W1e[e].fp32_data();
            const float* w3p = W3e[e].fp32_data();
            for (int f = 0; f < ffn_hidden; ++f) {
                float s1 = 0.0f, s3 = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    s1 += xt[d] * w1p[d * ffn_hidden + f];
                    s3 += xt[d] * w3p[d * ffn_hidden + f];
                }
                h1[f] = s1 / (1.0f + std::exp(-s1));
                h3[f] = s3;
            }

            const float* w2p = W2e[e].fp32_data();
            float* otp = out.fp32_data() + t * d_model;
            for (int d = 0; d < d_model; ++d) {
                float s = 0.0f;
                for (int f = 0; f < ffn_hidden; ++f) {
                    s += h1[f] * h3[f] * w2p[f * d_model + d];
                }
                otp[d] += w * s;
            }
        }
    }
}

void NolModel::head_forward(const Tensor& xN, const Tensor& z_N,
                            const Tensor& W_v, const Tensor& H_v,
                            float lambda_h, float c, float tau_o,
                            Tensor& logits, int batch, int seq_len,
                            int d_model, int d_hyp, int vocab_size) {
    int total = batch * seq_len;
    int dh1 = d_hyp + 1;

    // Safety check: W_v harus valid (tidak null)
    if (W_v.data == nullptr) {
        std::cerr << "[FATAL] W_v_head is null! Call rebuild_tied_embedding() after load.\n";
        return;
    }

    gemm(xN.fp32_data(), W_v.fp32_data(), logits.fp32_data(),
         total, vocab_size, d_model);

    float* lp = logits.fp32_data();
    const float* zp = z_N.fp32_data();
    const float* hvp = H_v.fp32_data();

    for (int t = 0; t < total; ++t) {
        for (int v = 0; v < vocab_size; ++v) {
            float geo_sim = (2.0f / c + 2.0f * linner(
                zp + t * dh1, hvp + v * dh1, dh1)) / tau_o;
            lp[t * vocab_size + v] += lambda_h * geo_sim;
        }
    }
}

// ==========================================
// FULL FORWARD PASS
// ==========================================

void NolModel::forward(const Tensor& token_ids, Tensor& logits, Tensor& aux_loss) {
    int batch = static_cast<int>(token_ids.shape[0]);
    int seq_len = static_cast<int>(token_ids.shape[1]);
    int total = batch * seq_len;
    int d = cfg.d_model;
    int dh1 = cfg.d_hyp + 1;

    float c = std::log(1.0f + std::exp(curvature)) + cfg.c_min;

    Tensor h({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    const float* ep = embed_E.fp32_data();
    float* hp = h.fp32_data();
    const float* tip = token_ids.fp32_data();
    for (int t = 0; t < total; ++t) {
        int tok = static_cast<int>(tip[t]);
        tok = std::max(0, std::min(tok, cfg.vocab_size - 1));
        std::memcpy(hp + t * d, ep + tok * d, d * sizeof(float));
    }

    aux_loss = Tensor({2}, DType::FP32);
    aux_loss.fill(0.0f);

    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& bp = blocks[l];

        Tensor x({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        rmsnorm_forward(h, bp.rmsnorm1_gamma, x, cfg.rmsnorm_eps);

        Tensor a({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        gqa_forward(x, bp.W_q, bp.W_k, bp.W_v, bp.W_o, a,
                    batch, seq_len, cfg.n_heads, cfg.n_kv_heads, cfg.d_head);

        Tensor z_proj({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
        gemm(x.fp32_data(), bp.W_z.fp32_data(), z_proj.fp32_data(), total, dh1, d);
        Tensor z({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
        for (int t = 0; t < total; ++t) {
            exp0(z_proj.fp32_data() + t * dh1, z.fp32_data() + t * dh1, dh1, c);
        }

        int hyp_out_dim = cfg.n_hyp_heads * cfg.d_hyp;
        Tensor u({static_cast<size_t>(total), static_cast<size_t>(hyp_out_dim)}, DType::FP32);
        u.fill(0.0f);
        if (cfg.enable_hga) {
            hga_forward(x, bp.W_q_hyp, bp.W_k_hyp, bp.W_v_hyp, u,
                        batch, seq_len, cfg.n_hyp_heads, cfg.d_hyp, c, cfg.tau_att);
        }

        Tensor m({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        if (cfg.enable_gated_fusion) {
            gated_fusion_forward(a, u, x, bp.W_g, bp.W_u, m, batch, seq_len, d);
        } else {
            m.copy_from(a);
        }

        h.add(m);

        if (cfg.enable_cgl) {
            cgl_forward(h, z, bp.anchors_l1, bp.anchors_l2, bp.parent_map,
                        bp.W_c, bp.gamma, bp.lambda_tilde, cfg.tau_c, c,
                        batch, seq_len, d, cfg.d_hyp,
                        cfg.n_top_anchors, cfg.n_top_anchors * cfg.n_children,
                        cfg.n_children);
        }

        Tensor x2({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        rmsnorm_forward(h, bp.rmsnorm2_gamma, x2, cfg.rmsnorm_eps);

        int n_total_experts = cfg.n_top_anchors + cfg.n_shared_experts;
        Tensor route_w({static_cast<size_t>(total), static_cast<size_t>(n_total_experts)}, DType::FP32);
        Tensor block_aux({2}, DType::FP32);
        block_aux.fill(0.0f);
        if (cfg.enable_tig) {
            tig_route(x2, z, bp.anchors_l1, bp.W_r, bp.beta, cfg.iso_delta,
                      cfg.tau_r, c, batch, seq_len, d, cfg.d_hyp,
                      cfg.n_top_anchors, cfg.top_k_experts, cfg.n_shared_experts,
                      route_w, block_aux);
        } else {
            route_w.fill(0.0f);
            float* rwp = route_w.fp32_data();
            for (int t = 0; t < total; ++t) {
                rwp[t * n_total_experts + cfg.n_top_anchors] = 1.0f;
            }
        }

        Tensor ffn_out({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        if (cfg.enable_moe) {
            swiglu_moe_forward(x2, bp.W1_shared, bp.W2_shared, bp.W3_shared,
                               bp.W1_expert, bp.W2_expert, bp.W3_expert,
                               route_w, ffn_out, batch, seq_len, d, cfg.ffn_hidden,
                               cfg.n_top_anchors, cfg.top_k_experts, cfg.n_shared_experts);
        } else {
            Tensor h1({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
            Tensor h3({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
            gemm(x2.fp32_data(), bp.W1_shared.fp32_data(), h1.fp32_data(), total, cfg.ffn_hidden, d);
            gemm(x2.fp32_data(), bp.W3_shared.fp32_data(), h3.fp32_data(), total, cfg.ffn_hidden, d);
            h1.silu_inplace();
            h1.mul(h3);
            gemm(h1.fp32_data(), bp.W2_shared.fp32_data(), ffn_out.fp32_data(), total, d, cfg.ffn_hidden);
        }
        h.add(ffn_out);
    }

    Tensor xN({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    rmsnorm_forward(h, rmsnorm_final_gamma, xN, cfg.rmsnorm_eps);

    auto& last_bp = blocks[cfg.n_layers - 1];
    Tensor zN_proj({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
    gemm(xN.fp32_data(), last_bp.W_z.fp32_data(), zN_proj.fp32_data(), total, dh1, d);
    Tensor z_N({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
    for (int t = 0; t < total; ++t) {
        exp0(zN_proj.fp32_data() + t * dh1, z_N.fp32_data() + t * dh1, dh1, c);
    }

    logits = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.vocab_size)}, DType::FP32);
    if (cfg.enable_hyp_head) {
        head_forward(xN, z_N, W_v_head, H_v, lambda_h, c, cfg.tau_head,
                     logits, batch, seq_len, d, cfg.d_hyp, cfg.vocab_size);
    } else {
        gemm(xN.fp32_data(), W_v_head.fp32_data(), logits.fp32_data(),
             total, cfg.vocab_size, d);
    }
}

// ==========================================
// UTILITIES
// ==========================================

size_t NolModel::param_count() const {
    size_t count = embed_E.numel + embed_T.numel;
    for (const auto& bp : blocks) {
        count += bp.rmsnorm1_gamma.numel + bp.W_q.numel + bp.W_k.numel;
        count += bp.W_v.numel + bp.W_o.numel;
        count += bp.W_q_hyp.numel + bp.W_k_hyp.numel + bp.W_v_hyp.numel;
        count += bp.W_g.numel + bp.W_u.numel;
        count += bp.W_z.numel + bp.W_c.numel;
        count += bp.anchors_l1.numel + bp.anchors_l2.numel;
        count += bp.W_r.numel + bp.rmsnorm2_gamma.numel;
        count += bp.W1_shared.numel + bp.W2_shared.numel + bp.W3_shared.numel;
        for (int e = 0; e < cfg.n_top_anchors; ++e) {
            count += bp.W1_expert[e].numel + bp.W2_expert[e].numel + bp.W3_expert[e].numel;
        }
    }
    count += rmsnorm_final_gamma.numel + H_v.numel;
    return count;
}

void NolModel::print_summary() const {
    std::cout << "\n\033[1;35m";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║              MODEL NOL - ARCHITECTURE SUMMARY           ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════╣\n\033[0m";
    std::cout << "║  Layers: " << std::setw(3) << cfg.n_layers
              << "  d_model: " << std::setw(4) << cfg.d_model
              << "  Heads: " << cfg.n_heads << "/" << cfg.n_kv_heads
              << "  Hyp: " << cfg.n_hyp_heads << "        ║\n";
    std::cout << "║  Vocab: " << std::setw(5) << cfg.vocab_size
              << "  FFN: " << std::setw(5) << cfg.ffn_hidden
              << "  Seq: " << std::setw(4) << cfg.max_seq_len
              << "  Anchors: " << cfg.n_top_anchors << "+"
              << (cfg.n_top_anchors * cfg.n_children) << "   ║\n";
    std::cout << "║  Total Parameters: " << std::setw(12) << param_count() << "              ║\n";
    std::cout << "║  T1-HGA:" << (cfg.enable_hga ? "ON " : "OFF")
              << "  T2-Fuse:" << (cfg.enable_gated_fusion ? "ON " : "OFF")
              << "  T3-CGL:" << (cfg.enable_cgl ? "ON " : "OFF")
              << "  T4-TIG:" << (cfg.enable_tig ? "ON " : "OFF")
              << "  T5:" << (cfg.enable_hyp_head ? "ON " : "OFF") << " ║\n";
    std::cout << "\033[1;35m╚═══════════════════════════════════════════════════════════╝\033[0m\n\n";
}

} // namespace model
} // namespace nol

// ==========================================
// UNIT TEST MODEL
// ==========================================
static void run_model_tests() {
    using namespace nol::model;
    using namespace nol::config;
    using namespace nol::core;
    std::cout << "[TEST] Memulai unit test model...\n";

    NolConfig cfg;
    cfg.vocab_size = 100;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.d_head = 16;
    cfg.ffn_hidden = 128;
    cfg.max_seq_len = 32;
    cfg.n_hyp_heads = 1;
    cfg.d_hyp = 8;
    cfg.n_top_anchors = 4;
    cfg.n_children = 4;
    cfg.top_k_experts = 2;
    cfg.n_shared_experts = 1;

    // Test 1: Konstruksi dan inisialisasi
    {
        NolModel model(cfg);
        model.initialize_weights(42);
        assert(model.param_count() > 0);
        model.print_summary();
    }

    // Test 2: Forward pass tidak crash
    {
        NolModel model(cfg);
        model.initialize_weights(42);

        int batch = 2, seq = 8;
        Tensor token_ids({static_cast<size_t>(batch), static_cast<size_t>(seq)}, DType::FP32);
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> tok_dist(0, cfg.vocab_size - 1);
        float* tip = token_ids.fp32_data();
        for (int i = 0; i < batch * seq; ++i) {
            tip[i] = static_cast<float>(tok_dist(rng));
        }

        Tensor logits, aux_loss;
        model.forward(token_ids, logits, aux_loss);

        assert(logits.shape[0] == static_cast<size_t>(batch * seq));
        assert(logits.shape[1] == static_cast<size_t>(cfg.vocab_size));

        const float* lp = logits.fp32_data();
        for (size_t i = 0; i < logits.numel; ++i) {
            assert(std::isfinite(lp[i]));
        }
        std::cout << "  Forward pass OK, logits shape: ["
                  << logits.shape[0] << ", " << logits.shape[1] << "]\n";
    }

    // Test 3: Forward dengan layer dimatikan
    {
        NolConfig cfg2 = cfg;
        cfg2.enable_hga = false;
        cfg2.enable_cgl = false;
        cfg2.enable_tig = false;
        cfg2.enable_hyp_head = false;
        cfg2.enable_moe = false;

        NolModel model(cfg2);
        model.initialize_weights(42);

        int batch = 1, seq = 4;
        Tensor token_ids({static_cast<size_t>(batch), static_cast<size_t>(seq)}, DType::FP32);
        token_ids.fill(5.0f);

        Tensor logits, aux_loss;
        model.forward(token_ids, logits, aux_loss);

        const float* lp = logits.fp32_data();
        for (size_t i = 0; i < logits.numel; ++i) {
            assert(std::isfinite(lp[i]));
        }
        std::cout << "  Forward (baseline mode) OK\n";
    }

    // Test 4: Tied embedding check
    {
        NolModel model(cfg);
        model.initialize_weights(42);
        assert(model.W_v_head.data == model.embed_E.data);
        assert(model.W_v_head.owns_memory == false);
    }

    // Test 5: Curvature computation
    {
        NolModel model(cfg);
        float c = std::log(1.0f + std::exp(model.curvature)) + cfg.c_min;
        assert(std::abs(c - cfg.c_init) < 1e-4f);
    }

    // Test 6: rebuild_tied_embedding setelah konstruksi tanpa init
    {
        NolModel model(cfg);
        // Tanpa initialize_weights, W_v_head.data harus null
        assert(model.W_v_head.data == nullptr);
        // Setelah rebuild, harus menunjuk ke embed_E
        model.rebuild_tied_embedding();
        assert(model.W_v_head.data == model.embed_E.data);
        assert(model.W_v_head.owns_memory == false);
    }

    std::cout << "[PASS] ✅ Semua unit test model berhasil!\n\n";
}

#ifdef NOL_TEST_MODEL
int main() {
    run_model_tests();
    return 0;
}
#endif
