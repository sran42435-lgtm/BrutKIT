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

    W_v_head.data = embed_E.data;
    W_v_head.shape = embed_E.shape;
    W_v_head.strides = embed_E.strides;
    W_v_head.numel = embed_E.numel;
    W_v_head.dtype = embed_E.dtype;
    W_v_head.owns_memory = false;
}

void NolModel::rebuild_tied_embedding() {
    W_v_head.data = embed_E.data;
    W_v_head.shape = embed_E.shape;
    W_v_head.strides = embed_E.strides;
    W_v_head.numel = embed_E.numel;
    W_v_head.dtype = embed_E.dtype;
    W_v_head.owns_memory = false;
}

// ==========================================
// COMPONENT FORWARD IMPLEMENTATIONS
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
// FORWARD WITH CACHE (Phase 1 + Phase 2)
// ==========================================

void NolModel::forward_with_cache(const Tensor& token_ids, Tensor& logits,
                                  Tensor& aux_loss, ForwardCache& cache) {
    int batch = static_cast<int>(token_ids.shape[0]);
    int seq_len = static_cast<int>(token_ids.shape[1]);
    int total = batch * seq_len;
    int d = cfg.d_model;
    int dh1 = cfg.d_hyp + 1;

    cache.batch_size = batch;
    cache.seq_len = seq_len;
    cache.total_tokens = total;
    cache.resize(cfg.n_layers);

    float c = std::log(1.0f + std::exp(curvature)) + cfg.c_min;

    // Save input
    cache.token_ids = Tensor(token_ids.shape, DType::FP32);
    cache.token_ids.copy_from(token_ids);

    // Embedding lookup
    cache.h0 = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    const float* ep = embed_E.fp32_data();
    float* hp = cache.h0.fp32_data();
    const float* tip = token_ids.fp32_data();
    for (int t = 0; t < total; ++t) {
        int tok = static_cast<int>(tip[t]);
        tok = std::max(0, std::min(tok, cfg.vocab_size - 1));
        std::memcpy(hp + t * d, ep + tok * d, d * sizeof(float));
    }

    cache.T_lookup = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.d_hyp)}, DType::FP32);
    const float* etp = embed_T.fp32_data();
    float* thp = cache.T_lookup.fp32_data();
    for (int t = 0; t < total; ++t) {
        int tok = static_cast<int>(tip[t]);
        tok = std::max(0, std::min(tok, cfg.vocab_size - 1));
        std::memcpy(thp + t * cfg.d_hyp, etp + tok * cfg.d_hyp, cfg.d_hyp * sizeof(float));
    }

    aux_loss = Tensor({2}, DType::FP32);
    aux_loss.fill(0.0f);

    Tensor h({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    h.copy_from(cache.h0);

    int d_model = cfg.n_heads * cfg.d_head;
    int kv_dim = cfg.n_kv_heads * cfg.d_head;

    for (int l = 0; l < cfg.n_layers; ++l) {
        auto& bp = blocks[l];
        auto& bc = cache.blocks[l];

        // Save input block
        bc.h_in = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        bc.h_in.copy_from(h);

        // RMSNorm1
        bc.x = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        rmsnorm_forward(h, bp.rmsnorm1_gamma, bc.x, cfg.rmsnorm_eps);

        // === GQA Forward with Cache ===
        bc.q_proj = Tensor({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
        bc.k_proj = Tensor({static_cast<size_t>(total), static_cast<size_t>(kv_dim)}, DType::FP32);
        bc.v_proj = Tensor({static_cast<size_t>(total), static_cast<size_t>(kv_dim)}, DType::FP32);
        
        gemm(bc.x.fp32_data(), bp.W_q.fp32_data(), bc.q_proj.fp32_data(), total, d_model, d);
        gemm(bc.x.fp32_data(), bp.W_k.fp32_data(), bc.k_proj.fp32_data(), total, kv_dim, d);
        gemm(bc.x.fp32_data(), bp.W_v.fp32_data(), bc.v_proj.fp32_data(), total, kv_dim, d);

        // Copy to RoPE buffers (RoPE is in-place)
        bc.q_rope = Tensor(bc.q_proj.shape, DType::FP32);
        bc.k_rope = Tensor(bc.k_proj.shape, DType::FP32);
        bc.q_rope.copy_from(bc.q_proj);
        bc.k_rope.copy_from(bc.k_proj);

        bc.q_rope.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        bc.k_rope.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)});
        rope_apply(bc.q_rope, bc.k_rope, seq_len, cfg.d_head, cfg.n_heads, cfg.n_kv_heads, 10000.0f);
        bc.q_rope.reshape({static_cast<size_t>(total), static_cast<size_t>(d_model)});
        bc.k_rope.reshape({static_cast<size_t>(total), static_cast<size_t>(kv_dim)});

        bc.attn_out = Tensor({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
        bc.attn_out.fill(0.0f);
        
        bc.attn_scores = Tensor({static_cast<size_t>(batch * cfg.n_heads), 
                                 static_cast<size_t>(seq_len), 
                                 static_cast<size_t>(seq_len)}, DType::FP32);
        bc.attn_scores.fill(0.0f);

        float scale = 1.0f / std::sqrt(static_cast<float>(cfg.d_head));
        int heads_per_kv = cfg.n_heads / cfg.n_kv_heads;

        for (int b = 0; b < batch; ++b) {
            for (int hh = 0; hh < cfg.n_heads; ++hh) {
                int kv_h = hh / heads_per_kv;
                for (int i = 0; i < seq_len; ++i) {
                    std::vector<float> scores(i + 1);
                    float max_score = -1e30f;
                    for (int j = 0; j <= i; ++j) {
                        float s = 0.0f;
                        for (int dd = 0; dd < cfg.d_head; ++dd) {
                            size_t q_idx = (b * seq_len + i) * d_model + hh * cfg.d_head + dd;
                            size_t k_idx = (b * seq_len + j) * kv_dim + kv_h * cfg.d_head + dd;
                            s += bc.q_rope.fp32_data()[q_idx] * bc.k_rope.fp32_data()[k_idx];
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
                    for (int j = 0; j <= i; ++j) {
                        scores[j] *= inv_sum;
                        bc.attn_scores.fp32_data()[(b * cfg.n_heads + hh) * seq_len * seq_len + i * seq_len + j] = scores[j];
                    }

                    for (int dd = 0; dd < cfg.d_head; ++dd) {
                        float val = 0.0f;
                        for (int j = 0; j <= i; ++j) {
                            size_t v_idx = (b * seq_len + j) * kv_dim + kv_h * cfg.d_head + dd;
                            val += scores[j] * bc.v_proj.fp32_data()[v_idx];
                        }
                        size_t out_idx = (b * seq_len + i) * d_model + hh * cfg.d_head + dd;
                        bc.attn_out.fp32_data()[out_idx] = val;
                    }
                }
            }
        }

        bc.a = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        gemm(bc.attn_out.fp32_data(), bp.W_o.fp32_data(), bc.a.fp32_data(), total, d, d_model);

        // z = exp0(W_z * x)
        bc.z_proj = Tensor({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
        gemm(bc.x.fp32_data(), bp.W_z.fp32_data(), bc.z_proj.fp32_data(), total, dh1, d);
        bc.z = Tensor({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
        for (int t = 0; t < total; ++t) {
            exp0(bc.z_proj.fp32_data() + t * dh1, bc.z.fp32_data() + t * dh1, dh1, c);
        }

        // HGA (T1)
        int hyp_out_dim = cfg.n_hyp_heads * cfg.d_hyp;
        bc.u = Tensor({static_cast<size_t>(total), static_cast<size_t>(hyp_out_dim)}, DType::FP32);
        bc.u.fill(0.0f);
        if (cfg.enable_hga) {
            hga_forward(bc.x, bp.W_q_hyp, bp.W_k_hyp, bp.W_v_hyp, bc.u,
                        batch, seq_len, cfg.n_hyp_heads, cfg.d_hyp, c, cfg.tau_att);
        }

        // Gated Fusion (T2)
        bc.m = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        if (cfg.enable_gated_fusion) {
            gated_fusion_forward(bc.a, bc.u, bc.x, bp.W_g, bp.W_u, bc.m, batch, seq_len, d);
        } else {
            bc.m.copy_from(bc.a);
        }

        h.add(bc.m);

        // CGL (T3)
        bc.h1 = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        bc.h1.copy_from(h);
        if (cfg.enable_cgl) {
            cgl_forward(h, bc.z, bp.anchors_l1, bp.anchors_l2, bp.parent_map,
                        bp.W_c, bp.gamma, bp.lambda_tilde, cfg.tau_c, c,
                        batch, seq_len, d, cfg.d_hyp,
                        cfg.n_top_anchors, cfg.n_top_anchors * cfg.n_children,
                        cfg.n_children);
        }
        bc.h2 = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        bc.h2.copy_from(h);

        // RMSNorm2
        bc.x2 = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        rmsnorm_forward(h, bp.rmsnorm2_gamma, bc.x2, cfg.rmsnorm_eps);

        // TIG routing
        int n_total_experts = cfg.n_top_anchors + cfg.n_shared_experts;
        bc.route_weights = Tensor({static_cast<size_t>(total), static_cast<size_t>(n_total_experts)}, DType::FP32);
        bc.tig_aux = Tensor({2}, DType::FP32);
        bc.tig_aux.fill(0.0f);
        if (cfg.enable_tig) {
            tig_route(bc.x2, bc.z, bp.anchors_l1, bp.W_r, bp.beta, cfg.iso_delta,
                      cfg.tau_r, c, batch, seq_len, d, cfg.d_hyp,
                      cfg.n_top_anchors, cfg.top_k_experts, cfg.n_shared_experts,
                      bc.route_weights, bc.tig_aux);
        } else {
            bc.route_weights.fill(0.0f);
            float* rwp = bc.route_weights.fp32_data();
            for (int t = 0; t < total; ++t) {
                rwp[t * n_total_experts + cfg.n_top_anchors] = 1.0f;
            }
        }

        // === SwiGLU MoE Forward with Cache ===
        bc.ffn_out = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        bc.ffn_out.fill(0.0f);

        bc.h1_shared = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.h3_shared = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.h1_shared_silu = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.gate_shared = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.ffn_shared_out = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);

        gemm(bc.x2.fp32_data(), bp.W1_shared.fp32_data(), bc.h1_shared.fp32_data(), total, cfg.ffn_hidden, d);
        gemm(bc.x2.fp32_data(), bp.W3_shared.fp32_data(), bc.h3_shared.fp32_data(), total, cfg.ffn_hidden, d);
        bc.h1_shared_silu.copy_from(bc.h1_shared);
        bc.h1_shared_silu.silu_inplace();
        bc.gate_shared.copy_from(bc.h1_shared_silu);
        bc.gate_shared.mul(bc.h3_shared);
        gemm(bc.gate_shared.fp32_data(), bp.W2_shared.fp32_data(), bc.ffn_shared_out.fp32_data(), total, d, cfg.ffn_hidden);
        bc.ffn_out.add(bc.ffn_shared_out);

        if (cfg.enable_moe) {
            bc.expert_h1.resize(cfg.n_top_anchors);
            bc.expert_h3.resize(cfg.n_top_anchors);
            bc.expert_h1_silu.resize(cfg.n_top_anchors);
            bc.expert_gate.resize(cfg.n_top_anchors);
            bc.expert_out.resize(cfg.n_top_anchors);

            const float* rwp = bc.route_weights.fp32_data();
            for (int e = 0; e < cfg.n_top_anchors; ++e) {
                bc.expert_h1[e] = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
                bc.expert_h3[e] = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
                bc.expert_h1_silu[e] = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
                bc.expert_gate[e] = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
                bc.expert_out[e] = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
                bc.expert_h1[e].fill(0.0f);
                bc.expert_h3[e].fill(0.0f);
                bc.expert_h1_silu[e].fill(0.0f);
                bc.expert_gate[e].fill(0.0f);
                bc.expert_out[e].fill(0.0f);

                bool expert_active = false;
                for (int t = 0; t < total; ++t) {
                    if (rwp[t * n_total_experts + e] > 1e-6f) {
                        expert_active = true;
                        break;
                    }
                }

                if (expert_active) {
                    gemm(bc.x2.fp32_data(), bp.W1_expert[e].fp32_data(), bc.expert_h1[e].fp32_data(), total, cfg.ffn_hidden, d);
                    gemm(bc.x2.fp32_data(), bp.W3_expert[e].fp32_data(), bc.expert_h3[e].fp32_data(), total, cfg.ffn_hidden, d);
                    bc.expert_h1_silu[e].copy_from(bc.expert_h1[e]);
                    bc.expert_h1_silu[e].silu_inplace();
                    bc.expert_gate[e].copy_from(bc.expert_h1_silu[e]);
                    bc.expert_gate[e].mul(bc.expert_h3[e]);
                    gemm(bc.expert_gate[e].fp32_data(), bp.W2_expert[e].fp32_data(), bc.expert_out[e].fp32_data(), total, d, cfg.ffn_hidden);

                    for (int t = 0; t < total; ++t) {
                        float w = rwp[t * n_total_experts + e];
                        if (w > 1e-6f) {
                            for (int dd = 0; dd < d; ++dd) {
                                bc.ffn_out.fp32_data()[t * d + dd] += w * bc.expert_out[e].fp32_data()[t * d + dd];
                            }
                        }
                    }
                }
            }
        }

        h.add(bc.ffn_out);

        bc.h_out = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
        bc.h_out.copy_from(h);
    }

    // Final RMSNorm
    cache.h_final = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    cache.h_final.copy_from(h);
    cache.xN = Tensor({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    rmsnorm_forward(h, rmsnorm_final_gamma, cache.xN, cfg.rmsnorm_eps);

    // z_N for T5
    auto& last_bp = blocks[cfg.n_layers - 1];
    cache.zN_proj = Tensor({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
    gemm(cache.xN.fp32_data(), last_bp.W_z.fp32_data(), cache.zN_proj.fp32_data(), total, dh1, d);
    cache.zN = Tensor({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
    for (int t = 0; t < total; ++t) {
        exp0(cache.zN_proj.fp32_data() + t * dh1, cache.zN.fp32_data() + t * dh1, dh1, c);
    }

    // Head
    cache.logits = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.vocab_size)}, DType::FP32);
    if (cfg.enable_hyp_head) {
        head_forward(cache.xN, cache.zN, W_v_head, H_v, lambda_h, c, cfg.tau_head,
                     cache.logits, batch, seq_len, d, cfg.d_hyp, cfg.vocab_size);
    } else {
        gemm(cache.xN.fp32_data(), W_v_head.fp32_data(), cache.logits.fp32_data(),
             total, cfg.vocab_size, d);
    }

    logits = Tensor(cache.logits.shape, DType::FP32);
    logits.copy_from(cache.logits);
    cache.aux_loss = Tensor({2}, DType::FP32);
    cache.aux_loss.copy_from(aux_loss);
}

// ==========================================
// BACKWARD PASS - FUNDAMENTAL OPS (Phase 1)
// ==========================================

void NolModel::rmsnorm_backward(const Tensor& x, const Tensor& gamma,
                                 const Tensor& grad_out,
                                 Tensor& grad_x, Tensor& grad_gamma,
                                 float eps) {
    assert(x.dtype == DType::FP32);
    size_t total = x.numel;
    size_t d = gamma.numel;
    size_t rows = total / d;

    const float* xp = x.fp32_data();
    const float* gp = gamma.fp32_data();
    const float* gop = grad_out.fp32_data();
    float* gxp = grad_x.fp32_data();
    float* ggp = grad_gamma.fp32_data();

    std::memset(ggp, 0, d * sizeof(float));

    for (size_t r = 0; r < rows; ++r) {
        const float* row = xp + r * d;
        const float* go_row = gop + r * d;
        float* gx_row = gxp + r * d;

        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) sum_sq += row[j] * row[j];
        float mean_sq = sum_sq / static_cast<float>(d);
        float s = 1.0f / std::sqrt(mean_sq + eps);
        float s3 = s * s * s;

        float sum_go_y = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            sum_go_y += go_row[j] * gp[j] * row[j];
        }

        for (size_t j = 0; j < d; ++j) {
            gx_row[j] = s * gp[j] * go_row[j] - s3 * row[j] * sum_go_y / static_cast<float>(d);
            ggp[j] += go_row[j] * row[j] * s;
        }
    }
}

void NolModel::linear_backward(const Tensor& x, const Tensor& grad_y,
                                Tensor& grad_x, Tensor& grad_W) {
    int B = static_cast<int>(x.shape[0]);
    int K = static_cast<int>(x.shape[1]);
    int N = static_cast<int>(grad_y.shape[1]);

    const float* xp = x.fp32_data();
    const float* gyp = grad_y.fp32_data();
    float* gxp = grad_x.fp32_data();
    float* gwp = grad_W.fp32_data();

    std::memset(gwp, 0, K * N * sizeof(float));
    for (int b = 0; b < B; ++b) {
        for (int k = 0; k < K; ++k) {
            float x_val = xp[b * K + k];
            for (int n = 0; n < N; ++n) {
                gwp[k * N + n] += x_val * gyp[b * N + n];
            }
        }
    }
    // Note: grad_x tidak dihitung karena tidak ada W di signature
    if (gxp) std::memset(gxp, 0, B * K * sizeof(float));
}

void NolModel::silu_backward(const Tensor& x, const Tensor& grad_y, Tensor& grad_x) {
    size_t n = x.numel;
    const float* xp = x.fp32_data();
    const float* gyp = grad_y.fp32_data();
    float* gxp = grad_x.fp32_data();

    for (size_t i = 0; i < n; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-xp[i]));
        float dsig = sig * (1.0f + xp[i] * (1.0f - sig));
        gxp[i] = gyp[i] * dsig;
    }
}

void NolModel::sigmoid_backward(const Tensor& y, const Tensor& grad_y, Tensor& grad_x) {
    size_t n = y.numel;
    const float* yp = y.fp32_data();
    const float* gyp = grad_y.fp32_data();
    float* gxp = grad_x.fp32_data();

    for (size_t i = 0; i < n; ++i) {
        gxp[i] = gyp[i] * yp[i] * (1.0f - yp[i]);
    }
}

// ==========================================
// BACKWARD PASS - PHASE 2 (GQA + SwiGLU)
// ==========================================

void NolModel::gqa_backward(const Tensor& x, const Tensor& W_q, const Tensor& W_k,
                             const Tensor& W_v, const Tensor& W_o,
                             const Tensor& grad_out, const BlockCache& cache,
                             Tensor& grad_x,
                             Tensor& grad_W_q, Tensor& grad_W_k,
                             Tensor& grad_W_v, Tensor& grad_W_o,
                             int batch, int seq_len, int n_heads,
                             int n_kv_heads, int d_head) {
    int d_model = n_heads * d_head;
    int kv_dim = n_kv_heads * d_head;
    int total = batch * seq_len;

    // 1. Backward through W_o projection: a = attn_out @ W_o
    // grad_attn_out = grad_out @ W_o^T
    // grad_W_o = attn_out^T @ grad_out
    Tensor grad_attn_out({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
    grad_attn_out.fill(0.0f);
    grad_W_o.fill(0.0f);

    // grad_attn_out[t, d] = sum_k grad_out[t, k] * W_o[k, d]
    // But W_o is [d, d] so W_o^T is [d, d]
    // Wait - check shapes: attn_out is [total, d_model], W_o is [d_model, d_model] (since d == d_model)
    // a[t, k] = sum_d attn_out[t, d] * W_o[d, k]
    // grad_attn_out[t, d] = sum_k grad_out[t, k] * W_o[d, k]
    for (int t = 0; t < total; ++t) {
        for (int dd = 0; dd < d_model; ++dd) {
            float sum = 0.0f;
            for (int k = 0; k < d_model; ++k) {
                sum += grad_out.fp32_data()[t * d_model + k] * W_o.fp32_data()[dd * d_model + k];
            }
            grad_attn_out.fp32_data()[t * d_model + dd] = sum;
        }
    }

    // grad_W_o[d, k] = sum_t attn_out[t, d] * grad_out[t, k]
    for (int dd = 0; dd < d_model; ++dd) {
        for (int k = 0; k < d_model; ++k) {
            float sum = 0.0f;
            for (int t = 0; t < total; ++t) {
                sum += cache.attn_out.fp32_data()[t * d_model + dd] * grad_out.fp32_data()[t * d_model + k];
            }
            grad_W_o.fp32_data()[dd * d_model + k] = sum;
        }
    }

    // 2. Backward through attention: attn_out[i, h, d] = sum_j scores[i, h, j] * V[j, kv_h, d]
    // grad_scores[i, h, j] = sum_d grad_attn_out[i, h, d] * V[j, kv_h, d]
    // grad_V[j, kv_h, d] = sum_i sum_h scores[i, h, j] * grad_attn_out[i, h, d]
    Tensor grad_scores({static_cast<size_t>(batch * n_heads), 
                        static_cast<size_t>(seq_len), 
                        static_cast<size_t>(seq_len)}, DType::FP32);
    grad_scores.fill(0.0f);

    Tensor grad_v_proj({static_cast<size_t>(total), static_cast<size_t>(kv_dim)}, DType::FP32);
    grad_v_proj.fill(0.0f);

    int heads_per_kv = n_heads / n_kv_heads;

    for (int b = 0; b < batch; ++b) {
        for (int hh = 0; hh < n_heads; ++hh) {
            int kv_h = hh / heads_per_kv;
            for (int i = 0; i < seq_len; ++i) {
                // grad_scores[i, j] for j <= i
                for (int j = 0; j <= i; ++j) {
                    float gs = 0.0f;
                    for (int dd = 0; dd < d_head; ++dd) {
                        size_t v_idx = (b * seq_len + j) * kv_dim + kv_h * d_head + dd;
                        size_t out_idx = (b * seq_len + i) * d_model + hh * d_head + dd;
                        gs += grad_attn_out.fp32_data()[out_idx] * cache.v_proj.fp32_data()[v_idx];
                    }
                    grad_scores.fp32_data()[(b * n_heads + hh) * seq_len * seq_len + i * seq_len + j] = gs;
                }

                // grad_V contribution from this (i, h)
                for (int j = 0; j <= i; ++j) {
                    float s = cache.attn_scores.fp32_data()[(b * n_heads + hh) * seq_len * seq_len + i * seq_len + j];
                    for (int dd = 0; dd < d_head; ++dd) {
                        size_t v_idx = (b * seq_len + j) * kv_dim + kv_h * d_head + dd;
                        size_t out_idx = (b * seq_len + i) * d_model + hh * d_head + dd;
                        grad_v_proj.fp32_data()[v_idx] += s * grad_attn_out.fp32_data()[out_idx];
                    }
                }
            }
        }
    }

    // 3. Backward through softmax: logits -> scores
    // scores[i, j] = softmax(logits[i, j]) for j <= i (causal mask)
    // grad_logits[i, j] = scores[i, j] * (grad_scores[i, j] - sum_k scores[i, k] * grad_scores[i, k])
    Tensor grad_logits_pre_rope({static_cast<size_t>(batch * n_heads), 
                                 static_cast<size_t>(seq_len), 
                                 static_cast<size_t>(seq_len)}, DType::FP32);
    grad_logits_pre_rope.fill(0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int hh = 0; hh < n_heads; ++hh) {
            for (int i = 0; i < seq_len; ++i) {
                float sum_k = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    size_t idx = (b * n_heads + hh) * seq_len * seq_len + i * seq_len + j;
                    sum_k += cache.attn_scores.fp32_data()[idx] * grad_scores.fp32_data()[idx];
                }
                for (int j = 0; j <= i; ++j) {
                    size_t idx = (b * n_heads + hh) * seq_len * seq_len + i * seq_len + j;
                    float s = cache.attn_scores.fp32_data()[idx];
                    float gs = grad_scores.fp32_data()[idx];
                    grad_logits_pre_rope.fp32_data()[idx] = s * (gs - sum_k);
                }
            }
        }
    }

    // 4. Backward through scale and Q·K dot product
    // logits[i, j] = (Q[i, h] · K[j, kv_h]) / sqrt(d_head)
    // grad_Q[i, h, d] = sum_j grad_logits[i, j] * K[j, kv_h, d] / sqrt(d_head)
    // grad_K[j, kv_h, d] = sum_i sum_h grad_logits[i, j] * Q[i, h, d] / sqrt(d_head)
    float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    Tensor grad_q_rope({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
    Tensor grad_k_rope({static_cast<size_t>(total), static_cast<size_t>(kv_dim)}, DType::FP32);
    grad_q_rope.fill(0.0f);
    grad_k_rope.fill(0.0f);

    for (int b = 0; b < batch; ++b) {
        for (int hh = 0; hh < n_heads; ++hh) {
            int kv_h = hh / heads_per_kv;
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j <= i; ++j) {
                    float gl = grad_logits_pre_rope.fp32_data()[(b * n_heads + hh) * seq_len * seq_len + i * seq_len + j];
                    gl *= scale;
                    for (int dd = 0; dd < d_head; ++dd) {
                        size_t q_idx = (b * seq_len + i) * d_model + hh * d_head + dd;
                        size_t k_idx = (b * seq_len + j) * kv_dim + kv_h * d_head + dd;
                        grad_q_rope.fp32_data()[q_idx] += gl * cache.k_rope.fp32_data()[k_idx];
                        grad_k_rope.fp32_data()[k_idx] += gl * cache.q_rope.fp32_data()[q_idx];
                    }
                }
            }
        }
    }

    // 5. Backward through RoPE (inverse rotation)
    // RoPE applies rotation R(theta). Backward is R(-theta) (transpose = inverse for rotation).
    Tensor grad_q_proj({static_cast<size_t>(total), static_cast<size_t>(d_model)}, DType::FP32);
    Tensor grad_k_proj({static_cast<size_t>(total), static_cast<size_t>(kv_dim)}, DType::FP32);
    grad_q_proj.copy_from(grad_q_rope);
    grad_k_proj.copy_from(grad_k_rope);

    int half = d_head / 2;
    float* gqp = grad_q_proj.fp32_data();
    float* gkp = grad_k_proj.fp32_data();

    for (int b = 0; b < batch; ++b) {
        for (int pos = 0; pos < seq_len; ++pos) {
            // Apply inverse RoPE to Q
            for (int hh = 0; hh < n_heads; ++hh) {
                size_t offset = (b * seq_len + pos) * d_model + hh * d_head;
                for (int i = 0; i < half; ++i) {
                    float theta = static_cast<float>(pos) / std::pow(10000.0f, 2.0f * i / d_head);
                    float cos_t = std::cos(theta);
                    float sin_t = std::sin(-theta);  // Inverse: use -sin
                    float q0 = gqp[offset + 2 * i];
                    float q1 = gqp[offset + 2 * i + 1];
                    gqp[offset + 2 * i] = q0 * cos_t + q1 * sin_t;  // R^T
                    gqp[offset + 2 * i + 1] = -q0 * sin_t + q1 * cos_t;
                }
            }
            // Apply inverse RoPE to K
            for (int hh = 0; hh < n_kv_heads; ++hh) {
                size_t offset = (b * seq_len + pos) * kv_dim + hh * d_head;
                for (int i = 0; i < half; ++i) {
                    float theta = static_cast<float>(pos) / std::pow(10000.0f, 2.0f * i / d_head);
                    float cos_t = std::cos(theta);
                    float sin_t = std::sin(-theta);
                    float k0 = gkp[offset + 2 * i];
                    float k1 = gkp[offset + 2 * i + 1];
                    gkp[offset + 2 * i] = k0 * cos_t + k1 * sin_t;
                    gkp[offset + 2 * i + 1] = -k0 * sin_t + k1 * cos_t;
                }
            }
        }
    }

    // 6. Backward through QKV projections: Q = x @ W_q, K = x @ W_k, V = x @ W_v
    // grad_x = grad_Q @ W_q^T + grad_K @ W_k^T + grad_V @ W_v^T
    // grad_W_q = x^T @ grad_Q, etc.
    grad_x.fill(0.0f);
    grad_W_q.fill(0.0f);
    grad_W_k.fill(0.0f);
    grad_W_v.fill(0.0f);

    // grad_W_q[i, j] = sum_t x[t, i] * grad_Q[t, j]
    // grad_x[t, i] += sum_j grad_Q[t, j] * W_q[i, j]
    for (int t = 0; t < total; ++t) {
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad_W_q.fp32_data()[i * d_model + j] += x.fp32_data()[t * d_model + i] * grad_q_proj.fp32_data()[t * d_model + j];
                grad_x.fp32_data()[t * d_model + i] += grad_q_proj.fp32_data()[t * d_model + j] * W_q.fp32_data()[i * d_model + j];
            }
            for (int j = 0; j < kv_dim; ++j) {
                grad_W_k.fp32_data()[i * kv_dim + j] += x.fp32_data()[t * d_model + i] * grad_k_proj.fp32_data()[t * kv_dim + j];
                grad_x.fp32_data()[t * d_model + i] += grad_k_proj.fp32_data()[t * kv_dim + j] * W_k.fp32_data()[i * kv_dim + j];
                grad_W_v.fp32_data()[i * kv_dim + j] += x.fp32_data()[t * d_model + i] * grad_v_proj.fp32_data()[t * kv_dim + j];
                grad_x.fp32_data()[t * d_model + i] += grad_v_proj.fp32_data()[t * kv_dim + j] * W_v.fp32_data()[i * kv_dim + j];
            }
        }
    }
}

void NolModel::swiglu_moe_backward(const Tensor& x,
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
                                    int n_experts) {
    int total = batch * seq_len;
    int n_total_experts = n_experts + 1;  // n_shared = 1

    grad_x.fill(0.0f);
    grad_W1s.fill(0.0f);
    grad_W2s.fill(0.0f);
    grad_W3s.fill(0.0f);

    // === SHARED EXPERT BACKWARD ===
    // ffn_shared_out = gate_shared @ W2s^T
    // gate_shared = SiLU(h1_shared) * h3_shared
    // h1_shared = x @ W1s, h3_shared = x @ W3s

    // 1. grad_gate_shared = grad_ffn_shared_out @ W2s
    // Note: in forward, ffn_out[t, d] = sum_f gate[t, f] * W2[f, d]
    // So grad_gate[t, f] = sum_d grad_out[t, d] * W2[f, d]
    Tensor grad_gate_shared({static_cast<size_t>(total), static_cast<size_t>(ffn_hidden)}, DType::FP32);
    grad_gate_shared.fill(0.0f);

    for (int t = 0; t < total; ++t) {
        for (int f = 0; f < ffn_hidden; ++f) {
            float sum = 0.0f;
            for (int dd = 0; dd < d_model; ++dd) {
                sum += grad_out.fp32_data()[t * d_model + dd] * W2s.fp32_data()[f * d_model + dd];
            }
            grad_gate_shared.fp32_data()[t * ffn_hidden + f] = sum;
        }
    }

    // 2. grad_W2s[f, d] = sum_t gate[t, f] * grad_out[t, d]
    for (int f = 0; f < ffn_hidden; ++f) {
        for (int dd = 0; dd < d_model; ++dd) {
            float sum = 0.0f;
            for (int t = 0; t < total; ++t) {
                sum += cache.gate_shared.fp32_data()[t * ffn_hidden + f] * grad_out.fp32_data()[t * d_model + dd];
            }
            grad_W2s.fp32_data()[f * d_model + dd] = sum;
        }
    }

    // 3. Backward through gate = SiLU(h1) * h3
    // grad_h1[t, f] = grad_gate[t, f] * h3[t, f] * silu'(h1[t, f])
    // grad_h3[t, f] = grad_gate[t, f] * silu(h1[t, f])
    Tensor grad_h1_shared({static_cast<size_t>(total), static_cast<size_t>(ffn_hidden)}, DType::FP32);
    Tensor grad_h3_shared({static_cast<size_t>(total), static_cast<size_t>(ffn_hidden)}, DType::FP32);

    for (int t = 0; t < total; ++t) {
        for (int f = 0; f < ffn_hidden; ++f) {
            float h1_val = cache.h1_shared.fp32_data()[t * ffn_hidden + f];
            float sig = 1.0f / (1.0f + std::exp(-h1_val));
            float silu_val = h1_val * sig;
            float silu_deriv = sig * (1.0f + h1_val * (1.0f - sig));

            float h3_val = cache.h3_shared.fp32_data()[t * ffn_hidden + f];
            float gg = grad_gate_shared.fp32_data()[t * ffn_hidden + f];

            grad_h1_shared.fp32_data()[t * ffn_hidden + f] = gg * h3_val * silu_deriv;
            grad_h3_shared.fp32_data()[t * ffn_hidden + f] = gg * silu_val;
        }
    }

    // 4. Backward through h1 = x @ W1s, h3 = x @ W3s
    // grad_W1s[i, f] = sum_t x[t, i] * grad_h1[t, f]
    // grad_W3s[i, f] = sum_t x[t, i] * grad_h3[t, f]
    // grad_x_shared[t, i] = sum_f grad_h1[t, f] * W1s[i, f] + grad_h3[t, f] * W3s[i, f]
    for (int t = 0; t < total; ++t) {
        for (int i = 0; i < d_model; ++i) {
            for (int f = 0; f < ffn_hidden; ++f) {
                float gh1 = grad_h1_shared.fp32_data()[t * ffn_hidden + f];
                float gh3 = grad_h3_shared.fp32_data()[t * ffn_hidden + f];
                grad_W1s.fp32_data()[i * ffn_hidden + f] += x.fp32_data()[t * d_model + i] * gh1;
                grad_W3s.fp32_data()[i * ffn_hidden + f] += x.fp32_data()[t * d_model + i] * gh3;
                grad_x.fp32_data()[t * d_model + i] += gh1 * W1s.fp32_data()[i * ffn_hidden + f];
                grad_x.fp32_data()[t * d_model + i] += gh3 * W3s.fp32_data()[i * ffn_hidden + f];
            }
        }
    }

    // === ROUTED EXPERTS BACKWARD ===
    grad_W1e.resize(n_experts);
    grad_W2e.resize(n_experts);
    grad_W3e.resize(n_experts);

    // Safety check: apakah expert weights tersedia?
    bool experts_available = !W1e.empty() && !W2e.empty() && !W3e.empty() && 
                              static_cast<int>(W1e.size()) >= n_experts &&
                              static_cast<int>(W2e.size()) >= n_experts &&
                              static_cast<int>(W3e.size()) >= n_experts;

    if (!experts_available || n_experts == 0) {
        for (int e = 0; e < n_experts; ++e) {
            grad_W1e[e] = Tensor({static_cast<size_t>(d_model), static_cast<size_t>(ffn_hidden)}, DType::FP32);
            grad_W2e[e] = Tensor({static_cast<size_t>(ffn_hidden), static_cast<size_t>(d_model)}, DType::FP32);
            grad_W3e[e] = Tensor({static_cast<size_t>(d_model), static_cast<size_t>(ffn_hidden)}, DType::FP32);
            grad_W1e[e].fill(0.0f);
            grad_W2e[e].fill(0.0f);
            grad_W3e[e].fill(0.0f);
        }
        return;
    }

    for (int e = 0; e < n_experts; ++e) {
        grad_W1e[e] = Tensor(W1e[e].shape, DType::FP32);
        grad_W2e[e] = Tensor(W2e[e].shape, DType::FP32);
        grad_W3e[e] = Tensor(W3e[e].shape, DType::FP32);
        grad_W1e[e].fill(0.0f);
        grad_W2e[e].fill(0.0f);
        grad_W3e[e].fill(0.0f);
    }

    if (n_experts == 0) return;

    const float* rwp = route_weights.fp32_data();

    for (int e = 0; e < n_experts; ++e) {
        // Check if expert has cache
        if (cache.expert_out.empty() || cache.expert_gate.empty()) continue;
        if (e >= static_cast<int>(cache.expert_out.size())) continue;
        if (e >= static_cast<int>(cache.expert_gate.size())) continue;
        if (e >= static_cast<int>(cache.expert_h1.size())) continue;
        if (e >= static_cast<int>(cache.expert_h3.size())) continue;

        for (int t = 0; t < total; ++t) {
            float w = rwp[t * n_total_experts + e];
            if (w < 1e-6f) continue;

            // grad_out_exp[t, d] = w * grad_out[t, d]
            // grad_gate_exp[t, f] = sum_d grad_out_exp[t, d] * W2e[e][f, d]
            for (int f = 0; f < ffn_hidden; ++f) {
                float gg = 0.0f;
                for (int dd = 0; dd < d_model; ++dd) {
                    gg += w * grad_out.fp32_data()[t * d_model + dd] * W2e[e].fp32_data()[f * d_model + dd];
                }

                // Backward through gate = SiLU(h1) * h3
                float h1_val = cache.expert_h1[e].fp32_data()[t * ffn_hidden + f];
                float sig = 1.0f / (1.0f + std::exp(-h1_val));
                float silu_val = h1_val * sig;
                float silu_deriv = sig * (1.0f + h1_val * (1.0f - sig));
                float h3_val = cache.expert_h3[e].fp32_data()[t * ffn_hidden + f];

                float gh1 = gg * h3_val * silu_deriv;
                float gh3 = gg * silu_val;

                // grad_W2e[f, d] += w * gate[t, f] * grad_out[t, d]
                for (int dd = 0; dd < d_model; ++dd) {
                    grad_W2e[e].fp32_data()[f * d_model + dd] += 
                        w * cache.expert_gate[e].fp32_data()[t * ffn_hidden + f] * grad_out.fp32_data()[t * d_model + dd];
                }

                // grad_W1e, grad_W3e, grad_x from this expert
                for (int i = 0; i < d_model; ++i) {
                    grad_W1e[e].fp32_data()[i * ffn_hidden + f] += x.fp32_data()[t * d_model + i] * gh1;
                    grad_W3e[e].fp32_data()[i * ffn_hidden + f] += x.fp32_data()[t * d_model + i] * gh3;
                    grad_x.fp32_data()[t * d_model + i] += gh1 * W1e[e].fp32_data()[i * ffn_hidden + f];
                    grad_x.fp32_data()[t * d_model + i] += gh3 * W3e[e].fp32_data()[i * ffn_hidden + f];
                }
            }
        }
    }
}

// ==========================================
// BACKWARD PASS - STUBS (Phase 3 - to be implemented)
// ==========================================

void NolModel::hga_backward(const Tensor& x, const Tensor& W_q_hyp,
                             const Tensor& W_k_hyp, const Tensor& W_v_hyp,
                             const Tensor& grad_out, const BlockCache& cache,
                             Tensor& grad_x,
                             Tensor& grad_W_q_hyp, Tensor& grad_W_k_hyp,
                             Tensor& grad_W_v_hyp,
                             int batch, int seq_len, int n_hyp_heads,
                             int d_hyp, float c, float tau) {
    grad_x.fill(0.0f);
    grad_W_q_hyp.fill(0.0f);
    grad_W_k_hyp.fill(0.0f);
    grad_W_v_hyp.fill(0.0f);
}

void NolModel::gated_fusion_backward(const Tensor& a, const Tensor& u,
                                      const Tensor& x, const Tensor& W_g,
                                      const Tensor& W_u, const Tensor& grad_m,
                                      const BlockCache& cache,
                                      Tensor& grad_a, Tensor& grad_u, Tensor& grad_x,
                                      Tensor& grad_W_g, Tensor& grad_W_u,
                                      int batch, int seq_len, int d_model) {
    grad_a.fill(0.0f);
    grad_u.fill(0.0f);
    grad_x.fill(0.0f);
    grad_W_g.fill(0.0f);
    grad_W_u.fill(0.0f);
}

void NolModel::cgl_backward(const Tensor& h, const Tensor& z,
                             const Tensor& grad_h2, const BlockCache& cache,
                             const Tensor& W_c, float gamma, float lambda_tilde,
                             float tau_c, float c,
                             Tensor& grad_h, Tensor& grad_z,
                             Tensor& grad_W_c, Tensor& grad_anchors_l1,
                             int batch, int seq_len, int d_model, int d_hyp,
                             int n_l1, int n_children) {
    grad_h.fill(0.0f);
    grad_z.fill(0.0f);
    grad_W_c.fill(0.0f);
    grad_anchors_l1.fill(0.0f);
}

void NolModel::tig_backward(const Tensor& x, const Tensor& z,
                             const Tensor& grad_ffn_in, const BlockCache& cache,
                             Tensor& grad_x, Tensor& grad_z,
                             Tensor& grad_W_r,
                             int batch, int seq_len, int d_model, int d_hyp,
                             int n_experts, float c, float tau_r, float beta) {
    grad_x.fill(0.0f);
    grad_z.fill(0.0f);
    grad_W_r.fill(0.0f);
}

void NolModel::head_backward(const Tensor& xN, const Tensor& z_N,
                              const Tensor& W_v, const Tensor& H_v,
                              const Tensor& grad_logits,
                              Tensor& grad_xN, Tensor& grad_zN,
                              Tensor& grad_W_v, Tensor& grad_H_v,
                              float lambda_h, float c, float tau_o,
                              int batch, int seq_len, int d_model,
                              int d_hyp, int vocab_size) {
    grad_xN.fill(0.0f);
    grad_zN.fill(0.0f);
    grad_W_v.fill(0.0f);
    grad_H_v.fill(0.0f);
}

// ==========================================
// BACKWARD ENTRY POINT (Phase 2 skeleton)
// ==========================================

void NolModel::backward(const Tensor& grad_logits, const ForwardCache& cache,
                         GradientAccumulator& grad_acc) {
    int total = cache.total_tokens;
    int d = cfg.d_model;
    int dh1 = cfg.d_hyp + 1;
    int batch = cache.batch_size;
    int seq_len = cache.seq_len;

    float c = std::log(1.0f + std::exp(curvature)) + cfg.c_min;

    // === Backward dari Head ===
    Tensor grad_xN({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    Tensor grad_zN({static_cast<size_t>(total), static_cast<size_t>(dh1)}, DType::FP32);
    Tensor grad_W_v({static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(d)}, DType::FP32);
    Tensor grad_H_v({static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(dh1)}, DType::FP32);
    head_backward(cache.xN, cache.zN, W_v_head, H_v, grad_logits,
                  grad_xN, grad_zN, grad_W_v, grad_H_v,
                  lambda_h, c, cfg.tau_head, batch, seq_len, d, cfg.d_hyp, cfg.vocab_size);

    // === Backward Final RMSNorm ===
    Tensor grad_h_final({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    Tensor grad_rmsnorm_final({static_cast<size_t>(d)}, DType::FP32);
    rmsnorm_backward(cache.h_final, rmsnorm_final_gamma, grad_xN,
                     grad_h_final, grad_rmsnorm_final, cfg.rmsnorm_eps);

    // === Backward through blocks (reverse order) ===
    Tensor grad_h({static_cast<size_t>(total), static_cast<size_t>(d)}, DType::FP32);
    grad_h.copy_from(grad_h_final);

    for (int l = cfg.n_layers - 1; l >= 0; --l) {
        // Stubs return zero for now - will be wired up in Phase 3-4
    }
}

void NolModel::accumulate_gradients(const ForwardCache& cache,
                                     const std::vector<Tensor>& grad_embed_E,
                                     const std::vector<Tensor>& grad_embed_T,
                                     const std::vector<std::vector<Tensor>>& grad_blocks,
                                     const Tensor& grad_rmsnorm_final,
                                     const Tensor& grad_H_v,
                                     GradientAccumulator& grad_acc) const {
    grad_acc.resize(param_count());
    grad_acc.zero();

    size_t offset = 0;
    auto add_tensor = [&](const Tensor& grad) {
        if (grad.numel == 0) return;
        std::memcpy(grad_acc.grads.data() + offset, grad.fp32_data(), grad.numel * sizeof(float));
        offset += grad.numel;
    };

    if (!grad_embed_E.empty()) add_tensor(grad_embed_E[0]);
    else offset += embed_E.numel;
    if (!grad_embed_T.empty()) add_tensor(grad_embed_T[0]);
    else offset += embed_T.numel;

    for (size_t l = 0; l < grad_blocks.size(); ++l) {
        for (const auto& t : grad_blocks[l]) {
            add_tensor(t);
        }
    }

    add_tensor(grad_rmsnorm_final);
    add_tensor(grad_H_v);

    assert(offset == param_count());
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

    // Test 6: rebuild_tied_embedding
    {
        NolModel model(cfg);
        assert(model.W_v_head.data == nullptr);
        model.rebuild_tied_embedding();
        assert(model.W_v_head.data == model.embed_E.data);
        assert(model.W_v_head.owns_memory == false);
    }

    // Test 7: forward_with_cache
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

        ForwardCache cache;
        Tensor logits, aux_loss;
        model.forward_with_cache(token_ids, logits, aux_loss, cache);

        assert(cache.batch_size == batch);
        assert(cache.seq_len == seq);
        assert(cache.total_tokens == batch * seq);
        assert(static_cast<int>(cache.blocks.size()) == cfg.n_layers);

        // Verify cache tensors exist
        assert(cache.blocks[0].q_proj.numel > 0);
        assert(cache.blocks[0].k_proj.numel > 0);
        assert(cache.blocks[0].v_proj.numel > 0);
        assert(cache.blocks[0].attn_scores.numel > 0);
        assert(cache.blocks[0].attn_out.numel > 0);
        assert(cache.blocks[0].h1_shared.numel > 0);
        assert(cache.blocks[0].gate_shared.numel > 0);

        std::cout << "  forward_with_cache OK (cached " << cache.blocks.size() << " blocks)\n";
    }

    // Test 8: RMSNorm backward gradient check
    {
        size_t d = 4;
        size_t rows = 2;
        Tensor x({rows, d}, DType::FP32);
        Tensor gamma({d}, DType::FP32);
        Tensor grad_out({rows, d}, DType::FP32);
        Tensor grad_x({rows, d}, DType::FP32);
        Tensor grad_gamma({d}, DType::FP32);

        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        float* xp = x.fp32_data();
        float* gp = gamma.fp32_data();
        float* gop = grad_out.fp32_data();
        for (size_t i = 0; i < x.numel; ++i) xp[i] = normal(rng);
        for (size_t i = 0; i < d; ++i) gp[i] = normal(rng);
        for (size_t i = 0; i < grad_out.numel; ++i) gop[i] = normal(rng);

        NolModel::rmsnorm_backward(x, gamma, grad_out, grad_x, grad_gamma, 1e-5f);

        float eps = 1e-4f;
        float max_err = 0.0f;

        auto compute_loss = [&](const Tensor& xv, const Tensor& gv) -> float {
            Tensor out({rows, d}, DType::FP32);
            NolModel::rmsnorm_forward(xv, gv, out, 1e-5f);
            const float* op = out.fp32_data();
            float sum = 0.0f;
            for (size_t i = 0; i < out.numel; ++i) {
                sum += gop[i] * op[i];
            }
            return sum;
        };

        for (size_t i = 0; i < x.numel; ++i) {
            Tensor xp_plus = Tensor(x.shape, DType::FP32);
            Tensor xp_minus = Tensor(x.shape, DType::FP32);
            xp_plus.copy_from(x);
            xp_minus.copy_from(x);
            xp_plus.fp32_data()[i] += eps;
            xp_minus.fp32_data()[i] -= eps;

            float fp = compute_loss(xp_plus, gamma);
            float fm = compute_loss(xp_minus, gamma);
            float numerical = (fp - fm) / (2.0f * eps);
            float analytical = grad_x.fp32_data()[i];
            float err = std::abs(numerical - analytical);
            if (err > max_err) max_err = err;
        }

        for (size_t i = 0; i < d; ++i) {
            Tensor gp_plus = Tensor(gamma.shape, DType::FP32);
            Tensor gp_minus = Tensor(gamma.shape, DType::FP32);
            gp_plus.copy_from(gamma);
            gp_minus.copy_from(gamma);
            gp_plus.fp32_data()[i] += eps;
            gp_minus.fp32_data()[i] -= eps;

            float fp = compute_loss(x, gp_plus);
            float fm = compute_loss(x, gp_minus);
            float numerical = (fp - fm) / (2.0f * eps);
            float analytical = grad_gamma.fp32_data()[i];
            float err = std::abs(numerical - analytical);
            if (err > max_err) max_err = err;
        }

        std::cout << "  RMSNorm backward gradient check: max_err = " << std::scientific << max_err << "\n";
        assert(max_err < 5e-3f);
        std::cout << "  RMSNorm backward OK\n";
    }

    // Test 9: SiLU backward gradient check
    {
        size_t n = 16;
        Tensor x({n}, DType::FP32);
        Tensor grad_y({n}, DType::FP32);
        Tensor grad_x({n}, DType::FP32);

        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        float* xp = x.fp32_data();
        float* gyp = grad_y.fp32_data();
        for (size_t i = 0; i < n; ++i) {
            xp[i] = normal(rng);
            gyp[i] = normal(rng);
        }

        NolModel::silu_backward(x, grad_y, grad_x);

        float eps = 1e-4f;
        float max_err = 0.0f;
        auto silu_val = [](float x) { return x / (1.0f + std::exp(-x)); };

        for (size_t i = 0; i < n; ++i) {
            float fp = silu_val(xp[i] + eps) * gyp[i];
            float fm = silu_val(xp[i] - eps) * gyp[i];
            float numerical = (fp - fm) / (2.0f * eps);
            float analytical = grad_x.fp32_data()[i];
            float err = std::abs(numerical - analytical);
            if (err > max_err) max_err = err;
        }

        std::cout << "  SiLU backward gradient check: max_err = " << std::scientific << max_err << "\n";
        assert(max_err < 1e-3f);
        std::cout << "  SiLU backward OK\n";
    }

    // Test 10: Sigmoid backward gradient check
    {
        size_t n = 16;
        Tensor x({n}, DType::FP32);
        Tensor y({n}, DType::FP32);
        Tensor grad_y({n}, DType::FP32);
        Tensor grad_x({n}, DType::FP32);

        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        float* xp = x.fp32_data();
        float* yp = y.fp32_data();
        float* gyp = grad_y.fp32_data();
        for (size_t i = 0; i < n; ++i) {
            xp[i] = normal(rng);
            yp[i] = 1.0f / (1.0f + std::exp(-xp[i]));
            gyp[i] = normal(rng);
        }

        NolModel::sigmoid_backward(y, grad_y, grad_x);

        float eps = 1e-4f;
        float max_err = 0.0f;
        auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

        for (size_t i = 0; i < n; ++i) {
            float fp = sig(xp[i] + eps) * gyp[i];
            float fm = sig(xp[i] - eps) * gyp[i];
            float numerical = (fp - fm) / (2.0f * eps);
            float analytical = grad_x.fp32_data()[i];
            float err = std::abs(numerical - analytical);
            if (err > max_err) max_err = err;
        }

        std::cout << "  Sigmoid backward gradient check: max_err = " << std::scientific << max_err << "\n";
        assert(max_err < 1e-3f);
        std::cout << "  Sigmoid backward OK\n";
    }

    // Test 11: GQA backward gradient check (BARU - Phase 2)
    {
        NolModel model(cfg);
        model.initialize_weights(42);

        int batch = 1, seq = 4;
        Tensor x({static_cast<size_t>(batch * seq), static_cast<size_t>(cfg.d_model)}, DType::FP32);
        std::mt19937 rng(99);
        std::normal_distribution<float> normal(0.0f, 0.5f);
        for (size_t i = 0; i < x.numel; ++i) x.fp32_data()[i] = normal(rng);

        Tensor grad_out({static_cast<size_t>(batch * seq), static_cast<size_t>(cfg.d_model)}, DType::FP32);
        for (size_t i = 0; i < grad_out.numel; ++i) grad_out.fp32_data()[i] = normal(rng);

        // Build cache manually with forward
        BlockCache bc;
        int d_model = cfg.n_heads * cfg.d_head;
        int kv_dim = cfg.n_kv_heads * cfg.d_head;

        bc.q_proj = Tensor({static_cast<size_t>(batch * seq), static_cast<size_t>(d_model)}, DType::FP32);
        bc.k_proj = Tensor({static_cast<size_t>(batch * seq), static_cast<size_t>(kv_dim)}, DType::FP32);
        bc.v_proj = Tensor({static_cast<size_t>(batch * seq), static_cast<size_t>(kv_dim)}, DType::FP32);

        auto& bp = model.blocks[0];
        gemm(x.fp32_data(), bp.W_q.fp32_data(), bc.q_proj.fp32_data(), batch * seq, d_model, cfg.d_model);
        gemm(x.fp32_data(), bp.W_k.fp32_data(), bc.k_proj.fp32_data(), batch * seq, kv_dim, cfg.d_model);
        gemm(x.fp32_data(), bp.W_v.fp32_data(), bc.v_proj.fp32_data(), batch * seq, kv_dim, cfg.d_model);

        bc.q_rope = Tensor(bc.q_proj.shape, DType::FP32);
        bc.k_rope = Tensor(bc.k_proj.shape, DType::FP32);
        bc.q_rope.copy_from(bc.q_proj);
        bc.k_rope.copy_from(bc.k_proj);
        bc.q_rope.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq), static_cast<size_t>(d_model)});
        bc.k_rope.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq), static_cast<size_t>(kv_dim)});
        NolModel::rope_apply(bc.q_rope, bc.k_rope, seq, cfg.d_head, cfg.n_heads, cfg.n_kv_heads, 10000.0f);
        bc.q_rope.reshape({static_cast<size_t>(batch * seq), static_cast<size_t>(d_model)});
        bc.k_rope.reshape({static_cast<size_t>(batch * seq), static_cast<size_t>(kv_dim)});

        bc.attn_out = Tensor({static_cast<size_t>(batch * seq), static_cast<size_t>(d_model)}, DType::FP32);
        bc.attn_out.fill(0.0f);
        bc.attn_scores = Tensor({static_cast<size_t>(batch * cfg.n_heads), 
                                  static_cast<size_t>(seq), 
                                  static_cast<size_t>(seq)}, DType::FP32);
        bc.attn_scores.fill(0.0f);

        float scale = 1.0f / std::sqrt(static_cast<float>(cfg.d_head));
        int heads_per_kv = cfg.n_heads / cfg.n_kv_heads;

        for (int b = 0; b < batch; ++b) {
            for (int hh = 0; hh < cfg.n_heads; ++hh) {
                int kv_h = hh / heads_per_kv;
                for (int i = 0; i < seq; ++i) {
                    std::vector<float> scores(i + 1);
                    float max_score = -1e30f;
                    for (int j = 0; j <= i; ++j) {
                        float s = 0.0f;
                        for (int dd = 0; dd < cfg.d_head; ++dd) {
                            size_t q_idx = (b * seq + i) * d_model + hh * cfg.d_head + dd;
                            size_t k_idx = (b * seq + j) * kv_dim + kv_h * cfg.d_head + dd;
                            s += bc.q_rope.fp32_data()[q_idx] * bc.k_rope.fp32_data()[k_idx];
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
                    for (int j = 0; j <= i; ++j) {
                        scores[j] *= inv_sum;
                        bc.attn_scores.fp32_data()[(b * cfg.n_heads + hh) * seq * seq + i * seq + j] = scores[j];
                    }
                    for (int dd = 0; dd < cfg.d_head; ++dd) {
                        float val = 0.0f;
                        for (int j = 0; j <= i; ++j) {
                            size_t v_idx = (b * seq + j) * kv_dim + kv_h * cfg.d_head + dd;
                            val += scores[j] * bc.v_proj.fp32_data()[v_idx];
                        }
                        size_t out_idx = (b * seq + i) * d_model + hh * cfg.d_head + dd;
                        bc.attn_out.fp32_data()[out_idx] = val;
                    }
                }
            }
        }

        bc.a = Tensor({static_cast<size_t>(batch * seq), static_cast<size_t>(cfg.d_model)}, DType::FP32);
        gemm(bc.attn_out.fp32_data(), bp.W_o.fp32_data(), bc.a.fp32_data(), batch * seq, cfg.d_model, d_model);

        // Now test backward
        Tensor grad_x({static_cast<size_t>(batch * seq), static_cast<size_t>(cfg.d_model)}, DType::FP32);
        Tensor grad_W_q(bp.W_q.shape, DType::FP32);
        Tensor grad_W_k(bp.W_k.shape, DType::FP32);
        Tensor grad_W_v(bp.W_v.shape, DType::FP32);
        Tensor grad_W_o(bp.W_o.shape, DType::FP32);

        NolModel::gqa_backward(x, bp.W_q, bp.W_k, bp.W_v, bp.W_o, grad_out, bc,
                               grad_x, grad_W_q, grad_W_k, grad_W_v, grad_W_o,
                               batch, seq, cfg.n_heads, cfg.n_kv_heads, cfg.d_head);

        // Numerical gradient check for grad_x (sample a few elements)
        float eps = 1e-3f;
        float max_err = 0.0f;
        int checks = 0;
        int max_checks = 10;

        auto compute_loss = [&](const Tensor& xv) -> float {
            // Forward with this x
            Tensor q_p({static_cast<size_t>(batch * seq), static_cast<size_t>(d_model)}, DType::FP32);
            Tensor k_p({static_cast<size_t>(batch * seq), static_cast<size_t>(kv_dim)}, DType::FP32);
            Tensor v_p({static_cast<size_t>(batch * seq), static_cast<size_t>(kv_dim)}, DType::FP32);
            gemm(xv.fp32_data(), bp.W_q.fp32_data(), q_p.fp32_data(), batch * seq, d_model, cfg.d_model);
            gemm(xv.fp32_data(), bp.W_k.fp32_data(), k_p.fp32_data(), batch * seq, kv_dim, cfg.d_model);
            gemm(xv.fp32_data(), bp.W_v.fp32_data(), v_p.fp32_data(), batch * seq, kv_dim, cfg.d_model);
            q_p.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq), static_cast<size_t>(d_model)});
            k_p.reshape({static_cast<size_t>(batch), static_cast<size_t>(seq), static_cast<size_t>(kv_dim)});
            NolModel::rope_apply(q_p, k_p, seq, cfg.d_head, cfg.n_heads, cfg.n_kv_heads, 10000.0f);
            q_p.reshape({static_cast<size_t>(batch * seq), static_cast<size_t>(d_model)});
            k_p.reshape({static_cast<size_t>(batch * seq), static_cast<size_t>(kv_dim)});

            Tensor aout({static_cast<size_t>(batch * seq), static_cast<size_t>(d_model)}, DType::FP32);
            aout.fill(0.0f);
            for (int b = 0; b < batch; ++b) {
                for (int hh = 0; hh < cfg.n_heads; ++hh) {
                    int kv_hh = hh / heads_per_kv;
                    for (int i = 0; i < seq; ++i) {
                        std::vector<float> sc(i + 1);
                        float mx = -1e30f;
                        for (int j = 0; j <= i; ++j) {
                            float s = 0.0f;
                            for (int dd = 0; dd < cfg.d_head; ++dd) {
                                s += q_p.fp32_data()[(b * seq + i) * d_model + hh * cfg.d_head + dd] *
                                     k_p.fp32_data()[(b * seq + j) * kv_dim + kv_hh * cfg.d_head + dd];
                            }
                            s *= scale;
                            sc[j] = s;
                            if (s > mx) mx = s;
                        }
                        float se = 0.0f;
                        for (int j = 0; j <= i; ++j) { sc[j] = std::exp(sc[j] - mx); se += sc[j]; }
                        for (int j = 0; j <= i; ++j) sc[j] /= se;
                        for (int dd = 0; dd < cfg.d_head; ++dd) {
                            float val = 0.0f;
                            for (int j = 0; j <= i; ++j) {
                                val += sc[j] * v_p.fp32_data()[(b * seq + j) * kv_dim + kv_hh * cfg.d_head + dd];
                            }
                            aout.fp32_data()[(b * seq + i) * d_model + hh * cfg.d_head + dd] = val;
                        }
                    }
                }
            }
            Tensor out({static_cast<size_t>(batch * seq), static_cast<size_t>(cfg.d_model)}, DType::FP32);
            gemm(aout.fp32_data(), bp.W_o.fp32_data(), out.fp32_data(), batch * seq, cfg.d_model, d_model);
            float loss = 0.0f;
            for (size_t i = 0; i < out.numel; ++i) loss += grad_out.fp32_data()[i] * out.fp32_data()[i];
            return loss;
        };

        for (size_t idx = 0; idx < std::min(x.numel, (size_t)20); ++idx) {
            Tensor xp = Tensor(x.shape, DType::FP32);
            Tensor xm = Tensor(x.shape, DType::FP32);
            xp.copy_from(x); xm.copy_from(x);
            xp.fp32_data()[idx] += eps;
            xm.fp32_data()[idx] -= eps;
            float fp = compute_loss(xp);
            float fm = compute_loss(xm);
            float numerical = (fp - fm) / (2.0f * eps);
            float analytical = grad_x.fp32_data()[idx];
            float err = std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-6f);
            if (err > max_err) max_err = err;
            checks++;
            if (checks >= max_checks) break;
        }

        std::cout << "  GQA backward gradient check: max_rel_err = " << std::scientific << max_err << "\n";
        assert(max_err < 5e-2f);  // 5% relative error tolerance for attention backward
        std::cout << "  GQA backward OK\n";
    }

    // Test 12: SwiGLU backward gradient check (BARU - Phase 2)
    {
        NolModel model(cfg);
        model.initialize_weights(42);

        int batch = 1, seq = 4;
        int total = batch * seq;
        Tensor x({static_cast<size_t>(total), static_cast<size_t>(cfg.d_model)}, DType::FP32);
        std::mt19937 rng(77);
        std::normal_distribution<float> normal(0.0f, 0.5f);
        for (size_t i = 0; i < x.numel; ++i) x.fp32_data()[i] = normal(rng);

        Tensor grad_out({static_cast<size_t>(total), static_cast<size_t>(cfg.d_model)}, DType::FP32);
        for (size_t i = 0; i < grad_out.numel; ++i) grad_out.fp32_data()[i] = normal(rng);

        auto& bp = model.blocks[0];
        BlockCache bc;
        bc.h1_shared = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.h3_shared = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.h1_shared_silu = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.gate_shared = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
        bc.ffn_shared_out = Tensor({static_cast<size_t>(total), static_cast<size_t>(cfg.d_model)}, DType::FP32);

        gemm(x.fp32_data(), bp.W1_shared.fp32_data(), bc.h1_shared.fp32_data(), total, cfg.ffn_hidden, cfg.d_model);
        gemm(x.fp32_data(), bp.W3_shared.fp32_data(), bc.h3_shared.fp32_data(), total, cfg.ffn_hidden, cfg.d_model);
        bc.h1_shared_silu.copy_from(bc.h1_shared);
        bc.h1_shared_silu.silu_inplace();
        bc.gate_shared.copy_from(bc.h1_shared_silu);
        bc.gate_shared.mul(bc.h3_shared);
        gemm(bc.gate_shared.fp32_data(), bp.W2_shared.fp32_data(), bc.ffn_shared_out.fp32_data(), total, cfg.d_model, cfg.ffn_hidden);

        Tensor grad_x({static_cast<size_t>(total), static_cast<size_t>(cfg.d_model)}, DType::FP32);
        Tensor grad_W1s(bp.W1_shared.shape, DType::FP32);
        Tensor grad_W2s(bp.W2_shared.shape, DType::FP32);
        Tensor grad_W3s(bp.W3_shared.shape, DType::FP32);

        std::vector<Tensor> empty_w1e, empty_w2e, empty_w3e;
        std::vector<Tensor> grad_w1e, grad_w2e, grad_w3e;
        Tensor route_w({static_cast<size_t>(total), static_cast<size_t>(cfg.n_top_anchors + 1)}, DType::FP32);
        route_w.fill(0.0f);

        NolConfig cfg_no_moe = cfg;
        cfg_no_moe.enable_moe = false;

        NolModel::swiglu_moe_backward(x, bp.W1_shared, bp.W2_shared, bp.W3_shared,
                                       empty_w1e, empty_w2e, empty_w3e,
                                       route_w, grad_out, bc,
                                       grad_x, grad_W1s, grad_W2s, grad_W3s,
                                       grad_w1e, grad_w2e, grad_w3e,
                                       batch, seq, cfg.d_model, cfg.ffn_hidden, cfg.n_top_anchors);

        // Numerical gradient check for grad_x
        float eps = 1e-3f;
        float max_err = 0.0f;

        auto compute_loss = [&](const Tensor& xv) -> float {
            Tensor h1({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
            Tensor h3({static_cast<size_t>(total), static_cast<size_t>(cfg.ffn_hidden)}, DType::FP32);
            gemm(xv.fp32_data(), bp.W1_shared.fp32_data(), h1.fp32_data(), total, cfg.ffn_hidden, cfg.d_model);
            gemm(xv.fp32_data(), bp.W3_shared.fp32_data(), h3.fp32_data(), total, cfg.ffn_hidden, cfg.d_model);
            h1.silu_inplace();
            h1.mul(h3);
            Tensor out({static_cast<size_t>(total), static_cast<size_t>(cfg.d_model)}, DType::FP32);
            gemm(h1.fp32_data(), bp.W2_shared.fp32_data(), out.fp32_data(), total, cfg.d_model, cfg.ffn_hidden);
            float loss = 0.0f;
            for (size_t i = 0; i < out.numel; ++i) loss += grad_out.fp32_data()[i] * out.fp32_data()[i];
            return loss;
        };

        for (size_t idx = 0; idx < std::min(x.numel, (size_t)15); ++idx) {
            Tensor xp = Tensor(x.shape, DType::FP32);
            Tensor xm = Tensor(x.shape, DType::FP32);
            xp.copy_from(x); xm.copy_from(x);
            xp.fp32_data()[idx] += eps;
            xm.fp32_data()[idx] -= eps;
            float fp = compute_loss(xp);
            float fm = compute_loss(xm);
            float numerical = (fp - fm) / (2.0f * eps);
            float analytical = grad_x.fp32_data()[idx];
            float err = std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-6f);
            if (err > max_err) max_err = err;
        }

        std::cout << "  SwiGLU backward gradient check: max_rel_err = " << std::scientific << max_err << "\n";
        assert(max_err < 5e-2f);
        std::cout << "  SwiGLU backward OK\n";
    }

    std::cout << "[PASS] ✅ Semua unit test model berhasil (Phase 1 + Phase 2)!\n\n";
}

#ifdef NOL_TEST_MODEL
int main() {
    run_model_tests();
    return 0;
}
#endif
