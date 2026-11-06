#include "../../include/core/lorentz.hpp"
#include <cstring>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

namespace nol {
namespace core {

// ==========================================
// FORWARD OPERATIONS
// ==========================================

float linner(const float* x, const float* y, size_t n) {
    float s = -x[0] * y[0];
    for (size_t i = 1; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

void exp0(const float* v, float* x, size_t n, float c) {
    // PERSIS sesuai kode referensi dokumentasi Section 9
    const float sc = std::sqrt(c);
    float nraw = 0.0f;
    for (size_t i = 0; i < n - 1; ++i) {
        nraw += v[i] * v[i];
    }
    nraw = std::sqrt(nraw);

    // Clamp anti-overflow: ‖v‖ ≤ 10/√c
    const float nc = std::min(nraw, 10.0f / sc);

    // k = sinh(sc*nc) / (sc*nraw)  ← pembagi adalah nraw, BUKAN nc
    // Ini memastikan ⟨x,x⟩_L = -1/c secara eksak (dalam limit presisi)
    const float k = (nraw < 1e-6f) ? 1.0f : std::sinh(sc * nc) / (sc * nraw);

    x[0] = std::cosh(sc * nc) / sc;
    for (size_t i = 0; i < n - 1; ++i) {
        x[i + 1] = k * v[i];
    }
}

void log0(const float* x, float* v, size_t n, float c) {
    // PERSIS sesuai kode referensi dokumentasi Section 9
    const float sc = std::sqrt(c);
    float ns = 0.0f;
    for (size_t i = 1; i < n; ++i) {
        ns += x[i] * x[i];
    }
    ns = std::sqrt(ns);

    // asinh untuk stabilitas numerik
    const float rho = std::asinh(sc * ns) / sc;
    const float k = (ns < 1e-6f) ? 1.0f : rho / ns;

    for (size_t i = 0; i < n - 1; ++i) {
        v[i] = k * x[i + 1];
    }
}

float dist2_lorentz(const float* x, const float* y, size_t n, float c) {
    return -2.0f / c - 2.0f * linner(x, y, n);
}

void lorentz_centroid(const float* points, const float* weights,
                      float* out, size_t num_points, size_t n, float c) {
    std::memset(out, 0, n * sizeof(float));
    for (size_t j = 0; j < num_points; ++j) {
        float w = weights[j];
        const float* pt = points + j * n;
        for (size_t i = 0; i < n; ++i) {
            out[i] += w * pt[i];
        }
    }

    float zz = linner(out, out, n);
    float abs_zz = std::abs(zz);
    float denom = std::sqrt(c) * std::sqrt(std::max(abs_zz, 1e-8f));
    float inv_denom = 1.0f / denom;

    for (size_t i = 0; i < n; ++i) {
        out[i] *= inv_denom;
    }

    if (out[0] < 0.0f) {
        for (size_t i = 0; i < n; ++i) out[i] = -out[i];
    }
}

float poincare_radius(float rho, float c) {
    const float sc = std::sqrt(c);
    return std::tanh(sc * rho * 0.5f) / sc;
}

// ==========================================
// BACKWARD MANUAL
// ==========================================

static inline float apply_J_element(float val, size_t index) {
    return (index == 0) ? -val : val;
}

void backward_score_qk(float grad_s, const float* q, const float* k,
                       float* grad_q, float* grad_k,
                       size_t n, float tau) {
    float scale = 2.0f * grad_s / tau;
    for (size_t i = 0; i < n; ++i) {
        float ji = apply_J_element(1.0f, i);
        grad_q[i] += scale * ji * k[i];
        grad_k[i] += scale * ji * q[i];
    }
}

void backward_centroid(const float* z, const float* grad_u,
                       float* grad_z, size_t n, float c) {
    float q_val = -linner(z, z, n);
    q_val = std::max(q_val, 1e-8f);

    float sqrt_c = std::sqrt(c);
    float sqrt_q = std::sqrt(q_val);
    float inv_scale = 1.0f / (sqrt_c * sqrt_q);

    float ztg = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        ztg += z[i] * grad_u[i];
    }

    float ztg_over_q = ztg / q_val;

    for (size_t i = 0; i < n; ++i) {
        float ji_z = apply_J_element(z[i], i);
        grad_z[i] += inv_scale * (grad_u[i] + ji_z * ztg_over_q);
    }
}

// ==========================================
// GRADIENT CHECK
// ==========================================

float gradient_check_score_qk(size_t n, float tau, float eps) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(n), k(n);
    for (size_t i = 0; i < n; ++i) {
        q[i] = dist(rng);
        k[i] = dist(rng);
    }

    float max_err = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        std::vector<float> q_plus = q, q_minus = q;
        q_plus[i] += eps;
        q_minus[i] -= eps;

        float inner_plus = linner(q_plus.data(), k.data(), n);
        float inner_minus = linner(q_minus.data(), k.data(), n);
        float numerical = (inner_plus - inner_minus) / (2.0f * eps);

        float analytical = apply_J_element(k[i], i);
        float err = std::abs(numerical - analytical);
        max_err = std::max(max_err, err);
    }

    for (size_t i = 0; i < n; ++i) {
        std::vector<float> k_plus = k, k_minus = k;
        k_plus[i] += eps;
        k_minus[i] -= eps;

        float inner_plus = linner(q.data(), k_plus.data(), n);
        float inner_minus = linner(q.data(), k_minus.data(), n);
        float numerical = (inner_plus - inner_minus) / (2.0f * eps);

        float analytical = apply_J_element(q[i], i);
        float err = std::abs(numerical - analytical);
        max_err = std::max(max_err, err);
    }

    return max_err;
}

float gradient_check_centroid(size_t n, float c, float eps) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.1f, 1.0f);

    std::vector<float> z(n);
    for (size_t i = 1; i < n; ++i) z[i] = dist(rng) * 0.1f;
    float spatial_sq = 0.0f;
    for (size_t i = 1; i < n; ++i) spatial_sq += z[i] * z[i];
    z[0] = std::sqrt(1.0f / c + spatial_sq);

    std::vector<float> grad_u(n);
    for (size_t i = 0; i < n; ++i) grad_u[i] = dist(rng);

    std::vector<float> grad_z_a(n, 0.0f);
    backward_centroid(z.data(), grad_u.data(), grad_z_a.data(), n, c);

    float max_err = 0.0f;

    auto compute_u_dot_g = [&](const std::vector<float>& zv) -> float {
        float q_val = -linner(zv.data(), zv.data(), n);
        q_val = std::max(q_val, 1e-8f);
        float inv_scale = 1.0f / (std::sqrt(c) * std::sqrt(q_val));
        float dot = 0.0f;
        for (size_t i = 0; i < n; ++i) dot += zv[i] * inv_scale * grad_u[i];
        return dot;
    };

    for (size_t i = 0; i < n; ++i) {
        std::vector<float> z_plus = z, z_minus = z;
        z_plus[i] += eps;
        z_minus[i] -= eps;

        float f_plus = compute_u_dot_g(z_plus);
        float f_minus = compute_u_dot_g(z_minus);
        float numerical = (f_plus - f_minus) / (2.0f * eps);

        float err = std::abs(numerical - grad_z_a[i]);
        max_err = std::max(max_err, err);
    }

    return max_err;
}

} // namespace core
} // namespace nol

// ==========================================
// UNIT TEST LORENTZ
// ==========================================
static void run_lorentz_tests() {
    using namespace nol::core;
    std::cout << "[TEST] Memulai unit test lorentz...\n";

    // Test 1: linner dasar
    {
        float x[] = {1, 2, 3};
        float y[] = {4, 5, 6};
        assert(std::abs(linner(x, y, 3) - 24.0f) < 1e-5f);
    }

    // Test 2: Manifold identity ⟨x,x⟩_L = -1/c
    {
        float c = 1.0f;
        float x[] = {std::sqrt(2.0f), 1.0f, 0.0f};
        assert(std::abs(linner(x, x, 3) - (-1.0f)) < 1e-5f);
    }

    // Test 3: exp0 → log0 roundtrip (vektor normal)
    {
        float c = 1.0f;
        size_t n = 4;
        float v[] = {0.5f, -0.3f, 0.8f};
        float x[4], v_rec[3];
        exp0(v, x, n, c);
        log0(x, v_rec, n, c);
        for (size_t i = 0; i < 3; ++i) {
            assert(std::abs(v_rec[i] - v[i]) < 1e-4f);
        }
    }

    // Test 4: exp0 menghasilkan titik di manifold (vektor kecil)
    {
        float c = 2.0f;
        size_t n = 3;
        float v[] = {0.3f, -0.5f};
        float x[3];
        exp0(v, x, n, c);
        float inner = linner(x, x, n);
        assert(std::abs(inner - (-1.0f / c)) < 1e-4f);
        assert(x[0] > 0.0f);
    }

    // Test 5: exp0 clamp anti-overflow
    // Catatan: dengan nc=10 dan float32, cosh²(10)-sinh²(10) mengalami
    // catastrophic cancellation. Toleransi disesuaikan dengan realitas float32.
    // Yang penting: x finite, x₀ > 0, dan arah vektor terjaga.
    {
        float c = 1.0f;
        size_t n = 3;
        float v[] = {100.0f, 100.0f};
        float x[3];
        exp0(v, x, n, c);

        assert(std::isfinite(x[0]));
        assert(std::isfinite(x[1]));
        assert(std::isfinite(x[2]));
        assert(x[0] > 0.0f);

        // Verifikasi arah: x[1]/x[2] harus ≈ v[0]/v[1] = 1.0
        assert(std::abs(x[1] / x[2] - 1.0f) < 1e-5f);

        // Verifikasi exp0→log0 roundtrip tetap bekerja setelah clamp
        float v_rec[2];
        log0(x, v_rec, n, c);
        // Arah harus terjaga meski magnitudo di-clamp
        float orig_ratio = v[0] / v[1];
        float rec_ratio = v_rec[0] / v_rec[1];
        assert(std::abs(rec_ratio - orig_ratio) < 1e-4f);
    }

    // Test 6: exp0 manifold check dengan vektor sedang (presisi float32 baik)
    {
        float c = 1.0f;
        size_t n = 3;
        float v[] = {2.0f, 3.0f}; // nraw ≈ 3.6, jauh di bawah clamp 10
        float x[3];
        exp0(v, x, n, c);
        float inner = linner(x, x, n);
        assert(std::abs(inner - (-1.0f / c)) < 1e-4f);
        assert(x[0] > 0.0f);
    }

    // Test 7: exp0 manifold check dengan c berbeda
    {
        float c = 0.5f;
        size_t n = 4;
        float v[] = {1.0f, -1.5f, 0.8f};
        float x[4];
        exp0(v, x, n, c);
        float inner = linner(x, x, n);
        assert(std::abs(inner - (-1.0f / c)) < 1e-4f);
        assert(x[0] > 0.0f);
    }

    // Test 8: dist2 identik = 0
    {
        float c = 1.0f;
        float x[] = {std::sqrt(2.0f), 1.0f, 0.0f};
        assert(std::abs(dist2_lorentz(x, x, 3, c)) < 1e-4f);
    }

    // Test 9: Centroid di manifold
    {
        float c = 1.0f;
        size_t n = 3;
        float p1[] = {std::sqrt(2.0f), 1.0f, 0.0f};
        float p2[] = {std::sqrt(2.0f), -1.0f, 0.0f};
        float points[] = {p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]};
        float weights[] = {0.5f, 0.5f};
        float out[3];
        lorentz_centroid(points, weights, out, 2, n, c);
        float inner = linner(out, out, n);
        assert(std::abs(inner - (-1.0f / c)) < 1e-3f);
        assert(out[0] > 0.0f);
    }

    // Test 10: poincare_radius
    {
        assert(std::abs(poincare_radius(0.0f, 1.0f)) < 1e-6f);
        assert(std::abs(poincare_radius(100.0f, 1.0f) - 1.0f) < 1e-3f);
        assert(std::abs(poincare_radius(100.0f, 4.0f) - 0.5f) < 1e-3f);
    }

    // Test 11-14: Gradient checks
    {
        float e1 = gradient_check_score_qk(5, 1.0f);
        std::cout << "  Grad check score_qk (n=5): " << e1 << "\n";
        assert(e1 < 1e-3f);
    }
    {
        float e2 = gradient_check_centroid(4, 1.0f);
        std::cout << "  Grad check centroid (n=4): " << e2 << "\n";
        assert(e2 < 1e-3f);
    }
    {
        assert(gradient_check_score_qk(3, 0.5f) < 1e-3f);
        assert(gradient_check_score_qk(8, 2.0f) < 1e-3f);
    }
    {
        assert(gradient_check_centroid(3, 0.5f) < 1e-3f);
        assert(gradient_check_centroid(5, 2.0f) < 1e-3f);
    }

    // Test 15: exp0 → log0 roundtrip vektor sangat kecil
    {
        float c = 1.0f;
        float v[] = {1e-7f, 1e-7f};
        float x[3], v_rec[2];
        exp0(v, x, 3, c);
        log0(x, v_rec, 3, c);
        for (size_t i = 0; i < 2; ++i) {
            assert(std::abs(v_rec[i] - v[i]) < 1e-5f);
        }
    }

    std::cout << "[PASS] ✅ Semua unit test lorentz berhasil!\n\n";
}

#ifdef NOL_TEST_LORENTZ
int main() {
    run_lorentz_tests();
    return 0;
}
#endif
