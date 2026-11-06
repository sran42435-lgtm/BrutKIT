#pragma once

#include <cstddef>
#include <cmath>

namespace nol {
namespace core {

/**
 * @brief Lorentz Geometry Engine - Pilar 2 Lapisan 12
 * Semua operasi dalam FP32. Kolom 0 = waktu.
 * Dimensi titik Lorentz = n (dimana n = d_hyp + 1).
 * Vektor tangen berdimensi n-1.
 */

// Produk dalam Lorentz: ⟨x,y⟩_L = -x₀y₀ + Σᵢ≥₁ xᵢyᵢ
float linner(const float* x, const float* y, size_t n);

// Exponential map: vektor tangen (n-1 dim) → titik hiperboloid (n dim)
// Clamp ‖v‖ ≤ 10/√c sebelum komputasi
void exp0(const float* v, float* x, size_t n, float c);

// Logarithmic map: titik hiperboloid (n dim) → vektor tangen (n-1 dim)
// Menggunakan asinh untuk stabilitas numerik
void log0(const float* x, float* v, size_t n, float c);

// Jarak kuadrat Lorentz: d²_L(x,y) = -2/c - 2⟨x,y⟩_L
// Nilai ini selalu ≤ 0, digunakan sebagai skor (negatif = lebih dekat)
float dist2_lorentz(const float* x, const float* y, size_t n, float c);

// Centroid Lorentz: μ = z / (√c · √|⟨z,z⟩_L|), z = Σⱼ wⱼ vⱼ
// points: [num_points x n], weights: [num_points], out: [n]
void lorentz_centroid(const float* points, const float* weights,
                      float* out, size_t num_points, size_t n, float c);

// Radius Poincaré untuk interpretasi/loss hierarki
// r = tanh(√c·ρ/2)/√c
float poincare_radius(float rho, float c);

// ==========================================
// BACKWARD MANUAL (Gradient Check Wajib)
// ==========================================

// Turunan skor HGA terhadap q dan k:
// s = (2/c + 2⟨q,k⟩_L)/τ
// ∂s/∂q = (2/τ)·J·k,  ∂s/∂k = (2/τ)·J·q
// J = diag(-1, 1, 1, ..., 1)
void backward_score_qk(float grad_s, const float* q, const float* k,
                       float* grad_q, float* grad_k,
                       size_t n, float tau);

// Turunan centroid Lorentz:
// u = z/(√c·√q), q = -⟨z,z⟩_L
// (∂u/∂z)ᵀ g = (1/(√c·√q)) · (g + J·z · (zᵀg)/q)
void backward_centroid(const float* z, const float* grad_u,
                       float* grad_z, size_t n, float c);

// Gradient check via finite difference
// Mengembalikan max absolute error antara analytical dan numerical gradient
float gradient_check_score_qk(size_t n, float tau, float eps = 1e-4f);
float gradient_check_centroid(size_t n, float c, float eps = 1e-4f);

} // namespace core
} // namespace nol
