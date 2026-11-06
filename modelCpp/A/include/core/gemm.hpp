#pragma once

#include <cstddef>
#include <cstdint>

namespace nol {
namespace core {

/**
 * @brief BLAS-like GEMM Blocked dengan SIMD & OpenMP.
 * Pilar 1 - Lapisan 4.
 * 
 * C = alpha * A(M×K) × B(K×N) + beta * C(M×N)
 * Semua matriks row-major, contiguous.
 */
void gemm(const float* A, const float* B, float* C,
          size_t M, size_t N, size_t K,
          float alpha = 1.0f, float beta = 0.0f);

/**
 * @brief Lorentz Inner Product via GEMM.
 * Trik: kolom waktu (index 0) dari Q dikalikan -1 sebelum dot product.
 * out[i] = sum_j( Q[i*n+j] * K[j] ) dengan Q[i*n+0] *= -1
 * 
 * Digunakan oleh HGA (T1) untuk menghitung ⟨q,k⟩_L sebagai satu GEMM biasa.
 * Q: [T x n], K: [n], out: [T]
 */
void gemm_lorentz_inner(const float* Q, const float* K, float* out,
                        size_t T, size_t n);

/**
 * @brief Batched Lorentz Inner Product untuk attention score matrix.
 * Scores[i][j] = ⟨Q[i], K[j]⟩_L untuk semua pasangan kausal (j <= i).
 * Q: [T x n], K: [T x n], Scores: [T x T]
 */
void gemm_lorentz_scores(const float* Q, const float* K, float* scores,
                         size_t T, size_t n);

} // namespace core
} // namespace nol
