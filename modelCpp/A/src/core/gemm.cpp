#include "../../include/core/gemm.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define NOL_HAS_AVX2_FMA 1
#else
#define NOL_HAS_AVX2_FMA 0
#endif

namespace nol {
namespace core {

static constexpr size_t BLOCK_M = 64;
static constexpr size_t BLOCK_N = 64;
static constexpr size_t BLOCK_K = 64;

// Kernel mikro: C_tile += A_tile × B_tile (selalu akumulasi)
static void micro_kernel(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C,
                         size_t M, size_t N, size_t K,
                         size_t lda, size_t ldb, size_t ldc) {
#if NOL_HAS_AVX2_FMA
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j + 7 < N; j += 8) {
            __m256 acc = _mm256_loadu_ps(&C[i * ldc + j]);
            for (size_t k = 0; k < K; ++k) {
                __m256 a_val = _mm256_set1_ps(A[i * lda + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k * ldb + j]);
                acc = _mm256_fmadd_ps(a_val, b_vec, acc);
            }
            _mm256_storeu_ps(&C[i * ldc + j], acc);
        }
        for (size_t j = (N / 8) * 8; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
#else
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = A[i * lda + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
#endif
}

void gemm(const float* A, const float* B, float* C,
          size_t M, size_t N, size_t K,
          float alpha, float beta) {
    if (M == 0 || N == 0 || K == 0) return;

    // Jika alpha == 0, hanya scale C by beta
    if (alpha == 0.0f) {
        if (beta == 0.0f) {
            std::memset(C, 0, M * N * sizeof(float));
        } else if (beta != 1.0f) {
            for (size_t i = 0; i < M * N; ++i) C[i] *= beta;
        }
        return;
    }

    // Gunakan buffer sementara untuk akumulasi A×B per tile
    // agar alpha dan beta bisa diterapkan secara terpisah
    #pragma omp parallel
    {
        // Buffer lokal per thread untuk menghindari race condition
        std::vector<float> tmp(BLOCK_M * BLOCK_N, 0.0f);

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t bm = 0; bm < M; bm += BLOCK_M) {
            for (size_t bn = 0; bn < N; bn += BLOCK_N) {
                size_t m_end = std::min(bm + BLOCK_M, M);
                size_t n_end = std::min(bn + BLOCK_N, N);
                size_t mb = m_end - bm;
                size_t nb = n_end - bn;

                // Nol-kan buffer sementara
                for (size_t i = 0; i < mb * nb; ++i) tmp[i] = 0.0f;

                // Akumulasi A×B ke buffer sementara
                for (size_t bk = 0; bk < K; bk += BLOCK_K) {
                    size_t k_end = std::min(bk + BLOCK_K, K);
                    size_t kb = k_end - bk;

                    micro_kernel(&A[bm * K + bk], &B[bk * N + bn],
                                 tmp.data(), mb, nb, kb, K, N, nb);
                }

                // Gabungkan: C = alpha * tmp + beta * C
                for (size_t i = 0; i < mb; ++i) {
                    for (size_t j = 0; j < nb; ++j) {
                        size_t c_idx = (bm + i) * N + (bn + j);
                        C[c_idx] = alpha * tmp[i * nb + j] + beta * C[c_idx];
                    }
                }
            }
        }
    }
}

// ==========================================
// LORENTZ INNER PRODUCT VIA GEMM
// ==========================================

void gemm_lorentz_inner(const float* Q, const float* K, float* out,
                        size_t T, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < T; ++t) {
        float sum = -Q[t * n + 0] * K[0];
#if NOL_HAS_AVX2_FMA
        size_t i = 1;
        __m256 acc = _mm256_setzero_ps();
        for (; i + 7 < n; i += 8) {
            __m256 q_vec = _mm256_loadu_ps(&Q[t * n + i]);
            __m256 k_vec = _mm256_loadu_ps(&K[i]);
            acc = _mm256_fmadd_ps(q_vec, k_vec, acc);
        }
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        sum += _mm_cvtss_f32(s);
        for (; i < n; ++i) sum += Q[t * n + i] * K[i];
#else
        for (size_t i = 1; i < n; ++i) {
            sum += Q[t * n + i] * K[i];
        }
#endif
        out[t] = sum;
    }
}

void gemm_lorentz_scores(const float* Q, const float* K, float* scores,
                         size_t T, size_t n) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            float sum = -Q[i * n + 0] * K[j * n + 0];
            for (size_t d = 1; d < n; ++d) {
                sum += Q[i * n + d] * K[j * n + d];
            }
            scores[i * T + j] = sum;
        }
        for (size_t j = i + 1; j < T; ++j) {
            scores[i * T + j] = -1e30f;
        }
    }
}

} // namespace core
} // namespace nol

// ==========================================
// UNIT TEST GEMM
// ==========================================
static void run_gemm_tests() {
    using namespace nol::core;
    std::cout << "[TEST] Memulai unit test gemm...\n";

    // Test 1: GEMM dasar 2x3 × 3x2 = 2x2
    {
        float A[] = {1,2,3, 4,5,6};
        float B[] = {7,8, 9,10, 11,12};
        float C[4] = {0};
        gemm(A, B, C, 2, 2, 3);
        assert(std::abs(C[0] - 58.0f) < 1e-3f);
        assert(std::abs(C[1] - 64.0f) < 1e-3f);
        assert(std::abs(C[2] - 139.0f) < 1e-3f);
        assert(std::abs(C[3] - 154.0f) < 1e-3f);
    }

    // Test 2: GEMM dengan alpha dan beta
    {
        float A[] = {1, 2, 3, 4};
        float B[] = {5, 6, 7, 8};
        float C[] = {1, 1, 1, 1};
        // C = 2*A*B + 3*C = 2*[[19,22],[43,50]] + 3*[[1,1],[1,1]] = [[41,47],[89,103]]
        gemm(A, B, C, 2, 2, 2, 2.0f, 3.0f);
        assert(std::abs(C[0] - 41.0f) < 1e-3f);
        assert(std::abs(C[1] - 47.0f) < 1e-3f);
        assert(std::abs(C[2] - 89.0f) < 1e-3f);
        assert(std::abs(C[3] - 103.0f) < 1e-3f);
    }

    // Test 3: GEMM identitas
    {
        float I[] = {1,0,0, 0,1,0, 0,0,1};
        float A[] = {1,2,3, 4,5,6, 7,8,9};
        float C[9] = {0};
        gemm(I, A, C, 3, 3, 3);
        for (int i = 0; i < 9; ++i) {
            assert(std::abs(C[i] - A[i]) < 1e-5f);
        }
    }

    // Test 4: GEMM ukuran besar
    {
        size_t M = 128, N = 128, K = 128;
        std::vector<float> A(M * K, 0.01f);
        std::vector<float> B(K * N, 0.01f);
        std::vector<float> C(M * N, 0.0f);
        gemm(A.data(), B.data(), C.data(), M, N, K);
        float expected = static_cast<float>(K) * 0.01f * 0.01f;
        for (size_t i = 0; i < M * N; ++i) {
            assert(std::abs(C[i] - expected) < 1e-3f);
        }
    }

    // Test 5: Lorentz Inner Product single vector
    {
        float Q[] = {1, 2, 3};
        float K[] = {4, 5, 6};
        float out[1];
        gemm_lorentz_inner(Q, K, out, 1, 3);
        assert(std::abs(out[0] - 24.0f) < 1e-5f);
    }

    // Test 6: Lorentz Inner Product batched
    {
        float Q[] = {1,2,3, 4,5,6};
        float K[] = {1,1,1};
        float out[2];
        gemm_lorentz_inner(Q, K, out, 2, 3);
        assert(std::abs(out[0] - 4.0f) < 1e-5f);
        assert(std::abs(out[1] - 7.0f) < 1e-5f);
    }

    // Test 7: Lorentz Scores kausal
    {
        float Q[] = {1,2, 3,4, 5,6};
        float K[] = {1,0, 0,1, 1,1};
        float scores[9];
        gemm_lorentz_scores(Q, K, scores, 3, 2);
        
        assert(std::abs(scores[0*3+0] - (-1.0f)) < 1e-5f);
        assert(std::abs(scores[1*3+0] - (-3.0f)) < 1e-5f);
        assert(std::abs(scores[1*3+1] - 4.0f) < 1e-5f);
        assert(std::abs(scores[2*3+0] - (-5.0f)) < 1e-5f);
        assert(std::abs(scores[2*3+1] - 6.0f) < 1e-5f);
        assert(std::abs(scores[2*3+2] - 1.0f) < 1e-5f);
        
        assert(scores[0*3+1] < -1e20f);
        assert(scores[0*3+2] < -1e20f);
        assert(scores[1*3+2] < -1e20f);
    }

    // Test 8: Edge case M=0
    {
        float A[1], B[1], C[1] = {99.0f};
        gemm(A, B, C, 0, 5, 3);
        assert(C[0] == 99.0f);
    }

    // Test 9: alpha=0 hanya scale beta
    {
        float A[] = {1,2,3,4};
        float B[] = {5,6,7,8};
        float C[] = {2, 3, 4, 5};
        gemm(A, B, C, 2, 2, 2, 0.0f, 2.0f);
        assert(std::abs(C[0] - 4.0f) < 1e-5f);
        assert(std::abs(C[1] - 6.0f) < 1e-5f);
        assert(std::abs(C[2] - 8.0f) < 1e-5f);
        assert(std::abs(C[3] - 10.0f) < 1e-5f);
    }

    // Test 10: beta=0 overwrite
    {
        float A[] = {1,0, 0,1};
        float B[] = {7,8, 9,10};
        float C[] = {99,99, 99,99};
        gemm(A, B, C, 2, 2, 2, 1.0f, 0.0f);
        assert(std::abs(C[0] - 7.0f) < 1e-5f);
        assert(std::abs(C[1] - 8.0f) < 1e-5f);
        assert(std::abs(C[2] - 9.0f) < 1e-5f);
        assert(std::abs(C[3] - 10.0f) < 1e-5f);
    }

    std::cout << "[PASS] ✅ Semua unit test gemm berhasil!\n\n";
}

#ifdef NOL_TEST_GEMM
int main() {
    run_gemm_tests();
    return 0;
}
#endif
