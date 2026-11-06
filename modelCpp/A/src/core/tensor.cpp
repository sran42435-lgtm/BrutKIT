#include "../../include/core/tensor.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iomanip>

#ifdef _WIN32
#include <malloc.h>
#define ALIGNED_ALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
#define ALIGNED_ALLOC(size, alignment) std::aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) std::free(ptr)
#endif

namespace nol {
namespace core {

// ==========================================
// CONSTRUCTOR / DESTRUCTOR / MOVE
// ==========================================

Tensor::Tensor() : data(nullptr), owns_memory(false), numel(0), dtype(DType::FP32) {}

Tensor::Tensor(const std::vector<size_t>& shape, DType dtype)
    : shape(shape), dtype(dtype), data(nullptr), owns_memory(true) {
    numel = 1;
    for (size_t s : shape) numel *= s;
    compute_strides();
    allocate();
}

Tensor::~Tensor() {
    if (owns_memory && data != nullptr) {
        deallocate();
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape(std::move(other.shape)), strides(std::move(other.strides)),
      numel(other.numel), dtype(other.dtype), data(other.data), owns_memory(other.owns_memory) {
    other.data = nullptr;
    other.owns_memory = false;
    other.numel = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_memory && data != nullptr) deallocate();
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        numel = other.numel;
        dtype = other.dtype;
        data = other.data;
        owns_memory = other.owns_memory;
        other.data = nullptr;
        other.owns_memory = false;
        other.numel = 0;
    }
    return *this;
}

// ==========================================
// MEMORY MANAGEMENT
// ==========================================

void Tensor::compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

size_t Tensor::get_aligned_size() const {
    size_t bytes = numel * (dtype == DType::FP32 ? sizeof(float) : sizeof(uint16_t));
    size_t alignment = 64;
    return (bytes + alignment - 1) & ~(alignment - 1);
}

void Tensor::allocate() {
    if (numel == 0) {
        data = nullptr;
        return;
    }
    size_t alloc_size = get_aligned_size();
    data = ALIGNED_ALLOC(alloc_size, 64);
    if (!data) throw std::bad_alloc();
    std::memset(data, 0, alloc_size);
}

void Tensor::deallocate() {
    if (data) {
        ALIGNED_FREE(data);
        data = nullptr;
    }
}

void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_numel = 1;
    for (size_t s : new_shape) new_numel *= s;
    if (new_numel != numel) {
        throw std::runtime_error("Reshape: total elemen harus sama");
    }
    shape = new_shape;
    compute_strides();
}

// ==========================================
// TYPE-SAFE ACCESSORS
// ==========================================

float* Tensor::fp32_data() {
    if (dtype != DType::FP32) throw std::runtime_error("Tensor bukan FP32");
    return static_cast<float*>(data);
}

const float* Tensor::fp32_data() const {
    if (dtype != DType::FP32) throw std::runtime_error("Tensor bukan FP32");
    return static_cast<const float*>(data);
}

uint16_t* Tensor::bf16_data() {
    if (dtype != DType::BF16) throw std::runtime_error("Tensor bukan BF16");
    return static_cast<uint16_t*>(data);
}

const uint16_t* Tensor::bf16_data() const {
    if (dtype != DType::BF16) throw std::runtime_error("Tensor bukan BF16");
    return static_cast<const uint16_t*>(data);
}

// ==========================================
// ELEMENTWISE OPS (PILAR 1 - LAPISAN 5)
// ==========================================

void Tensor::fill(float value) {
    if (dtype == DType::FP32) {
        float* ptr = fp32_data();
        for (size_t i = 0; i < numel; ++i) ptr[i] = value;
    } else {
        uint16_t bv = fp32_to_bf16(value);
        uint16_t* ptr = bf16_data();
        for (size_t i = 0; i < numel; ++i) ptr[i] = bv;
    }
}

void Tensor::copy_from(const Tensor& other) {
    if (numel != other.numel || dtype != other.dtype) {
        throw std::runtime_error("copy_from: numel atau dtype tidak cocok");
    }
    size_t bytes = numel * (dtype == DType::FP32 ? sizeof(float) : sizeof(uint16_t));
    std::memcpy(data, other.data, bytes);
}

void Tensor::add(const Tensor& other) {
    if (numel != other.numel) throw std::runtime_error("add: numel tidak cocok");
    if (dtype == DType::FP32 && other.dtype == DType::FP32) {
        float* a = fp32_data();
        const float* b = other.fp32_data();
        for (size_t i = 0; i < numel; ++i) a[i] += b[i];
    } else if (dtype == DType::BF16 && other.dtype == DType::BF16) {
        uint16_t* a = bf16_data();
        const uint16_t* b = other.bf16_data();
        for (size_t i = 0; i < numel; ++i) {
            a[i] = fp32_to_bf16(bf16_to_fp32(a[i]) + bf16_to_fp32(b[i]));
        }
    } else {
        throw std::runtime_error("add: dtype harus sama");
    }
}

void Tensor::mul(const Tensor& other) {
    if (numel != other.numel) throw std::runtime_error("mul: numel tidak cocok");
    if (dtype == DType::FP32 && other.dtype == DType::FP32) {
        float* a = fp32_data();
        const float* b = other.fp32_data();
        for (size_t i = 0; i < numel; ++i) a[i] *= b[i];
    } else if (dtype == DType::BF16 && other.dtype == DType::BF16) {
        uint16_t* a = bf16_data();
        const uint16_t* b = other.bf16_data();
        for (size_t i = 0; i < numel; ++i) {
            a[i] = fp32_to_bf16(bf16_to_fp32(a[i]) * bf16_to_fp32(b[i]));
        }
    } else {
        throw std::runtime_error("mul: dtype harus sama");
    }
}

void Tensor::scale(float alpha) {
    if (dtype == DType::FP32) {
        float* ptr = fp32_data();
        for (size_t i = 0; i < numel; ++i) ptr[i] *= alpha;
    } else {
        uint16_t* ptr = bf16_data();
        for (size_t i = 0; i < numel; ++i) {
            ptr[i] = fp32_to_bf16(bf16_to_fp32(ptr[i]) * alpha);
        }
    }
}

void Tensor::silu_inplace() {
    if (dtype == DType::FP32) {
        float* ptr = fp32_data();
        for (size_t i = 0; i < numel; ++i) {
            ptr[i] = ptr[i] / (1.0f + std::exp(-ptr[i]));
        }
    } else {
        uint16_t* ptr = bf16_data();
        for (size_t i = 0; i < numel; ++i) {
            float v = bf16_to_fp32(ptr[i]);
            ptr[i] = fp32_to_bf16(v / (1.0f + std::exp(-v)));
        }
    }
}

void Tensor::sigmoid_inplace() {
    if (dtype == DType::FP32) {
        float* ptr = fp32_data();
        for (size_t i = 0; i < numel; ++i) {
            ptr[i] = 1.0f / (1.0f + std::exp(-ptr[i]));
        }
    } else {
        uint16_t* ptr = bf16_data();
        for (size_t i = 0; i < numel; ++i) {
            float v = bf16_to_fp32(ptr[i]);
            ptr[i] = fp32_to_bf16(1.0f / (1.0f + std::exp(-v)));
        }
    }
}

void Tensor::rmsnorm_inplace(const Tensor& gamma, float eps) {
    if (dtype != DType::FP32 || gamma.dtype != DType::FP32) {
        throw std::runtime_error("rmsnorm: wajib FP32");
    }
    if (shape.empty()) return;

    size_t last_dim = shape.back();
    if (gamma.numel != last_dim) {
        throw std::runtime_error("rmsnorm: gamma numel harus sama dengan dimensi terakhir");
    }

    float* ptr = fp32_data();
    const float* g = gamma.fp32_data();
    size_t rows = numel / last_dim;

    for (size_t r = 0; r < rows; ++r) {
        float* row = ptr + r * last_dim;
        float sum_sq = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            sum_sq += row[j] * row[j];
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(last_dim) + eps);
        for (size_t j = 0; j < last_dim; ++j) {
            row[j] = row[j] * inv_rms * g[j];
        }
    }
}

void Tensor::softmax_inplace(int dim) {
    if (dtype != DType::FP32) throw std::runtime_error("softmax: wajib FP32");
    if (dim < 0 || dim >= static_cast<int>(shape.size())) {
        throw std::runtime_error("softmax: dim out of range");
    }

    size_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= shape[i];
    size_t inner = 1;
    for (int i = dim + 1; i < static_cast<int>(shape.size()); ++i) inner *= shape[i];
    size_t dim_size = shape[dim];

    float* ptr = fp32_data();

    for (size_t o = 0; o < outer; ++o) {
        for (size_t in = 0; in < inner; ++in) {
            size_t base = o * dim_size * inner + in;
            float max_val = -1e30f;
            for (size_t d = 0; d < dim_size; ++d) {
                float v = ptr[base + d * inner];
                if (v > max_val) max_val = v;
            }
            float sum_exp = 0.0f;
            for (size_t d = 0; d < dim_size; ++d) {
                size_t idx = base + d * inner;
                ptr[idx] = std::exp(ptr[idx] - max_val);
                sum_exp += ptr[idx];
            }
            float inv_sum = 1.0f / sum_exp;
            for (size_t d = 0; d < dim_size; ++d) {
                ptr[base + d * inner] *= inv_sum;
            }
        }
    }
}

void Tensor::print(const std::string& name, int max_elements) const {
    std::cout << name << " shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << shape[i];
    }
    std::cout << "] dtype=" << (dtype == DType::FP32 ? "FP32" : "BF16")
              << " numel=" << numel << "\n";

    int count = std::min(static_cast<int>(numel), max_elements);
    std::cout << "  [";
    for (int i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (dtype == DType::FP32) {
            std::cout << std::fixed << std::setprecision(4) << fp32_data()[i];
        } else {
            std::cout << std::fixed << std::setprecision(4) << bf16_to_fp32(bf16_data()[i]);
        }
    }
    if (static_cast<int>(numel) > max_elements) std::cout << ", ...";
    std::cout << "]\n";
}

} // namespace core
} // namespace nol

// ==========================================
// UNIT TEST TENSOR
// ==========================================
static void run_tensor_tests() {
    using namespace nol::core;
    std::cout << "[TEST] Memulai unit test tensor...\n";

    // Test 1: Konstruktor & alokasi
    {
        Tensor t({2, 3}, DType::FP32);
        assert(t.numel == 6);
        assert(t.shape.size() == 2);
        assert(t.strides[0] == 3);
        assert(t.strides[1] == 1);
        assert(t.data != nullptr);
        // Zero-initialized
        for (size_t i = 0; i < 6; ++i) assert(t.fp32_data()[i] == 0.0f);
    }

    // Test 2: Fill & Scale
    {
        Tensor t({4}, DType::FP32);
        t.fill(2.0f);
        for (size_t i = 0; i < 4; ++i) assert(t.fp32_data()[i] == 2.0f);
        t.scale(3.0f);
        for (size_t i = 0; i < 4; ++i) assert(t.fp32_data()[i] == 6.0f);
    }

    // Test 3: Add & Mul
    {
        Tensor a({4}, DType::FP32);
        Tensor b({4}, DType::FP32);
        a.fill(2.0f);
        b.fill(3.0f);
        a.add(b);
        for (size_t i = 0; i < 4; ++i) assert(a.fp32_data()[i] == 5.0f);
        a.mul(b);
        for (size_t i = 0; i < 4; ++i) assert(a.fp32_data()[i] == 15.0f);
    }

    // Test 4: SiLU
    {
        Tensor t({1}, DType::FP32);
        t.fp32_data()[0] = 0.0f;
        t.silu_inplace();
        assert(std::abs(t.fp32_data()[0] - 0.0f) < 1e-6f); // silu(0) = 0
    }

    // Test 5: Sigmoid
    {
        Tensor t({1}, DType::FP32);
        t.fp32_data()[0] = 0.0f;
        t.sigmoid_inplace();
        assert(std::abs(t.fp32_data()[0] - 0.5f) < 1e-6f); // sigmoid(0) = 0.5
    }

    // Test 6: RMSNorm
    {
        Tensor x({2, 4}, DType::FP32);
        Tensor gamma({4}, DType::FP32);
        gamma.fill(1.0f);
        float* xp = x.fp32_data();
        // Row 0: [1, 1, 1, 1] -> rms = 1.0 -> normalized = [1,1,1,1]
        for (int i = 0; i < 4; ++i) xp[i] = 1.0f;
        // Row 1: [2, 2, 2, 2] -> rms = 2.0 -> normalized = [1,1,1,1]
        for (int i = 4; i < 8; ++i) xp[i] = 2.0f;
        x.rmsnorm_inplace(gamma);
        for (size_t i = 0; i < 8; ++i) {
            assert(std::abs(xp[i] - 1.0f) < 1e-5f);
        }
    }

    // Test 7: Softmax
    {
        Tensor t({1, 3}, DType::FP32);
        float* p = t.fp32_data();
        p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f;
        t.softmax_inplace(1);
        float sum = p[0] + p[1] + p[2];
        assert(std::abs(sum - 1.0f) < 1e-5f);
        assert(p[2] > p[1] && p[1] > p[0]); // urutan terjaga
    }

    // Test 8: BF16 conversion
    {
        float original = 3.14f;
        uint16_t bf = fp32_to_bf16(original);
        float recovered = bf16_to_fp32(bf);
        assert(std::abs(recovered - original) < 0.02f); // BF16 presisi terbatas
    }

    // Test 9: BF16 elementwise
    {
        Tensor a({4}, DType::BF16);
        Tensor b({4}, DType::BF16);
        a.fill(2.0f);
        b.fill(3.0f);
        a.add(b);
        for (size_t i = 0; i < 4; ++i) {
            assert(std::abs(bf16_to_fp32(a.bf16_data()[i]) - 5.0f) < 0.05f);
        }
    }

    // Test 10: Move semantics
    {
        Tensor a({10}, DType::FP32);
        a.fill(7.0f);
        Tensor b = std::move(a);
        assert(b.numel == 10);
        assert(b.fp32_data()[0] == 7.0f);
        assert(a.data == nullptr);
        assert(a.numel == 0);
    }

    // Test 11: Reshape
    {
        Tensor t({2, 3}, DType::FP32);
        t.reshape({3, 2});
        assert(t.shape[0] == 3 && t.shape[1] == 2);
        assert(t.numel == 6);
    }

    // Test 12: Print tidak crash
    {
        Tensor t({2, 3}, DType::FP32);
        t.fill(1.5f);
        t.print("TestTensor");
    }

    std::cout << "[PASS] ✅ Semua unit test tensor berhasil!\n\n";
}

// main HARUS di luar namespace
#ifdef NOL_TEST_TENSOR
int main() {
    run_tensor_tests();
    return 0;
}
#endif
