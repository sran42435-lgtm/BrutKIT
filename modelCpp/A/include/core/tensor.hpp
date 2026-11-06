#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

namespace nol {
namespace core {

enum class DType {
    FP32,
    BF16
};

/**
 * @brief Tensor Primitif dengan dukungan FP32 & BF16.
 * Pilar 1 - Lapisan 1: Manajemen alokasi memori tensor.
 * Pilar 1 - Lapisan 5: Operasi elementwise dasar.
 */
class Tensor {
public:
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t numel;
    DType dtype;
    void* data;
    bool owns_memory;

    // Constructors & Destructor
    Tensor();
    Tensor(const std::vector<size_t>& shape, DType dtype = DType::FP32);
    ~Tensor();

    // Non-copyable
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Movable
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Memory
    void allocate();
    void deallocate();
    void reshape(const std::vector<size_t>& new_shape);

    // Type-safe accessors
    float* fp32_data();
    const float* fp32_data() const;
    uint16_t* bf16_data();
    const uint16_t* bf16_data() const;

    // Elementwise Ops (Lapisan 5)
    void fill(float value);
    void copy_from(const Tensor& other);
    void add(const Tensor& other);
    void mul(const Tensor& other);
    void scale(float alpha);
    void silu_inplace();
    void sigmoid_inplace();
    void rmsnorm_inplace(const Tensor& gamma, float eps = 1e-5f);
    void softmax_inplace(int dim);

    // Debug
    void print(const std::string& name = "Tensor", int max_elements = 10) const;

private:
    void compute_strides();
    size_t get_aligned_size() const;
};

// Utilitas konversi BF16 <-> FP32
inline float bf16_to_fp32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t fp32_to_bf16(float fp32) {
    uint32_t bits;
    std::memcpy(&bits, &fp32, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

} // namespace core
} // namespace nol
