#include "architecture.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef ARCH_SELFTEST
#include <iostream>
#endif

namespace {

constexpr double kPi = 3.14159265358979323846;

std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double uniform01(std::uint64_t& state) {
    std::uint64_t x = splitmix64(state);
    return static_cast<double>(x >> 11) / 9007199254740992.0;
}

double gaussian01(std::uint64_t& state) {
    double u1 = uniform01(state);
    double u2 = uniform01(state);
    if (u1 < 1e-12) {
        u1 = 1e-12;
    }
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

double safeC(float c) {
    return c > 1e-12f ? static_cast<double>(c) : 1e-12;
}

double sigmoidDouble(double x) {
    if (x >= 0.0) {
        double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    double e = std::exp(x);
    return e / (1.0 + e);
}

float sigmoidFloat(float x) {
    return static_cast<float>(sigmoidDouble(static_cast<double>(x)));
}

double siluDouble(double x) {
    return x * sigmoidDouble(x);
}

double siluPrimeDouble(double x) {
    double s = sigmoidDouble(x);
    return s * (1.0 + x * (1.0 - s));
}

void randomNormalTensorState(arch::Tensor& t, float mean, float stddev, std::uint64_t& state) {
    if (t.data.empty()) {
        return;
    }
    if (stddev < 0.0f) {
        stddev = -stddev;
    }
    for (float& x : t.data) {
        x = mean + stddev * static_cast<float>(gaussian01(state));
    }
}

arch::LinearParams makeLinearParams(std::size_t out, std::size_t in, std::uint64_t& state, float biasFill = 0.0f) {
    arch::LinearParams lin;
    lin.w.shape = {static_cast<std::int64_t>(out), static_cast<std::int64_t>(in)};
    lin.b.shape = {static_cast<std::int64_t>(out)};
    lin.w.data.assign(out * in, 0.0f);
    lin.b.data.assign(out, biasFill);
    double denom = static_cast<double>(out) + static_cast<double>(in);
    float std = denom > 0.0 ? static_cast<float>(std::sqrt(2.0 / denom)) : 1.0f;
    randomNormalTensorState(lin.w, 0.0f, std, state);
    return lin;
}

arch::SwiGLUParams makeSwiGLUParams(std::size_t d, std::size_t ffn, std::uint64_t& state) {
    arch::SwiGLUParams f;
    f.w1 = makeLinearParams(ffn, d, state, 0.0f);
    f.w3 = makeLinearParams(ffn, d, state, 0.0f);
    f.w2 = makeLinearParams(d, ffn, state, 0.0f);
    return f;
}

arch::RMSNormParams makeRMSNorm(std::size_t d) {
    arch::RMSNormParams n;
    n.weight.shape = {static_cast<std::int64_t>(d)};
    n.weight.data.assign(d, 1.0f);
    n.eps = 1e-6f;
    return n;
}

void linearForwardRow(const float* x, std::size_t in, const arch::LinearParams& lin, float* y) {
    if (lin.w.shape.size() != 2) {
        throw std::runtime_error("linear weight must be 2D");
    }
    std::size_t out = static_cast<std::size_t>(lin.w.shape[0]);
    std::size_t inParam = static_cast<std::size_t>(lin.w.shape[1]);
    if (in != inParam) {
        throw std::runtime_error("linear input dimension mismatch");
    }
    if (lin.b.data.size() != out) {
        throw std::runtime_error("linear bias size mismatch");
    }
    for (std::size_t o = 0; o < out; ++o) {
        double sum = static_cast<double>(lin.b.data[o]);
        for (std::size_t k = 0; k < in; ++k) {
            sum += static_cast<double>(x[k]) * static_cast<double>(lin.w.data[o * in + k]);
        }
        y[o] = static_cast<float>(sum);
    }
}

void swigluForwardRow(const float* x, std::size_t d, const arch::SwiGLUParams& ffn, float* out) {
    if (ffn.w1.w.shape.size() != 2) {
        throw std::runtime_error("swiglu w1 must be 2D");
    }
    std::size_t ffnHidden = static_cast<std::size_t>(ffn.w1.w.shape[0]);
    std::vector<float> y1(ffnHidden, 0.0f);
    std::vector<float> y3(ffnHidden, 0.0f);
    linearForwardRow(x, d, ffn.w1, y1.data());
    linearForwardRow(x, d, ffn.w3, y3.data());
    for (std::size_t i = 0; i < ffnHidden; ++i) {
        y1[i] = static_cast<float>(siluDouble(static_cast<double>(y1[i])) * static_cast<double>(y3[i]));
    }
    linearForwardRow(y1.data(), ffnHidden, ffn.w2, out);
}

void ensureTensorLike(arch::Tensor& g, const arch::Tensor& ref) {
    g.shape = ref.shape;
    g.data.assign(ref.data.size(), 0.0f);
}

void ensureLinearGrad(const arch::LinearParams& p, arch::LinearGrad& g) {
    ensureTensorLike(g.dw, p.w);
    ensureTensorLike(g.db, p.b);
}

void ensureEmbeddingGrad(const arch::EmbeddingParams& p, arch::EmbeddingGrad& g) {
    ensureTensorLike(g.dtable, p.table);
}

void ensureRMSNormGrad(const arch::RMSNormParams& p, arch::RMSNormGrad& g) {
    ensureTensorLike(g.dweight, p.weight);
}

void ensureGQAGrad(const arch::GQAParams& p, arch::GQAGrad& g) {
    ensureLinearGrad(p.wq, g.wq);
    ensureLinearGrad(p.wk, g.wk);
    ensureLinearGrad(p.wv, g.wv);
    ensureLinearGrad(p.wo, g.wo);
}

void ensureSwiGLUGrad(const arch::SwiGLUParams& p, arch::SwiGLUGrad& g) {
    ensureLinearGrad(p.w1, g.w1);
    ensureLinearGrad(p.w3, g.w3);
    ensureLinearGrad(p.w2, g.w2);
}

void ensureHGAGrad(const arch::HGAParams& p, arch::HGAGrad& g) {
    ensureLinearGrad(p.wq, g.wq);
    ensureLinearGrad(p.wk, g.wk);
    ensureLinearGrad(p.wv, g.wv);
}

void ensureFusionGrad(const arch::FusionParams& p, arch::FusionGrad& g) {
    ensureLinearGrad(p.wg, g.wg);
    ensureLinearGrad(p.wu, g.wu);
}

void ensureCGLGrad(const arch::CGLParams& p, arch::CGLGrad& g) {
    ensureTensorLike(g.danchors, p.anchors);
    ensureTensorLike(g.dlambdaRaw, p.lambdaRaw);
    ensureTensorLike(g.dgamma, p.gamma);
    ensureLinearGrad(p.wc, g.wc);
}

void ensureTigGrad(const arch::TigParams& p, arch::TigGrad& g) {
    ensureLinearGrad(p.wr, g.wr);
    ensureTensorLike(g.danchors, p.anchors);
}

void ensureMoEGrad(const arch::MoEParams& p, arch::MoEGrad& g) {
    ensureSwiGLUGrad(p.shared, g.shared);
    g.experts.resize(p.experts.size());
    for (std::size_t i = 0; i < p.experts.size(); ++i) {
        ensureSwiGLUGrad(p.experts[i], g.experts[i]);
    }
}

void ensureHeadGrad(const arch::HeadParams& p, arch::HeadGrad& g) {
    ensureLinearGrad(p.euclidean, g.euclidean);
    ensureTensorLike(g.dhyperTable, p.hyperTable);
    ensureLinearGrad(p.wzHead, g.wzHead);
    ensureTensorLike(g.dlambdaH, p.lambdaH);
}

void ensureBlockGrad(const arch::BlockParams& p, arch::BlockGrad& g) {
    ensureRMSNormGrad(p.norm1, g.norm1);
    ensureRMSNormGrad(p.norm2, g.norm2);
    ensureGQAGrad(p.attn, g.attn);
    ensureLinearGrad(p.wz, g.wz);
    ensureHGAGrad(p.hga, g.hga);
    ensureFusionGrad(p.fusion, g.fusion);
    ensureCGLGrad(p.cgl, g.cgl);
    ensureTigGrad(p.tig, g.tig);
    ensureMoEGrad(p.moe, g.moe);
}

void ensureModelGrad(const arch::NolModelParams& p, arch::ModelGrad& g) {
    ensureEmbeddingGrad(p.embed, g.embed);
    g.blocks.resize(p.blocks.size());
    for (std::size_t i = 0; i < p.blocks.size(); ++i) {
        ensureBlockGrad(p.blocks[i], g.blocks[i]);
    }
    ensureRMSNormGrad(p.finalNorm, g.finalNorm);
    ensureHeadGrad(p.head, g.head);
}

void addTensorInPlace(arch::Tensor& dst, const arch::Tensor& src) {
    if (src.data.empty()) {
        return;
    }
    if (dst.data.size() != src.data.size()) {
        throw std::runtime_error("addTensorInPlace size mismatch");
    }
    for (std::size_t i = 0; i < dst.data.size(); ++i) {
        dst.data[i] += src.data[i];
    }
}

void addLinearGrad(arch::LinearGrad& dst, const arch::LinearGrad& src) {
    addTensorInPlace(dst.dw, src.dw);
    addTensorInPlace(dst.db, src.db);
}

void addSwiGLUGrad(arch::SwiGLUGrad& dst, const arch::SwiGLUGrad& src) {
    addLinearGrad(dst.w1, src.w1);
    addLinearGrad(dst.w3, src.w3);
    addLinearGrad(dst.w2, src.w2);
}

void softmaxBackward(const float* probs, const float* gradProbs, float* gradScores, int n) {
    if (n <= 0) {
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(gradProbs[i]) * static_cast<double>(probs[i]);
    }
    for (int i = 0; i < n; ++i) {
        double g = static_cast<double>(gradProbs[i]) - sum;
        gradScores[i] = static_cast<float>(static_cast<double>(probs[i]) * g);
    }
}

double vectorNormPtr(const float* p, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    }
    return std::sqrt(s);
}

} // namespace

namespace arch {

Tensor makeTensor(const std::vector<std::int64_t>& shape, float fill) {
    Tensor t;
    resizeTensor(t, shape, fill);
    return t;
}

std::size_t tensorNumel(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (std::int64_t d : shape) {
        if (d < 0) {
            return 0;
        }
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

bool resizeTensor(Tensor& t, const std::vector<std::int64_t>& shape, float fill) {
    std::size_t n = tensorNumel(shape);
    t.shape = shape;
    if (n == 0) {
        t.data.clear();
        return true;
    }
    t.data.assign(n, fill);
    return true;
}

void fillTensor(Tensor& t, float value) {
    std::fill(t.data.begin(), t.data.end(), value);
}

void randomNormalTensor(Tensor& t, float mean, float stddev, std::uint64_t seed) {
    if (t.data.empty()) {
        return;
    }
    std::uint64_t state = seed == 0 ? 123456789ULL : seed;
    if (stddev < 0.0f) {
        stddev = -stddev;
    }
    for (float& x : t.data) {
        x = mean + stddev * static_cast<float>(gaussian01(state));
    }
}

void gemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta) {
    if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2) {
        throw std::runtime_error("gemm requires 2D tensors");
    }
    std::int64_t M64 = A.shape[0];
    std::int64_t K64 = A.shape[1];
    std::int64_t K2_64 = B.shape[0];
    std::int64_t N64 = B.shape[1];
    if (M64 < 0 || K64 < 0 || K2_64 < 0 || N64 < 0) {
        throw std::runtime_error("gemm shape dimension is negative");
    }
    if (K64 != K2_64) {
        throw std::runtime_error("gemm inner dimensions do not match");
    }
    if (C.shape[0] != M64 || C.shape[1] != N64) {
        throw std::runtime_error("gemm output shape mismatch");
    }
    std::size_t M = static_cast<std::size_t>(M64);
    std::size_t K = static_cast<std::size_t>(K64);
    std::size_t N = static_cast<std::size_t>(N64);
    if (A.data.size() != M * K) {
        throw std::runtime_error("gemm A data size mismatch");
    }
    if (B.data.size() != K * N) {
        throw std::runtime_error("gemm B data size mismatch");
    }
    if (C.data.size() != M * N) {
        throw std::runtime_error("gemm C data size mismatch");
    }
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += static_cast<double>(A.data[i * K + k]) * static_cast<double>(B.data[k * N + j]);
            }
            double old = static_cast<double>(C.data[i * N + j]);
            double out = static_cast<double>(alpha) * sum + static_cast<double>(beta) * old;
            C.data[i * N + j] = static_cast<float>(out);
        }
    }
}

void gemmNaive(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta) {
    gemm(A, B, C, alpha, beta);
}

void addInPlace(Tensor& dst, const Tensor& src) {
    addTensorInPlace(dst, src);
}

void mulInPlace(Tensor& dst, const Tensor& src) {
    if (dst.data.size() != src.data.size()) {
        throw std::runtime_error("mulInPlace size mismatch");
    }
    for (std::size_t i = 0; i < dst.data.size(); ++i) {
        dst.data[i] *= src.data[i];
    }
}

void sigmoidInPlace(Tensor& t) {
    for (float& value : t.data) {
        value = sigmoidFloat(value);
    }
}

void siluInPlace(Tensor& t) {
    for (float& value : t.data) {
        value = static_cast<float>(siluDouble(static_cast<double>(value)));
    }
}

void softmaxInPlace(float* row, int n) {
    if (n <= 0 || row == nullptr) {
        return;
    }
    float maxv = row[0];
    for (int i = 1; i < n; ++i) {
        if (row[i] > maxv) {
            maxv = row[i];
        }
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(row[i]) - static_cast<double>(maxv));
    }
    if (sum <= 0.0) {
        float uniform = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) {
            row[i] = uniform;
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        double e = std::exp(static_cast<double>(row[i]) - static_cast<double>(maxv));
        row[i] = static_cast<float>(e / sum);
    }
}

float linner(const float* x, const float* y, int n) {
    if (n <= 0 || x == nullptr || y == nullptr) {
        return 0.0f;
    }
    double s = -static_cast<double>(x[0]) * static_cast<double>(y[0]);
    for (int i = 1; i < n; ++i) {
        s += static_cast<double>(x[i]) * static_cast<double>(y[i]);
    }
    return static_cast<float>(s);
}

void exp0(const float* v, float* x, int n, float c) {
    if (n <= 0 || v == nullptr || x == nullptr) {
        return;
    }
    double cc = safeC(c);
    double sc = std::sqrt(cc);
    if (n == 1) {
        x[0] = static_cast<float>(1.0 / sc);
        return;
    }
    double norm = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        norm += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    }
    norm = std::sqrt(norm);
    double docMaxNorm = 10.0 / sc;
    double fp32SafeMaxNorm = 3.5 / sc;
    double maxNorm = std::min(docMaxNorm, fp32SafeMaxNorm);
    double nc = std::min(norm, maxNorm);
    double x0 = std::cosh(sc * nc) / sc;
    double k = 1.0;
    if (norm >= 1e-7) {
        k = std::sinh(sc * nc) / (sc * norm);
    }
    x[0] = static_cast<float>(x0);
    for (int i = 0; i < n - 1; ++i) {
        x[i + 1] = static_cast<float>(k * static_cast<double>(v[i]));
    }
}

void log0(const float* x, float* v, int n, float c) {
    if (n <= 0 || x == nullptr || v == nullptr) {
        return;
    }
    if (n == 1) {
        return;
    }
    double cc = safeC(c);
    double sc = std::sqrt(cc);
    double ns = 0.0;
    for (int i = 1; i < n; ++i) {
        ns += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    ns = std::sqrt(ns);
    double rho = std::asinh(sc * ns) / sc;
    double k = 1.0;
    if (ns >= 1e-7) {
        k = rho / ns;
    }
    for (int i = 0; i < n - 1; ++i) {
        v[i] = static_cast<float>(k * static_cast<double>(x[i + 1]));
    }
}

float dist2Lorentz(const float* x, const float* y, int n, float c) {
    if (n <= 0 || x == nullptr || y == nullptr) {
        return 0.0f;
    }
    double cc = safeC(c);
    double d2 = -2.0 / cc - 2.0 * static_cast<double>(linner(x, y, n));
    if (d2 < 0.0) {
        d2 = 0.0;
    }
    return static_cast<float>(d2);
}

float distLorentz(const float* x, const float* y, int n, float c) {
    double d2 = static_cast<double>(dist2Lorentz(x, y, n, c));
    if (d2 < 0.0) {
        d2 = 0.0;
    }
    return static_cast<float>(std::sqrt(d2));
}

void lorentzCentroidFromWeightedSum(const float* z, float* mu, int n, float c) {
    if (n <= 0 || z == nullptr || mu == nullptr) {
        return;
    }
    double cc = safeC(c);
    double q = -static_cast<double>(linner(z, z, n));
    if (q < 1e-12) {
        q = 1e-12;
    }
    double inv = 1.0 / (std::sqrt(cc) * std::sqrt(q));
    for (int i = 0; i < n; ++i) {
        mu[i] = static_cast<float>(inv * static_cast<double>(z[i]));
    }
}

void lorentzCentroid(const float* const* points, const float* weights, int count, int n, float c, float* mu) {
    if (n <= 0 || mu == nullptr) {
        return;
    }
    double cc = safeC(c);
    if (count <= 0 || points == nullptr || weights == nullptr) {
        mu[0] = static_cast<float>(1.0 / std::sqrt(cc));
        for (int i = 1; i < n; ++i) {
            mu[i] = 0.0f;
        }
        return;
    }
    std::vector<float> z(static_cast<std::size_t>(n), 0.0f);
    for (int j = 0; j < count; ++j) {
        if (points[j] == nullptr) {
            continue;
        }
        float w = weights[j];
        if (w == 0.0f) {
            continue;
        }
        for (int i = 0; i < n; ++i) {
            z[static_cast<std::size_t>(i)] += w * points[j][i];
        }
    }
    lorentzCentroidFromWeightedSum(z.data(), mu, n, c);
}

float poincareRadius(float rho, float c) {
    double cc = safeC(c);
    double sc = std::sqrt(cc);
    double r = rho > 0.0f ? static_cast<double>(rho) : 0.0;
    return static_cast<float>(std::tanh(sc * r * 0.5) / sc);
}

void lorentzScoreBackward(const float* q, const float* k, float* dq, float* dk, int n, float c, float invTau, float gradScore) {
    (void)c;
    if (n <= 0 || q == nullptr || k == nullptr) {
        return;
    }
    float coef = 2.0f * invTau * gradScore;
    if (dq != nullptr) {
        dq[0] += coef * (-k[0]);
        for (int i = 1; i < n; ++i) {
            dq[i] += coef * k[i];
        }
    }
    if (dk != nullptr) {
        dk[0] += coef * (-q[0]);
        for (int i = 1; i < n; ++i) {
            dk[i] += coef * q[i];
        }
    }
}

void lorentzCentroidBackward(const float* z, const float* gradMu, float* gradZ, int n, float c) {
    if (n <= 0 || z == nullptr || gradMu == nullptr || gradZ == nullptr) {
        return;
    }
    double cc = safeC(c);
    double q = -static_cast<double>(linner(z, z, n));
    if (q < 1e-12) {
        q = 1e-12;
    }
    double a = 1.0 / (std::sqrt(cc) * std::sqrt(q));
    double dot = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(z[i]) * static_cast<double>(gradMu[i]);
    }
    double coef = dot / q;
    gradZ[0] += static_cast<float>(a * (static_cast<double>(gradMu[0]) - coef * static_cast<double>(z[0])));
    for (int i = 1; i < n; ++i) {
        gradZ[i] += static_cast<float>(a * (static_cast<double>(gradMu[i]) + coef * static_cast<double>(z[i])));
    }
}

void exp0Backward(const float* v, const float* gradX, float* gradV, int n, float c) {
    if (n <= 0 || v == nullptr || gradX == nullptr || gradV == nullptr) {
        return;
    }
    if (n == 1) {
        return;
    }
    std::fill(gradV, gradV + (n - 1), 0.0f);
    const double eps = 1e-3;
    std::vector<float> xp(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> xm(static_cast<std::size_t>(n), 0.0f);
    std::vector<float> vp(v, v + (n - 1));
    for (int i = 0; i < n - 1; ++i) {
        double orig = vp[static_cast<std::size_t>(i)];
        vp[static_cast<std::size_t>(i)] = static_cast<float>(orig + eps);
        exp0(vp.data(), xp.data(), n, c);
        vp[static_cast<std::size_t>(i)] = static_cast<float>(orig - eps);
        exp0(vp.data(), xm.data(), n, c);
        vp[static_cast<std::size_t>(i)] = static_cast<float>(orig);
        double fp = 0.0;
        double fm = 0.0;
        for (int j = 0; j < n; ++j) {
            fp += static_cast<double>(xp[static_cast<std::size_t>(j)]) * static_cast<double>(gradX[j]);
            fm += static_cast<double>(xm[static_cast<std::size_t>(j)]) * static_cast<double>(gradX[j]);
        }
        gradV[i] = static_cast<float>((fp - fm) / (2.0 * eps));
    }
}

void log0Backward(const float* x, const float* gradV, float* gradX, int n, float c) {
    if (n <= 0 || x == nullptr || gradV == nullptr || gradX == nullptr) {
        return;
    }
    std::fill(gradX, gradX + n, 0.0f);
    if (n <= 1) {
        return;
    }
    double cc = safeC(c);
    double sc = std::sqrt(cc);
    double r = vectorNormPtr(x + 1, static_cast<std::size_t>(n - 1));
    if (r < 1e-7) {
        for (int i = 0; i < n - 1; ++i) {
            gradX[i + 1] = gradV[i];
        }
        return;
    }
    double rho = std::asinh(sc * r) / sc;
    double f = rho / r;
    double rhoPrime = 1.0 / std::sqrt(1.0 + cc * r * r);
    double fp = (rhoPrime * r - rho) / (r * r);
    double dot = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        dot += static_cast<double>(gradV[i]) * static_cast<double>(x[i + 1]);
    }
    for (int i = 0; i < n - 1; ++i) {
        double yi = static_cast<double>(x[i + 1]);
        double gi = static_cast<double>(gradV[i]);
        gradX[i + 1] = static_cast<float>(f * gi + fp * yi * dot / r);
    }
}

void linearForward(const Tensor& x, const LinearParams& lin, Tensor& y) {
    if (x.shape.size() != 2) {
        throw std::runtime_error("linear input must be 2D");
    }
    if (lin.w.shape.size() != 2) {
        throw std::runtime_error("linear weight must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t in = static_cast<std::size_t>(x.shape[1]);
    std::size_t out = static_cast<std::size_t>(lin.w.shape[0]);
    std::size_t inParam = static_cast<std::size_t>(lin.w.shape[1]);
    if (in != inParam) {
        throw std::runtime_error("linear input dimension mismatch");
    }
    if (x.data.size() != T * in) {
        throw std::runtime_error("linear input data size mismatch");
    }
    if (lin.b.data.size() != out) {
        throw std::runtime_error("linear bias size mismatch");
    }
    y.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(out)};
    y.data.assign(T * out, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        linearForwardRow(x.data.data() + t * in, in, lin, y.data.data() + t * out);
    }
}

void linearBackward(const Tensor& x, const LinearParams& lin, const Tensor& gradY, Tensor& gradX, LinearGrad& grad) {
    ensureLinearGrad(lin, grad);
    if (x.shape.size() != 2 || gradY.shape.size() != 2) {
        throw std::runtime_error("linearBackward requires 2D tensors");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t in = static_cast<std::size_t>(x.shape[1]);
    std::size_t out = static_cast<std::size_t>(lin.w.shape[0]);
    gradX.shape = x.shape;
    gradX.data.assign(T * in, 0.0f);
    if (gradY.data.size() != T * out) {
        throw std::runtime_error("linearBackward gradY size mismatch");
    }
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t k = 0; k < in; ++k) {
            double sum = 0.0;
            for (std::size_t o = 0; o < out; ++o) {
                sum += static_cast<double>(gradY.data[t * out + o]) * static_cast<double>(lin.w.data[o * in + k]);
            }
            gradX.data[t * in + k] = static_cast<float>(sum);
        }
    }
    for (std::size_t o = 0; o < out; ++o) {
        double bsum = 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            bsum += static_cast<double>(gradY.data[t * out + o]);
        }
        grad.db.data[o] = static_cast<float>(bsum);
        for (std::size_t k = 0; k < in; ++k) {
            double sum = 0.0;
            for (std::size_t t = 0; t < T; ++t) {
                sum += static_cast<double>(gradY.data[t * out + o]) * static_cast<double>(x.data[t * in + k]);
            }
            grad.dw.data[o * in + k] = static_cast<float>(sum);
        }
    }
}

void embeddingForward(const EmbeddingParams& emb, const std::vector<int>& ids, Tensor& out) {
    if (emb.table.shape.size() != 2) {
        throw std::runtime_error("embedding table must be 2D");
    }
    std::size_t vocab = static_cast<std::size_t>(emb.table.shape[0]);
    std::size_t d = static_cast<std::size_t>(emb.table.shape[1]);
    std::size_t T = ids.size();
    out.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(d)};
    out.data.assign(T * d, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        int id = ids[t];
        if (id < 0 || static_cast<std::size_t>(id) >= vocab) {
            throw std::runtime_error("token id out of embedding range");
        }
        std::copy_n(emb.table.data.data() + static_cast<std::size_t>(id) * d, d, out.data.data() + t * d);
    }
}

void embeddingBackward(const std::vector<int>& ids, const Tensor& gradOut, EmbeddingGrad& grad) {
    if (grad.dtable.shape.size() != 2) {
        throw std::runtime_error("embedding grad table must be initialized");
    }
    std::size_t d = static_cast<std::size_t>(grad.dtable.shape[1]);
    std::size_t T = ids.size();
    if (gradOut.data.size() != T * d) {
        throw std::runtime_error("embeddingBackward grad size mismatch");
    }
    std::fill(grad.dtable.data.begin(), grad.dtable.data.end(), 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        int id = ids[t];
        if (id < 0) {
            continue;
        }
        std::size_t row = static_cast<std::size_t>(id);
        if (row >= static_cast<std::size_t>(grad.dtable.shape[0])) {
            continue;
        }
        for (std::size_t i = 0; i < d; ++i) {
            grad.dtable.data[row * d + i] += gradOut.data[t * d + i];
        }
    }
}

void rmsNormForward(const Tensor& x, const RMSNormParams& norm, Tensor& y) {
    if (x.shape.size() != 2) {
        throw std::runtime_error("rmsnorm input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    y.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(d)};
    y.data.assign(T * d, 0.0f);
    if (d == 0) {
        return;
    }
    if (norm.weight.data.size() != d) {
        throw std::runtime_error("rmsnorm weight size mismatch");
    }
    for (std::size_t t = 0; t < T; ++t) {
        const float* xr = x.data.data() + t * d;
        float* yr = y.data.data() + t * d;
        double ms = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            ms += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        }
        ms /= static_cast<double>(d);
        double rstd = 1.0 / std::sqrt(ms + static_cast<double>(norm.eps));
        for (std::size_t i = 0; i < d; ++i) {
            yr[i] = static_cast<float>(static_cast<double>(xr[i]) * rstd * static_cast<double>(norm.weight.data[i]));
        }
    }
}

void rmsNormBackward(const Tensor& x, const RMSNormParams& norm, const Tensor& gradY, Tensor& gradX, RMSNormGrad& grad) {
    ensureRMSNormGrad(norm, grad);
    if (x.shape.size() != 2) {
        throw std::runtime_error("rmsNormBackward input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    gradX.shape = x.shape;
    gradX.data.assign(T * d, 0.0f);
    if (d == 0) {
        return;
    }
    if (gradY.data.size() != T * d) {
        throw std::runtime_error("rmsNormBackward grad size mismatch");
    }
    std::fill(grad.dweight.data.begin(), grad.dweight.data.end(), 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        const float* xr = x.data.data() + t * d;
        const float* gyr = gradY.data.data() + t * d;
        float* gxr = gradX.data.data() + t * d;
        double ms = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            ms += static_cast<double>(xr[i]) * static_cast<double>(xr[i]);
        }
        ms /= static_cast<double>(d);
        double rstd = 1.0 / std::sqrt(ms + static_cast<double>(norm.eps));
        double dot = 0.0;
        for (std::size_t i = 0; i < d; ++i) {
            double xhat = static_cast<double>(xr[i]) * rstd;
            grad.dweight.data[i] += static_cast<float>(static_cast<double>(gyr[i]) * xhat);
            dot += static_cast<double>(gyr[i]) * static_cast<double>(norm.weight.data[i]) * static_cast<double>(xr[i]);
        }
        double correction = rstd * rstd * rstd * dot / static_cast<double>(d);
        for (std::size_t i = 0; i < d; ++i) {
            double first = rstd * static_cast<double>(norm.weight.data[i]) * static_cast<double>(gyr[i]);
            double second = correction * static_cast<double>(xr[i]);
            gxr[i] = static_cast<float>(first - second);
        }
    }
}

void ropeForward(Tensor& q, Tensor& k, int n_heads, int n_kv_heads, int d_head, const std::vector<int>& positions, float base) {
    if (q.shape.size() != 2 || k.shape.size() != 2) {
        throw std::runtime_error("rope q/k must be 2D");
    }
    if (d_head < 2) {
        return;
    }
    std::size_t T = static_cast<std::size_t>(q.shape[0]);
    if (positions.size() < T) {
        throw std::runtime_error("rope positions size mismatch");
    }
    std::size_t qStride = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(d_head);
    std::size_t kStride = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(d_head);
    if (q.shape[1] != static_cast<std::int64_t>(qStride)) {
        throw std::runtime_error("rope q shape mismatch");
    }
    if (k.shape[1] != static_cast<std::int64_t>(kStride)) {
        throw std::runtime_error("rope k shape mismatch");
    }
    for (std::size_t t = 0; t < T; ++t) {
        double pos = static_cast<double>(positions[t]);
        for (int pair = 0; pair + 1 < d_head; pair += 2) {
            double theta = std::pow(static_cast<double>(base), -2.0 * static_cast<double>(pair) / static_cast<double>(d_head));
            double angle = pos * theta;
            double co = std::cos(angle);
            double si = std::sin(angle);
            for (int h = 0; h < n_heads; ++h) {
                std::size_t idx = t * qStride + static_cast<std::size_t>(h) * static_cast<std::size_t>(d_head) + static_cast<std::size_t>(pair);
                double x1 = static_cast<double>(q.data[idx]);
                double x2 = static_cast<double>(q.data[idx + 1]);
                q.data[idx] = static_cast<float>(x1 * co - x2 * si);
                q.data[idx + 1] = static_cast<float>(x1 * si + x2 * co);
            }
            for (int h = 0; h < n_kv_heads; ++h) {
                std::size_t idx = t * kStride + static_cast<std::size_t>(h) * static_cast<std::size_t>(d_head) + static_cast<std::size_t>(pair);
                double x1 = static_cast<double>(k.data[idx]);
                double x2 = static_cast<double>(k.data[idx + 1]);
                k.data[idx] = static_cast<float>(x1 * co - x2 * si);
                k.data[idx + 1] = static_cast<float>(x1 * si + x2 * co);
            }
        }
    }
}

void ropeBackward(Tensor& gradQ, Tensor& gradK, int n_heads, int n_kv_heads, int d_head, const std::vector<int>& positions, float base) {
    if (gradQ.shape.size() != 2 || gradK.shape.size() != 2) {
        throw std::runtime_error("ropeBackward grad q/k must be 2D");
    }
    if (d_head < 2) {
        return;
    }
    std::size_t T = static_cast<std::size_t>(gradQ.shape[0]);
    if (positions.size() < T) {
        throw std::runtime_error("ropeBackward positions size mismatch");
    }
    std::size_t qStride = static_cast<std::size_t>(n_heads) * static_cast<std::size_t>(d_head);
    std::size_t kStride = static_cast<std::size_t>(n_kv_heads) * static_cast<std::size_t>(d_head);
    for (std::size_t t = 0; t < T; ++t) {
        double pos = static_cast<double>(positions[t]);
        for (int pair = 0; pair + 1 < d_head; pair += 2) {
            double theta = std::pow(static_cast<double>(base), -2.0 * static_cast<double>(pair) / static_cast<double>(d_head));
            double angle = pos * theta;
            double co = std::cos(angle);
            double si = std::sin(angle);
            for (int h = 0; h < n_heads; ++h) {
                std::size_t idx = t * qStride + static_cast<std::size_t>(h) * static_cast<std::size_t>(d_head) + static_cast<std::size_t>(pair);
                double g1 = static_cast<double>(gradQ.data[idx]);
                double g2 = static_cast<double>(gradQ.data[idx + 1]);
                gradQ.data[idx] = static_cast<float>(g1 * co + g2 * si);
                gradQ.data[idx + 1] = static_cast<float>(-g1 * si + g2 * co);
            }
            for (int h = 0; h < n_kv_heads; ++h) {
                std::size_t idx = t * kStride + static_cast<std::size_t>(h) * static_cast<std::size_t>(d_head) + static_cast<std::size_t>(pair);
                double g1 = static_cast<double>(gradK.data[idx]);
                double g2 = static_cast<double>(gradK.data[idx + 1]);
                gradK.data[idx] = static_cast<float>(g1 * co + g2 * si);
                gradK.data[idx + 1] = static_cast<float>(-g1 * si + g2 * co);
            }
        }
    }
}

void gqaForward(const Tensor& x, const GQAParams& attn, const std::vector<int>& positions, Tensor& out) {
    if (x.shape.size() != 2) {
        throw std::runtime_error("gqa input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    if (attn.n_heads <= 0 || attn.n_kv_heads <= 0 || attn.d_head <= 0) {
        throw std::runtime_error("invalid gqa dimensions");
    }
    if (d != static_cast<std::size_t>(attn.n_heads) * static_cast<std::size_t>(attn.d_head)) {
        throw std::runtime_error("gqa d_model must equal n_heads * d_head");
    }
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor ctx;
    linearForward(x, attn.wq, q);
    linearForward(x, attn.wk, k);
    linearForward(x, attn.wv, v);
    ropeForward(q, k, attn.n_heads, attn.n_kv_heads, attn.d_head, positions, attn.ropeBase);
    std::size_t qStride = static_cast<std::size_t>(attn.n_heads) * static_cast<std::size_t>(attn.d_head);
    std::size_t kvStride = static_cast<std::size_t>(attn.n_kv_heads) * static_cast<std::size_t>(attn.d_head);
    ctx.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(qStride)};
    ctx.data.assign(T * qStride, 0.0f);
    for (int h = 0; h < attn.n_heads; ++h) {
        int kvh = h % attn.n_kv_heads;
        std::size_t qOff = static_cast<std::size_t>(h) * static_cast<std::size_t>(attn.d_head);
        std::size_t kvOff = static_cast<std::size_t>(kvh) * static_cast<std::size_t>(attn.d_head);
        for (std::size_t i = 0; i < T; ++i) {
            std::size_t len = i + 1;
            std::vector<float> scores(len, 0.0f);
            for (std::size_t j = 0; j < len; ++j) {
                double dot = 0.0;
                for (int a = 0; a < attn.d_head; ++a) {
                    dot += static_cast<double>(q.data[i * qStride + qOff + static_cast<std::size_t>(a)]) *
                           static_cast<double>(k.data[j * kvStride + kvOff + static_cast<std::size_t>(a)]);
                }
                scores[j] = static_cast<float>(dot * static_cast<double>(attn.scale));
            }
            softmaxInPlace(scores.data(), static_cast<int>(len));
            for (int a = 0; a < attn.d_head; ++a) {
                double acc = 0.0;
                for (std::size_t j = 0; j < len; ++j) {
                    acc += static_cast<double>(scores[j]) * static_cast<double>(v.data[j * kvStride + kvOff + static_cast<std::size_t>(a)]);
                }
                ctx.data[i * qStride + qOff + static_cast<std::size_t>(a)] = static_cast<float>(acc);
            }
        }
    }
    linearForward(ctx, attn.wo, out);
}

void gqaBackward(const Tensor& x, const GQAParams& attn, const std::vector<int>& positions, const Tensor& gradOut, Tensor& gradX, GQAGrad& grad) {
    ensureGQAGrad(attn, grad);
    if (x.shape.size() != 2) {
        throw std::runtime_error("gqaBackward input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor ctx;
    linearForward(x, attn.wq, q);
    linearForward(x, attn.wk, k);
    linearForward(x, attn.wv, v);
    ropeForward(q, k, attn.n_heads, attn.n_kv_heads, attn.d_head, positions, attn.ropeBase);
    std::size_t qStride = static_cast<std::size_t>(attn.n_heads) * static_cast<std::size_t>(attn.d_head);
    std::size_t kvStride = static_cast<std::size_t>(attn.n_kv_heads) * static_cast<std::size_t>(attn.d_head);
    ctx.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(qStride)};
    ctx.data.assign(T * qStride, 0.0f);
    for (int h = 0; h < attn.n_heads; ++h) {
        int kvh = h % attn.n_kv_heads;
        std::size_t qOff = static_cast<std::size_t>(h) * static_cast<std::size_t>(attn.d_head);
        std::size_t kvOff = static_cast<std::size_t>(kvh) * static_cast<std::size_t>(attn.d_head);
        for (std::size_t i = 0; i < T; ++i) {
            std::size_t len = i + 1;
            std::vector<float> probs(len, 0.0f);
            for (std::size_t j = 0; j < len; ++j) {
                double dot = 0.0;
                for (int a = 0; a < attn.d_head; ++a) {
                    dot += static_cast<double>(q.data[i * qStride + qOff + static_cast<std::size_t>(a)]) *
                           static_cast<double>(k.data[j * kvStride + kvOff + static_cast<std::size_t>(a)]);
                }
                probs[j] = static_cast<float>(dot * static_cast<double>(attn.scale));
            }
            softmaxInPlace(probs.data(), static_cast<int>(len));
            for (int a = 0; a < attn.d_head; ++a) {
                double acc = 0.0;
                for (std::size_t j = 0; j < len; ++j) {
                    acc += static_cast<double>(probs[j]) * static_cast<double>(v.data[j * kvStride + kvOff + static_cast<std::size_t>(a)]);
                }
                ctx.data[i * qStride + qOff + static_cast<std::size_t>(a)] = static_cast<float>(acc);
            }
        }
    }
    Tensor gradCtx;
    linearBackward(ctx, attn.wo, gradOut, gradCtx, grad.wo);
    Tensor gradQ = makeTensor(q.shape, 0.0f);
    Tensor gradK = makeTensor(k.shape, 0.0f);
    Tensor gradV = makeTensor(v.shape, 0.0f);
    for (int h = 0; h < attn.n_heads; ++h) {
        int kvh = h % attn.n_kv_heads;
        std::size_t qOff = static_cast<std::size_t>(h) * static_cast<std::size_t>(attn.d_head);
        std::size_t kvOff = static_cast<std::size_t>(kvh) * static_cast<std::size_t>(attn.d_head);
        for (std::size_t i = 0; i < T; ++i) {
            std::size_t len = i + 1;
            std::vector<float> probs(len, 0.0f);
            std::vector<float> gradScores(len, 0.0f);
            std::vector<float> dScores(len, 0.0f);
            for (std::size_t j = 0; j < len; ++j) {
                double dot = 0.0;
                for (int a = 0; a < attn.d_head; ++a) {
                    dot += static_cast<double>(q.data[i * qStride + qOff + static_cast<std::size_t>(a)]) *
                           static_cast<double>(k.data[j * kvStride + kvOff + static_cast<std::size_t>(a)]);
                }
                probs[j] = static_cast<float>(dot * static_cast<double>(attn.scale));
            }
            softmaxInPlace(probs.data(), static_cast<int>(len));
            for (std::size_t j = 0; j < len; ++j) {
                double g = 0.0;
                for (int a = 0; a < attn.d_head; ++a) {
                    g += static_cast<double>(gradCtx.data[i * qStride + qOff + static_cast<std::size_t>(a)]) *
                         static_cast<double>(v.data[j * kvStride + kvOff + static_cast<std::size_t>(a)]);
                }
                gradScores[j] = static_cast<float>(g);
            }
            softmaxBackward(probs.data(), gradScores.data(), dScores.data(), static_cast<int>(len));
            for (std::size_t j = 0; j < len; ++j) {
                double ds = static_cast<double>(dScores[j]) * static_cast<double>(attn.scale);
                for (int a = 0; a < attn.d_head; ++a) {
                    gradQ.data[i * qStride + qOff + static_cast<std::size_t>(a)] +=
                        static_cast<float>(ds * static_cast<double>(k.data[j * kvStride + kvOff + static_cast<std::size_t>(a)]));
                    gradK.data[j * kvStride + kvOff + static_cast<std::size_t>(a)] +=
                        static_cast<float>(ds * static_cast<double>(q.data[i * qStride + qOff + static_cast<std::size_t>(a)]));
                    gradV.data[j * kvStride + kvOff + static_cast<std::size_t>(a)] +=
                        static_cast<float>(static_cast<double>(probs[j]) * static_cast<double>(gradCtx.data[i * qStride + qOff + static_cast<std::size_t>(a)]));
                }
            }
        }
    }
    ropeBackward(gradQ, gradK, attn.n_heads, attn.n_kv_heads, attn.d_head, positions, attn.ropeBase);
    Tensor gradXq;
    Tensor gradXk;
    Tensor gradXv;
    linearBackward(x, attn.wq, gradQ, gradXq, grad.wq);
    linearBackward(x, attn.wk, gradK, gradXk, grad.wk);
    linearBackward(x, attn.wv, gradV, gradXv, grad.wv);
    gradX = gradXq;
    addTensorInPlace(gradX, gradXk);
    addTensorInPlace(gradX, gradXv);
}

void swigluForward(const Tensor& x, const SwiGLUParams& ffn, Tensor& out) {
    if (x.shape.size() != 2) {
        throw std::runtime_error("swiglu input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    if (ffn.w2.w.shape.size() != 2) {
        throw std::runtime_error("swiglu w2 must be 2D");
    }
    std::size_t outDim = static_cast<std::size_t>(ffn.w2.w.shape[0]);
    out.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(outDim)};
    out.data.assign(T * outDim, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        swigluForwardRow(x.data.data() + t * d, d, ffn, out.data.data() + t * outDim);
    }
}

void swigluBackward(const Tensor& x, const SwiGLUParams& ffn, const Tensor& gradOut, Tensor& gradX, SwiGLUGrad& grad) {
    ensureSwiGLUGrad(ffn, grad);
    if (x.shape.size() != 2) {
        throw std::runtime_error("swigluBackward input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t ffnHidden = static_cast<std::size_t>(ffn.w1.w.shape[0]);
    Tensor y1;
    Tensor y3;
    Tensor act;
    linearForward(x, ffn.w1, y1);
    linearForward(x, ffn.w3, y3);
    act.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(ffnHidden)};
    act.data.assign(T * ffnHidden, 0.0f);
    for (std::size_t i = 0; i < T * ffnHidden; ++i) {
        act.data[i] = static_cast<float>(siluDouble(static_cast<double>(y1.data[i])) * static_cast<double>(y3.data[i]));
    }
    Tensor gradAct;
    linearBackward(act, ffn.w2, gradOut, gradAct, grad.w2);
    Tensor gradY1 = makeTensor(y1.shape, 0.0f);
    Tensor gradY3 = makeTensor(y3.shape, 0.0f);
    for (std::size_t i = 0; i < T * ffnHidden; ++i) {
        double ga = static_cast<double>(gradAct.data[i]);
        double x1 = static_cast<double>(y1.data[i]);
        double x3 = static_cast<double>(y3.data[i]);
        double silu = siluDouble(x1);
        double siluP = siluPrimeDouble(x1);
        gradY1.data[i] = static_cast<float>(ga * x3 * siluP);
        gradY3.data[i] = static_cast<float>(ga * silu);
    }
    Tensor gradX1;
    Tensor gradX3;
    linearBackward(x, ffn.w1, gradY1, gradX1, grad.w1);
    linearBackward(x, ffn.w3, gradY3, gradX3, grad.w3);
    gradX = gradX1;
    addTensorInPlace(gradX, gradX3);
}

void hgaForward(const Tensor& x, const HGAParams& hga, Tensor& u) {
    if (x.shape.size() != 2) {
        throw std::runtime_error("hga input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    if (hga.n_hyp_heads <= 0 || hga.d_hyp <= 0) {
        u.shape = {static_cast<std::int64_t>(T), 0};
        u.data.clear();
        return;
    }
    std::size_t hypOut = static_cast<std::size_t>(hga.n_hyp_heads) * static_cast<std::size_t>(hga.d_hyp);
    u.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(hypOut)};
    u.data.assign(T * hypOut, 0.0f);
    Tensor qTan;
    Tensor kTan;
    Tensor vTan;
    linearForward(x, hga.wq, qTan);
    linearForward(x, hga.wk, kTan);
    linearForward(x, hga.wv, vTan);
    int n = hga.d_hyp + 1;
    std::vector<float> qPts(T * static_cast<std::size_t>(hga.n_hyp_heads) * static_cast<std::size_t>(n), 0.0f);
    std::vector<float> kPts(qPts.size(), 0.0f);
    std::vector<float> vPts(qPts.size(), 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        for (int h = 0; h < hga.n_hyp_heads; ++h) {
            std::size_t tanOff = t * hypOut + static_cast<std::size_t>(h) * static_cast<std::size_t>(hga.d_hyp);
            std::size_t ptOff = (t * static_cast<std::size_t>(hga.n_hyp_heads) + static_cast<std::size_t>(h)) * static_cast<std::size_t>(n);
            exp0(qTan.data.data() + tanOff, qPts.data() + ptOff, n, hga.c);
            exp0(kTan.data.data() + tanOff, kPts.data() + ptOff, n, hga.c);
            exp0(vTan.data.data() + tanOff, vPts.data() + ptOff, n, hga.c);
        }
    }
    for (std::size_t i = 0; i < T; ++i) {
        for (int h = 0; h < hga.n_hyp_heads; ++h) {
            std::size_t len = i + 1;
            std::vector<float> scores(len, 0.0f);
            const float* qptr = qPts.data() + (i * static_cast<std::size_t>(hga.n_hyp_heads) + static_cast<std::size_t>(h)) * static_cast<std::size_t>(n);
            for (std::size_t j = 0; j < len; ++j) {
                const float* kptr = kPts.data() + (j * static_cast<std::size_t>(hga.n_hyp_heads) + static_cast<std::size_t>(h)) * static_cast<std::size_t>(n);
                double s = (2.0 / static_cast<double>(hga.c) + 2.0 * static_cast<double>(linner(qptr, kptr, n))) * static_cast<double>(hga.invTau);
                scores[j] = static_cast<float>(s);
            }
            softmaxInPlace(scores.data(), static_cast<int>(len));
            std::vector<float> zsum(static_cast<std::size_t>(n), 0.0f);
            for (std::size_t j = 0; j < len; ++j) {
                float w = scores[j];
                if (w == 0.0f) {
                    continue;
                }
                const float* vptr = vPts.data() + (j * static_cast<std::size_t>(hga.n_hyp_heads) + static_cast<std::size_t>(h)) * static_cast<std::size_t>(n);
                for (int dim = 0; dim < n; ++dim) {
                    zsum[static_cast<std::size_t>(dim)] += w * vptr[dim];
                }
            }
            std::vector<float> mu(static_cast<std::size_t>(n), 0.0f);
            lorentzCentroidFromWeightedSum(zsum.data(), mu.data(), n, hga.c);
            std::vector<float> tangent(static_cast<std::size_t>(hga.d_hyp), 0.0f);
            log0(mu.data(), tangent.data(), n, hga.c);
            std::size_t uOff = i * hypOut + static_cast<std::size_t>(h) * static_cast<std::size_t>(hga.d_hyp);
            for (int dim = 0; dim < hga.d_hyp; ++dim) {
                u.data[uOff + static_cast<std::size_t>(dim)] = tangent[static_cast<std::size_t>(dim)];
            }
        }
    }
}

void hgaBackward(const Tensor& x, const HGAParams& hga, const Tensor& gradU, Tensor& gradX, HGAGrad& grad) {
    ensureHGAGrad(hga, grad);
    if (x.shape.size() != 2) {
        throw std::runtime_error("hgaBackward input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    gradX.shape = x.shape;
    gradX.data.assign(x.data.size(), 0.0f);
    if (hga.n_hyp_heads <= 0 || hga.d_hyp <= 0 || gradU.data.empty()) {
        return;
    }
    std::size_t hypOut = static_cast<std::size_t>(hga.n_hyp_heads) * static_cast<std::size_t>(hga.d_hyp);
    Tensor qTan;
    Tensor kTan;
    Tensor vTan;
    linearForward(x, hga.wq, qTan);
    linearForward(x, hga.wk, kTan);
    linearForward(x, hga.wv, vTan);
    int n = hga.d_hyp + 1;
    std::size_t heads = static_cast<std::size_t>(hga.n_hyp_heads);
    std::vector<float> qPts(T * heads * static_cast<std::size_t>(n), 0.0f);
    std::vector<float> kPts(qPts.size(), 0.0f);
    std::vector<float> vPts(qPts.size(), 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            std::size_t tanOff = t * hypOut + h * static_cast<std::size_t>(hga.d_hyp);
            std::size_t ptOff = (t * heads + h) * static_cast<std::size_t>(n);
            exp0(qTan.data.data() + tanOff, qPts.data() + ptOff, n, hga.c);
            exp0(kTan.data.data() + tanOff, kPts.data() + ptOff, n, hga.c);
            exp0(vTan.data.data() + tanOff, vPts.data() + ptOff, n, hga.c);
        }
    }
    std::vector<float> gradQPts(qPts.size(), 0.0f);
    std::vector<float> gradKPts(kPts.size(), 0.0f);
    std::vector<float> gradVPts(vPts.size(), 0.0f);
    for (std::size_t i = 0; i < T; ++i) {
        for (std::size_t h = 0; h < heads; ++h) {
            std::size_t len = i + 1;
            std::vector<float> probs(len, 0.0f);
            std::vector<float> gradProbs(len, 0.0f);
            std::vector<float> gradScores(len, 0.0f);
            const float* qptr = qPts.data() + (i * heads + h) * static_cast<std::size_t>(n);
            for (std::size_t j = 0; j < len; ++j) {
                const float* kptr = kPts.data() + (j * heads + h) * static_cast<std::size_t>(n);
                double s = (2.0 / static_cast<double>(hga.c) + 2.0 * static_cast<double>(linner(qptr, kptr, n))) * static_cast<double>(hga.invTau);
                probs[j] = static_cast<float>(s);
            }
            softmaxInPlace(probs.data(), static_cast<int>(len));
            std::vector<float> zsum(static_cast<std::size_t>(n), 0.0f);
            for (std::size_t j = 0; j < len; ++j) {
                float w = probs[j];
                if (w == 0.0f) {
                    continue;
                }
                const float* vptr = vPts.data() + (j * heads + h) * static_cast<std::size_t>(n);
                for (int dim = 0; dim < n; ++dim) {
                    zsum[static_cast<std::size_t>(dim)] += w * vptr[dim];
                }
            }
            std::vector<float> mu(static_cast<std::size_t>(n), 0.0f);
            lorentzCentroidFromWeightedSum(zsum.data(), mu.data(), n, hga.c);
            const float* gradTan = gradU.data.data() + i * hypOut + h * static_cast<std::size_t>(hga.d_hyp);
            std::vector<float> gradMu(static_cast<std::size_t>(n), 0.0f);
            std::vector<float> gradZsum(static_cast<std::size_t>(n), 0.0f);
            log0Backward(mu.data(), gradTan, gradMu.data(), n, hga.c);
            lorentzCentroidBackward(zsum.data(), gradMu.data(), gradZsum.data(), n, hga.c);
            for (std::size_t j = 0; j < len; ++j) {
                const float* vptr = vPts.data() + (j * heads + h) * static_cast<std::size_t>(n);
                double dot = 0.0;
                for (int dim = 0; dim < n; ++dim) {
                    dot += static_cast<double>(vptr[dim]) * static_cast<double>(gradZsum[static_cast<std::size_t>(dim)]);
                }
                gradProbs[j] = static_cast<float>(dot);
            }
            softmaxBackward(probs.data(), gradProbs.data(), gradScores.data(), static_cast<int>(len));
            float* gradQhead = gradQPts.data() + (i * heads + h) * static_cast<std::size_t>(n);
            for (std::size_t j = 0; j < len; ++j) {
                if (gradScores[j] == 0.0f) {
                    continue;
                }
                const float* kptr = kPts.data() + (j * heads + h) * static_cast<std::size_t>(n);
                float* gradKhead = gradKPts.data() + (j * heads + h) * static_cast<std::size_t>(n);
                lorentzScoreBackward(qptr, kptr, gradQhead, gradKhead, n, hga.c, hga.invTau, gradScores[j]);
                float w = probs[j];
                float* gradVhead = gradVPts.data() + (j * heads + h) * static_cast<std::size_t>(n);
                for (int dim = 0; dim < n; ++dim) {
                    gradVhead[dim] += w * gradZsum[static_cast<std::size_t>(dim)];
                }
            }
        }
    }
    std::vector<float> gradQTan(T * hypOut, 0.0f);
    std::vector<float> gradKTan(T * hypOut, 0.0f);
    std::vector<float> gradVTan(T * hypOut, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t h = 0; h < heads; ++h) {
            std::size_t tanOff = t * hypOut + h * static_cast<std::size_t>(hga.d_hyp);
            std::size_t ptOff = (t * heads + h) * static_cast<std::size_t>(n);
            exp0Backward(qTan.data.data() + tanOff, gradQPts.data() + ptOff, gradQTan.data() + tanOff, n, hga.c);
            exp0Backward(kTan.data.data() + tanOff, gradKPts.data() + ptOff, gradKTan.data() + tanOff, n, hga.c);
            exp0Backward(vTan.data.data() + tanOff, gradVPts.data() + ptOff, gradVTan.data() + tanOff, n, hga.c);
        }
    }
    Tensor gradQTanT;
    Tensor gradKTanT;
    Tensor gradVTanT;
    gradQTanT.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(hypOut)};
    gradKTanT.shape = gradQTanT.shape;
    gradVTanT.shape = gradQTanT.shape;
    gradQTanT.data = gradQTan;
    gradKTanT.data = gradKTan;
    gradVTanT.data = gradVTan;
    Tensor gradXq;
    Tensor gradXk;
    Tensor gradXv;
    linearBackward(x, hga.wq, gradQTanT, gradXq, grad.wq);
    linearBackward(x, hga.wk, gradKTanT, gradXk, grad.wk);
    linearBackward(x, hga.wv, gradVTanT, gradXv, grad.wv);
    gradX = gradXq;
    addTensorInPlace(gradX, gradXk);
    addTensorInPlace(gradX, gradXv);
}

void gatedFusionForward(const Tensor& x, const Tensor& a, const Tensor& u, const FusionParams& fusion, Tensor& m) {
    if (x.shape.size() != 2 || a.shape.size() != 2) {
        throw std::runtime_error("fusion inputs must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    if (a.data.size() != T * d) {
        throw std::runtime_error("fusion a size mismatch");
    }
    m.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(d)};
    m.data = a.data;
    if (u.data.empty()) {
        return;
    }
    Tensor gate;
    Tensor proj;
    linearForward(x, fusion.wg, gate);
    sigmoidInPlace(gate);
    linearForward(u, fusion.wu, proj);
    if (gate.data.size() != T * d || proj.data.size() != T * d) {
        throw std::runtime_error("fusion internal size mismatch");
    }
    for (std::size_t i = 0; i < T * d; ++i) {
        m.data[i] += gate.data[i] * proj.data[i];
    }
}

void gatedFusionBackward(const Tensor& x, const Tensor& a, const Tensor& u, const FusionParams& fusion, const Tensor& gradM, Tensor& gradX, Tensor& gradU, FusionGrad& grad) {
    (void)a;
    ensureFusionGrad(fusion, grad);
    if (x.shape.size() != 2 || gradM.shape.size() != 2) {
        throw std::runtime_error("gatedFusionBackward inputs must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    gradX.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(d)};
    gradX.data.assign(T * d, 0.0f);
    if (u.data.empty()) {
        gradU = Tensor();
        return;
    }
    gradU.shape = u.shape;
    gradU.data.assign(u.data.size(), 0.0f);
    Tensor gatePre;
    Tensor gate;
    Tensor proj;
    linearForward(x, fusion.wg, gatePre);
    gate = gatePre;
    sigmoidInPlace(gate);
    linearForward(u, fusion.wu, proj);
    Tensor gradGate = makeTensor(gate.shape, 0.0f);
    Tensor gradProj = makeTensor(proj.shape, 0.0f);
    Tensor gradGatePre = makeTensor(gatePre.shape, 0.0f);
    for (std::size_t i = 0; i < T * d; ++i) {
        double gm = static_cast<double>(gradM.data[i]);
        double g = static_cast<double>(gate.data[i]);
        double p = static_cast<double>(proj.data[i]);
        gradGate.data[i] = static_cast<float>(gm * p);
        gradProj.data[i] = static_cast<float>(gm * g);
        gradGatePre.data[i] = static_cast<float>(gm * p * g * (1.0 - g));
    }
    linearBackward(x, fusion.wg, gradGatePre, gradX, grad.wg);
    linearBackward(u, fusion.wu, gradProj, gradU, grad.wu);
}

void cglDeltaForward(const Tensor& z, const CGLParams& cgl, Tensor& delta) {
    if (z.shape.size() != 2) {
        throw std::runtime_error("cgl z must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(z.shape[0]);
    std::size_t n = static_cast<std::size_t>(z.shape[1]);
    std::size_t dHyp = n > 0 ? n - 1 : 0;
    delta.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(dHyp)};
    delta.data.assign(T * dHyp, 0.0f);
    if (dHyp == 0 || cgl.anchors.shape.size() != 2) {
        return;
    }
    std::size_t A = static_cast<std::size_t>(cgl.anchors.shape[0]);
    if (A == 0) {
        return;
    }
    if (static_cast<std::size_t>(cgl.anchors.shape[1]) != dHyp) {
        throw std::runtime_error("cgl anchor dimension mismatch");
    }
    if (cgl.parent.size() != A) {
        throw std::runtime_error("cgl parent size mismatch");
    }
    std::vector<float> anchorPts(A * n, 0.0f);
    for (std::size_t k = 0; k < A; ++k) {
        exp0(cgl.anchors.data.data() + k * dHyp, anchorPts.data() + k * n, static_cast<int>(n), cgl.c);
    }
    std::vector<float> root(n, 0.0f);
    root[0] = static_cast<float>(1.0 / std::sqrt(safeC(cgl.c)));
    double lambda = sigmoidDouble(static_cast<double>(cgl.lambdaRaw.data.empty() ? 0.0f : cgl.lambdaRaw.data[0]));
    double lambda2 = lambda * lambda;
    for (std::size_t t = 0; t < T; ++t) {
        const float* zptr = z.data.data() + t * n;
        std::vector<float> logits(A, 0.0f);
        for (std::size_t k = 0; k < A; ++k) {
            float d2 = dist2Lorentz(zptr, anchorPts.data() + k * n, static_cast<int>(n), cgl.c);
            logits[k] = -d2 / cgl.tauCore;
        }
        softmaxInPlace(logits.data(), static_cast<int>(A));
        std::vector<float> m(n, 0.0f);
        for (std::size_t k = 0; k < A; ++k) {
            float p = logits[k];
            if (p == 0.0f) {
                continue;
            }
            double coeff = static_cast<double>(p) * lambda;
            for (std::size_t dim = 0; dim < n; ++dim) {
                m[dim] += static_cast<float>(coeff * static_cast<double>(anchorPts[k * n + dim]));
            }
            int par = cgl.parent[k];
            double coeff2 = static_cast<double>(p) * lambda2;
            if (par >= 0 && static_cast<std::size_t>(par) < A) {
                for (std::size_t dim = 0; dim < n; ++dim) {
                    m[dim] += static_cast<float>(coeff2 * static_cast<double>(anchorPts[static_cast<std::size_t>(par) * n + dim]));
                }
            } else {
                for (std::size_t dim = 0; dim < n; ++dim) {
                    m[dim] += static_cast<float>(coeff2 * static_cast<double>(root[dim]));
                }
            }
        }
        std::vector<float> zplus(n, 0.0f);
        for (std::size_t dim = 0; dim < n; ++dim) {
            zplus[dim] = zptr[dim] + m[dim];
        }
        std::vector<float> mu(n, 0.0f);
        lorentzCentroidFromWeightedSum(zplus.data(), mu.data(), static_cast<int>(n), cgl.c);
        std::vector<float> tanMu(dHyp, 0.0f);
        std::vector<float> tanZ(dHyp, 0.0f);
        log0(mu.data(), tanMu.data(), static_cast<int>(n), cgl.c);
        log0(zptr, tanZ.data(), static_cast<int>(n), cgl.c);
        for (std::size_t i = 0; i < dHyp; ++i) {
            delta.data[t * dHyp + i] = tanMu[i] - tanZ[i];
        }
    }
}

// ACTIVATE_FULL_CGL_GEOMETRY_BACKWARD_AFTER_PIPELINE_STABLE
//
// Safe zero-gradient stub for hyperbolic anchor/lambda geometry.
// Gradients for wc and gamma remain active in nolBlockBackward.
void cglGeometryBackward(const Tensor& z, const Tensor& gradDelta, Tensor& gradZ) {
    (void)gradDelta;
    gradZ.shape = z.shape;
    gradZ.data.assign(z.data.size(), 0.0f);
}

void tigForward(const Tensor& x, const Tensor& z, const TigParams& tig, TigOutput& out) {
    if (x.shape.size() != 2 || z.shape.size() != 2) {
        throw std::runtime_error("tig inputs must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t n = static_cast<std::size_t>(z.shape[1]);
    if (tig.nTop <= 0 || tig.topK <= 0 || n == 0) {
        out.topK = 0;
        out.probs.shape = {static_cast<std::int64_t>(T), 0};
        out.probs.data.clear();
        out.selected.clear();
        out.weights.shape = {0};
        out.weights.data.clear();
        return;
    }
    std::size_t nTop = static_cast<std::size_t>(tig.nTop);
    std::size_t topK = static_cast<std::size_t>(tig.topK);
    out.topK = tig.topK;
    out.probs.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(nTop)};
    out.probs.data.assign(T * nTop, 0.0f);
    out.selected.assign(T * topK, -1);
    out.weights.shape = {static_cast<std::int64_t>(T * topK)};
    out.weights.data.assign(T * topK, 0.0f);
    if (tig.anchors.shape.size() != 2) {
        throw std::runtime_error("tig anchors must be 2D");
    }
    if (static_cast<std::size_t>(tig.anchors.shape[0]) < nTop) {
        throw std::runtime_error("tig anchors count mismatch");
    }
    std::size_t dHyp = n - 1;
    if (static_cast<std::size_t>(tig.anchors.shape[1]) != dHyp) {
        throw std::runtime_error("tig anchor dimension mismatch");
    }
    std::vector<float> anchorPts(nTop * n, 0.0f);
    for (std::size_t e = 0; e < nTop; ++e) {
        exp0(tig.anchors.data.data() + e * dHyp, anchorPts.data() + e * n, static_cast<int>(n), tig.c);
    }
    Tensor rBase;
    linearForward(x, tig.wr, rBase);
    if (rBase.data.size() != T * nTop) {
        throw std::runtime_error("tig routing base size mismatch");
    }
    for (std::size_t t = 0; t < T; ++t) {
        const float* zptr = z.data.data() + t * n;
        std::vector<float> r(nTop, 0.0f);
        for (std::size_t e = 0; e < nTop; ++e) {
            float d2 = dist2Lorentz(zptr, anchorPts.data() + e * n, static_cast<int>(n), tig.c);
            r[e] = rBase.data[t * nTop + e] + tig.beta * (-d2 / tig.tauRoute);
        }
        std::vector<float> p = r;
        softmaxInPlace(p.data(), static_cast<int>(nTop));
        for (std::size_t e = 0; e < nTop; ++e) {
            out.probs.data[t * nTop + e] = p[e];
        }
        std::size_t eStar = 0;
        float best = r[0];
        for (std::size_t e = 1; e < nTop; ++e) {
            if (r[e] > best) {
                best = r[e];
                eStar = e;
            }
        }
        std::vector<float> distStar(nTop, 0.0f);
        for (std::size_t e = 0; e < nTop; ++e) {
            distStar[e] = distLorentz(anchorPts.data() + e * n, anchorPts.data() + eStar * n, static_cast<int>(n), tig.c);
        }
        std::vector<std::size_t> chosen;
        chosen.push_back(eStar);
        if (topK > 1) {
            std::size_t second = std::numeric_limits<std::size_t>::max();
            float secondVal = -std::numeric_limits<float>::infinity();
            for (std::size_t e = 0; e < nTop; ++e) {
                if (e == eStar) {
                    continue;
                }
                if (distStar[e] <= tig.isoDelta && r[e] > secondVal) {
                    secondVal = r[e];
                    second = e;
                }
            }
            if (second != std::numeric_limits<std::size_t>::max()) {
                chosen.push_back(second);
            }
        }
        std::size_t count = std::min(chosen.size(), topK);
        std::vector<float> selR(count, 0.0f);
        for (std::size_t i = 0; i < count; ++i) {
            selR[i] = r[chosen[i]];
            out.selected[t * topK + i] = static_cast<int>(chosen[i]);
        }
        softmaxInPlace(selR.data(), static_cast<int>(count));
        for (std::size_t i = 0; i < count; ++i) {
            out.weights.data[t * topK + i] = selR[i];
        }
    }
}

void tigBackward(const Tensor& x, const Tensor& z, const TigParams& tig, const TigOutput& route, const Tensor& gradWeights, Tensor& gradX, Tensor& gradZ, TigGrad& grad) {
    ensureTigGrad(tig, grad);
    gradX.shape = x.shape;
    gradX.data.assign(x.data.size(), 0.0f);
    gradZ.shape = z.shape;
    gradZ.data.assign(z.data.size(), 0.0f);
    if (x.shape.size() != 2 || z.shape.size() != 2) {
        return;
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t n = static_cast<std::size_t>(z.shape[1]);
    if (tig.nTop <= 0 || tig.topK <= 0 || n == 0) {
        return;
    }
    std::size_t nTop = static_cast<std::size_t>(tig.nTop);
    std::size_t topK = static_cast<std::size_t>(tig.topK);
    std::size_t dHyp = n - 1;
    std::vector<float> anchorPts(nTop * n, 0.0f);
    for (std::size_t e = 0; e < nTop; ++e) {
        exp0(tig.anchors.data.data() + e * dHyp, anchorPts.data() + e * n, static_cast<int>(n), tig.c);
    }
    Tensor rBase;
    linearForward(x, tig.wr, rBase);
    Tensor gradR = makeTensor({static_cast<std::int64_t>(T), static_cast<std::int64_t>(nTop)}, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        const float* zptr = z.data.data() + t * n;
        std::vector<float> r(nTop, 0.0f);
        for (std::size_t e = 0; e < nTop; ++e) {
            float d2 = dist2Lorentz(zptr, anchorPts.data() + e * n, static_cast<int>(n), tig.c);
            r[e] = rBase.data[t * nTop + e] + tig.beta * (-d2 / tig.tauRoute);
        }
        std::vector<std::size_t> sel;
        std::vector<float> selProbs;
        std::vector<float> gradSel;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            int id = route.selected[t * topK + slot];
            if (id < 0) {
                continue;
            }
            sel.push_back(static_cast<std::size_t>(id));
            selProbs.push_back(route.weights.data[t * topK + slot]);
            gradSel.push_back(gradWeights.data[t * topK + slot]);
        }
        if (sel.empty()) {
            continue;
        }
        std::vector<float> gradSelScores(sel.size(), 0.0f);
        softmaxBackward(selProbs.data(), gradSel.data(), gradSelScores.data(), static_cast<int>(sel.size()));
        for (std::size_t i = 0; i < sel.size(); ++i) {
            gradR.data[t * nTop + sel[i]] = gradSelScores[i];
        }
    }
    Tensor gradXBase;
    linearBackward(x, tig.wr, gradR, gradXBase, grad.wr);
    addTensorInPlace(gradX, gradXBase);
    std::vector<float> gradAnchorPts(nTop * n, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        const float* zptr = z.data.data() + t * n;
        float* zgrad = gradZ.data.data() + t * n;
        for (std::size_t e = 0; e < nTop; ++e) {
            float gr = gradR.data[t * nTop + e];
            if (gr == 0.0f) {
                continue;
            }
            const float* aptr = anchorPts.data() + e * n;
            float* agrad = gradAnchorPts.data() + e * n;
            lorentzScoreBackward(zptr, aptr, zgrad, agrad, static_cast<int>(n), tig.c, 1.0f / tig.tauRoute, tig.beta * gr);
        }
    }
    for (std::size_t e = 0; e < nTop; ++e) {
        exp0Backward(tig.anchors.data.data() + e * dHyp, gradAnchorPts.data() + e * n, grad.danchors.data.data() + e * dHyp, static_cast<int>(n), tig.c);
    }
}

void moeForward(const Tensor& x, const MoEParams& moe, const TigOutput& route, Tensor& out) {
    swigluForward(x, moe.shared, out);
    if (!moe.useMoe || moe.experts.empty()) {
        return;
    }
    if (x.shape.size() != 2) {
        throw std::runtime_error("moe input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    std::size_t topK = static_cast<std::size_t>(route.topK);
    if (topK == 0) {
        return;
    }
    if (route.selected.size() < T * topK) {
        return;
    }
    if (route.weights.data.size() < T * topK) {
        return;
    }
    std::vector<float> expertOut(d, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        const float* xrow = x.data.data() + t * d;
        for (std::size_t slot = 0; slot < topK; ++slot) {
            int id = route.selected[t * topK + slot];
            float w = route.weights.data[t * topK + slot];
            if (id < 0 || static_cast<std::size_t>(id) >= moe.experts.size() || w == 0.0f) {
                continue;
            }
            swigluForwardRow(xrow, d, moe.experts[static_cast<std::size_t>(id)], expertOut.data());
            for (std::size_t i = 0; i < d; ++i) {
                out.data[t * d + i] += w * expertOut[i];
            }
        }
    }
}

void moeBackward(const Tensor& x, const Tensor& z, const TigParams& tig, const MoEParams& moe, const TigOutput& route, const Tensor& gradOut, Tensor& gradX, Tensor& gradZ, TigGrad& tigGrad, MoEGrad& grad) {
    ensureMoEGrad(moe, grad);
    gradX.shape = x.shape;
    gradX.data.assign(x.data.size(), 0.0f);
    if (!z.data.empty()) {
        gradZ.shape = z.shape;
        gradZ.data.assign(z.data.size(), 0.0f);
    }
    Tensor gradXShared;
    swigluBackward(x, moe.shared, gradOut, gradXShared, grad.shared);
    addTensorInPlace(gradX, gradXShared);
    if (!moe.useMoe || moe.experts.empty() || route.topK <= 0) {
        return;
    }
    std::size_t T = static_cast<std::size_t>(x.shape[0]);
    std::size_t d = static_cast<std::size_t>(x.shape[1]);
    std::size_t topK = static_cast<std::size_t>(route.topK);
    Tensor gradWeights = makeTensor({static_cast<std::int64_t>(T * topK)}, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t slot = 0; slot < topK; ++slot) {
            int id = route.selected[t * topK + slot];
            float w = route.weights.data[t * topK + slot];
            if (id < 0 || static_cast<std::size_t>(id) >= moe.experts.size()) {
                continue;
            }
            Tensor xRow = makeTensor({1, static_cast<std::int64_t>(d)}, 0.0f);
            Tensor gradRow = makeTensor({1, static_cast<std::int64_t>(d)}, 0.0f);
            Tensor fwdRow = makeTensor({1, static_cast<std::int64_t>(d)}, 0.0f);
            for (std::size_t i = 0; i < d; ++i) {
                xRow.data[i] = x.data[t * d + i];
                gradRow.data[i] = gradOut.data[t * d + i];
            }
            swigluForward(xRow, moe.experts[static_cast<std::size_t>(id)], fwdRow);
            double dotv = 0.0;
            for (std::size_t i = 0; i < d; ++i) {
                dotv += static_cast<double>(fwdRow.data[i]) * static_cast<double>(gradRow.data[i]);
            }
            gradWeights.data[t * topK + slot] = static_cast<float>(dotv);
            for (std::size_t i = 0; i < d; ++i) {
                gradRow.data[i] = w * gradOut.data[t * d + i];
            }
            SwiGLUGrad tmp;
            Tensor gradXRow;
            swigluBackward(xRow, moe.experts[static_cast<std::size_t>(id)], gradRow, gradXRow, tmp);
            addSwiGLUGrad(grad.experts[static_cast<std::size_t>(id)], tmp);
            for (std::size_t i = 0; i < d; ++i) {
                gradX.data[t * d + i] += gradXRow.data[i];
            }
        }
    }
    if (z.data.empty()) {
        return;
    }
    Tensor gradXTig;
    Tensor gradZTig;
    tigBackward(x, z, tig, route, gradWeights, gradXTig, gradZTig, tigGrad);
    addTensorInPlace(gradX, gradXTig);
    addTensorInPlace(gradZ, gradZTig);
}

void headForward(const Tensor& xN, const Tensor& zN, const HeadParams& head, Tensor& logits) {
    linearForward(xN, head.euclidean, logits);
    if (!head.enableT5 || head.hyperTable.shape.size() != 2) {
        return;
    }
    std::size_t T = static_cast<std::size_t>(xN.shape[0]);
    std::size_t vocab = static_cast<std::size_t>(head.hyperTable.shape[0]);
    std::size_t dHyp = static_cast<std::size_t>(head.hyperTable.shape[1]);
    if (dHyp == 0) {
        return;
    }
    if (logits.shape.size() != 2 || static_cast<std::size_t>(logits.shape[1]) != vocab) {
        throw std::runtime_error("head logits vocab mismatch");
    }
    std::size_t n = dHyp + 1;
    std::vector<float> hyperPts(vocab * n, 0.0f);
    for (std::size_t v = 0; v < vocab; ++v) {
        exp0(head.hyperTable.data.data() + v * dHyp, hyperPts.data() + v * n, static_cast<int>(n), head.c);
    }
    std::vector<float> zPts(T * n, 0.0f);
    bool useProvidedZ =
        !zN.data.empty() &&
        zN.shape.size() == 2 &&
        static_cast<std::size_t>(zN.shape[0]) == T &&
        static_cast<std::size_t>(zN.shape[1]) == n;
    if (useProvidedZ) {
        std::copy_n(zN.data.data(), T * n, zPts.data());
    } else {
        Tensor zTan;
        linearForward(xN, head.wzHead, zTan);
        for (std::size_t t = 0; t < T; ++t) {
            exp0(zTan.data.data() + t * dHyp, zPts.data() + t * n, static_cast<int>(n), head.c);
        }
    }
    float lambda = head.lambdaH.data.empty() ? 0.0f : head.lambdaH.data[0];
    for (std::size_t t = 0; t < T; ++t) {
        const float* zptr = zPts.data() + t * n;
        for (std::size_t v = 0; v < vocab; ++v) {
            const float* hptr = hyperPts.data() + v * n;
            double sim = (2.0 / static_cast<double>(head.c) + 2.0 * static_cast<double>(linner(zptr, hptr, static_cast<int>(n)))) / static_cast<double>(head.tauHead);
            logits.data[t * vocab + v] += static_cast<float>(static_cast<double>(lambda) * sim);
        }
    }
}

void headBackward(const Tensor& xN, const HeadParams& head, const Tensor& gradLogits, Tensor& gradXN, HeadGrad& grad) {
    ensureHeadGrad(head, grad);
    linearBackward(xN, head.euclidean, gradLogits, gradXN, grad.euclidean);
    if (!head.enableT5 || head.hyperTable.shape.size() != 2) {
        return;
    }
    std::size_t T = static_cast<std::size_t>(xN.shape[0]);
    std::size_t vocab = static_cast<std::size_t>(head.hyperTable.shape[0]);
    std::size_t dHyp = static_cast<std::size_t>(head.hyperTable.shape[1]);
    if (dHyp == 0) {
        return;
    }
    std::size_t n = dHyp + 1;
    std::vector<float> hyperPts(vocab * n, 0.0f);
    for (std::size_t v = 0; v < vocab; ++v) {
        exp0(head.hyperTable.data.data() + v * dHyp, hyperPts.data() + v * n, static_cast<int>(n), head.c);
    }
    Tensor zTan;
    linearForward(xN, head.wzHead, zTan);
    std::vector<float> zPts(T * n, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        exp0(zTan.data.data() + t * dHyp, zPts.data() + t * n, static_cast<int>(n), head.c);
    }
    std::vector<float> gradZPts(T * n, 0.0f);
    std::vector<float> gradHyperPts(vocab * n, 0.0f);
    double gradLambda = 0.0;
    float lambda = head.lambdaH.data.empty() ? 0.0f : head.lambdaH.data[0];
    for (std::size_t t = 0; t < T; ++t) {
        const float* zptr = zPts.data() + t * n;
        float* zgrad = gradZPts.data() + t * n;
        for (std::size_t v = 0; v < vocab; ++v) {
            const float* hptr = hyperPts.data() + v * n;
            float* hgrad = gradHyperPts.data() + v * n;
            double score = (2.0 / static_cast<double>(head.c) + 2.0 * static_cast<double>(linner(zptr, hptr, static_cast<int>(n)))) / static_cast<double>(head.tauHead);
            double gl = static_cast<double>(gradLogits.data[t * vocab + v]);
            gradLambda += gl * score;
            if (lambda != 0.0f && gl != 0.0) {
                lorentzScoreBackward(zptr, hptr, zgrad, hgrad, static_cast<int>(n), head.c, 1.0f / head.tauHead, static_cast<float>(static_cast<double>(lambda) * gl));
            }
        }
    }
    grad.dlambdaH.data[0] = static_cast<float>(gradLambda);
    std::vector<float> gradZTan(T * dHyp, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        exp0Backward(zTan.data.data() + t * dHyp, gradZPts.data() + t * n, gradZTan.data() + t * dHyp, static_cast<int>(n), head.c);
    }
    Tensor gradZTanT;
    gradZTanT.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(dHyp)};
    gradZTanT.data = gradZTan;
    Tensor gradXWz;
    linearBackward(xN, head.wzHead, gradZTanT, gradXWz, grad.wzHead);
    addTensorInPlace(gradXN, gradXWz);
    for (std::size_t v = 0; v < vocab; ++v) {
        exp0Backward(head.hyperTable.data.data() + v * dHyp, gradHyperPts.data() + v * n, grad.dhyperTable.data.data() + v * dHyp, static_cast<int>(n), head.c);
    }
}

void nolBlockForward(const Tensor& hIn, const BlockParams& block, const std::vector<int>& positions, Tensor& hOut) {
    if (hIn.shape.size() != 2) {
        throw std::runtime_error("block input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(hIn.shape[0]);
    std::size_t d = static_cast<std::size_t>(hIn.shape[1]);
    if (d != static_cast<std::size_t>(block.d_model)) {
        throw std::runtime_error("block d_model mismatch");
    }
    hOut.shape = hIn.shape;
    hOut.data = hIn.data;
    Tensor x;
    rmsNormForward(hIn, block.norm1, x);
    Tensor a;
    gqaForward(x, block.attn, positions, a);
    Tensor zPts;
    std::size_t n = 0;
    if (block.d_hyp > 0) {
        Tensor zTan;
        linearForward(x, block.wz, zTan);
        n = static_cast<std::size_t>(block.d_hyp) + 1;
        zPts.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(n)};
        zPts.data.assign(T * n, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            exp0(zTan.data.data() + t * static_cast<std::size_t>(block.d_hyp), zPts.data.data() + t * n, static_cast<int>(n), block.cgl.c);
        }
    }
    Tensor u;
    if (block.enableT1 && block.d_hyp > 0 && block.n_hyp_heads > 0) {
        hgaForward(x, block.hga, u);
    }
    Tensor m;
    if (block.enableT2) {
        gatedFusionForward(x, a, u, block.fusion, m);
    } else {
        m = a;
    }
    if (m.data.size() != T * d) {
        throw std::runtime_error("block fusion output size mismatch");
    }
    for (std::size_t i = 0; i < T * d; ++i) {
        hOut.data[i] += m.data[i];
    }
    if (block.enableT3 && block.d_hyp > 0) {
        Tensor delta;
        cglDeltaForward(zPts, block.cgl, delta);
        Tensor proj;
        linearForward(delta, block.cgl.wc, proj);
        if (proj.data.size() != T * d) {
            throw std::runtime_error("cgl projection size mismatch");
        }
        if (block.cgl.gamma.data.size() != d) {
            throw std::runtime_error("cgl gamma size mismatch");
        }
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t i = 0; i < d; ++i) {
                hOut.data[t * d + i] += block.cgl.gamma.data[i] * proj.data[t * d + i];
            }
        }
    }
    Tensor x2;
    rmsNormForward(hOut, block.norm2, x2);
    TigOutput route;
    if (block.enableT4 && block.d_hyp > 0 && block.moe.useMoe) {
        tigForward(x2, zPts, block.tig, route);
    }
    Tensor ffnOut;
    moeForward(x2, block.moe, route, ffnOut);
    if (ffnOut.data.size() != T * d) {
        throw std::runtime_error("block ffn output size mismatch");
    }
    for (std::size_t i = 0; i < T * d; ++i) {
        hOut.data[i] += ffnOut.data[i];
    }
}

void nolBlockBackward(const Tensor& hIn, const BlockParams& block, const std::vector<int>& positions, const Tensor& gradHOut, Tensor& gradHIn, BlockGrad& grad) {
    ensureBlockGrad(block, grad);
    if (hIn.shape.size() != 2) {
        throw std::runtime_error("blockBackward input must be 2D");
    }
    std::size_t T = static_cast<std::size_t>(hIn.shape[0]);
    std::size_t d = static_cast<std::size_t>(hIn.shape[1]);
    Tensor x;
    Tensor a;
    Tensor zPts;
    Tensor zTan;
    Tensor u;
    Tensor m;
    Tensor delta;
    Tensor proj;
    Tensor h2;
    Tensor x2;
    TigOutput route;
    rmsNormForward(hIn, block.norm1, x);
    gqaForward(x, block.attn, positions, a);
    if (block.d_hyp > 0) {
        linearForward(x, block.wz, zTan);
        std::size_t n = static_cast<std::size_t>(block.d_hyp) + 1;
        zPts.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(n)};
        zPts.data.assign(T * n, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            exp0(zTan.data.data() + t * static_cast<std::size_t>(block.d_hyp), zPts.data.data() + t * n, static_cast<int>(n), block.cgl.c);
        }
    }
    if (block.enableT1 && block.d_hyp > 0 && block.n_hyp_heads > 0) {
        hgaForward(x, block.hga, u);
    }
    if (block.enableT2) {
        gatedFusionForward(x, a, u, block.fusion, m);
    } else {
        m = a;
    }
    Tensor h1 = hIn;
    addTensorInPlace(h1, m);
    if (block.enableT3 && block.d_hyp > 0) {
        cglDeltaForward(zPts, block.cgl, delta);
        linearForward(delta, block.cgl.wc, proj);
        h2 = h1;
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t i = 0; i < d; ++i) {
                h2.data[t * d + i] += block.cgl.gamma.data[i] * proj.data[t * d + i];
            }
        }
    } else {
        h2 = h1;
    }
    rmsNormForward(h2, block.norm2, x2);
    if (block.enableT4 && block.d_hyp > 0 && block.moe.useMoe) {
        tigForward(x2, zPts, block.tig, route);
    }
    Tensor gradX2;
    Tensor gradZ;
    Tensor gradH2 = gradHOut;
    moeBackward(x2, zPts, block.tig, block.moe, route, gradHOut, gradX2, gradZ, grad.tig, grad.moe);
    Tensor gradH2FromNorm;
    rmsNormBackward(h2, block.norm2, gradX2, gradH2FromNorm, grad.norm2);
    addTensorInPlace(gradH2, gradH2FromNorm);
    if (block.enableT3 && block.d_hyp > 0) {
        Tensor gradProj = makeTensor(proj.shape, 0.0f);
        Tensor gradGamma = makeTensor(block.cgl.gamma.shape, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            for (std::size_t i = 0; i < d; ++i) {
                double gh = static_cast<double>(gradH2.data[t * d + i]);
                double p = static_cast<double>(proj.data[t * d + i]);
                double g = static_cast<double>(block.cgl.gamma.data[i]);
                gradProj.data[t * d + i] = static_cast<float>(g * gh);
                gradGamma.data[i] += static_cast<float>(gh * p);
            }
        }
        Tensor gradDelta;
        linearBackward(delta, block.cgl.wc, gradProj, gradDelta, grad.cgl.wc);
        grad.cgl.dgamma = gradGamma;
        Tensor gradZCgl;
        cglGeometryBackward(zPts, gradDelta, gradZCgl);
        addTensorInPlace(gradZ, gradZCgl);
    }
    Tensor gradH1 = gradH2;
    Tensor gradXfusion = makeTensor({static_cast<std::int64_t>(T), static_cast<std::int64_t>(d)}, 0.0f);
    Tensor gradUfusion;
    Tensor gradA = gradH1;
    if (block.enableT2) {
        gatedFusionBackward(x, a, u, block.fusion, gradH1, gradXfusion, gradUfusion, grad.fusion);
    } else {
        gradUfusion = Tensor();
    }
    Tensor gradXhga = makeTensor({static_cast<std::int64_t>(T), static_cast<std::int64_t>(d)}, 0.0f);
    if (block.enableT1 && !gradUfusion.data.empty()) {
        hgaBackward(x, block.hga, gradUfusion, gradXhga, grad.hga);
    }
    Tensor gradXattn;
    gqaBackward(x, block.attn, positions, gradA, gradXattn, grad.attn);
    Tensor gradXwz = makeTensor({static_cast<std::int64_t>(T), static_cast<std::int64_t>(d)}, 0.0f);
    if (block.d_hyp > 0 && !gradZ.data.empty()) {
        std::size_t n = static_cast<std::size_t>(block.d_hyp) + 1;
        Tensor gradZTan = makeTensor({static_cast<std::int64_t>(T), static_cast<std::int64_t>(block.d_hyp)}, 0.0f);
        for (std::size_t t = 0; t < T; ++t) {
            exp0Backward(
                zTan.data.data() + t * static_cast<std::size_t>(block.d_hyp),
                gradZ.data.data() + t * n,
                gradZTan.data.data() + t * static_cast<std::size_t>(block.d_hyp),
                static_cast<int>(n),
                block.cgl.c
            );
        }
        linearBackward(x, block.wz, gradZTan, gradXwz, grad.wz);
    }
    Tensor gradX = gradXfusion;
    addTensorInPlace(gradX, gradXhga);
    addTensorInPlace(gradX, gradXattn);
    addTensorInPlace(gradX, gradXwz);
    Tensor gradHInFromNorm;
    rmsNormBackward(hIn, block.norm1, gradX, gradHInFromNorm, grad.norm1);
    gradHIn = gradH1;
    addTensorInPlace(gradHIn, gradHInFromNorm);
}

void nolModelForward(const NolModelParams& model, const std::vector<int>& ids, const std::vector<int>& positions, Tensor& logits) {
    if (positions.size() != ids.size()) {
        throw std::runtime_error("positions size must match ids size");
    }
    Tensor h;
    embeddingForward(model.embed, ids, h);
    for (const BlockParams& block : model.blocks) {
        Tensor next;
        nolBlockForward(h, block, positions, next);
        h = std::move(next);
    }
    Tensor xN;
    rmsNormForward(h, model.finalNorm, xN);
    Tensor zN;
    if (model.head.enableT5 && model.head.wzHead.w.shape.size() == 2) {
        std::size_t dHyp = static_cast<std::size_t>(model.head.wzHead.w.shape[0]);
        if (dHyp > 0) {
            Tensor zTan;
            linearForward(xN, model.head.wzHead, zTan);
            std::size_t T = static_cast<std::size_t>(xN.shape[0]);
            std::size_t n = dHyp + 1;
            zN.shape = {static_cast<std::int64_t>(T), static_cast<std::int64_t>(n)};
            zN.data.assign(T * n, 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                exp0(zTan.data.data() + t * dHyp, zN.data.data() + t * n, static_cast<int>(n), model.head.c);
            }
        }
    }
    headForward(xN, zN, model.head, logits);
}

void nolModelBackward(const NolModelParams& model, const std::vector<int>& ids, const std::vector<int>& positions, const Tensor& gradLogits, ModelGrad& grad) {
    ensureModelGrad(model, grad);
    std::vector<Tensor> hidden;
    hidden.reserve(model.blocks.size() + 1);
    Tensor h;
    embeddingForward(model.embed, ids, h);
    hidden.push_back(h);
    for (const BlockParams& block : model.blocks) {
        Tensor next;
        nolBlockForward(h, block, positions, next);
        h = std::move(next);
        hidden.push_back(h);
    }
    Tensor xN;
    rmsNormForward(h, model.finalNorm, xN);
    Tensor gradXN;
    headBackward(xN, model.head, gradLogits, gradXN, grad.head);
    Tensor gradH;
    rmsNormBackward(h, model.finalNorm, gradXN, gradH, grad.finalNorm);
    for (std::size_t li = model.blocks.size(); li > 0; --li) {
        Tensor gradHIn;
        nolBlockBackward(hidden[li - 1], model.blocks[li - 1], positions, gradH, gradHIn, grad.blocks[li - 1]);
        gradH = gradHIn;
    }
    embeddingBackward(ids, gradH, grad.embed);
}

NolModelParams createRandomModel(const ArchitectureDims& dimsIn) {
    ArchitectureDims dims = dimsIn;
    if (dims.d_model <= 0) {
        throw std::runtime_error("d_model must be positive");
    }
    if (dims.n_heads <= 0) {
        throw std::runtime_error("n_heads must be positive");
    }
    if (dims.d_model % dims.n_heads != 0) {
        throw std::runtime_error("d_model must be divisible by n_heads");
    }
    if (dims.d_head <= 0) {
        dims.d_head = dims.d_model / dims.n_heads;
    }
    if (dims.d_head != dims.d_model / dims.n_heads) {
        throw std::runtime_error("d_head must equal d_model / n_heads");
    }
    if (dims.n_kv_heads <= 0) {
        dims.n_kv_heads = 1;
    }
    if (dims.n_kv_heads > dims.n_heads) {
        throw std::runtime_error("n_kv_heads must be <= n_heads");
    }
    if (dims.n_heads % dims.n_kv_heads != 0) {
        throw std::runtime_error("n_heads must be divisible by n_kv_heads");
    }
    if (dims.n_layers <= 0) {
        throw std::runtime_error("n_layers must be positive");
    }
    if (dims.vocab <= 0) {
        throw std::runtime_error("vocab must be positive");
    }
    if (dims.ffn_hidden <= 0) {
        dims.ffn_hidden = dims.d_model * 2;
    }
    if (dims.d_hyp < 0) {
        dims.d_hyp = 0;
    }
    if (dims.n_hyp_heads < 0) {
        dims.n_hyp_heads = 0;
    }
    if (dims.n_top_anchors < 0) {
        dims.n_top_anchors = 0;
    }
    if (dims.n_children < 0) {
        dims.n_children = 0;
    }
    bool hyperUsable = dims.d_hyp > 0;
    dims.enableT1 = dims.enableT1 && hyperUsable && dims.n_hyp_heads > 0;
    dims.enableT2 = dims.enableT2 && hyperUsable && dims.n_hyp_heads > 0;
    dims.enableT3 = dims.enableT3 && hyperUsable;
    dims.enableT4 = dims.enableT4 && hyperUsable && dims.useMoe && dims.n_top_anchors > 0;
    dims.enableT5 = dims.enableT5 && hyperUsable;
    std::uint64_t state = dims.seed == 0 ? 123456789ULL : dims.seed;
    std::size_t V = static_cast<std::size_t>(dims.vocab);
    std::size_t d = static_cast<std::size_t>(dims.d_model);
    std::size_t L = static_cast<std::size_t>(dims.n_layers);
    std::size_t H = static_cast<std::size_t>(dims.n_heads);
    std::size_t KV = static_cast<std::size_t>(dims.n_kv_heads);
    std::size_t Dh = static_cast<std::size_t>(dims.d_head);
    std::size_t F = static_cast<std::size_t>(dims.ffn_hidden);
    std::size_t dHyp = static_cast<std::size_t>(dims.d_hyp);
    std::size_t HH = static_cast<std::size_t>(dims.n_hyp_heads);
    std::size_t hypOut = HH * dHyp;
    std::size_t nTop = static_cast<std::size_t>(dims.n_top_anchors);
    std::size_t nChildren = static_cast<std::size_t>(dims.n_children);
    std::size_t totalAnchors = nTop + nTop * nChildren;
    NolModelParams model;
    model.vocab = dims.vocab;
    model.d_model = dims.d_model;
    model.embed.table.shape = {static_cast<std::int64_t>(V), static_cast<std::int64_t>(d)};
    model.embed.table.data.assign(V * d, 0.0f);
    randomNormalTensorState(model.embed.table, 0.0f, 0.02f, state);
    model.finalNorm = makeRMSNorm(d);
    model.blocks.reserve(L);
    for (std::size_t layer = 0; layer < L; ++layer) {
        BlockParams block;
        block.d_model = dims.d_model;
        block.d_hyp = dims.d_hyp;
        block.n_hyp_heads = dims.n_hyp_heads;
        block.ropeBase = dims.ropeBase;
        block.enableT1 = dims.enableT1;
        block.enableT2 = dims.enableT2;
        block.enableT3 = dims.enableT3;
        block.enableT4 = dims.enableT4;
        block.norm1 = makeRMSNorm(d);
        block.norm2 = makeRMSNorm(d);
        block.attn.n_heads = dims.n_heads;
        block.attn.n_kv_heads = dims.n_kv_heads;
        block.attn.d_head = dims.d_head;
        block.attn.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dims.d_head)));
        block.attn.ropeBase = dims.ropeBase;
        block.attn.wq = makeLinearParams(H * Dh, d, state, 0.0f);
        block.attn.wk = makeLinearParams(KV * Dh, d, state, 0.0f);
        block.attn.wv = makeLinearParams(KV * Dh, d, state, 0.0f);
        block.attn.wo = makeLinearParams(d, H * Dh, state, 0.0f);
        block.wz = makeLinearParams(dHyp, d, state, 0.0f);
        block.hga.n_hyp_heads = dims.n_hyp_heads;
        block.hga.d_hyp = dims.d_hyp;
        block.hga.c = dims.cInit;
        block.hga.invTau = 1.0f / dims.tauAtt;
        block.hga.wq = makeLinearParams(hypOut, d, state, 0.0f);
        block.hga.wk = makeLinearParams(hypOut, d, state, 0.0f);
        block.hga.wv = makeLinearParams(hypOut, d, state, 0.0f);
        block.fusion.wg = makeLinearParams(d, d, state, -2.0f);
        block.fusion.wu = makeLinearParams(d, hypOut, state, 0.0f);
        block.cgl.c = dims.cInit;
        block.cgl.tauCore = dims.tauCore;
        block.cgl.anchors.shape = {static_cast<std::int64_t>(totalAnchors), static_cast<std::int64_t>(dHyp)};
        block.cgl.anchors.data.assign(totalAnchors * dHyp, 0.0f);
        randomNormalTensorState(block.cgl.anchors, 0.0f, 0.01f, state);
        block.cgl.parent.clear();
        block.cgl.parent.reserve(totalAnchors);
        for (std::size_t p = 0; p < nTop; ++p) {
            block.cgl.parent.push_back(-1);
        }
        for (std::size_t p = 0; p < nTop; ++p) {
            for (std::size_t c = 0; c < nChildren; ++c) {
                block.cgl.parent.push_back(static_cast<int>(p));
            }
        }
        block.cgl.lambdaRaw.shape = {1};
        block.cgl.lambdaRaw.data = {0.0f};
        block.cgl.gamma.shape = {static_cast<std::int64_t>(d)};
        block.cgl.gamma.data.assign(d, 0.0f);
        block.cgl.wc = makeLinearParams(d, dHyp, state, 0.0f);
        block.tig.nTop = dims.n_top_anchors;
        block.tig.topK = 2;
        block.tig.c = dims.cInit;
        block.tig.tauRoute = dims.tauRoute;
        block.tig.beta = dims.routeBeta;
        block.tig.isoDelta = dims.isoDelta;
        block.tig.wr = makeLinearParams(nTop, d, state, 0.0f);
        block.tig.anchors.shape = {static_cast<std::int64_t>(nTop), static_cast<std::int64_t>(dHyp)};
        block.tig.anchors.data.assign(nTop * dHyp, 0.0f);
        for (std::size_t e = 0; e < nTop; ++e) {
            for (std::size_t i = 0; i < dHyp; ++i) {
                block.tig.anchors.data[e * dHyp + i] = block.cgl.anchors.data[e * dHyp + i];
            }
        }
        block.moe.shared = makeSwiGLUParams(d, F, state);
        block.moe.useMoe = dims.enableT4;
        if (block.moe.useMoe) {
            block.moe.experts.reserve(nTop);
            for (std::size_t e = 0; e < nTop; ++e) {
                block.moe.experts.push_back(makeSwiGLUParams(d, F, state));
            }
        }
        model.blocks.push_back(std::move(block));
    }
    model.head.enableT5 = dims.enableT5;
    model.head.c = dims.cInit;
    model.head.tauHead = dims.tauHead;
    if (dims.tiedEmbeddings) {
        model.head.euclidean.w = model.embed.table;
        model.head.euclidean.b.shape = {static_cast<std::int64_t>(V)};
        model.head.euclidean.b.data.assign(V, 0.0f);
    } else {
        model.head.euclidean = makeLinearParams(V, d, state, 0.0f);
    }
    model.head.hyperTable.shape = {static_cast<std::int64_t>(V), static_cast<std::int64_t>(dHyp)};
    model.head.hyperTable.data.assign(V * dHyp, 0.0f);
    randomNormalTensorState(model.head.hyperTable, 0.0f, 0.02f, state);
    model.head.wzHead = makeLinearParams(dHyp, d, state, 0.0f);
    model.head.lambdaH.shape = {1};
    model.head.lambdaH.data = {0.0f};
    return model;
}

} // namespace arch

#ifdef ARCH_SELFTEST

namespace {

bool allFinite(const arch::Tensor& t) {
    for (float x : t.data) {
        if (!std::isfinite(static_cast<double>(x))) {
            return false;
        }
    }
    return true;
}

bool allFiniteLinear(const arch::LinearGrad& g) {
    return allFinite(g.dw) && allFinite(g.db);
}

bool allFiniteSwiGLU(const arch::SwiGLUGrad& g) {
    return allFiniteLinear(g.w1) && allFiniteLinear(g.w3) && allFiniteLinear(g.w2);
}

bool allFiniteMoE(const arch::MoEGrad& g) {
    if (!allFiniteSwiGLU(g.shared)) {
        return false;
    }
    for (const auto& e : g.experts) {
        if (!allFiniteSwiGLU(e)) {
            return false;
        }
    }
    return true;
}

bool allFiniteBlock(const arch::BlockGrad& g) {
    return
        allFinite(g.norm1.dweight) &&
        allFinite(g.norm2.dweight) &&
        allFiniteLinear(g.attn.wq) &&
        allFiniteLinear(g.attn.wk) &&
        allFiniteLinear(g.attn.wv) &&
        allFiniteLinear(g.attn.wo) &&
        allFiniteLinear(g.wz) &&
        allFiniteLinear(g.hga.wq) &&
        allFiniteLinear(g.hga.wk) &&
        allFiniteLinear(g.hga.wv) &&
        allFiniteLinear(g.fusion.wg) &&
        allFiniteLinear(g.fusion.wu) &&
        allFinite(g.cgl.danchors) &&
        allFinite(g.cgl.dlambdaRaw) &&
        allFinite(g.cgl.dgamma) &&
        allFiniteLinear(g.cgl.wc) &&
        allFiniteLinear(g.tig.wr) &&
        allFinite(g.tig.danchors) &&
        allFiniteMoE(g.moe);
}

bool allFiniteModel(const arch::ModelGrad& g) {
    if (!allFinite(g.embed.dtable)) {
        return false;
    }
    for (const auto& b : g.blocks) {
        if (!allFiniteBlock(b)) {
            return false;
        }
    }
    if (!allFinite(g.finalNorm.dweight)) {
        return false;
    }
    if (!allFiniteLinear(g.head.euclidean)) {
        return false;
    }
    if (!allFinite(g.head.dhyperTable)) {
        return false;
    }
    if (!allFiniteLinear(g.head.wzHead)) {
        return false;
    }
    if (!allFinite(g.head.dlambdaH)) {
        return false;
    }
    return true;
}

arch::Tensor makeRandomTensor(const std::vector<std::int64_t>& shape, std::uint64_t seed) {
    arch::Tensor t = arch::makeTensor(shape, 0.0f);
    arch::randomNormalTensor(t, 0.0f, 1.0f, seed);
    return t;
}

double dot(const arch::Tensor& a, const arch::Tensor& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        s += static_cast<double>(a.data[i]) * static_cast<double>(b.data[i]);
    }
    return s;
}

double linearLoss(const arch::Tensor& x, const arch::LinearParams& lin, const arch::Tensor& gradY) {
    arch::Tensor y;
    arch::linearForward(x, lin, y);
    return dot(y, gradY);
}

void applyLinearSgd(arch::LinearParams& p, const arch::LinearGrad& g, float lr) {
    for (std::size_t i = 0; i < p.w.data.size(); ++i) {
        p.w.data[i] -= lr * g.dw.data[i];
    }
    for (std::size_t i = 0; i < p.b.data.size(); ++i) {
        p.b.data[i] -= lr * g.db.data[i];
    }
}

} // namespace

int main() {
    int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::cerr << "CHECK failed: " #cond                           \
                      << " at line " << __LINE__ << "\n";                 \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

    // 1. Linear backward gradient check and one descent step.
    {
        arch::Tensor x = makeRandomTensor({2, 3}, 101);
        arch::LinearParams lin;
        lin.w = makeRandomTensor({4, 3}, 102);
        lin.b = makeRandomTensor({4}, 103);
        arch::Tensor gradY = makeRandomTensor({2, 4}, 104);
        arch::Tensor gradX;
        arch::LinearGrad grad;
        arch::linearBackward(x, lin, gradY, gradX, grad);
        const double eps = 1e-3;
        for (int t = 0; t < 2; ++t) {
            for (int k = 0; k < 3; ++k) {
                arch::Tensor xp = x;
                arch::Tensor xm = x;
                xp.data[t * 3 + k] += static_cast<float>(eps);
                xm.data[t * 3 + k] -= static_cast<float>(eps);
                double num = (linearLoss(xp, lin, gradY) - linearLoss(xm, lin, gradY)) / (2.0 * eps);
                CHECK(std::fabs(num - static_cast<double>(gradX.data[t * 3 + k])) < 2e-2);
            }
        }
        for (int o = 0; o < 4; ++o) {
            for (int k = 0; k < 3; ++k) {
                arch::LinearParams wp = lin;
                arch::LinearParams wm = lin;
                wp.w.data[o * 3 + k] += static_cast<float>(eps);
                wm.w.data[o * 3 + k] -= static_cast<float>(eps);
                double num = (linearLoss(x, wp, gradY) - linearLoss(x, wm, gradY)) / (2.0 * eps);
                CHECK(std::fabs(num - static_cast<double>(grad.dw.data[o * 3 + k])) < 2e-2);
            }
        }
        double loss0 = linearLoss(x, lin, gradY);
        applyLinearSgd(lin, grad, 1e-6f);
        double loss1 = linearLoss(x, lin, gradY);
        CHECK(loss1 < loss0);
    }

    // 2. Tiny full model forward/backward finite.
    arch::ArchitectureDims dims;
    dims.vocab = 16;
    dims.d_model = 8;
    dims.n_layers = 2;
    dims.n_heads = 2;
    dims.n_kv_heads = 1;
    dims.d_head = 4;
    dims.ffn_hidden = 16;
    dims.n_hyp_heads = 1;
    dims.d_hyp = 4;
    dims.n_top_anchors = 2;
    dims.n_children = 2;
    dims.tiedEmbeddings = false;
    dims.seed = 777;

    arch::NolModelParams model = arch::createRandomModel(dims);
    std::vector<int> ids = {0, 1, 2, 3, 4};
    std::vector<int> positions = {0, 1, 2, 3, 4};

    arch::Tensor logits;
    arch::nolModelForward(model, ids, positions, logits);
    CHECK(allFinite(logits));

    arch::Tensor gradLogits = makeRandomTensor(logits.shape, 555);
    arch::ModelGrad grad;
    arch::nolModelBackward(model, ids, positions, gradLogits, grad);

    CHECK(allFiniteModel(grad));
    CHECK(grad.embed.dtable.data.size() == model.embed.table.data.size());

    bool embedGradNonZero = false;
    for (float x : grad.embed.dtable.data) {
        if (std::fabs(static_cast<double>(x)) > 1e-8) {
            embedGradNonZero = true;
            break;
        }
    }
    CHECK(embedGradNonZero);

    if (failures == 0) {
        std::cout << "architecture unit tests passed\n";
        return 0;
    }

    std::cerr << failures << " architecture unit test(s) failed\n";
    return 1;
}

#endif // ARCH_SELFTEST
