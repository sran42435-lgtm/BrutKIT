#ifndef PROYEK_ARCHITECTURE_HPP
#define PROYEK_ARCHITECTURE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace arch {

struct Tensor {
    std::vector<std::int64_t> shape;
    std::vector<float> data;
};

Tensor makeTensor(const std::vector<std::int64_t>& shape, float fill = 0.0f);

std::size_t tensorNumel(const std::vector<std::int64_t>& shape);

bool resizeTensor(
    Tensor& t,
    const std::vector<std::int64_t>& shape,
    float fill = 0.0f
);

void fillTensor(Tensor& t, float value);

void randomNormalTensor(
    Tensor& t,
    float mean,
    float stddev,
    std::uint64_t seed
);

void gemm(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha = 1.0f,
    float beta = 0.0f
);

void gemmNaive(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    float alpha = 1.0f,
    float beta = 0.0f
);

void addInPlace(Tensor& dst, const Tensor& src);

void mulInPlace(Tensor& dst, const Tensor& src);

void sigmoidInPlace(Tensor& t);

void siluInPlace(Tensor& t);

void softmaxInPlace(float* row, int n);

float linner(const float* x, const float* y, int n);

void exp0(const float* v, float* x, int n, float c);

void log0(const float* x, float* v, int n, float c);

float dist2Lorentz(const float* x, const float* y, int n, float c);

float distLorentz(const float* x, const float* y, int n, float c);

void lorentzCentroidFromWeightedSum(
    const float* z,
    float* mu,
    int n,
    float c
);

void lorentzCentroid(
    const float* const* points,
    const float* weights,
    int count,
    int n,
    float c,
    float* mu
);

float poincareRadius(float rho, float c);

void lorentzScoreBackward(
    const float* q,
    const float* k,
    float* dq,
    float* dk,
    int n,
    float c,
    float invTau,
    float gradScore
);

void lorentzCentroidBackward(
    const float* z,
    const float* gradMu,
    float* gradZ,
    int n,
    float c
);

void exp0Backward(
    const float* v,
    const float* gradX,
    float* gradV,
    int n,
    float c
);

void log0Backward(
    const float* x,
    const float* gradV,
    float* gradX,
    int n,
    float c
);

struct LinearParams {
    Tensor w; // [out, in]
    Tensor b; // [out]
};

struct EmbeddingParams {
    Tensor table; // [vocab, d_model]
};

struct RMSNormParams {
    Tensor weight; // [d_model]
    float eps = 1e-6f;
};

struct GQAParams {
    LinearParams wq;
    LinearParams wk;
    LinearParams wv;
    LinearParams wo;

    int n_heads = 0;
    int n_kv_heads = 0;
    int d_head = 0;

    float scale = 1.0f;
    float ropeBase = 10000.0f;
};

struct SwiGLUParams {
    LinearParams w1;
    LinearParams w3;
    LinearParams w2;
};

struct HGAParams {
    LinearParams wq;
    LinearParams wk;
    LinearParams wv;

    int n_hyp_heads = 0;
    int d_hyp = 0;

    float c = 1.0f;
    float invTau = 1.0f;
};

struct FusionParams {
    LinearParams wg;
    LinearParams wu;
};

struct CGLParams {
    Tensor anchors; // [num_anchors, d_hyp], tangent vectors
    std::vector<int> parent;

    Tensor lambdaRaw; // [1]
    Tensor gamma;     // [d_model]

    LinearParams wc; // [d_model, d_hyp]

    float c = 1.0f;
    float tauCore = 1.0f;
};

struct TigParams {
    LinearParams wr; // [n_top_anchors, d_model]
    Tensor anchors;  // [n_top_anchors, d_hyp]

    int nTop = 0;
    int topK = 2;

    float c = 1.0f;
    float tauRoute = 1.0f;
    float beta = 1.0f;
    float isoDelta = 2.0f;
};

struct TigOutput {
    Tensor probs; // [T, nTop]

    std::vector<int> selected; // [T * topK], -1 means unused

    Tensor weights; // [T * topK]

    int topK = 0;
};

struct MoEParams {
    SwiGLUParams shared;
    std::vector<SwiGLUParams> experts;

    bool useMoe = false;
};

struct HeadParams {
    LinearParams euclidean; // [vocab, d_model]

    Tensor hyperTable; // [vocab, d_hyp], tangent vectors
    LinearParams wzHead; // [d_hyp, d_model]

    Tensor lambdaH; // [1]

    bool enableT5 = false;

    float c = 1.0f;
    float tauHead = 1.0f;
};

struct BlockParams {
    RMSNormParams norm1;
    RMSNormParams norm2;

    GQAParams attn;

    LinearParams wz; // [d_hyp, d_model]

    HGAParams hga;
    FusionParams fusion;

    CGLParams cgl;
    TigParams tig;

    MoEParams moe;

    bool enableT1 = true;
    bool enableT2 = true;
    bool enableT3 = true;
    bool enableT4 = true;

    int d_model = 0;
    int d_hyp = 0;
    int n_hyp_heads = 0;

    float ropeBase = 10000.0f;
};

struct NolModelParams {
    EmbeddingParams embed;

    std::vector<BlockParams> blocks;

    RMSNormParams finalNorm;
    HeadParams head;

    int vocab = 0;
    int d_model = 0;
};

struct ArchitectureDims {
    int vocab = 32;
    int d_model = 64;
    int n_layers = 2;

    int n_heads = 4;
    int n_kv_heads = 2;
    int d_head = 16;

    int ffn_hidden = 128;

    int n_hyp_heads = 2;
    int d_hyp = 16;

    int n_top_anchors = 4;
    int n_children = 4;

    bool enableT1 = true;
    bool enableT2 = true;
    bool enableT3 = true;
    bool enableT4 = true;
    bool enableT5 = true;

    bool useMoe = true;
    bool tiedEmbeddings = true;

    float cInit = 1.0f;
    float tauAtt = 1.0f;
    float tauCore = 1.0f;
    float tauRoute = 1.0f;
    float tauHead = 1.0f;

    float isoDelta = 2.0f;
    float routeBeta = 1.0f;

    float ropeBase = 10000.0f;

    std::uint64_t seed = 1234;
};

struct LinearGrad {
    Tensor dw;
    Tensor db;
};

struct EmbeddingGrad {
    Tensor dtable;
};

struct RMSNormGrad {
    Tensor dweight;
};

struct GQAGrad {
    LinearGrad wq;
    LinearGrad wk;
    LinearGrad wv;
    LinearGrad wo;
};

struct SwiGLUGrad {
    LinearGrad w1;
    LinearGrad w3;
    LinearGrad w2;
};

struct HGAGrad {
    LinearGrad wq;
    LinearGrad wk;
    LinearGrad wv;
};

struct FusionGrad {
    LinearGrad wg;
    LinearGrad wu;
};

struct CGLGrad {
    Tensor danchors;
    Tensor dlambdaRaw;
    Tensor dgamma;
    LinearGrad wc;
};

struct TigGrad {
    LinearGrad wr;
    Tensor danchors;
};

struct MoEGrad {
    SwiGLUGrad shared;
    std::vector<SwiGLUGrad> experts;
};

struct HeadGrad {
    LinearGrad euclidean;
    Tensor dhyperTable;
    LinearGrad wzHead;
    Tensor dlambdaH;
};

struct BlockGrad {
    RMSNormGrad norm1;
    RMSNormGrad norm2;

    GQAGrad attn;

    LinearGrad wz;

    HGAGrad hga;
    FusionGrad fusion;

    CGLGrad cgl;
    TigGrad tig;

    MoEGrad moe;
};

struct ModelGrad {
    EmbeddingGrad embed;

    std::vector<BlockGrad> blocks;

    RMSNormGrad finalNorm;
    HeadGrad head;
};

void linearForward(
    const Tensor& x,
    const LinearParams& lin,
    Tensor& y
);

void linearBackward(
    const Tensor& x,
    const LinearParams& lin,
    const Tensor& gradY,
    Tensor& gradX,
    LinearGrad& grad
);

void embeddingForward(
    const EmbeddingParams& emb,
    const std::vector<int>& ids,
    Tensor& out
);

void embeddingBackward(
    const std::vector<int>& ids,
    const Tensor& gradOut,
    EmbeddingGrad& grad
);

void rmsNormForward(
    const Tensor& x,
    const RMSNormParams& norm,
    Tensor& y
);

void rmsNormBackward(
    const Tensor& x,
    const RMSNormParams& norm,
    const Tensor& gradY,
    Tensor& gradX,
    RMSNormGrad& grad
);

void ropeForward(
    Tensor& q,
    Tensor& k,
    int n_heads,
    int n_kv_heads,
    int d_head,
    const std::vector<int>& positions,
    float base
);

void ropeBackward(
    Tensor& gradQ,
    Tensor& gradK,
    int n_heads,
    int n_kv_heads,
    int d_head,
    const std::vector<int>& positions,
    float base
);

void gqaForward(
    const Tensor& x,
    const GQAParams& attn,
    const std::vector<int>& positions,
    Tensor& out
);

void gqaBackward(
    const Tensor& x,
    const GQAParams& attn,
    const std::vector<int>& positions,
    const Tensor& gradOut,
    Tensor& gradX,
    GQAGrad& grad
);

void swigluForward(
    const Tensor& x,
    const SwiGLUParams& ffn,
    Tensor& out
);

void swigluBackward(
    const Tensor& x,
    const SwiGLUParams& ffn,
    const Tensor& gradOut,
    Tensor& gradX,
    SwiGLUGrad& grad
);

void hgaForward(
    const Tensor& x,
    const HGAParams& hga,
    Tensor& u
);

void hgaBackward(
    const Tensor& x,
    const HGAParams& hga,
    const Tensor& gradU,
    Tensor& gradX,
    HGAGrad& grad
);

void gatedFusionForward(
    const Tensor& x,
    const Tensor& a,
    const Tensor& u,
    const FusionParams& fusion,
    Tensor& m
);

void gatedFusionBackward(
    const Tensor& x,
    const Tensor& a,
    const Tensor& u,
    const FusionParams& fusion,
    const Tensor& gradM,
    Tensor& gradX,
    Tensor& gradU,
    FusionGrad& grad
);

void cglDeltaForward(
    const Tensor& z,
    const CGLParams& cgl,
    Tensor& delta
);

void cglGeometryBackward(
    const Tensor& z,
    const Tensor& gradDelta,
    Tensor& gradZ
);

void tigForward(
    const Tensor& x,
    const Tensor& z,
    const TigParams& tig,
    TigOutput& out
);

void tigBackward(
    const Tensor& x,
    const Tensor& z,
    const TigParams& tig,
    const TigOutput& route,
    const Tensor& gradWeights,
    Tensor& gradX,
    Tensor& gradZ,
    TigGrad& grad
);

void moeForward(
    const Tensor& x,
    const MoEParams& moe,
    const TigOutput& route,
    Tensor& out
);

void moeBackward(
    const Tensor& x,
    const Tensor& z,
    const TigParams& tig,
    const MoEParams& moe,
    const TigOutput& route,
    const Tensor& gradOut,
    Tensor& gradX,
    Tensor& gradZ,
    TigGrad& tigGrad,
    MoEGrad& grad
);

void headForward(
    const Tensor& xN,
    const Tensor& zN,
    const HeadParams& head,
    Tensor& logits
);

void headBackward(
    const Tensor& xN,
    const HeadParams& head,
    const Tensor& gradLogits,
    Tensor& gradXN,
    HeadGrad& grad
);

void nolBlockForward(
    const Tensor& hIn,
    const BlockParams& block,
    const std::vector<int>& positions,
    Tensor& hOut
);

void nolBlockBackward(
    const Tensor& hIn,
    const BlockParams& block,
    const std::vector<int>& positions,
    const Tensor& gradHOut,
    Tensor& gradHIn,
    BlockGrad& grad
);

void nolModelForward(
    const NolModelParams& model,
    const std::vector<int>& ids,
    const std::vector<int>& positions,
    Tensor& logits
);

void nolModelBackward(
    const NolModelParams& model,
    const std::vector<int>& ids,
    const std::vector<int>& positions,
    const Tensor& gradLogits,
    ModelGrad& grad
);

NolModelParams createRandomModel(const ArchitectureDims& dimsIn);

} // namespace arch

#endif // PROYEK_ARCHITECTURE_HPP
