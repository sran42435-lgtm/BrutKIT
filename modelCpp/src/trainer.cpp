#include "trainer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef TRAINER_SELFTEST
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

bool isBlankLine(const std::string& s) {
    for (char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    return true;
}

int normalizeToken(
    int id,
    int vocabSize,
    int fallback
) {
    if (id < 0) {
        if (fallback >= 0) {
            return fallback;
        }

        return 0;
    }

    if (vocabSize <= 0) {
        return id;
    }

    if (id < vocabSize) {
        return id;
    }

    if (fallback >= 0 && fallback < vocabSize) {
        return fallback;
    }

    return 0;
}

std::size_t storageNumel(const StorageTensor& t) {
    std::size_t n = 1;

    for (std::uint64_t d : t.shape) {
        n *= static_cast<std::size_t>(d);
    }

    return n;
}

StorageTensor makeStorageTensor(
    const std::string& name,
    const arch::Tensor& t
) {
    StorageTensor s;

    s.name = name;
    s.dtype = "F32";
    s.shape.reserve(t.shape.size());

    for (std::int64_t d : t.shape) {
        s.shape.push_back(
            static_cast<std::uint64_t>(d > 0 ? d : 0)
        );
    }

    s.data = t.data;
    return s;
}

void sanitizeTensor(arch::Tensor* t) {
    if (t == nullptr) {
        return;
    }

    for (float& x : t->data) {
        if (!std::isfinite(static_cast<double>(x))) {
            x = 0.0f;
        }
    }
}

void ensureTensorLike(arch::Tensor& dst, const arch::Tensor& ref) {
    dst.shape = ref.shape;
    dst.data.assign(ref.data.size(), 0.0f);
}

int sampleToken(
    const float* logits,
    int vocab,
    float temperature,
    int topK,
    float topP,
    std::uint64_t seed
) {
    if (vocab <= 0 || logits == nullptr) {
        return -1;
    }

    std::vector<int> order(static_cast<std::size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);

    if (temperature <= 0.0f) {
        int best = 0;
        double bestv = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < vocab; ++i) {
            double x = static_cast<double>(logits[i]);

            if (std::isfinite(x) && x > bestv) {
                bestv = x;
                best = i;
            }
        }

        return best;
    }

    double maxv = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < vocab; ++i) {
        double x = static_cast<double>(logits[i]);

        if (std::isfinite(x) && x > maxv) {
            maxv = x;
        }
    }

    if (!std::isfinite(maxv)) {
        maxv = 0.0;
    }

    std::vector<float> probs(static_cast<std::size_t>(vocab), 0.0f);
    double sum = 0.0;

    for (int i = 0; i < vocab; ++i) {
        double x = static_cast<double>(logits[i]);

        if (!std::isfinite(x)) {
            x = maxv;
        }

        double e = std::exp((x - maxv) / static_cast<double>(temperature));

        if (!std::isfinite(e) || e < 0.0) {
            e = 0.0;
        }

        probs[static_cast<std::size_t>(i)] = static_cast<float>(e);
        sum += e;
    }

    if (sum <= 0.0) {
        std::sort(
            order.begin(),
            order.end(),
            [&](int a, int b) {
                double xa = static_cast<double>(logits[a]);
                double xb = static_cast<double>(logits[b]);

                if (!std::isfinite(xa)) {
                    xa = -std::numeric_limits<double>::infinity();
                }

                if (!std::isfinite(xb)) {
                    xb = -std::numeric_limits<double>::infinity();
                }

                return xa > xb;
            }
        );

        return order[0];
    }

    for (float& p : probs) {
        p = static_cast<float>(static_cast<double>(p) / sum);
    }

    std::sort(
        order.begin(),
        order.end(),
        [&](int a, int b) {
            return probs[static_cast<std::size_t>(a)] >
                   probs[static_cast<std::size_t>(b)];
        }
    );

    int keep = vocab;

    if (topK > 0 && topK < keep) {
        keep = topK;
    }

    if (topP > 0.0f && topP < 1.0f) {
        double cum = 0.0;
        int pkeep = 1;

        for (int i = 0; i < vocab; ++i) {
            cum += static_cast<double>(
                probs[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]
            );

            pkeep = i + 1;

            if (cum >= static_cast<double>(topP)) {
                break;
            }
        }

        if (pkeep < keep) {
            keep = pkeep;
        }
    }

    if (keep < 1) {
        keep = 1;
    }

    if (keep > vocab) {
        keep = vocab;
    }

    double selectedSum = 0.0;

    for (int i = 0; i < keep; ++i) {
        selectedSum += static_cast<double>(
            probs[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]
        );
    }

    if (selectedSum <= 0.0) {
        return order[0];
    }

    std::uint64_t state = seed == 0 ? 123456789ULL : seed;
    double r = uniform01(state) * selectedSum;
    double acc = 0.0;

    for (int i = 0; i < keep; ++i) {
        acc += static_cast<double>(
            probs[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])]
        );

        if (r <= acc) {
            return order[static_cast<std::size_t>(i)];
        }
    }

    return order[static_cast<std::size_t>(keep - 1)];
}

} // namespace

namespace tr {

std::vector<NamedParam> collectParams(arch::NolModelParams& model) {
    std::vector<NamedParam> out;

    auto add = [&](
        const std::string& name,
        arch::Tensor* t,
        bool decay
    ) {
        if (t != nullptr && !t->data.empty()) {
            out.push_back(NamedParam{name, t, decay});
        }
    };

    add("embed.table", &model.embed.table, true);

    for (std::size_t li = 0; li < model.blocks.size(); ++li) {
        arch::BlockParams& b = model.blocks[li];
        std::string bp = "blocks." + std::to_string(li) + ".";

        add(bp + "norm1.weight", &b.norm1.weight, false);

        add(bp + "attn.wq.w", &b.attn.wq.w, true);
        add(bp + "attn.wq.b", &b.attn.wq.b, false);

        add(bp + "attn.wk.w", &b.attn.wk.w, true);
        add(bp + "attn.wk.b", &b.attn.wk.b, false);

        add(bp + "attn.wv.w", &b.attn.wv.w, true);
        add(bp + "attn.wv.b", &b.attn.wv.b, false);

        add(bp + "attn.wo.w", &b.attn.wo.w, true);
        add(bp + "attn.wo.b", &b.attn.wo.b, false);

        add(bp + "wz.w", &b.wz.w, true);
        add(bp + "wz.b", &b.wz.b, false);

        add(bp + "hga.wq.w", &b.hga.wq.w, true);
        add(bp + "hga.wq.b", &b.hga.wq.b, false);

        add(bp + "hga.wk.w", &b.hga.wk.w, true);
        add(bp + "hga.wk.b", &b.hga.wk.b, false);

        add(bp + "hga.wv.w", &b.hga.wv.w, true);
        add(bp + "hga.wv.b", &b.hga.wv.b, false);

        add(bp + "fusion.wg.w", &b.fusion.wg.w, true);
        add(bp + "fusion.wg.b", &b.fusion.wg.b, false);

        add(bp + "fusion.wu.w", &b.fusion.wu.w, true);
        add(bp + "fusion.wu.b", &b.fusion.wu.b, false);

        add(bp + "cgl.anchors", &b.cgl.anchors, true);
        add(bp + "cgl.lambdaRaw", &b.cgl.lambdaRaw, false);
        add(bp + "cgl.gamma", &b.cgl.gamma, false);

        add(bp + "cgl.wc.w", &b.cgl.wc.w, true);
        add(bp + "cgl.wc.b", &b.cgl.wc.b, false);

        add(bp + "tig.wr.w", &b.tig.wr.w, true);
        add(bp + "tig.wr.b", &b.tig.wr.b, false);
        add(bp + "tig.anchors", &b.tig.anchors, true);

        add(bp + "moe.shared.w1.w", &b.moe.shared.w1.w, true);
        add(bp + "moe.shared.w1.b", &b.moe.shared.w1.b, false);

        add(bp + "moe.shared.w3.w", &b.moe.shared.w3.w, true);
        add(bp + "moe.shared.w3.b", &b.moe.shared.w3.b, false);

        add(bp + "moe.shared.w2.w", &b.moe.shared.w2.w, true);
        add(bp + "moe.shared.w2.b", &b.moe.shared.w2.b, false);

        for (std::size_t e = 0; e < b.moe.experts.size(); ++e) {
            std::string ep = bp + "moe.experts." + std::to_string(e) + ".";

            add(ep + "w1.w", &b.moe.experts[e].w1.w, true);
            add(ep + "w1.b", &b.moe.experts[e].w1.b, false);

            add(ep + "w3.w", &b.moe.experts[e].w3.w, true);
            add(ep + "w3.b", &b.moe.experts[e].w3.b, false);

            add(ep + "w2.w", &b.moe.experts[e].w2.w, true);
            add(ep + "w2.b", &b.moe.experts[e].w2.b, false);
        }

        add(bp + "norm2.weight", &b.norm2.weight, false);
    }

    add("final_norm.weight", &model.finalNorm.weight, false);

    add("head.euclidean.w", &model.head.euclidean.w, true);
    add("head.euclidean.b", &model.head.euclidean.b, false);

    add("head.hyperTable", &model.head.hyperTable, true);

    add("head.wzHead.w", &model.head.wzHead.w, true);
    add("head.wzHead.b", &model.head.wzHead.b, false);

    add("head.lambdaH", &model.head.lambdaH, false);

    return out;
}

std::vector<arch::Tensor*> collectGrads(arch::ModelGrad& grad) {
    std::vector<arch::Tensor*> out;

    auto add = [&](arch::Tensor* t) {
        if (t != nullptr && !t->data.empty()) {
            out.push_back(t);
        }
    };

    add(&grad.embed.dtable);

    for (std::size_t li = 0; li < grad.blocks.size(); ++li) {
        arch::BlockGrad& g = grad.blocks[li];

        add(&g.norm1.dweight);

        add(&g.attn.wq.dw);
        add(&g.attn.wq.db);

        add(&g.attn.wk.dw);
        add(&g.attn.wk.db);

        add(&g.attn.wv.dw);
        add(&g.attn.wv.db);

        add(&g.attn.wo.dw);
        add(&g.attn.wo.db);

        add(&g.wz.dw);
        add(&g.wz.db);

        add(&g.hga.wq.dw);
        add(&g.hga.wq.db);

        add(&g.hga.wk.dw);
        add(&g.hga.wk.db);

        add(&g.hga.wv.dw);
        add(&g.hga.wv.db);

        add(&g.fusion.wg.dw);
        add(&g.fusion.wg.db);

        add(&g.fusion.wu.dw);
        add(&g.fusion.wu.db);

        add(&g.cgl.danchors);
        add(&g.cgl.dlambdaRaw);
        add(&g.cgl.dgamma);

        add(&g.cgl.wc.dw);
        add(&g.cgl.wc.db);

        add(&g.tig.wr.dw);
        add(&g.tig.wr.db);
        add(&g.tig.danchors);

        add(&g.moe.shared.w1.dw);
        add(&g.moe.shared.w1.db);

        add(&g.moe.shared.w3.dw);
        add(&g.moe.shared.w3.db);

        add(&g.moe.shared.w2.dw);
        add(&g.moe.shared.w2.db);

        for (std::size_t e = 0; e < g.moe.experts.size(); ++e) {
            add(&g.moe.experts[e].w1.dw);
            add(&g.moe.experts[e].w1.db);

            add(&g.moe.experts[e].w3.dw);
            add(&g.moe.experts[e].w3.db);

            add(&g.moe.experts[e].w2.dw);
            add(&g.moe.experts[e].w2.db);
        }

        add(&g.norm2.dweight);
    }

    add(&grad.finalNorm.dweight);

    add(&grad.head.euclidean.dw);
    add(&grad.head.euclidean.db);

    add(&grad.head.dhyperTable);

    add(&grad.head.wzHead.dw);
    add(&grad.head.wzHead.db);

    add(&grad.head.dlambdaH);

    return out;
}

float currentLearningRate(const Config& cfg, std::int64_t step) {
    double base = static_cast<double>(cfg.learningRate);
    double minLr = static_cast<double>(cfg.minLr);

    if (minLr > base) {
        minLr = base;
    }

    if (step <= 0) {
        return static_cast<float>(base);
    }

    if (cfg.warmupSteps > 0 && step <= cfg.warmupSteps) {
        double w = static_cast<double>(cfg.warmupSteps);
        return static_cast<float>(base * static_cast<double>(step) / w);
    }

    if (
        cfg.scheduleType == "cosine" &&
        cfg.totalSteps > cfg.warmupSteps
    ) {
        double start = static_cast<double>(cfg.warmupSteps);
        double end = static_cast<double>(cfg.totalSteps);
        double cur = static_cast<double>(step);

        double progress = (cur - start) / (end - start);

        if (progress < 0.0) {
            progress = 0.0;
        }

        if (progress > 1.0) {
            progress = 1.0;
        }

        double lr = minLr + 0.5 * (base - minLr) *
                    (1.0 + std::cos(kPi * progress));

        return static_cast<float>(lr);
    }

    return static_cast<float>(base);
}

double crossEntropy(
    const arch::Tensor& logits,
    const std::vector<int>& targets,
    arch::Tensor& gradLogits
) {
    gradLogits.shape = logits.shape;
    gradLogits.data.assign(logits.data.size(), 0.0f);

    if (logits.shape.size() != 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (targets.size() != static_cast<std::size_t>(logits.shape[0])) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (logits.shape[1] <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::size_t T = static_cast<std::size_t>(logits.shape[0]);
    std::size_t V = static_cast<std::size_t>(logits.shape[1]);

    if (logits.data.size() != T * V) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double loss = 0.0;
    std::size_t valid = 0;

    for (std::size_t t = 0; t < T; ++t) {
        int target = targets[t];

        if (target < 0 || static_cast<std::size_t>(target) >= V) {
            continue;
        }

        const float* row = logits.data.data() + t * V;
        float* grow = gradLogits.data.data() + t * V;

        double maxv = -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < V; ++i) {
            double x = static_cast<double>(row[i]);

            if (std::isfinite(x) && x > maxv) {
                maxv = x;
            }
        }

        if (!std::isfinite(maxv)) {
            maxv = 0.0;
        }

        double sum = 0.0;

        for (std::size_t i = 0; i < V; ++i) {
            double x = static_cast<double>(row[i]);

            if (!std::isfinite(x)) {
                x = maxv;
            }

            double e = std::exp(x - maxv);

            if (!std::isfinite(e) || e < 0.0) {
                e = 0.0;
            }

            grow[i] = static_cast<float>(e);
            sum += e;
        }

        if (sum <= 0.0) {
            for (std::size_t i = 0; i < V; ++i) {
                grow[i] = 0.0f;
            }

            continue;
        }

        double logSum = std::log(sum);
        double targetLogit = static_cast<double>(row[static_cast<std::size_t>(target)]);

        if (!std::isfinite(targetLogit)) {
            targetLogit = maxv;
        }

        loss += -targetLogit + maxv + logSum;

        double invSum = 1.0 / sum;

        for (std::size_t i = 0; i < V; ++i) {
            grow[i] = static_cast<float>(static_cast<double>(grow[i]) * invSum);
        }

        grow[static_cast<std::size_t>(target)] -= 1.0f;
        ++valid;
    }

    if (valid == 0) {
        return 0.0;
    }

    double invValid = 1.0 / static_cast<double>(valid);

    for (float& x : gradLogits.data) {
        x = static_cast<float>(static_cast<double>(x) * invValid);
    }

    return loss * invValid;
}

void AdamW::reset(std::size_t n) {
    states_.assign(n, State());
    step_ = 0;
}

double AdamW::step(
    std::vector<NamedParam>& params,
    const std::vector<arch::Tensor*>& grads,
    const Config& cfg,
    std::int64_t nextStep
) {
    if (states_.size() != params.size()) {
        reset(params.size());
    }

    if (grads.size() != params.size()) {
        return 0.0;
    }

    step_ = nextStep;

    for (std::size_t i = 0; i < params.size(); ++i) {
        sanitizeTensor(grads[i]);
    }

    double gradNorm = 0.0;

    for (std::size_t i = 0; i < params.size(); ++i) {
        arch::Tensor* g = grads[i];

        if (g == nullptr) {
            continue;
        }

        for (float x : g->data) {
            double d = static_cast<double>(x);
            gradNorm += d * d;
        }
    }

    gradNorm = std::sqrt(gradNorm);

    if (
        cfg.maxGradNorm > 0.0f &&
        gradNorm > static_cast<double>(cfg.maxGradNorm) &&
        gradNorm > 1e-12
    ) {
        double scale = static_cast<double>(cfg.maxGradNorm) / gradNorm;

        for (std::size_t i = 0; i < params.size(); ++i) {
            arch::Tensor* g = grads[i];

            if (g == nullptr) {
                continue;
            }

            for (float& x : g->data) {
                x = static_cast<float>(static_cast<double>(x) * scale);
            }
        }
    }

    double lr = static_cast<double>(currentLearningRate(cfg, nextStep));

    double b1 = static_cast<double>(cfg.beta1);
    double b2 = static_cast<double>(cfg.beta2);
    double eps = static_cast<double>(cfg.eps);
    double wd = static_cast<double>(cfg.weightDecay);

    double bc1 = 1.0 - std::pow(b1, static_cast<double>(nextStep));
    double bc2 = 1.0 - std::pow(b2, static_cast<double>(nextStep));

    if (bc1 <= 0.0) {
        bc1 = 1e-12;
    }

    if (bc2 <= 0.0) {
        bc2 = 1e-12;
    }

    for (std::size_t i = 0; i < params.size(); ++i) {
        arch::Tensor* p = params[i].tensor;
        arch::Tensor* g = grads[i];

        if (p == nullptr || g == nullptr) {
            continue;
        }

        if (p->data.size() != g->data.size()) {
            continue;
        }

        State& s = states_[i];

        if (s.m.data.size() != p->data.size()) {
            ensureTensorLike(s.m, *p);
        }

        if (s.v.data.size() != p->data.size()) {
            ensureTensorLike(s.v, *p);
        }

        for (std::size_t j = 0; j < p->data.size(); ++j) {
            double gp = static_cast<double>(g->data[j]);
            double mp = static_cast<double>(s.m.data[j]);
            double vp = static_cast<double>(s.v.data[j]);

            mp = b1 * mp + (1.0 - b1) * gp;
            vp = b2 * vp + (1.0 - b2) * gp * gp;

            s.m.data[j] = static_cast<float>(mp);
            s.v.data[j] = static_cast<float>(vp);

            double mhat = mp / bc1;
            double vhat = vp / bc2;

            double update = lr * mhat / (std::sqrt(vhat) + eps);

            double pv = static_cast<double>(p->data[j]);

            if (params[i].decay && wd != 0.0) {
                pv -= lr * wd * pv;
            }

            pv -= update;

            if (!std::isfinite(pv)) {
                pv = 0.0;
            }

            p->data[j] = static_cast<float>(pv);
        }
    }

    return static_cast<float>(lr);
}

void AdamW::exportStates(
    const std::vector<NamedParam>& params,
    std::map<std::string, StorageTensor>& out
) const {
    out.clear();

    for (std::size_t i = 0; i < params.size(); ++i) {
        const NamedParam& np = params[i];

        if (np.tensor == nullptr) {
            continue;
        }

        arch::Tensor mz;
        arch::Tensor vz;

        ensureTensorLike(mz, *np.tensor);
        ensureTensorLike(vz, *np.tensor);

        if (i < states_.size()) {
            if (states_[i].m.data.size() == mz.data.size()) {
                mz.data = states_[i].m.data;
            }

            if (states_[i].v.data.size() == vz.data.size()) {
                vz.data = states_[i].v.data;
            }
        }

        out["optimizer." + np.name + ".m"] =
            makeStorageTensor("optimizer." + np.name + ".m", mz);

        out["optimizer." + np.name + ".v"] =
            makeStorageTensor("optimizer." + np.name + ".v", vz);
    }
}

bool AdamW::importStates(
    const std::vector<NamedParam>& params,
    const std::map<std::string, StorageTensor>& in,
    std::string& err
) {
    states_.resize(params.size());

    for (std::size_t i = 0; i < params.size(); ++i) {
        const NamedParam& np = params[i];

        if (np.tensor == nullptr) {
            continue;
        }

        ensureTensorLike(states_[i].m, *np.tensor);
        ensureTensorLike(states_[i].v, *np.tensor);

        std::string mk = "optimizer." + np.name + ".m";
        std::string vk = "optimizer." + np.name + ".v";

        auto mit = in.find(mk);

        if (mit != in.end()) {
            if (storageNumel(mit->second) != states_[i].m.data.size()) {
                err = "optimizer state size mismatch: " + mk;
                return false;
            }

            states_[i].m.data = mit->second.data;
        }

        auto vit = in.find(vk);

        if (vit != in.end()) {
            if (storageNumel(vit->second) != states_[i].v.data.size()) {
                err = "optimizer state size mismatch: " + vk;
                return false;
            }

            states_[i].v.data = vit->second.data;
        }
    }

    err.clear();
    return true;
}

bool Dataset::loadFromFile(
    const std::string& path,
    const Tokenizer& tok,
    const Config& cfg,
    std::string& err
) {
    std::string text;

    if (!readFileString(path, text, err)) {
        return false;
    }

    return loadFromText(text, tok, cfg, err);
}

bool Dataset::loadFromText(
    const std::string& text,
    const Tokenizer& tok,
    const Config& cfg,
    std::string& err
) {
    chunks_.clear();
    cursor_ = 0;

    if (cfg.seqLen <= 0) {
        err = "seqLen must be positive";
        return false;
    }

    std::istringstream iss(text);
    std::string line;

    std::size_t seqLen = static_cast<std::size_t>(cfg.seqLen);

    int pad = tok.padId();

    if (pad < 0) {
        pad = 0;
    }

    if (cfg.vocabSize > 0 && pad >= cfg.vocabSize) {
        pad = 0;
    }

    auto fit = [&](int id) -> int {
        return normalizeToken(id, cfg.vocabSize, pad);
    };

    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (isBlankLine(line)) {
            continue;
        }

        std::vector<int> raw = tok.encode(line);

        if (raw.empty()) {
            continue;
        }

        std::vector<int> doc;
        doc.reserve(raw.size() + 2);

        if (cfg.includeBos) {
            doc.push_back(fit(tok.bosId()));
        }

        for (int id : raw) {
            doc.push_back(fit(id));
        }

        if (cfg.includeEos) {
            doc.push_back(fit(tok.eosId()));
        }

        if (doc.empty()) {
            continue;
        }

        if (doc.size() < seqLen) {
            doc.resize(seqLen, pad);
        }

        for (std::size_t pos = 0; pos < doc.size();) {
            std::size_t take = std::min(seqLen, doc.size() - pos);

            std::vector<int> chunk(
                doc.begin() + static_cast<std::ptrdiff_t>(pos),
                doc.begin() + static_cast<std::ptrdiff_t>(pos + take)
            );

            if (chunk.size() < seqLen) {
                chunk.resize(seqLen, pad);
            }

            chunks_.push_back(std::move(chunk));
            pos += seqLen;
        }
    }

    if (chunks_.empty()) {
        err = "dataset produced no sequences";
        return false;
    }

    err.clear();
    return true;
}

bool Dataset::nextSequence(std::vector<int>& ids) {
    if (cursor_ >= chunks_.size()) {
        return false;
    }

    ids = chunks_[cursor_++];
    return true;
}

Trainer::Trainer(
    arch::NolModelParams modelIn,
    Tokenizer tokenizerIn,
    Config cfgIn
)
    : model_(std::move(modelIn)),
      tokenizer_(std::move(tokenizerIn)),
      cfg_(cfgIn) {
    cfg_.vocabSize = model_.vocab;
    params_ = collectParams(model_);
    opt_.reset(params_.size());
}

std::map<std::string, StorageTensor> Trainer::exportWeights() const {
    std::map<std::string, StorageTensor> out;

    for (const NamedParam& np : params_) {
        if (np.tensor == nullptr) {
            continue;
        }

        out[np.name] = makeStorageTensor(np.name, *np.tensor);
    }

    return out;
}

bool Trainer::importWeights(
    const std::map<std::string, StorageTensor>& in,
    std::string& err
) {
    for (const NamedParam& np : params_) {
        if (np.tensor == nullptr) {
            continue;
        }

        auto it = in.find(np.name);

        if (it == in.end()) {
            err = "missing checkpoint tensor: " + np.name;
            return false;
        }

        if (storageNumel(it->second) != np.tensor->data.size()) {
            err = "checkpoint tensor size mismatch: " + np.name;
            return false;
        }

        np.tensor->data = it->second.data;
    }

    err.clear();
    return true;
}

double Trainer::trainSequence(
    const std::vector<int>& ids,
    std::string& err
) {
    err.clear();

    if (ids.size() < 2) {
        err = "sequence too short";
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<int> safeIds = ids;

    int pad = tokenizer_.padId();

    if (pad < 0) {
        pad = 0;
    }

    if (model_.vocab > 0 && pad >= model_.vocab) {
        pad = 0;
    }

    for (int& id : safeIds) {
        id = normalizeToken(id, model_.vocab, pad);
    }

    std::vector<int> targets(safeIds.size(), -1);

    for (std::size_t i = 0; i + 1 < safeIds.size(); ++i) {
        if (safeIds[i + 1] != pad) {
            targets[i] = safeIds[i + 1];
        }
    }

    std::vector<int> positions(safeIds.size());
    std::iota(positions.begin(), positions.end(), 0);

    arch::Tensor logits;
    arch::nolModelForward(model_, safeIds, positions, logits);

    if (
        logits.shape.size() != 2 ||
        static_cast<std::size_t>(logits.shape[0]) != safeIds.size()
    ) {
        err = "invalid logits shape";
        return std::numeric_limits<double>::quiet_NaN();
    }

    arch::Tensor gradLogits;
    double loss = crossEntropy(logits, targets, gradLogits);

    if (!std::isfinite(loss)) {
        err = "non-finite loss";
        return loss;
    }

    arch::ModelGrad mg;
    arch::nolModelBackward(model_, safeIds, positions, gradLogits, mg);

    std::vector<arch::Tensor*> grads = collectGrads(mg);

    if (grads.size() != params_.size()) {
        err = "gradient parameter mismatch";
        return std::numeric_limits<double>::quiet_NaN();
    }

    opt_.step(params_, grads, cfg_, globalStep_ + 1);

    ++globalStep_;
    lastLoss_ = loss;

    return loss;
}

bool Trainer::trainFile(
    const std::string& path,
    const std::string& modelDir,
    std::string& err
) {
    Dataset ds;

    if (!ds.loadFromFile(path, tokenizer_, cfg_, err)) {
        return false;
    }

    for (int ep = epoch_; ep < cfg_.epochs; ++ep) {
        ds.reset();

        std::vector<int> ids;

        while (ds.nextSequence(ids)) {
            if (ids.size() < 2) {
                continue;
            }

            double loss = trainSequence(ids, err);

            if (!err.empty()) {
                return false;
            }

            if (!std::isfinite(loss)) {
                err = "non-finite training loss";
                return false;
            }

            if (
                cfg_.saveEverySteps > 0 &&
                !modelDir.empty() &&
                globalStep_ > 0 &&
                globalStep_ % cfg_.saveEverySteps == 0
            ) {
                if (!saveCheckpoint(modelDir, err)) {
                    return false;
                }
            }
        }

        epoch_ = ep + 1;

        if (!modelDir.empty()) {
            if (!saveCheckpoint(modelDir, err)) {
                return false;
            }
        }
    }

    err.clear();
    return true;
}

bool Trainer::saveCheckpoint(
    const std::string& modelDir,
    std::string& err
) {
    CheckpointState st;

    st.modelName = "nol-model";
    st.epoch = epoch_;
    st.step = static_cast<int>(globalStep_);
    st.totalSteps = cfg_.totalSteps;
    st.stepsPerEpoch = 0;
    st.learningRate = currentLearningRate(cfg_, globalStep_ + 1);
    st.globalSeed = cfg_.seed;
    st.dataSeed = cfg_.seed;
    st.resumeReason = "manual-save";
    st.dirty = true;

    st.metrics.initialized = true;
    st.metrics.loss = static_cast<float>(lastLoss_);
    st.metrics.lmLoss = static_cast<float>(lastLoss_);
    st.metrics.acc = 0.0f;

    if (std::isfinite(lastLoss_) && lastLoss_ < 50.0) {
        st.metrics.ppl = static_cast<float>(std::exp(lastLoss_));
    } else {
        st.metrics.ppl = 0.0f;
    }

    st.schedule.type = cfg_.scheduleType;
    st.schedule.warmupSteps = cfg_.warmupSteps;
    st.schedule.totalSteps = cfg_.totalSteps;
    st.schedule.currentStep = static_cast<int>(globalStep_);
    st.schedule.currentLr = st.learningRate;

    std::map<std::string, StorageTensor> weights = exportWeights();
    std::map<std::string, StorageTensor> optimizer;

    opt_.exportStates(params_, optimizer);

    return ::saveCheckpoint(modelDir, st, weights, optimizer, err);
}

bool Trainer::loadCheckpoint(
    const std::string& modelDir,
    std::string& err
) {
    CheckpointState st;
    std::map<std::string, StorageTensor> weights;
    std::map<std::string, StorageTensor> optimizer;

    if (!::loadCheckpoint(modelDir, st, weights, optimizer, err)) {
        return false;
    }

    if (!importWeights(weights, err)) {
        return false;
    }

    if (!opt_.importStates(params_, optimizer, err)) {
        return false;
    }

    globalStep_ = static_cast<std::int64_t>(st.step);
    epoch_ = st.epoch;
    lastLoss_ = static_cast<double>(st.metrics.loss);

    cfg_.vocabSize = model_.vocab;

    err.clear();
    return true;
}

std::string Trainer::generate(
    const std::string& prompt,
    int maxNewTokens,
    float temperature,
    int topK,
    float topP,
    std::string& err
) const {
    err.clear();

    if (maxNewTokens <= 0) {
        return tokenizer_.decode(tokenizer_.encode(prompt));
    }

    std::vector<int> ids = tokenizer_.encode(prompt);

    int pad = tokenizer_.padId();

    if (pad < 0) {
        pad = 0;
    }

    if (model_.vocab > 0 && pad >= model_.vocab) {
        pad = 0;
    }

    for (int& id : ids) {
        id = normalizeToken(id, model_.vocab, pad);
    }

    int bos = normalizeToken(tokenizer_.bosId(), model_.vocab, pad);

    if (cfg_.includeBos && (ids.empty() || ids.front() != bos)) {
        ids.insert(ids.begin(), bos);
    }

    if (ids.empty()) {
        ids.push_back(bos);
    }

    int eosRaw = tokenizer_.eosId();
    int eos = -1;

    if (eosRaw >= 0 && eosRaw < model_.vocab) {
        eos = eosRaw;
    }

    for (int step = 0; step < maxNewTokens; ++step) {
        std::vector<int> positions(ids.size());
        std::iota(positions.begin(), positions.end(), 0);

        arch::Tensor logits;
        arch::nolModelForward(model_, ids, positions, logits);

        if (
            logits.shape.size() != 2 ||
            static_cast<std::size_t>(logits.shape[0]) != ids.size() ||
            logits.shape[1] <= 0
        ) {
            err = "invalid generation logits";
            return tokenizer_.decode(ids);
        }

        int vocab = static_cast<int>(logits.shape[1]);
        const float* last = logits.data.data() +
            (ids.size() - 1) * static_cast<std::size_t>(vocab);

        std::uint64_t seed = cfg_.seed;

        if (seed == 0) {
            seed = 123456789ULL;
        }

        seed ^= static_cast<std::uint64_t>(step + 1) * 0x9e3779b97f4a7c15ULL;

        int next = sampleToken(
            last,
            vocab,
            temperature,
            topK,
            topP,
            seed
        );

        if (next < 0) {
            err = "sampling failed";
            break;
        }

        next = normalizeToken(next, model_.vocab, pad);

        if (eos >= 0 && next == eos) {
            break;
        }

        ids.push_back(next);
    }

    return tokenizer_.decode(ids);
}

} // namespace tr

#ifdef TRAINER_SELFTEST

namespace {

bool allFiniteTensor(const arch::Tensor& t) {
    for (float x : t.data) {
        if (!std::isfinite(static_cast<double>(x))) {
            return false;
        }
    }

    return true;
}

double ceLoss(
    const arch::Tensor& logits,
    const std::vector<int>& targets
) {
    arch::Tensor grad;
    return tr::crossEntropy(logits, targets, grad);
}

void removeIfExistsLocal(const std::string& path) {
    (void)std::remove(path.c_str());
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

    std::string err;

    // 1. Cross entropy gradient check.
    {
        arch::Tensor logits;
        logits.shape = {2, 3};
        logits.data = {
            1.0f, 2.0f, 0.5f,
            -1.0f, 0.0f, 1.0f
        };

        std::vector<int> targets = {1, -1};
        arch::Tensor grad;

        double loss = tr::crossEntropy(logits, targets, grad);

        CHECK(std::isfinite(loss));
        CHECK(allFiniteTensor(grad));

        const double eps = 1e-3;

        for (std::size_t i = 0; i < logits.data.size(); ++i) {
            arch::Tensor lp = logits;
            arch::Tensor lm = logits;

            lp.data[i] += static_cast<float>(eps);
            lm.data[i] -= static_cast<float>(eps);

            double num = (ceLoss(lp, targets) - ceLoss(lm, targets)) / (2.0 * eps);

            CHECK(
                std::fabs(num - static_cast<double>(grad.data[i])) < 2e-2
            );
        }
    }

    // 2. Tiny trainer smoke test.
    //
    // Vocab dihitung dari tokenizer byte agar tidak terjadi
    // "token id out of embedding range".
    std::string corpusPath = "trainer_test_corpus.txt";
    std::string corpus =
        "alpha beta gamma alpha beta gamma\n"
        "delta epsilon zeta delta epsilon zeta\n"
        "one two three one two three\n";

    Tokenizer tok = Tokenizer::makeByte();

    std::vector<int> probe = tok.encode(corpus);

    int maxToken = 0;

    for (int id : probe) {
        if (id > maxToken) {
            maxToken = id;
        }
    }

    auto considerSpecial = [&](int id) {
        if (id > maxToken) {
            maxToken = id;
        }
    };

    considerSpecial(tok.bosId());
    considerSpecial(tok.eosId());
    considerSpecial(tok.padId());

    arch::ArchitectureDims dims;
    dims.vocab = std::max(16, maxToken + 1);
    dims.d_model = 8;
    dims.n_layers = 1;
    dims.n_heads = 2;
    dims.n_kv_heads = 1;
    dims.d_head = 4;
    dims.ffn_hidden = 16;

    dims.n_hyp_heads = 1;
    dims.d_hyp = 4;

    dims.n_top_anchors = 2;
    dims.n_children = 1;

    dims.enableT1 = true;
    dims.enableT2 = true;
    dims.enableT3 = true;
    dims.enableT4 = true;
    dims.enableT5 = true;

    dims.useMoe = true;
    dims.tiedEmbeddings = false;

    dims.seed = 4242;

    arch::NolModelParams model = arch::createRandomModel(dims);

    tr::Config cfg;
    cfg.seqLen = 8;
    cfg.epochs = 2;
    cfg.totalSteps = 20;
    cfg.warmupSteps = 2;
    cfg.learningRate = 1e-3f;
    cfg.minLr = 1e-5f;
    cfg.weightDecay = 0.0f;
    cfg.maxGradNorm = 1.0f;
    cfg.seed = 99;
    cfg.includeBos = true;
    cfg.includeEos = true;
    cfg.saveEverySteps = 0;
    cfg.scheduleType = "cosine";
    cfg.vocabSize = dims.vocab;

    tr::Trainer trainer(std::move(model), std::move(tok), cfg);

    CHECK(writeFileString(corpusPath, corpus, err));

    std::string modelDir = "trainer_test_model";

    CHECK(trainer.trainFile(corpusPath, modelDir, err));
    CHECK(std::isfinite(trainer.lastLoss()));
    CHECK(trainer.globalStep() > 0);

    // 3. Checkpoint roundtrip preserves forward output.
    {
        std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<int> positions(ids.size());
        std::iota(positions.begin(), positions.end(), 0);

        arch::Tensor before;
        arch::nolModelForward(trainer.model(), ids, positions, before);

        CHECK(allFiniteTensor(before));

        CHECK(trainer.saveCheckpoint(modelDir, err));

        trainer.model().embed.table.data[0] += 1.0f;

        arch::Tensor perturbed;
        arch::nolModelForward(trainer.model(), ids, positions, perturbed);

        bool changed = false;

        for (std::size_t i = 0; i < before.data.size(); ++i) {
            if (
                std::fabs(
                    static_cast<double>(before.data[i]) -
                    static_cast<double>(perturbed.data[i])
                ) > 1e-5
            ) {
                changed = true;
                break;
            }
        }

        CHECK(changed);

        CHECK(trainer.loadCheckpoint(modelDir, err));

        arch::Tensor after;
        arch::nolModelForward(trainer.model(), ids, positions, after);

        CHECK(allFiniteTensor(after));

        for (std::size_t i = 0; i < before.data.size(); ++i) {
            CHECK(
                std::fabs(
                    static_cast<double>(before.data[i]) -
                    static_cast<double>(after.data[i])
                ) < 1e-4
            );
        }
    }

    // 4. Generation smoke test.
    {
        std::string out = trainer.generate(
            "alpha",
            3,
            0.0f,
            0,
            1.0f,
            err
        );

        CHECK(err.empty());
        (void)out;
    }

    // 5. Dataset isolation: blank lines are skipped.
    {
        tr::Dataset ds;

        CHECK(ds.loadFromText(
            "\n   \nalpha beta\n\n",
            trainer.tokenizer(),
            trainer.config(),
            err
        ));

        CHECK(ds.size() >= 1);
    }

    removeIfExistsLocal(corpusPath);

    removeIfExistsLocal(modelDir + "/checkpoint/state.json");
    removeIfExistsLocal(modelDir + "/checkpoint/weights.safetensors");
    removeIfExistsLocal(modelDir + "/checkpoint/optimizer.safetensors");
    removeIfExistsLocal(modelDir + "/checkpoint/resume.log");

    if (failures == 0) {
        std::cout << "trainer unit tests passed\n";
        return 0;
    }

    std::cerr << failures << " trainer unit test(s) failed\n";
    return 1;
}

#endif // TRAINER_SELFTEST
