#ifndef PROYEK_TRAINER_HPP
#define PROYEK_TRAINER_HPP

#include "architecture.hpp"
#include "storage.hpp"
#include "tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tr {

struct Config {
    int seqLen = 32;
    int epochs = 1;

    int totalSteps = 0;
    int warmupSteps = 0;

    float learningRate = 1e-3f;
    float minLr = 1e-5f;

    float weightDecay = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float maxGradNorm = 1.0f;

    std::uint64_t seed = 1234;

    bool includeBos = true;
    bool includeEos = true;

    int saveEverySteps = 0;
    std::string scheduleType = "cosine";

    // Jika > 0, dataset akan menormalkan token out-of-vocab ke pad/fallback.
    // Trainer otomatis mengisi field ini dari model.vocab.
    int vocabSize = 0;
};

struct NamedParam {
    std::string name;
    arch::Tensor* tensor = nullptr;
    bool decay = false;
};

std::vector<NamedParam> collectParams(arch::NolModelParams& model);

std::vector<arch::Tensor*> collectGrads(arch::ModelGrad& grad);

float currentLearningRate(const Config& cfg, std::int64_t step);

double crossEntropy(
    const arch::Tensor& logits,
    const std::vector<int>& targets,
    arch::Tensor& gradLogits
);

class AdamW {
public:
    struct State {
        arch::Tensor m;
        arch::Tensor v;
    };

    AdamW() = default;

    void reset(std::size_t n);

    std::size_t size() const {
        return states_.size();
    }

    std::int64_t stepCount() const {
        return step_;
    }

    double step(
        std::vector<NamedParam>& params,
        const std::vector<arch::Tensor*>& grads,
        const Config& cfg,
        std::int64_t nextStep
    );

    void exportStates(
        const std::vector<NamedParam>& params,
        std::map<std::string, StorageTensor>& out
    ) const;

    bool importStates(
        const std::vector<NamedParam>& params,
        const std::map<std::string, StorageTensor>& in,
        std::string& err
    );

private:
    std::vector<State> states_;
    std::int64_t step_ = 0;
};

class Dataset {
public:
    bool loadFromFile(
        const std::string& path,
        const Tokenizer& tok,
        const Config& cfg,
        std::string& err
    );

    bool loadFromText(
        const std::string& text,
        const Tokenizer& tok,
        const Config& cfg,
        std::string& err
    );

    std::size_t size() const {
        return chunks_.size();
    }

    void reset() {
        cursor_ = 0;
    }

    bool nextSequence(std::vector<int>& ids);

private:
    std::vector<std::vector<int>> chunks_;
    std::size_t cursor_ = 0;
};

class Trainer {
public:
    Trainer(
        arch::NolModelParams modelIn,
        Tokenizer tokenizerIn,
        Config cfgIn
    );

    const arch::NolModelParams& model() const {
        return model_;
    }

    arch::NolModelParams& model() {
        return model_;
    }

    const Tokenizer& tokenizer() const {
        return tokenizer_;
    }

    const Config& config() const {
        return cfg_;
    }

    std::int64_t globalStep() const {
        return globalStep_;
    }

    int epoch() const {
        return epoch_;
    }

    double lastLoss() const {
        return lastLoss_;
    }

    double trainSequence(
        const std::vector<int>& ids,
        std::string& err
    );

    bool trainFile(
        const std::string& path,
        const std::string& modelDir,
        std::string& err
    );

    bool saveCheckpoint(
        const std::string& modelDir,
        std::string& err
    );

    bool loadCheckpoint(
        const std::string& modelDir,
        std::string& err
    );

    std::string generate(
        const std::string& prompt,
        int maxNewTokens,
        float temperature,
        int topK,
        float topP,
        std::string& err
    ) const;

private:
    arch::NolModelParams model_;
    Tokenizer tokenizer_;
    Config cfg_;

    std::vector<NamedParam> params_;
    AdamW opt_;

    std::int64_t globalStep_ = 0;
    int epoch_ = 0;
    double lastLoss_ = 0.0;

    std::map<std::string, StorageTensor> exportWeights() const;

    bool importWeights(
        const std::map<std::string, StorageTensor>& in,
        std::string& err
    );
};

} // namespace tr

#endif // PROYEK_TRAINER_HPP
