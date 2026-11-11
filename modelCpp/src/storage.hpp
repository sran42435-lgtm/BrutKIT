#ifndef PROYEK_STORAGE_HPP
#define PROYEK_STORAGE_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

enum class JsonType {
    Null,
    Bool,
    Number,
    String,
    Array,
    Object
};

struct JsonValue {
    JsonType type = JsonType::Null;

    bool boolValue = false;
    double number = 0.0;
    std::string str;

    std::vector<std::shared_ptr<JsonValue>> array;
    std::map<std::string, std::shared_ptr<JsonValue>> object;

    const JsonValue* find(const std::string& key) const;

    bool isNull() const;
    bool isBool() const;
    bool isNumber() const;
    bool isString() const;
    bool isArray() const;
    bool isObject() const;
};

std::string toJson(const JsonValue& value, int indent = -1);

bool parseJson(
    const std::string& text,
    JsonValue& out,
    std::string& err
);

bool jsonGetString(
    const JsonValue& obj,
    const std::string& key,
    std::string& out
);

bool jsonGetNumber(
    const JsonValue& obj,
    const std::string& key,
    double& out
);

bool jsonGetBool(
    const JsonValue& obj,
    const std::string& key,
    bool& out
);

bool jsonGetInt(
    const JsonValue& obj,
    const std::string& key,
    long long& out
);

bool jsonArrayToU64(
    const JsonValue& arr,
    std::vector<uint64_t>& out
);

std::string jsonStringOr(
    const JsonValue& obj,
    const std::string& key,
    const std::string& defaultValue
);

double jsonNumberOr(
    const JsonValue& obj,
    const std::string& key,
    double defaultValue
);

bool jsonBoolOr(
    const JsonValue& obj,
    const std::string& key,
    bool defaultValue
);

long long jsonIntOr(
    const JsonValue& obj,
    const std::string& key,
    long long defaultValue
);

struct StorageTensor {
    std::string name;
    std::string dtype = "F32";
    std::vector<uint64_t> shape;
    std::vector<float> data;
};

class SafetensorsWriter {
public:
    void addTensor(const StorageTensor& tensor);

    void setMetadata(
        const std::string& key,
        const std::string& value
    );

    bool write(
        const std::string& path,
        std::string& err
    ) const;

private:
    std::vector<StorageTensor> tensors_;
    std::map<std::string, std::string> metadata_;
};

class SafetensorsReader {
public:
    bool load(
        const std::string& path,
        std::string& err
    );

    std::vector<std::string> tensorNames() const;

    StorageTensor getTensor(const std::string& name) const;

    const std::map<std::string, std::string>& metadata() const;

private:
    std::vector<StorageTensor> tensors_;
    std::map<std::string, std::string> metadata_;
};

struct MetricsSnapshot {
    bool initialized = false;

    float loss = 0.0f;
    float lmLoss = 0.0f;
    float hclLoss = 0.0f;
    float hierLoss = 0.0f;
    float isoLoss = 0.0f;
    float balLoss = 0.0f;

    float acc = 0.0f;
    float ppl = 0.0f;
    float ctxMastery = 0.0f;
    float topicMixRate = 0.0f;

    float gradNorm = 0.0f;
    float lr = 0.0f;
};

struct ScheduleState {
    std::string type = "cosine";

    int warmupSteps = 0;
    int totalSteps = 0;
    int currentStep = 0;

    float currentLr = 0.0f;
};

struct CheckpointState {
    int schemaVersion = 1;

    std::string modelName;
    std::string createdAt;
    std::string updatedAt;

    int epoch = 0;
    int step = 0;
    int totalSteps = 0;
    int stepsPerEpoch = 0;
    int remainingSteps = 0;

    float learningRate = 0.0f;

    uint64_t globalSeed = 0;
    uint64_t dataSeed = 0;

    MetricsSnapshot metrics;
    ScheduleState schedule;

    std::string resumeReason;
    bool dirty = false;
};

bool saveCheckpoint(
    const std::string& modelDir,
    const CheckpointState& state,
    const std::map<std::string, StorageTensor>& weights,
    const std::map<std::string, StorageTensor>& optimizer,
    std::string& err
);

bool loadCheckpoint(
    const std::string& modelDir,
    CheckpointState& state,
    std::map<std::string, StorageTensor>& weights,
    std::map<std::string, StorageTensor>& optimizer,
    std::string& err
);

bool appendResumeEvent(
    const std::string& modelDir,
    const std::string& event,
    const JsonValue& data,
    std::string& err
);

struct ExportOptions {
    bool includeTokenizer = true;
    bool includeOptimizer = false;
    bool stripEma = true;

    std::string dtype = "F32";
    std::string cardText;

    bool overwrite = false;
};

bool exportModelBundle(
    const std::string& sourceDir,
    const std::string& targetDir,
    const ExportOptions& options,
    const std::map<std::string, StorageTensor>& weights,
    const std::string& configJson,
    const std::string& tokenizerJson,
    std::string& err
);

class FileLock {
public:
    explicit FileLock(std::string path);

    ~FileLock() noexcept;

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

    bool acquire(std::string& err);

    void release();

    bool locked() const;

private:
    std::string path_;
    bool locked_ = false;
};

bool pathExists(const std::string& path);

bool ensureDirectory(
    const std::string& path,
    std::string& err
);

std::string pathJoin(
    const std::string& a,
    const std::string& b
);

std::string pathBase(const std::string& path);

bool readFileString(
    const std::string& path,
    std::string& out,
    std::string& err
);

bool writeFileString(
    const std::string& path,
    const std::string& content,
    std::string& err
);

void removeIfExists(const std::string& path);

#endif // PROYEK_STORAGE_HPP
