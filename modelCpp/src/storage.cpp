#include "storage.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

#ifdef STORAGE_SELFTEST
#include <iostream>
#endif

namespace {

bool isSpaceJson(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

bool isPathSep(char c) {
    return c == '/' || c == '\\';
}

int hexDigit(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }

    if (c >= 'a' && c <= 'f') {
        return 10 + (c - 'a');
    }

    if (c >= 'A' && c <= 'F') {
        return 10 + (c - 'A');
    }

    return -1;
}

void appendUtf8(std::string& out, uint32_t cp) {
    if (cp < 0x80U) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (cp >> 6)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else if (cp < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (cp >> 12)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (cp >> 18)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 12) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((cp >> 6) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
    }
}

class JsonParser {
public:
    explicit JsonParser(const std::string& text)
        : s_(text), pos_(0) {}

    bool parse(JsonValue& out) {
        skipWs();

        if (!parseValue(out)) {
            return false;
        }

        skipWs();

        if (pos_ != s_.size()) {
            return fail("trailing characters after JSON value");
        }

        return true;
    }

    const std::string& error() const {
        return err_;
    }

private:
    const std::string& s_;
    std::size_t pos_;
    std::string err_;

    bool fail(const std::string& msg) {
        err_ = msg;
        return false;
    }

    void skipWs() {
        while (pos_ < s_.size() && isSpaceJson(s_[pos_])) {
            ++pos_;
        }
    }

    bool peek(char& c) const {
        if (pos_ >= s_.size()) {
            return false;
        }

        c = s_[pos_];
        return true;
    }

    bool parseValue(JsonValue& out) {
        skipWs();

        char c = 0;

        if (!peek(c)) {
            return fail("unexpected end of JSON input");
        }

        if (c == '{') {
            return parseObject(out);
        }

        if (c == '[') {
            return parseArray(out);
        }

        if (c == '"') {
            out = JsonValue();
            out.type = JsonType::String;
            return parseStringRaw(out.str);
        }

        if (c == 't') {
            return parseLiteralTrue(out);
        }

        if (c == 'f') {
            return parseLiteralFalse(out);
        }

        if (c == 'n') {
            return parseLiteralNull(out);
        }

        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parseNumber(out);
        }

        return fail("unexpected character in JSON value");
    }

    bool parseLiteralTrue(JsonValue& out) {
        if (pos_ + 4 > s_.size() || s_.compare(pos_, 4, "true") != 0) {
            return fail("invalid literal, expected true");
        }

        out = JsonValue();
        out.type = JsonType::Bool;
        out.boolValue = true;
        pos_ += 4;
        return true;
    }

    bool parseLiteralFalse(JsonValue& out) {
        if (pos_ + 5 > s_.size() || s_.compare(pos_, 5, "false") != 0) {
            return fail("invalid literal, expected false");
        }

        out = JsonValue();
        out.type = JsonType::Bool;
        out.boolValue = false;
        pos_ += 5;
        return true;
    }

    bool parseLiteralNull(JsonValue& out) {
        if (pos_ + 4 > s_.size() || s_.compare(pos_, 4, "null") != 0) {
            return fail("invalid literal, expected null");
        }

        out = JsonValue();
        out.type = JsonType::Null;
        pos_ += 4;
        return true;
    }

    bool parseNumber(JsonValue& out) {
        std::size_t start = pos_;

        if (pos_ < s_.size() && s_[pos_] == '-') {
            ++pos_;
        }

        if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            return fail("invalid number");
        }

        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }

        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;

            if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                return fail("invalid fraction in number");
            }

            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                ++pos_;
            }
        }

        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;

            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) {
                ++pos_;
            }

            if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                return fail("invalid exponent in number");
            }

            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                ++pos_;
            }
        }

        std::string token = s_.substr(start, pos_ - start);
        char* end = nullptr;

        double value = std::strtod(token.c_str(), &end);

        if (end == nullptr || static_cast<const char*>(end) != token.c_str() + token.size()) {
            return fail("invalid numeric token");
        }

        if (!std::isfinite(value)) {
            return fail("non-finite number is not valid JSON");
        }

        out = JsonValue();
        out.type = JsonType::Number;
        out.number = value;
        return true;
    }

    bool parseHex4(uint32_t& out) {
        if (pos_ + 4 > s_.size()) {
            return fail("truncated unicode escape");
        }

        uint32_t value = 0;

        for (int i = 0; i < 4; ++i) {
            int d = hexDigit(s_[pos_++]);

            if (d < 0) {
                return fail("invalid hexadecimal digit in unicode escape");
            }

            value = (value << 4U) | static_cast<uint32_t>(d);
        }

        out = value;
        return true;
    }

    bool parseStringRaw(std::string& out) {
        out.clear();

        if (pos_ >= s_.size() || s_[pos_] != '"') {
            return fail("expected string");
        }

        ++pos_;

        while (true) {
            if (pos_ >= s_.size()) {
                return fail("unterminated string");
            }

            unsigned char uc = static_cast<unsigned char>(s_[pos_++]);

            if (uc == '"') {
                return true;
            }

            if (uc == '\\') {
                if (pos_ >= s_.size()) {
                    return fail("unterminated escape sequence");
                }

                char e = s_[pos_++];

                switch (e) {
                    case '"':
                        out.push_back('"');
                        break;

                    case '\\':
                        out.push_back('\\');
                        break;

                    case '/':
                        out.push_back('/');
                        break;

                    case 'b':
                        out.push_back('\b');
                        break;

                    case 'f':
                        out.push_back('\f');
                        break;

                    case 'n':
                        out.push_back('\n');
                        break;

                    case 'r':
                        out.push_back('\r');
                        break;

                    case 't':
                        out.push_back('\t');
                        break;

                    case 'u': {
                        uint32_t cp = 0;

                        if (!parseHex4(cp)) {
                            return false;
                        }

                        if (cp >= 0xD800U && cp <= 0xDBFFU) {
                            std::size_t save = pos_;

                            if (
                                pos_ + 1 < s_.size() &&
                                s_[pos_] == '\\' &&
                                s_[pos_ + 1] == 'u'
                            ) {
                                pos_ += 2;

                                uint32_t low = 0;

                                if (!parseHex4(low)) {
                                    return false;
                                }

                                if (low >= 0xDC00U && low <= 0xDFFFU) {
                                    cp =
                                        0x10000U +
                                        ((cp - 0xD800U) << 10U) +
                                        (low - 0xDC00U);
                                } else {
                                    pos_ = save;
                                    appendUtf8(out, 0xFFFDU);
                                }
                            } else {
                                appendUtf8(out, 0xFFFDU);
                            }
                        } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
                            appendUtf8(out, 0xFFFDU);
                        }

                        if (cp < 0xD800U || cp > 0xDFFFU) {
                            appendUtf8(out, cp);
                        }

                        break;
                    }

                    default:
                        return fail("invalid escape sequence");
                }

                continue;
            }

            if (uc < 0x20U) {
                return fail("unescaped control character in string");
            }

            out.push_back(static_cast<char>(uc));
        }
    }

    bool parseObject(JsonValue& out) {
        out = JsonValue();
        out.type = JsonType::Object;

        ++pos_;
        skipWs();

        char c = 0;

        if (peek(c) && c == '}') {
            ++pos_;
            return true;
        }

        while (true) {
            skipWs();

            if (!peek(c) || c != '"') {
                return fail("object key must be a string");
            }

            std::string key;

            if (!parseStringRaw(key)) {
                return false;
            }

            skipWs();

            if (!peek(c) || c != ':') {
                return fail("expected ':' after object key");
            }

            ++pos_;
            skipWs();

            JsonValue value;

            if (!parseValue(value)) {
                return false;
            }

            auto ptr = std::make_shared<JsonValue>();
            *ptr = std::move(value);
            out.object[key] = ptr;

            skipWs();

            if (!peek(c)) {
                return fail("unterminated object");
            }

            if (c == ',') {
                ++pos_;
                continue;
            }

            if (c == '}') {
                ++pos_;
                return true;
            }

            return fail("expected ',' or '}' in object");
        }
    }

    bool parseArray(JsonValue& out) {
        out = JsonValue();
        out.type = JsonType::Array;

        ++pos_;
        skipWs();

        char c = 0;

        if (peek(c) && c == ']') {
            ++pos_;
            return true;
        }

        while (true) {
            skipWs();

            JsonValue value;

            if (!parseValue(value)) {
                return false;
            }

            auto ptr = std::make_shared<JsonValue>();
            *ptr = std::move(value);
            out.array.push_back(ptr);

            skipWs();

            if (!peek(c)) {
                return fail("unterminated array");
            }

            if (c == ',') {
                ++pos_;
                continue;
            }

            if (c == ']') {
                ++pos_;
                return true;
            }

            return fail("expected ',' or ']' in array");
        }
    }
};

void serializeString(std::ostringstream& os, const std::string& s) {
    os << '"';

    for (unsigned char c : s) {
        switch (c) {
            case '"':
                os << "\\\"";
                break;

            case '\\':
                os << "\\\\";
                break;

            case '\b':
                os << "\\b";
                break;

            case '\f':
                os << "\\f";
                break;

            case '\n':
                os << "\\n";
                break;

            case '\r':
                os << "\\r";
                break;

            case '\t':
                os << "\\t";
                break;

            default:
                if (c < 0x20U) {
                    const char* hex = "0123456789abcdef";
                    os << "\\u00";
                    os << hex[(c >> 4) & 0x0fU];
                    os << hex[c & 0x0fU];
                } else {
                    os << static_cast<char>(c);
                }

                break;
        }
    }

    os << '"';
}

void writeSpaces(std::ostringstream& os, int count) {
    for (int i = 0; i < count; ++i) {
        os << ' ';
    }
}

void serializeValue(
    std::ostringstream& os,
    const JsonValue& v,
    int indent,
    int level
) {
    switch (v.type) {
        case JsonType::Null:
            os << "null";
            break;

        case JsonType::Bool:
            os << (v.boolValue ? "true" : "false");
            break;

        case JsonType::Number:
            if (std::isfinite(v.number)) {
                os << v.number;
            } else {
                os << "null";
            }

            break;

        case JsonType::String:
            serializeString(os, v.str);
            break;

        case JsonType::Array:
            if (v.array.empty()) {
                os << "[]";
                break;
            }

            os << '[';

            for (std::size_t i = 0; i < v.array.size(); ++i) {
                if (i != 0) {
                    os << ',';
                }

                if (indent >= 0) {
                    os << '\n';
                    writeSpaces(os, indent * (level + 1));
                }

                if (v.array[i]) {
                    serializeValue(os, *v.array[i], indent, level + 1);
                } else {
                    os << "null";
                }
            }

            if (indent >= 0) {
                os << '\n';
                writeSpaces(os, indent * level);
            }

            os << ']';
            break;

        case JsonType::Object:
            if (v.object.empty()) {
                os << "{}";
                break;
            }

            os << '{';

            {
                bool first = true;

                for (const auto& kv : v.object) {
                    if (!first) {
                        os << ',';
                    }

                    first = false;

                    if (indent >= 0) {
                        os << '\n';
                        writeSpaces(os, indent * (level + 1));
                    }

                    serializeString(os, kv.first);
                    os << ':';

                    if (indent >= 0) {
                        os << ' ';
                    }

                    if (kv.second) {
                        serializeValue(os, *kv.second, indent, level + 1);
                    } else {
                        os << "null";
                    }
                }
            }

            if (indent >= 0) {
                os << '\n';
                writeSpaces(os, indent * level);
            }

            os << '}';
            break;
    }
}

std::shared_ptr<JsonValue> makeStringValue(const std::string& value) {
    auto ptr = std::make_shared<JsonValue>();
    ptr->type = JsonType::String;
    ptr->str = value;
    return ptr;
}

std::shared_ptr<JsonValue> makeNumberValue(double value) {
    auto ptr = std::make_shared<JsonValue>();
    ptr->type = JsonType::Number;
    ptr->number = value;
    return ptr;
}

std::shared_ptr<JsonValue> makeBoolValue(bool value) {
    auto ptr = std::make_shared<JsonValue>();
    ptr->type = JsonType::Bool;
    ptr->boolValue = value;
    return ptr;
}

void jsonSetString(
    JsonValue& obj,
    const std::string& key,
    const std::string& value
) {
    obj.type = JsonType::Object;
    obj.object[key] = makeStringValue(value);
}

void jsonSetNumber(
    JsonValue& obj,
    const std::string& key,
    double value
) {
    obj.type = JsonType::Object;
    obj.object[key] = makeNumberValue(value);
}

void jsonSetBool(
    JsonValue& obj,
    const std::string& key,
    bool value
) {
    obj.type = JsonType::Object;
    obj.object[key] = makeBoolValue(value);
}

void jsonSetInt(
    JsonValue& obj,
    const std::string& key,
    long long value
) {
    jsonSetNumber(obj, key, static_cast<double>(value));
}

void jsonSetNested(
    JsonValue& obj,
    const std::string& key,
    const JsonValue& value
) {
    obj.type = JsonType::Object;

    auto ptr = std::make_shared<JsonValue>();
    *ptr = value;
    obj.object[key] = ptr;
}

void jsonPushString(
    JsonValue& arr,
    const std::string& value
) {
    arr.type = JsonType::Array;
    arr.array.push_back(makeStringValue(value));
}

std::string currentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);

    std::tm tm{};

#ifdef _WIN32
    (void)gmtime_s(&tm, &tt);
#else
    (void)gmtime_r(&tt, &tm);
#endif

    char buf[32];
    int written = std::snprintf(
        buf,
        sizeof(buf),
        "%04d-%02d-%02dT%02d:%02d:%02dZ",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    );

    (void)written;
    return std::string(buf);
}

bool mkdirOne(
    const std::string& path,
    std::string& err
) {
    if (path.empty()) {
        return true;
    }

#ifdef _WIN32
    if (_mkdir(path.c_str()) == 0) {
        return true;
    }
#else
    if (::mkdir(path.c_str(), 0755) == 0) {
        return true;
    }
#endif

    if (errno == EEXIST) {
        return true;
    }

    err = "cannot create directory: " + path;
    return false;
}

bool ensureDirectoryImpl(
    const std::string& path,
    std::string& err
) {
    if (path.empty()) {
        return true;
    }

    std::string cur;

    for (std::size_t i = 0; i < path.size(); ++i) {
        cur.push_back(path[i]);

        if (isPathSep(path[i]) && cur.size() > 1) {
            std::string dir = cur.substr(0, cur.size() - 1);

            if (!mkdirOne(dir, err)) {
                return false;
            }
        }
    }

    if (!cur.empty() && !isPathSep(cur.back())) {
        if (!mkdirOne(cur, err)) {
            return false;
        }
    }

    return true;
}

bool readFileBytes(
    const std::string& path,
    std::vector<unsigned char>& out,
    std::string& err
) {
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
        err = "cannot open file: " + path;
        return false;
    }

    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();

    if (end < 0) {
        err = "cannot determine file size: " + path;
        return false;
    }

    f.seekg(0, std::ios::beg);

    std::size_t size = static_cast<std::size_t>(end);
    out.resize(size);

    if (size > 0) {
        f.read(
            reinterpret_cast<char*>(out.data()),
            static_cast<std::streamsize>(size)
        );

        if (static_cast<std::size_t>(f.gcount()) != size) {
            err = "failed reading file: " + path;
            return false;
        }
    }

    return true;
}

bool writeFileBytesImpl(
    const std::string& path,
    const void* data,
    std::size_t size,
    std::string& err
) {
    std::size_t slash = path.find_last_of("/\\");

    if (slash != std::string::npos && slash > 0) {
        std::string parent = path.substr(0, slash);

        if (!ensureDirectoryImpl(parent, err)) {
            return false;
        }
    }

    std::string tmp = path + ".tmp";

    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);

    if (!out.is_open()) {
        err = "cannot open temp file: " + tmp;
        return false;
    }

    if (size > 0) {
        out.write(
            static_cast<const char*>(data),
            static_cast<std::streamsize>(size)
        );
    }

    out.flush();

    if (!out.good()) {
        err = "failed writing temp file: " + tmp;
        out.close();
        removeIfExists(tmp);
        return false;
    }

    out.close();

#ifdef _WIN32
    if (pathExists(path)) {
        (void)std::remove(path.c_str());
    }
#endif

    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "failed renaming temp file: " + tmp + " -> " + path;
        removeIfExists(tmp);
        return false;
    }

    return true;
}

void writeLE64(
    std::vector<unsigned char>& out,
    uint64_t x
) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<unsigned char>((x >> (8 * i)) & 0xffULL));
    }
}

uint64_t readLE64(const unsigned char* p) {
    uint64_t x = 0;

    for (int i = 0; i < 8; ++i) {
        x |= static_cast<uint64_t>(p[i]) << (8 * i);
    }

    return x;
}

uint64_t tensorNumel(const std::vector<uint64_t>& shape) {
    uint64_t n = 1;

    for (uint64_t d : shape) {
        if (d == 0) {
            return 0;
        }

        n *= d;
    }

    return n;
}

void metricsToJson(
    const MetricsSnapshot& m,
    JsonValue& out
) {
    out = JsonValue();
    out.type = JsonType::Object;

    jsonSetBool(out, "initialized", m.initialized);

    jsonSetNumber(out, "loss", m.loss);
    jsonSetNumber(out, "lm_loss", m.lmLoss);
    jsonSetNumber(out, "hcl_loss", m.hclLoss);
    jsonSetNumber(out, "hier_loss", m.hierLoss);
    jsonSetNumber(out, "iso_loss", m.isoLoss);
    jsonSetNumber(out, "bal_loss", m.balLoss);

    jsonSetNumber(out, "acc", m.acc);
    jsonSetNumber(out, "ppl", m.ppl);
    jsonSetNumber(out, "ctx_mastery", m.ctxMastery);
    jsonSetNumber(out, "topic_mix_rate", m.topicMixRate);

    jsonSetNumber(out, "grad_norm", m.gradNorm);
    jsonSetNumber(out, "lr", m.lr);
}

bool metricsFromJson(
    const JsonValue& in,
    MetricsSnapshot& m,
    std::string& err
) {
    (void)err;

    m = MetricsSnapshot();

    if (!in.isObject()) {
        return false;
    }

    m.initialized = jsonBoolOr(in, "initialized", false);

    if (!m.initialized && in.find("loss") != nullptr) {
        m.initialized = true;
    }

    m.loss = static_cast<float>(jsonNumberOr(in, "loss", 0.0));
    m.lmLoss = static_cast<float>(jsonNumberOr(in, "lm_loss", 0.0));
    m.hclLoss = static_cast<float>(jsonNumberOr(in, "hcl_loss", 0.0));
    m.hierLoss = static_cast<float>(jsonNumberOr(in, "hier_loss", 0.0));
    m.isoLoss = static_cast<float>(jsonNumberOr(in, "iso_loss", 0.0));
    m.balLoss = static_cast<float>(jsonNumberOr(in, "bal_loss", 0.0));

    m.acc = static_cast<float>(jsonNumberOr(in, "acc", 0.0));
    m.ppl = static_cast<float>(jsonNumberOr(in, "ppl", 0.0));
    m.ctxMastery = static_cast<float>(jsonNumberOr(in, "ctx_mastery", 0.0));
    m.topicMixRate = static_cast<float>(jsonNumberOr(in, "topic_mix_rate", 0.0));

    m.gradNorm = static_cast<float>(jsonNumberOr(in, "grad_norm", 0.0));
    m.lr = static_cast<float>(jsonNumberOr(in, "lr", 0.0));

    return true;
}

void scheduleToJson(
    const ScheduleState& s,
    JsonValue& out
) {
    out = JsonValue();
    out.type = JsonType::Object;

    jsonSetString(out, "type", s.type);
    jsonSetInt(out, "warmup_steps", s.warmupSteps);
    jsonSetInt(out, "total_steps", s.totalSteps);
    jsonSetInt(out, "current_step", s.currentStep);
    jsonSetNumber(out, "current_lr", s.currentLr);
}

bool scheduleFromJson(
    const JsonValue& in,
    ScheduleState& s,
    std::string& err
) {
    (void)err;

    s = ScheduleState();

    if (!in.isObject()) {
        return false;
    }

    s.type = jsonStringOr(in, "type", "cosine");
    s.warmupSteps = static_cast<int>(jsonIntOr(in, "warmup_steps", 0));
    s.totalSteps = static_cast<int>(jsonIntOr(in, "total_steps", 0));
    s.currentStep = static_cast<int>(jsonIntOr(in, "current_step", 0));
    s.currentLr = static_cast<float>(jsonNumberOr(in, "current_lr", 0.0));

    return true;
}

void checkpointStateToJson(
    const CheckpointState& st,
    JsonValue& out
) {
    out = JsonValue();
    out.type = JsonType::Object;

    jsonSetInt(out, "schema_version", st.schemaVersion);
    jsonSetString(out, "model_name", st.modelName);
    jsonSetString(out, "created_at", st.createdAt);
    jsonSetString(out, "updated_at", st.updatedAt);

    jsonSetInt(out, "epoch", st.epoch);
    jsonSetInt(out, "step", st.step);
    jsonSetInt(out, "total_steps", st.totalSteps);
    jsonSetInt(out, "steps_per_epoch", st.stepsPerEpoch);
    jsonSetInt(out, "remaining_steps", st.remainingSteps);

    jsonSetNumber(out, "learning_rate", st.learningRate);

    jsonSetString(out, "global_seed", std::to_string(st.globalSeed));
    jsonSetString(out, "data_seed", std::to_string(st.dataSeed));

    JsonValue metrics;
    metricsToJson(st.metrics, metrics);
    jsonSetNested(out, "metrics", metrics);

    JsonValue schedule;
    scheduleToJson(st.schedule, schedule);
    jsonSetNested(out, "schedule", schedule);

    jsonSetString(out, "resume_reason", st.resumeReason);
    jsonSetBool(out, "dirty", st.dirty);
}

bool uint64FromString(
    const std::string& s,
    uint64_t& out
) {
    if (s.empty()) {
        out = 0;
        return true;
    }

    try {
        std::size_t pos = 0;
        unsigned long long v = std::stoull(s, &pos);

        if (pos != s.size()) {
            return false;
        }

        out = static_cast<uint64_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

bool checkpointStateFromJson(
    const JsonValue& in,
    CheckpointState& st,
    std::string& err
) {
    st = CheckpointState();

    if (!in.isObject()) {
        err = "checkpoint state root must be an object";
        return false;
    }

    st.schemaVersion = static_cast<int>(jsonIntOr(in, "schema_version", 1));

    if (st.schemaVersion < 1) {
        err = "unsupported checkpoint schema version";
        return false;
    }

    st.modelName = jsonStringOr(in, "model_name", "");
    st.createdAt = jsonStringOr(in, "created_at", "");
    st.updatedAt = jsonStringOr(in, "updated_at", "");

    st.epoch = static_cast<int>(jsonIntOr(in, "epoch", 0));
    st.step = static_cast<int>(jsonIntOr(in, "step", 0));
    st.totalSteps = static_cast<int>(jsonIntOr(in, "total_steps", 0));
    st.stepsPerEpoch = static_cast<int>(jsonIntOr(in, "steps_per_epoch", 0));
    st.remainingSteps = static_cast<int>(jsonIntOr(in, "remaining_steps", 0));

    st.learningRate = static_cast<float>(jsonNumberOr(in, "learning_rate", 0.0));

    std::string globalSeedStr = jsonStringOr(in, "global_seed", "0");
    std::string dataSeedStr = jsonStringOr(in, "data_seed", "0");

    if (!uint64FromString(globalSeedStr, st.globalSeed)) {
        err = "invalid global_seed in checkpoint state";
        return false;
    }

    if (!uint64FromString(dataSeedStr, st.dataSeed)) {
        err = "invalid data_seed in checkpoint state";
        return false;
    }

    const JsonValue* metrics = in.find("metrics");

    if (metrics != nullptr) {
        if (!metricsFromJson(*metrics, st.metrics, err)) {
            err = "invalid metrics object in checkpoint state";
            return false;
        }
    }

    const JsonValue* schedule = in.find("schedule");

    if (schedule != nullptr) {
        if (!scheduleFromJson(*schedule, st.schedule, err)) {
            err = "invalid schedule object in checkpoint state";
            return false;
        }
    }

    st.resumeReason = jsonStringOr(in, "resume_reason", "");
    st.dirty = jsonBoolOr(in, "dirty", false);

    return true;
}

} // namespace

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != JsonType::Object) {
        return nullptr;
    }

    auto it = object.find(key);

    if (it == object.end()) {
        return nullptr;
    }

    return it->second.get();
}

bool JsonValue::isNull() const {
    return type == JsonType::Null;
}

bool JsonValue::isBool() const {
    return type == JsonType::Bool;
}

bool JsonValue::isNumber() const {
    return type == JsonType::Number;
}

bool JsonValue::isString() const {
    return type == JsonType::String;
}

bool JsonValue::isArray() const {
    return type == JsonType::Array;
}

bool JsonValue::isObject() const {
    return type == JsonType::Object;
}

std::string toJson(const JsonValue& value, int indent) {
    std::ostringstream os;
    os << std::setprecision(17);
    serializeValue(os, value, indent, 0);
    return os.str();
}

bool parseJson(
    const std::string& text,
    JsonValue& out,
    std::string& err
) {
    JsonParser parser(text);
    out = JsonValue();

    if (!parser.parse(out)) {
        err = parser.error();
        return false;
    }

    err.clear();
    return true;
}

bool jsonGetString(
    const JsonValue& obj,
    const std::string& key,
    std::string& out
) {
    const JsonValue* v = obj.find(key);

    if (v == nullptr || v->type != JsonType::String) {
        return false;
    }

    out = v->str;
    return true;
}

bool jsonGetNumber(
    const JsonValue& obj,
    const std::string& key,
    double& out
) {
    const JsonValue* v = obj.find(key);

    if (v == nullptr || v->type != JsonType::Number) {
        return false;
    }

    if (!std::isfinite(v->number)) {
        return false;
    }

    out = v->number;
    return true;
}

bool jsonGetBool(
    const JsonValue& obj,
    const std::string& key,
    bool& out
) {
    const JsonValue* v = obj.find(key);

    if (v == nullptr || v->type != JsonType::Bool) {
        return false;
    }

    out = v->boolValue;
    return true;
}

bool jsonGetInt(
    const JsonValue& obj,
    const std::string& key,
    long long& out
) {
    double d = 0.0;

    if (!jsonGetNumber(obj, key, d)) {
        return false;
    }

    if (
        d < static_cast<double>(LLONG_MIN) ||
        d > static_cast<double>(LLONG_MAX)
    ) {
        return false;
    }

    out = static_cast<long long>(d);
    return true;
}

bool jsonArrayToU64(
    const JsonValue& arr,
    std::vector<uint64_t>& out
) {
    out.clear();

    if (!arr.isArray()) {
        return false;
    }

    for (const auto& ptr : arr.array) {
        if (!ptr || ptr->type != JsonType::Number) {
            return false;
        }

        if (!std::isfinite(ptr->number) || ptr->number < 0.0) {
            return false;
        }

        if (ptr->number > static_cast<double>(UINT64_MAX)) {
            return false;
        }

        out.push_back(static_cast<uint64_t>(ptr->number));
    }

    return true;
}

std::string jsonStringOr(
    const JsonValue& obj,
    const std::string& key,
    const std::string& defaultValue
) {
    std::string v;

    if (jsonGetString(obj, key, v)) {
        return v;
    }

    return defaultValue;
}

double jsonNumberOr(
    const JsonValue& obj,
    const std::string& key,
    double defaultValue
) {
    double v = 0.0;

    if (jsonGetNumber(obj, key, v)) {
        return v;
    }

    return defaultValue;
}

bool jsonBoolOr(
    const JsonValue& obj,
    const std::string& key,
    bool defaultValue
) {
    bool v = false;

    if (jsonGetBool(obj, key, v)) {
        return v;
    }

    return defaultValue;
}

long long jsonIntOr(
    const JsonValue& obj,
    const std::string& key,
    long long defaultValue
) {
    long long v = 0;

    if (jsonGetInt(obj, key, v)) {
        return v;
    }

    return defaultValue;
}

bool pathExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

bool ensureDirectory(
    const std::string& path,
    std::string& err
) {
    return ensureDirectoryImpl(path, err);
}

std::string pathJoin(
    const std::string& a,
    const std::string& b
) {
    if (a.empty()) {
        return b;
    }

    if (b.empty()) {
        return a;
    }

    if (isPathSep(a.back())) {
        return a + b;
    }

    return a + "/" + b;
}

std::string pathBase(const std::string& path) {
    std::size_t pos = path.find_last_of("/\\");

    if (pos == std::string::npos) {
        return path;
    }

    if (pos + 1 >= path.size()) {
        return std::string();
    }

    return path.substr(pos + 1);
}

bool readFileString(
    const std::string& path,
    std::string& out,
    std::string& err
) {
    std::vector<unsigned char> bytes;

    if (!readFileBytes(path, bytes, err)) {
        return false;
    }

    out.assign(
        reinterpret_cast<const char*>(bytes.data()),
        bytes.size()
    );

    return true;
}

bool writeFileString(
    const std::string& path,
    const std::string& content,
    std::string& err
) {
    return writeFileBytesImpl(
        path,
        content.data(),
        content.size(),
        err
    );
}

void removeIfExists(const std::string& path) {
    (void)std::remove(path.c_str());
}

void SafetensorsWriter::addTensor(const StorageTensor& tensor) {
    tensors_.push_back(tensor);
}

void SafetensorsWriter::setMetadata(
    const std::string& key,
    const std::string& value
) {
    metadata_[key] = value;
}

bool SafetensorsWriter::write(
    const std::string& path,
    std::string& err
) const {
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (tensors_[i].name == tensors_[j].name) {
                err = "duplicate tensor name: " + tensors_[i].name;
                return false;
            }
        }
    }

    std::vector<uint64_t> offsets;
    offsets.reserve(tensors_.size());

    uint64_t totalData = 0;

    for (const StorageTensor& t : tensors_) {
        if (t.dtype != "F32") {
            err = "unsupported dtype: " + t.dtype;
            return false;
        }

        uint64_t numel = tensorNumel(t.shape);
        uint64_t bytes = numel * 4ULL;

        if (static_cast<uint64_t>(t.data.size()) != numel) {
            err = "tensor data size mismatch for: " + t.name;
            return false;
        }

        if (static_cast<uint64_t>(t.data.size()) * 4ULL != bytes) {
            err = "tensor byte size mismatch for: " + t.name;
            return false;
        }

        offsets.push_back(totalData);
        totalData += bytes;
    }

    JsonValue header;
    header.type = JsonType::Object;

    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        const StorageTensor& t = tensors_[i];

        JsonValue meta;
        meta.type = JsonType::Object;

        jsonSetString(meta, "dtype", t.dtype);

        JsonValue shape;
        shape.type = JsonType::Array;

        for (uint64_t d : t.shape) {
            shape.array.push_back(makeNumberValue(static_cast<double>(d)));
        }

        meta.object["shape"] = std::make_shared<JsonValue>(shape);

        uint64_t begin = offsets[i];
        uint64_t end = begin + static_cast<uint64_t>(t.data.size()) * 4ULL;

        JsonValue dataOffsets;
        dataOffsets.type = JsonType::Array;
        dataOffsets.array.push_back(makeNumberValue(static_cast<double>(begin)));
        dataOffsets.array.push_back(makeNumberValue(static_cast<double>(end)));

        meta.object["data_offsets"] = std::make_shared<JsonValue>(dataOffsets);

        header.object[t.name] = std::make_shared<JsonValue>(meta);
    }

    if (!metadata_.empty()) {
        JsonValue md;
        md.type = JsonType::Object;

        for (const auto& kv : metadata_) {
            jsonSetString(md, kv.first, kv.second);
        }

        header.object["__metadata__"] = std::make_shared<JsonValue>(md);
    }

    std::string headerJson = toJson(header, -1);

    std::vector<unsigned char> file;
    file.reserve(
        8 +
        headerJson.size() +
        static_cast<std::size_t>(totalData)
    );

    writeLE64(file, static_cast<uint64_t>(headerJson.size()));

    file.insert(
        file.end(),
        reinterpret_cast<const unsigned char*>(headerJson.data()),
        reinterpret_cast<const unsigned char*>(headerJson.data()) + headerJson.size()
    );

    for (const StorageTensor& t : tensors_) {
        const unsigned char* begin =
            reinterpret_cast<const unsigned char*>(t.data.data());

        const unsigned char* end =
            begin + t.data.size() * sizeof(float);

        file.insert(file.end(), begin, end);
    }

    return writeFileBytesImpl(
        path,
        file.data(),
        file.size(),
        err
    );
}

bool SafetensorsReader::load(
    const std::string& path,
    std::string& err
) {
    tensors_.clear();
    metadata_.clear();

    std::vector<unsigned char> bytes;

    if (!readFileBytes(path, bytes, err)) {
        return false;
    }

    if (bytes.size() < 8) {
        err = "safetensors file too small";
        return false;
    }

    uint64_t headerLen = readLE64(bytes.data());

    if (headerLen > static_cast<uint64_t>(bytes.size()) - 8ULL) {
        err = "safetensors header length exceeds file size";
        return false;
    }

    std::string headerJson(
        reinterpret_cast<const char*>(bytes.data() + 8),
        static_cast<std::size_t>(headerLen)
    );

    JsonValue root;

    if (!parseJson(headerJson, root, err)) {
        return false;
    }

    if (!root.isObject()) {
        err = "safetensors header must be a JSON object";
        return false;
    }

    std::size_t dataStart = 8 + static_cast<std::size_t>(headerLen);

    if (dataStart > bytes.size()) {
        err = "safetensors data start exceeds file size";
        return false;
    }

    uint64_t available = static_cast<uint64_t>(bytes.size() - dataStart);

    for (const auto& kv : root.object) {
        if (kv.first == "__metadata__") {
            if (kv.second && kv.second->isObject()) {
                for (const auto& mkv : kv.second->object) {
                    if (mkv.second && mkv.second->isString()) {
                        metadata_[mkv.first] = mkv.second->str;
                    }
                }
            }

            continue;
        }

        if (!kv.second || !kv.second->isObject()) {
            err = "invalid tensor metadata for: " + kv.first;
            return false;
        }

        const JsonValue& meta = *kv.second;

        std::string dtype;

        if (!jsonGetString(meta, "dtype", dtype)) {
            err = "missing dtype for tensor: " + kv.first;
            return false;
        }

        if (dtype != "F32") {
            err = "unsupported dtype for tensor " + kv.first + ": " + dtype;
            return false;
        }

        const JsonValue* shapeJson = meta.find("shape");

        if (shapeJson == nullptr) {
            err = "missing shape for tensor: " + kv.first;
            return false;
        }

        std::vector<uint64_t> shape;

        if (!jsonArrayToU64(*shapeJson, shape)) {
            err = "invalid shape for tensor: " + kv.first;
            return false;
        }

        const JsonValue* offsetsJson = meta.find("data_offsets");

        if (
            offsetsJson == nullptr ||
            !offsetsJson->isArray() ||
            offsetsJson->array.size() != 2
        ) {
            err = "invalid data_offsets for tensor: " + kv.first;
            return false;
        }

        const JsonValue& beginJson = *offsetsJson->array[0];
        const JsonValue& endJson = *offsetsJson->array[1];

        if (
            !beginJson.isNumber() ||
            !endJson.isNumber() ||
            beginJson.number < 0.0 ||
            endJson.number < 0.0 ||
            beginJson.number > static_cast<double>(UINT64_MAX) ||
            endJson.number > static_cast<double>(UINT64_MAX)
        ) {
            err = "invalid data_offsets values for tensor: " + kv.first;
            return false;
        }

        uint64_t begin = static_cast<uint64_t>(beginJson.number);
        uint64_t end = static_cast<uint64_t>(endJson.number);

        if (end < begin) {
            err = "inverted data_offsets for tensor: " + kv.first;
            return false;
        }

        if (end > available) {
            err = "data_offsets out of bounds for tensor: " + kv.first;
            return false;
        }

        uint64_t needed = end - begin;
        uint64_t numel = tensorNumel(shape);

        if (needed != numel * 4ULL) {
            err = "data size mismatch for tensor: " + kv.first;
            return false;
        }

        StorageTensor t;
        t.name = kv.first;
        t.dtype = dtype;
        t.shape = shape;
        t.data.resize(static_cast<std::size_t>(numel));

        std::memcpy(
            t.data.data(),
            bytes.data() + dataStart + static_cast<std::size_t>(begin),
            static_cast<std::size_t>(needed)
        );

        tensors_.push_back(std::move(t));
    }

    err.clear();
    return true;
}

std::vector<std::string> SafetensorsReader::tensorNames() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());

    for (const StorageTensor& t : tensors_) {
        names.push_back(t.name);
    }

    return names;
}

StorageTensor SafetensorsReader::getTensor(const std::string& name) const {
    for (const StorageTensor& t : tensors_) {
        if (t.name == name) {
            return t;
        }
    }

    return StorageTensor();
}

const std::map<std::string, std::string>& SafetensorsReader::metadata() const {
    return metadata_;
}

bool saveCheckpoint(
    const std::string& modelDir,
    const CheckpointState& state,
    const std::map<std::string, StorageTensor>& weights,
    const std::map<std::string, StorageTensor>& optimizer,
    std::string& err
) {
    CheckpointState st = state;

    st.updatedAt = currentTimestamp();

    if (st.createdAt.empty()) {
        st.createdAt = st.updatedAt;
    }

    if (st.totalSteps >= st.step) {
        st.remainingSteps = st.totalSteps - st.step;
    } else {
        st.remainingSteps = 0;
    }

    std::string dir = pathJoin(modelDir, "checkpoint");

    if (!ensureDirectory(dir, err)) {
        return false;
    }

    SafetensorsWriter weightWriter;

    for (const auto& kv : weights) {
        StorageTensor t = kv.second;

        if (t.name.empty()) {
            t.name = kv.first;
        }

        weightWriter.addTensor(t);
    }

    if (!weightWriter.write(pathJoin(dir, "weights.safetensors"), err)) {
        return false;
    }

    std::string optimizerPath = pathJoin(dir, "optimizer.safetensors");

    if (optimizer.empty()) {
        removeIfExists(optimizerPath);
    } else {
        SafetensorsWriter optimizerWriter;

        for (const auto& kv : optimizer) {
            StorageTensor t = kv.second;

            if (t.name.empty()) {
                t.name = kv.first;
            }

            optimizerWriter.addTensor(t);
        }

        if (!optimizerWriter.write(optimizerPath, err)) {
            return false;
        }
    }

    JsonValue stateJson;
    checkpointStateToJson(st, stateJson);

    if (!writeFileString(
        pathJoin(dir, "state.json"),
        toJson(stateJson, 2),
        err
    )) {
        return false;
    }

    JsonValue event;
    event.type = JsonType::Object;

    jsonSetInt(event, "step", st.step);
    jsonSetInt(event, "remaining_steps", st.remainingSteps);
    jsonSetString(
        event,
        "reason",
        st.resumeReason.empty() ? std::string("checkpoint") : st.resumeReason
    );

    if (!appendResumeEvent(modelDir, "checkpoint_saved", event, err)) {
        return false;
    }

    err.clear();
    return true;
}

bool loadCheckpoint(
    const std::string& modelDir,
    CheckpointState& state,
    std::map<std::string, StorageTensor>& weights,
    std::map<std::string, StorageTensor>& optimizer,
    std::string& err
) {
    std::string dir = pathJoin(modelDir, "checkpoint");
    std::string statePath = pathJoin(dir, "state.json");

    std::string stateText;

    if (!readFileString(statePath, stateText, err)) {
        return false;
    }

    JsonValue root;

    if (!parseJson(stateText, root, err)) {
        return false;
    }

    if (!checkpointStateFromJson(root, state, err)) {
        return false;
    }

    weights.clear();

    std::string weightsPath = pathJoin(dir, "weights.safetensors");
    SafetensorsReader weightReader;

    if (!weightReader.load(weightsPath, err)) {
        return false;
    }

    for (const std::string& name : weightReader.tensorNames()) {
        weights.emplace(name, weightReader.getTensor(name));
    }

    optimizer.clear();

    std::string optimizerPath = pathJoin(dir, "optimizer.safetensors");

    if (pathExists(optimizerPath)) {
        SafetensorsReader optReader;

        if (!optReader.load(optimizerPath, err)) {
            return false;
        }

        for (const std::string& name : optReader.tensorNames()) {
            optimizer.emplace(name, optReader.getTensor(name));
        }
    }

    err.clear();
    return true;
}

bool appendResumeEvent(
    const std::string& modelDir,
    const std::string& event,
    const JsonValue& data,
    std::string& err
) {
    std::string dir = pathJoin(modelDir, "checkpoint");

    if (!ensureDirectory(dir, err)) {
        return false;
    }

    JsonValue line;
    line.type = JsonType::Object;

    jsonSetString(line, "ts", currentTimestamp());
    jsonSetString(line, "event", event);

    if (data.isObject() && !data.object.empty()) {
        jsonSetNested(line, "data", data);
    }

    std::string path = pathJoin(dir, "resume.log");

    std::ofstream out(path, std::ios::app);

    if (!out.is_open()) {
        err = "cannot open resume log: " + path;
        return false;
    }

    out << toJson(line, -1) << '\n';
    out.flush();

    if (!out.good()) {
        err = "failed writing resume log: " + path;
        return false;
    }

    err.clear();
    return true;
}

bool exportModelBundle(
    const std::string& sourceDir,
    const std::string& targetDir,
    const ExportOptions& options,
    const std::map<std::string, StorageTensor>& weights,
    const std::string& configJson,
    const std::string& tokenizerJson,
    std::string& err
) {
    if (targetDir.empty()) {
        err = "target directory is empty";
        return false;
    }

    if (pathExists(targetDir) && !options.overwrite) {
        err = "target model directory already exists: " + targetDir;
        return false;
    }

    if (!options.dtype.empty() && options.dtype != "F32") {
        err = "unsupported export dtype: " + options.dtype;
        return false;
    }

    if (!ensureDirectory(targetDir, err)) {
        return false;
    }

    SafetensorsWriter writer;
    writer.setMetadata("format", "nol-model");
    writer.setMetadata("architecture", "hybrid-hyperbolic-transformer");

    for (const auto& kv : weights) {
        if (options.stripEma && kv.first.find("ema_") != std::string::npos) {
            continue;
        }

        StorageTensor t = kv.second;

        if (t.name.empty()) {
            t.name = kv.first;
        }

        writer.addTensor(t);
    }

    std::string modelPath = pathJoin(targetDir, "model.safetensors");

    if (!writer.write(modelPath, err)) {
        return false;
    }

    std::vector<std::string> files;
    files.push_back("model.safetensors");

    if (!configJson.empty()) {
        std::string configPath = pathJoin(targetDir, "config.json");

        if (!writeFileString(configPath, configJson, err)) {
            return false;
        }

        files.push_back("config.json");
    }

    if (options.includeTokenizer && !tokenizerJson.empty()) {
        std::string tokenizerPath = pathJoin(targetDir, "tokenizer.json");

        if (!writeFileString(tokenizerPath, tokenizerJson, err)) {
            return false;
        }

        files.push_back("tokenizer.json");
    }

    if (options.includeOptimizer) {
        std::string optSource =
            pathJoin(pathJoin(sourceDir, "checkpoint"), "optimizer.safetensors");

        if (pathExists(optSource)) {
            std::vector<unsigned char> bytes;

            if (!readFileBytes(optSource, bytes, err)) {
                return false;
            }

            std::string optTarget = pathJoin(targetDir, "optimizer.safetensors");

            if (!writeFileBytesImpl(
                optTarget,
                bytes.data(),
                bytes.size(),
                err
            )) {
                return false;
            }

            files.push_back("optimizer.safetensors");
        }
    }

    JsonValue card;
    card.type = JsonType::Object;

    jsonSetString(card, "name", pathBase(targetDir));
    jsonSetString(card, "architecture", "ModelNol-Hybrid-Hyperbolic-Transformer");
    jsonSetString(card, "created_at", currentTimestamp());
    jsonSetString(card, "source_dir", sourceDir);

    if (!options.cardText.empty()) {
        jsonSetString(card, "description", options.cardText);
    }

    JsonValue fileList;
    fileList.type = JsonType::Array;

    for (const std::string& f : files) {
        jsonPushString(fileList, f);
    }

    jsonSetNested(card, "files", fileList);

    std::string cardPath = pathJoin(targetDir, "model-card.json");

    if (!writeFileString(cardPath, toJson(card, 2), err)) {
        return false;
    }

    err.clear();
    return true;
}

FileLock::FileLock(std::string path)
    : path_(std::move(path)) {}

FileLock::~FileLock() noexcept {
    release();
}

bool FileLock::acquire(std::string& err) {
    if (locked_) {
        return true;
    }

    if (pathExists(path_)) {
        err = "lock file exists: " + path_;
        return false;
    }

    std::string content = currentTimestamp() + "\n";

    if (!writeFileBytesImpl(
        path_,
        content.data(),
        content.size(),
        err
    )) {
        return false;
    }

    locked_ = true;
    err.clear();
    return true;
}

void FileLock::release() {
    if (locked_) {
        removeIfExists(path_);
        locked_ = false;
    }
}

bool FileLock::locked() const {
    return locked_;
}

#ifdef STORAGE_SELFTEST

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

    // 1. JSON parse and serialize.
    {
        JsonValue root;

        CHECK(parseJson(
            R"({"a":1,"b":"x\ny","c":[1,2],"d":true,"e":null})",
            root,
            err
        ));

        CHECK(root.isObject());
        CHECK(root.object.size() == 5u);

        const JsonValue* a = root.find("a");
        CHECK(a != nullptr);
        CHECK(a && a->isNumber());
        if (a) {
            CHECK(a->number == 1.0);
        }

        const JsonValue* b = root.find("b");
        CHECK(b != nullptr);
        CHECK(b && b->isString());
        if (b) {
            CHECK(b->str == "x\ny");
        }

        const JsonValue* c = root.find("c");
        CHECK(c != nullptr);
        CHECK(c && c->isArray());
        if (c) {
            CHECK(c->array.size() == 2u);
        }

        const JsonValue* d = root.find("d");
        CHECK(d != nullptr);
        CHECK(d && d->isBool());
        if (d) {
            CHECK(d->boolValue == true);
        }

        const JsonValue* e = root.find("e");
        CHECK(e != nullptr);
        CHECK(e && e->isNull());

        std::string compact = toJson(root, -1);
        JsonValue root2;

        CHECK(parseJson(compact, root2, err));
        CHECK(root2.isObject());
        CHECK(root2.object.size() == 5u);
    }

    // 2. Unicode escape.
    {
        JsonValue uni;

        CHECK(parseJson(R"({"u":"\u0041"})", uni, err));

        const JsonValue* u = uni.find("u");
        CHECK(u != nullptr);
        CHECK(u && u->isString());
        if (u) {
            CHECK(u->str == "A");
        }
    }

    // 3. Safetensors roundtrip.
    {
        StorageTensor t1;
        t1.name = "embed";
        t1.dtype = "F32";
        t1.shape = {2ULL, 3ULL};
        t1.data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };

        StorageTensor t2;
        t2.name = "norm";
        t2.dtype = "F32";
        t2.shape = {4ULL};
        t2.data = {0.1f, 0.2f, 0.3f, 0.4f};

        SafetensorsWriter writer;
        writer.addTensor(t1);
        writer.addTensor(t2);
        writer.setMetadata("format", "nol-model");

        std::string path = "storage_test.safetensors";

        CHECK(writer.write(path, err));

        SafetensorsReader reader;
        CHECK(reader.load(path, err));

        CHECK(reader.tensorNames().size() == 2u);

        StorageTensor rt1 = reader.getTensor("embed");
        CHECK(rt1.name == "embed");
        CHECK(rt1.shape.size() == 2u);

        if (rt1.shape.size() == 2u) {
            CHECK(rt1.shape[0] == 2ULL);
            CHECK(rt1.shape[1] == 3ULL);
        }

        CHECK(rt1.data.size() == t1.data.size());

        if (rt1.data.size() == t1.data.size()) {
            for (std::size_t i = 0; i < rt1.data.size(); ++i) {
                CHECK(rt1.data[i] == t1.data[i]);
            }
        }

        StorageTensor rt2 = reader.getTensor("norm");
        CHECK(rt2.name == "norm");
        CHECK(rt2.data.size() == t2.data.size());

        if (rt2.data.size() == t2.data.size()) {
            for (std::size_t i = 0; i < rt2.data.size(); ++i) {
                CHECK(rt2.data[i] == t2.data[i]);
            }
        }

        auto mit = reader.metadata().find("format");
        CHECK(mit != reader.metadata().end());

        if (mit != reader.metadata().end()) {
            CHECK(mit->second == "nol-model");
        }

        removeIfExists(path);
    }

    // 4. Checkpoint save/load.
    {
        std::string modelDir = "storage_test_model";

        CheckpointState st;
        st.modelName = "mawar";
        st.epoch = 1;
        st.step = 10;
        st.totalSteps = 100;
        st.stepsPerEpoch = 50;
        st.learningRate = 0.001f;
        st.globalSeed = 42;
        st.dataSeed = 7;
        st.resumeReason = "test";
        st.dirty = true;

        st.metrics.initialized = true;
        st.metrics.loss = 2.0f;
        st.metrics.lmLoss = 1.9f;
        st.metrics.acc = 0.3f;
        st.metrics.ppl = 7.389f;

        st.schedule.type = "cosine";
        st.schedule.warmupSteps = 5;
        st.schedule.totalSteps = 100;
        st.schedule.currentStep = 10;
        st.schedule.currentLr = 0.001f;

        StorageTensor t1;
        t1.name = "embed";
        t1.dtype = "F32";
        t1.shape = {2ULL, 3ULL};
        t1.data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };

        StorageTensor t2;
        t2.name = "adam.m";
        t2.dtype = "F32";
        t2.shape = {4ULL};
        t2.data = {0.1f, 0.2f, 0.3f, 0.4f};

        std::map<std::string, StorageTensor> weights;
        weights["embed"] = t1;

        std::map<std::string, StorageTensor> optimizer;
        optimizer["adam.m"] = t2;

        CHECK(saveCheckpoint(modelDir, st, weights, optimizer, err));

        CheckpointState st2;
        std::map<std::string, StorageTensor> w2;
        std::map<std::string, StorageTensor> o2;

        CHECK(loadCheckpoint(modelDir, st2, w2, o2, err));

        CHECK(st2.modelName == "mawar");
        CHECK(st2.epoch == 1);
        CHECK(st2.step == 10);
        CHECK(st2.totalSteps == 100);
        CHECK(st2.remainingSteps == 90);
        CHECK(st2.globalSeed == 42ULL);
        CHECK(st2.dataSeed == 7ULL);
        CHECK(st2.dirty == true);
        CHECK(st2.resumeReason == "test");

        CHECK(std::fabs(st2.learningRate - 0.001f) < 1e-6f);
        CHECK(std::fabs(st2.metrics.loss - 2.0f) < 1e-6f);
        CHECK(std::fabs(st2.metrics.acc - 0.3f) < 1e-6f);
        CHECK(st2.schedule.type == "cosine");
        CHECK(st2.schedule.totalSteps == 100);

        CHECK(w2.find("embed") != w2.end());
        CHECK(o2.find("adam.m") != o2.end());

        CHECK(pathExists(pathJoin(pathJoin(modelDir, "checkpoint"), "resume.log")));
    }

    // 5. Export bundle.
    {
        std::string modelDir = "storage_test_model";
        std::string exportDir = "storage_test_export";

        CheckpointState st2;
        std::map<std::string, StorageTensor> w2;
        std::map<std::string, StorageTensor> o2;

        CHECK(loadCheckpoint(modelDir, st2, w2, o2, err));

        ExportOptions options;
        options.overwrite = true;
        options.cardText = "unit test";

        CHECK(exportModelBundle(
            modelDir,
            exportDir,
            options,
            w2,
            "{\"ok\":true}",
            "{\"type\":\"byte\"}",
            err
        ));

        CHECK(pathExists(pathJoin(exportDir, "model.safetensors")));
        CHECK(pathExists(pathJoin(exportDir, "config.json")));
        CHECK(pathExists(pathJoin(exportDir, "tokenizer.json")));
        CHECK(pathExists(pathJoin(exportDir, "model-card.json")));
    }

    // 6. File lock.
    {
        std::string modelDir = "storage_test_model";
        std::string lockPath = pathJoin(modelDir, "LOCK");

        removeIfExists(lockPath);

        FileLock lock(lockPath);
        CHECK(lock.acquire(err));

        FileLock lock2(lockPath);
        CHECK(!lock2.acquire(err));

        lock.release();

        CHECK(lock2.acquire(err));
        lock2.release();

        removeIfExists(lockPath);
    }

    // Cleanup files, directories may remain.
    removeIfExists("storage_test.safetensors");
    removeIfExists(pathJoin(pathJoin("storage_test_model", "checkpoint"), "state.json"));
    removeIfExists(pathJoin(pathJoin("storage_test_model", "checkpoint"), "weights.safetensors"));
    removeIfExists(pathJoin(pathJoin("storage_test_model", "checkpoint"), "optimizer.safetensors"));
    removeIfExists(pathJoin(pathJoin("storage_test_model", "checkpoint"), "resume.log"));
    removeIfExists(pathJoin("storage_test_export", "model.safetensors"));
    removeIfExists(pathJoin("storage_test_export", "config.json"));
    removeIfExists(pathJoin("storage_test_export", "tokenizer.json"));
    removeIfExists(pathJoin("storage_test_export", "model-card.json"));

    if (failures == 0) {
        std::cout << "storage unit tests passed\n";
        return 0;
    }

    std::cerr << failures << " storage unit test(s) failed\n";
    return 1;
}

#endif // STORAGE_SELFTEST
