#include "tokenizer.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef TOKENIZER_SELFTEST
#include <iostream>
#endif

namespace {

constexpr int kSpecialCount = 4;

bool isReservedToken(const std::string& token) {
    return
        token == "<pad>" ||
        token == "<unk>" ||
        token == "<bos>" ||
        token == "<eos>";
}

std::string replacementUtf8() {
    std::string r;
    r.push_back(static_cast<char>(0xEF));
    r.push_back(static_cast<char>(0xBF));
    r.push_back(static_cast<char>(0xBD));
    return r;
}

std::vector<std::string> splitUtf8CodePoints(const std::string& s) {
    std::vector<std::string> out;

    std::size_t i = 0;

    while (i < s.size()) {
        unsigned char lead = static_cast<unsigned char>(s[i]);

        int len = 0;

        if (lead < 0x80U) {
            len = 1;
        } else if (lead >= 0xC2U && lead <= 0xDFU) {
            len = 2;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            len = 3;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            len = 4;
        } else {
            out.push_back(replacementUtf8());
            ++i;
            continue;
        }

        if (i + static_cast<std::size_t>(len) > s.size()) {
            out.push_back(replacementUtf8());
            ++i;
            continue;
        }

        bool ok = true;

        for (int j = 1; j < len; ++j) {
            unsigned char cont = static_cast<unsigned char>(s[i + static_cast<std::size_t>(j)]);

            if ((cont & 0xC0U) != 0x80U) {
                ok = false;
                break;
            }
        }

        if (!ok) {
            out.push_back(replacementUtf8());
            ++i;
            continue;
        }

        out.push_back(s.substr(i, static_cast<std::size_t>(len)));
        i += static_cast<std::size_t>(len);
    }

    return out;
}

bool readFileString(
    const std::string& path,
    std::string& out,
    std::string& err
) {
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
        err = "cannot open file: " + path;
        return false;
    }

    std::ostringstream ss;
    ss << f.rdbuf();

    if (f.bad()) {
        err = "failed reading file: " + path;
        return false;
    }

    out = ss.str();
    err.clear();
    return true;
}

bool writeFileString(
    const std::string& path,
    const std::string& content,
    std::string& err
) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);

    if (!f.is_open()) {
        err = "cannot open file for writing: " + path;
        return false;
    }

    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    f.flush();

    if (!f.good()) {
        err = "failed writing file: " + path;
        return false;
    }

    err.clear();
    return true;
}

void removeIfExists(const std::string& path) {
    (void)std::remove(path.c_str());
}

std::string escapeJsonString(const std::string& s) {
    std::string out;
    out.push_back('"');

    for (char ch : s) {
        unsigned char c = static_cast<unsigned char>(ch);

        switch (c) {
            case '"':
                out += "\\\"";
                break;

            case '\\':
                out += "\\\\";
                break;

            case '\b':
                out += "\\b";
                break;

            case '\f':
                out += "\\f";
                break;

            case '\n':
                out += "\\n";
                break;

            case '\r':
                out += "\\r";
                break;

            case '\t':
                out += "\\t";
                break;

            default:
                if (c < 0x20U) {
                    const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0x0fU];
                    out += hex[c & 0x0fU];
                } else {
                    out.push_back(static_cast<char>(c));
                }

                break;
        }
    }

    out.push_back('"');
    return out;
}

struct TinyJsonValue {
    enum class Type {
        Number,
        String,
        Array,
        Object
    };

    Type type = Type::Number;

    double number = 0.0;
    std::string str;

    std::vector<std::shared_ptr<TinyJsonValue>> array;
    std::map<std::string, std::shared_ptr<TinyJsonValue>> object;
};

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

void appendUtf8(std::string& out, std::uint32_t cp) {
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

class TinyJsonParser {
public:
    explicit TinyJsonParser(const std::string& text)
        : text_(text), pos_(0) {}

    bool parse(TinyJsonValue& out) {
        skipWs();

        if (!parseValue(out)) {
            return false;
        }

        skipWs();

        if (pos_ != text_.size()) {
            return fail("trailing characters after JSON value");
        }

        return true;
    }

    const std::string& error() const {
        return err_;
    }

private:
    const std::string& text_;
    std::size_t pos_;
    std::string err_;

    bool fail(const std::string& msg) {
        err_ = msg;
        return false;
    }

    void skipWs() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    bool peek(char& c) const {
        if (pos_ >= text_.size()) {
            return false;
        }

        c = text_[pos_];
        return true;
    }

    bool parseValue(TinyJsonValue& out) {
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
            out = TinyJsonValue();
            out.type = TinyJsonValue::Type::String;
            return parseStringRaw(out.str);
        }

        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parseNumber(out);
        }

        return fail("unsupported JSON value");
    }

    bool parseNumber(TinyJsonValue& out) {
        std::size_t start = pos_;

        if (pos_ < text_.size() && text_[pos_] == '-') {
            ++pos_;
        }

        if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            return fail("invalid number");
        }

        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }

        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;

            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                return fail("invalid fraction in number");
            }

            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }

        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;

            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }

            if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                return fail("invalid exponent in number");
            }

            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }

        std::string token = text_.substr(start, pos_ - start);
        char* end = nullptr;

        double value = std::strtod(token.c_str(), &end);

        if (
            end == nullptr ||
            static_cast<const char*>(end) != token.c_str() + token.size()
        ) {
            return fail("invalid numeric token");
        }

        if (!std::isfinite(value)) {
            return fail("non-finite number is not valid JSON");
        }

        out = TinyJsonValue();
        out.type = TinyJsonValue::Type::Number;
        out.number = value;
        return true;
    }

    bool parseHex4(std::uint32_t& out) {
        if (pos_ + 4 > text_.size()) {
            return fail("truncated unicode escape");
        }

        std::uint32_t value = 0;

        for (int i = 0; i < 4; ++i) {
            int d = hexDigit(text_[pos_++]);

            if (d < 0) {
                return fail("invalid hexadecimal digit in unicode escape");
            }

            value = (value << 4U) | static_cast<std::uint32_t>(d);
        }

        out = value;
        return true;
    }

    bool parseStringRaw(std::string& out) {
        out.clear();

        if (pos_ >= text_.size() || text_[pos_] != '"') {
            return fail("expected string");
        }

        ++pos_;

        while (true) {
            if (pos_ >= text_.size()) {
                return fail("unterminated string");
            }

            unsigned char uc = static_cast<unsigned char>(text_[pos_++]);

            if (uc == '"') {
                return true;
            }

            if (uc == '\\') {
                if (pos_ >= text_.size()) {
                    return fail("unterminated escape sequence");
                }

                char e = text_[pos_++];

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
                        std::uint32_t cp = 0;

                        if (!parseHex4(cp)) {
                            return false;
                        }

                        if (cp >= 0xD800U && cp <= 0xDFFFU) {
                            appendUtf8(out, 0xFFFDU);
                        } else {
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

    bool parseObject(TinyJsonValue& out) {
        out = TinyJsonValue();
        out.type = TinyJsonValue::Type::Object;

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

            TinyJsonValue value;

            if (!parseValue(value)) {
                return false;
            }

            auto ptr = std::make_shared<TinyJsonValue>();
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

    bool parseArray(TinyJsonValue& out) {
        out = TinyJsonValue();
        out.type = TinyJsonValue::Type::Array;

        ++pos_;
        skipWs();

        char c = 0;

        if (peek(c) && c == ']') {
            ++pos_;
            return true;
        }

        while (true) {
            skipWs();

            TinyJsonValue value;

            if (!parseValue(value)) {
                return false;
            }

            auto ptr = std::make_shared<TinyJsonValue>();
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

const TinyJsonValue* findMember(
    const TinyJsonValue& obj,
    const std::string& key
) {
    if (obj.type != TinyJsonValue::Type::Object) {
        return nullptr;
    }

    auto it = obj.object.find(key);

    if (it == obj.object.end()) {
        return nullptr;
    }

    return it->second.get();
}

bool parseTinyJson(
    const std::string& text,
    TinyJsonValue& out,
    std::string& err
) {
    TinyJsonParser parser(text);
    out = TinyJsonValue();

    if (!parser.parse(out)) {
        err = parser.error();
        return false;
    }

    err.clear();
    return true;
}

} // namespace

Tokenizer::Tokenizer() {
    initByte();
}

Tokenizer Tokenizer::makeByte() {
    Tokenizer t;
    t.initByte();
    return t;
}

void Tokenizer::initByte() {
    kind_ = Kind::Byte;

    idToToken_.clear();
    idToToken_.push_back("<pad>");
    idToToken_.push_back("<unk>");
    idToToken_.push_back("<bos>");
    idToToken_.push_back("<eos>");

    padId_ = 0;
    unkId_ = 1;
    bosId_ = 2;
    eosId_ = 3;
    firstByteId_ = kSpecialCount;

    merges_.clear();
    rebuildIndex();
}

void Tokenizer::rebuildIndex() {
    tokenToId_.clear();

    for (std::size_t i = 0; i < idToToken_.size(); ++i) {
        tokenToId_[idToToken_[i]] = static_cast<int>(i);
    }

    mergeRank_.clear();

    for (std::size_t i = 0; i < merges_.size(); ++i) {
        const std::pair<std::string, std::string>& p = merges_[i];

        if (mergeRank_.find(p) == mergeRank_.end()) {
            mergeRank_[p] = static_cast<int>(i);
        }
    }
}

int Tokenizer::vocabSize() const {
    if (kind_ == Kind::Byte) {
        return firstByteId_ + 256;
    }

    return static_cast<int>(idToToken_.size());
}

bool Tokenizer::isSpecialId(int id) const {
    return
        id == padId_ ||
        id == unkId_ ||
        id == bosId_ ||
        id == eosId_;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;

    if (kind_ == Kind::Byte) {
        out.reserve(text.size());

        for (std::size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            out.push_back(firstByteId_ + static_cast<int>(c));
        }

        return out;
    }

    std::vector<std::string> seq = splitUtf8CodePoints(text);

    while (true) {
        bool found = false;
        int bestRank = std::numeric_limits<int>::max();
        std::pair<std::string, std::string> bestPair;

        for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
            std::pair<std::string, std::string> pair = std::make_pair(seq[i], seq[i + 1]);
            auto it = mergeRank_.find(pair);

            if (it != mergeRank_.end()) {
                if (!found || it->second < bestRank) {
                    found = true;
                    bestRank = it->second;
                    bestPair = pair;
                }
            }
        }

        if (!found) {
            break;
        }

        std::string merged = bestPair.first + bestPair.second;
        std::vector<std::string> next;
        next.reserve(seq.size());

        for (std::size_t i = 0; i < seq.size();) {
            if (
                i + 1 < seq.size() &&
                seq[i] == bestPair.first &&
                seq[i + 1] == bestPair.second
            ) {
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(seq[i]);
                ++i;
            }
        }

        if (next.size() == seq.size()) {
            break;
        }

        seq.swap(next);
    }

    out.reserve(seq.size());

    for (const std::string& token : seq) {
        auto it = tokenToId_.find(token);

        if (it != tokenToId_.end()) {
            out.push_back(it->second);
        } else {
            out.push_back(unkId_);
        }
    }

    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;

    if (kind_ == Kind::Byte) {
        out.reserve(ids.size());

        for (int id : ids) {
            if (id >= firstByteId_ && id < firstByteId_ + 256) {
                unsigned char c = static_cast<unsigned char>(id - firstByteId_);
                out.push_back(static_cast<char>(c));
            }
        }

        return out;
    }

    int vs = vocabSize();

    for (int id : ids) {
        if (id < 0 || id >= vs) {
            continue;
        }

        if (isSpecialId(id)) {
            continue;
        }

        out += idToToken_[static_cast<std::size_t>(id)];
    }

    return out;
}

bool Tokenizer::trainFromText(
    const std::string& corpusPath,
    int targetVocabSize,
    std::uint64_t seed,
    std::string& err
) {
    (void)seed;

    std::string corpus;

    if (!readFileString(corpusPath, corpus, err)) {
        return false;
    }

    if (targetVocabSize < kSpecialCount) {
        err = "target vocab size must be at least " + std::to_string(kSpecialCount);
        return false;
    }

    std::vector<std::string> symbols = splitUtf8CodePoints(corpus);

    std::set<std::string> unique;

    for (const std::string& s : symbols) {
        if (!isReservedToken(s)) {
            unique.insert(s);
        }
    }

    int maxTokens = targetVocabSize - kSpecialCount;

    if (maxTokens < 0) {
        err = "target vocab size is too small";
        return false;
    }

    if (unique.size() > static_cast<std::size_t>(maxTokens)) {
        err =
            "target vocab size is too small to represent all Unicode code points in corpus; "
            "use byte tokenizer or increase vocab size";
        return false;
    }

    kind_ = Kind::Bpe;
    firstByteId_ = kSpecialCount;

    idToToken_.clear();
    idToToken_.reserve(static_cast<std::size_t>(kSpecialCount) + unique.size());

    idToToken_.push_back("<pad>");
    idToToken_.push_back("<unk>");
    idToToken_.push_back("<bos>");
    idToToken_.push_back("<eos>");

    padId_ = 0;
    unkId_ = 1;
    bosId_ = 2;
    eosId_ = 3;

    for (const std::string& token : unique) {
        idToToken_.push_back(token);
    }

    merges_.clear();
    rebuildIndex();

    std::vector<std::string> seq = symbols;

    while (static_cast<int>(idToToken_.size()) < targetVocabSize) {
        std::map<std::pair<std::string, std::string>, long long> counts;

        for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
            std::pair<std::string, std::string> pair = std::make_pair(seq[i], seq[i + 1]);
            ++counts[pair];
        }

        if (counts.empty()) {
            break;
        }

        bool found = false;
        long long bestCount = 0;
        std::pair<std::string, std::string> bestPair;

        for (const auto& kv : counts) {
            std::string merged = kv.first.first + kv.first.second;

            if (isReservedToken(merged)) {
                continue;
            }

            if (
                !found ||
                kv.second > bestCount ||
                (kv.second == bestCount && kv.first < bestPair)
            ) {
                found = true;
                bestCount = kv.second;
                bestPair = kv.first;
            }
        }

        if (!found) {
            break;
        }

        std::string merged = bestPair.first + bestPair.second;

        if (tokenToId_.find(merged) == tokenToId_.end()) {
            idToToken_.push_back(merged);
        }

        if (mergeRank_.find(bestPair) == mergeRank_.end()) {
            merges_.push_back(bestPair);
        }

        rebuildIndex();

        std::vector<std::string> next;
        next.reserve(seq.size());

        for (std::size_t i = 0; i < seq.size();) {
            if (
                i + 1 < seq.size() &&
                seq[i] == bestPair.first &&
                seq[i + 1] == bestPair.second
            ) {
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(seq[i]);
                ++i;
            }
        }

        if (next.size() == seq.size()) {
            break;
        }

        seq.swap(next);
    }

    err.clear();
    return true;
}

bool Tokenizer::saveJson(
    const std::string& path,
    std::string& err
) const {
    std::ostringstream os;

    os << "{\n";
    os << "  \"kind\": \"" << (kind_ == Kind::Byte ? "byte" : "bpe") << "\",\n";
    os << "  \"version\": 1,\n";
    os << "  \"special_tokens\": {";
    os << "\"<pad>\":" << padId_ << ",";
    os << "\"<unk>\":" << unkId_ << ",";
    os << "\"<bos>\":" << bosId_ << ",";
    os << "\"<eos>\":" << eosId_;
    os << "},\n";
    os << "  \"first_byte_id\": " << firstByteId_ << ",\n";

    os << "  \"vocab\": [";

    std::size_t vocabLimit = idToToken_.size();

    if (kind_ == Kind::Byte) {
        vocabLimit = static_cast<std::size_t>(kSpecialCount);
    }

    for (std::size_t i = 0; i < vocabLimit; ++i) {
        if (i != 0) {
            os << ",";
        }

        os << "\n    " << escapeJsonString(idToToken_[i]);
    }

    if (vocabLimit != 0) {
        os << "\n  ";
    }

    os << "],\n";

    os << "  \"merges\": [";

    for (std::size_t i = 0; i < merges_.size(); ++i) {
        if (i != 0) {
            os << ",";
        }

        os << "\n    [";
        os << escapeJsonString(merges_[i].first);
        os << ",";
        os << escapeJsonString(merges_[i].second);
        os << "]";
    }

    if (!merges_.empty()) {
        os << "\n  ";
    }

    os << "]\n";
    os << "}\n";

    return writeFileString(path, os.str(), err);
}

bool Tokenizer::loadJson(
    const std::string& path,
    Tokenizer& out,
    std::string& err
) {
    out = Tokenizer();

    std::string text;

    if (!readFileString(path, text, err)) {
        return false;
    }

    TinyJsonValue root;

    if (!parseTinyJson(text, root, err)) {
        return false;
    }

    if (root.type != TinyJsonValue::Type::Object) {
        err = "tokenizer JSON root must be an object";
        return false;
    }

    const TinyJsonValue* kindV = findMember(root, "kind");

    if (kindV == nullptr || kindV->type != TinyJsonValue::Type::String) {
        err = "missing or invalid tokenizer kind";
        return false;
    }

    std::string kind = kindV->str;

    const TinyJsonValue* firstByteV = findMember(root, "first_byte_id");
    int firstByteId = kSpecialCount;

    if (firstByteV != nullptr) {
        if (firstByteV->type != TinyJsonValue::Type::Number) {
            err = "first_byte_id must be a number";
            return false;
        }

        firstByteId = static_cast<int>(firstByteV->number);

        if (firstByteId < kSpecialCount) {
            err = "first_byte_id is too small";
            return false;
        }
    }

    const TinyJsonValue* vocabV = findMember(root, "vocab");

    if (vocabV == nullptr || vocabV->type != TinyJsonValue::Type::Array) {
        err = "missing or invalid tokenizer vocab";
        return false;
    }

    const TinyJsonValue* mergesV = findMember(root, "merges");

    if (mergesV == nullptr || mergesV->type != TinyJsonValue::Type::Array) {
        err = "missing or invalid tokenizer merges";
        return false;
    }

    if (kind == "byte") {
        out.initByte();
        out.firstByteId_ = firstByteId;
        err.clear();
        return true;
    }

    if (kind != "bpe") {
        err = "unsupported tokenizer kind: " + kind;
        return false;
    }

    out.kind_ = Kind::Bpe;
    out.firstByteId_ = firstByteId;
    out.idToToken_.clear();

    for (const auto& ptr : vocabV->array) {
        if (!ptr || ptr->type != TinyJsonValue::Type::String) {
            err = "tokenizer vocab entries must be strings";
            return false;
        }

        out.idToToken_.push_back(ptr->str);
    }

    if (out.idToToken_.size() < static_cast<std::size_t>(kSpecialCount)) {
        err = "tokenizer vocab must contain special tokens";
        return false;
    }

    if (
        out.idToToken_[0] != "<pad>" ||
        out.idToToken_[1] != "<unk>" ||
        out.idToToken_[2] != "<bos>" ||
        out.idToToken_[3] != "<eos>"
    ) {
        err = "tokenizer special tokens must be <pad>, <unk>, <bos>, <eos>";
        return false;
    }

    out.padId_ = 0;
    out.unkId_ = 1;
    out.bosId_ = 2;
    out.eosId_ = 3;

    out.merges_.clear();

    for (const auto& mergePtr : mergesV->array) {
        if (
            !mergePtr ||
            mergePtr->type != TinyJsonValue::Type::Array ||
            mergePtr->array.size() != 2
        ) {
            err = "each merge must be an array of two strings";
            return false;
        }

        const TinyJsonValue* a = mergePtr->array[0].get();
        const TinyJsonValue* b = mergePtr->array[1].get();

        if (
            a == nullptr || a->type != TinyJsonValue::Type::String ||
            b == nullptr || b->type != TinyJsonValue::Type::String
        ) {
            err = "merge entries must be strings";
            return false;
        }

        out.merges_.push_back(std::make_pair(a->str, b->str));
    }

    out.rebuildIndex();
    err.clear();
    return true;
}

#ifdef TOKENIZER_SELFTEST

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

    // 1. Byte tokenizer basic roundtrip.
    {
        Tokenizer tok = Tokenizer::makeByte();

        CHECK(tok.kind() == Tokenizer::Kind::Byte);
        CHECK(tok.vocabSize() == 260);
        CHECK(tok.padId() == 0);
        CHECK(tok.unkId() == 1);
        CHECK(tok.bosId() == 2);
        CHECK(tok.eosId() == 3);

        std::vector<int> ids = tok.encode("abc");

        CHECK(ids.size() == 3);
        CHECK(tok.decode(ids) == "abc");
    }

    // 2. Byte tokenizer UTF-8 roundtrip.
    {
        Tokenizer tok = Tokenizer::makeByte();

        std::string text = "héllo 🌍";
        std::vector<int> ids = tok.encode(text);

        CHECK(tok.decode(ids) == text);
    }

    // 3. Byte tokenizer save/load.
    {
        Tokenizer tok = Tokenizer::makeByte();

        CHECK(tok.saveJson("tokenizer_test_byte.json", err));

        Tokenizer loaded;
        CHECK(Tokenizer::loadJson("tokenizer_test_byte.json", loaded, err));

        CHECK(loaded.kind() == Tokenizer::Kind::Byte);
        CHECK(loaded.vocabSize() == 260);

        std::vector<int> ids = loaded.encode("hello");
        CHECK(loaded.decode(ids) == "hello");
    }

    // 4. BPE training and roundtrip.
    {
        std::string corpusPath = "tokenizer_test_corpus.txt";
        std::string corpus =
            "aaabbbccc aaabbbccc bbbc ccc aaa\n"
            "abc abc abc bca cab\n";

        CHECK(writeFileString(corpusPath, corpus, err));

        Tokenizer tok;
        CHECK(tok.trainFromText(corpusPath, 24, 123, err));

        CHECK(tok.kind() == Tokenizer::Kind::Bpe);
        CHECK(tok.vocabSize() <= 24);
        CHECK(tok.vocabSize() >= kSpecialCount);

        std::string probe = "aaabbbccc";
        std::vector<int> ids = tok.encode(probe);
        std::string dec = tok.decode(ids);

        CHECK(dec == probe);
    }

    // 5. BPE save/load roundtrip.
    {
        Tokenizer tok;
        std::string corpusPath = "tokenizer_test_corpus.txt";

        CHECK(tok.trainFromText(corpusPath, 24, 123, err));
        CHECK(tok.saveJson("tokenizer_test_bpe.json", err));

        Tokenizer loaded;
        CHECK(Tokenizer::loadJson("tokenizer_test_bpe.json", loaded, err));

        CHECK(loaded.kind() == Tokenizer::Kind::Bpe);
        CHECK(loaded.vocabSize() == tok.vocabSize());

        std::string probe = "abc bca";
        std::vector<int> ids = loaded.encode(probe);
        CHECK(loaded.decode(ids) == probe);
    }

    // 6. Unknown token fallback to <unk>.
    {
        std::string corpusPath = "tokenizer_test_small_corpus.txt";
        CHECK(writeFileString(corpusPath, "ab ab ab\n", err));

        Tokenizer tok;
        CHECK(tok.trainFromText(corpusPath, 12, 1, err));

        std::vector<int> ids = tok.encode("z");

        CHECK(ids.size() == 1);
        CHECK(ids[0] == tok.unkId());
    }

    // 7. Invalid JSON load must fail.
    {
        std::string badPath = "tokenizer_test_bad.json";
        CHECK(writeFileString(badPath, "not json", err));

        Tokenizer loaded;
        CHECK(!Tokenizer::loadJson(badPath, loaded, err));
        CHECK(!err.empty());
    }

    // 8. Missing file load must fail.
    {
        Tokenizer loaded;
        CHECK(!Tokenizer::loadJson("tokenizer_does_not_exist.json", loaded, err));
        CHECK(!err.empty());
    }

    removeIfExists("tokenizer_test_byte.json");
    removeIfExists("tokenizer_test_bpe.json");
    removeIfExists("tokenizer_test_corpus.txt");
    removeIfExists("tokenizer_test_small_corpus.txt");
    removeIfExists("tokenizer_test_bad.json");

    if (failures == 0) {
        std::cout << "tokenizer unit tests passed\n";
        return 0;
    }

    std::cerr << failures << " tokenizer unit test(s) failed\n";
    return 1;
}

#endif // TOKENIZER_SELFTEST
