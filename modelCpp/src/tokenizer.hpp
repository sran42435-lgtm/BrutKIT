#ifndef PROYEK_TOKENIZER_HPP
#define PROYEK_TOKENIZER_HPP

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

class Tokenizer {
public:
    enum class Kind {
        Byte,
        Bpe
    };

    Tokenizer();

    static Tokenizer makeByte();

    Kind kind() const {
        return kind_;
    }

    int vocabSize() const;

    int padId() const {
        return padId_;
    }

    int unkId() const {
        return unkId_;
    }

    int bosId() const {
        return bosId_;
    }

    int eosId() const {
        return eosId_;
    }

    bool isSpecialId(int id) const;

    std::vector<int> encode(const std::string& text) const;

    std::string decode(const std::vector<int>& ids) const;

    bool trainFromText(
        const std::string& corpusPath,
        int targetVocabSize,
        std::uint64_t seed,
        std::string& err
    );

    bool saveJson(
        const std::string& path,
        std::string& err
    ) const;

    static bool loadJson(
        const std::string& path,
        Tokenizer& out,
        std::string& err
    );

private:
    void initByte();

    void rebuildIndex();

    Kind kind_;

    std::vector<std::string> idToToken_;

    std::map<std::string, int> tokenToId_;

    std::vector<std::pair<std::string, std::string>> merges_;

    std::map<std::pair<std::string, std::string>, int> mergeRank_;

    int padId_;
    int unkId_;
    int bosId_;
    int eosId_;
    int firstByteId_;
};

#endif // PROYEK_TOKENIZER_HPP
