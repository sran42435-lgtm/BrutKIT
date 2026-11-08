#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace nol {
namespace token {

/**
 * @brief BPE Tokenizer untuk Model Nol.
 * 
 * Fitur:
 * - Byte-level BPE (fallback ke byte jika char tidak ada di vocab)
 * - Encode: string → vector<int>
 * - Decode: vector<int> → string
 * - Load vocab dari file (format: satu token per baris)
 * - Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
 * - Jika vocab file tidak ada, generate vocab sederhana (char-level + common pairs)
 */
class BPETokenizer {
public:
    static constexpr int PAD_ID = 0;
    static constexpr int UNK_ID = 1;
    static constexpr int BOS_ID = 2;
    static constexpr int EOS_ID = 3;
    static constexpr int NUM_SPECIAL = 4;

    BPETokenizer();

    /**
     * @brief Load vocabulary dari file.
     * Format: satu token per baris, baris ke-N = token ID N.
     * @param path Path ke file vocab
     * @return true jika berhasil
     */
    bool load_vocab(const std::string& path);

    /**
     * @brief Generate vocab sederhana dari corpus teks.
     * Char-level tokens + special tokens.
     * @param texts Corpus teks untuk membangun vocab
     * @param max_vocab_size Ukuran maksimum vocabulary
     */
    void build_vocab(const std::vector<std::string>& texts, int max_vocab_size);

    /**
     * @brief Encode string ke token IDs.
     * @param text Input string
     * @param add_bos Tambahkan <bos> di awal
     * @param add_eos Tambahkan <eos> di akhir
     * @return Vector token IDs
     */
    std::vector<int> encode(const std::string& text,
                            bool add_bos = false, bool add_eos = false) const;

    /**
     * @brief Decode token IDs ke string.
     * @param ids Token IDs
     * @return Decoded string
     */
    std::string decode(const std::vector<int>& ids) const;

    /**
     * @brief Get vocabulary size.
     */
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

    /**
     * @brief Save vocab ke file.
     */
    bool save_vocab(const std::string& path) const;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    void add_token(const std::string& token);
    int lookup(const std::string& token) const;
};

} // namespace token
} // namespace nol
