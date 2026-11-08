#include "../../include/token/tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <set>
#include <unordered_set>

namespace nol {
namespace token {

BPETokenizer::BPETokenizer() {
    // Initialize special tokens
    add_token("<pad>");  // 0
    add_token("<unk>");  // 1
    add_token("<bos>");  // 2
    add_token("<eos>");  // 3
}

void BPETokenizer::add_token(const std::string& token) {
    if (token_to_id_.find(token) == token_to_id_.end()) {
        int id = static_cast<int>(id_to_token_.size());
        id_to_token_.push_back(token);
        token_to_id_[token] = id;
    }
}

int BPETokenizer::lookup(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    return UNK_ID;
}

bool BPETokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] Cannot open vocab file: " << path << "\n";
        return false;
    }

    // Clear existing vocab
    id_to_token_.clear();
    token_to_id_.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            add_token(line);
        }
    }

    file.close();

    // Ensure special tokens exist
    if (vocab_size() < NUM_SPECIAL) {
        std::cerr << "[Tokenizer] Warning: vocab file has fewer than "
                  << NUM_SPECIAL << " entries. Special tokens may be missing.\n";
    }

    std::cout << "[Tokenizer] Loaded vocab: " << vocab_size() << " tokens from " << path << "\n";
    return true;
}

void BPETokenizer::build_vocab(const std::vector<std::string>& texts, int max_vocab_size) {
    // Clear and reinitialize with special tokens
    id_to_token_.clear();
    token_to_id_.clear();
    add_token("<pad>");
    add_token("<unk>");
    add_token("<bos>");
    add_token("<eos>");

    // Count character frequencies
    std::unordered_map<std::string, int> char_freq;
    for (const auto& text : texts) {
        for (char c : text) {
            std::string ch(1, c);
            char_freq[ch]++;
        }
    }

    // Sort by frequency
    std::vector<std::pair<std::string, int>> sorted_chars(char_freq.begin(), char_freq.end());
    std::sort(sorted_chars.begin(), sorted_chars.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Add single characters up to max_vocab_size
    int remaining = max_vocab_size - NUM_SPECIAL;
    for (const auto& [ch, freq] : sorted_chars) {
        if (static_cast<int>(id_to_token_.size()) >= max_vocab_size) break;
        add_token(ch);
    }

    // Add common bigrams if space remains
    if (static_cast<int>(id_to_token_.size()) < max_vocab_size) {
        std::unordered_map<std::string, int> bigram_freq;
        for (const auto& text : texts) {
            for (size_t i = 0; i + 1 < text.size(); ++i) {
                std::string bigram = text.substr(i, 2);
                bigram_freq[bigram]++;
            }
        }

        std::vector<std::pair<std::string, int>> sorted_bigrams(
            bigram_freq.begin(), bigram_freq.end());
        std::sort(sorted_bigrams.begin(), sorted_bigrams.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [bg, freq] : sorted_bigrams) {
            if (static_cast<int>(id_to_token_.size()) >= max_vocab_size) break;
            if (freq >= 2) { // Only add bigrams appearing at least twice
                add_token(bg);
            }
        }
    }

    std::cout << "[Tokenizer] Built vocab: " << vocab_size() << " tokens\n";
}

std::vector<int> BPETokenizer::encode(const std::string& text,
                                      bool add_bos, bool add_eos) const {
    std::vector<int> ids;

    if (add_bos) ids.push_back(BOS_ID);

    if (vocab_size() <= NUM_SPECIAL) {
        // No real vocab built yet, use byte-level fallback
        for (char c : text) {
            ids.push_back(UNK_ID);
        }
    } else {
        // Greedy longest-match encoding
        size_t i = 0;
        while (i < text.size()) {
            bool matched = false;

            // Try longest match first (up to 4 chars)
            for (int len = std::min(4, static_cast<int>(text.size() - i)); len >= 1; --len) {
                std::string substr = text.substr(i, len);
                int id = lookup(substr);
                if (id != UNK_ID || (len == 1 && id == UNK_ID)) {
                    if (id != UNK_ID) {
                        ids.push_back(id);
                    } else {
                        // Single char not in vocab → UNK
                        ids.push_back(UNK_ID);
                    }
                    i += len;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                ids.push_back(UNK_ID);
                i++;
            }
        }
    }

    if (add_eos) ids.push_back(EOS_ID);

    return ids;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        const std::string& token = id_to_token_[id];
        // Skip special tokens
        if (token == "<pad>" || token == "<unk>" || token == "<bos>" || token == "<eos>") {
            continue;
        }
        result += token;
    }
    return result;
}

bool BPETokenizer::save_vocab(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] Cannot save vocab to: " << path << "\n";
        return false;
    }

    for (const auto& token : id_to_token_) {
        file << token << "\n";
    }

    file.close();
    std::cout << "[Tokenizer] Saved vocab: " << vocab_size() << " tokens to " << path << "\n";
    return true;
}

} // namespace token
} // namespace nol

// ==========================================
// UNIT TEST TOKENIZER
// ==========================================
static void run_tokenizer_tests() {
    using namespace nol::token;
    std::cout << "[TEST] Memulai unit test tokenizer...\n";

    // Test 1: Default construction has special tokens
    {
        BPETokenizer tok;
        assert(tok.vocab_size() == 4);
        assert(tok.decode({0}) == "");   // <pad> skipped
        assert(tok.decode({1}) == "");   // <unk> skipped
        assert(tok.decode({2}) == "");   // <bos> skipped
        assert(tok.decode({3}) == "");   // <eos> skipped
        std::cout << "  Special tokens OK\n";
    }

    // Test 2: Build vocab from corpus
    {
        BPETokenizer tok;
        std::vector<std::string> corpus = {
            "hello world",
            "hello there",
            "world peace",
            "the quick brown fox",
            "hello hello hello"
        };
        tok.build_vocab(corpus, 100);

        assert(tok.vocab_size() > 4);
        assert(tok.vocab_size() <= 100);

        // Common chars should be in vocab
        std::vector<int> ids = tok.encode("hello");
        assert(!ids.empty());
        for (int id : ids) {
            assert(id >= 0 && id < tok.vocab_size());
        }
        std::cout << "  Build vocab OK (" << tok.vocab_size() << " tokens)\n";
    }

    // Test 3: Encode/decode roundtrip
    {
        BPETokenizer tok;
        std::vector<std::string> corpus = {"abcdef", "abcabc", "defdef"};
        tok.build_vocab(corpus, 50);

        std::string original = "abc";
        std::vector<int> ids = tok.encode(original);
        std::string decoded = tok.decode(ids);
        assert(decoded == original);
        std::cout << "  Encode/decode roundtrip OK\n";
    }

    // Test 4: BOS/EOS handling
    {
        BPETokenizer tok;
        std::vector<std::string> corpus = {"test"};
        tok.build_vocab(corpus, 50);

        auto ids_none = tok.encode("test", false, false);
        auto ids_bos = tok.encode("test", true, false);
        auto ids_eos = tok.encode("test", false, true);
        auto ids_both = tok.encode("test", true, true);

        assert(ids_bos.front() == BPETokenizer::BOS_ID);
        assert(ids_eos.back() == BPETokenizer::EOS_ID);
        assert(ids_both.front() == BPETokenizer::BOS_ID);
        assert(ids_both.back() == BPETokenizer::EOS_ID);
        assert(ids_bos.size() == ids_none.size() + 1);
        assert(ids_eos.size() == ids_none.size() + 1);
        assert(ids_both.size() == ids_none.size() + 2);
        std::cout << "  BOS/EOS OK\n";
    }

    // Test 5: Save/load vocab roundtrip
    {
        BPETokenizer tok1;
        std::vector<std::string> corpus = {"hello world test"};
        tok1.build_vocab(corpus, 50);

        assert(tok1.save_vocab("/tmp/test_vocab.txt"));

        BPETokenizer tok2;
        assert(tok2.load_vocab("/tmp/test_vocab.txt"));

        assert(tok1.vocab_size() == tok2.vocab_size());

        std::string text = "hello";
        auto ids1 = tok1.encode(text);
        auto ids2 = tok2.encode(text);
        assert(ids1 == ids2);

        std::remove("/tmp/test_vocab.txt");
        std::cout << "  Save/load vocab OK\n";
    }

    // Test 6: Unknown characters → UNK
    {
        BPETokenizer tok;
        std::vector<std::string> corpus = {"abc"};
        tok.build_vocab(corpus, 10);

        auto ids = tok.encode("xyz");
        for (int id : ids) {
            assert(id == BPETokenizer::UNK_ID);
        }
        std::cout << "  Unknown chars → UNK OK\n";
    }

    // Test 7: Empty string
    {
        BPETokenizer tok;
        auto ids = tok.encode("");
        assert(ids.empty());
        std::cout << "  Empty string OK\n";
    }

    std::cout << "[PASS] ✅ Semua unit test tokenizer berhasil!\n\n";
}

#ifdef NOL_TEST_TOKENIZER
int main() {
    run_tokenizer_tests();
    return 0;
}
#endif
