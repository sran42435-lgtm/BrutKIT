# core/router.py
#
# Auto Domain Router + Semantic Search Engine.
#
# Tugas utama:
# 1. Extract keywords per domain (hybrid: frequency × uniqueness)
# 2. Score prompt ke domain berdasarkan keywords
# 3. Semantic search: cosine similarity dengan cached embeddings
# 4. Hybrid routing: keyword-first, semantic fallback
# 5. Persistence: save/load keywords + embeddings ke disk
#
# Pendekatan:
# - Pure NumPy (tidak butuh FAISS/ChromaDB)
# - Lightweight untuk deployment di CPU
# - Portabel (ikut di-bundle dengan safetensors)
#
# Keterhubungan:
# - config.py           : parameter MoE
# - core/architecture.py: get_hidden_states() untuk embeddings
# - core/tokenizer.py   : tokenisasi prompt untuk keyword matching
# - testing/playground.py: auto-routing saat inference

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================================
# NUMPY IMPORT
# ============================================================================

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "core/router.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config


# ============================================================================
# STOPWORDS (Indonesia + Inggris)
# ============================================================================
# Kata-kata umum yang tidak berguna untuk routing karena muncul di semua domain.

STOPWORDS: frozenset = frozenset({
    # Indonesia
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "oleh",
    "adalah", "adalah", "ini", "itu", "atau", "juga", "sebagai", "serta",
    "karena", "sehingga", "ketika", "saat", "jika", "agar", "supaya",
    "tetapi", "namun", "akan", "telah", "sudah", "belum", "tidak", "bukan",
    "sangat", "lebih", "paling", "saja", "hanya", "dapat", "bisa",
    "memiliki", "merupakan", "menjadi", "melakukan", "menggunakan",
    "a", "an", "the", "adalah", "si", "sang", "para", "kaum",
    # Inggris
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "can", "may",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "that", "this", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "him", "her", "his", "we", "us", "our", "you", "your",
    "not", "no", "yes", "all", "any", "some", "many", "much", "more",
    # Angka dan simbol
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
})


# ============================================================================
# TEXT HELPERS
# ============================================================================

def _simple_tokenize(text: str) -> List[str]:
    """
    Tokenisasi sederhana: split on non-alphanumeric, lowercase.

    Contoh:
        "Mobil adalah KENDARAAN!" → ["mobil", "adalah", "kendaraan"]
    """
    text = text.lower()
    # Ganti semua non-alphanumeric dengan spasi
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Hitung cosine similarity antara dua vektor.

    Return:
    - float dalam rentang [-1, 1], atau 0.0 jika salah satu vektor nol
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()

    if a.shape != b.shape:
        raise ValueError(
            f"Shape tidak cocok: a={a.shape}, b={b.shape}"
        )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    dot = np.dot(a, b)
    return float(dot / (norm_a * norm_b))


def _cosine_similarity_matrix(
    query: np.ndarray,
    corpus: np.ndarray,
) -> np.ndarray:
    """
    Hitung cosine similarity antara 1 query dengan banyak corpus embeddings.

    Input:
    - query: [D]
    - corpus: [N, D]

    Output:
    - similarities: [N]
    """
    query = np.asarray(query, dtype=np.float32).flatten()
    corpus = np.asarray(corpus, dtype=np.float32)

    if corpus.ndim != 2:
        raise ValueError(f"Corpus harus 2D, diterima {corpus.ndim}D")

    N, D = corpus.shape

    if query.shape[0] != D:
        raise ValueError(
            f"Dimensi tidak cocok: query={query.shape}, corpus={corpus.shape}"
        )

    # Normalize
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(N, dtype=np.float32)
    query = query / query_norm

    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus_norms = np.where(corpus_norms < 1e-10, 1.0, corpus_norms)
    corpus_normalized = corpus / corpus_norms

    similarities = corpus_normalized @ query
    return similarities.astype(np.float32)


# ============================================================================
# KEYWORD EXTRACTOR
# ============================================================================

class KeywordExtractor:
    """
    Ekstrak keywords dari korpus untuk setiap domain.

    Metode yang didukung:
    - "frequency" : top-N kata paling frequent di domain
    - "tfidf"     : TF-IDF (kata yang unik untuk domain ini)
    - "hybrid"    : frequency × uniqueness score (rekomendasi)
    """

    def __init__(self, method: str = "hybrid"):
        if method not in {"frequency", "tfidf", "hybrid"}:
            raise ValueError(
                f"Method tidak valid: {method}. "
                "Pilihan: 'frequency', 'tfidf', 'hybrid'"
            )
        self.method = method

    def extract(
        self,
        corpus_by_domain: Dict[str, List[str]],
        top_k: int = 30,
    ) -> Dict[str, List[str]]:
        """
        Ekstrak keywords untuk setiap domain.

        Input:
        - corpus_by_domain: dict {domain: list of paragraphs}

        Output:
        - dict {domain: list of top-k keywords}
        """
        if not corpus_by_domain:
            return {}

        if self.method == "frequency":
            return self._extract_by_frequency(corpus_by_domain, top_k)
        elif self.method == "tfidf":
            return self._extract_by_tfidf(corpus_by_domain, top_k)
        else:  # hybrid
            return self._extract_hybrid(corpus_by_domain, top_k)

    def _extract_by_frequency(
        self,
        corpus_by_domain: Dict[str, List[str]],
        top_k: int,
    ) -> Dict[str, List[str]]:
        """Paling simple: top-k kata paling frequent per domain."""
        result = {}

        for domain, paragraphs in corpus_by_domain.items():
            counter = Counter()
            for para in paragraphs:
                counter.update(_simple_tokenize(para))

            top_words = [w for w, _ in counter.most_common(top_k)]
            result[domain] = top_words

        return result

    def _extract_by_tfidf(
        self,
        corpus_by_domain: Dict[str, List[str]],
        top_k: int,
    ) -> Dict[str, List[str]]:
        """
        TF-IDF sederhana:
        - TF = frequency dalam domain
        - IDF = log(total_domains / domain_count)
        """
        domains = list(corpus_by_domain.keys())
        total_domains = len(domains)

        # Hitung document frequency (di berapa domain kata muncul)
        doc_freq: Counter = Counter()
        for domain, paragraphs in corpus_by_domain.items():
            words_in_domain = set()
            for para in paragraphs:
                words_in_domain.update(_simple_tokenize(para))
            doc_freq.update(words_in_domain)

        # Hitung TF-IDF per domain
        result = {}
        for domain, paragraphs in corpus_by_domain.items():
            tf_counter = Counter()
            for para in paragraphs:
                tf_counter.update(_simple_tokenize(para))

            tfidf_scores = {}
            for word, tf in tf_counter.items():
                df = doc_freq.get(word, 1)
                idf = math.log(total_domains / df)
                tfidf_scores[word] = tf * idf

            # Sort by score descending
            sorted_words = sorted(
                tfidf_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            top_words = [w for w, _ in sorted_words[:top_k]]
            result[domain] = top_words

        return result

    def _extract_hybrid(
        self,
        corpus_by_domain: Dict[str, List[str]],
        top_k: int,
    ) -> Dict[str, List[str]]:
        """
        Hybrid: frequency × uniqueness.

        Score = log(1 + tf) × (total_domains / df)

        Kata yang frequent DAN unik di satu domain dapat score tinggi.
        Kata yang frequent tapi muncul di semua domain (stopword) score rendah.
        """
        domains = list(corpus_by_domain.keys())
        total_domains = len(domains)

        # Hitung document frequency
        doc_freq: Counter = Counter()
        for domain, paragraphs in corpus_by_domain.items():
            words_in_domain = set()
            for para in paragraphs:
                words_in_domain.update(_simple_tokenize(para))
            doc_freq.update(words_in_domain)

        # Hitung hybrid score per domain
        result = {}
        for domain, paragraphs in corpus_by_domain.items():
            tf_counter = Counter()
            for para in paragraphs:
                tf_counter.update(_simple_tokenize(para))

            hybrid_scores = {}
            for word, tf in tf_counter.items():
                df = doc_freq.get(word, 1)
                uniqueness = total_domains / df  # tinggi jika unik
                score = math.log(1 + tf) * uniqueness
                hybrid_scores[word] = score

            sorted_words = sorted(
                hybrid_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            top_words = [w for w, _ in sorted_words[:top_k]]
            result[domain] = top_words

        return result


# ============================================================================
# SEMANTIC SEARCH ENGINE
# ============================================================================

class SemanticSearchEngine:
    """
    Semantic search berbasis cosine similarity dengan cached embeddings.

    Cache structure:
    - domain_embeddings: {domain: np.ndarray [N, D]}
    - domain_paragraphs: {domain: list of paragraphs}
    """

    def __init__(self):
        self.domain_embeddings: Dict[str, np.ndarray] = {}
        self.domain_paragraphs: Dict[str, List[str]] = {}

    def build_cache(
        self,
        model,
        tokenizer,
        corpus_by_domain: Dict[str, List[str]],
        pooling: str = "mean",
        verbose: bool = True,
    ) -> None:
        """
        Bangun cache embeddings dari semua paragraf per domain.

        Parameter:
        - model: CustomTransformerLM instance
        - tokenizer: Tokenizer instance
        - corpus_by_domain: dict {domain: list of paragraphs}
        - pooling: "mean", "last", atau "none"
        - verbose: cetak progress
        """
        self.domain_embeddings = {}
        self.domain_paragraphs = {}

        for domain, paragraphs in corpus_by_domain.items():
            if not paragraphs:
                continue

            embeddings = []
            valid_paragraphs = []

            for para in paragraphs:
                # Encode paragraph (tanpa BOS/EOS untuk embedding)
                ids = tokenizer.encode(para, add_bos=False, add_eos=False)
                if not ids:
                    continue

                # Truncate jika terlalu panjang
                max_len = int(getattr(model.config.model, 'max_position_embeddings', 96))
                if len(ids) > max_len:
                    ids = ids[:max_len]

                input_ids = np.array([ids], dtype=np.int64)
                emb = model.get_hidden_states(input_ids, pooling=pooling)
                embeddings.append(emb[0])  # [D]
                valid_paragraphs.append(para)

            if embeddings:
                self.domain_embeddings[domain] = np.stack(embeddings, axis=0)
                self.domain_paragraphs[domain] = valid_paragraphs

                if verbose:
                    print(
                        f"[Router] Domain {domain}: "
                        f"{len(embeddings)} embeddings cached"
                    )

    def search(
        self,
        query_embedding: np.ndarray,
        domain: Optional[str] = None,
        top_k: int = 2,
    ) -> Tuple[List[str], List[float]]:
        """
        Cari paragraf paling mirip dengan query.

        Parameter:
        - query_embedding: [D] vektor query
        - domain: batasi search ke domain ini (None = semua domain)
        - top_k: jumlah paragraf teratas yang dikembalikan

        Return:
        - (paragraphs, similarities): tuple dengan top-k results
        """
        if not self.domain_embeddings:
            return [], []

        # Gabungkan semua embeddings dari domain yang relevan
        if domain is not None and domain in self.domain_embeddings:
            search_domains = [domain]
        else:
            search_domains = list(self.domain_embeddings.keys())

        if not search_domains:
            return [], []

        all_embeddings = []
        all_paragraphs = []
        all_domain_labels = []

        for d in search_domains:
            if d not in self.domain_embeddings:
                continue
            all_embeddings.append(self.domain_embeddings[d])
            all_paragraphs.extend(self.domain_paragraphs[d])
            all_domain_labels.extend([d] * len(self.domain_paragraphs[d]))

        if not all_embeddings:
            return [], []

        corpus = np.concatenate(all_embeddings, axis=0)
        similarities = _cosine_similarity_matrix(query_embedding, corpus)

        # Ambil top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        top_paragraphs = [all_paragraphs[i] for i in top_indices]
        top_sims = [float(similarities[i]) for i in top_indices]

        return top_paragraphs, top_sims

    def save_cache(self, cache_path: Path) -> None:
        """
        Simpan cache ke file .npz.

        Format:
        - {domain}_embeddings: np.ndarray
        - {domain}_paragraphs: JSON string
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {}

        for domain, emb in self.domain_embeddings.items():
            safe_key = re.sub(r'[^A-Za-z0-9_]', '_', domain)
            save_dict[f"{safe_key}__embeddings"] = emb

        # Paragraphs disimpan sebagai JSON string
        paragraphs_json = {
            domain: paras for domain, paras in self.domain_paragraphs.items()
        }
        save_dict["__paragraphs_json__"] = np.array(
            [json.dumps(paragraphs_json, ensure_ascii=False)],
            dtype=object,
        )

        np.savez_compressed(cache_path, **save_dict)

    def load_cache(self, cache_path: Path) -> bool:
        """
        Load cache dari file .npz.

        Return:
        - True jika berhasil, False jika file tidak ada / gagal
        """
        cache_path = Path(cache_path)

        if not cache_path.exists():
            return False

        # Tambah .npz jika belum ada
        if not str(cache_path).endswith('.npz'):
            cache_path = cache_path.with_suffix('.npz')

        if not cache_path.exists():
            return False

        try:
            with np.load(cache_path, allow_pickle=True) as data:
                self.domain_embeddings = {}
                self.domain_paragraphs = {}

                # Load paragraphs dari JSON
                if "__paragraphs_json__" in data:
                    paragraphs_json = json.loads(str(data["__paragraphs_json__"][0]))
                    self.domain_paragraphs = paragraphs_json

                # Load embeddings
                for key in data.files:
                    if key == "__paragraphs_json__":
                        continue
                    if key.endswith("__embeddings"):
                        # Reconstruct domain name dari key
                        domain = key[:-len("__embeddings")]
                        self.domain_embeddings[domain] = data[key]

                return True

        except Exception as exc:
            print(f"[Router] Gagal load cache: {exc}")
            return False


# ============================================================================
# MAIN ROUTER
# ============================================================================

class DomainRouter:
    """
    Auto Domain Router untuk MoE inference.

    Cara kerja:
    1. User input prompt
    2. Keyword scoring untuk setiap domain
    3. Pilih domain dengan score tertinggi
    4. (Opsional) Semantic search untuk inject context

    Contoh pakai:
        router = DomainRouter(config)
        router.load_keywords_and_cache(model, tokenizer, paragraphs_by_domain)

        domain, confidence = router.route("apa itu fotosintesis?")
        # → ("SAINS", 0.94)

        context_paragraphs = router.semantic_search(
            prompt="fotosintesis",
            domain=domain,
            top_k=2,
        )
    """

    def __init__(self, config: Config = CONFIG):
        self.config = config

        # MoE parameters dari config
        self.moe_auto_routing = bool(
            getattr(config.model, 'moe_auto_routing', True)
        )
        self.confidence_threshold = float(
            getattr(config.model, 'moe_router_confidence_threshold', 0.3)
        )
        self.router_top_keywords = int(
            getattr(config.model, 'moe_router_top_keywords', 30)
        )
        self.semantic_top_k = int(
            getattr(config.model, 'moe_semantic_top_k', 2)
        )
        self.keywords_per_domain = int(
            getattr(config.training, 'moe_keywords_per_domain', 30)
        )
        self.keyword_method = getattr(
            config.training, 'moe_keyword_method', 'hybrid'
        )

        # Data structures
        self.domain_keywords: Dict[str, List[str]] = {}
        self.keyword_weights: Dict[str, Dict[str, float]] = {}
        self.search_engine = SemanticSearchEngine()

        # Force domain (user override via /force command)
        self.force_domain: Optional[str] = None

    # ====================================================================
    # KEYWORD MANAGEMENT
    # ====================================================================

    def build_keywords(
        self,
        corpus_by_domain: Dict[str, List[str]],
        verbose: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Extract keywords dari corpus.

        Return:
        - dict {domain: list of keywords}
        """
        extractor = KeywordExtractor(method=self.keyword_method)
        self.domain_keywords = extractor.extract(
            corpus_by_domain,
            top_k=self.keywords_per_domain,
        )

        # Build weights (uniform untuk sekarang, bisa diperluas)
        self.keyword_weights = {}
        for domain, keywords in self.domain_keywords.items():
            # Weight menurun dari 1.0 ke 0.5 berdasarkan rank
            self.keyword_weights[domain] = {
                word: 1.0 - (i / (2 * len(keywords)))
                for i, word in enumerate(keywords)
            }

        if verbose:
            print(f"[Router] Keywords dibangun untuk {len(self.domain_keywords)} domain:")
            for domain, kws in self.domain_keywords.items():
                preview = kws[:5]
                print(f"  {domain}: {preview}... ({len(kws)} total)")

        return self.domain_keywords

    def score_domain(self, prompt: str) -> Dict[str, float]:
        """
        Hitung score setiap domain untuk prompt ini.

        Score = sum of weights dari keywords yang match.

        Return:
        - dict {domain: score}
        """
        prompt_words = set(_simple_tokenize(prompt))

        if not prompt_words:
            return {}

        scores = {}
        for domain, keywords in self.domain_keywords.items():
            weights = self.keyword_weights.get(domain, {})
            score = 0.0
            for word in prompt_words:
                if word in weights:
                    score += weights[word]
            scores[domain] = score

        return scores

    def route(
        self,
        prompt: str,
        available_domains: Optional[List[str]] = None,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Pilih domain untuk prompt ini.

        Return:
        - (domain, confidence, all_scores)
        - domain: nama domain terpilih (atau "GENERAL" jika fallback)
        - confidence: confidence score [0, 1]
        - all_scores: dict semua scores untuk debugging
        """
        # Force domain override
        if self.force_domain is not None:
            return (self.force_domain, 1.0, {self.force_domain: 1.0})

        # Score semua domain
        all_scores = self.score_domain(prompt)

        if not all_scores:
            return ("GENERAL", 0.0, {})

        # Filter ke available domains jika diberikan
        if available_domains is not None:
            all_scores = {
                d: s for d, s in all_scores.items()
                if d in available_domains
            }

        if not all_scores:
            return ("GENERAL", 0.0, {})

        # Normalize scores ke [0, 1]
        max_score = max(all_scores.values())
        if max_score <= 0:
            return ("GENERAL", 0.0, all_scores)

        normalized = {d: s / max_score for d, s in all_scores.items()}

        # Pilih top domain
        best_domain = max(normalized, key=normalized.get)
        best_score = normalized[best_domain]

        # Check confidence threshold
        if best_score < self.confidence_threshold:
            return ("GENERAL", best_score, all_scores)

        return (best_domain, best_score, all_scores)

    # ====================================================================
    # SEMANTIC SEARCH
    # ====================================================================

    def semantic_search(
        self,
        prompt: str,
        model,
        tokenizer,
        domain: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Cari paragraf paling mirip dengan prompt.

        Parameter:
        - prompt: teks query
        - model: CustomTransformerLM instance
        - tokenizer: Tokenizer instance
        - domain: batasi search ke domain ini (None = semua)
        - top_k: jumlah paragraf yang dikembalikan

        Return:
        - (paragraphs, similarities)
        """
        if top_k is None:
            top_k = self.semantic_top_k

        if top_k <= 0:
            return [], []

        if not self.search_engine.domain_embeddings:
            return [], []

        # Encode prompt
        ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
        if not ids:
            return [], []

        max_len = int(getattr(model.config.model, 'max_position_embeddings', 96))
        if len(ids) > max_len:
            ids = ids[:max_len]

        input_ids = np.array([ids], dtype=np.int64)
        query_embedding = model.get_hidden_states(input_ids, pooling="mean")[0]

        return self.search_engine.search(
            query_embedding=query_embedding,
            domain=domain,
            top_k=top_k,
        )

    def build_semantic_cache(
        self,
        model,
        tokenizer,
        corpus_by_domain: Dict[str, List[str]],
        verbose: bool = True,
    ) -> None:
        """Bangun cache embeddings dari corpus."""
        self.search_engine.build_cache(
            model=model,
            tokenizer=tokenizer,
            corpus_by_domain=corpus_by_domain,
            pooling="mean",
            verbose=verbose,
        )

    # ====================================================================
    # PERSISTENCE
    # ====================================================================

    def save_keywords(self, path: Path) -> None:
        """Simpan keywords ke file JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "domain_keywords": self.domain_keywords,
            "keyword_weights": self.keyword_weights,
            "config": {
                "method": self.keyword_method,
                "top_k": self.keywords_per_domain,
                "confidence_threshold": self.confidence_threshold,
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_keywords(self, path: Path) -> bool:
        """
        Load keywords dari file JSON.

        Return:
        - True jika berhasil
        """
        path = Path(path)

        if not path.exists():
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.domain_keywords = data.get("domain_keywords", {})
            self.keyword_weights = data.get("keyword_weights", {})

            return True

        except Exception as exc:
            print(f"[Router] Gagal load keywords: {exc}")
            return False

    def save_cache(self, cache_path: Path) -> None:
        """Simpan semantic cache ke .npz."""
        self.search_engine.save_cache(cache_path)

    def load_cache(self, cache_path: Path) -> bool:
        """Load semantic cache dari .npz."""
        return self.search_engine.load_cache(cache_path)

    # ====================================================================
    # FORCE DOMAIN (User Override)
    # ====================================================================

    def set_force_domain(self, domain: Optional[str]) -> None:
        """
        Force router untuk selalu pakai domain tertentu.

        Set None untuk disable.
        """
        self.force_domain = domain

    def clear_force_domain(self) -> None:
        """Hapus force domain."""
        self.force_domain = None

    # ====================================================================
    # DEBUG
    # ====================================================================

    def debug_route(self, prompt: str) -> str:
        """
        Return string debug untuk prompt.

        Format:
            "DOMAIN: confidence% [matched_words]"
        """
        domain, confidence, all_scores = self.route(prompt)
        scores_str = ", ".join(
            f"{d}={s:.2f}" for d, s in sorted(all_scores.items(), key=lambda x: -x[1])[:5]
        )

        prompt_words = set(_simple_tokenize(prompt))
        matched = []
        for word in prompt_words:
            for d, weights in self.keyword_weights.items():
                if word in weights and d == domain:
                    matched.append(word)
                    break

        matched_str = ", ".join(matched[:5]) if matched else "(none)"

        return (
            f"[{domain}] conf={confidence:.2%} "
            f"| matched: {matched_str} "
            f"| top scores: {scores_str}"
        )


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("core/router.py - Test Suite")
    print("=" * 60)

    # ================================================================
    # Test 1: Keyword Extraction
    # ================================================================
    print("\n[Test 1] Keyword Extraction")
    print("-" * 60)

    corpus = {
        "MOBIL": [
            "Mobil adalah kendaraan beroda empat yang menggunakan mesin.",
            "Rem mobil berfungsi untuk menghentikan kendaraan.",
            "Bahan bakar digunakan untuk menggerakkan mesin mobil.",
        ],
        "SAINS": [
            "Air adalah senyawa kimia dengan rumus H2O.",
            "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan.",
            "Gravitasi adalah gaya tarik menarik antara dua benda bermassa.",
        ],
        "FILSAFAT": [
            "Filsafat adalah ilmu yang mempelajari pertanyaan mendasar.",
            "Etika adalah cabang filsafat yang membahas baik dan buruk.",
            "Logika adalah ilmu tentang prinsip penalaran yang valid.",
        ],
    }

    for method in ["frequency", "tfidf", "hybrid"]:
        extractor = KeywordExtractor(method=method)
        keywords = extractor.extract(corpus, top_k=5)
        print(f"\n  Method: {method}")
        for domain, kws in keywords.items():
            print(f"    {domain}: {kws}")

    # ================================================================
    # Test 2: Domain Scoring
    # ================================================================
    print("\n[Test 2] Domain Scoring")
    print("-" * 60)

    router = DomainRouter(CONFIG)
    router.build_keywords(corpus, verbose=False)

    test_prompts = [
        "apa itu fotosintesis?",
        "bagaimana cara kerja rem mobil?",
        "jelaskan tentang etika",
        "saya suka makan nasi goreng",  # tidak ada domain match
    ]

    for prompt in test_prompts:
        domain, conf, scores = router.route(prompt)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  → Domain: {domain} (confidence: {conf:.2%})")
        print(f"    Scores: {scores}")

    # ================================================================
    # Test 3: Force Domain
    # ================================================================
    print("\n[Test 3] Force Domain Override")
    print("-" * 60)

    router.set_force_domain("FILSAFAT")
    domain, conf, _ = router.route("apa itu mobil?")
    print(f"  Force FILSAFAT + prompt mobil → {domain} (conf={conf:.2%})")

    router.clear_force_domain()
    domain, conf, _ = router.route("apa itu mobil?")
    print(f"  After clear + prompt mobil → {domain} (conf={conf:.2%})")

    # ================================================================
    # Test 4: Save/Load Keywords
    # ================================================================
    print("\n[Test 4] Save/Load Keywords")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        keywords_path = Path(tmp) / "keywords.json"

        router.save_keywords(keywords_path)
        print(f"  Saved to: {keywords_path}")

        router2 = DomainRouter(CONFIG)
        success = router2.load_keywords(keywords_path)
        print(f"  Loaded: {success}")
        print(f"  Domains: {list(router2.domain_keywords.keys())}")

    # ================================================================
    # Test 5: Cosine Similarity
    # ================================================================
    print("\n[Test 5] Cosine Similarity")
    print("-" * 60)

    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = np.array([0.9, 0.1, 0.0], dtype=np.float32)

    print(f"  sim(a, b) = {_cosine_similarity(a, b):.4f} (expected 1.0)")
    print(f"  sim(a, c) = {_cosine_similarity(a, c):.4f} (expected 0.0)")
    print(f"  sim(a, d) = {_cosine_similarity(a, d):.4f} (expected ~0.995)")

    # Matrix similarity
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus_arr = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.7],
    ], dtype=np.float32)
    sims = _cosine_similarity_matrix(query, corpus_arr)
    print(f"  Matrix sims: {sims}")

    # ================================================================
    # Test 6: Debug Output
    # ================================================================
    print("\n[Test 6] Debug Output")
    print("-" * 60)

    for prompt in test_prompts:
        debug_str = router.debug_route(prompt)
        print(f"  {debug_str}")

    print("\n" + "=" * 60)
    print("core/router.py test OK")
    print("=" * 60)
