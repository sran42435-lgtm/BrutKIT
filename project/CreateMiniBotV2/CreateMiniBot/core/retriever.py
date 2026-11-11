# core/retriever.py
#
# BM25 Retrieval Engine untuk RAG (Retrieval-Augmented Generation).
#
# Tugas utama:
# 1. Build inverted index dari corpus (paragraf-paragraf)
# 2. Retrieve paragraf paling relevan berdasarkan query
# 3. Query expansion fallback jika retrieval gagal
# 4. Save/load index ke JSON (portable)
# 5. Dynamic threshold untuk filtering hasil
#
# Pendekatan:
# - BM25 (Best Matching 25) - algoritma text retrieval klasik
# - Inverted index untuk performa query cepat
# - Pure Python + NumPy (tidak butuh FAISS/ChromaDB)
# - Lightweight dan portable
#
# Mitigasi risiko:
# - Risiko #1 (Retrieval gagal): Query expansion + dynamic threshold + graceful degradation
# - Risiko #3 (Dataset besar → lambat): Inverted index O(Q × avg_docs_per_term)
#
# Keterhubungan:
# - config.py           : parameter retrieval (RetrievalConfig)
# - pipeline/trainer.py : load corpus dari datasets/
# - testing/playground.py: retrieve context sebelum generation
# - core/model_exporter.py: bundle index ke safetensors

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import CONFIG, Config, RetrievalConfig


# ============================================================================
# STOPWORDS (Indonesia + Inggris)
# ============================================================================

STOPWORDS: frozenset = frozenset({
    # Indonesia - Kata Tanya
    "apa", "siapa", "kapan", "dimana", "mengapa", "bagaimana", "berapa",
    # Indonesia - Kata Ganti
    "saya", "kamu", "dia", "kami", "mereka", "kita", "anda", "ini", "itu",
    # Indonesia - Kata Hubung
    "dan", "atau", "tetapi", "namun", "yang", "dengan", "untuk", "dari",
    "di", "pada", "oleh", "dalam", "tentang", "antara",
    # Indonesia - Kata Modal
    "akan", "sudah", "telah", "bisa", "dapat", "harus", "mungkin",
    # Indonesia - Kata Kerja Umum
    "adalah", "merupakan", "yaitu", "ada", "menjadi", "memiliki",
    # Inggris - Articles & Pronouns
    "the", "a", "an", "this", "that", "it", "they", "we", "you",
    # Inggris - Prepositions & Conjunctions
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "and", "or",
    # Inggris - Verbs
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    # Angka
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
})


# ============================================================================
# SYNONYM DICTIONARY (untuk Query Expansion)
# ============================================================================

SYNONYMS: Dict[str, List[str]] = {
    # Kata tanya
    "apa": ["pengertian", "definisi", "maksud"],
    "jelaskan": ["uraikan", "ceritakan", "deskripsikan"],
    "bagaimana": ["cara", "proses", "mekanisme"],
    "mengapa": ["alasan", "sebab", "kenapa"],
    
    # Kata kerja umum
    "membuat": ["menciptakan", "memproduksi", "menghasilkan"],
    "menggunakan": ["memakai", "memanfaatkan", "mengaplikasikan"],
    "berfungsi": ["berguna", "bermanfaat", "bertujuan"],
    
    # Kata sifat umum
    "besar": ["luas", "raya", "akbar"],
    "kecil": ["mini", "mungil", "tipis"],
    "penting": ["krusial", "vital", "esensial"],
    
    # Domain-specific (bisa ditambah sesuai dataset)
    "mobil": ["kendaraan", "otomotif", "transportasi"],
    "fotosintesis": ["proses tumbuhan", "pembuatan makanan tanaman"],
    "komputer": ["pc", "laptop", "mesin komputasi"],
}


# ============================================================================
# TEXT PROCESSING HELPERS
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalisasi teks: lowercase, hapus punctuation berlebih, rapikan spasi.
    """
    text = text.lower()
    # Ganti punctuation dengan spasi
    text = re.sub(r'[^\w\s]', ' ', text)
    # Rapikan multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenisasi teks menjadi list kata.
    
    Args:
        text: teks input
        remove_stopwords: jika True, stopwords akan di-filter
    
    Returns:
        List of tokens (lowercase, alphanumeric only)
    """
    text = normalize_text(text)
    tokens = text.split()
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    
    return tokens


def expand_query(query: str, expansion_count: int = 2) -> str:
    """
    Expand query dengan menambahkan sinonim.
    
    Args:
        query: query asli
        expansion_count: jumlah sinonim per kata (default 2)
    
    Returns:
        Expanded query string
    """
    tokens = tokenize(query, remove_stopwords=False)
    expanded_tokens = list(tokens)  # Copy
    
    for token in tokens:
        if token in SYNONYMS:
            synonyms = SYNONYMS[token][:expansion_count]
            expanded_tokens.extend(synonyms)
    
    return " ".join(expanded_tokens)


# ============================================================================
# BM25 RETRIEVER
# ============================================================================

class BM25Retriever:
    """
    BM25-based retrieval engine dengan inverted index.
    
    Contoh pakai:
        retriever = BM25Retriever(config)
        retriever.build_index(corpus_paragraphs)
        
        results = retriever.retrieve("apa itu fotosintesis?", top_k=3)
        # results: [(paragraph_text, score), ...]
    """
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.retrieval_config: RetrievalConfig = config.retrieval
        
        # BM25 parameters
        self.k1: float = self.retrieval_config.bm25_k1
        self.b: float = self.retrieval_config.bm25_b
        
        # Corpus data
        self.corpus: List[str] = []  # List of paragraphs
        self.doc_lengths: List[int] = []  # Length of each document (in tokens)
        self.avg_doc_length: float = 0.0
        
        # Inverted index: term -> {doc_id: term_frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        
        # IDF cache: term -> idf_value
        self.idf_cache: Dict[str, float] = {}
        
        # Statistics
        self.total_docs: int = 0
        self.is_built: bool = False
    
    # ========================================================================
    # INDEX BUILDING
    # ========================================================================
    
    def build_index(self, corpus: List[str], verbose: bool = True) -> None:
        """
        Build inverted index dari corpus.
        
        Args:
            corpus: list of paragraph strings
            verbose: print progress
        """
        if not corpus:
            raise ValueError("Corpus tidak boleh kosong.")
        
        self.corpus = corpus
        self.total_docs = len(corpus)
        
        # Reset index
        self.inverted_index = defaultdict(dict)
        self.doc_lengths = []
        self.idf_cache = {}
        
        if verbose:
            print(f"[Retriever] Building index dari {self.total_docs} paragraf...")
        
        # Build inverted index
        for doc_id, paragraph in enumerate(corpus):
            tokens = tokenize(paragraph, remove_stopwords=True)
            self.doc_lengths.append(len(tokens))
            
            # Hitung term frequency untuk dokumen ini
            term_freq = Counter(tokens)
            
            for term, freq in term_freq.items():
                self.inverted_index[term][doc_id] = freq
        
        # Hitung average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        
        # Pre-compute IDF untuk semua terms
        self._compute_idf()
        
        self.is_built = True
        
        if verbose:
            print(f"[Retriever] Index selesai dibangun:")
            print(f"  - Total dokumen: {self.total_docs}")
            print(f"  - Total unique terms: {len(self.inverted_index)}")
            print(f"  - Average doc length: {self.avg_doc_length:.2f} tokens")
    
    def _compute_idf(self) -> None:
        """
        Pre-compute IDF (Inverse Document Frequency) untuk semua terms.
        
        IDF = log((N - df + 0.5) / (df + 0.5))
        dimana:
        - N = total documents
        - df = document frequency (jumlah dokumen yang mengandung term)
        """
        N = self.total_docs
        
        for term, doc_dict in self.inverted_index.items():
            df = len(doc_dict)
            # BM25 IDF formula (dengan smoothing)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_cache[term] = idf
    
    # ========================================================================
    # RETRIEVAL
    # ========================================================================
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        use_dynamic_threshold: bool = True,
        enable_expansion_fallback: bool = True,
        verbose: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve paragraf paling relevan untuk query.
        
        Args:
            query: query string
            top_k: jumlah hasil teratas (default dari config)
            min_score: minimum BM25 score (default dari config)
            use_dynamic_threshold: gunakan dynamic threshold jika min_score None
            enable_expansion_fallback: coba query expansion jika tidak ada hasil
            verbose: print debug info
        
        Returns:
            List of (paragraph_text, score) tuples, sorted by score descending
        """
        if not self.is_built:
            raise RuntimeError("Index belum dibangun. Panggil build_index() terlebih dahulu.")
        
        # Default parameters dari config
        if top_k is None:
            top_k = self.retrieval_config.top_k
        if min_score is None:
            min_score = self.retrieval_config.min_score
        
        # Tokenize query
        query_tokens = tokenize(query, remove_stopwords=True)
        
        if not query_tokens:
            if verbose:
                print("[Retriever] Query kosong setelah tokenization.")
            return []
        
        # Hitung BM25 scores
        scores = self._compute_bm25_scores(query_tokens)
        
        if not scores:
            if verbose:
                print("[Retriever] Tidak ada skor yang dihitung.")
            
            # Fallback: query expansion
            if enable_expansion_fallback and self.retrieval_config.enable_query_expansion_fallback:
                if verbose:
                    print("[Retriever] Mencoba query expansion...")
                expanded_query = expand_query(
                    query,
                    expansion_count=self.retrieval_config.query_expansion_count
                )
                return self.retrieve(
                    expanded_query,
                    top_k=top_k,
                    min_score=min_score,
                    use_dynamic_threshold=use_dynamic_threshold,
                    enable_expansion_fallback=False,  # Prevent infinite recursion
                    verbose=verbose,
                )
            
            return []
        
        # Apply threshold
        if min_score is not None:
            # Static threshold
            scores = [(doc_id, score) for doc_id, score in scores if score >= min_score]
        elif use_dynamic_threshold and self.retrieval_config.dynamic_threshold_ratio > 0:
            # Dynamic threshold: gunakan X% dari top score
            max_score = max(score for _, score in scores)
            threshold = max_score * self.retrieval_config.dynamic_threshold_ratio
            scores = [(doc_id, score) for doc_id, score in scores if score >= threshold]
        
        if not scores:
            if verbose:
                print("[Retriever] Tidak ada hasil yang melewati threshold.")
            return []
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil top-k
        top_scores = scores[:top_k]
        
        # Convert ke (paragraph_text, score)
        results = [
            (self.corpus[doc_id], score)
            for doc_id, score in top_scores
        ]
        
        if verbose:
            print(f"[Retriever] Ditemukan {len(results)} hasil:")
            for i, (text, score) in enumerate(results, 1):
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  {i}. Score={score:.3f}: {preview}")
        
        return results
    
    def _compute_bm25_scores(self, query_tokens: List[str]) -> List[Tuple[int, float]]:
        """
        Hitung BM25 scores untuk semua dokumen berdasarkan query tokens.
        
        BM25 formula:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
        
        dimana:
        - f(qi, D) = frequency of term qi in document D
        - |D| = length of document D
        - avgdl = average document length
        - k1, b = BM25 parameters
        """
        scores: Dict[int, float] = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self.idf_cache.get(term, 0.0)
            doc_dict = self.inverted_index[term]
            
            for doc_id, tf in doc_dict.items():
                doc_len = self.doc_lengths[doc_id]
                
                # BM25 term score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_length
                )
                
                term_score = idf * (numerator / denominator)
                scores[doc_id] += term_score
        
        return list(scores.items())
    
    # ========================================================================
    # PERSISTENCE (Save/Load Index)
    # ========================================================================
    
    def save_index(self, path: Path) -> None:
        """
        Save index ke file JSON.
        
        Format:
        {
            "version": "1.0",
            "config": {...},
            "corpus": [...],
            "doc_lengths": [...],
            "inverted_index": {...},
            "idf_cache": {...},
            "stats": {...}
        }
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert inverted index ke serializable format
        # defaultdict -> dict, int keys -> string keys (JSON requirement)
        inverted_index_serializable = {
            term: {str(doc_id): freq for doc_id, freq in doc_dict.items()}
            for term, doc_dict in self.inverted_index.items()
        }
        
        data = {
            "version": "1.0",
            "config": {
                "k1": self.k1,
                "b": self.b,
            },
            "corpus": self.corpus,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "inverted_index": inverted_index_serializable,
            "idf_cache": self.idf_cache,
            "stats": {
                "total_docs": self.total_docs,
                "total_terms": len(self.inverted_index),
            },
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[Retriever] Index disimpan ke: {path}")
    
    def load_index(self, path: Path) -> bool:
        """
        Load index dari file JSON.
        
        Returns:
            True jika berhasil, False jika gagal
        """
        path = Path(path)
        
        if not path.exists():
            print(f"[Retriever] File index tidak ditemukan: {path}")
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate version
            version = data.get("version")
            if version != "1.0":
                print(f"[Retriever] Warning: Version mismatch (expected 1.0, got {version})")
            
            # Load config
            config = data.get("config", {})
            self.k1 = config.get("k1", self.k1)
            self.b = config.get("b", self.b)
            
            # Load corpus
            self.corpus = data.get("corpus", [])
            self.doc_lengths = data.get("doc_lengths", [])
            self.avg_doc_length = data.get("avg_doc_length", 0.0)
            
            # Load inverted index (convert string keys back to int)
            inverted_index_raw = data.get("inverted_index", {})
            self.inverted_index = defaultdict(dict)
            for term, doc_dict in inverted_index_raw.items():
                for doc_id_str, freq in doc_dict.items():
                    self.inverted_index[term][int(doc_id_str)] = freq
            
            # Load IDF cache
            self.idf_cache = data.get("idf_cache", {})
            
            # Load stats
            stats = data.get("stats", {})
            self.total_docs = stats.get("total_docs", len(self.corpus))
            
            self.is_built = True
            
            print(f"[Retriever] Index dimuat dari: {path}")
            print(f"  - Total dokumen: {self.total_docs}")
            print(f"  - Total unique terms: {len(self.inverted_index)}")
            
            return True
        
        except Exception as exc:
            print(f"[Retriever] Gagal load index: {exc}")
            return False
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_corpus_stats(self) -> Dict:
        """
        Return statistik corpus.
        """
        return {
            "total_docs": self.total_docs,
            "total_terms": len(self.inverted_index),
            "avg_doc_length": self.avg_doc_length,
            "min_doc_length": min(self.doc_lengths) if self.doc_lengths else 0,
            "max_doc_length": max(self.doc_lengths) if self.doc_lengths else 0,
        }
    
    def get_top_terms(self, n: int = 20) -> List[Tuple[str, int]]:
        """
        Return top-N terms berdasarkan document frequency.
        """
        term_df = [
            (term, len(doc_dict))
            for term, doc_dict in self.inverted_index.items()
        ]
        term_df.sort(key=lambda x: x[1], reverse=True)
        return term_df[:n]
    
    def clear(self) -> None:
        """
        Clear semua data index.
        """
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0.0
        self.inverted_index = defaultdict(dict)
        self.idf_cache = {}
        self.total_docs = 0
        self.is_built = False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_corpus_from_directory(
    datasets_dir: Path,
    max_paragraph_length: int = 500,
    verbose: bool = True,
) -> List[str]:
    """
    Load semua paragraf dari file .txt di datasets_dir.
    
    Args:
        datasets_dir: path ke folder datasets
        max_paragraph_length: truncate paragraf jika lebih panjang dari ini
        verbose: print progress
    
    Returns:
        List of paragraph strings
    """
    datasets_dir = Path(datasets_dir)
    
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory tidak ditemukan: {datasets_dir}")
    
    paragraphs = []
    
    # Load semua file .txt
    for txt_file in sorted(datasets_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by double newline (paragraph separator)
        raw_paragraphs = content.split("\n\n")
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Truncate jika terlalu panjang
            if len(para) > max_paragraph_length:
                para = para[:max_paragraph_length] + "..."
            
            paragraphs.append(para)
    
    if verbose:
        print(f"[Retriever] Loaded {len(paragraphs)} paragraphs dari {datasets_dir}")
    
    return paragraphs


def build_retriever_from_corpus(
    corpus: List[str],
    config: Config = CONFIG,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> BM25Retriever:
    """
    Convenience function: build retriever dari corpus dan save index.
    
    Args:
        corpus: list of paragraph strings
        config: Config object
        save_path: path untuk save index (optional)
        verbose: print progress
    
    Returns:
        BM25Retriever instance (sudah di-build)
    """
    retriever = BM25Retriever(config)
    retriever.build_index(corpus, verbose=verbose)
    
    if save_path:
        retriever.save_index(save_path)
    
    return retriever


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("core/retriever.py - Test Suite")
    print("=" * 70)
    
    # ================================================================
    # Test 1: Basic tokenization
    # ================================================================
    print("\n[Test 1] Tokenization")
    print("-" * 70)
    
    test_text = "Mobil adalah kendaraan beroda empat yang menggunakan mesin."
    tokens = tokenize(test_text, remove_stopwords=True)
    print(f"Input: {test_text}")
    print(f"Tokens (stopwords removed): {tokens}")
    
    # ================================================================
    # Test 2: Query expansion
    # ================================================================
    print("\n[Test 2] Query Expansion")
    print("-" * 70)
    
    query = "apa itu fotosintesis"
    expanded = expand_query(query, expansion_count=2)
    print(f"Original: {query}")
    print(f"Expanded: {expanded}")
    
    # ================================================================
    # Test 3: Build index dari sample corpus
    # ================================================================
    print("\n[Test 3] Build Index")
    print("-" * 70)
    
    sample_corpus = [
        "Mobil adalah kendaraan beroda empat yang menggunakan mesin sebagai penggerak utama.",
        "Rem mobil berfungsi untuk mengurangi kecepatan dan menghentikan kendaraan.",
        "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan menggunakan cahaya matahari.",
        "Air adalah senyawa kimia dengan rumus H2O yang penting bagi kehidupan.",
        "Komputer adalah perangkat elektronik yang dapat memproses data dan menjalankan program.",
    ]
    
    retriever = BM25Retriever(CONFIG)
    retriever.build_index(sample_corpus, verbose=True)
    
    # Print stats
    stats = retriever.get_corpus_stats()
    print(f"\nCorpus stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Print top terms
    top_terms = retriever.get_top_terms(10)
    print(f"\nTop 10 terms (by document frequency):")
    for term, df in top_terms:
        print(f"  {term}: {df} docs")
    
    # ================================================================
    # Test 4: Retrieval
    # ================================================================
    print("\n[Test 4] Retrieval")
    print("-" * 70)
    
    test_queries = [
        "apa itu mobil",
        "bagaimana cara kerja rem",
        "fotosintesis",
        "air dan kimia",
        "pertanyaan yang tidak relevan dengan corpus",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve(query, top_k=2, verbose=True)
        
        if not results:
            print("  → Tidak ada hasil")
    
    # ================================================================
    # Test 5: Save/Load index
    # ================================================================
    print("\n[Test 5] Save/Load Index")
    print("-" * 70)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index.json"
        
        # Save
        retriever.save_index(index_path)
        
        # Load ke retriever baru
        retriever2 = BM25Retriever(CONFIG)
        success = retriever2.load_index(index_path)
        
        if success:
            # Test retrieval dengan loaded index
            query = "apa itu mobil"
            results = retriever2.retrieve(query, top_k=2, verbose=True)
            print(f"\nRetrieval dengan loaded index berhasil!")
        else:
            print(f"\nGagal load index!")
    
    print("\n" + "=" * 70)
    print("core/retriever.py test OK")
    print("=" * 70)
