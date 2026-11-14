"""
config.py
Pusat Pengendali Keputusan untuk Hybrid Fragmented Compressed Memory (HF-CM).
"""
from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True)
class InputLimits:
    MAX_INPUT_CHARACTERS: Final[int] = 8000
    MIN_INPUT_CHARACTERS: Final[int] = 1
    DEFAULT_TIMEZONE: Final[str] = "Asia/Jakarta"

@dataclass(frozen=True)
class FastMemoryLimits:
    MAX_FAST_TURNS: Final[int] = 10
    MAX_CLARIFICATION_ROUNDS: Final[int] = 3
    PENDING_RELEVANCE_THRESHOLD: Final[float] = 0.60
    PENDING_EXPIRATION_HOURS: Final[int] = 24
    # [NEW] Ditambahkan sesuai Spesifikasi Penguatan
    PENDING_SWITCH_THRESHOLD: Final[float] = 0.30
    PENDING_TTL_SECONDS: Final[int] = 86400
    # [REASONING PURITY] Short Pending Safety (Prioritas #3).
    # Input pendek TIDAK otomatis dianggap relevan; harus divalidasi
    # terhadap kandidat pending. Ini mencegah "C++" dianggap jawaban
    # untuk pending "Bisnis Kopi / Kedai Kopi?".
    SHORT_PENDING_INPUT_THRESHOLD: Final[int] = 3
    SHORT_PENDING_MIN_MATCH: Final[int] = 1
    PENDING_OPTION_REFERENCE_WEIGHT: Final[float] = 0.25

@dataclass(frozen=True)
class SignalConstants:
    MIN_TOKEN_LENGTH: Final[int] = 3
    MAX_ANALYSIS_TOKENS: Final[int] = 500
    MIN_SIGNAL_STRENGTH: Final[float] = 0.10

@dataclass(frozen=True)
class ActivationWeights:
    """Bobot untuk scoring Topic Key (Layer 4)."""
    # [UPDATED] Bobot disesuaikan agar total = 1.0 dengan penambahan Unique & Cross-Session
    TOKEN_WEIGHT: Final[float] = 0.20
    DESCRIPTION_WEIGHT: Final[float] = 0.15
    CO_OCCURRENCE_WEIGHT: Final[float] = 0.15
    CONTINUITY_WEIGHT: Final[float] = 0.20
    TEMPORAL_WEIGHT: Final[float] = 0.10
    # [NEW] Bobot untuk kelangkaan token global (IDF-like)
    UNIQUE_TOKEN_WEIGHT: Final[float] = 0.10
    # [NEW] Bobot untuk bukti lintas sesi dari Graph
    CROSS_SESSION_WEIGHT: Final[float] = 0.10

@dataclass(frozen=True)
class DecisionThresholds:
    """
    SPEK TINGGI: Pengendali utama Ambiguity Gate & Topic Switching.
    """
    HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.75
    CLEAR_MARGIN: Final[float] = 0.15 
    AMBIGUITY_MARGIN: Final[float] = 0.05
    MINIMUM_RETRIEVAL_SCORE: Final[float] = 0.10  # [FIX] Diturunkan dari 0.30 ke 0.10 agar query pendek tidak langsung digugurkan (NO_MATCH) sebelum masuk Ambiguity Gate.
    # [NEW] Ditambahkan sesuai Spesifikasi Penguatan
    MIN_TOPIC_ACTIVATION_SCORE: Final[float] = 0.15
    MIN_CONFIDENT_MARGIN: Final[float] = 0.20
    MAX_CANDIDATES: Final[int] = 5

@dataclass(frozen=True)
class TemporalConstants:
    TEMPORAL_MATCH_WEIGHT: Final[float] = 0.80
    TEMPORAL_TOLERANCE_HOURS: Final[int] = 2
    MAX_TEMPORAL_CANDIDATES: Final[int] = 5
    # [NEW] Ditambahkan sesuai Spesifikasi Penguatan
    TEMPORAL_DAY_WINDOW: Final[int] = 1
    TEMPORAL_WEEK_WINDOW: Final[int] = 7
    TEMPORAL_MONTH_WINDOW: Final[int] = 30
    TEMPORAL_YEAR_WINDOW: Final[int] = 365
    TEMPORAL_HOUR_TOLERANCE: Final[int] = 2
    # [REASONING PURITY] Temporal Distance Quality (Prioritas #5).
    # Filter temporal ≠ ranking temporal. Dua fragment di dalam range yang
    # sama tetap harus dibedakan berdasarkan kedekatan waktu dengan query.
    TEMPORAL_DISTANCE_DECAY: Final[float] = 0.10       # per jam
    TEMPORAL_EXACT_MATCH_SCORE: Final[float] = 1.0     # distance == 0
    TEMPORAL_HOUR_PRECISION_WEIGHT: Final[float] = 1.5 # bobot ekstra jika query menyebut jam spesifik
    TEMPORAL_DAY_PRECISION_WEIGHT: Final[float] = 1.0  # bobot baseline untuk range hari

@dataclass(frozen=True)
class RetrievalLimits:
    MAX_RETRIEVED_FRAGMENTS: Final[int] = 5
    MAX_PAYLOAD_BYTES: Final[int] = 10 * 1024 * 1024
    CHECKSUM_ALGORITHM: Final[str] = "sha256"

@dataclass(frozen=True)
class CompressionConstants:
    COMPRESSION_ALGORITHM: Final[str] = "zstd"
    COMPRESSION_LEVEL: Final[int] = 3
    MIN_COMPRESSION_SIZE_BYTES: Final[int] = 256
    ALLOW_RAW_FALLBACK: Final[bool] = False 

@dataclass(frozen=True)
class ContextBudget:
    MAX_CONTEXT_TOKENS: Final[int] = 4000
    MAX_MEMORY_TOKENS: Final[int] = 2500
    MAX_FRAGMENTS_IN_CONTEXT: Final[int] = 3
    CURRENT_INPUT_PRIORITY: Final[float] = 1.0

@dataclass(frozen=True)
class LLMConstants:
    MODEL_NAME: Final[str] = "default-api-model"
    MAX_OUTPUT_TOKENS: Final[int] = 1000
    TEMPERATURE: Final[float] = 0.7
    REQUEST_TIMEOUT_SECONDS: Final[int] = 30
    MAX_RETRIES: Final[int] = 2

@dataclass(frozen=True)
class ContinuityConstants:
    CONTINUITY_THRESHOLD: Final[float] = 0.65
    NEW_TOPIC_THRESHOLD: Final[float] = 0.70
    CONTINUITY_WINDOW_SIZE: Final[int] = 4
    MEDIAN_CONTINUITY_THRESHOLD: Final[float] = 0.60
    MIN_SUBSTANTIVE_SCORE: Final[float] = 0.50
    # [NEW] Ditambahkan sesuai Spesifikasi Penguatan
    MIN_CONTINUITY_SCORE: Final[float] = 0.40
    CONTINUITY_STABILITY_THRESHOLD: Final[float] = 0.65
    # [REASONING PURITY] Continuity Purity (Prioritas #1).
    # final_score (relevance kandidat) TIDAK BOLEH disamakan dengan
    # continuity_score (apakah turn melanjutkan state aktif).
    # Rumus: continuity = overlap + history + fragments - novelty - contradiction
    ACTIVE_TOPIC_OVERLAP_WEIGHT: Final[float] = 0.35
    CONTINUITY_HISTORY_WEIGHT: Final[float] = 0.25
    CONTINUITY_FRAGMENT_WEIGHT: Final[float] = 0.20
    CONTINUITY_NOVELTY_PENALTY: Final[float] = 0.20
    CONTINUITY_CONTRADICTION_PENALTY: Final[float] = 0.15
    # Continuity dianggap cukup kuat di atas ambang ini.
    STRONG_CONTINUITY_THRESHOLD: Final[float] = 0.55

# [NEW] Class baru sesuai Spesifikasi Penguatan
@dataclass(frozen=True)
class TopicSwitchConstants:
    NOVELTY_THRESHOLD: Final[float] = 0.70
    MIN_ACTIVE_TOPIC_OVERLAP: Final[float] = 0.15
    TOPIC_SWITCH_THRESHOLD: Final[float] = 0.60

# [NEW] Class baru untuk Fragment-Level Ranking (Review Kedua)
@dataclass(frozen=True)
class FragmentRankingConstants:
    """Bobot dan batas untuk menentukan fragment mana yang paling relevan di dalam satu Topic Key."""
    FRAGMENT_TOKEN_WEIGHT: Final[float] = 0.40
    FRAGMENT_COOC_WEIGHT: Final[float] = 0.20
    FRAGMENT_TEMPORAL_WEIGHT: Final[float] = 0.20
    FRAGMENT_RECENCY_WEIGHT: Final[float] = 0.20
    MIN_FRAGMENT_RELEVANCE_SCORE: Final[float] = 0.10

@dataclass(frozen=True)
class GraphConstants:
    EDGE_CREATION_THRESHOLD: Final[float] = 0.80
    MAX_EDGES_PER_NODE: Final[int] = 20
    MIN_RELATIONSHIP_CONFIDENCE: Final[float] = 0.75
    # [NEW] Batas minimum skor graph untuk cross-session reactivation
    MIN_CROSS_SESSION_SCORE: Final[float] = 0.30
    # [REASONING PURITY] Fragment Graph Evidence (Prioritas #4).
    # graph_score fragment dihitung dari edge nyata, bukan dari
    # relationship_strength metadata fragment.
    DIRECT_CONTINUATION_WEIGHT: Final[float] = 1.0
    DIRECT_CROSS_SESSION_WEIGHT: Final[float] = 1.2   # cross-session lebih kuat dari continuation
    INDIRECT_GRAPH_DECAY: Final[float] = 0.5          # decay per hop pada path
    MAX_GRAPH_PATH_DEPTH: Final[int] = 3              # batas penelusuran path graph
    FRAGMENT_GRAPH_MIN_SCORE: Final[float] = 0.05     # skor graph di bawah ini dianggap 0

@dataclass(frozen=True)
class StorageConstants:
    DB_TIMEOUT_SECONDS: Final[int] = 5
    MAX_TRANSACTION_RETRIES: Final[int] = 3
    SQLITE_DB_PATH: Final[str] = "data/memory.db"


# ==========================================
# [ASSOCIATIVE EVIDENCE LOCKING] GRUP BARU
# Fase ini tidak menambah fitur, tetapi mengunci KUALITAS BUKTI.
# Prinsip: skor tinggi != bukti kuat.
# ==========================================

@dataclass(frozen=True)
class AssociativeEvidenceConstants:
    """
    Pengendali kualitas bukti asosiatif (Associative Evidence Locking).

    Menjawab: "Berapa jenis bukti yang benar-benar mendukung kandidat ini?"
    bukan hanya "Seberapa besar skor akhirnya?"
    """
    # [ALIAS] Nama kanonik dokumentasi untuk CROSS_SESSION_WEIGHT.
    # PENTING: ini BUKAN bobot ke-8. Total ActivationWeights tetap 1.00.
    # Retrieval harus memakai SALAH SATU (CROSS_SESSION_WEIGHT atau GRAPH_WEIGHT),
    # TIDAK BOLEH menjumlahkan keduanya.
    GRAPH_WEIGHT: Final[float] = 0.10

    # Sebuah evidence baru dihitung "mendukung" jika >= batas ini.
    EVIDENCE_SUPPORT_THRESHOLD: Final[float] = 0.20

    # Minimal berapa JENIS bukti yang mendukung agar boleh auto-retrieve.
    MIN_SUPPORTING_EVIDENCE: Final[int] = 2

    # Confidence minimum sebelum retrieval otomatis dianggap aman.
    MIN_ASSOCIATIVE_CONFIDENCE: Final[float] = 0.55

    # Batas toleransi kontradiksi. Di atas ini -> CONTRADICTORY_EVIDENCE.
    MAX_CONTRADICTION_SCORE: Final[float] = 0.35

    # Token match dianggap "kuat" mulai dari nilai ini.
    STRONG_TOKEN_MATCH_THRESHOLD: Final[float] = 0.50

    # KONTRADIKSI: token cocok kuat TAPI deskripsi hampir tidak cocok
    # = indikasi intent topik sebenarnya berbeda.
    # Contoh: query "lanjutkan bisnis kopi" vs kandidat "Kedai kopi terdekat".
    DESCRIPTION_CONTRADICTION_THRESHOLD: Final[float] = 0.10
    DESCRIPTION_CONTRADICTION_PENALTY: Final[float] = 0.20

    # KONTRADIKSI TEMPORAL: user menyebut waktu spesifik, kandidat tidak punya
    # fragment di rentang waktu itu sama sekali.
    TEMPORAL_CONTRADICTION_PENALTY: Final[float] = 0.10

    # Deteksi dominasi: jika satu evidence menyumbang >= rasio ini dari total
    # positive evidence, kandidat dianggap single-signal dominated.
    EVIDENCE_DOMINANCE_RATIO: Final[float] = 0.75

    # [GUARDRAIL EKSPLISIT] Continuity adalah bukti yang SAH untuk continuation.
    # Tanpa exemption ini, "Modalnya sekitar 50 juta cukup?" (overlap token = 0,
    # hanya continuity) akan ditolak MIN_SUPPORTING_EVIDENCE dan memecah Scenario 1.
    # Efek exemption: confidence DI-CAP, bukan kandidat DITOLAK.
    CONTINUITY_EXEMPT_FROM_MIN_SUPPORT: Final[bool] = True


@dataclass(frozen=True)
class CandidateSafetyConstants:
    """
    Pengaman agar satu skor besar tidak menghasilkan kepercayaan palsu.
    """
    # Jika hanya 1 jenis bukti aktif, confidence tidak boleh melewati ini.
    SINGLE_SIGNAL_CONFIDENCE_CAP: Final[float] = 0.70

    # Jika hanya 2 jenis bukti aktif, confidence masih dibatasi.
    DUAL_SIGNAL_CONFIDENCE_CAP: Final[float] = 0.85

    # Margin minimum antara kandidat #1 dan #2 agar boleh dianggap tunggal.
    MIN_TOP_CANDIDATE_MARGIN: Final[float] = 0.05


@dataclass(frozen=True)
class CoOccurrenceConstants:
    """
    Co-occurrence naik dari literal-adjacent menjadi asosiatif berbasis window.

    Level 1 (Adjacent) : "bisnis kopi"        -> bisnis <-> kopi
    Level 2 (Window)   : "bisnis kecil di bidang kopi" -> bisnis <-> kopi tetap tercatat
    """
    CO_OCCURRENCE_WINDOW_SIZE: Final[int] = 5

    # Pasangan dengan kekuatan di bawah ini tidak dianggap evidence.
    CO_OCCURRENCE_MIN_STRENGTH: Final[float] = 0.10

    # Semakin jauh jarak token dalam window, semakin lemah hubungannya.
    # distance=1 -> 1.0, distance=2 -> 0.75, distance=3 -> 0.5625, dst.
    CO_OCCURRENCE_DECAY: Final[float] = 0.75

    # Pasangan bersebelahan langsung tetap mendapat nilai penuh.
    ADJACENT_PAIR_BONUS: Final[float] = 1.0


@dataclass(frozen=True)
class MemoryMutationConstants:
    """
    EVIDENCE BEFORE MUTATION.

    ANALYZE -> SCORE -> VALIDATE EVIDENCE -> DECIDE -> BARU UPDATE MEMORY

    Mencegah "evidence pollution": token C++/pointer masuk ke Token Profile,
    Co-occurrence, dan Helper Description milik cluster Bisnis Kopi.
    """
    # Confidence minimum agar boleh memperbarui token_profile & co_occurrence.
    # Diselaraskan dengan MIN_TOPIC_ACTIVATION_SCORE (0.15) agar kandidat yang
    # lolos activation gate tidak otomatis diblokir saat mutasi.
    MIN_EVIDENCE_TO_MUTATE_TOPIC: Final[float] = 0.15

    # Deskripsi adalah identitas manusia-readable -> butuh bukti lebih kuat.
    MIN_EVIDENCE_TO_UPDATE_DESCRIPTION: Final[float] = 0.30

    # Minimal jenis bukti pendukung agar mutasi diizinkan.
    MIN_SUPPORT_TO_MUTATE_TOPIC: Final[int] = 1

    # [GUARDRAIL EKSPLISIT] Continuation yang hanya didukung continuity tetap
    # boleh memperluas cluster (Scenario 1: Modal / Lokasi tanpa kata "kopi").
    # Yang DITOLAK adalah mutasi dengan bukti lemah ATAU kontradiksi tinggi.
    ALLOW_CONTINUITY_ONLY_MUTATION: Final[bool] = True

    # Kontradiksi di atas batas ini memblokir mutasi walaupun skor akhir tinggi.
    MUTATION_CONTRADICTION_BLOCK: Final[float] = 0.35


@dataclass(frozen=True)
class TokenQualityConstants:
    """
    Tidak semua token memiliki kekuatan identitas yang sama.

    "bagaimana" != "kopi" != "Rp50 juta"
    """
    # Token fungsional / kata tanya / kata ganti: nilai informasi rendah.
    LOW_INFORMATION_TOKENS: Final[frozenset] = frozenset({
        "saya", "aku", "kamu", "anda", "dia", "mereka", "kita", "kami",
        "yang", "dan", "atau", "untuk", "dengan", "dari", "pada", "tentang",
        "bagaimana", " gimana", "apa", "siapa", "kapan", "dimana", "mana",
        "mengapa", "kenapa", "berapa", "apakah", "itu", "ini", "tersebut",
        "ingin", "mau", "bisa", "boleh", "tolong", "coba", "saja", "dong",
        "yah", "nih", "sih", "deh", "lah", "kah", "pun", "juga", "lagi",
        "sudah", "belum", "akan", "sedang", "telah", "masih", "pernah",
        "bahas", "bahasan", "jelaskan", "sebut", "sebutkan", "bilang",
        "maksud", "maksudnya", "sebenarnya", "kira", "pikir", "rasa",
        "lanjutkan", "lanjut", "terus", "lalu", "kemudian", "jadi", "kalau",
        "nah", "selanjutnya", "berikutnya", "kembali", "tadi", "sebelumnya",
        "kita", "bahas", "buat", "bikin", "minta", "kasih", "beri",
    })

    # Token numerik / identifier: nilai identitas tertinggi.
    # Contoh: "rp50", "50", "juta", "15", "2023", "v2"
    NUMERIC_IDENTIFIER_PATTERN: Final[str] = r"^(rp)?\d+([.,]\d+)?$|^\d+[a-z]+$|^v\d+$"

    # Token >= panjang ini dan bukan low-information dianggap TOPIC SIGNAL.
    MIN_TOPIC_SIGNAL_LENGTH: Final[int] = 4

    # Token yang muncul di <= rasio ini dari seluruh Topic Key = RARE SIGNAL.
    RARE_TOKEN_DF_RATIO: Final[float] = 0.25

    # Bobot kualitas token (pengali, bukan pengganti ActivationWeights).
    LOW_INFORMATION_WEIGHT: Final[float] = 0.25
    NORMAL_WEIGHT: Final[float] = 1.0
    TOPIC_SIGNAL_WEIGHT: Final[float] = 1.25
    RARE_SIGNAL_WEIGHT: Final[float] = 1.5
    NUMERIC_IDENTIFIER_WEIGHT: Final[float] = 2.0


# ==========================================
# [REASONING PURITY] GRUP BARU
# Penguncian kemurnian hubungan antar-sinyal.
# Prinsip: skor yang menjawab satu pertanyaan tidak boleh dipakai untuk
# menjawab pertanyaan lain, dan banyak sinyal dari sumber sama tidak
# otomatis berarti banyak bukti independen.
# ==========================================

@dataclass(frozen=True)
class EvidenceFamilyConstants:
    """
    Evidence Family Independence (Prioritas #2).

    Token + Unique + Co-occurrence = 1 keluarga (LEXICAL), bukan 3 evidence.
    Description + Temporal + Continuity = 1 keluarga (CONTEXTUAL).
    Graph / Cross-session = 1 keluarga (RELATIONAL).

    support_count lama = jumlah sinyal aktif (raw)
    family_support_count = jumlah keluarga berbeda yang aktif (independen)
    """
    # Minimal berapa keluarga bukti yang aktif agar confidence penuh.
    # Di-set 1 (bukan 2) karena CONTINUITY_EXEMPT_FROM_FAMILY_CHECK
    # melindungi Scenario 1. Confidence tetap di-cap oleh SINGLE_FAMILY_*.
    MIN_EVIDENCE_FAMILIES: Final[int] = 1

    # Jika hanya 1 keluarga bukti aktif, confidence tidak boleh melewati ini.
    # 0.80 dipilih agar lebih permisif dari SINGLE_SIGNAL_CONFIDENCE_CAP (0.70)
    # karena "satu keluarga" bisa berisi banyak sinyal (e.g. 3 lexical).
    SINGLE_FAMILY_CONFIDENCE_CAP: Final[float] = 0.80

    # Bobot keluarga saat menghitung evidence_strength.
    # RELATIONAL diberi bobot tertinggi karena paling independen dari query.
    LEXICAL_FAMILY_WEIGHT: Final[float] = 0.30
    CONTEXTUAL_FAMILY_WEIGHT: Final[float] = 0.35
    RELATIONAL_FAMILY_WEIGHT: Final[float] = 0.35

    # [GUARDRAIL] Continuity = CONTEXTUAL family yang sah.
    # Query "Modalnya sekitar 50 juta cukup?" terhadap "Bisnis Kopi" hanya
    # memiliki 1 keluarga (CONTEXTUAL via continuity). Tanpa exemption,
    # kandidat ini jatuh ke INSUFFICIENT_EVIDENCE dan cluster pecah.
    CONTINUITY_EXEMPT_FROM_FAMILY_CHECK: Final[bool] = True

    # Dominasi: jika satu keluarga menyumbang >= rasio ini dari total strength.
    SINGLE_FAMILY_DOMINANCE_RATIO: Final[float] = 0.80


@dataclass(frozen=True)
class HF_CMConfig:
    """Global Configuration Aggregator."""
    input: Final[InputLimits] = InputLimits()
    fast_memory: Final[FastMemoryLimits] = FastMemoryLimits()
    signal: Final[SignalConstants] = SignalConstants()
    activation: Final[ActivationWeights] = ActivationWeights()
    decision: Final[DecisionThresholds] = DecisionThresholds()
    temporal: Final[TemporalConstants] = TemporalConstants()
    retrieval: Final[RetrievalLimits] = RetrievalLimits()
    compression: Final[CompressionConstants] = CompressionConstants()
    context: Final[ContextBudget] = ContextBudget()
    llm: Final[LLMConstants] = LLMConstants()
    continuity: Final[ContinuityConstants] = ContinuityConstants()
    topic_switch: Final[TopicSwitchConstants] = TopicSwitchConstants() # [NEW]
    fragment_ranking: Final[FragmentRankingConstants] = FragmentRankingConstants() # [NEW]
    graph: Final[GraphConstants] = GraphConstants()
    storage: Final[StorageConstants] = StorageConstants()
    # [ASSOCIATIVE EVIDENCE LOCKING] Grup pengendali kualitas bukti
    evidence: Final[AssociativeEvidenceConstants] = AssociativeEvidenceConstants()
    candidate_safety: Final[CandidateSafetyConstants] = CandidateSafetyConstants()
    co_occurrence: Final[CoOccurrenceConstants] = CoOccurrenceConstants()
    mutation: Final[MemoryMutationConstants] = MemoryMutationConstants()
    token_quality: Final[TokenQualityConstants] = TokenQualityConstants()
    # [REASONING PURITY] Grup pengendali independensi bukti
    evidence_family: Final[EvidenceFamilyConstants] = EvidenceFamilyConstants()

CONFIG: Final[HF_CMConfig] = HF_CMConfig()
