"""
config.py
Pusat Pengendali Keputusan untuk Hybrid Fragmented Compressed Memory (HF-CM).

[BATCH A2 - ASSOCIATION EXPANSION]
Menambahkan 2 grup constants baru:
    - AssociationConstants        : routing local-first, cross-topic, cross-fragment
    - ResponseMetadataConstants   : kontrak metadata dari LLM

Constants ini mendukung 3 file baru yang akan ditambahkan:
    - response_metadata.py  (Batch B)
    - topic_router.py       (Batch D)
    - association_engine.py (Batch E)
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
    # [FIX — Batch B] Noise floor diturunkan dari 0.05 → 0.01.
    #
    # Alasan: turn_continuity_score untuk continuation yang SAH tapi lemah
    # (overlap nol + novelty tinggi) bernilai sekitar 0.05:
    #     0.25 (history) - 0.20 (novelty penalty) = 0.04999999999999999
    # Dengan floor 0.05, edge CONTINUES yang legit ini dibuang karena
    # presisi floating point (0.0499... < 0.05). Floor 0.01 tetap membuang
    # noise near-zero tanpa membuang bukti continuation lemah yang valid.
    FRAGMENT_GRAPH_MIN_SCORE: Final[float] = 0.01

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
        # [FIX — Batch B] Ambiguous clarification responses.
        # Kata-kata ini adalah RESPONS terhadap pertanyaan klarifikasi,
        # bukan sinyal identitas topik. Tanpa ini, "terserah" (8 huruf)
        # diklasifikasikan TOPIC_SIGNAL dan dianggap meaningful token,
        # sehingga pending di-pause padahal user masih menjawab klarifikasi.
        "terserah", "terserahmu", "bebas", "entah", "ragu", "bingung",
        "pilih", "pilihkan", "silakan", "silahkan", "apapun", "apaun",
        "manapun", "semua", "semuanya", "dua", "duanya", "salah",
        # Affirmation / negation pendek terhadap klarifikasi.
        "ya", "iya", "oke", "ok", "baik", "baiklah", "setuju", "benar",
        "betul", "tidak", "bukan", "jangan", "gak", "nggak", "enggak",
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


# ==========================================
# [HIERARCHICAL FRAGMENT EXPANSION] GRUP BARU
#
# Fragment bukan lagi daftar datar. Ia adalah node hierarkis beridentitas
# dengan parent, depth, fragment_key, dan helper description sendiri.
#
#     TOPIC KEY: Bisnis Kopi
#         ├── [depth 0] Pembukaan Bisnis Kopi
#         ├── [depth 1] Modal Bisnis Kopi
#         │       ├── [depth 2] Modal Murah vs Mahal
#         │       └── [depth 2] Harga Pasaran Terkini
#         └── [depth 1] Lokasi Usaha
#
# Setiap constant di grup ini punya SATU penggunaan nyata dan SATU perilaku
# yang bisa diuji. Tidak ada constant dekoratif.
# ==========================================

@dataclass(frozen=True)
class FragmentHierarchyConstants:
    """
    [MEKANISME 3 + 5] Struktur hierarki fragment.

    DEPTH SEMANTICS (DIKUNCI):
        depth 0 = root / opening fragment
        depth 1 = section utama
        depth 2 = child spesifik
        depth 3 = spesialisasi lebih dalam (hanya jika MAX_FRAGMENT_DEPTH >= 3)

    MAX_FRAGMENT_DEPTH adalah HARD STRUCTURAL BOUNDARY, bukan saran.
    """
    # Batas kedalaman struktural. Default 2 sesuai spesifikasi:
    #   root(0) → section(1) → child spesifik(2)
    # Fragment baru TIDAK boleh menciptakan depth 3 dengan setting ini.
    MAX_FRAGMENT_DEPTH: Final[int] = 2

    # Skor minimum agar fragment baru ditempatkan sebagai CHILD dari parent itu.
    # 0.30 berarti ~2 token overlap dari 4-5 token bermakna sudah cukup,
    # sehingga "Modal kopi murah vs mahal" bisa masuk di bawah "Modal Bisnis Kopi".
    MIN_PARENT_MATCH_SCORE: Final[float] = 0.30

    # Margin minimum parent terbaik vs runner-up.
    # Mencegah penempatan arbitrer ketika dua parent sama-sama cocok.
    MIN_CHILD_MATCH_MARGIN: Final[float] = 0.08

    # Depth untuk branch baru ketika TIDAK ada parent yang cukup kuat.
    # Bukan 0 — root hanya untuk fragment pembuka TopicKey.
    NEW_BRANCH_DEPTH: Final[int] = 1

    # Maksimum fragment yang ditampilkan saat trend navigation di dalam TopicKey.
    MAX_TREND_FRAGMENTS: Final[int] = 5

    # Maksimum kandidat ambiguity yang ditampilkan ke user.
    MAX_AMBIGUITY_CANDIDATES: Final[int] = 5

    # Batas panjang fragment_helper_description (karakter).
    # Helper adalah metadata retrieval, bukan salinan payload.
    MAX_FRAGMENT_HELPER_LENGTH: Final[int] = 200

    # Maksimum kontribusi child yang boleh masuk ke helper parent.
    # Mencegah parent dengan 100 child membengkak tanpa batas.
    MAX_PARENT_HELPER_CONTRIBUTIONS: Final[int] = 5

    # Jumlah token maksimum untuk fragment_key (nama bagian).
    FRAGMENT_KEY_MAX_TOKENS: Final[int] = 5

    #Traversal ancestor maksimum saat propagasi (safety bound terhadap siklus).
    MAX_ANCESTOR_TRAVERSAL: Final[int] = 10


@dataclass(frozen=True)
class TrendConstants:
    """
    [MEKANISME 5] Trend-based continuation selection.

    Dipakai HANYA ketika user meminta continuation tanpa menunjuk bagian spesifik:
        "Lanjutkan topik sebelumnya" → top-N TopicKey
        "Lanjutkan bisnis kopi"      → top-N fragment depth tertentu

    ATURAN UTAMA (DIKUNCI):
        Jumlah anak / fragment adalah SINYAL TREND UTAMA.
        Recency dan temporal hanya bukti TAMBAHAN.

    Karena itu CHILD_COUNT_WEIGHT (0.60) > RECENCY + TEMPORAL (0.40).

    PENTING: banyak child ≠ pasti maksud user.
    Trend = navigation / candidate prioritization, BUKAN final topic identity.
    """
    # Batas kandidat yang ditampilkan.
    TREND_TOPIC_LIMIT: Final[int] = 5
    TREND_FRAGMENT_LIMIT: Final[int] = 5

    # Bobot sinyal. Total = 1.00.
    TREND_CHILD_COUNT_WEIGHT: Final[float] = 0.60   # SINYAL UTAMA
    TREND_RECENCY_WEIGHT: Final[float] = 0.20       # bukti tambahan
    TREND_TEMPORAL_WEIGHT: Final[float] = 0.20      # bukti tambahan

    # Normalisasi child_count agar 500 child tidak mendominasi tak terbatas.
    # child_count dinormalisasi: min(count, CAP) / CAP
    TREND_COUNT_NORMALIZATION_CAP: Final[int] = 20

    # Decay recency untuk trend (per hari).
    TREND_RECENCY_DECAY_DAYS: Final[float] = 30.0

    # Trend expansion HANYA jika current_depth < MAX_FRAGMENT_DEPTH.
    # Setelah user memilih node di depth maksimum, tidak ada expansion lagi.
    ENABLE_DEPTH_AWARE_EXPANSION: Final[bool] = True


@dataclass(frozen=True)
class ClarificationAccumulationConstants:
    """
    [MEKANISME 1] Recursive Ambiguity Context Accumulation.

    Ambiguity bukan "satu pertanyaan → satu jawaban → evaluasi".
    Ia adalah STATE YANG BERKEMBANG:

        original_input + history[0] + history[1] + ... + negative evidence
            ↓
        re-rank → candidate set MENYEMPIT

    Retrieval TIDAK boleh hanya menganalisis jawaban terakhir.
    "yang rencana kopi" sendiri terlalu miskin bukti.
    """
    # Batas total karakter akumulasi (original + seluruh history).
    CLARIFICATION_HISTORY_MAX_CHARS: Final[int] = 2000

    # Maksimum entri history yang disimpan.
    MAX_CLARIFICATION_HISTORY_ENTRIES: Final[int] = 10

    # Kekuatan penalti untuk kandidat yang terkena negative evidence.
    # "yang penting bukan topik cpp" → kandidat C++ dipenalti sebesar ini.
    CLARIFICATION_NEGATION_WEIGHT: Final[float] = 0.50

    # Bobot combined context (original + history) vs jawaban terakhir saja.
    # 0.70 berarti konteks akumulatif mendominasi, jawaban terakhir melengkapi.
    CLARIFICATION_COMBINED_INPUT_WEIGHT: Final[float] = 0.70

    # Kandidat dengan skor di bawah ini DIELIMINASI dari shortlist berikutnya.
    # Contoh: A=0.81, B=0.79, C=0.20 → C gugur, round berikut hanya A & B.
    MIN_CANDIDATE_RETENTION_SCORE: Final[float] = 0.35

    # Kandidat juga harus berada dalam rasio ini dari skor tertinggi.
    # 0.60 × 0.81 = 0.486 → C=0.20 tetap gugur.
    CANDIDATE_RETENTION_RATIO: Final[float] = 0.60

    # Jumlah maksimum negation scope yang diproses per input.
    MAX_NEGATION_SCOPES: Final[int] = 5

    # Panjang maksimum target negation (token) setelah marker.
    # "bukan topik cpp" → target ["cpp"], bukan seluruh sisa kalimat.
    NEGATION_SCOPE_TOKEN_WINDOW: Final[int] = 3


@dataclass(frozen=True)
class ContinuationIntentConstants:
    """
    [MEKANISME 2] Continuation Intent Protection.

    NO_MATCH + continuation intent ≠ NEW_TOPIC.

    User yang berkata "lanjutkan topik ini" tetapi sistem belum tahu topiknya
    TIDAK boleh dianggap membuka topik baru. Sistem masuk state
    UNRESOLVED_CONTINUATION dan TIDAK mempersist fragment TopicKey baru.
    """
    # Marker multi-kata (dicocokkan sebagai substring pada raw text lower).
    CONTINUATION_PHRASE_MARKERS: Final[frozenset] = frozenset({
        "topik sebelumnya", "topik sebelumnya.", "pembahasan sebelumnya",
        "yang kemarin", "pembahasan tadi", "yang kita bahas",
        "topik ini", "topik tadi", "bahasan tadi", "yang tadi",
        "lanjutkan topik", "lanjut topik", "lanjutkan pembahasan",
        "lanjut pembahasan", "lanjutkan yang", "lanjut yang",
        "kembali ke topik", "balik ke topik",
    })

    # Marker satu kata (dicocokkan sebagai token).
    CONTINUATION_TOKEN_MARKERS: Final[frozenset] = frozenset({
        "lanjutkan", "lanjut", "teruskan", "sambung",
    })

    # Skor minimum kandidat agar continuation dianggap TERSELESAIKAN.
    # Di bawah ini → UNRESOLVED_CONTINUATION (bukan NEW_TOPIC).
    MIN_CONTINUATION_CANDIDATE_SCORE: Final[float] = 0.25

    # Margin antar kandidat continuation untuk memutuskan ambiguous vs clear.
    CONTINUATION_AMBIGUITY_MARGIN: Final[float] = 0.05

    # [GUARDRAIL UTAMA] Blokir pembuatan TopicKey baru saat continuation intent
    # tidak terselesaikan. Ini inti Mekanisme 2.
    BLOCK_NEW_TOPIC_ON_CONTINUATION: Final[bool] = True

    # [MEKANISME 1] Explicit cancellation / topic switch.
    # Input yang jelas memilih topik lain TIDAK boleh dipaksa menjadi
    # jawaban ambiguity.
    CANCELLATION_PHRASE_MARKERS: Final[frozenset] = frozenset({
        "ganti topik", "lupakan saja", "tidak jadi", "bahas yang lain",
        "bahas topik lain", "topik lain saja", "sudahlah", "skip saja",
        "lewati saja", "batal saja", "gak jadi", "nggak jadi",
    })

    CANCELLATION_TOKEN_MARKERS: Final[frozenset] = frozenset({
        "lupakan", "batal", "skip", "lewati", "cancel",
    })

    # Kekuatan bukti bahwa user benar-benar berpindah topik.
    CANCELLATION_SWITCH_WEIGHT: Final[float] = 0.80


@dataclass(frozen=True)
class CompositeSourceConstants:
    """
    [MEKANISME 4] Cross-Key Composite Fragment.

    Satu respons baru dapat berasal dari beberapa fragment/topik berbeda.

    OWNERSHIP TETAP SATU:
        primary_topic_key_id  (single foreign key, tidak dipecah)
        + source_fragment_ids (N sumber)
        + N graph edges       (REFERENCES / RELATED_TO)

    BUKAN dual primary TopicKey — itu akan merusak foreign-key ownership.
    """
    # Skor minimum agar sebuah fragment dianggap BERKONTRIBUSI.
    MIN_SOURCE_CONTRIBUTION_SCORE: Final[float] = 0.35

    # Minimal berapa sumber berbeda agar disebut composite.
    MIN_DISTINCT_SOURCES: Final[int] = 2

    # Maksimum sumber yang dicatat (mencegah ledger membengkak).
    MAX_SOURCE_FRAGMENTS: Final[int] = 5

    # [KUNCI ANTI-FALSE-COMPOSITE] Sumber harus didukung >= 2 keluarga bukti.
    # Token overlap saja TIDAK cukup — kalau tidak, semua fragment bertema
    # "kopi" akan otomatis saling menjadi source.
    MIN_SOURCE_EVIDENCE_FAMILIES: Final[int] = 2

    # Overlap token minimum sebagai salah satu bukti kontribusi.
    MIN_SOURCE_TOKEN_OVERLAP: Final[float] = 0.25

    # Bonus kontribusi jika sumber berasal dari TopicKey berbeda.
    CROSS_TOPIC_CONTRIBUTION_BONUS: Final[float] = 0.15

    # Tipe edge yang dibuat dari sumber ke fragment composite.
    COMPOSITE_EDGE_TYPE: Final[str] = "REFERENCES"


@dataclass(frozen=True)
class HelperPropagationConstants:
    """
    [MEKANISME 3] Hierarchical Helper Aggregation.

    Alur:
        RAW FRAGMENT TEXT
            → EXTRACT OWN HELPER   (SEBELUM kompresi)
            → PERSIST EXACT PAYLOAD (lossless)
            → ATTACH TO HIERARCHY
            → AGGREGATE CHILD HELPER INTO PARENT
            → PROPAGATE UPWARD
            → REPEAT UNTIL ROOT
            → UPDATE TOPIC KEY HELPER

    YANG NAIK BUKAN SELURUH ISI CHILD.
    Yang naik: representasi ringkas, terkontrol, deduplicated.

    ❌ parent.helper += child.helper  (tanpa batas → metadata pollution)
    ✅ parent.aggregated = own + selected representative child evidence
    """
    # Batas panjang aggregated helper (karakter).
    MAX_HELPER_LENGTH: Final[int] = 240

    # Maksimum entri descendant evidence per node.
    MAX_DESCENDANT_EVIDENCE: Final[int] = 5

    # Ambang kemiripan untuk menganggap dua helper contribution DUPLIKAT.
    # Jaccard-style token overlap. 0.75 = 75% token sama → dianggap duplikat.
    # Mencegah "modal kopi murah", "modal kopi murah terbaru",
    # "modal kopi murah terbaru 2026" menumpuk sebagai 3 entri terpisah.
    DEDUP_SIMILARITY_THRESHOLD: Final[float] = 0.75

    # Batas rantai propagasi (safety terhadap siklus / hierarki sangat dalam).
    MAX_PROPAGATION_DEPTH: Final[int] = 5

    # Panjang token minimum yang boleh masuk helper.
    MIN_HELPER_TOKEN_LENGTH: Final[int] = 3

    # Maksimum token per kontribusi child (ringkas, bukan paragraf).
    MAX_HELPER_CONTRIBUTION_TOKENS: Final[int] = 12

    # Apakah propagasi mencapai TopicKey.helper_description.
    PROPAGATE_TO_TOPIC_KEY: Final[bool] = True

    # [GUARDRAIL] Propagasi TUNDUK pada mutation gate.
    # Input low-information ("Bagaimana itu?") tidak boleh mengubah helper
    # parent maupun TopicKey. Ini menjaga invariant test_evidence_7.
    REQUIRE_MUTATION_PERMISSION: Final[bool] = True

    # Prefix untuk bagian agregat agar manusia bisa membedakan own vs warisan.
    CHILD_SUMMARY_PREFIX: Final[str] = "Mencakup: "

    # Pemisah antar kontribusi child dalam aggregated helper.
    CONTRIBUTION_SEPARATOR: Final[str] = "; "


# ==========================================
# [ASSOCIATION EXPANSION - Batch A2] GRUP BARU
#
# Constants untuk mendukung mekanisme baru:
#   - ResponseMetadata (LLM-generated identity)
#   - TopicRouter (local-first routing)
#   - AssociationEngine (cross-topic & cross-fragment relationships)
#
# Tiga lapisan konsep yang DIKUNCI:
#   1. Ownership      : topic_key_id (single, structural)
#   2. Hierarchy      : parent_fragment_id + depth
#   3. Association    : banyak target lintas topik/fragment
#
# Setiap constant di grup ini punya SATU penggunaan nyata di
# response_metadata.py / topic_router.py / association_engine.py.
# Tidak ada constant dekoratif.
# ==========================================

@dataclass(frozen=True)
class ResponseMetadataConstants:
    """
    [ASSOCIATION EXPANSION] Kontrak metadata dari LLM.

    ResponseMetadata adalah structured output yang dihasilkan LLM
    bersamaan dengan response_text. Metadata ini berisi:
        - fragment_key (identitas internal)
        - helper_description
        - topic/fragment hints
        - positive/negative clues
        - confidence

    Prinsip kunci:
        LLM menentukan identitas fragment SEBELUM respons disimpan,
        bukan sistem menebak identitas itu dari heuristic SETELAH
        jawaban selesai.

    Namun LLM bisa salah / hallucinate / provider bisa tidak mendukung
    structured output, sehingga validator diperlukan.
    """
    # Confidence minimum agar metadata dipercaya sebagai PRIMARY identity.
    # Di bawah threshold ini, fallback ke heuristic FragmentIdentityExtractor.
    # 0.45 dipilih agar tidak terlalu permisif (hallucination) tapi juga
    # tidak terlalu ketat (metadata valid tapi low-confidence terbuang).
    MIN_METADATA_CONFIDENCE: Final[float] = 0.45

    # Batas panjang fragment_key (karakter).
    # fragment_key adalah NAMA BAGIAN singkat (e.g. "Modal Bisnis Kopi"),
    # bukan paragraf. Ini BERBEDA dengan MAX_FRAGMENT_HELPER_LENGTH (200)
    # yang berlaku untuk helper_description.
    MAX_FRAGMENT_KEY_LENGTH: Final[int] = 80

    # Batas panjang helper_description dari metadata LLM.
    # Sama dengan MAX_FRAGMENT_HELPER_LENGTH di hierarchy (sinkronisasi).
    MAX_HELPER_DESCRIPTION_LENGTH: Final[int] = 200

    # Batas panjang primary_topic_hint (karakter).
    # Hint biasanya singkat: nama TopicKey yang sudah dikenal.
    MAX_TOPIC_HINT_LENGTH: Final[int] = 100

    # Maksimum jumlah positive_clues yang dipertahankan.
    # Mencegah metadata bloated dengan daftar clue yang tak terbatas.
    MAX_POSITIVE_CLUES: Final[int] = 10

    # Maksimum jumlah negative_clues yang dipertahankan.
    MAX_NEGATIVE_CLUES: Final[int] = 5

    # Maksimum jumlah related_topic_hints.
    MAX_RELATED_TOPIC_HINTS: Final[int] = 5

    # Maksimum jumlah related_fragment_hints.
    MAX_RELATED_FRAGMENT_HINTS: Final[int] = 5

    # Izinkan metadata PARSIAL: sebagian field valid, sebagian ditolak.
    # True berarti: fragment_key valid + topic_hints invalid → fragment_key
    # tetap dipakai, topic_hints dibuang.
    # False berarti: satu field invalid → seluruh metadata ditolak.
    # Disarankan True untuk robustness.
    ALLOW_PARTIAL_METADATA: Final[bool] = True

    # Apakah topic/fragment hint HARUS match dengan known target.
    # True  : hint "Bisnis Kopi" hanya valid jika TopicKey itu ada.
    # False : hint boleh berupa nama bebas, dipakai sebagai soft signal.
    # Disarankan False karena LLM bisa menggunakan nama yang tidak persis
    # match (synonyms, abbreviations) tapi masih valid secara semantic.
    REQUIRE_KNOWN_TARGET_FOR_HINT: Final[bool] = False

    # Fallback ke heuristic FragmentIdentityExtractor jika metadata:
    #   - None (provider tidak mendukung)
    #   - malformed (parse error)
    #   - confidence < MIN_METADATA_CONFIDENCE
    # True = fallback aman (sistem tetap jalan).
    # False = strict mode (metadata invalid = fragment tidak ter-identity).
    METADATA_FALLBACK_ON_MALFORMED: Final[bool] = True

    # Versi kontrak metadata saat ini.
    # Jika LLM mengirim metadata_version != ini, validator bisa:
    #   - accept (backward compat)
    #   - warn (deprecation)
    #   - reject (breaking change)
    CURRENT_METADATA_VERSION: Final[str] = "1"

    # Skor minimum untuk menganggap satu clue (positive/negative) valid.
    # Di bawah ini clue di-discard sebagai noise.
    MIN_CLUE_STRENGTH: Final[float] = 0.20


@dataclass(frozen=True)
class AssociationConstants:
    """
    [ASSOCIATION EXPANSION] Threshold & bobot untuk association engine.

    Tiga lapis keputusan:
        1. LOCAL ROUTING : active topic cocok? -> ACTIVE_LOCAL_MATCH
        2. TOPIC ASSOC   : fragment baru terkait ke TopicKey lain?
        3. FRAGMENT ASSOC: fragment baru terkait ke fragment lain?

    Masing-masing punya threshold tersendiri, TIDAK BOLEH digabung,
    karena granularity-nya berbeda.

    Prinsip kunci:
        Association TIDAK menggantikan Ownership (single topic_key_id)
        maupun Hierarchy (single parent_fragment_id).
        Association adalah LAYER TAMBAHAN untuk cross-topic & cross-fragment.
    """
    # [LOCAL ROUTING] Threshold agar active topic dianggap cocok.
    # Jika active_topic_score >= ini → ACTIVE_LOCAL_MATCH (skip global search).
    # 0.45 dipilih lebih tinggi dari MIN_TOPIC_ACTIVATION_SCORE (0.15)
    # karena local routing adalah KEPUTUSAN UTAMA, bukan sekadar lolos gate.
    ACTIVE_TOPIC_MATCH_THRESHOLD: Final[float] = 0.45

    # [LOCAL ROUTING] Margin minimum antara active topic dan kandidat global
    # terbaik agar local match menang. Mencegah kasus:
    #   active score = 0.50, global best = 0.48 → local menang (margin OK)
    #   active score = 0.50, global best = 0.55 → global menang
    LOCAL_VS_GLOBAL_MARGIN: Final[float] = 0.03

    # [TOPIC ASSOCIATION] Threshold agar fragment baru dianggap terkait ke
    # TopicKey lain (selain primary ownership-nya).
    # Lebih rendah dari ACTIVE_TOPIC_MATCH_THRESHOLD karena association
    # adalah bukti TAMBAHAN, bukan routing decision.
    TOPIC_ASSOCIATION_THRESHOLD: Final[float] = 0.35

    # [FRAGMENT ASSOCIATION] Threshold agar fragment baru dianggap terkait
    # ke fragment spesifik lain.
    # Lebih rendah dari TOPIC_ASSOCIATION_THRESHOLD karena fragment lebih
    # spesifik dan match fragment-level adalah bukti lebih kuat.
    FRAGMENT_ASSOCIATION_THRESHOLD: Final[float] = 0.30

    # [CLUE WEIGHTS] Bonus/penalty per clue type.
    # POSITIVE_CLUE_BONUS ditambahkan ke association score per clue match.
    POSITIVE_CLUE_BONUS: Final[float] = 0.10
    # NEGATIVE_CLUE_PENALTY dikurangkan dari association score per clue match.
    NEGATIVE_CLUE_PENALTY: Final[float] = 0.15
    # METADATA_HINT_BONUS: bonus jika related_topic_hints/fragment_hints match.
    METADATA_HINT_BONUS: Final[float] = 0.12

    # [DOMINANCE] Margin minimum agar sumber dianggap PRIMARY (vs SECONDARY).
    # Jika score_primary - score_secondary >= ini → PRIMARY.
    # Jika kurang → keduanya SECONDARY (tidak ada yang dominan).
    DOMINANT_SOURCE_MARGIN: Final[float] = 0.15

    # [DOMINANCE] Skor minimum untuk dianggap PRIMARY.
    # Jika skor tertinggi di bawah ini, semua sumber dianggap SECONDARY.
    MIN_PRIMARY_SCORE: Final[float] = 0.60

    # [LIMITS] Batas jumlah associations per fragment.
    MAX_TOPIC_ASSOCIATIONS: Final[int] = 5
    MAX_FRAGMENT_ASSOCIATIONS: Final[int] = 5
    MAX_DOMINANT_SOURCES: Final[int] = 3

    # [ANTI-FALSE-ASSOCIATION] Sama seperti composite (MIN_SOURCE_EVIDENCE_FAMILIES):
    # association harus didukung >= 2 keluarga bukti. Token overlap saja tidak cukup.
    MIN_ASSOCIATION_EVIDENCE_FAMILIES: Final[int] = 2

    # [CROSS-TOPIC] Threshold khusus untuk cross-topic association.
    # Lebih rendah dari TOPIC_ASSOCIATION_THRESHOLD biasa karena
    # cross-topic signal sudah inherently lebih kuat (lintas context).
    CROSS_TOPIC_ASSOCIATION_THRESHOLD: Final[float] = 0.30

    # [ASSOCIATION EVIDENCE] Minimum token overlap ratio untuk mempertimbangkan
    # association. Di bawah ini, langsung ditolak tanpa perlu cek keluarga bukti.
    MIN_ASSOCIATION_TOKEN_OVERLAP: Final[float] = 0.20

    # [EXCLUSION] Skor penalti untuk target yang terkena EXCLUSION clue.
    # EXCLUSION = hard rejection (dari negation scope).
    # Target yang kena exclusion langsung di-drop (score = 0).
    # Nilai ini untuk observability saja, bukan perhitungan.
    EXCLUSION_PENALTY: Final[float] = 1.0


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
    # [HIERARCHICAL FRAGMENT EXPANSION] Grup pengendali 5 mekanisme baru
    hierarchy: Final[FragmentHierarchyConstants] = FragmentHierarchyConstants()
    trend: Final[TrendConstants] = TrendConstants()
    clarification: Final[ClarificationAccumulationConstants] = ClarificationAccumulationConstants()
    continuation: Final[ContinuationIntentConstants] = ContinuationIntentConstants()
    composite: Final[CompositeSourceConstants] = CompositeSourceConstants()
    helper_propagation: Final[HelperPropagationConstants] = HelperPropagationConstants()
    # [ASSOCIATION EXPANSION - Batch A2] Grup pengendali metadata LLM & association
    response_metadata: Final[ResponseMetadataConstants] = ResponseMetadataConstants()
    association: Final[AssociationConstants] = AssociationConstants()

CONFIG: Final[HF_CMConfig] = HF_CMConfig()
