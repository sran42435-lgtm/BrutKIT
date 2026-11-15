"""
models.py
Kontrak Data & State Machine untuk Hybrid Fragmented Compressed Memory (HF-CM).

File ini mendefinisikan bentuk data yang dipertukarkan antar-layer.
ATURAN MUTLAK: Semua pertukaran data harus menggunakan Pydantic models di bawah ini.

[HIERARCHICAL FRAGMENT EXPANSION]
Fragment bukan lagi daftar datar di bawah TopicKey. Fragment adalah NODE HIERARKIS
yang memiliki identitas bagian sendiri (fragment_key), helper description sendiri,
parent, depth, dan hubungan lintas-key.

    TOPIC KEY: Bisnis Kopi
        │
        ├── [depth 0] Pembukaan Bisnis Kopi
        ├── [depth 1] Modal Bisnis Kopi
        │       ├── [depth 2] Modal Murah vs Mahal
        │       └── [depth 2] Harga Pasaran Terkini
        └── [depth 1] Lokasi Usaha

Helper Description mengalir ke ATAS (bukan payload):
    CHILD HELPER → PARENT AGGREGATE → ANCESTOR → ROOT → TOPIC KEY HELPER

Exact memory tetap lossless di payload. Yang naik hanya representasi ringkas.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

# ==========================================
# ENUMS (State & Decision Definitions)
# ==========================================

class PendingStatus(str, Enum):
    ACTIVE = "ACTIVE"         # Menunggu klarifikasi user
    RESOLVED = "RESOLVED"     # Ambiguitas terselesaikan
    PAUSED = "PAUSED"         # User berpindah topik, pending ditunda
    EXPIRED = "EXPIRED"       # Melewati batas waktu/round

class DecisionType(str, Enum):
    CLEAR = "CLEAR"           # Kandidat tunggal yang kuat
    AMBIGUOUS = "AMBIGUOUS"   # Selisih skor terlalu tipis, wajib klarifikasi
    INSUFFICIENT = "INSUFFICIENT" # Bukti terlalu lemah
    NO_MATCH = "NO_MATCH"     # Tidak ada key yang cocok
    NEW_TOPIC = "NEW_TOPIC"   # Topik baru yang eksplisit
    # [ASSOCIATIVE EVIDENCE LOCKING] Dua state baru.
    # Ketiganya TIDAK sama dan tidak boleh diperlakukan sama:
    #   AMBIGUOUS              -> ada beberapa kandidat kuat
    #   INSUFFICIENT_EVIDENCE  -> tidak ada kandidat yang cukup terbukti
    #   CONTRADICTORY_EVIDENCE -> ada kecocokan, tetapi bukti saling bertentangan
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    # [MEKANISME 2] Continuation Intent Protection.
    #
    # User JELAS meminta melanjutkan sesuatu ("lanjutkan topik sebelumnya"),
    # tetapi sistem tidak punya bukti cukup untuk menentukan APA yang dilanjutkan.
    #
    # Ini BUKAN NO_MATCH dan BUKAN NEW_TOPIC:
    #   NO_MATCH + continuation intent ≠ NEW_TOPIC
    #
    # Konsekuensi persistence: input ini TIDAK boleh menjadi fragment TopicKey baru,
    # dan respons AI-nya juga TIDAK boleh dipersist sebagai fragment TopicKey baru,
    # sampai candidate ditemukan / ambiguity jelas / user eksplisit buka topik baru.
    UNRESOLVED_CONTINUATION = "UNRESOLVED_CONTINUATION"

class MemoryClassification(str, Enum):
    CONTINUE_TOPIC = "CONTINUE_TOPIC"   # Melanjutkan Topic Key aktif
    NEW_TOPIC = "NEW_TOPIC"             # Membentuk cluster topik baru
    # [NEW] Ditambahkan untuk mendukung skenario reaktivasi (Coffee -> C++ -> Coffee)
    REACTIVATE_TOPIC = "REACTIVATE_TOPIC" 
    CLARIFICATION_ONLY = "CLARIFICATION_ONLY" # Hanya kontrol state, tidak masuk memori besar
    DISCARD = "DISCARD"                 # Tidak layak disimpan (noise)

class RelationshipType(str, Enum):
    CONTINUES = "CONTINUES"
    RELATED_TO = "RELATED_TO"
    REFERENCES = "REFERENCES"
    CROSS_SESSION_LINK = "CROSS_SESSION_LINK"

# [NEW] Enum untuk observabilitas rekonstruksi memori (Review Kedua)
class ReconstructionStatus(str, Enum):
    FULL_SUCCESS = "FULL_SUCCESS"       # Semua fragment berhasil didekompresi
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS" # Sebagian fragment gagal (korup/hilang), sebagian berhasil
    FAILED = "FAILED"                   # Tidak ada memori yang berhasil direkonstruksi
    NO_MEMORY_REQUESTED = "NO_MEMORY_REQUESTED" # Tidak ada permintaan memori

# [ASSOCIATIVE EVIDENCE LOCKING] Tier kualitas token.
# "bagaimana" tidak boleh bernilai setara dengan "kopi" atau "Rp50 juta".
class TokenQuality(str, Enum):
    LOW_INFORMATION = "LOW_INFORMATION"       # saya, ingin, bagaimana, tentang
    NORMAL = "NORMAL"                         # token informasi biasa
    TOPIC_SIGNAL = "TOPIC_SIGNAL"             # kopi, bisnis, pointer
    RARE_SIGNAL = "RARE_SIGNAL"               # muncul di sangat sedikit Topic Key
    NUMERIC_IDENTIFIER = "NUMERIC_IDENTIFIER" # rp50, juta, 15, 2023


# [REASONING PURITY] Evidence Family (Prioritas #2).
# Tiga sumber bukti yang benar-benar INDEPENDEN secara epistemik.
# Banyak sinyal dari keluarga yang sama TIDAK dihitung sebagai bukti ganda.
class EvidenceFamily(str, Enum):
    LEXICAL = "LEXICAL"             # Token, Unique Token, Co-occurrence
    CONTEXTUAL = "CONTEXTUAL"       # Helper Description, Temporal, Continuity
    RELATIONAL = "RELATIONAL"       # Graph, Cross-session relationship


# [MEKANISME 5] Jenis node untuk trend navigation.
# Trend dipakai HANYA untuk navigasi/prioritisasi kandidat, BUKAN identitas final.
class TrendNodeType(str, Enum):
    TOPIC_KEY = "TOPIC_KEY"   # "Lanjutkan topik sebelumnya" → top-N TopicKey
    FRAGMENT = "FRAGMENT"     # "Lanjutkan bisnis kopi" → top-N fragment depth tertentu


# [MEKANISME 5] Hasil penempatan fragment baru dalam hierarchy.
class FragmentPlacementKind(str, Enum):
    ROOT = "ROOT"                       # Fragment pertama TopicKey → depth 0
    CHILD = "CHILD"                     # Parent kuat ditemukan → depth parent + 1
    SIBLING = "SIBLING"                 # Tidak ada parent cukup kuat → branch baru
    DEPTH_CAPPED_ATTACH = "DEPTH_CAPPED_ATTACH"  # MAX_DEPTH tercapai → attach, bukan perdalam


# ==========================================
# LAYER 1: INPUT GATE
# ==========================================

class InputEnvelope(BaseModel):
    """Representasi standar input mentah dari user."""
    input_id: str
    session_id: str
    raw_text: str
    normalized_text: str
    timestamp: datetime
    timezone: str = "Asia/Jakarta"


# ==========================================
# [MEKANISME 1] NEGATION SCOPE
# ==========================================

class NegationScope(BaseModel):
    """
    [MEKANISME 1] Negative evidence dengan SCOPE, bukan sekadar keberadaan kata.

    "bukan cpp"  → marker="bukan", target=["cpp"]   → cpp DIELIMINASI
    "cpp bukan?" → TIDAK menghasilkan scope ini      → cpp TIDAK dieliminasi

    Karena itu negation harus bagian dari signal extraction dengan posisi,
    bukan `if "bukan" in tokens`.
    """
    marker: str                                     # "bukan", "jangan", "kecuali", "bukan topik"
    target_tokens: list[str] = Field(default_factory=list)
    marker_position: int = 0                        # index karakter marker di raw text
    scope_end_position: int = 0                     # batas akhir scope negation
    raw_fragment: str = ""                          # potongan teks yang memicu scope


# ==========================================
# LAYER 3: SIGNAL ANALYSIS
# ==========================================

class SignalProfile(BaseModel):
    """Hasil ekstraksi sinyal dari bahasa manusia."""
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    raw_tokens: list[str] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    normalized_tokens: list[str] = Field(default_factory=list)
    unique_terms: list[str] = Field(default_factory=list)
    unique_tokens: list[str] = Field(default_factory=list)
    topic_signals: list[str] = Field(default_factory=list)
    co_occurrence_signals: list[str] = Field(default_factory=list)
    temporal_signals: list[str] = Field(default_factory=list)
    novel_tokens: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    # [ASSOCIATIVE EVIDENCE LOCKING] Kualitas bukti token.
    # token_quality memetakan setiap token unik ke tier-nya.
    token_quality: dict[str, TokenQuality] = Field(default_factory=dict)
    # Token bernilai identitas tinggi (numerik / identifier).
    numeric_tokens: list[str] = Field(default_factory=list)
    # Token yang jarang muncul di seluruh Topic Key.
    rare_tokens: list[str] = Field(default_factory=list)
    # Token fungsional yang tidak boleh mendominasi scoring.
    low_information_tokens: list[str] = Field(default_factory=list)
    # Co-occurrence berbasis window (bukan hanya bersebelahan).
    # Format sama dengan co_occurrence_signals: "token_a↔token_b"
    window_co_occurrence_signals: list[str] = Field(default_factory=list)
    # Kekuatan per pasangan window: "token_a↔token_b" -> bobot ter-decay.
    window_pair_strength: dict[str, float] = Field(default_factory=dict)
    # [MEKANISME 1] Negative / exclusion evidence.
    # Token yang secara eksplisit DITOLAK user ("bukan cpp" → ["cpp"]).
    excluded_tokens: list[str] = Field(default_factory=list)
    # Scope negation lengkap (marker + target + posisi) untuk audit.
    negation_scopes: list[NegationScope] = Field(default_factory=list)
    # [MEKANISME 2] Continuation intent.
    # True jika input meminta MELANJUTKAN memori, bukan membuka topik baru.
    continuation_intent: bool = False
    continuation_markers: list[str] = Field(default_factory=list)
    # [MEKANISME 1] Explicit cancellation / topic switch.
    # "ganti topik", "lupakan", "batal", "bahas yang lain", "C++ saja"
    cancellation_intent: bool = False
    cancellation_markers: list[str] = Field(default_factory=list)

# ==========================================
# LAYER 4: TOPIC KEY ACTIVATION
# ==========================================

class TopicKey(BaseModel):
    """Pintu asosiasi menuju kelompok memori."""
    key_id: str
    title: str
    helper_description: str
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    representative_tokens: list[str] = Field(default_factory=list)
    unique_tokens: list[str] = Field(default_factory=list)
    token_profile: dict[str, int] = Field(default_factory=dict)
    co_occurrence_profile: dict[str, int] = Field(default_factory=dict)
    created_at: datetime
    updated_at: Optional[datetime] = None
    activation_count: int = 0
    related_fragment_ids: list[str] = Field(default_factory=list)
    status: str = "ACTIVE"
    # [MEKANISME 3] Helper Propagation ke TopicKey.
    #
    # helper_description di atas = identitas OWN TopicKey (dibuat saat topik lahir).
    # aggregated_helper_description = ringkasan TERAGREGASI dari root fragments
    # beserta cabang-cabang yang hidup di bawahnya.
    #
    # Dipisah agar:
    #   1. propagasi bisa diaudit (mana own, mana warisan descendant)
    #   2. pertumbuhan bisa dibatasi tanpa merusak identitas asli
    #   3. mutation gate tetap bisa menolak propagasi (low-info input)
    aggregated_helper_description: str = ""
    # Fragment mana saja yang sudah menyumbang ke aggregated helper.
    # Dipakai untuk dedup agar child yang sama tidak menyumbang berulang.
    helper_source_fragment_ids: list[str] = Field(default_factory=list)

# [ASSOCIATIVE EVIDENCE LOCKING] Profil bukti lengkap satu kandidat.
# Tujuannya bukan menyimpan skor, tetapi menjawab:
# "Berapa JENIS bukti yang benar-benar mendukung kandidat ini?"
class EvidenceProfile(BaseModel):
    """Dimensi kedua dari penilaian kandidat: EVIDENCE STRENGTH."""
    token_score: float = 0.0
    description_score: float = 0.0
    co_occurrence_score: float = 0.0
    unique_token_score: float = 0.0
    temporal_score: float = 0.0
    continuity_score: float = 0.0
    graph_score: float = 0.0

    # Berapa jenis bukti yang >= EVIDENCE_SUPPORT_THRESHOLD.
    supporting_evidence_count: int = 0
    # Berapa jenis bukti yang saling bertentangan.
    contradiction_count: int = 0
    # Nama bukti yang mendukung (untuk audit / debugging).
    supporting_evidence_names: list[str] = Field(default_factory=list)
    # Alasan kontradiksi (untuk audit / debugging).
    contradiction_notes: list[str] = Field(default_factory=list)
    # Kekuatan bukti gabungan (0..1), independen dari final_score.
    evidence_strength: float = 0.0
    # True jika satu evidence menyumbang >= EVIDENCE_DOMINANCE_RATIO.
    is_single_signal_dominated: bool = False
    # [REASONING PURITY] Evidence Family (Prioritas #2).
    # Keluarga bukti yang aktif: LEXICAL / CONTEXTUAL / RELATIONAL.
    # 3 sinyal lexical = 1 keluarga, bukan 3 bukti independen.
    active_evidence_families: list[EvidenceFamily] = Field(default_factory=list)
    evidence_family_count: int = 0
    is_single_family_dominated: bool = False
    # Pemetaan sinyal -> keluarga (untuk audit).
    signal_to_family: dict[str, EvidenceFamily] = Field(default_factory=dict)

class CandidateScore(BaseModel):
    """Hasil scoring aktivasi Topic Key."""
    key_id: str
    title: str
    token_score: float = 0.0
    description_score: float = 0.0
    co_occurrence_score: float = 0.0
    continuity_score: float = 0.0
    temporal_score: float = 0.0
    # [NEW] Skor tambahan untuk Unique Token dan Cross-Session Evidence
    unique_token_score: float = 0.0
    cross_session_score: float = 0.0
    final_score: float = 0.0
    # [ASSOCIATIVE EVIDENCE LOCKING] Kandidat WAJIB membawa alasan pemilihannya.
    # graph_score adalah nama kanonik dokumentasi; nilainya selaras dengan
    # cross_session_score (tidak dihitung dua kali).
    graph_score: float = 0.0
    # Dimensi 1 = final_score (relevance). Dimensi 2 = evidence_strength.
    evidence_strength: float = 0.0
    support_count: int = 0
    contradiction_score: float = 0.0
    # Confidence setelah safety cap diterapkan. BUKAN sekadar final_score.
    confidence: float = 0.0
    # Profil bukti lengkap untuk audit.
    evidence_profile: Optional[EvidenceProfile] = None
    # Catatan keputusan (misal: "single-signal capped", "description contradiction").
    evidence_notes: list[str] = Field(default_factory=list)
    # [REASONING PURITY] Evidence Family (Prioritas #2).
    # Jumlah keluarga bukti berbeda yang mendukung kandidat ini.
    # family_support_count=1 + banyak sinyal = bukti TIDAK beragam.
    family_support_count: int = 0
    evidence_families: list[EvidenceFamily] = Field(default_factory=list)
    is_single_family_dominated: bool = False
    # [REASONING PURITY] Continuity Purity (Prioritas #1).
    # turn_continuity_score = "seberapa kuat kandidat ini melanjutkan state aktif"
    # BERBEDA dari continuity_score di evidence profile yang merupakan salah satu
    # komponen final_score. Field ini dipakai untuk relationship_strength fragment.
    turn_continuity_score: float = 0.0
    # [MEKANISME 1] Exclusion evidence.
    # Skor penalti akibat user secara eksplisit menolak kandidat ini
    # ("yang penting bukan topik cpp").
    exclusion_score: float = 0.0
    # Token kandidat yang terkena negation scope user.
    excluded_matched_tokens: list[str] = Field(default_factory=list)
    # True jika kandidat ini sudah dieliminasi dari shortlist klarifikasi.
    is_excluded: bool = False
    # [MEKANISME 3] Skor dari aggregated helper (warisan descendant).
    # Terpisah dari description_score agar terlihat mana bukti own vs warisan.
    aggregated_helper_score: float = 0.0

# ==========================================
# LAYER 6 & 7: FRAGMENTS & PAYLOADS
# ==========================================

class FragmentMetadata(BaseModel):
    """
    Metadata ringan fragment. Dibaca tanpa membuka payload besar.

    [MEKANISME 3 + 5] Fragment adalah NODE HIERARKIS beridentitas.

    IMMUTABILITY CONTRACT:
        model_config frozen=True DIPERTAHANKAN. Ini penjaga integritas
        exact memory. Helper propagation TIDAK memutasi instance ini.
        Alur propagasi:
            parent.model_copy(update={"aggregated_helper_description": ...})
                → instance BARU
                → storage.update_fragment_helper(session, instance_baru)
        Dengan begitu payload reference & content_hash tidak pernah berubah.

    CHILD COUNT:
        TIDAK disimpan sebagai field (spesifikasi butir 1.3).
        child_count adalah DERIVED value:
            COUNT(FragmentDB.parent_fragment_id == fragment_id)
        Storage menyediakan query terindex. Jika nanti di-cache, itu harus
        diperlakukan sebagai cache, bukan sumber kebenaran.
    """
    model_config = ConfigDict(frozen=True) # Immutable setelah dibuat
    
    fragment_id: str
    topic_key_id: str
    session_id: str
    created_at: datetime
    temporal_index: Optional[datetime] = None
    payload_reference: str # Merujuk ke payload_id
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    content_hash: Optional[str] = None
    sequence_index: int = 0
    source_fragment_ids: list[str] = Field(default_factory=list)
    relationship_strength: float = 0.0
    # [NEW] Skor relevansi fragment terhadap query saat ini (untuk Fragment-Level Ranking)
    relevance_score: float = 0.0
    # [CRITICAL FIX] Profil token & co-occurrence independen milik Fragment untuk True Query-to-Fragment Relevance
    token_profile: dict[str, int] = Field(default_factory=dict)
    co_occurrence_profile: dict[str, int] = Field(default_factory=dict)
    # ==========================================
    # [MEKANISME 3] IDENTITAS FRAGMENT SEBAGAI SUBTOPIK
    # ==========================================
    # Nama bagian ini sendiri. BUKAN TopicKey baru.
    # Contoh: "Modal Bisnis Kopi", "Lokasi Usaha", "Modal Murah vs Mahal"
    fragment_key: str = ""
    # Helper description OWN fragment ini — "fragment ini membahas apa?"
    # Diekstrak dari exact text SEBELUM payload dikompres.
    fragment_helper_description: str = ""
    # ==========================================
    # [MEKANISME 5] HIERARCHY STRUCTURE
    # ==========================================
    # Parent eksplisit. None = root/opening fragment.
    parent_fragment_id: Optional[str] = None
    # Kedalaman node. Definisi DIKUNCI:
    #   depth 0 = root / opening
    #   depth 1 = section utama
    #   depth 2 = child spesifik
    #   depth 3 = spesialisasi lebih dalam (hanya jika MAX_FRAGMENT_DEPTH >= 3)
    depth: int = 0
    # ==========================================
    # [MEKANISME 3] HELPER PROPAGATION (child → parent → root → TopicKey)
    # ==========================================
    # Ringkasan helper dari anak-cucu. BOUNDED + DEDUPLICATED.
    # BUKAN append mentah seluruh isi child.
    descendant_helper_evidence: list[str] = Field(default_factory=list)
    # Representasi agregat: own helper + ringkasan descendant terpilih.
    # Inilah yang dipakai retrieval untuk membedakan "Modal Bisnis Kopi"
    # dari "Lokasi Bisnis Kopi" meski keduanya di TopicKey yang sama.
    aggregated_helper_description: str = ""
    # Fragment mana yang sudah menyumbang ke descendant evidence (dedup).
    helper_contributor_ids: list[str] = Field(default_factory=list)
    # ==========================================
    # [MEKANISME 4] CROSS-KEY COMPOSITE
    # ==========================================
    # True jika fragment ini lahir dari >= 2 sumber bermakna yang berbeda.
    # Primary ownership TETAP topic_key_id (single foreign key).
    # Multi-source direpresentasikan lewat source_fragment_ids + graph edges,
    # BUKAN dengan dua primary TopicKey.
    is_composite: bool = False
    # TopicKey asal dari source fragments (derived, untuk audit/navigation).
    source_topic_key_ids: list[str] = Field(default_factory=list)

class CompressedPayload(BaseModel):
    """Memori besar yang dikompresi lossless."""
    model_config = ConfigDict(frozen=True) # Immutable, mencegah korupsi historis
    
    payload_id: str
    compressed_blob: bytes
    checksum: str
    compression_algorithm: str = "zstd"
    original_size: int
    compressed_size: int

# [ASSOCIATIVE EVIDENCE LOCKING] Memori hasil rekonstruksi membawa metadata bukti.
# Context Builder harus mengurutkan berdasarkan kualitas bukti,
# BUKAN sekadar "memori terbaru".
class RetrievedMemory(BaseModel):
    """Exact memory yang sudah direkonstruksi + metadata kualitas bukti."""
    fragment_id: str
    text: str
    timestamp: str
    topic_key_id: str
    # Kualitas bukti fragment-level.
    relevance_score: float = 0.0
    temporal_score: float = 0.0
    graph_score: float = 0.0
    evidence_strength: float = 0.0
    # Penanda apakah fragment cocok persis dengan constraint waktu user.
    exact_temporal_match: bool = False
    # [REASONING PURITY] Fragment Graph Evidence (Prioritas #4).
    # Skor graph dihitung dari edge nyata (CONTINUES / CROSS_SESSION_LINK),
    # BUKAN dari relationship_strength metadata. 0 jika tidak ada edge.
    fragment_graph_score: float = 0.0
    graph_edge_count: int = 0
    graph_path_types: list[str] = Field(default_factory=list)
    # [REASONING PURITY] Temporal Distance Quality (Prioritas #5).
    # Skor berdasarkan kedekatan fragment dengan waktu yang diminta,
    # bukan hanya "apakah masuk range".
    temporal_distance_score: float = 0.0
    temporal_distance_minutes: Optional[float] = None
    # ==========================================
    # [MEKANISME 3] HIERARCHY CONTEXT FRAMING
    # ==========================================
    # Dipakai ContextBuilder untuk membingkai memori sebagai struktur:
    #   TOPIC: Bisnis Kopi
    #   SECTION: Modal Bisnis Kopi (depth 1)
    #   SUBSECTION: Modal Murah vs Mahal (depth 2)
    #   EXACT MEMORY: <text tidak diubah>
    #
    # PENTING: framing ini HANYA metadata. Field `text` di atas tetap
    # exact memory apa adanya — tidak boleh dimodifikasi.
    topic_key_title: str = ""
    fragment_key: str = ""
    fragment_helper_description: str = ""
    parent_fragment_id: Optional[str] = None
    depth: int = 0
    # [MEKANISME 4] Sumber composite (untuk framing "berasal dari A + B").
    source_fragment_ids: list[str] = Field(default_factory=list)
    is_composite: bool = False

# ==========================================
# LAYER 2 & 15: FAST STATE & AMBIGUITY
# ==========================================

class PendingState(BaseModel):
    """
    State sementara untuk menangani ambiguitas dan klarifikasi.

    [MEKANISME 1] Pending bukan lagi "satu pertanyaan → satu jawaban".
    Ia adalah STATE YANG BERKEMBANG:

        original_input
            + clarification_history[0]
            + clarification_history[1]
            + ...
            + negative evidence
            ↓
        combined evidence → re-rank → candidate set MENYEMPIT

    SUMBER KEBENARAN = original_input + clarification_history.
    combined_clarification_text hanyalah CACHE/derived value, bukan source of truth.
    """
    pending_id: str
    session_id: str
    original_input: str
    candidate_keys: list[str]
    candidate_scores: dict[str, float]
    assistant_question: str
    clarification_round: int = 1
    created_at: datetime
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    last_updated_at: Optional[datetime] = None
    status: PendingStatus = PendingStatus.ACTIVE
    temporal_reference: Optional[str] = None
    origin_session_id: Optional[str] = None
    # ==========================================
    # [MEKANISME 1] RECURSIVE AMBIGUITY ACCUMULATION
    # ==========================================
    # Jawaban klarifikasi berurutan (round 1, round 2, ...).
    # Inilah sumber kebenaran akumulasi, bersama original_input.
    clarification_history: list[str] = Field(default_factory=list)
    # DERIVED CACHE: original_input + seluruh history, siap di-extract.
    # Boleh diregenerasi kapan saja dari original_input + history.
    combined_clarification_text: str = ""
    # Kandidat yang dieliminasi oleh negative evidence ("bukan topik cpp").
    excluded_candidate_keys: list[str] = Field(default_factory=list)
    # Token yang ditolak user, dipakai untuk exclusion scoring.
    excluded_tokens: list[str] = Field(default_factory=list)
    # Riwayat penyempitan kandidat per round.
    # Contoh: [[A,B,C], [A,B], [A]] — membuktikan C tidak muncul lagi di round 2.
    candidate_history: list[list[str]] = Field(default_factory=list)


# ==========================================
# [MEKANISME 3] SESSION ACTIVE STATE ("STATE NOW")
# ==========================================

class SessionActiveState(BaseModel):
    """
    [MEKANISME 3] Current Conversational State.

    Ketika percakapan sedang membahas "Bisnis Kopi → Modal Bisnis Kopi",
    sistem tidak perlu selalu membandingkan ulang seluruh kandidat untuk
    setiap kelanjutan lokal.

    TETAPI: active state BUKAN probabilitas permanen.
    Jika evidence menunjukkan "bisnis travel", sistem tetap harus berpindah.
    Active state adalah EVIDENCE KUAT, bukan pengunci.
    """
    session_id: str
    active_topic_key_id: Optional[str] = None
    active_fragment_id: Optional[str] = None
    active_depth: int = 0
    last_updated: Optional[datetime] = None


# ==========================================
# [MEKANISME 5] FRAGMENT PLACEMENT
# ==========================================

class FragmentPlacementResult(BaseModel):
    """
    [MEKANISME 5] Hasil Fragment Placement Analysis.

    Alur:
        New Fragment
            → compare with root/opening context
            → compare with depth-1 fragments
            → find strongest parent candidate
            ├── strong enough  → CHILD (depth parent + 1)
            └── not strong     → SIBLING (branch baru)

    MAX_FRAGMENT_DEPTH adalah HARD STRUCTURAL BOUNDARY.
    Jika parent terbaik sudah di depth maksimum, sistem TIDAK boleh
    menciptakan depth+1. Ia harus DEPTH_CAPPED_ATTACH pada node terdalam
    yang diizinkan.
    """
    kind: FragmentPlacementKind = FragmentPlacementKind.ROOT
    parent_fragment_id: Optional[str] = None
    depth: int = 0
    # Skor parent terbaik dan runner-up (untuk membuktikan margin).
    match_score: float = 0.0
    runner_up_score: float = 0.0
    margin: float = 0.0
    # True jika placement dibatasi oleh MAX_FRAGMENT_DEPTH.
    depth_capped: bool = False
    reason: str = ""
    # Kandidat parent yang dipertimbangkan (untuk audit "bukan sekadar fragment terbaru").
    considered_parent_ids: list[str] = Field(default_factory=list)


class FragmentCandidateScore(BaseModel):
    """
    [MEKANISME 3 + 5] Skor kandidat fragment.

    Dipakai untuk DUA keperluan:
        1. Fragment-level retrieval (query → fragment mana yang relevan)
        2. Hierarchy placement (fragment baru → parent mana yang tepat)

    Memisahkan fragment_key / own helper / aggregated helper / token / cooc
    agar terlihat sumber bukti mana yang memenangkan penempatan.
    """
    fragment_id: str
    fragment_key: str = ""
    topic_key_id: str = ""
    depth: int = 0
    parent_fragment_id: Optional[str] = None
    # Skor per sumber bukti.
    fragment_key_score: float = 0.0
    helper_score: float = 0.0
    aggregated_helper_score: float = 0.0
    token_score: float = 0.0
    co_occurrence_score: float = 0.0
    temporal_score: float = 0.0
    graph_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0
    # Bukti pendukung (untuk audit).
    supporting_evidence_names: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ==========================================
# [MEKANISME 4] CROSS-KEY COMPOSITE SOURCE
# ==========================================

class FragmentSourceContribution(BaseModel):
    """
    [MEKANISME 4] Kontribusi satu sumber terhadap respons baru.

    Tidak cukup hanya token overlap. Harus ada:
        evidence threshold + distinct source evidence + contribution strength
    agar semua fragment bertema "kopi" tidak otomatis saling menjadi source.
    """
    fragment_id: str
    topic_key_id: str = ""
    fragment_key: str = ""
    contribution_score: float = 0.0
    supporting_evidence_names: list[str] = Field(default_factory=list)
    # True jika sumber ini berasal dari TopicKey berbeda dari primary.
    is_cross_topic: bool = False


class CompositeSourceAnalysis(BaseModel):
    """
    [MEKANISME 4] Hasil deteksi multi-source response.

    Jika >= 2 sumber benar-benar berkontribusi:
        source_fragment_ids = [A, B, ...]
        + graph edges A → NewFragment, B → NewFragment

    Primary TopicKey TETAP SATU (ownership tidak dipecah).
    """
    is_composite: bool = False
    sources: list[FragmentSourceContribution] = Field(default_factory=list)
    source_fragment_ids: list[str] = Field(default_factory=list)
    distinct_topic_count: int = 0
    primary_topic_key_id: Optional[str] = None
    reason: str = ""


# ==========================================
# [MEKANISME 3] HELPER PROPAGATION RESULT
# ==========================================

class HelperPropagationResult(BaseModel):
    """
    [MEKANISME 3] Audit propagasi helper child → parent → root → TopicKey.

    Yang naik BUKAN seluruh isi child, melainkan representasi ringkas,
    terkontrol, deduplicated, dengan batas panjang dan jumlah kontribusi.
    """
    propagated: bool = False
    updated_fragment_ids: list[str] = Field(default_factory=list)
    updated_topic_key_id: Optional[str] = None
    contributions_added: int = 0
    contributions_deduplicated: int = 0
    # True jika ada kontribusi dipotong karena batas panjang/jumlah.
    truncated: bool = False
    # Kedalaman rantai propagasi yang benar-benar dilalui.
    chain_depth: int = 0
    reason: str = ""


# ==========================================
# [MEKANISME 5] TREND NAVIGATION
# ==========================================

class TrendCandidate(BaseModel):
    """
    [MEKANISME 5] Kandidat navigasi berbasis aktivitas cabang.

    Dipakai HANYA ketika user meminta continuation tanpa menunjuk bagian spesifik:
        "Lanjutkan topik sebelumnya"  → TOPIC_KEY candidates
        "Lanjutkan bisnis kopi"       → FRAGMENT candidates (depth tertentu)

    SINYAL UTAMA = jumlah anak / fragment (branch richness).
    Recency & temporal hanya bukti TAMBAHAN.

    PENTING: banyak child ≠ pasti maksud user.
    Trend = navigation / candidate prioritization evidence,
            BUKAN final topic identity.
    """
    node_id: str
    node_type: TrendNodeType = TrendNodeType.TOPIC_KEY
    label: str = ""
    helper_description: str = ""
    depth: int = 0
    parent_fragment_id: Optional[str] = None
    # Sinyal utama.
    child_count: int = 0
    fragment_count: int = 0
    # Sinyal tambahan.
    recency_score: float = 0.0
    temporal_score: float = 0.0
    evidence_score: float = 0.0
    # Skor komposit untuk ranking.
    trend_score: float = 0.0
    # Apakah node ini masih bisa di-expand (depth < MAX_FRAGMENT_DEPTH).
    expandable: bool = True


# ==========================================
# [MEKANISME 1] CLARIFICATION ACCUMULATION VALUE OBJECT
# ==========================================

class ClarificationAccumulation(BaseModel):
    """
    [MEKANISME 1] Paket evidence klarifikasi yang dikirim ke RetrievalEngine.

    Retrieval tidak boleh hanya menganalisis jawaban terakhir.
    "yang rencana kopi" sendiri terlalu miskin bukti; ia harus dibaca bersama
    "lanjutkan topik kopi kemarin".
    """
    original_input: str = ""
    clarification_history: list[str] = Field(default_factory=list)
    combined_text: str = ""
    excluded_tokens: list[str] = Field(default_factory=list)
    excluded_candidate_keys: list[str] = Field(default_factory=list)
    negation_scopes: list[NegationScope] = Field(default_factory=list)
    round_number: int = 0
    # Kandidat yang masih kompetitif dari round sebelumnya (untuk narrowing).
    retained_candidate_keys: list[str] = Field(default_factory=list)


# ==========================================
# LAYER 13: GRAPH RELATIONSHIPS
# ==========================================

class MemoryGraphEdge(BaseModel):
    """Hubungan antar node dalam graf memori."""
    edge_id: str
    # [NEW] Renamed dari source_id/target_id agar eksplisit node-based sesuai Spesifikasi
    source_node_id: str # Bisa TopicKey ID atau Fragment ID
    target_node_id: str
    relationship_type: RelationshipType
    weight: float = 1.0
    # [NEW] Field diperkuat sesuai Spesifikasi Penguatan
    strength: float = 1.0
    evidence: str = ""
    created_at: datetime

# ==========================================
# LAYER 5, 9, 11, 16: ENGINE DECISIONS & I/O
# ==========================================

class EngineDecision(BaseModel):
    """Keputusan yang diambil oleh Decision Gate."""
    decision_type: DecisionType
    selected_key_id: Optional[str] = None
    selected_fragment_ids: list[str] = Field(default_factory=list)
    pending_state: Optional[PendingState] = None
    message: str = ""
    # [ASSOCIATIVE EVIDENCE LOCKING] Keputusan harus bisa diaudit.
    confidence: float = 0.0
    support_count: int = 0
    evidence_strength: float = 0.0
    contradiction_score: float = 0.0
    decision_reason: str = ""
    # [REASONING PURITY] Evidence Family (Prioritas #2).
    family_support_count: int = 0
    evidence_families: list[EvidenceFamily] = Field(default_factory=list)
    # [MEKANISME 1] Kandidat yang dieliminasi oleh negative evidence.
    excluded_candidate_keys: list[str] = Field(default_factory=list)
    # [MEKANISME 2] Intent melanjutkan terdeteksi pada input ini.
    continuation_intent: bool = False
    # [MEKANISME 5] Kandidat navigasi saat continuation tanpa target spesifik.
    trend_candidates: list[TrendCandidate] = Field(default_factory=list)

class ModelRequest(BaseModel):
    """Konteks yang dirakit untuk dikirim ke Model API."""
    system_context: str = ""
    past_memory: str = ""
    current_user_message: str

class ModelResponse(BaseModel):
    """Respons murni dari Model API."""
    response_text: str
    tokens_used: int = 0

class TurnAnalysisResult(BaseModel):
    """Hasil analisis apakah turn baru layak menjadi memori."""
    classification: MemoryClassification
    continuity_score: float = 0.0
    new_topic_score: float = 0.0
    active_topic_key_id: Optional[str] = None
    # [ASSOCIATIVE EVIDENCE LOCKING] EVIDENCE BEFORE MUTATION.
    # Field ini yang menentukan boleh/tidaknya token_profile, co_occurrence,
    # dan helper_description milik Topic Key diperbarui.
    evidence_strength: float = 0.0
    support_count: int = 0
    contradiction_score: float = 0.0
    mutation_allowed: bool = True
    description_update_allowed: bool = True
    mutation_reason: str = ""
    # [REASONING PURITY] Continuity Purity (Prioritas #1).
    # turn_continuity_score menjawab: "seberapa kuat turn ini melanjutkan
    # state aktif?" — BERBEDA dari continuity_score kandidat.
    # Inilah nilai yang dipersist ke relationship_strength fragment.
    turn_continuity_score: float = 0.0
    # Sumber bukti continuity (untuk audit: "overlap", "history", "fragments").
    continuity_evidence_sources: list[str] = Field(default_factory=list)
    # [MEKANISME 2] Continuation Intent Protection.
    continuation_intent: bool = False
    # True jika continuation intent ada TAPI target tidak teridentifikasi.
    # Konsekuensi: JANGAN persist fragment TopicKey baru.
    unresolved_continuation: bool = False
    # [MEKANISME 3] Active fragment state yang dipakai sebagai evidence.
    active_fragment_id: Optional[str] = None
    active_depth: int = 0

class EngineResult(BaseModel):
    """Output final dari Global Orchestrator (Layer 16) ke User/App."""
    assistant_response: str
    decision: EngineDecision
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    active_topic_key_id: Optional[str] = None
    persistence_status: str = "SUCCESS" # SUCCESS, FAILED, SKIPPED
    # [NEW] Status observabilitas rekonstruksi memori persis (Review Kedua)
    reconstruction_status: ReconstructionStatus = ReconstructionStatus.NO_MEMORY_REQUESTED
    # [ASSOCIATIVE EVIDENCE LOCKING] Memori terstruktur dengan metadata bukti.
    # retrieved_memory_ids tetap dipertahankan untuk kompatibilitas.
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list)
    # Ringkasan bukti untuk transparansi / debugging.
    evidence_summary: str = ""
    # [ASSOCIATIVE EVIDENCE LOCKING] SELURUH kandidat, termasuk yang DITOLAK.
    # Tanpa ini, pemanggil hanya bisa melihat pemenang dan tidak bisa menjawab
    # "KENAPA memori B tidak dipilih" (Dokumentasi butir 4: kandidat harus
    # membawa alasan pemilihannya). Setiap entri membawa support_count,
    # contradiction_score, confidence, dan evidence_notes.
    candidates: list[CandidateScore] = Field(default_factory=list)
    # [REASONING PURITY] Ringkasan kemurnian reasoning (untuk audit/debug).
    reasoning_purity_summary: str = ""
    # [FIX — Batch D] Continuity Purity observability.
    #
    # Nilai turn_continuity_score AKTUAL yang dihitung saat turn ini diproses,
    # dan yang dipersist ke fragment.relationship_strength.
    #
    # Kenapa perlu di-expose:
    #   Menghitung ulang nilai ini SETELAH persist tidak menghasilkan angka yang
    #   sama, karena 3 inputnya sudah berubah state:
    #     1. token_profile TopicKey sudah diperbarui dengan token turn ini
    #     2. history_ratio berubah karena fragment baru sudah masuk window
    #     3. active_frag_rel mengambil fragment terbaru — yaitu fragment yang
    #        sedang diuji sendiri (self-reference)
    #
    #   Dengan field ini, invariant "relationship_strength == turn_continuity"
    #   dapat diverifikasi langsung tanpa rekonstruksi ulang.
    turn_continuity_score: float = 0.0
    # Sumber bukti continuity untuk audit: "overlap", "history", "fragments",
    # "-novelty(0.20)", "-contradiction(0.15)".
    continuity_evidence_sources: list[str] = Field(default_factory=list)
    # ==========================================
    # [MEKANISME 3] ACTIVE FRAGMENT STATE
    # ==========================================
    active_fragment_id: Optional[str] = None
    active_depth: int = 0
    # ==========================================
    # [MEKANISME 5] HIERARCHY PLACEMENT
    # ==========================================
    # Fragment yang baru dibuat pada turn ini (jika ada).
    created_fragment_id: Optional[str] = None
    fragment_placement: Optional[FragmentPlacementResult] = None
    # ==========================================
    # [MEKANISME 3] HELPER PROPAGATION
    # ==========================================
    helper_propagation: Optional[HelperPropagationResult] = None
    # ==========================================
    # [MEKANISME 4] COMPOSITE MULTI-SOURCE
    # ==========================================
    composite_analysis: Optional[CompositeSourceAnalysis] = None
    # ==========================================
    # [MEKANISME 5] TREND NAVIGATION
    # ==========================================
    trend_candidates: list[TrendCandidate] = Field(default_factory=list)
    # ==========================================
    # [MEKANISME 1] CLARIFICATION ACCUMULATION
    # ==========================================
    clarification_round: int = 0
    clarification_history: list[str] = Field(default_factory=list)
    excluded_candidate_keys: list[str] = Field(default_factory=list)
    # ==========================================
    # [MEKANISME 2] CONTINUATION PROTECTION
    # ==========================================
    continuation_intent: bool = False
