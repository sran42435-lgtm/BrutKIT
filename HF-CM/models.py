"""
models.py
Kontrak Data & State Machine untuk Hybrid Fragmented Compressed Memory (HF-CM).

File ini mendefinisikan bentuk data yang dipertukarkan antar-layer.
ATURAN MUTLAK: Semua pertukaran data harus menggunakan Pydantic models di bawah ini.
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

# ==========================================
# LAYER 6 & 7: FRAGMENTS & PAYLOADS
# ==========================================

class FragmentMetadata(BaseModel):
    """Metadata ringan fragment. Dibaca tanpa membuka payload besar."""
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

class CompressedPayload(BaseModel):
    """Memori besar yang dikompresi lossless."""
    model_config = ConfigDict(frozen=True) # Immutable, mencegah korupsi historis
    
    payload_id: str
    compressed_blob: bytes
    checksum: str
    compression_algorithm: str = "zstd"
    original_size: int
    compressed_size: int

# ==========================================
# LAYER 2 & 15: FAST STATE & AMBIGUITY
# ==========================================

class PendingState(BaseModel):
    """State sementara untuk menangani ambiguitas dan klarifikasi."""
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

class EngineResult(BaseModel):
    """Output final dari Global Orchestrator (Layer 16) ke User/App."""
    assistant_response: str
    decision: EngineDecision
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    active_topic_key_id: Optional[str] = None
    persistence_status: str = "SUCCESS" # SUCCESS, FAILED, SKIPPED
    # [NEW] Status observabilitas rekonstruksi memori persis (Review Kedua)
    reconstruction_status: ReconstructionStatus = ReconstructionStatus.NO_MEMORY_REQUESTED
