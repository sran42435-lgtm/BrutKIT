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

@dataclass(frozen=True)
class SignalConstants:
    MIN_TOKEN_LENGTH: Final[int] = 3
    MAX_ANALYSIS_TOKENS: Final[int] = 500
    MIN_SIGNAL_STRENGTH: Final[float] = 0.10

@dataclass(frozen=True)
class ActivationWeights:
    """Bobot untuk scoring Topic Key (Layer 4)."""
    TOKEN_WEIGHT: Final[float] = 0.25
    DESCRIPTION_WEIGHT: Final[float] = 0.20
    CO_OCCURRENCE_WEIGHT: Final[float] = 0.15
    CONTINUITY_WEIGHT: Final[float] = 0.40
    TEMPORAL_WEIGHT: Final[float] = 0.00

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

# [NEW] Class baru sesuai Spesifikasi Penguatan
@dataclass(frozen=True)
class TopicSwitchConstants:
    NOVELTY_THRESHOLD: Final[float] = 0.70
    MIN_ACTIVE_TOPIC_OVERLAP: Final[float] = 0.15
    TOPIC_SWITCH_THRESHOLD: Final[float] = 0.60

@dataclass(frozen=True)
class GraphConstants:
    EDGE_CREATION_THRESHOLD: Final[float] = 0.80
    MAX_EDGES_PER_NODE: Final[int] = 20
    MIN_RELATIONSHIP_CONFIDENCE: Final[float] = 0.75

@dataclass(frozen=True)
class StorageConstants:
    DB_TIMEOUT_SECONDS: Final[int] = 5
    MAX_TRANSACTION_RETRIES: Final[int] = 3
    SQLITE_DB_PATH: Final[str] = "data/memory.db"


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
    graph: Final[GraphConstants] = GraphConstants()
    storage: Final[StorageConstants] = StorageConstants()

CONFIG: Final[HF_CMConfig] = HF_CMConfig()
