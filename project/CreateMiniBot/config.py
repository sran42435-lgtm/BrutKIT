# config.py
#
# Konfigurasi global untuk RAG-based Training Pipeline Engine.
#
# Arsitektur: Single-Head + RAG + Constrained Generation
# Dataset: SQuAD 2.0 (142K samples) + Custom Indonesian
# Model size target: ~40M parameters (balanced untuk 142K samples)
#
# Fitur Utama:
# - Retrieval-based context injection (BM25)
# - Single LM head (no domain contamination)
# - Context-conditioned training
# - Constrained generation (tokens only from context)
# - SOTA: RoPE, Weight Tying, Soft-Capping, DropPath
#
# Implementasi ini sengaja tidak bergantung pada pustaka ML eksternal.
# Hanya menggunakan standard library Python + NumPy.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, Optional, Tuple


# ==============================================================================
# ROOT PATH
# ==============================================================================

ROOT_DIR: Final[Path] = Path(__file__).resolve().parent


# ==============================================================================
# DAFTAR DTYPE YANG DIIZINKAN
# ==============================================================================

ALLOWED_TRAIN_DTYPES: Final[frozenset] = frozenset(
    {
        "float32",
        "float16",
        "bfloat16",
    }
)

ALLOWED_EXPORT_DTYPES: Final[frozenset] = frozenset(
    {
        "float32",
        "float16",
        "bfloat16",
        "int8",
        "int4",
    }
)


# ==============================================================================
# PATH CONFIG
# ==============================================================================

@dataclass(frozen=True)
class PathsConfig:
    """
    Konfigurasi lokasi folder dan file penting.
    
    Sesuai struktur proyek:
    CreateMiniBot/
    ├── datasets/
    │   ├── *.txt (raw corpus user - Indonesian)
    │   ├── squad/ (SQuAD 2.0)
    │   ├── tydi/ (TyDi QA, optional)
    │   ├── indoqa/ (IndoQA, optional)
    │   └── merged/ (unified QA dataset)
    │       ├── qa_train.json
    │       ├── qa_val.json
    │       └── qa_test.json
    ├── output_models/
    │   ├── otak_model.safetensors
    │   ├── otak_engine.py (standalone inference)
    │   ├── training_checkpoint.npz
    │   └── retrieval_index.json
    ├── vocab.json
    """
    
    datasets_dir: Path = field(default_factory=lambda: ROOT_DIR / "datasets")
    output_models_dir: Path = field(default_factory=lambda: ROOT_DIR / "output_models")
    vocab_path: Path = field(default_factory=lambda: ROOT_DIR / "vocab.json")
    final_model_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "otak_model.safetensors"
    )
    checkpoint_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "training_checkpoint.npz"
    )
    
    # Path untuk standalone engine yang ikut di-export
    engine_standalone_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "otak_engine.py"
    )
    
    # Path untuk retrieval index (BM25)
    retrieval_index_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "retrieval_index.json"
    )
    
    # Path untuk converted QA dataset (legacy - single file)
    qa_dataset_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "qa_dataset.json"
    )
    
    # ================================================================
    # MERGED DATASET PATHS (BARU)
    # ================================================================
    # Path untuk unified QA dataset dari prepare_datasets.py
    merged_qa_train_path: Path = field(
        default_factory=lambda: ROOT_DIR / "datasets" / "merged" / "qa_train.json"
    )
    merged_qa_val_path: Path = field(
        default_factory=lambda: ROOT_DIR / "datasets" / "merged" / "qa_val.json"
    )
    merged_qa_test_path: Path = field(
        default_factory=lambda: ROOT_DIR / "datasets" / "merged" / "qa_test.json"
    )
    merged_summary_path: Path = field(
        default_factory=lambda: ROOT_DIR / "datasets" / "merged" / "summary.json"
    )

    def ensure_dirs(self) -> None:
        """
        Membuat folder-folder penting jika belum ada.
        """
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.output_models_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SPECIAL TOKENS CONFIG
# ==============================================================================

@dataclass(frozen=True)
class SpecialTokensConfig:
    """
    Token khusus yang wajib ada dalam tokenizer.
    
    Urutan ID dikunci:
    0 -> [PAD]       : padding
    1 -> [UNK]       : unknown token
    2 -> [BOS]       : begin of sequence
    3 -> [EOS]       : end of sequence
    4 ->  : pemisah dokumen
    
    Catatan:
    - String literal dikonkatenasi untuk menghindari rendering markdown
      yang mungkin menghapus tag  sebagai HTML.
    """
    
    pad: str = "[PAD]"
    unk: str = "[UNK]"
    bos: str = "[BOS]"
    eos: str = "[EOS]"
    endoftext: str = '<' + '|' + 'endoftext' + '|' + '>'
    
    @property
    def ordered_tokens(self) -> Tuple[str, ...]:
        return (
            self.pad,
            self.unk,
            self.bos,
            self.eos,
            self.endoftext,
        )

    @property
    def token_to_id(self) -> Dict[str, int]:
        return {token: idx for idx, token in enumerate(self.ordered_tokens)}
    
    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    @property
    def endoftext_id(self) -> int:
        return 4


# ==============================================================================
# TRAINING CONFIG
# ==============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    """
    Hyperparameter latihan untuk 142K QA samples + ~40M params model.
    
    Dipakai oleh:
    - pipeline/trainer.py
    - pipeline/weight_manager.py
    - pipeline/evaluator.py
    
    Philosophy:
    - Learning rate moderat (2e-4) untuk stable convergence
    - Epochs minimal (3) untuk hindari overfitting di data besar
    - Regularization rendah (data besar = kurang perlu regularization)
    - Sequence length panjang (512) untuk fit context + Q + A
    """
    
    # ================================================================
    # CORE TRAINING PARAMETERS (OPTIMIZED untuk 142K samples)
    # ================================================================
    
    # Learning rate: moderat untuk model medium
    # Standard untuk GPT-2 (117M) = 2.5e-4, untuk 40M bisa sedikit lebih tinggi
    learning_rate: float = 2e-4
    
    # Batch size: 4-8 optimal untuk sequence_length 256 di CPU
    # Memory estimate: ~2-3GB dengan batch=4, seq=256, 40M params
    batch_size: int = 4
    
    # Epochs: 3 sudah cukup untuk 142K samples
    # Lebih dari itu = overfitting
    epochs: int = 3
    
    # Sequence length: cukup untuk context (300 tokens) + Q (20) + A (30) + special (10)
    sequence_length: int = 384
    
    # ================================================================
    # REGULARIZATION (rendah karena data besar)
    # ================================================================
    
    # Dropout: rendah (0.1), data besar sudah cukup regularize
    dropout_rate: float = 0.1
    
    # Weight decay: standard (0.01), tidak terlalu agresif
    weight_decay: float = 0.01
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # ================================================================
    # OPTIMIZER (AdamW)
    # ================================================================
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # ================================================================
    # LEARNING RATE SCHEDULING
    # ================================================================
    
    # Warmup: 1000 steps (~250 batches) untuk stability di awal
    lr_warmup_steps: int = 1000
    
    # Cosine decay: tidak pakai (biar constant LR untuk simplicity)
    lr_decay_steps: int = 0
    lr_decay_factor: float = 0.5
    lr_min: float = 1e-6
    
    # ================================================================
    # GRADIENT & CHECKPOINT
    # ================================================================
    
    # Gradient accumulation: 1 (no accumulation, sudah pakai batch=4)
    gradient_accumulation_steps: int = 1
    
    # Early stopping: disabled (3 epochs sudah optimal)
    early_stopping_patience: int = 0

    # Evaluasi dan checkpoint
    eval_every_n_steps: int = 5000  # Evaluate setiap 5000 steps
    save_checkpoint_every_n_steps: int = 5000  # Checkpoint setiap 5000 steps
    
    # Inline training dari playground
    inline_training_steps: int = 100
    
    # Reproducibility
    seed: int = 1337
    
    # ================================================================
    # RAG TRAINING PARAMETERS (optimized untuk SQuAD)
    # ================================================================
    
    # Context dropout: rendah karena data besar
    # Model belajar dari variasi alami di SQuAD
    context_dropout_rate: float = 0.05
    
    # Context perturbation: rendah, SQuAD sudah variatif
    context_perturbation_rate: float = 0.1
    
    # Progressive training: disabled (data sudah balanced)
    progressive_stages: int = 0
    
    # Multi-context: 2 paragraphs (SQuAD context rata-rata cukup panjang)
    max_context_paragraphs: int = 2
    
    # Data augmentation: 1 (no augmentation, data sudah cukup)
    augmentation_factor: int = 1
    
    # ================================================================
    # CONSTRAINED GENERATION (BARU - untuk training)
    # ================================================================
    
    # Mask vocab saat training: force model pakai tokens dari context
    # 0.0 = disabled (standard training)
    # 0.5 = mask 50% tokens yang tidak ada di context
    # 1.0 = hanya allow tokens dari context (paling strict)
    constrained_training_ratio: float = 0.0  # Off saat training, dipakai saat inference
    
    # Loss weight untuk "impossible questions" (SQuAD 2.0 feature)
    # Higher weight = model lebih serius belajar bilang "tidak tahu"
    impossible_loss_weight: float = 1.0


# ==============================================================================
# MODEL CONFIG
# ==============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Parameter arsitektur model (~40M params, balanced untuk 142K samples).
    
    Dipakai oleh:
    - core/architecture.py
    - core/model_exporter.py
    - core/tokenizer.py untuk vocab_size
    - testing/playground.py
    
    Perhitungan parameter:
    - Embedding: 8000 × 512 = 4.1M
    - 8 layers × (4×512² + 3×512×2048) = 34.9M
    - LM head (weight tying): 0 (shared dengan embedding)
    - Total: ~39M params
    
    Rasio dengan data: 39M / 142K = ~275 params/sample ✅
    """
    
    # ================================================================
    # ARCHITECTURE (40M params target)
    # ================================================================
    
    # Vocab size: 8000 cukup untuk English + Indonesian
    # SQuAD mostly English, tapi butuh coverage untuk both languages
    vocab_size: int = 8000
    
    # Hidden dimension
    embedding_dim: int = 512
    
    # Attention heads
    num_attention_heads: int = 8
    
    # Layers: 8 cukup untuk model medium
    num_layers: int = 8
    
    # Context length: 512 tokens (context 300 + Q 20 + A 30 + special)
    max_position_embeddings: int = 512
    
    # LayerNorm epsilon
    layer_norm_eps: float = 1e-5
    
    # FFN dimension: 4x embedding (standard)
    ffn_hidden_dim: int = 2048
    
    # ================================================================
    # PRECISION
    # ================================================================
    train_dtype: str = "float32"
    export_dtype: str = "float16"  # Save space untuk deployment
    
    # ================================================================
    # SOTA UPGRADE PARAMETERS
    # ================================================================
    
    # RoPE: Rotary Position Embedding (critical untuk long context)
    use_rope: bool = True
    
    # Weight Tying: share embed_tokens dengan lm_head
    # Hemat 4M params, improve generalization
    use_weight_tying: bool = True
    
    # Attention logit soft-capping (tanh)
    attention_logit_cap: float = 30.0
    
    # Final logit soft-capping (tanh)
    final_logit_cap: float = 50.0
    
    # Stochastic Depth (DropPath) - rendah karena model tidak terlalu dalam
    drop_path_rate: float = 0.1
    
    # ================================================================
    # CONSTRAINED GENERATION FLAGS (BARU)
    # ================================================================
    
    # Enable constrained generation saat inference
    # Hanya tokens dari context yang boleh di-generate
    use_constrained_generation: bool = True
    
    # Cross-attention layers (indices where cross-attn is applied)
    # None = apply di semua layers
    # [2, 4, 6] = apply di layer 2, 4, 6 (untuk efficiency)
    cross_attention_layers: Optional[Tuple[int, ...]] = None

    @property
    def head_dim(self) -> int:
        """Dimensi per attention head."""
        if self.num_attention_heads == 0:
            return 0
        return self.embedding_dim // self.num_attention_heads
    
    @property
    def estimated_params(self) -> int:
        """Estimasi jumlah parameter."""
        V = self.vocab_size
        D = self.embedding_dim
        L = self.num_layers
        F = self.ffn_hidden_dim
        
        embedding_params = V * D
        lm_head_params = 0 if self.use_weight_tying else V * D
        
        layer_params = L * (
            2 * D  # 2 LayerNorms
            + 4 * D * D  # Q, K, V, O projections
            + 3 * D * F  # SwiGLU: gate, up, down
        )
        
        return embedding_params + layer_params + lm_head_params


# ==============================================================================
# RETRIEVAL CONFIG
# ==============================================================================

@dataclass(frozen=True)
class RetrievalConfig:
    """
    Parameter untuk retrieval engine (BM25).
    
    Dipakai oleh:
    - core/retriever.py
    - testing/playground.py
    - core/model_exporter.py
    """
    
    # Jumlah paragraf teratas yang diambil sebagai konteks
    # SQuAD context sudah cukup panjang, top-2 optimal
    top_k: int = 3
    
    # Minimum BM25 score untuk dianggap relevan
    min_score: Optional[float] = None
    
    # Dynamic threshold: 30% dari top score
    dynamic_threshold_ratio: float = 0.3
    
    # Query expansion: 2 sinonim per kata
    query_expansion_count: int = 2
    
    # Enable query expansion sebagai fallback
    enable_query_expansion_fallback: bool = True
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Auto-rebuild index saat startup
    auto_rebuild_index: bool = True
    
    # Maximum context length per paragraf (karakter)
    # SQuAD paragraphs rata-rata 600 chars
    max_paragraph_length: int = 800


# ==============================================================================
# GLOBAL CONFIG
# ==============================================================================

@dataclass(frozen=True)
class Config:
    """
    Konfigurasi utama yang menggabungkan seluruh sub-config.
    """
    
    paths: PathsConfig = field(default_factory=PathsConfig)
    special_tokens: SpecialTokensConfig = field(default_factory=SpecialTokensConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def ensure_dirs(self) -> None:
        """Wrapper untuk membuat folder-folder kerja."""
        self.paths.ensure_dirs()

    def validate(self) -> None:
        """Validasi seluruh konfigurasi."""
        t = self.training
        m = self.model
        s = self.special_tokens
        r = self.retrieval

        # ------------------------------------------------------------------
        # Validasi special tokens dan vocab
        # ------------------------------------------------------------------
        if m.vocab_size < len(s.ordered_tokens):
            raise ValueError(
                "model.vocab_size terlalu kecil. "
                f"Minimal harus {len(s.ordered_tokens)} untuk menampung seluruh "
                f"special tokens: {s.ordered_tokens}"
            )

        # ------------------------------------------------------------------
        # Validasi arsitektur model
        # ------------------------------------------------------------------
        if m.embedding_dim <= 0:
            raise ValueError("model.embedding_dim harus lebih besar dari 0.")

        if m.num_attention_heads <= 0:
            raise ValueError("model.num_attention_heads harus lebih besar dari 0.")

        if m.embedding_dim % m.num_attention_heads != 0:
            raise ValueError(
                "model.embedding_dim harus habis dibagi model.num_attention_heads."
            )

        if m.num_layers < 0:
            raise ValueError("model.num_layers tidak boleh negatif.")

        if not (0.0 <= t.dropout_rate < 1.0):
            raise ValueError("training.dropout_rate harus berada pada rentang [0, 1).")

        if m.max_position_embeddings <= 0:
            raise ValueError("model.max_position_embeddings harus lebih besar dari 0.")

        if m.ffn_hidden_dim <= 0:
            raise ValueError("model.ffn_hidden_dim harus lebih besar dari 0.")

        if m.layer_norm_eps <= 0.0:
            raise ValueError("model.layer_norm_eps harus lebih besar dari 0.")

        if m.train_dtype not in ALLOWED_TRAIN_DTYPES:
            raise ValueError(
                f"model.train_dtype tidak valid. Pilihan: {sorted(ALLOWED_TRAIN_DTYPES)}"
            )

        if m.export_dtype not in ALLOWED_EXPORT_DTYPES:
            raise ValueError(
                f"model.export_dtype tidak valid. Pilihan: {sorted(ALLOWED_EXPORT_DTYPES)}"
            )

        # ------------------------------------------------------------------
        # Validasi SOTA parameters
        # ------------------------------------------------------------------
        if m.use_rope:
            head_dim = m.embedding_dim // m.num_attention_heads
            if head_dim % 2 != 0:
                raise ValueError(
                    "RoPE memerlukan head_dim genap. "
                    f"Sekarang: {head_dim}. Pastikan embedding_dim / num_attention_heads genap."
                )

        if m.attention_logit_cap <= 0.0:
            raise ValueError("model.attention_logit_cap harus lebih besar dari 0.")

        if m.final_logit_cap <= 0.0:
            raise ValueError("model.final_logit_cap harus lebih besar dari 0.")

        if not (0.0 <= m.drop_path_rate < 1.0):
            raise ValueError("model.drop_path_rate harus berada pada rentang [0, 1).")

        # ------------------------------------------------------------------
        # Validasi cross_attention_layers
        # ------------------------------------------------------------------
        if m.cross_attention_layers is not None:
            for layer_idx in m.cross_attention_layers:
                if layer_idx < 0 or layer_idx >= m.num_layers:
                    raise ValueError(
                        f"cross_attention_layers index {layer_idx} out of range "
                        f"(must be in [0, {m.num_layers - 1}])"
                    )

        # ------------------------------------------------------------------
        # Validasi RAG parameters
        # ------------------------------------------------------------------
        if r.top_k < 1:
            raise ValueError("retrieval.top_k harus >= 1.")

        if r.min_score is not None and r.min_score < 0.0:
            raise ValueError("retrieval.min_score tidak boleh negatif.")

        if not (0.0 <= r.dynamic_threshold_ratio <= 1.0):
            raise ValueError(
                "retrieval.dynamic_threshold_ratio harus berada pada rentang [0, 1]."
            )

        if r.query_expansion_count < 0:
            raise ValueError("retrieval.query_expansion_count tidak boleh negatif.")

        if r.bm25_k1 <= 0.0:
            raise ValueError("retrieval.bm25_k1 harus lebih besar dari 0.")

        if not (0.0 <= r.bm25_b <= 1.0):
            raise ValueError("retrieval.bm25_b harus berada pada rentang [0, 1].")

        if r.max_paragraph_length < 50:
            raise ValueError("retrieval.max_paragraph_length harus >= 50.")

        # ------------------------------------------------------------------
        # Validasi training parameters (RAG-specific)
        # ------------------------------------------------------------------
        if not (0.0 <= t.context_dropout_rate <= 1.0):
            raise ValueError(
                "training.context_dropout_rate harus berada pada rentang [0, 1]."
            )

        if not (0.0 <= t.context_perturbation_rate <= 1.0):
            raise ValueError(
                "training.context_perturbation_rate harus berada pada rentang [0, 1]."
            )

        if t.progressive_stages < 0:
            raise ValueError("training.progressive_stages tidak boleh negatif.")

        if t.max_context_paragraphs < 1:
            raise ValueError("training.max_context_paragraphs harus >= 1.")

        if t.augmentation_factor < 1:
            raise ValueError("training.augmentation_factor harus >= 1.")

        if not (0.0 <= t.constrained_training_ratio <= 1.0):
            raise ValueError(
                "training.constrained_training_ratio harus berada pada rentang [0, 1]."
            )

        if t.impossible_loss_weight < 0.0:
            raise ValueError("training.impossible_loss_weight tidak boleh negatif.")

        # ------------------------------------------------------------------
        # Validasi hyperparameter training
        # ------------------------------------------------------------------
        if t.learning_rate <= 0.0:
            raise ValueError("training.learning_rate harus lebih besar dari 0.")

        if t.batch_size <= 0:
            raise ValueError("training.batch_size harus lebih besar dari 0.")

        if t.epochs < 0:
            raise ValueError("training.epochs tidak boleh negatif.")

        if t.sequence_length <= 0:
            raise ValueError("training.sequence_length harus lebih besar dari 0.")

        if t.sequence_length > m.max_position_embeddings:
            raise ValueError(
                "training.sequence_length tidak boleh lebih besar dari "
                "model.max_position_embeddings."
            )

        if t.weight_decay < 0.0:
            raise ValueError("training.weight_decay tidak boleh negatif.")

        if t.grad_clip_norm <= 0.0:
            raise ValueError("training.grad_clip_norm harus lebih besar dari 0.")

        if not (0.0 <= t.adam_beta1 < 1.0):
            raise ValueError("training.adam_beta1 harus berada pada rentang [0, 1).")

        if not (0.0 <= t.adam_beta2 < 1.0):
            raise ValueError("training.adam_beta2 harus berada pada rentang [0, 1).")

        if t.adam_epsilon <= 0.0:
            raise ValueError("training.adam_epsilon harus lebih besar dari 0.")

        if t.lr_warmup_steps < 0:
            raise ValueError("training.lr_warmup_steps tidak boleh negatif.")

        if t.lr_decay_steps < 0:
            raise ValueError("training.lr_decay_steps tidak boleh negatif.")

        if not (0.0 < t.lr_decay_factor <= 1.0):
            raise ValueError("training.lr_decay_factor harus berada pada rentang (0, 1].")

        if t.lr_min < 0.0:
            raise ValueError("training.lr_min tidak boleh negatif.")

        if t.lr_min > t.learning_rate:
            raise ValueError("training.lr_min tidak boleh lebih besar dari learning_rate.")

        if t.gradient_accumulation_steps < 1:
            raise ValueError("training.gradient_accumulation_steps harus >= 1.")

        if t.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience tidak boleh negatif.")

        if t.eval_every_n_steps < 0:
            raise ValueError("training.eval_every_n_steps tidak boleh negatif.")

        if t.save_checkpoint_every_n_steps < 0:
            raise ValueError("training.save_checkpoint_every_n_steps tidak boleh negatif.")

        if t.inline_training_steps < 1:
            raise ValueError("training.inline_training_steps harus >= 1.")


# ==============================================================================
# INSTANCE GLOBAL
# ==============================================================================

CONFIG: Final[Config] = Config()

# Validasi langsung saat module di-import
CONFIG.validate()


# ==============================================================================
# DIRECT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    CONFIG.ensure_dirs()

    print("=" * 70)
    print("CONFIGURATION SUMMARY (Path B - Constrained Generation)")
    print("=" * 70)
    
    print(f"\n📁 Paths:")
    print(f"  Root directory        : {ROOT_DIR}")
    print(f"  Datasets directory    : {CONFIG.paths.datasets_dir}")
    print(f"  Output models dir     : {CONFIG.paths.output_models_dir}")
    print(f"  Vocab path            : {CONFIG.paths.vocab_path}")
    print(f"  Final model path      : {CONFIG.paths.final_model_path}")
    print(f"  Checkpoint path       : {CONFIG.paths.checkpoint_path}")
    print(f"  Engine standalone     : {CONFIG.paths.engine_standalone_path}")
    print(f"  Retrieval index       : {CONFIG.paths.retrieval_index_path}")
    print(f"  Merged QA train       : {CONFIG.paths.merged_qa_train_path}")
    print(f"  Merged QA val         : {CONFIG.paths.merged_qa_val_path}")
    print(f"  Merged QA test        : {CONFIG.paths.merged_qa_test_path}")

    print(f"\n🔤 Special tokens:")
    print(f"  [PAD]                 : {CONFIG.special_tokens.pad_id}")
    print(f"  [UNK]                 : {CONFIG.special_tokens.unk_id}")
    print(f"  [BOS]                 : {CONFIG.special_tokens.bos_id}")
    print(f"  [EOS]                 : {CONFIG.special_tokens.eos_id}")
    print(f"     : {CONFIG.special_tokens.endoftext_id}")

    print(f"\n🎯 Model Architecture:")
    print(f"  vocab_size            : {CONFIG.model.vocab_size:,}")
    print(f"  embedding_dim         : {CONFIG.model.embedding_dim}")
    print(f"  num_attention_heads   : {CONFIG.model.num_attention_heads}")
    print(f"  num_layers            : {CONFIG.model.num_layers}")
    print(f"  max_position          : {CONFIG.model.max_position_embeddings}")
    print(f"  ffn_hidden_dim        : {CONFIG.model.ffn_hidden_dim}")
    print(f"  head_dim              : {CONFIG.model.head_dim}")
    print(f"  estimated_params      : {CONFIG.model.estimated_params:,} (~{CONFIG.model.estimated_params/1e6:.1f}M)")
    print(f"  train_dtype           : {CONFIG.model.train_dtype}")
    print(f"  export_dtype          : {CONFIG.model.export_dtype}")
    
    print(f"\n🔬 SOTA Features:")
    print(f"  use_rope              : {CONFIG.model.use_rope}")
    print(f"  use_weight_tying      : {CONFIG.model.use_weight_tying}")
    print(f"  attention_logit_cap   : {CONFIG.model.attention_logit_cap}")
    print(f"  final_logit_cap       : {CONFIG.model.final_logit_cap}")
    print(f"  drop_path_rate        : {CONFIG.model.drop_path_rate}")
    print(f"  cross_attention_layers: {CONFIG.model.cross_attention_layers}")
    print(f"  use_constrained_gen   : {CONFIG.model.use_constrained_generation}")

    print(f"\n📚 Training:")
    print(f"  learning_rate         : {CONFIG.training.learning_rate}")
    print(f"  batch_size            : {CONFIG.training.batch_size}")
    print(f"  epochs                : {CONFIG.training.epochs}")
    print(f"  sequence_length       : {CONFIG.training.sequence_length}")
    print(f"  dropout_rate          : {CONFIG.training.dropout_rate}")
    print(f"  weight_decay          : {CONFIG.training.weight_decay}")
    print(f"  grad_clip_norm        : {CONFIG.training.grad_clip_norm}")
    print(f"  lr_warmup_steps       : {CONFIG.training.lr_warmup_steps}")
    print(f"  gradient_accumulation : {CONFIG.training.gradient_accumulation_steps}")
    print(f"  early_stopping        : {CONFIG.training.early_stopping_patience}")
    print(f"  eval_every_n_steps    : {CONFIG.training.eval_every_n_steps}")
    print(f"  save_checkpoint_every : {CONFIG.training.save_checkpoint_every_n_steps}")

    print(f"\n🔎 RAG Training:")
    print(f"  context_dropout_rate  : {CONFIG.training.context_dropout_rate}")
    print(f"  context_perturbation  : {CONFIG.training.context_perturbation_rate}")
    print(f"  progressive_stages    : {CONFIG.training.progressive_stages}")
    print(f"  max_context_paragraphs: {CONFIG.training.max_context_paragraphs}")
    print(f"  augmentation_factor   : {CONFIG.training.augmentation_factor}")
    print(f"  constrained_training  : {CONFIG.training.constrained_training_ratio}")
    print(f"  impossible_loss_weight: {CONFIG.training.impossible_loss_weight}")

    print(f"\n🔍 Retrieval (BM25):")
    print(f"  top_k                 : {CONFIG.retrieval.top_k}")
    print(f"  min_score             : {CONFIG.retrieval.min_score}")
    print(f"  dynamic_threshold     : {CONFIG.retrieval.dynamic_threshold_ratio}")
    print(f"  query_expansion_count : {CONFIG.retrieval.query_expansion_count}")
    print(f"  bm25_k1               : {CONFIG.retrieval.bm25_k1}")
    print(f"  bm25_b                : {CONFIG.retrieval.bm25_b}")
    print(f"  auto_rebuild_index    : {CONFIG.retrieval.auto_rebuild_index}")
    print(f"  max_paragraph_length  : {CONFIG.retrieval.max_paragraph_length}")

    print(f"\n📊 Training Estimates:")
    train_samples = 113899  # dari prepare_datasets.py
    steps_per_epoch = train_samples // CONFIG.training.batch_size
    total_steps = steps_per_epoch * CONFIG.training.epochs
    time_per_step = 0.3  # seconds (estimated for 40M model)
    total_time_hours = (total_steps * time_per_step) / 3600
    
    print(f"  Train samples         : {train_samples:,}")
    print(f"  Steps per epoch       : {steps_per_epoch:,}")
    print(f"  Total training steps  : {total_steps:,}")
    print(f"  Estimated time        : {total_time_hours:.1f} hours (CPU)")

    print("\n" + "=" * 70)
    print("✅ config.py valid dan optimized untuk 142K samples!")
    print("=" * 70)
    print("\n🚀 Next step: Update core/architecture.py untuk support:")
    print("   - Constrained generation (vocab masking)")
    print("   - Cross-attention ke context")
    print("   - SQuAD-specific QA format")
