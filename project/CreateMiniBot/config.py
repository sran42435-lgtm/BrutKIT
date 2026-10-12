# config.py
#
# Konfigurasi global untuk RAG-based Training Pipeline Engine.
# File ini menjadi sumber parameter tunggal untuk seluruh modul:
# - core/tokenizer.py
# - core/architecture.py
# - core/retriever.py (BARU)
# - pipeline/trainer.py
# - pipeline/evaluator.py
# - pipeline/weight_manager.py
# - core/model_exporter.py
# - testing/playground.py
# - main.py
#
# Arsitektur: Single-Head + RAG (Retrieval-Augmented Generation)
#
# Fitur Utama:
# - Retrieval-based context injection (BM25)
# - Single LM head (no domain contamination)
# - Context-conditioned training
# - SOTA: RoPE, Weight Tying, Soft-Capping, DropPath
#
# Implementasi ini sengaja tidak bergantung pada pustaka ML eksternal.
# Hanya menggunakan standard library Python.

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
    │   ├── *.txt (raw corpus)
    │   └── *.json (converted QA format, optional)
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
    
    # Path untuk converted QA dataset (hasil konversi dari raw corpus)
    qa_dataset_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "qa_dataset.json"
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
    Hyperparameter latihan.
    
    Dipakai oleh:
    - pipeline/trainer.py
    - pipeline/weight_manager.py
    - pipeline/evaluator.py
    """
    
    # Learning rate
    learning_rate: float = 3e-4
    
    batch_size: int = 2
    epochs: int = 500
    sequence_length: int = 64
    
    # Regularization
    dropout_rate: float = 0.5
    weight_decay: float = 0.3
    
    # Gradient clipping
    grad_clip_norm: float = 1.0
    
    # AdamW optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    
    # Learning rate scheduling
    lr_warmup_steps: int = 100
    lr_decay_steps: int = 0
    lr_decay_factor: float = 0.5
    lr_min: float = 1e-6
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Early stopping
    early_stopping_patience: int = 20

    # Evaluasi dan checkpoint
    eval_every_n_steps: int = 0
    save_checkpoint_every_n_steps: int = 0
    
    # Inline training dari playground
    inline_training_steps: int = 10
    
    # Reproducibility
    seed: int = 1337
    
    # ================================================================
    # RAG TRAINING PARAMETERS
    # ================================================================
    
    # Context dropout rate (probabilitas training TANPA konteks)
    # 0.0 = selalu dengan konteks
    # 0.2 = 20% waktu tanpa konteks (recommended)
    # Mitigasi risiko: model hallucinate tanpa konteks
    context_dropout_rate: float = 0.3
    
    # Context perturbation (probabilitas shuffle kalimat dalam konteks)
    # 0.0 = konteks selalu dalam urutan asli
    # 0.3 = 30% waktu kalimat di-shuffle (recommended)
    # Mitigasi risiko: model hafal urutan kata
    context_perturbation_rate: float = 0.4
    
    # Progressive training stages (0 = disable)
    # Jika > 0, training dibagi menjadi N stages dengan difficulty meningkat
    # Stage 1: single context, simple questions
    # Stage 2: multi context, complex questions
    # Stage 3: context dropout + noise injection
    progressive_stages: int = 3
    
    # Multi-context: jumlah paragraf yang digabung sebagai konteks
    # 1 = single paragraph context
    # 3 = combine 3 paragraphs (recommended)
    max_context_paragraphs: int = 3
    
    # Data augmentation: jumlah variasi per QA pair
    # 1 = no augmentation (hanya template asli)
    # 3 = 3 variasi per QA pair (paraphrase, permute, truncate)
    augmentation_factor: int = 5


# ==============================================================================
# MODEL CONFIG
# ==============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Parameter arsitektur model.
    
    Dipakai oleh:
    - core/architecture.py
    - core/model_exporter.py
    - core/tokenizer.py untuk vocab_size
    - testing/playground.py
    """
    
    vocab_size: int = 4096
    embedding_dim: int = 768
    num_attention_heads: int = 12
    num_layers: int = 12
    
    # Context length maksimal
    max_position_embeddings: int = 96
    
    # Epsilon untuk LayerNorm / RMSNorm
    layer_norm_eps: float = 1e-5
    
    # Dimensi Feed-Forward Network
    ffn_hidden_dim: int = 3072
    
    # Presisi saat training dan export
    train_dtype: str = "float32"
    export_dtype: str = "float16"
    
    # ================================================================
    # SOTA UPGRADE PARAMETERS
    # ================================================================
    
    # RoPE: Rotary Position Embedding
    use_rope: bool = True
    
    # Weight Tying: share bobot embed_tokens dengan lm_head
    use_weight_tying: bool = True
    
    # Attention logit soft-capping (tanh)
    attention_logit_cap: float = 30.0
    
    # Final logit soft-capping (tanh)
    final_logit_cap: float = 50.0
    
    # Stochastic Depth (DropPath) base rate
    drop_path_rate: float = 0.2

    @property
    def head_dim(self) -> int:
        """
        Dimensi per attention head.
        """
        if self.num_attention_heads == 0:
            return 0
        return self.embedding_dim // self.num_attention_heads


# ==============================================================================
# RETRIEVAL CONFIG (BARU)
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
    # 1 = hanya 1 paragraf paling relevan
    # 3 = top 3 paragraf (recommended)
    # 5 = top 5 paragraf
    top_k: int = 3
    
    # Minimum BM25 score untuk dianggap relevan
    # 0.0 = semua paragraf dianggap relevan (tidak recommended)
    # 0.1 = threshold rendah (ambil banyak, mungkin ada noise)
    # 0.5 = threshold ketat (hanya yang sangat relevan)
    # None = gunakan dynamic threshold (recommended)
    min_score: Optional[float] = None
    
    # Dynamic threshold: gunakan X% dari top score sebagai threshold
    # 0.3 = paragraf dengan score < 30% dari top score akan di-filter
    # 0.5 = paragraf dengan score < 50% dari top score akan di-filter
    # Recommended: 0.3 - 0.5
    dynamic_threshold_ratio: float = 0.3
    
    # Query expansion: jumlah sinonim yang ditambahkan per kata
    # 0 = disable query expansion
    # 2 = tambahkan 2 sinonim per kata (recommended)
    # 5 = tambahkan 5 sinonim per kata (aggressive)
    query_expansion_count: int = 2
    
    # Enable query expansion sebagai fallback jika retrieval gagal
    # True = jika tidak ada hasil, coba expand query dan retrieve lagi
    # False = jika tidak ada hasil, langsung return None
    enable_query_expansion_fallback: bool = True
    
    # BM25 parameters
    # k1: term frequency saturation (1.2 - 2.0, recommended 1.5)
    bm25_k1: float = 1.5
    
    # b: document length normalization (0.0 - 1.0, recommended 0.75)
    bm25_b: float = 0.75
    
    # Auto-rebuild index saat startup jika corpus berubah
    auto_rebuild_index: bool = True
    
    # Maximum context length (karakter) per paragraf
    # Paragraf lebih panjang akan di-truncate
    # 200 = pendek (hemat token)
    # 500 = medium (recommended)
    # 1000 = panjang (lebih banyak info, tapi boros token)
    max_paragraph_length: int = 500


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
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)  # BARU

    def ensure_dirs(self) -> None:
        """
        Wrapper untuk membuat folder-folder kerja.
        """
        self.paths.ensure_dirs()

    def validate(self) -> None:
        """
        Validasi seluruh konfigurasi agar tidak ada parameter yang saling
        bertentangan sebelum modul lain dijalankan.
        """
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

        # Validasi learning rate scheduling
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

        # Validasi gradient accumulation
        if t.gradient_accumulation_steps < 1:
            raise ValueError("training.gradient_accumulation_steps harus >= 1.")

        # Validasi early stopping
        if t.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience tidak boleh negatif.")

        # Validasi eval/checkpoint
        if t.eval_every_n_steps < 0:
            raise ValueError("training.eval_every_n_steps tidak boleh negatif.")

        if t.save_checkpoint_every_n_steps < 0:
            raise ValueError("training.save_checkpoint_every_n_steps tidak boleh negatif.")

        # Validasi inline training
        if t.inline_training_steps < 1:
            raise ValueError("training.inline_training_steps harus >= 1.")


# ==============================================================================
# INSTANCE GLOBAL
# ==============================================================================

CONFIG: Final[Config] = Config()

# Validasi langsung saat module di-import agar error konfigurasi terlihat dini.
CONFIG.validate()


# ==============================================================================
# DIRECT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    CONFIG.ensure_dirs()

    print("config.py valid.")
    print(f"Root directory      : {ROOT_DIR}")
    print(f"Datasets directory  : {CONFIG.paths.datasets_dir}")
    print(f"Output models dir   : {CONFIG.paths.output_models_dir}")
    print(f"Vocab path          : {CONFIG.paths.vocab_path}")
    print(f"Final model path    : {CONFIG.paths.final_model_path}")
    print(f"Checkpoint path     : {CONFIG.paths.checkpoint_path}")
    print(f"Engine standalone   : {CONFIG.paths.engine_standalone_path}")
    print(f"Retrieval index     : {CONFIG.paths.retrieval_index_path}")
    print(f"QA dataset          : {CONFIG.paths.qa_dataset_path}")
    print()
    print("Special tokens:")
    print(f"  [PAD]              : {CONFIG.special_tokens.pad_id}")
    print(f"  [UNK]              : {CONFIG.special_tokens.unk_id}")
    print(f"  [BOS]              : {CONFIG.special_tokens.bos_id}")
    print(f"  [EOS]              : {CONFIG.special_tokens.eos_id}")
    print(f"    : {CONFIG.special_tokens.endoftext_id}")
    print()
    print("Training config:")
    print(f"  learning_rate                : {CONFIG.training.learning_rate}")
    print(f"  batch_size                   : {CONFIG.training.batch_size}")
    print(f"  epochs                       : {CONFIG.training.epochs}")
    print(f"  sequence_length              : {CONFIG.training.sequence_length}")
    print(f"  dropout_rate                 : {CONFIG.training.dropout_rate}")
    print(f"  weight_decay                 : {CONFIG.training.weight_decay}")
    print(f"  grad_clip_norm               : {CONFIG.training.grad_clip_norm}")
    print(f"  lr_warmup_steps              : {CONFIG.training.lr_warmup_steps}")
    print(f"  lr_decay_steps               : {CONFIG.training.lr_decay_steps}")
    print(f"  lr_decay_factor              : {CONFIG.training.lr_decay_factor}")
    print(f"  lr_min                       : {CONFIG.training.lr_min}")
    print(f"  gradient_accumulation_steps  : {CONFIG.training.gradient_accumulation_steps}")
    print(f"  early_stopping_patience      : {CONFIG.training.early_stopping_patience}")
    print(f"  eval_every_n_steps           : {CONFIG.training.eval_every_n_steps}")
    print(f"  save_checkpoint_every_n_steps: {CONFIG.training.save_checkpoint_every_n_steps}")
    print(f"  inline_training_steps        : {CONFIG.training.inline_training_steps}")
    print()
    print("Training (RAG-specific):")
    print(f"  context_dropout_rate         : {CONFIG.training.context_dropout_rate}")
    print(f"  context_perturbation_rate    : {CONFIG.training.context_perturbation_rate}")
    print(f"  progressive_stages           : {CONFIG.training.progressive_stages}")
    print(f"  max_context_paragraphs       : {CONFIG.training.max_context_paragraphs}")
    print(f"  augmentation_factor          : {CONFIG.training.augmentation_factor}")
    print()
    print("Model config:")
    print(f"  vocab_size        : {CONFIG.model.vocab_size}")
    print(f"  embedding_dim     : {CONFIG.model.embedding_dim}")
    print(f"  num_heads         : {CONFIG.model.num_attention_heads}")
    print(f"  num_layers        : {CONFIG.model.num_layers}")
    print(f"  max_position      : {CONFIG.model.max_position_embeddings}")
    print(f"  ffn_hidden_dim    : {CONFIG.model.ffn_hidden_dim}")
    print(f"  head_dim          : {CONFIG.model.head_dim}")
    print(f"  train_dtype       : {CONFIG.model.train_dtype}")
    print(f"  export_dtype      : {CONFIG.model.export_dtype}")
    print()
    print("SOTA features:")
    print(f"  use_rope              : {CONFIG.model.use_rope}")
    print(f"  use_weight_tying      : {CONFIG.model.use_weight_tying}")
    print(f"  attention_logit_cap   : {CONFIG.model.attention_logit_cap}")
    print(f"  final_logit_cap       : {CONFIG.model.final_logit_cap}")
    print(f"  drop_path_rate        : {CONFIG.model.drop_path_rate}")
    print()
    print("Retrieval config (BM25):")
    print(f"  top_k                        : {CONFIG.retrieval.top_k}")
    print(f"  min_score                    : {CONFIG.retrieval.min_score}")
    print(f"  dynamic_threshold_ratio      : {CONFIG.retrieval.dynamic_threshold_ratio}")
    print(f"  query_expansion_count        : {CONFIG.retrieval.query_expansion_count}")
    print(f"  enable_query_expansion_fb    : {CONFIG.retrieval.enable_query_expansion_fallback}")
    print(f"  bm25_k1                      : {CONFIG.retrieval.bm25_k1}")
    print(f"  bm25_b                       : {CONFIG.retrieval.bm25_b}")
    print(f"  auto_rebuild_index           : {CONFIG.retrieval.auto_rebuild_index}")
    print(f"  max_paragraph_length         : {CONFIG.retrieval.max_paragraph_length}")
