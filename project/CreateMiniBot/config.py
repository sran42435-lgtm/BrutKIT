# config.py
#
# Konfigurasi global untuk MLOps / Training Pipeline Engine.
# File ini menjadi sumber parameter tunggal untuk seluruh modul:
# - core/tokenizer.py
# - core/architecture.py
# - core/router.py
# - pipeline/trainer.py
# - pipeline/evaluator.py
# - pipeline/weight_manager.py
# - core/model_exporter.py
# - testing/playground.py
# - main.py
#
# Fitur Unggulan:
# - Dynamic MoE: auto-detect domain dari file dataset baru
# - Auto Routing: keyword-based routing tanpa hardcode domain
# - Standalone Engine: support export ke format portable
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

ALLOWED_KEYWORD_METHODS: Final[frozenset] = frozenset(
    {
        "frequency",  # Top-N kata paling frequent
        "tfidf",      # TF-IDF: kata unik yang khas untuk domain
        "hybrid",     # Frequency × uniqueness score (rekomendasi)
    }
)

ALLOWED_MOE_TRAIN_MODES: Final[frozenset] = frozenset(
    {
        "per_domain",  # Train per domain (backbone shared, head specific)
        "mixed",       # Mix semua domain dalam satu epoch (traditional)
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
    ├── output_models/
    │   ├── otak_model.safetensors
    │   ├── otak_engine.py          (standalone inference)
    │   ├── training_checkpoint.npz
    │   ├── domain_keywords.json
    │   ├── semantic_cache.npz
    │   └── moe_cache/              (cached embeddings per domain)
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

    # Folder untuk cached semantic embeddings per domain
    moe_cache_dir: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "moe_cache"
    )

    def ensure_dirs(self) -> None:
        """
        Membuat folder-folder penting jika belum ada.
        """
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.output_models_dir.mkdir(parents=True, exist_ok=True)
        self.final_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.moe_cache_dir.mkdir(parents=True, exist_ok=True)


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
    4 ->  : pemisah dokumen (mencegah topic drift)

    Perbedaan [EOS] vs :
    - [EOS]    : akhir satu utterance/kalimat pendek
    -  : pemisah antar dokumen/paragraf dalam korpus training

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
    dropout_rate: float = 0.4
    weight_decay: float = 0.15

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
    early_stopping_patience: int = 5

    # Evaluasi dan checkpoint
    eval_every_n_steps: int = 0
    save_checkpoint_every_n_steps: int = 0

    # Inline training dari playground
    inline_training_steps: int = 10

    # Reproducibility
    seed: int = 1337

    # ================================================================
    # MOE TRAINING PARAMETERS
    # ================================================================

    # Metode ekstraksi keywords per domain
    # "frequency" : top-N kata paling frequent
    # "tfidf"     : TF-IDF (kata unik yang khas)
    # "hybrid"    : frequency × uniqueness (rekomendasi)
    moe_keyword_method: str = "hybrid"

    # Jumlah keywords yang diekstrak per domain
    moe_keywords_per_domain: int = 30

    # Mode training MoE
    # "per_domain" : latih per domain secara bergantian (backbone shared)
    # "mixed"      : campur semua domain dalam satu epoch
    moe_train_mode: str = "per_domain"

    # Jumlah epochs per domain dalam mode per_domain
    # Total epochs = epochs_per_domain × num_domains
    moe_epochs_per_domain: int = 30

    # Rebuild semantic cache setelah training selesai
    moe_rebuild_cache_after_training: bool = True


# ==============================================================================
# MODEL CONFIG
# ==============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Parameter arsitektur model.

    Dipakai oleh:
    - core/architecture.py
    - core/router.py
    - core/model_exporter.py
    - core/tokenizer.py untuk vocab_size
    - testing/playground.py
    """

    vocab_size: int = 4096
    embedding_dim: int = 512
    num_attention_heads: int = 16
    num_layers: int = 6

    # Context length maksimal
    max_position_embeddings: int = 96

    # Epsilon untuk LayerNorm / RMSNorm
    layer_norm_eps: float = 1e-5

    # Dimensi Feed-Forward Network
    ffn_hidden_dim: int = 1024

    # Presisi saat training dan export
    train_dtype: str = "float32"
    export_dtype: str = "float16"

    # ================================================================
    # SOTA UPGRADE PARAMETERS
    # ================================================================

    # RoPE: Rotary Position Embedding
    use_rope: bool = True

    # Weight Tying: share bobot embed_tokens dengan lm_head.
    # CATATAN: akan otomatis di-disable jika use_moe=True
    # karena MoE membutuhkan multiple LM heads yang berbeda.
    use_weight_tying: bool = True

    # Attention logit soft-capping (tanh)
    attention_logit_cap: float = 30.0

    # Final logit soft-capping (tanh)
    final_logit_cap: float = 50.0

    # Stochastic Depth (DropPath) base rate
    drop_path_rate: float = 0.1

    # ================================================================
    # MOE PARAMETERS
    # ================================================================

    # Aktifkan Mixture of Experts dengan dynamic LM heads.
    # Jika True, model akan memiliki:
    #   - Shared backbone (17M params)
    #   - Multiple LM heads (satu per domain yang terdeteksi dari datasets/)
    #   - GENERAL head sebagai fallback
    #
    # Jika False, model behave seperti biasa (single LM head).
    use_moe: bool = True

    # Auto-routing saat inference
    # True  : otomatis pilih LM head berdasarkan keyword di prompt
    # False : user harus pilih domain manual via /domain command
    moe_auto_routing: bool = True

    # Confidence threshold untuk router
    # Jika score domain teratas < threshold, fallback ke GENERAL head
    # Range: 0.0 - 1.0
    moe_router_confidence_threshold: float = 0.3

    # Selalu buat head GENERAL sebagai fallback
    # Head ini dilatih dengan data campuran semua domain
    moe_always_create_general_head: bool = True

    # Jumlah top keywords yang digunakan untuk scoring router
    # Lebih tinggi = lebih akurat tapi lebih lambat
    moe_router_top_keywords: int = 30

    # Semantic search: jumlah paragraf relevan yang di-inject ke prompt
    # 0 = disable semantic context injection
    moe_semantic_top_k: int = 2

    # Dimension untuk cached embeddings (semantic search)
    # Jika None, pakai embedding_dim
    moe_embedding_dim: Optional[int] = None

    @property
    def head_dim(self) -> int:
        """
        Dimensi per attention head.
        """
        if self.num_attention_heads == 0:
            return 0
        return self.embedding_dim // self.num_attention_heads

    @property
    def effective_use_weight_tying(self) -> bool:
        """
        Weight tying otomatis di-disable jika MoE aktif.
        """
        if self.use_moe:
            return False
        return self.use_weight_tying

    @property
    def effective_moe_embedding_dim(self) -> int:
        """
        Dimensi embedding untuk semantic search.
        Default ke embedding_dim jika tidak diset.
        """
        if self.moe_embedding_dim is None:
            return self.embedding_dim
        return self.moe_embedding_dim


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
        # Validasi MOE parameters
        # ------------------------------------------------------------------
        if t.moe_keyword_method not in ALLOWED_KEYWORD_METHODS:
            raise ValueError(
                f"training.moe_keyword_method tidak valid. "
                f"Pilihan: {sorted(ALLOWED_KEYWORD_METHODS)}"
            )

        if t.moe_train_mode not in ALLOWED_MOE_TRAIN_MODES:
            raise ValueError(
                f"training.moe_train_mode tidak valid. "
                f"Pilihan: {sorted(ALLOWED_MOE_TRAIN_MODES)}"
            )

        if t.moe_keywords_per_domain < 1:
            raise ValueError("training.moe_keywords_per_domain harus >= 1.")

        if t.moe_epochs_per_domain < 1:
            raise ValueError("training.moe_epochs_per_domain harus >= 1.")

        if not (0.0 <= m.moe_router_confidence_threshold <= 1.0):
            raise ValueError(
                "model.moe_router_confidence_threshold harus berada pada rentang [0, 1]."
            )

        if m.moe_router_top_keywords < 1:
            raise ValueError("model.moe_router_top_keywords harus >= 1.")

        if m.moe_semantic_top_k < 0:
            raise ValueError("model.moe_semantic_top_k tidak boleh negatif.")

        if m.moe_embedding_dim is not None and m.moe_embedding_dim <= 0:
            raise ValueError("model.moe_embedding_dim harus > 0 jika diset.")

        # Warning jika MoE aktif tapi weight tying juga aktif
        if m.use_moe and m.use_weight_tying:
            # Ini bukan error, hanya info - weight tying akan auto-disable
            pass

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
    print(f"MoE cache dir       : {CONFIG.paths.moe_cache_dir}")
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
    print("Training (MoE):")
    print(f"  moe_keyword_method           : {CONFIG.training.moe_keyword_method}")
    print(f"  moe_keywords_per_domain      : {CONFIG.training.moe_keywords_per_domain}")
    print(f"  moe_train_mode               : {CONFIG.training.moe_train_mode}")
    print(f"  moe_epochs_per_domain        : {CONFIG.training.moe_epochs_per_domain}")
    print(f"  moe_rebuild_cache_after_train: {CONFIG.training.moe_rebuild_cache_after_training}")
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
    print(f"  use_rope                          : {CONFIG.model.use_rope}")
    print(f"  use_weight_tying                  : {CONFIG.model.use_weight_tying}")
    print(f"  effective_use_weight_tying        : {CONFIG.model.effective_use_weight_tying}")
    print(f"  attention_logit_cap               : {CONFIG.model.attention_logit_cap}")
    print(f"  final_logit_cap                   : {CONFIG.model.final_logit_cap}")
    print(f"  drop_path_rate                    : {CONFIG.model.drop_path_rate}")
    print()
    print("MoE features:")
    print(f"  use_moe                           : {CONFIG.model.use_moe}")
    print(f"  moe_auto_routing                  : {CONFIG.model.moe_auto_routing}")
    print(f"  moe_router_confidence_threshold   : {CONFIG.model.moe_router_confidence_threshold}")
    print(f"  moe_always_create_general_head    : {CONFIG.model.moe_always_create_general_head}")
    print(f"  moe_router_top_keywords           : {CONFIG.model.moe_router_top_keywords}")
    print(f"  moe_semantic_top_k                : {CONFIG.model.moe_semantic_top_k}")
    print(f"  moe_embedding_dim                 : {CONFIG.model.effective_moe_embedding_dim}")
