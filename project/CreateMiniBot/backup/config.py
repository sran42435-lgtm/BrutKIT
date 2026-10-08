# config.py
#
# Konfigurasi global untuk MLOps / Training Pipeline Engine.
# File ini menjadi sumber parameter tunggal untuk seluruh modul:
# - core/tokenizer.py
# - core/architecture.py
# - pipeline/trainer.py
# - pipeline/evaluator.py
# - pipeline/weight_manager.py
# - core/model_exporter.py
# - testing/playground.py
# - main.py
#
# Implementasi ini sengaja tidak bergantung pada pustaka ML eksternal.
# Hanya menggunakan standard library Python.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, Tuple


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
    plural-dev/
    ├── datasets/
    ├── output_models/
    ├── vocab.json
    """

    datasets_dir: Path = field(default_factory=lambda: ROOT_DIR / "datasets")
    output_models_dir: Path = field(default_factory=lambda: ROOT_DIR / "output_models")
    vocab_path: Path = field(default_factory=lambda: ROOT_DIR / "vocab.json")
    final_model_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "otak_model.safetensors"
    )
    checkpoint_path: Path = field(
        default_factory=lambda: ROOT_DIR / "output_models" / "training_checkpoint.json"
    )

    def ensure_dirs(self) -> None:
        """
        Membuat folder-folder penting jika belum ada.

        Dipanggil secara eksplisit oleh main.py atau saat pengguna menjalankan:
            python config.py
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
    0 -> [PAD]
    1 -> [UNK]
    2 -> [BOS]
    3 -> [EOS]
    """

    pad: str = "[PAD]"
    unk: str = "[UNK]"
    bos: str = "[BOS]"
    eos: str = "[EOS]"

    @property
    def ordered_tokens(self) -> Tuple[str, ...]:
        return (self.pad, self.unk, self.bos, self.eos)

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

    learning_rate: float = 3e-4
    batch_size: int = 2
    epochs: int = 101
    sequence_length: int = 24

    # Regularization
    weight_decay: float = 0.01

    # Gradient clipping untuk mencegah exploding gradient
    grad_clip_norm: float = 1.0

    # AdamW optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # Reproducibility
    seed: int = 1337


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

    vocab_size: int = 512
    embedding_dim: int = 96
    num_attention_heads: int = 4
    num_layers: int = 3
    dropout_rate: float = 0.0

    # Context length maksimal
    max_position_embeddings: int = 96

    # Epsilon untuk LayerNorm / RMSNorm
    layer_norm_eps: float = 1e-5

    # Dimensi Feed-Forward Network
    # Contoh pada spesifikasi menggunakan 2048 -> 5632
    ffn_hidden_dim: int = 192

    # Presisi saat training dan export
    train_dtype: str = "float32"
    export_dtype: str = "float16"

    @property
    def head_dim(self) -> int:
        """
        Dimensi per attention head.

        Rumus:
            head_dim = embedding_dim / num_attention_heads
        """
        if self.num_attention_heads == 0:
            return 0
        return self.embedding_dim // self.num_attention_heads


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
                "Minimal harus menampung seluruh special tokens."
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

        if not (0.0 <= m.dropout_rate < 1.0):
            raise ValueError("model.dropout_rate harus berada pada rentang [0, 1).")

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
    # Jika file ini dijalankan langsung, buat folder-folder yang dibutuhkan
    # dan tampilkan ringkasan konfigurasi.

    CONFIG.ensure_dirs()

    print("config.py valid.")
    print(f"Root directory      : {ROOT_DIR}")
    print(f"Datasets directory  : {CONFIG.paths.datasets_dir}")
    print(f"Output models dir   : {CONFIG.paths.output_models_dir}")
    print(f"Vocab path          : {CONFIG.paths.vocab_path}")
    print(f"Final model path    : {CONFIG.paths.final_model_path}")
    print(f"Checkpoint path     : {CONFIG.paths.checkpoint_path}")
    print()
    print("Training config:")
    print(f"  learning_rate     : {CONFIG.training.learning_rate}")
    print(f"  batch_size        : {CONFIG.training.batch_size}")
    print(f"  epochs            : {CONFIG.training.epochs}")
    print(f"  sequence_length   : {CONFIG.training.sequence_length}")
    print(f"  weight_decay      : {CONFIG.training.weight_decay}")
    print(f"  grad_clip_norm    : {CONFIG.training.grad_clip_norm}")
    print()
    print("Model config:")
    print(f"  vocab_size        : {CONFIG.model.vocab_size}")
    print(f"  embedding_dim     : {CONFIG.model.embedding_dim}")
    print(f"  num_heads         : {CONFIG.model.num_attention_heads}")
    print(f"  num_layers        : {CONFIG.model.num_layers}")
    print(f"  dropout_rate      : {CONFIG.model.dropout_rate}")
    print(f"  max_position      : {CONFIG.model.max_position_embeddings}")
    print(f"  ffn_hidden_dim    : {CONFIG.model.ffn_hidden_dim}")
    print(f"  head_dim          : {CONFIG.model.head_dim}")
    print(f"  train_dtype       : {CONFIG.model.train_dtype}")
    print(f"  export_dtype      : {CONFIG.model.export_dtype}")
