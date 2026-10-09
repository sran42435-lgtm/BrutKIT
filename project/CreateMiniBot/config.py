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
        default_factory=lambda: ROOT_DIR / "output_models" / "training_checkpoint.npz"
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

    # Learning rate: diturunkan dari 3e-4 ke 1e-4 untuk stabilitas
    learning_rate: float = 3e-4

    batch_size: int = 2
    epochs: int = 500
    sequence_length: int = 24

    # Regularization: dinaikkan dari 0.0 ke 0.1 untuk mencegah overfit
    dropout_rate: float = 0.4
    weight_decay: float = 0.1

    # Gradient clipping untuk mencegah exploding gradient
    grad_clip_norm: float = 1.0

    # AdamW optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # Learning rate scheduling
    # Warmup: LR naik perlahan di awal training
    lr_warmup_steps: int = 100

    # Decay: LR turun setelah step tertentu
    # 0 = tidak ada decay, > 0 = decay setiap N step
    lr_decay_steps: int = 0

    # Faktor decay (misal 0.5 = LR dibagi 2 setiap lr_decay_steps)
    lr_decay_factor: float = 0.5

    # Minimum learning rate (LR tidak akan turun di bawah ini)
    lr_min: float = 1e-6

    # Gradient accumulation untuk effective batch size lebih besar
    # effective_batch_size = batch_size * gradient_accumulation_steps
    # 1 = tidak ada accumulation
    gradient_accumulation_steps: int = 1

    # Early stopping
    # 0 = tidak ada early stopping
    # > 0 = stop jika loss tidak turun selama N epoch
    early_stopping_patience: int = 5

    # Evaluasi dan checkpoint
    # 0 = hanya di akhir epoch
    # > 0 = evaluasi/checkpoint setiap N step
    eval_every_n_steps: int = 0
    save_checkpoint_every_n_steps: int = 0

    # Inline training dari playground
    # Jumlah step default saat user mengetik "training" di interactive mode
    inline_training_steps: int = 10

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

    # RoPE: Rotary Position Embedding (menggantikan sinusoidal absolut).
    # Menerapkan rotasi pada Q dan K secara relatif, jauh lebih baik untuk
    # menjaga koherensi konteks panjang.
    use_rope: bool = True

    # Weight Tying: share bobot embed_tokens dengan lm_head.
    # Memangkas parameter, menyelaraskan ruang representasi input-output,
    # dan mencegah model menghafal token ID secara berlebihan.
    use_weight_tying: bool = True

    # Attention logit soft-capping (tanh).
    # Mencegah attention scores meledak tanpa memotong gradient keras.
    # Nilai 30.0 mengikuti Gemma.
    attention_logit_cap: float = 30.0

    # Final logit soft-capping (tanh).
    # Mencegah logits output meledak sebelum masuk ke Cross-Entropy Loss.
    # Nilai 50.0 mengikuti Gemma.
    final_logit_cap: float = 50.0

    # Stochastic Depth (DropPath) base rate.
    # Probabilitas drop meningkat linear dari 0 di layer pertama ke
    # drop_path_rate di layer terakhir.
    # Nilai 0.1 adalah rekomendasi umum untuk model Transformer kecil-menengah.
    drop_path_rate: float = 0.1

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
