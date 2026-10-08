# pipeline/weight_manager.py
#
# Weight Manager / Optimizer Engine untuk Training Pipeline Engine.
#
# Tugas utama:
# 1. Menginisialisasi optimizer AdamW.
# 2. Menyimpan state optimizer (m, v, step).
# 3. Melakukan gradient clipping dengan sanitasi NaN/Inf.
# 4. Memperbarui bobot model setelah backward pass.
# 5. Menyimpan dan memuat checkpoint training (termasuk epoch & global step).
# 6. Mendukung learning rate scheduling (warmup & decay).
#
# Implementasi:
# - Tidak memakai optimizer dari framework ML eksternal.
# - Memakai NumPy sebagai pustaka primitif numerik.
#
# Keterhubungan:
# - config.py            : hyperparameter optimizer
# - core/architecture.py : parameter model dan gradient
# - pipeline/trainer.py  : memanggil step() setelah backward
# - main.py              : checkpoint / resume training

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
# Memastikan project root ada di sys.path, sehingga file di dalam folder
# pipeline/ tetap bisa meng-import config.py meskipun dijalankan langsung:
#   python pipeline/weight_manager.py
# atau:
#   cd pipeline && python weight_manager.py

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================================
# NUMPY IMPORT
# ============================================================================

try:
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "pipeline/weight_manager.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config


class WeightManager:
    """
    Pengelola pembaruan bobot model.

    WeightManager menerima model yang sudah memiliki gradient pada setiap
    parameter, lalu menerapkan AdamW + gradient clipping.

    Fitur tambahan:
    - Sanitasi gradient NaN/Inf
    - Learning rate scheduling (warmup & decay)
    - Penyimpanan epoch & global step di checkpoint

    Alur pemakaian normal:

        model.zero_grad()
        logits = model.forward(input_ids, training=True)
        loss, grad_logits, metrics = evaluator.compute_loss_and_grad(logits, targets)
        model.backward(grad_logits)
        weight_manager.step()
    """

    def __init__(self, model, config: Config = CONFIG):
        self.model = model
        self.config = config

        # Ambil seluruh parameter model.
        # Dictionary ini berisi nama parameter -> objek Parameter.
        self.params = model.parameters()

        # Hyperparameter optimizer
        self.base_learning_rate = float(config.training.learning_rate)
        self.learning_rate = self.base_learning_rate  # LR aktif (bisa berubah karena scheduling)
        self.beta1 = float(config.training.adam_beta1)
        self.beta2 = float(config.training.adam_beta2)
        self.epsilon = float(config.training.adam_epsilon)
        self.weight_decay = float(config.training.weight_decay)
        self.grad_clip_norm = float(config.training.grad_clip_norm)

        # Learning rate scheduling
        self.lr_warmup_steps = int(config.training.lr_warmup_steps)
        self.lr_decay_steps = int(config.training.lr_decay_steps)
        self.lr_decay_factor = float(config.training.lr_decay_factor)
        self.lr_min = float(config.training.lr_min)

        # Step optimizer dimulai dari 0.
        # Akan bertambah menjadi 1 pada pemanggilan step() pertama.
        self.step_count = 0

        # State untuk resume training
        # last_epoch: epoch terakhir yang selesai (0-indexed)
        # global_step: total step training yang sudah dijalankan
        self.last_epoch = 0
        self.global_step = 0

        # State AdamW
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

        for name, param in self.params.items():
            self.m[name] = np.zeros_like(param.data, dtype=np.float32)
            self.v[name] = np.zeros_like(param.data, dtype=np.float32)

        # Lokasi checkpoint.
        # Config menggunakan .json, tetapi checkpoint optimizer berisi array
        # biner, jadi kita pakai format .npz dengan nama dasar yang sama.
        self.checkpoint_path = Path(config.paths.checkpoint_path).with_suffix(".npz")

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def zero_grad(self) -> None:
        """
        Mengosongkan seluruh gradient pada model.
        """
        self.model.zero_grad()

    def set_learning_rate(self, lr: float) -> None:
        """
        Set learning rate secara manual (override scheduling).
        """
        self.learning_rate = max(self.lr_min, float(lr))

    def update_learning_rate(self) -> float:
        """
        Update learning rate berdasarkan scheduling.

        1. Warmup: LR naik linear dari lr_min ke base_learning_rate
        2. Decay: LR turun setiap lr_decay_steps
        3. Clamp ke lr_min

        Return:
        - learning rate yang sedang aktif
        """
        if self.step_count < self.lr_warmup_steps and self.lr_warmup_steps > 0:
            # Warmup: linear dari lr_min ke base_learning_rate
            progress = self.step_count / self.lr_warmup_steps
            self.learning_rate = self.lr_min + (self.base_learning_rate - self.lr_min) * progress
        else:
            # Setelah warmup, cek apakah perlu decay
            self.learning_rate = self.base_learning_rate

            if self.lr_decay_steps > 0:
                # Hitung berapa kali decay sudah terjadi
                steps_after_warmup = max(0, self.step_count - self.lr_warmup_steps)
                num_decays = steps_after_warmup // self.lr_decay_steps

                if num_decays > 0:
                    self.learning_rate = self.base_learning_rate * (self.lr_decay_factor ** num_decays)

        # Clamp ke minimum
        self.learning_rate = max(self.lr_min, self.learning_rate)

        return self.learning_rate

    def step(self, zero_grad: bool = True) -> Dict[str, float]:
        """
        Melakukan satu langkah pembaruan bobot.

        Tahapan:
        1. Sanitasi gradient NaN/Inf.
        2. Gradient clipping.
        3. Update learning rate (scheduling).
        4. Update momentum AdamW.
        5. Terapkan weight decay.
        6. Update parameter.
        7. Sanitasi bobot dari NaN/Inf.
        8. Increment counters.
        9. Opsional zero gradient.
        """
        # Sanitasi gradient sebelum clipping
        self._sanitize_gradients()

        # Gradient clipping
        grad_norm = self.clip_gradients()

        # Jika gradient tidak valid setelah sanitasi, jangan update bobot
        if not np.isfinite(grad_norm) or grad_norm == 0.0:
            if zero_grad:
                self.zero_grad()

            # Tetap increment global_step agar scheduling berjalan
            self.global_step += 1
            self.step_count += 1

            return {
                "step": float(self.step_count),
                "global_step": float(self.global_step),
                "grad_norm": float(grad_norm) if np.isfinite(grad_norm) else 0.0,
                "update_applied": 0.0,
                "learning_rate": float(self.learning_rate),
            }

        # Update learning rate berdasarkan scheduling
        current_lr = self.update_learning_rate()

        self.step_count += 1
        self.global_step += 1

        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon
        weight_decay = self.weight_decay

        bias_correction1 = 1.0 - (beta1 ** self.step_count)
        bias_correction2 = 1.0 - (beta2 ** self.step_count)

        for name, param in self.params.items():
            # Pastikan gradient bersih
            grad = np.nan_to_num(
                param.grad,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)

            param.grad = grad

            m = self.m[name]
            v = self.v[name]

            # Update biased first moment estimate
            m *= beta1
            m += (1.0 - beta1) * grad

            # Update biased second raw moment estimate
            v *= beta2
            v += (1.0 - beta2) * grad * grad

            # Bias correction
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Adam update
            update = m_hat / (np.sqrt(v_hat) + epsilon)

            # AdamW decoupled weight decay
            if weight_decay > 0.0:
                update = update + weight_decay * param.data

            # Update bobot
            param.data -= current_lr * update

            # Sanitasi bobot agar tidak ada NaN/Inf yang menetap
            param.data = np.nan_to_num(
                param.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)

        if zero_grad:
            self.zero_grad()

        return {
            "step": float(self.step_count),
            "global_step": float(self.global_step),
            "grad_norm": float(grad_norm),
            "update_applied": 1.0,
            "learning_rate": float(current_lr),
        }

    def _sanitize_gradients(self) -> None:
        """
        Membersihkan gradient dari NaN/Inf sebelum clipping.
        """
        for param in self.params.values():
            param.grad = np.nan_to_num(
                param.grad,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)

    def clip_gradients(self) -> float:
        """
        Melakukan global gradient norm clipping.

        Return:
        - grad_norm sebelum clipping
        """
        max_norm = self.grad_clip_norm

        total_sq = 0.0

        for param in self.params.values():
            total_sq += float(np.sum(param.grad * param.grad))

        if not np.isfinite(total_sq):
            # Gradient tidak valid, reset semua
            for param in self.params.values():
                param.grad.fill(0.0)
            return 0.0

        total_norm = float(np.sqrt(total_sq))

        if max_norm > 0.0 and total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)

            for param in self.params.values():
                param.grad *= scale

                # Sanitasi lagi setelah scaling (inf * 0 bisa jadi NaN)
                param.grad = np.nan_to_num(
                    param.grad,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32, copy=False)

        return total_norm

    # ========================================================================
    # STATE DICT
    # ========================================================================

    def state_dict(self) -> Dict:
        """
        Mengembalikan state optimizer dalam bentuk dictionary.
        """
        return {
            "step": self.step_count,
            "global_step": self.global_step,
            "last_epoch": self.last_epoch,
            "learning_rate": self.learning_rate,
            "base_learning_rate": self.base_learning_rate,
            "m": {
                name: state.copy()
                for name, state in self.m.items()
            },
            "v": {
                name: state.copy()
                for name, state in self.v.items()
            },
        }

    def load_state_dict(self, state: Dict) -> None:
        """
        Memuat state optimizer dari dictionary.
        """
        if "step" not in state:
            raise ValueError("state_dict WeightManager tidak memiliki 'step'.")

        self.step_count = int(state["step"])
        self.global_step = int(state.get("global_step", self.step_count))
        self.last_epoch = int(state.get("last_epoch", 0))

        if "learning_rate" in state:
            self.learning_rate = float(state["learning_rate"])

        if "base_learning_rate" in state:
            self.base_learning_rate = float(state["base_learning_rate"])

        if "m" in state:
            for name, param in self.params.items():
                if name not in state["m"]:
                    raise KeyError(f"State m untuk parameter {name} tidak ditemukan.")

                arr = np.asarray(state["m"][name], dtype=np.float32)

                if arr.shape != param.data.shape:
                    raise ValueError(
                        f"Shape state m untuk {name} tidak cocok. "
                        f"Diharapkan {param.data.shape}, diterima {arr.shape}."
                    )

                self.m[name] = arr

        if "v" in state:
            for name, param in self.params.items():
                if name not in state["v"]:
                    raise KeyError(f"State v untuk parameter {name} tidak ditemukan.")

                arr = np.asarray(state["v"][name], dtype=np.float32)

                if arr.shape != param.data.shape:
                    raise ValueError(
                        f"Shape state v untuk {name} tidak cocok. "
                        f"Diharapkan {param.data.shape}, diterima {arr.shape}."
                    )

                self.v[name] = arr

    # ========================================================================
    # CHECKPOINT
    # ========================================================================

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Menyimpan checkpoint model + optimizer ke file .npz.

        Jika path tidak diberikan, memakai:
            output_models/training_checkpoint.npz
        """
        if path is None:
            path = self.checkpoint_path
        else:
            path = Path(path)

        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "step": np.array(self.step_count, dtype=np.int64),
            "global_step": np.array(self.global_step, dtype=np.int64),
            "last_epoch": np.array(self.last_epoch, dtype=np.int64),
            "learning_rate": np.array(self.learning_rate, dtype=np.float32),
            "base_learning_rate": np.array(self.base_learning_rate, dtype=np.float32),
        }

        for name, param in self.params.items():
            safe_name = name.replace(".", "__")

            payload[f"model__{safe_name}"] = param.data.astype(np.float32)
            payload[f"adam_m__{safe_name}"] = self.m[name].astype(np.float32)
            payload[f"adam_v__{safe_name}"] = self.v[name].astype(np.float32)

        np.savez_compressed(path, **payload)

        return path

    def load_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Memuat checkpoint model + optimizer dari file .npz.

        Setelah load, attribute berikut tersedia:
        - self.last_epoch: epoch terakhir yang selesai
        - self.global_step: total step training
        - self.step_count: optimizer step
        """
        if path is None:
            path = self.checkpoint_path
        else:
            path = Path(path)

        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan di: {path}")

        data = np.load(path, allow_pickle=False)

        if "step" not in data:
            raise ValueError("File checkpoint tidak valid: tidak memiliki 'step'.")

        self.step_count = int(data["step"])
        self.global_step = int(data.get("global_step", self.step_count))
        self.last_epoch = int(data.get("last_epoch", 0))

        if "learning_rate" in data:
            self.learning_rate = float(data["learning_rate"])

        if "base_learning_rate" in data:
            self.base_learning_rate = float(data["base_learning_rate"])
        else:
            # Fallback untuk checkpoint lama
            self.base_learning_rate = self.learning_rate

        for name, param in self.params.items():
            safe_name = name.replace(".", "__")

            model_key = f"model__{safe_name}"
            m_key = f"adam_m__{safe_name}"
            v_key = f"adam_v__{safe_name}"

            if model_key not in data:
                raise KeyError(f"Checkpoint tidak memiliki bobot untuk: {name}")

            if m_key not in data or v_key not in data:
                raise KeyError(f"Checkpoint tidak memiliki optimizer state untuk: {name}")

            model_arr = np.array(data[model_key], dtype=np.float32)
            m_arr = np.array(data[m_key], dtype=np.float32)
            v_arr = np.array(data[v_key], dtype=np.float32)

            if model_arr.shape != param.data.shape:
                raise ValueError(
                    f"Shape bobot checkpoint untuk {name} tidak cocok. "
                    f"Diharapkan {param.data.shape}, diterima {model_arr.shape}."
                )

            if m_arr.shape != param.data.shape:
                raise ValueError(
                    f"Shape optimizer m checkpoint untuk {name} tidak cocok. "
                    f"Diharapkan {param.data.shape}, diterima {m_arr.shape}."
                )

            if v_arr.shape != param.data.shape:
                raise ValueError(
                    f"Shape optimizer v checkpoint untuk {name} tidak cocok. "
                    f"Diharapkan {param.data.shape}, diterima {v_arr.shape}."
                )

            param.data = model_arr
            param.grad = np.zeros_like(model_arr, dtype=np.float32)

            self.m[name] = m_arr
            self.v[name] = v_arr

        return path


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    from config import (
        Config,
        ModelConfig,
        PathsConfig,
        SpecialTokensConfig,
        TrainingConfig,
    )
    from core.architecture import CustomTransformerLM

    small_config = Config(
        paths=PathsConfig(),
        special_tokens=SpecialTokensConfig(),
        training=TrainingConfig(
            learning_rate=3e-4,
            batch_size=1,
            epochs=1,
            sequence_length=8,
            weight_decay=0.01,
            grad_clip_norm=1.0,
            seed=1337,
        ),
        model=ModelConfig(
            vocab_size=64,
            embedding_dim=32,
            num_attention_heads=4,
            num_layers=1,
            dropout_rate=0.0,
            max_position_embeddings=32,
            layer_norm_eps=1e-5,
            ffn_hidden_dim=48,
            train_dtype="float32",
            export_dtype="float32",
        ),
    )

    model = CustomTransformerLM(small_config)
    weight_manager = WeightManager(model, small_config)

    input_ids = np.array([[2, 5, 7, 10, 3]], dtype=np.int64)

    # Forward pass
    model.zero_grad()
    logits = model.forward(input_ids, training=True)

    # Gradient dummy.
    # Pada pipeline nyata, gradient ini datang dari evaluator.compute_loss_and_grad()
    # setelah Cross-Entropy Loss dihitung.
    grad_logits = np.ones_like(logits, dtype=np.float32)

    # Backward pass mengisi gradient ke seluruh parameter model
    model.backward(grad_logits)

    # Update bobot memakai AdamW
    info = weight_manager.step()

    print("pipeline/weight_manager.py test OK")
    print(f"Parameter count : {model.parameter_count():,}")
    print(f"Step info       : {info}")

    # Test checkpoint
    ckpt_path = weight_manager.save_checkpoint()
    print(f"Checkpoint saved : {ckpt_path}")

    weight_manager.load_checkpoint(ckpt_path)
    print("Checkpoint loaded OK")
