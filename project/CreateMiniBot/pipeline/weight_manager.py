# pipeline/weight_manager.py
#
# Weight Manager / Optimizer Engine untuk Training Pipeline Engine.
#
# VERSI SOTA-ENHANCED (v2.0):
# - AdamW optimizer dengan gradient clipping
# - Learning rate scheduling (warmup + cosine + decay)
# - EMA Weights (Polyak Averaging) - save/load/apply
# - Sanitasi gradient & bobot dari NaN/Inf
# - Checkpoint save/load dengan backward compatibility
#
# Tugas utama:
# 1. Menginisialisasi optimizer AdamW.
# 2. Menyimpan state optimizer (m, v, step).
# 3. Melakukan gradient clipping dengan sanitasi NaN/Inf.
# 4. Memperbarui bobot model setelah backward pass.
# 5. Menyimpan dan memuat checkpoint training (termasuk epoch & global step).
# 6. Mendukung learning rate scheduling (warmup & decay).
# 7. 🎯 NEW: Menyimpan/memuat EMA weights untuk inference yang lebih smooth.
#
# Implementasi:
# - Tidak memakai optimizer dari framework ML eksternal.
# - Memakai NumPy sebagai pustaka primitif numerik.
#
# Keterhubungan:
# - config.py            : hyperparameter optimizer + EMA
# - core/architecture.py : parameter model dan gradient
# - pipeline/trainer.py  : memanggil step() setelah backward, update EMA
# - main.py              : checkpoint / resume training
# - testing/playground.py: apply EMA weights untuk inference

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
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
    Pengelola pembaruan bobot model dengan SOTA enhancements.

    WeightManager menerima model yang sudah memiliki gradient pada setiap
    parameter, lalu menerapkan AdamW + gradient clipping.

    Fitur:
    - Sanitasi gradient NaN/Inf
    - Gradient clipping (global norm)
    - Learning rate scheduling (warmup + decay)
    - 🎯 EMA Weights (Polyak Averaging) untuk inference lebih smooth
    - Penyimpanan epoch & global step di checkpoint

    Alur pemakaian normal:

        model.zero_grad()
        logits = model.forward(input_ids, training=True)
        loss, grad_logits, metrics = evaluator.compute_loss_and_grad(logits, targets)
        model.backward(grad_logits)
        weight_manager.step()
        
        # (Optional) Update EMA di trainer:
        # trainer._update_ema()
    """

    def __init__(self, model, config: Config = CONFIG):
        self.model = model
        self.config = config

        # Ambil seluruh parameter model.
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
        self.step_count = 0

        # State untuk resume training
        self.last_epoch = 0
        self.global_step = 0

        # State AdamW
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

        for name, param in self.params.items():
            self.m[name] = np.zeros_like(param.data, dtype=np.float32)
            self.v[name] = np.zeros_like(param.data, dtype=np.float32)

        # ================================================================
        # 🎯 EMA WEIGHTS STATE (BARU)
        # ================================================================
        # EMA (Exponential Moving Average) weights untuk smoother inference
        # Diupdate setiap step: ema = decay * ema + (1 - decay) * current
        self.ema_weights: Dict[str, np.ndarray] = {}
        self.ema_decay: float = float(getattr(config.training, 'ema_decay', 0.999))
        self.use_ema: bool = bool(getattr(config.training, 'use_ema', True))
        self._ema_initialized: bool = False

        # Lokasi checkpoint.
        self.checkpoint_path = Path(config.paths.checkpoint_path).with_suffix(".npz")

    # ========================================================================
    # PUBLIC API - OPTIMIZER
    # ========================================================================

    def zero_grad(self) -> None:
        """Mengosongkan seluruh gradient pada model."""
        self.model.zero_grad()

    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate secara manual (override scheduling)."""
        self.learning_rate = max(self.lr_min, float(lr))

    def update_learning_rate(self) -> float:
        """
        Update learning rate berdasarkan scheduling.

        1. Warmup: LR naik linear dari lr_min ke base_learning_rate
        2. Decay: LR turun setiap lr_decay_steps
        3. Clamp ke lr_min

        Catatan: Trainer SOTA menggunakan cosine schedule yang di-apply
        via set_learning_rate(), jadi method ini hanya untuk fallback.

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

            # Sanitasi bobot
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
        """Membersihkan gradient dari NaN/Inf sebelum clipping."""
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
            for param in self.params.values():
                param.grad.fill(0.0)
            return 0.0

        total_norm = float(np.sqrt(total_sq))

        if max_norm > 0.0 and total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)

            for param in self.params.values():
                param.grad *= scale

                param.grad = np.nan_to_num(
                    param.grad,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32, copy=False)

        return total_norm

    # ========================================================================
    # 🎯 EMA WEIGHTS (BARU)
    # ========================================================================

    def initialize_ema(self) -> None:
        """
        Initialize EMA weights dengan current model weights.
        
        Dipanggil sekali di awal training atau setelah load checkpoint
        yang tidak punya EMA.
        """
        self.ema_weights = {
            name: param.data.copy()
            for name, param in self.params.items()
        }
        self._ema_initialized = True

    def update_ema(self) -> None:
        """
        Update EMA weights dengan current model weights.
        
        Formula: ema = decay * ema + (1 - decay) * current
        
        Dipanggil setiap step oleh trainer (jika use_ema=True).
        """
        if not self.use_ema:
            return
        
        # Initialize jika belum
        if not self._ema_initialized:
            self.initialize_ema()
            return
        
        decay = self.ema_decay
        
        for name, param in self.params.items():
            if name in self.ema_weights:
                self.ema_weights[name] = (
                    decay * self.ema_weights[name]
                    + (1.0 - decay) * param.data
                )

    def apply_ema_to_model(self) -> bool:
        """
        Apply EMA weights ke model (untuk inference).
        
        WARNING: Ini akan REPLACE current model weights dengan EMA.
        Biasanya dilakukan:
        - Di akhir training (final model)
        - Sebelum evaluation
        - Sebelum export
        
        Return:
        - True jika sukses, False jika EMA belum tersedia
        """
        if not self._ema_initialized or not self.ema_weights:
            return False
        
        for name, param in self.params.items():
            if name in self.ema_weights:
                param.data = self.ema_weights[name].copy()
        
        return True

    def get_ema_weights(self) -> Dict[str, np.ndarray]:
        """Return copy dari EMA weights dict."""
        if not self.ema_weights:
            return {}
        
        return {
            name: arr.copy()
            for name, arr in self.ema_weights.items()
        }

    def set_ema_weights(self, ema_dict: Dict[str, np.ndarray]) -> None:
        """
        Set EMA weights dari external dict.
        
        Args:
            ema_dict: dict nama parameter → numpy array
        """
        self.ema_weights = {
            name: np.asarray(arr, dtype=np.float32).copy()
            for name, arr in ema_dict.items()
        }
        self._ema_initialized = True

    def has_ema_weights(self) -> bool:
        """Check apakah EMA weights sudah di-initialize."""
        return self._ema_initialized and len(self.ema_weights) > 0

    # ========================================================================
    # STATE DICT
    # ========================================================================

    def state_dict(self) -> Dict:
        """Mengembalikan state optimizer dalam bentuk dictionary."""
        state = {
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
        
        # 🎯 Include EMA weights di state_dict
        if self._ema_initialized and self.ema_weights:
            state["ema_weights"] = {
                name: arr.copy()
                for name, arr in self.ema_weights.items()
            }
            state["ema_decay"] = self.ema_decay
            state["ema_initialized"] = True
        
        return state

    def load_state_dict(self, state: Dict) -> None:
        """Memuat state optimizer dari dictionary."""
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

        # 🎯 Load EMA weights jika ada
        if state.get("ema_initialized", False) and "ema_weights" in state:
            self.set_ema_weights(state["ema_weights"])
            if "ema_decay" in state:
                self.ema_decay = float(state["ema_decay"])

    # ========================================================================
    # CHECKPOINT
    # ========================================================================

    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Menyimpan checkpoint model + optimizer + EMA ke file .npz.

        Format keys:
        - step, global_step, last_epoch: metadata
        - learning_rate, base_learning_rate: LR state
        - model__{name}: model weights
        - adam_m__{name}: Adam first moment
        - adam_v__{name}: Adam second moment
        - ema__{name}: 🎯 EMA weights (BARU)
        - ema_decay, ema_initialized: EMA metadata
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

        # 🎯 EMA metadata
        payload["ema_decay"] = np.array(self.ema_decay, dtype=np.float32)
        payload["ema_initialized"] = np.array(self._ema_initialized, dtype=bool)

        for name, param in self.params.items():
            safe_name = name.replace(".", "__")

            payload[f"model__{safe_name}"] = param.data.astype(np.float32)
            payload[f"adam_m__{safe_name}"] = self.m[name].astype(np.float32)
            payload[f"adam_v__{safe_name}"] = self.v[name].astype(np.float32)
            
            # 🎯 Save EMA weights
            if self._ema_initialized and name in self.ema_weights:
                payload[f"ema__{safe_name}"] = self.ema_weights[name].astype(np.float32)

        np.savez_compressed(path, **payload)

        return path

    def load_checkpoint(self, path: Optional[Path] = None) -> Path:
        """
        Memuat checkpoint model + optimizer + EMA dari file .npz.

        Backward compatible:
        - Jika checkpoint lama (tanpa EMA), EMA akan di-initialize dari model weights
        - Jika EMA ada di checkpoint, langsung di-load
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
            self.base_learning_rate = self.learning_rate

        # Load model weights dan optimizer state
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

        # 🎯 Load EMA weights (dengan backward compatibility)
        ema_initialized_in_ckpt = bool(data.get("ema_initialized", False))
        
        if "ema_decay" in data:
            self.ema_decay = float(data["ema_decay"])
        
        if ema_initialized_in_ckpt:
            # Load EMA weights dari checkpoint
            self.ema_weights = {}
            ema_loaded_count = 0
            
            for name, param in self.params.items():
                safe_name = name.replace(".", "__")
                ema_key = f"ema__{safe_name}"
                
                if ema_key in data:
                    ema_arr = np.array(data[ema_key], dtype=np.float32)
                    
                    if ema_arr.shape == param.data.shape:
                        self.ema_weights[name] = ema_arr
                        ema_loaded_count += 1
            
            if ema_loaded_count > 0:
                self._ema_initialized = True
            else:
                # EMA flag ada tapi tidak ada weights - initialize dari model
                self._ema_initialized = False
                if self.use_ema:
                    self.initialize_ema()
        else:
            # Checkpoint lama tanpa EMA - initialize dari current model weights
            self._ema_initialized = False
            if self.use_ema:
                self.initialize_ema()

        return path

    def load_ema_weights(self, path: Optional[Path] = None) -> bool:
        """
        Hanya load EMA weights dari checkpoint (tanpa load model weights).
        
        Berguna untuk:
        - Apply EMA weights ke model yang sudah di-load
        - Compare EMA vs regular weights
        
        Return:
        - True jika EMA berhasil di-load, False jika tidak ada
        """
        if path is None:
            path = self.checkpoint_path
        else:
            path = Path(path)

        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        if not path.exists():
            return False

        try:
            data = np.load(path, allow_pickle=False)
            
            if not bool(data.get("ema_initialized", False)):
                return False
            
            ema_dict = {}
            for name, param in self.params.items():
                safe_name = name.replace(".", "__")
                ema_key = f"ema__{safe_name}"
                
                if ema_key in data:
                    ema_arr = np.array(data[ema_key], dtype=np.float32)
                    if ema_arr.shape == param.data.shape:
                        ema_dict[name] = ema_arr
            
            if ema_dict:
                self.set_ema_weights(ema_dict)
                return True
            
            return False
        
        except Exception:
            return False


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    import tempfile
    
    from config import (
        Config,
        ModelConfig,
        PathsConfig,
        SpecialTokensConfig,
        TrainingConfig,
    )
    from core.architecture import CustomTransformerLM

    print("=" * 70)
    print("pipeline/weight_manager.py SOTA Test Suite")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "output_models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        small_config = Config(
            paths=PathsConfig(
                output_models_dir=output_dir,
                checkpoint_path=output_dir / "training_checkpoint.npz",
            ),
            special_tokens=SpecialTokensConfig(),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=1,
                epochs=1,
                sequence_length=8,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
                use_ema=True,
                ema_decay=0.999,
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

        # --------------------------------------------------------------------
        # Test 1: Basic Step
        # --------------------------------------------------------------------
        print("\n[Test 1] Basic Optimizer Step")
        input_ids = np.array([[2, 5, 7, 10, 3]], dtype=np.int64)

        model.zero_grad()
        logits = model.forward(input_ids, training=True)
        grad_logits = np.ones_like(logits, dtype=np.float32)
        model.backward(grad_logits)

        info = weight_manager.step()

        print(f"  Parameter count : {model.parameter_count():,}")
        print(f"  Step info       : {info}")
        assert info["update_applied"] == 1.0
        assert info["step"] == 1

        # --------------------------------------------------------------------
        # Test 2: EMA Initialization
        # --------------------------------------------------------------------
        print("\n[Test 2] EMA Initialization")
        weight_manager.initialize_ema()
        print(f"  EMA initialized   : {weight_manager._ema_initialized}")
        print(f"  EMA weights count : {len(weight_manager.ema_weights)}")
        print(f"  EMA decay         : {weight_manager.ema_decay}")
        assert weight_manager.has_ema_weights()

        # --------------------------------------------------------------------
        # Test 3: EMA Update
        # --------------------------------------------------------------------
        print("\n[Test 3] EMA Update")
        # Simpan initial EMA untuk comparison
        initial_ema = {k: v.copy() for k, v in weight_manager.ema_weights.items()}
        
        # Do a few more steps to change model weights
        for _ in range(3):
            model.zero_grad()
            logits = model.forward(input_ids, training=True)
            grad_logits = np.random.randn(*logits.shape).astype(np.float32) * 0.1
            model.backward(grad_logits)
            weight_manager.step()
            weight_manager.update_ema()
        
        # Check EMA has changed
        changed_count = 0
        for name in initial_ema:
            if not np.allclose(initial_ema[name], weight_manager.ema_weights[name]):
                changed_count += 1
        
        print(f"  Steps done       : {weight_manager.step_count}")
        print(f"  EMA tensors changed: {changed_count}/{len(initial_ema)}")
        assert changed_count > 0, "EMA weights should change after updates"

        # --------------------------------------------------------------------
        # Test 4: Checkpoint Save (with EMA)
        # --------------------------------------------------------------------
        print("\n[Test 4] Checkpoint Save (with EMA)")
        ckpt_path = weight_manager.save_checkpoint()
        print(f"  Checkpoint saved : {ckpt_path}")
        print(f"  File size        : {ckpt_path.stat().st_size:,} bytes")
        
        # Verify EMA keys in checkpoint
        data = np.load(ckpt_path, allow_pickle=False)
        ema_keys = [k for k in data.files if k.startswith("ema__")]
        print(f"  EMA keys in ckpt : {len(ema_keys)}")
        print(f"  EMA initialized  : {bool(data['ema_initialized'])}")
        print(f"  EMA decay        : {float(data['ema_decay'])}")

        # --------------------------------------------------------------------
        # Test 5: Checkpoint Load (with EMA)
        # --------------------------------------------------------------------
        print("\n[Test 5] Checkpoint Load (with EMA)")
        # Save EMA state before load
        ema_before = {k: v.copy() for k, v in weight_manager.ema_weights.items()}
        
        # Create fresh model and manager
        model2 = CustomTransformerLM(small_config)
        wm2 = WeightManager(model2, small_config)
        
        # Load checkpoint
        loaded_path = wm2.load_checkpoint(ckpt_path)
        print(f"  Loaded from      : {loaded_path}")
        print(f"  Step count       : {wm2.step_count}")
        print(f"  Global step      : {wm2.global_step}")
        print(f"  EMA initialized  : {wm2._ema_initialized}")
        print(f"  EMA weights count: {len(wm2.ema_weights)}")
        
        # Verify EMA weights match
        ema_match = True
        for name in ema_before:
            if name in wm2.ema_weights:
                if not np.allclose(ema_before[name], wm2.ema_weights[name]):
                    ema_match = False
                    break
        
        print(f"  EMA weights match: {ema_match}")
        assert ema_match, "EMA weights should match after load"

        # --------------------------------------------------------------------
        # Test 6: Apply EMA to Model
        # --------------------------------------------------------------------
        print("\n[Test 6] Apply EMA to Model")
        # Save current model weights
        model_weights_before = {
            name: param.data.copy()
            for name, param in wm2.params.items()
        }
        
        # Apply EMA
        success = wm2.apply_ema_to_model()
        print(f"  Apply success    : {success}")
        
        # Verify model weights changed to EMA
        changed = 0
        for name in model_weights_before:
            if not np.allclose(model_weights_before[name], wm2.params[name].data):
                changed += 1
        
        print(f"  Tensors changed  : {changed}/{len(model_weights_before)}")
        assert changed > 0, "Model weights should change after EMA apply"

        # --------------------------------------------------------------------
        # Test 7: Backward Compatibility (old checkpoint without EMA)
        # --------------------------------------------------------------------
        print("\n[Test 7] Backward Compatibility (old checkpoint without EMA)")
        # Create a fake old checkpoint (without ema__ keys)
        old_ckpt_path = output_dir / "old_checkpoint.npz"
        
        old_payload = {
            "step": np.array(10, dtype=np.int64),
            "global_step": np.array(10, dtype=np.int64),
            "last_epoch": np.array(2, dtype=np.int64),
            "learning_rate": np.array(3e-4, dtype=np.float32),
            "base_learning_rate": np.array(3e-4, dtype=np.float32),
            # No ema_initialized, no ema_decay, no ema__ keys
        }
        
        for name, param in wm2.params.items():
            safe_name = name.replace(".", "__")
            old_payload[f"model__{safe_name}"] = param.data
            old_payload[f"adam_m__{safe_name}"] = wm2.m[name]
            old_payload[f"adam_v__{safe_name}"] = wm2.v[name]
        
        np.savez_compressed(old_ckpt_path, **old_payload)
        
        # Load with new manager
        model3 = CustomTransformerLM(small_config)
        wm3 = WeightManager(model3, small_config)
        wm3.load_checkpoint(old_ckpt_path)
        
        print(f"  Old ckpt loaded  : True")
        print(f"  Step count       : {wm3.step_count}")
        print(f"  EMA auto-init    : {wm3._ema_initialized}")
        print(f"  EMA weights count: {len(wm3.ema_weights)}")
        assert wm3._ema_initialized, "EMA should auto-initialize from model weights"
        assert len(wm3.ema_weights) == len(wm3.params), "All params should have EMA"

        # --------------------------------------------------------------------
        # Test 8: Learning Rate Scheduling
        # --------------------------------------------------------------------
        print("\n[Test 8] Learning Rate Scheduling")
        model4 = CustomTransformerLM(small_config)
        wm4 = WeightManager(model4, small_config)
        
        lrs = []
        for step in range(20):
            lr = wm4.update_learning_rate()
            lrs.append(lr)
            wm4.step_count += 1
        
        print(f"  Warmup LR progression:")
        print(f"    Step 0  : {lrs[0]:.6e}")
        print(f"    Step 50 : {lrs[5] if len(lrs) > 5 else 'N/A'}")
        print(f"    Step 100: {lrs[10] if len(lrs) > 10 else 'N/A'}")
        print(f"    Step 150: {lrs[15] if len(lrs) > 15 else 'N/A'}")

        # --------------------------------------------------------------------
        # Test 9: Gradient Clipping
        # --------------------------------------------------------------------
        print("\n[Test 9] Gradient Clipping")
        model5 = CustomTransformerLM(small_config)
        wm5 = WeightManager(model5, small_config)
        
        # Set large gradients
        for param in wm5.params.values():
            param.grad = np.random.randn(*param.shape).astype(np.float32) * 100.0
        
        grad_norm_before = float(np.sqrt(sum(
            float(np.sum(p.grad * p.grad)) for p in wm5.params.values()
        )))
        
        grad_norm_returned = wm5.clip_gradients()
        
        grad_norm_after = float(np.sqrt(sum(
            float(np.sum(p.grad * p.grad)) for p in wm5.params.values()
        )))
        
        print(f"  Grad norm before : {grad_norm_before:.4f}")
        print(f"  Grad norm returned: {grad_norm_returned:.4f}")
        print(f"  Grad norm after  : {grad_norm_after:.4f}")
        print(f"  Max allowed      : {wm5.grad_clip_norm}")
        assert grad_norm_after <= wm5.grad_clip_norm + 1e-5

        # --------------------------------------------------------------------
        # Test 10: State Dict Round-trip
        # --------------------------------------------------------------------
        print("\n[Test 10] State Dict Round-trip")
        state = weight_manager.state_dict()
        
        model6 = CustomTransformerLM(small_config)
        wm6 = WeightManager(model6, small_config)
        wm6.load_state_dict(state)
        
        print(f"  Step preserved     : {wm6.step_count == weight_manager.step_count}")
        print(f"  Global step match  : {wm6.global_step == weight_manager.global_step}")
        print(f"  LR match           : {abs(wm6.learning_rate - weight_manager.learning_rate) < 1e-9}")
        print(f"  EMA match          : {wm6.has_ema_weights()}")
        
        assert wm6.step_count == weight_manager.step_count
        assert wm6.global_step == weight_manager.global_step

        print("\n" + "=" * 70)
        print("✅ pipeline/weight_manager.py (SOTA + EMA) ALL TESTS PASSED")
        print("=" * 70)
