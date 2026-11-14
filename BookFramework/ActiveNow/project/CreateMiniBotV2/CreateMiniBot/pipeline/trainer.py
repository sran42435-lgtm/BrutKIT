# pipeline/trainer.py
#
# Trainer / Training Loop Engine untuk RAG-based Training Pipeline.
#
# VERSI v2.3 (FINAL):
# Arsitektur: Single-Head + RAG + Constrained Generation
#
# 🎯 SELF-CORRECTING TRAINING PHILOSOPHY:
# "Model berlatih keras, memperbaiki setiap kesalahan, menjadi jauh lebih baik"
#
# Teknik SOTA yang diimplementasi:
# 1. Cosine Annealing LR + Warmup (GPT-3/LLaMA style)
# 2. Label Smoothing (cegah overconfidence)
# 3. Focal Loss (fokus pada hard examples)
# 4. Token Importance Weighting (answer tokens 3x penting)
# 5. Adaptive Gradient Clipping - AGC (NFNet)
# 6. Loss Spike Detection + Auto-Recovery
# 7. Curriculum Learning (easy → hard)
# 8. Self-Correction Loop (DeepMind style)
# 9. EMA Weights (Exponential Moving Average)
# 10. Dynamic Sample Difficulty Scoring
#
# Fitur v2.2:
# - Flexible Checkpoint System (per-steps + per-epochs)
# - Export otak model BARENG checkpoint (anti mati-lampu!)
# - Auto-Stop pintar (loss + accuracy threshold)
# - Rolling metrics untuk convergence detection
#
# 🆕 Fitur v2.3:
# - Metrics tracking (loss, PPL, accuracy) di setiap checkpoint
# - get_resume_info() untuk main.py auto-continue
# - Metrics persisten di checkpoint untuk resume + /status
#
# Keterhubungan:
# - config.py              : hyperparameter training + RAG
# - core/tokenizer.py      : encode QA samples
# - core/architecture.py   : model (single LM head + cross-attention)
# - core/retriever.py      : build index setelah training (optional)
# - core/model_exporter.py : export otak model
# - pipeline/evaluator.py  : menghitung loss dan gradient
# - pipeline/weight_manager.py : memperbarui bobot + track metrics

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
        "pipeline/trainer.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config

# Import retriever (optional)
try:
    from core.retriever import BM25Retriever, load_corpus_from_directory
    _RETRIEVER_AVAILABLE = True
except ImportError:
    _RETRIEVER_AVAILABLE = False

# Import ModelExporter untuk export otak model
try:
    from core.model_exporter import ModelExporter
    _EXPORTER_AVAILABLE = True
except ImportError:
    _EXPORTER_AVAILABLE = False


# ============================================================================
# SOTA TRAINING CONSTANTS
# ============================================================================

LOSS_SPIKE_SIGMA = 3.0
LOSS_SPIKE_WINDOW = 50
AUTO_STOP_ROLLING_WINDOW = 100

DEFAULT_EMA_DECAY = 0.999
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_ANSWER_TOKEN_WEIGHT = 3.0
DEFAULT_AGC_FACTOR = 0.01
DEFAULT_SELF_CORRECTION_MAX_ITER = 2
DEFAULT_HARD_SAMPLE_LOSS_THRESHOLD = 2.0


class Trainer:
    """
    Mesin latihan SOTA untuk RAG (Retrieval-Augmented Generation).

    🎯 Self-Correcting Training Philosophy:
    "Model berlatih keras, memperbaiki setiap kesalahan, menjadi jauh lebih baik"

    Fitur v2.2:
    - Flexible Checkpoint (per-steps + per-epochs)
    - Export otak model bareng checkpoint (anti mati-lampu!)
    - Auto-Stop pintar berdasarkan loss + accuracy threshold
    - Rolling metrics untuk convergence detection
    
    🆕 Fitur v2.3:
    - Metrics tracking (loss, PPL, accuracy) di setiap checkpoint
    - Resume info untuk auto-continue di main.py
    """

    LOG_EVERY_STEPS = 10
    NO_CONTEXT_ANSWER = "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut."

    def __init__(
        self,
        model,
        tokenizer,
        evaluator,
        weight_manager,
        config: Config = CONFIG,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.weight_manager = weight_manager
        self.config = config

        self.datasets_dir = Path(config.paths.datasets_dir)
        self.qa_dataset_path = Path(config.paths.qa_dataset_path)
        self.batch_size = int(config.training.batch_size)
        self.epochs = int(config.training.epochs)
        self.sequence_length = int(config.training.sequence_length)
        self.seed = int(config.training.seed)

        # Early stopping
        self.early_stopping_patience = int(config.training.early_stopping_patience)

        # Inline training
        self.inline_training_steps = int(config.training.inline_training_steps)

        # ================================================================
        # 🎯 FLEXIBLE CHECKPOINT SYSTEM
        # ================================================================
        self.save_checkpoint_every_n_steps = int(
            getattr(config.training, 'save_checkpoint_every_n_steps', 500)
        )
        self.save_checkpoint_every_n_epochs = int(
            getattr(config.training, 'save_checkpoint_every_n_epochs', 1)
        )
        
        # Export otak model bareng checkpoint
        self.export_brain_on_checkpoint = bool(
            getattr(config.training, 'export_brain_on_checkpoint', True)
        )

        # ================================================================
        # 🎯 AUTO-STOP CONFIGURATION
        # ================================================================
        self.auto_stop_enabled = bool(
            getattr(config.training, 'auto_stop_enabled', True)
        )
        self.auto_stop_loss_threshold = float(
            getattr(config.training, 'auto_stop_loss_threshold', 0.8)
        )
        self.auto_stop_accuracy_threshold = float(
            getattr(config.training, 'auto_stop_accuracy_threshold', 0.75)
        )
        self.auto_stop_min_steps = int(
            getattr(config.training, 'auto_stop_min_steps', 1000)
        )
        self.auto_stop_patience_steps = int(
            getattr(config.training, 'auto_stop_patience_steps', 100)
        )
        self.auto_stop_check_interval = int(
            getattr(config.training, 'auto_stop_check_interval', 50)
        )

        # ================================================================
        # RAG-SPECIFIC PARAMETERS
        # ================================================================
        self.context_dropout_rate = float(
            getattr(config.training, 'context_dropout_rate', 0.2)
        )
        self.context_perturbation_rate = float(
            getattr(config.training, 'context_perturbation_rate', 0.3)
        )
        self.progressive_stages = int(
            getattr(config.training, 'progressive_stages', 3)
        )
        self.max_context_paragraphs = int(
            getattr(config.training, 'max_context_paragraphs', 3)
        )

        # ================================================================
        # 🎯 SOTA TRAINING PARAMETERS
        # ================================================================
        self.use_cosine_schedule = bool(
            getattr(config.training, 'use_cosine_schedule', True)
        )
        self.base_learning_rate = float(config.training.learning_rate)
        self.lr_warmup_steps = int(config.training.lr_warmup_steps)
        self.lr_min = float(getattr(config.training, 'lr_min', 1e-6))
        
        self.label_smoothing = float(
            getattr(config.training, 'label_smoothing', DEFAULT_LABEL_SMOOTHING)
        )
        
        self.use_focal_loss = bool(
            getattr(config.training, 'use_focal_loss', True)
        )
        self.focal_gamma = float(
            getattr(config.training, 'focal_gamma', DEFAULT_FOCAL_GAMMA)
        )
        
        self.use_token_weighting = bool(
            getattr(config.training, 'use_token_weighting', True)
        )
        self.answer_token_weight = float(
            getattr(config.training, 'answer_token_weight', DEFAULT_ANSWER_TOKEN_WEIGHT)
        )
        
        self.use_agc = bool(getattr(config.training, 'use_agc', True))
        self.agc_factor = float(getattr(config.training, 'agc_factor', DEFAULT_AGC_FACTOR))
        
        self.use_loss_spike_detection = bool(
            getattr(config.training, 'use_loss_spike_detection', True)
        )
        
        self.use_curriculum = bool(getattr(config.training, 'use_curriculum', True))
        
        self.use_self_correction = bool(
            getattr(config.training, 'use_self_correction', True)
        )
        self.self_correction_max_iter = int(
            getattr(config.training, 'self_correction_max_iter', DEFAULT_SELF_CORRECTION_MAX_ITER)
        )
        self.hard_sample_loss_threshold = float(
            getattr(config.training, 'hard_sample_loss_threshold', DEFAULT_HARD_SAMPLE_LOSS_THRESHOLD)
        )
        
        self.use_ema = bool(getattr(config.training, 'use_ema', True))
        self.ema_decay = float(getattr(config.training, 'ema_decay', DEFAULT_EMA_DECAY))

        # ================================================================
        # STATE
        # ================================================================
        self.start_epoch = 0
        self.start_global_step = 0

        self.pad_id = int(config.special_tokens.pad_id)
        self.bos_id = int(config.special_tokens.bos_id)
        self.eos_id = int(config.special_tokens.eos_id)
        self.endoftext_id = int(config.special_tokens.endoftext_id)

        self.qa_samples: List[Dict] = []

        self._rng = np.random.default_rng(self.seed)
        self._py_rng = random.Random(self.seed)

        self._retriever: Optional['BM25Retriever'] = None
        
        # ================================================================
        # 🎯 SOTA TRACKING STATE
        # ================================================================
        self._loss_history: deque = deque(maxlen=LOSS_SPIKE_WINDOW)
        self._ema_weights: Optional[Dict[str, np.ndarray]] = None
        self._sample_difficulty: Dict[int, float] = {}
        self._self_correction_stats = {
            "total_attempts": 0,
            "corrections_applied": 0,
            "spikes_detected": 0,
            "hard_samples_retrained": 0,
        }
        
        # Rolling metrics untuk auto-stop
        self._rolling_losses: deque = deque(maxlen=AUTO_STOP_ROLLING_WINDOW)
        self._rolling_correct: int = 0
        self._rolling_total: int = 0
        self._consecutive_target_met: int = 0
        self._auto_stop_triggered: bool = False

        if self.batch_size <= 0:
            raise ValueError("batch_size harus lebih besar dari 0.")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length harus lebih besar dari 0.")

    # ========================================================================
    # 🎯 SOTA: LEARNING RATE SCHEDULING
    # ========================================================================

    def _get_learning_rate(self, step: int, total_steps: int) -> float:
        """Cosine Annealing dengan Linear Warmup."""
        if not self.use_cosine_schedule:
            return self.base_learning_rate
        
        if step < self.lr_warmup_steps:
            progress = step / max(1, self.lr_warmup_steps)
            return self.lr_min + (self.base_learning_rate - self.lr_min) * progress
        
        cosine_steps = max(1, total_steps - self.lr_warmup_steps)
        progress = (step - self.lr_warmup_steps) / cosine_steps
        progress = min(1.0, progress)
        
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.base_learning_rate - self.lr_min) * cosine_factor

    def _apply_lr_to_weight_manager(self, step: int, total_steps: int) -> None:
        """Update learning rate di weight manager."""
        new_lr = self._get_learning_rate(step, total_steps)
        if hasattr(self.weight_manager, 'learning_rate'):
            self.weight_manager.learning_rate = new_lr

    # ========================================================================
    # 🎯 SOTA: LABEL SMOOTHING
    # ========================================================================

    def _apply_label_smoothing(self, targets: np.ndarray, vocab_size: int) -> np.ndarray:
        """Apply label smoothing ke target distribution."""
        if self.label_smoothing <= 0.0:
            return None
        
        B, T = targets.shape
        eps = self.label_smoothing
        smooth_targets = np.full((B, T, vocab_size), eps / vocab_size, dtype=np.float32)
        
        for b in range(B):
            for t in range(T):
                token_id = targets[b, t]
                if 0 <= token_id < vocab_size and token_id != self.pad_id:
                    smooth_targets[b, t, token_id] = 1.0 - eps + (eps / vocab_size)
        
        return smooth_targets

    # ========================================================================
    # 🎯 SOTA: FOCAL LOSS
    # ========================================================================

    def _compute_focal_weights(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute focal loss weights: (1 - p_t)^gamma"""
        if not self.use_focal_loss or self.focal_gamma <= 0.0:
            return None
        
        B, T, V = logits.shape
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        p_t = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                token_id = targets[b, t]
                if 0 <= token_id < V:
                    p_t[b, t] = probs[b, t, token_id]
        
        focal_weights = np.power(1.0 - p_t, self.focal_gamma)
        return focal_weights.astype(np.float32)

    # ========================================================================
    # 🎯 SOTA: TOKEN IMPORTANCE WEIGHTING
    # ========================================================================

    def _compute_token_weights(self, input_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
        """Compute per-token weights."""
        if not self.use_token_weighting:
            return None
        
        B, T = target_ids.shape
        weights = np.ones((B, T), dtype=np.float32)
        weights[target_ids == self.pad_id] = 0.0
        
        for b in range(B):
            input_seq = input_ids[b]
            eos_positions = np.where(input_seq == self.eos_id)[0]
            if len(eos_positions) > 0:
                answer_start = eos_positions[0] + 1
                if answer_start < T:
                    weights[b, answer_start:] = self.answer_token_weight
        
        return weights

    # ========================================================================
    # 🎯 SOTA: AGC
    # ========================================================================

    def _apply_agc(self) -> None:
        """Adaptive Gradient Clipping dari NFNet."""
        if not self.use_agc:
            return
        
        for name, param in self.model.parameters().items():
            if param.grad is None:
                continue
            
            grad_norm = np.linalg.norm(param.grad)
            weight_norm = np.linalg.norm(param.data)
            
            if grad_norm < 1e-8 or weight_norm < 1e-8:
                continue
            
            ratio = grad_norm / weight_norm
            if ratio > self.agc_factor:
                scale_factor = (self.agc_factor * weight_norm) / grad_norm
                param.grad = param.grad * scale_factor

    # ========================================================================
    # 🎯 SOTA: LOSS SPIKE DETECTION
    # ========================================================================

    def _detect_loss_spike(self, current_loss: float) -> bool:
        """Deteksi loss spike."""
        if not self.use_loss_spike_detection:
            return False
        
        if len(self._loss_history) < 10:
            self._loss_history.append(current_loss)
            return False
        
        history_array = np.array(list(self._loss_history))
        mean_loss = np.mean(history_array)
        std_loss = np.std(history_array)
        
        spike_threshold = mean_loss + LOSS_SPIKE_SIGMA * std_loss
        is_spike = current_loss > spike_threshold
        
        self._loss_history.append(current_loss)
        
        if is_spike:
            self._self_correction_stats["spikes_detected"] += 1
        
        return is_spike

    def _recover_from_spike(self) -> None:
        """Recovery dari loss spike."""
        if hasattr(self.weight_manager, 'learning_rate'):
            current_lr = self.weight_manager.learning_rate
            self.weight_manager.learning_rate = current_lr * 0.5

    # ========================================================================
    # 🎯 SOTA: CURRICULUM LEARNING
    # ========================================================================

    def _compute_sample_difficulty(self, sample: Dict) -> float:
        """Estimate difficulty."""
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        
        context_len = len(context.split())
        question_len = len(question.split())
        answer_len = len(answer.split())
        
        diff_length = min(1.0, (context_len + question_len) / 300.0)
        diff_answer = min(1.0, answer_len / 50.0)
        
        return 0.6 * diff_length + 0.4 * diff_answer

    def _sort_samples_by_curriculum(
        self, samples: List[Dict], epoch: int, total_epochs: int,
    ) -> List[Dict]:
        """Sort samples easy → hard."""
        if not self.use_curriculum:
            return samples
        
        sample_with_diff = [(s, self._compute_sample_difficulty(s)) for s in samples]
        sample_with_diff.sort(key=lambda x: x[1])
        
        progress = (epoch + 1) / total_epochs
        
        if progress < 0.3:
            threshold = 0.4
        elif progress < 0.7:
            threshold = 0.7
        else:
            threshold = 1.0
        
        filtered = [s for s, d in sample_with_diff if d <= threshold]
        self._py_rng.shuffle(filtered)
        return filtered

    # ========================================================================
    # 🎯 SOTA: SELF-CORRECTION
    # ========================================================================

    def _self_correct_batch(
        self, inputs: np.ndarray, targets: np.ndarray, initial_loss: float,
    ) -> Tuple[float, int]:
        """Self-correction loop."""
        if not self.use_self_correction:
            return initial_loss, 0
        
        current_loss = initial_loss
        corrections = 0
        
        for iteration in range(self.self_correction_max_iter):
            if current_loss < self.hard_sample_loss_threshold:
                break
            
            self.model.zero_grad()
            logits = self.model.forward(inputs, training=True)
            loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(logits, targets)
            
            num_valid = int(metrics["num_valid_tokens"])
            if num_valid == 0:
                break
            
            self.model.backward(grad_logits)
            self._apply_agc()
            
            original_lr = getattr(self.weight_manager, 'learning_rate', None)
            if original_lr is not None:
                self.weight_manager.learning_rate = original_lr * 0.5
            
            self.weight_manager.step(zero_grad=True)
            
            if original_lr is not None:
                self.weight_manager.learning_rate = original_lr
            
            current_loss = float(loss)
            corrections += 1
        
        self._self_correction_stats["corrections_applied"] += corrections
        return current_loss, corrections

    # ========================================================================
    # 🎯 SOTA: EMA WEIGHTS
    # ========================================================================

    def _update_ema(self) -> None:
        """Update EMA weights."""
        if not self.use_ema:
            return
        
        current_weights = {
            name: param.data.copy()
            for name, param in self.model.parameters().items()
        }
        
        if self._ema_weights is None:
            self._ema_weights = current_weights
        else:
            for name in current_weights:
                if name in self._ema_weights:
                    self._ema_weights[name] = (
                        self.ema_decay * self._ema_weights[name]
                        + (1.0 - self.ema_decay) * current_weights[name]
                    )
                else:
                    self._ema_weights[name] = current_weights[name]

    def apply_ema_to_model(self) -> None:
        """Apply EMA weights ke model."""
        if self._ema_weights is None:
            return
        
        for name, param in self.model.parameters().items():
            if name in self._ema_weights:
                param.data = self._ema_weights[name].copy()

    # ========================================================================
    # 🎯 AUTO-STOP LOGIC
    # ========================================================================

    def _update_rolling_metrics(self, loss: float, num_correct: int, num_tokens: int) -> None:
        """Update rolling metrics untuk auto-stop check."""
        self._rolling_losses.append(loss)
        self._rolling_correct += num_correct
        self._rolling_total += num_tokens

    def _check_auto_stop(self, current_step: int, verbose: bool = True) -> bool:
        """
        Cek apakah training harus stop berdasarkan loss + accuracy threshold.
        
        Returns:
            True jika training harus stop
        """
        if not self.auto_stop_enabled:
            return False
        
        if current_step < self.auto_stop_min_steps:
            return False
        
        if len(self._rolling_losses) < AUTO_STOP_ROLLING_WINDOW:
            return False
        
        avg_loss = float(np.mean(list(self._rolling_losses)))
        
        if self._rolling_total > 0:
            rolling_accuracy = self._rolling_correct / self._rolling_total
        else:
            rolling_accuracy = 0.0
        
        loss_ok = avg_loss < self.auto_stop_loss_threshold
        acc_ok = rolling_accuracy > self.auto_stop_accuracy_threshold
        
        if loss_ok and acc_ok:
            self._consecutive_target_met += 1
        else:
            self._consecutive_target_met = 0
        
        if self._consecutive_target_met >= self.auto_stop_patience_steps:
            if verbose:
                print()
                print("=" * 70)
                print("🎯 AUTO-STOP TRIGGERED!")
                print("=" * 70)
                print(f"  Step                  : {current_step:,}")
                print(f"  Average loss          : {avg_loss:.6f} (< {self.auto_stop_loss_threshold})")
                print(f"  Rolling accuracy      : {rolling_accuracy:.4f} (> {self.auto_stop_accuracy_threshold})")
                print(f"  Consecutive meets     : {self._consecutive_target_met}")
                print(f"  Target sudah tercapai! Training akan dihentikan.")
                print("=" * 70)
            self._auto_stop_triggered = True
            return True
        
        if verbose and current_step % self.auto_stop_check_interval == 0:
            if loss_ok or acc_ok:
                partial_status = []
                if loss_ok:
                    partial_status.append(f"✅loss={avg_loss:.4f}")
                if acc_ok:
                    partial_status.append(f"✅acc={rolling_accuracy:.4f}")
                print(f"  [Auto-stop] Progress: {' '.join(partial_status)} | consecutive={self._consecutive_target_met}/{self.auto_stop_patience_steps}")
        
        return False

    def reset_rolling_metrics(self) -> None:
        """Reset rolling metrics."""
        self._rolling_losses.clear()
        self._rolling_correct = 0
        self._rolling_total = 0

    # ========================================================================
    # 🎯 CHECKPOINT + BRAIN EXPORT + METRICS HELPER (v2.3 UPDATED)
    # ========================================================================

    def _save_checkpoint(
        self, 
        verbose: bool = True, 
        context: str = "",
        export_brain: Optional[bool] = None,
        current_metrics: Optional[Dict] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Save checkpoint + export otak model + track metrics.
        
        Args:
            verbose: print progress
            context: context message
            export_brain: override config
            current_metrics: dict dengan loss, perplexity, accuracy (v2.3 NEW)
        """
        # 🆕 v2.3: Update metrics di weight_manager sebelum save
        if current_metrics and hasattr(self.weight_manager, 'update_metrics'):
            self.weight_manager.update_metrics(
                loss=current_metrics.get("loss", 0.0),
                perplexity=current_metrics.get("perplexity", 1.0),
                accuracy=current_metrics.get("accuracy", 0.0),
            )
        
        # 1. Save training session (cepat)
        ckpt_path = self.weight_manager.save_checkpoint()
        
        # 2. Export otak model (lambat tapi aman)
        should_export_brain = (
            export_brain if export_brain is not None 
            else self.export_brain_on_checkpoint
        )
        
        brain_path = None
        if should_export_brain and _EXPORTER_AVAILABLE:
            try:
                exporter = ModelExporter(self.config)
                brain_path = exporter.save(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    include_vocab=True,
                    include_rag_assets=False,
                    verbose=False,
                )
            except Exception as e:
                if verbose:
                    print(f"  [Checkpoint] ⚠️ Failed to export brain: {e}")
        
        # 3. Print info
        if verbose:
            if context:
                print(f"  [Checkpoint] 💾 {context}:")
            else:
                print(f"  [Checkpoint] 💾 Saved:")
            
            print(f"    📝 Training session : {ckpt_path.name}")
            
            # 🆕 v2.3: Print metrics
            if current_metrics:
                print(f"    📊 Metrics          : loss={current_metrics.get('loss', 0):.4f} ppl={current_metrics.get('perplexity', 1):.2f} acc={current_metrics.get('accuracy', 0):.4f}")
            
            if brain_path:
                try:
                    brain_size_mb = brain_path.stat().st_size / 1024 / 1024
                    print(f"    🧠 Brain model      : {brain_path.name} ({brain_size_mb:.1f} MB)")
                except Exception:
                    print(f"    🧠 Brain model      : {brain_path.name}")
            elif should_export_brain and not _EXPORTER_AVAILABLE:
                print(f"    🧠 Brain export     : skipped (ModelExporter not available)")
        
        return ckpt_path, brain_path

    # ========================================================================
    # 🆕 RESUME INFO (untuk main.py auto-continue)
    # ========================================================================

    def get_resume_info(self) -> Dict:
        """
        Get informasi resume dari checkpoint.
        
        Dipakai main.py untuk auto-continue training setelah --resume.
        
        Return:
            Dict dengan start_epoch, global_step, last_metrics, remaining info
        """
        last_metrics = {}
        if hasattr(self.weight_manager, 'get_last_metrics'):
            last_metrics = self.weight_manager.get_last_metrics()
        
        remaining_epochs = max(0, self.epochs - self.start_epoch)
        
        return {
            "start_epoch": self.start_epoch,
            "global_step": self.start_global_step,
            "last_metrics": last_metrics,
            "total_epochs_configured": self.epochs,
            "remaining_epochs": remaining_epochs,
            "has_more_to_train": remaining_epochs > 0,
            "auto_stop_triggered": self._auto_stop_triggered,
        }

    # ========================================================================
    # RESUME STATE
    # ========================================================================

    def set_resume_state(self, start_epoch: int = 0, global_step: int = 0) -> None:
        self.start_epoch = max(0, int(start_epoch))
        self.start_global_step = max(0, int(global_step))

    # ========================================================================
    # DATASET LOADING
    # ========================================================================

    def load_qa_dataset(self, path: Optional[Path] = None, verbose: bool = True) -> int:
        """Load QA dataset."""
        if path is None:
            merged_train = Path(self.config.paths.merged_qa_train_path)
            if merged_train.exists():
                path = merged_train
                if verbose:
                    print(f"[Trainer] 📚 Menggunakan merged dataset: {path}")
            else:
                path = self.qa_dataset_path
        else:
            path = Path(path)

        if not path.exists():
            if verbose:
                print(f"[Trainer] ⚠️ QA dataset tidak ditemukan: {path}")
                print(f"[Trainer] Jalankan: python scripts/prepare_datasets.py")
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            samples = data.get("samples", [])
            valid_samples = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                if "question" not in sample:
                    continue
                valid_samples.append(sample)

            self.qa_samples = valid_samples

            if verbose:
                metadata = data.get("metadata", {})
                print(f"[Trainer] ✅ Loaded {len(valid_samples)} QA samples dari {path.name}")
                print(f"[Trainer]    Total samples: {metadata.get('total', len(valid_samples))}")

                stats = metadata.get("stats", {})
                if stats:
                    print(f"[Trainer]    By source: {dict(stats.get('by_source', {}))}")
                    print(f"[Trainer]    By language: {dict(stats.get('by_language', {}))}")

            return len(valid_samples)

        except Exception as exc:
            if verbose:
                print(f"[Trainer] ❌ Gagal load QA dataset: {exc}")
            return 0

    def get_qa_samples(self) -> List[Dict]:
        """Return list QA samples."""
        return list(self.qa_samples)

    # ========================================================================
    # CONTEXT MANIPULATION
    # ========================================================================

    def _split_sentences(self, text: str) -> List[str]:
        """Split teks jadi kalimat."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []
        return sentences

    def _shuffle_sentences(self, text: str) -> str:
        """Shuffle kalimat."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text
        self._py_rng.shuffle(sentences)
        return ' '.join(sentences)

    def _apply_context_dropout(self, context: str) -> Tuple[str, bool]:
        """Randomly hapus konteks."""
        if not context:
            return context, True

        if self._py_rng.random() < self.context_dropout_rate:
            return "", True

        return context, False

    def _apply_context_perturbation(self, context: str) -> str:
        """Randomly perturbasi konteks."""
        if not context:
            return context

        if self._py_rng.random() >= self.context_perturbation_rate:
            return context

        perturbation_type = self._py_rng.choice(["shuffle", "truncate"])

        if perturbation_type == "shuffle":
            return self._shuffle_sentences(context)

        elif perturbation_type == "truncate":
            sentences = self._split_sentences(context)
            if len(sentences) <= 1:
                mid = len(context) // 2
                return context[:mid].strip()
            ratio = self._py_rng.uniform(0.5, 0.8)
            n_keep = max(1, int(len(sentences) * ratio))
            return ' '.join(sentences[:n_keep])

        return context

    # ========================================================================
    # SAMPLE PREPARATION
    # ========================================================================

    def _prepare_sample(self, sample: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Siapkan satu QA sample."""
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        is_impossible = sample.get("is_impossible", False)

        if not question:
            return None

        if is_impossible or not answer:
            answer = self.NO_CONTEXT_ANSWER

        context, is_dropped = self._apply_context_dropout(context)
        if is_dropped and context == "":
            answer = self.NO_CONTEXT_ANSWER

        if context:
            context = self._apply_context_perturbation(context)

        try:
            if context:
                input_ids, target_ids = self.tokenizer.split_qa_for_training(
                    context=context, question=question, answer=answer,
                )
            else:
                full_ids = self.tokenizer.encode_qa_no_context(
                    question=question, answer=answer,
                    add_bos=True, add_eos=True,
                )
                input_text = f"Pertanyaan: {question} Jawaban:"
                input_ids = self.tokenizer.encode(input_text, add_bos=True, add_eos=False)
                target_ids = full_ids[len(input_ids):]
        except Exception:
            return None

        if not input_ids or not target_ids:
            return None

        full_sequence = input_ids + target_ids
        chunk_size = self.sequence_length + 1

        if len(full_sequence) > chunk_size:
            full_sequence = full_sequence[:chunk_size]
        elif len(full_sequence) < chunk_size:
            padding_needed = chunk_size - len(full_sequence)
            full_sequence = full_sequence + [self.pad_id] * padding_needed

        return np.array(full_sequence, dtype=np.int64)

    def _prepare_sample_progressive(
        self, sample: Dict, stage: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Siapkan sample dengan progressive difficulty."""
        if stage == 1:
            orig_dropout = self.context_dropout_rate
            orig_perturb = self.context_perturbation_rate
            self.context_dropout_rate = 0.0
            self.context_perturbation_rate = 0.0
            result = self._prepare_sample(sample)
            self.context_dropout_rate = orig_dropout
            self.context_perturbation_rate = orig_perturb
            return result

        elif stage == 2:
            orig_dropout = self.context_dropout_rate
            orig_perturb = self.context_perturbation_rate
            self.context_dropout_rate = min(0.1, self.context_dropout_rate)
            self.context_perturbation_rate = min(0.15, self.context_perturbation_rate)
            result = self._prepare_sample(sample)
            self.context_dropout_rate = orig_dropout
            self.context_perturbation_rate = orig_perturb
            return result

        else:
            return self._prepare_sample(sample)

    # ========================================================================
    # BATCHING dengan CURRICULUM
    # ========================================================================

    def _get_current_stage(self, epoch: int, total_epochs: int) -> int:
        """Hitung current progressive stage."""
        if self.progressive_stages <= 0 or total_epochs <= 0:
            return 1

        stage_size = total_epochs / self.progressive_stages
        stage = int(epoch / stage_size) + 1
        return min(stage, self.progressive_stages)

    def iter_batches(
        self,
        epoch: int = 0,
        total_epochs: Optional[int] = None,
        shuffle: bool = True,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """Yield batches dengan curriculum learning."""
        if total_epochs is None:
            total_epochs = self.epochs

        if not self.qa_samples:
            return

        samples = self._sort_samples_by_curriculum(
            self.qa_samples, epoch, total_epochs
        )

        stage = self._get_current_stage(epoch, total_epochs)

        prepared: List[np.ndarray] = []
        for sample in samples:
            result = self._prepare_sample_progressive(sample, stage)
            if result is not None:
                prepared.append(result)

        if not prepared:
            return

        for i in range(0, len(prepared), self.batch_size):
            chunk = prepared[i : i + self.batch_size]
            if not chunk:
                continue

            stacked = np.stack(chunk, axis=0)
            inputs = stacked[:, :-1]
            targets = stacked[:, 1:]

            yield inputs, targets

    # ========================================================================
    # 🎯 SOTA: SINGLE TRAINING STEP
    # ========================================================================

    def _train_step_sota(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        step: int,
        total_steps: int,
    ) -> Tuple[float, dict]:
        """Satu training step dengan semua SOTA techniques."""
        self._apply_lr_to_weight_manager(step, total_steps)

        self.model.zero_grad()
        logits = self.model.forward(inputs, training=True)

        loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(
            logits, targets,
        )
        
        num_valid_tokens = int(metrics["num_valid_tokens"])
        if num_valid_tokens == 0:
            return 0.0, metrics
        
        if self.label_smoothing > 0.0:
            smooth_factor = self.label_smoothing
            vocab_size = logits.shape[-1]
            uniform_grad = -1.0 / vocab_size
            grad_logits = (1.0 - smooth_factor) * grad_logits + smooth_factor * uniform_grad
        
        if self.use_focal_loss:
            focal_weights = self._compute_focal_weights(logits, targets)
            if focal_weights is not None:
                focal_weights = focal_weights[:, :, np.newaxis]
                grad_logits = grad_logits * focal_weights
        
        if self.use_token_weighting:
            token_weights = self._compute_token_weights(inputs, targets)
            if token_weights is not None:
                token_weights = token_weights[:, :, np.newaxis]
                grad_logits = grad_logits * token_weights

        self.model.backward(grad_logits)
        self._apply_agc()

        is_spike = self._detect_loss_spike(float(loss))
        if is_spike:
            self._recover_from_spike()

        final_loss = float(loss)
        if self.use_self_correction and final_loss > self.hard_sample_loss_threshold:
            final_loss, corrections = self._self_correct_batch(
                inputs, targets, final_loss
            )
            self._self_correction_stats["total_attempts"] += 1
            if corrections > 0:
                self._self_correction_stats["hard_samples_retrained"] += 1

        step_info = self.weight_manager.step(zero_grad=True)
        self._update_ema()

        metrics["loss"] = final_loss
        metrics["is_spike"] = is_spike
        metrics["learning_rate"] = step_info.get("learning_rate", self.base_learning_rate)

        return final_loss, metrics

    # ========================================================================
    # PUBLIC API: FULL TRAINING (SOTA-ENHANCED)
    # ========================================================================

    def run_epochs(
        self,
        max_steps: Optional[int] = None,
        save_checkpoints: bool = False,
        verbose: bool = True,
        start_epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> List[dict]:
        """Main training entry point dengan SOTA techniques + auto-stop."""
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        if not self.qa_samples:
            n_samples = self.load_qa_dataset(verbose=verbose)
            if n_samples == 0:
                if verbose:
                    print("[Trainer] ❌ Tidak ada data untuk dilatih.")
                return []

        if start_epoch is None:
            start_epoch = self.start_epoch
        if global_step is None:
            global_step = self.start_global_step

        total_epochs = self.epochs

        if start_epoch >= total_epochs:
            new_total = start_epoch + total_epochs
            if verbose:
                print(
                    f"[Trainer] Melanjutkan dengan {total_epochs} epoch baru "
                    f"(epoch {start_epoch + 1} - {new_total})."
                )
            total_epochs = new_total

        steps_per_epoch = len(self.qa_samples) // self.batch_size
        total_steps = total_epochs * steps_per_epoch

        if verbose:
            print("=" * 70)
            print("🎯 SOTA RAG TRAINING (v2.3 FINAL)")
            print("=" * 70)
            print(f"  QA samples          : {len(self.qa_samples):,}")
            print(f"  Total epochs        : {total_epochs}")
            print(f"  Batch size          : {self.batch_size}")
            print(f"  Sequence length     : {self.sequence_length}")
            print(f"  Steps per epoch     : {steps_per_epoch:,}")
            print(f"  Total steps         : {total_steps:,}")
            print(f"  Progressive stages  : {self.progressive_stages}")
            print(f"  Context dropout     : {self.context_dropout_rate:.1%}")
            print(f"  Context perturb     : {self.context_perturbation_rate:.1%}")
            print()
            print("  🎯 SOTA Techniques Aktif:")
            print(f"    Cosine LR + Warmup     : {self.use_cosine_schedule}")
            print(f"    Label Smoothing        : {self.label_smoothing}")
            print(f"    Focal Loss (γ={self.focal_gamma})   : {self.use_focal_loss}")
            print(f"    Token Weighting (×{self.answer_token_weight}) : {self.use_token_weighting}")
            print(f"    AGC (factor={self.agc_factor})    : {self.use_agc}")
            print(f"    Loss Spike Detection   : {self.use_loss_spike_detection}")
            print(f"    Curriculum Learning    : {self.use_curriculum}")
            print(f"    Self-Correction        : {self.use_self_correction}")
            print(f"    EMA Weights (decay={self.ema_decay}): {self.use_ema}")
            print()
            print("  🎯 Flexible Checkpoint:")
            if self.save_checkpoint_every_n_steps > 0:
                print(f"    Per steps              : every {self.save_checkpoint_every_n_steps} steps")
            else:
                print(f"    Per steps              : disabled")
            if self.save_checkpoint_every_n_epochs > 0:
                print(f"    Per epochs             : every {self.save_checkpoint_every_n_epochs} epoch(s)")
            else:
                print(f"    Per epochs             : disabled")
            print(f"    Export brain           : {self.export_brain_on_checkpoint}")
            print()
            print("  🎯 Auto-Stop:")
            if self.auto_stop_enabled:
                print(f"    Status                 : enabled")
                print(f"    Loss target            : < {self.auto_stop_loss_threshold}")
                print(f"    Accuracy target        : > {self.auto_stop_accuracy_threshold}")
                print(f"    Min steps              : {self.auto_stop_min_steps}")
                print(f"    Patience               : {self.auto_stop_patience_steps} checks")
                print(f"    Check interval         : every {self.auto_stop_check_interval} steps")
            else:
                print(f"    Status                 : disabled")
            print("=" * 70)

        history: List[dict] = []
        stop_training = False
        best_loss = float("inf")
        patience_counter = 0
        last_epoch_report = None  # 🆕 v2.3: Track last report untuk final checkpoint

        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_correct = 0
            steps_done = 0
            epoch_spikes = 0

            self._rng = np.random.default_rng(self.seed + epoch)
            self._py_rng = random.Random(self.seed + epoch)

            stage = self._get_current_stage(epoch, total_epochs)

            if verbose:
                print(
                    f"\n[Trainer] Epoch {epoch + 1}/{total_epochs} dimulai "
                    f"(Stage {stage}/{self.progressive_stages})"
                )

            batch_iterator = self.iter_batches(
                epoch=epoch,
                total_epochs=total_epochs,
                shuffle=True,
            )

            for inputs, targets in batch_iterator:
                if max_steps is not None and global_step >= int(max_steps):
                    stop_training = True
                    break

                loss, metrics = self._train_step_sota(
                    inputs, targets, global_step, total_steps
                )

                num_valid_tokens = int(metrics.get("num_valid_tokens", 0))

                if num_valid_tokens > 0:
                    epoch_loss += loss * num_valid_tokens
                    epoch_tokens += num_valid_tokens
                    num_correct_step = int(metrics.get("num_correct", 0))
                    epoch_correct += num_correct_step
                    
                    self._update_rolling_metrics(loss, num_correct_step, num_valid_tokens)
                    
                    if metrics.get("is_spike", False):
                        epoch_spikes += 1

                    if verbose and steps_done % self.LOG_EVERY_STEPS == 0:
                        spike_marker = " ⚠️SPIKE" if metrics.get("is_spike", False) else ""
                        print(
                            f"  [Stage {stage}] step={steps_done} "
                            f"global_step={global_step} "
                            f"loss={loss:.6f} "
                            f"ppl={metrics.get('perplexity', 1.0):.4f} "
                            f"acc={metrics.get('accuracy', 0.0):.4f} "
                            f"lr={metrics.get('learning_rate', 0.0):.2e}"
                            f"{spike_marker}"
                        )

                    # 🆕 v2.3: CHECKPOINT PER STEPS + METRICS + EXPORT BRAIN
                    if (
                        save_checkpoints
                        and self.save_checkpoint_every_n_steps > 0
                        and global_step > 0
                        and global_step % self.save_checkpoint_every_n_steps == 0
                    ):
                        self._save_checkpoint(
                            verbose=verbose,
                            context=f"Step {global_step}",
                            current_metrics={
                                "loss": float(loss),
                                "perplexity": float(metrics.get('perplexity', 1.0)),
                                "accuracy": float(metrics.get('accuracy', 0.0)),
                            },
                        )
                    
                    # AUTO-STOP CHECK
                    if (
                        global_step > 0 
                        and global_step % self.auto_stop_check_interval == 0
                    ):
                        if self._check_auto_stop(global_step, verbose=verbose):
                            if save_checkpoints:
                                if verbose:
                                    print("\n[Trainer] 💾 Saving final checkpoint + brain...")
                                self._save_checkpoint(
                                    verbose=verbose,
                                    context="🎯 AUTO-STOP FINAL",
                                    export_brain=True,
                                    current_metrics={
                                        "loss": float(loss),
                                        "perplexity": float(metrics.get('perplexity', 1.0)),
                                        "accuracy": float(metrics.get('accuracy', 0.0)),
                                    },
                                )
                            stop_training = True
                            break
                else:
                    self.model.zero_grad()

                steps_done += 1
                global_step += 1

            if steps_done == 0:
                if verbose:
                    print("[Trainer] ⚠️ Tidak ada batch yang diproses.")

                epoch_report = {
                    "epoch": epoch + 1,
                    "stage": stage,
                    "loss": 0.0,
                    "perplexity": 1.0,
                    "accuracy": 0.0,
                    "num_valid_tokens": 0,
                    "num_correct": 0,
                    "steps": 0,
                    "spikes": 0,
                    "corrections": 0,
                }
            else:
                avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0.0
                accuracy = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0.0
                perplexity = float(np.exp(avg_loss)) if avg_loss < 100 else float("inf")

                epoch_report = {
                    "epoch": epoch + 1,
                    "stage": stage,
                    "loss": float(avg_loss),
                    "perplexity": float(perplexity),
                    "accuracy": float(accuracy),
                    "num_valid_tokens": int(epoch_tokens),
                    "num_correct": int(epoch_correct),
                    "steps": int(steps_done),
                    "spikes": epoch_spikes,
                    "corrections": 0,
                    "learning_rate": self._get_learning_rate(global_step, total_steps),
                    "auto_stop_triggered": self._auto_stop_triggered,
                }
                
                last_epoch_report = epoch_report  # 🆕 v2.3: Track untuk final checkpoint

                if verbose:
                    print(
                        f"[Trainer] Epoch {epoch + 1} selesai (Stage {stage}) | "
                        f"loss={avg_loss:.6f} | "
                        f"ppl={perplexity:.4f} | "
                        f"acc={accuracy:.4f} | "
                        f"tokens={epoch_tokens:,} | "
                        f"spikes={epoch_spikes}"
                    )

            history.append(epoch_report)

            self.weight_manager.last_epoch = epoch + 1
            self.weight_manager.global_step = global_step

            # 🆕 v2.3: CHECKPOINT PER EPOCHS + METRICS + EXPORT BRAIN
            if save_checkpoints:
                should_save_epoch = (
                    self.save_checkpoint_every_n_epochs > 0
                    and (epoch + 1) % self.save_checkpoint_every_n_epochs == 0
                )
                
                if should_save_epoch:
                    self._save_checkpoint(
                        verbose=verbose,
                        context=f"Epoch {epoch + 1}",
                        current_metrics={
                            "loss": epoch_report.get("loss", 0.0),
                            "perplexity": epoch_report.get("perplexity", 1.0),
                            "accuracy": epoch_report.get("accuracy", 0.0),
                        },
                    )

            # Early stopping
            if self.early_stopping_patience > 0 and steps_done > 0:
                current_loss = epoch_report.get("loss", float("inf"))

                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(
                            f"[Trainer] Early stopping: loss tidak turun selama "
                            f"{self.early_stopping_patience} epoch."
                        )
                    stop_training = True

            if stop_training:
                if verbose and not self._auto_stop_triggered:
                    print("[Trainer] Training dihentikan.")
                break

        # FINAL: Apply EMA weights
        if self.use_ema and self._ema_weights is not None and not self._auto_stop_triggered:
            if verbose:
                print("\n[Trainer] 🎯 Applying EMA weights ke model...")
            self.apply_ema_to_model()
            
            if save_checkpoints:
                # 🆕 v2.3: Pass last_epoch_report metrics ke final checkpoint
                final_metrics = None
                if last_epoch_report:
                    final_metrics = {
                        "loss": last_epoch_report.get("loss", 0.0),
                        "perplexity": last_epoch_report.get("perplexity", 1.0),
                        "accuracy": last_epoch_report.get("accuracy", 0.0),
                    }
                
                self._save_checkpoint(
                    verbose=verbose,
                    context="Final (EMA applied)",
                    current_metrics=final_metrics,
                )

        # FINAL: Print SOTA stats
        if verbose:
            print("\n" + "=" * 70)
            print("📊 SOTA TRAINING STATISTICS")
            print("=" * 70)
            stats = self._self_correction_stats
            print(f"  Total spikes detected    : {stats['spikes_detected']}")
            print(f"  Self-correction attempts : {stats['total_attempts']}")
            print(f"  Corrections applied      : {stats['corrections_applied']}")
            print(f"  Hard samples retrained   : {stats['hard_samples_retrained']}")
            
            if self._auto_stop_triggered:
                print(f"  Auto-stop triggered      : ✅ YES (target tercapai)")
            else:
                print(f"  Auto-stop triggered      : ❌ NO")
            
            print("=" * 70)

        if _RETRIEVER_AVAILABLE:
            self._build_retriever_index(verbose=verbose)

        return history

    # ========================================================================
    # INLINE TRAINING
    # ========================================================================

    def train_steps(
        self,
        steps: int,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        """Inline training untuk N steps."""
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        if not self.qa_samples:
            self.load_qa_dataset(verbose=verbose)

        if not self.qa_samples:
            if verbose:
                print("[Trainer] ❌ Tidak ada data untuk dilatih.")
            return []

        steps = max(1, int(steps))
        current_global_step = getattr(self.weight_manager, "global_step", 0)
        total_steps = current_global_step + steps

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        steps_done = 0
        epochs_traversed = 0
        last_checkpoint_step = current_global_step
        last_loss = 0.0
        last_perplexity = 1.0
        last_accuracy = 0.0

        if verbose:
            print(f"[Trainer] Inline SOTA training: target {steps} step")

        while steps_done < steps:
            self._rng = np.random.default_rng(self.seed + current_global_step + steps_done)
            self._py_rng = random.Random(self.seed + current_global_step + steps_done)

            batch_iterator = self.iter_batches(
                epoch=epochs_traversed,
                total_epochs=max(1, steps // max(1, len(self.qa_samples) // self.batch_size)),
                shuffle=True,
            )
            batches_this_epoch = 0

            for inputs, targets in batch_iterator:
                if steps_done >= steps:
                    break

                loss, metrics = self._train_step_sota(
                    inputs, targets,
                    current_global_step + steps_done,
                    total_steps,
                )

                num_valid_tokens = int(metrics.get("num_valid_tokens", 0))

                if num_valid_tokens > 0:
                    total_loss += loss * num_valid_tokens
                    total_tokens += num_valid_tokens
                    num_correct_step = int(metrics.get("num_correct", 0))
                    total_correct += num_correct_step
                    
                    # Track last metrics untuk checkpoint
                    last_loss = float(loss)
                    last_perplexity = float(metrics.get('perplexity', 1.0))
                    last_accuracy = float(metrics.get('accuracy', 0.0))
                    
                    self._update_rolling_metrics(loss, num_correct_step, num_valid_tokens)

                    if verbose:
                        print(
                            f"[Trainer] step={steps_done} "
                            f"loss={loss:.6f} "
                            f"ppl={metrics.get('perplexity', 1.0):.4f} "
                            f"acc={metrics.get('accuracy', 0.0):.4f} "
                            f"lr={metrics.get('learning_rate', 0.0):.2e}"
                        )
                    
                    current_step = current_global_step + steps_done
                    
                    # 🆕 v2.3: CHECKPOINT PER STEPS + METRICS
                    if (
                        save_checkpoint
                        and self.save_checkpoint_every_n_steps > 0
                        and current_step > 0
                        and current_step % self.save_checkpoint_every_n_steps == 0
                        and current_step != last_checkpoint_step
                    ):
                        self._save_checkpoint(
                            verbose=verbose,
                            context=f"Inline Step {current_step}",
                            current_metrics={
                                "loss": last_loss,
                                "perplexity": last_perplexity,
                                "accuracy": last_accuracy,
                            },
                        )
                        last_checkpoint_step = current_step
                    
                    # AUTO-STOP CHECK
                    if (
                        current_step > 0
                        and current_step % self.auto_stop_check_interval == 0
                    ):
                        if self._check_auto_stop(current_step, verbose=verbose):
                            if save_checkpoint:
                                self._save_checkpoint(
                                    verbose=verbose,
                                    context="🎯 INLINE AUTO-STOP FINAL",
                                    export_brain=True,
                                    current_metrics={
                                        "loss": last_loss,
                                        "perplexity": last_perplexity,
                                        "accuracy": last_accuracy,
                                    },
                                )
                            break
                else:
                    self.model.zero_grad()

                steps_done += 1
                batches_this_epoch += 1

                self.weight_manager.global_step = current_global_step + steps_done

            epochs_traversed += 1

            if batches_this_epoch == 0:
                break
            
            if self._auto_stop_triggered:
                break

        if steps_done == 0:
            if verbose:
                print("[Trainer] Tidak ada batch untuk training.")
            return []

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = float(np.exp(avg_loss)) if avg_loss < 100 else float("inf")

        report = {
            "epoch": getattr(self.weight_manager, "last_epoch", 0),
            "loss": float(avg_loss),
            "perplexity": float(perplexity),
            "accuracy": float(accuracy),
            "num_valid_tokens": int(total_tokens),
            "num_correct": int(total_correct),
            "steps": int(steps_done),
            "epochs_traversed": int(epochs_traversed),
            "auto_stop_triggered": self._auto_stop_triggered,
        }

        if verbose:
            print(
                f"[Trainer] Inline SOTA training selesai | "
                f"steps={steps_done} | "
                f"loss={avg_loss:.6f} | "
                f"ppl={perplexity:.4f} | "
                f"acc={accuracy:.4f}"
            )
            if self._auto_stop_triggered:
                print(f"[Trainer] 🎯 AUTO-STOP: target tercapai!")

        # 🆕 v2.3: Final checkpoint dengan aggregated metrics
        if save_checkpoint and not self._auto_stop_triggered:
            self._save_checkpoint(
                verbose=verbose,
                context="Inline Training Final",
                current_metrics={
                    "loss": float(avg_loss),
                    "perplexity": float(perplexity),
                    "accuracy": float(accuracy),
                },
            )

        if _RETRIEVER_AVAILABLE:
            self._build_retriever_index(verbose=verbose)

        return [report]

    def train_epochs_inline(
        self,
        epochs: int,
        min_steps: int = 0,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        """Inline training untuk N epochs."""
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        if not self.qa_samples:
            self.load_qa_dataset(verbose=verbose)

        if not self.qa_samples:
            if verbose:
                print("[Trainer] ❌ Tidak ada data untuk dilatih.")
            return []

        epochs = max(1, int(epochs))
        min_steps = max(0, int(min_steps))

        current_global_step = getattr(self.weight_manager, "global_step", 0)
        current_epoch = getattr(self.weight_manager, "last_epoch", 0)

        if verbose:
            print(
                f"[Trainer] Inline SOTA training: "
                f"target {epochs} epoch"
                + (f", minimal {min_steps} step" if min_steps > 0 else "")
            )

        self.set_resume_state(
            start_epoch=current_epoch,
            global_step=current_global_step,
        )

        original_epochs = self.epochs
        self.epochs = current_epoch + epochs

        history = self.run_epochs(
            max_steps=None,
            save_checkpoints=save_checkpoint,
            verbose=verbose,
        )

        self.epochs = original_epochs

        return history

    # ========================================================================
    # RETRIEVER INDEX BUILDING
    # ========================================================================

    def _build_retriever_index(self, verbose: bool = True) -> None:
        """Bangun BM25 retrieval index."""
        if not _RETRIEVER_AVAILABLE:
            if verbose:
                print("[Trainer] core/retriever.py tidak tersedia.")
            return

        if not self.datasets_dir.exists():
            if verbose:
                print(f"[Trainer] Datasets dir tidak ditemukan.")
            return

        if verbose:
            print()
            print("=" * 70)
            print("BUILDING RETRIEVAL INDEX")
            print("=" * 70)

        try:
            max_para_len = int(getattr(self.config.retrieval, 'max_paragraph_length', 500))
            corpus = load_corpus_from_directory(
                self.datasets_dir,
                max_paragraph_length=max_para_len,
                verbose=verbose,
            )

            if not corpus:
                if verbose:
                    print("[Trainer] ⚠️ Corpus kosong.")
                return

            self._retriever = BM25Retriever(self.config)
            self._retriever.build_index(corpus, verbose=verbose)

            index_path = Path(self.config.paths.retrieval_index_path)
            self._retriever.save_index(index_path)

        except Exception as exc:
            if verbose:
                print(f"[Trainer] ❌ Gagal build retriever index: {exc}")

    def get_retriever(self) -> Optional['BM25Retriever']:
        """Return retriever instance."""
        return self._retriever

    def get_sota_stats(self) -> Dict:
        """Return SOTA training statistics."""
        return dict(self._self_correction_stats)
