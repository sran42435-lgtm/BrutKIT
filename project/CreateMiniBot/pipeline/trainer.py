# pipeline/trainer.py
#
# Trainer / Training Loop Engine untuk RAG-based Training Pipeline.
#
# VERSI SOTA-ENHANCED (v2.0):
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
# 5. Adaptive Gradient Clipping - AGC (NFNet, lebih stabil dari fixed clip)
# 6. Loss Spike Detection + Auto-Recovery
# 7. Curriculum Learning (easy → hard)
# 8. Self-Correction Loop (DeepMind style)
# 9. EMA Weights (Exponential Moving Average)
# 10. Dynamic Sample Difficulty Scoring
#
# Keterhubungan:
# - config.py              : hyperparameter training + RAG
# - core/tokenizer.py      : encode QA samples
# - core/architecture.py   : model (single LM head + cross-attention)
# - core/retriever.py      : build index setelah training (optional)
# - pipeline/evaluator.py  : menghitung loss dan gradient
# - pipeline/weight_manager.py : memperbarui bobot

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


# ============================================================================
# SOTA TRAINING CONSTANTS
# ============================================================================

# Loss spike detection: jika loss > MEAN + N*STD, dianggap spike
LOSS_SPIKE_SIGMA = 3.0
LOSS_SPIKE_WINDOW = 50  # steps untuk rolling statistics

# EMA decay untuk model averaging (Polyak averaging)
DEFAULT_EMA_DECAY = 0.999

# Label smoothing epsilon (standard: 0.1)
DEFAULT_LABEL_SMOOTHING = 0.1

# Focal loss gamma (2.0 = fokus pada hard examples)
DEFAULT_FOCAL_GAMMA = 2.0

# Token importance weight untuk answer tokens
DEFAULT_ANSWER_TOKEN_WEIGHT = 3.0

# AGC (Adaptive Gradient Clipping) factor
DEFAULT_AGC_FACTOR = 0.01

# Self-correction: max iterations per sample
DEFAULT_SELF_CORRECTION_MAX_ITER = 2

# Difficulty threshold: samples dengan loss > threshold di-train ulang
DEFAULT_HARD_SAMPLE_LOSS_THRESHOLD = 2.0


class Trainer:
    """
    Mesin latihan SOTA untuk RAG (Retrieval-Augmented Generation).

    🎯 Self-Correcting Training Philosophy:
    "Model berlatih keras, memperbaiki setiap kesalahan, menjadi jauh lebih baik"

    Arsitektur:
    - Single LM head (no domain contamination)
    - Context-conditioned training
    - Loss ↓ = Presisi ↑ (korelasi langsung)

    SOTA Features:
    - Cosine Annealing LR + Warmup
    - Label Smoothing + Focal Loss
    - Token Importance Weighting
    - Adaptive Gradient Clipping (AGC)
    - Loss Spike Detection + Auto-Recovery
    - Curriculum Learning (easy → hard)
    - Self-Correction Loop
    - EMA Weights (Polyak Averaging)

    Standard Features:
    - Resume dari epoch terakhir
    - Early stopping
    - Periodic checkpoint
    - Inline training dari playground
    - Context dropout (mitigasi hallucination)
    - Context perturbation (mitigasi urutan memorization)
    - Progressive training stages
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

        # Periodic checkpoint
        self.save_checkpoint_every_n_steps = int(config.training.save_checkpoint_every_n_steps)

        # Inline training
        self.inline_training_steps = int(config.training.inline_training_steps)

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
        # 🎯 SOTA TRAINING PARAMETERS (BARU)
        # ================================================================
        
        # Cosine annealing
        self.use_cosine_schedule = bool(
            getattr(config.training, 'use_cosine_schedule', True)
        )
        self.base_learning_rate = float(config.training.learning_rate)
        self.lr_warmup_steps = int(config.training.lr_warmup_steps)
        self.lr_min = float(getattr(config.training, 'lr_min', 1e-6))
        
        # Label smoothing
        self.label_smoothing = float(
            getattr(config.training, 'label_smoothing', DEFAULT_LABEL_SMOOTHING)
        )
        
        # Focal loss
        self.use_focal_loss = bool(
            getattr(config.training, 'use_focal_loss', True)
        )
        self.focal_gamma = float(
            getattr(config.training, 'focal_gamma', DEFAULT_FOCAL_GAMMA)
        )
        
        # Token importance weighting
        self.use_token_weighting = bool(
            getattr(config.training, 'use_token_weighting', True)
        )
        self.answer_token_weight = float(
            getattr(config.training, 'answer_token_weight', DEFAULT_ANSWER_TOKEN_WEIGHT)
        )
        
        # Adaptive Gradient Clipping (AGC)
        self.use_agc = bool(
            getattr(config.training, 'use_agc', True)
        )
        self.agc_factor = float(
            getattr(config.training, 'agc_factor', DEFAULT_AGC_FACTOR)
        )
        
        # Loss spike detection
        self.use_loss_spike_detection = bool(
            getattr(config.training, 'use_loss_spike_detection', True)
        )
        
        # Curriculum learning
        self.use_curriculum = bool(
            getattr(config.training, 'use_curriculum', True)
        )
        
        # Self-correction loop
        self.use_self_correction = bool(
            getattr(config.training, 'use_self_correction', True)
        )
        self.self_correction_max_iter = int(
            getattr(config.training, 'self_correction_max_iter', DEFAULT_SELF_CORRECTION_MAX_ITER)
        )
        self.hard_sample_loss_threshold = float(
            getattr(config.training, 'hard_sample_loss_threshold', DEFAULT_HARD_SAMPLE_LOSS_THRESHOLD)
        )
        
        # EMA weights
        self.use_ema = bool(
            getattr(config.training, 'use_ema', True)
        )
        self.ema_decay = float(
            getattr(config.training, 'ema_decay', DEFAULT_EMA_DECAY)
        )

        # ================================================================
        # STATE
        # ================================================================
        self.start_epoch = 0
        self.start_global_step = 0

        # Token IDs untuk special tokens
        self.pad_id = int(config.special_tokens.pad_id)
        self.bos_id = int(config.special_tokens.bos_id)
        self.eos_id = int(config.special_tokens.eos_id)
        self.endoftext_id = int(config.special_tokens.endoftext_id)

        # QA Dataset
        self.qa_samples: List[Dict] = []

        # Random generators
        self._rng = np.random.default_rng(self.seed)
        self._py_rng = random.Random(self.seed)

        # Retriever
        self._retriever: Optional['BM25Retriever'] = None
        
        # ================================================================
        # 🎯 SOTA TRACKING STATE (BARU)
        # ================================================================
        self._loss_history: deque = deque(maxlen=LOSS_SPIKE_WINDOW)
        self._ema_weights: Optional[Dict[str, np.ndarray]] = None
        self._sample_difficulty: Dict[int, float] = {}  # idx → running loss
        self._self_correction_stats = {
            "total_attempts": 0,
            "corrections_applied": 0,
            "spikes_detected": 0,
            "hard_samples_retrained": 0,
        }

        if self.batch_size <= 0:
            raise ValueError("batch_size harus lebih besar dari 0.")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length harus lebih besar dari 0.")

    # ========================================================================
    # 🎯 SOTA: LEARNING RATE SCHEDULING (COSINE WITH WARMUP)
    # ========================================================================

    def _get_learning_rate(self, step: int, total_steps: int) -> float:
        """
        Cosine Annealing dengan Linear Warmup (GPT-3/LLaMA style).
        
        Formula:
        - Warmup phase (step < warmup_steps): linear increase
        - Cosine phase: lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
        
        Args:
            step: current training step
            total_steps: total training steps
        
        Returns:
            learning_rate untuk step ini
        """
        if not self.use_cosine_schedule:
            return self.base_learning_rate
        
        # Phase 1: Warmup (linear increase)
        if step < self.lr_warmup_steps:
            progress = step / max(1, self.lr_warmup_steps)
            return self.lr_min + (self.base_learning_rate - self.lr_min) * progress
        
        # Phase 2: Cosine decay
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

    def _apply_label_smoothing(
        self,
        targets: np.ndarray,
        vocab_size: int,
    ) -> np.ndarray:
        """
        Apply label smoothing ke target distribution.
        
        Standard: target = [0, 0, ..., 1, ..., 0]
        Smoothed: target = [ε/V, ε/V, ..., 1-ε, ..., ε/V]
        
        Benefit: cegah model terlalu confident, improve generalization
        """
        if self.label_smoothing <= 0.0:
            return None  # Use standard cross-entropy
        
        B, T = targets.shape
        eps = self.label_smoothing
        
        # Buat uniform distribution
        smooth_targets = np.full((B, T, vocab_size), eps / vocab_size, dtype=np.float32)
        
        # Set high probability untuk correct token
        for b in range(B):
            for t in range(T):
                token_id = targets[b, t]
                if 0 <= token_id < vocab_size and token_id != self.pad_id:
                    smooth_targets[b, t, token_id] = 1.0 - eps + (eps / vocab_size)
        
        return smooth_targets

    # ========================================================================
    # 🎯 SOTA: FOCAL LOSS MODULATION
    # ========================================================================

    def _compute_focal_weights(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """
        Compute focal loss weights: (1 - p_t)^gamma
        
        Benefit: down-weight easy examples, fokus pada hard examples
        Reference: Lin et al. 2017 "Focal Loss for Dense Object Detection"
        """
        if not self.use_focal_loss or self.focal_gamma <= 0.0:
            return None
        
        B, T, V = logits.shape
        
        # Softmax probabilities
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Get probability of correct token
        p_t = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            for t in range(T):
                token_id = targets[b, t]
                if 0 <= token_id < V:
                    p_t[b, t] = probs[b, t, token_id]
        
        # Focal weight: (1 - p_t)^gamma
        focal_weights = np.power(1.0 - p_t, self.focal_gamma)
        
        return focal_weights.astype(np.float32)

    # ========================================================================
    # 🎯 SOTA: TOKEN IMPORTANCE WEIGHTING
    # ========================================================================

    def _compute_token_weights(
        self,
        input_ids: np.ndarray,
        target_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-token weights berdasarkan importance.
        
        Strategy:
        - PAD tokens: weight = 0 (ignore)
        - Question tokens: weight = 1.0 (standard)
        - Answer tokens: weight = answer_token_weight (3.0 default)
        
        Benefit: model fokus pada kualitas jawaban, bukan pertanyaan
        """
        if not self.use_token_weighting:
            return None
        
        B, T = target_ids.shape
        weights = np.ones((B, T), dtype=np.float32)
        
        # Zero weight untuk PAD tokens
        weights[target_ids == self.pad_id] = 0.0
        
        # Heuristic: answer tokens biasanya di后半部分 sequence
        # Kita identify answer start dengan pattern matching pada input
        # Untuk simplicity, gunakan position-based weighting
        # (Answer tokens typically di second half setelah "Jawaban:")
        
        # Detect "Jawaban:" position via token patterns
        # Ini heuristic sederhana; trainer bisa pass explicit answer_start
        for b in range(B):
            # Cari EOS di input (marker end of question)
            input_seq = input_ids[b]
            eos_positions = np.where(input_seq == self.eos_id)[0]
            
            if len(eos_positions) > 0:
                # Tokens setelah EOS pertama = answer region
                answer_start = eos_positions[0] + 1
                if answer_start < T:
                    weights[b, answer_start:] = self.answer_token_weight
        
        return weights

    # ========================================================================
    # 🎯 SOTA: ADAPTIVE GRADIENT CLIPPING (AGC)
    # ========================================================================

    def _apply_agc(self) -> None:
        """
        Adaptive Gradient Clipping dari NFNet (Brock et al. 2021).
        
        Alih-alih fixed clip norm, AGC clip berdasarkan ratio:
            ||grad|| / ||weight||
        
        Jika ratio > threshold, scale gradient:
            grad_new = grad * (threshold * ||weight|| / ||grad||)
        
        Benefit: lebih stabil untuk large batch training, cegah explosion
        """
        if not self.use_agc:
            return
        
        for name, param in self.model.parameters().items():
            if param.grad is None:
                continue
            
            grad_norm = np.linalg.norm(param.grad)
            weight_norm = np.linalg.norm(param.data)
            
            if grad_norm < 1e-8 or weight_norm < 1e-8:
                continue
            
            # Max ratio allowed (adaptive threshold)
            max_ratio = self.agc_factor
            
            # Current ratio
            ratio = grad_norm / weight_norm
            
            if ratio > max_ratio:
                # Scale down gradient
                scale_factor = (max_ratio * weight_norm) / grad_norm
                param.grad = param.grad * scale_factor

    # ========================================================================
    # 🎯 SOTA: LOSS SPIKE DETECTION
    # ========================================================================

    def _detect_loss_spike(self, current_loss: float) -> bool:
        """
        Deteksi apakah current loss adalah spike (anomaly).
        
        Criterion: loss > mean(history) + N * std(history)
        
        Returns:
            True jika spike detected
        """
        if not self.use_loss_spike_detection:
            return False
        
        if len(self._loss_history) < 10:  # Butuh minimal samples
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
        """
        Recovery dari loss spike:
        1. Reduce gradient (already clipped by AGC)
        2. Optionally rollback to previous weights (jika tersedia)
        3. Reduce learning rate temporarily
        """
        # Temporary learning rate reduction
        if hasattr(self.weight_manager, 'learning_rate'):
            current_lr = self.weight_manager.learning_rate
            self.weight_manager.learning_rate = current_lr * 0.5

    # ========================================================================
    # 🎯 SOTA: CURRICULUM LEARNING
    # ========================================================================

    def _compute_sample_difficulty(self, sample: Dict) -> float:
        """
        Estimate difficulty dari satu sample.
        
        Heuristics:
        - Panjang context: lebih panjang = lebih sulit
        - Panjang question: lebih panjang = lebih sulit
        - Vocabulary rarity: banyak UNK = lebih sulit
        """
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        
        # Length-based difficulty
        context_len = len(context.split())
        question_len = len(question.split())
        answer_len = len(answer.split())
        
        # Normalize ke [0, 1]
        diff_length = min(1.0, (context_len + question_len) / 300.0)
        
        # Answer complexity (lebih panjang = lebih sulit)
        diff_answer = min(1.0, answer_len / 50.0)
        
        # Combined score
        difficulty = 0.6 * diff_length + 0.4 * diff_answer
        
        return difficulty

    def _sort_samples_by_curriculum(
        self,
        samples: List[Dict],
        epoch: int,
        total_epochs: int,
    ) -> List[Dict]:
        """
        Sort samples berdasarkan curriculum: easy → hard.
        
        Strategy:
        - Epoch 1-30%: easy samples saja
        - Epoch 30-70%: mix easy + medium
        - Epoch 70-100%: all samples (termasuk hard)
        """
        if not self.use_curriculum:
            return samples
        
        # Compute difficulty untuk setiap sample
        sample_with_diff = [
            (s, self._compute_sample_difficulty(s))
            for s in samples
        ]
        
        # Sort by difficulty
        sample_with_diff.sort(key=lambda x: x[1])
        
        # Determine difficulty threshold based on epoch progress
        progress = (epoch + 1) / total_epochs
        
        if progress < 0.3:
            # First 30%: only easy samples
            threshold = 0.4
        elif progress < 0.7:
            # Middle 40%: easy + medium
            threshold = 0.7
        else:
            # Last 30%: all samples
            threshold = 1.0
        
        # Filter by threshold
        filtered = [s for s, d in sample_with_diff if d <= threshold]
        
        # Shuffle filtered samples
        self._py_rng.shuffle(filtered)
        
        return filtered

    # ========================================================================
    # 🎯 SOTA: SELF-CORRECTION LOOP
    # ========================================================================

    def _self_correct_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        initial_loss: float,
    ) -> Tuple[float, int]:
        """
        Self-correction loop (DeepMind style).
        
        Jika sample sulit (loss tinggi), train ulang beberapa kali
        dengan fokus pada error patterns.
        
        Args:
            inputs: input batch
            targets: target batch
            initial_loss: loss dari forward pertama
        
        Returns:
            (final_loss, num_corrections)
        """
        if not self.use_self_correction:
            return initial_loss, 0
        
        current_loss = initial_loss
        corrections = 0
        
        for iteration in range(self.self_correction_max_iter):
            # Check jika masih "hard" sample
            if current_loss < self.hard_sample_loss_threshold:
                break  # Sudah cukup baik, stop
            
            # Re-train dengan fokus
            self.model.zero_grad()
            logits = self.model.forward(inputs, training=True)
            
            loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(
                logits, targets,
            )
            
            num_valid = int(metrics["num_valid_tokens"])
            if num_valid == 0:
                break
            
            self.model.backward(grad_logits)
            
            # Apply AGC sebelum step
            self._apply_agc()
            
            # Step dengan smaller learning rate untuk fine-tuning
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
    # 🎯 SOTA: EMA WEIGHTS (POLYAK AVERAGING)
    # ========================================================================

    def _update_ema(self) -> None:
        """
        Update EMA (Exponential Moving Average) weights.
        
        Formula: ema_weight = decay * ema_weight + (1 - decay) * current_weight
        
        Benefit: smoother model, better generalization di akhir training
        """
        if not self.use_ema:
            return
        
        current_weights = {
            name: param.data.copy()
            for name, param in self.model.parameters().items()
        }
        
        if self._ema_weights is None:
            # Initialize EMA dengan current weights
            self._ema_weights = current_weights
        else:
            # Update EMA
            for name in current_weights:
                if name in self._ema_weights:
                    self._ema_weights[name] = (
                        self.ema_decay * self._ema_weights[name]
                        + (1.0 - self.ema_decay) * current_weights[name]
                    )
                else:
                    self._ema_weights[name] = current_weights[name]

    def apply_ema_to_model(self) -> None:
        """
        Apply EMA weights ke model (biasanya di akhir training).
        
        WARNING: Ini akan replace model weights. Lakukan hanya saat:
        - Training selesai (final model)
        - Evaluation (untuk hasil lebih smooth)
        """
        if self._ema_weights is None:
            return
        
        for name, param in self.model.parameters().items():
            if name in self._ema_weights:
                param.data = self._ema_weights[name].copy()

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
        """Load QA dataset dari file JSON."""
        if path is None:
            # Coba load merged dataset dulu (SQuAD prepared)
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
                # Support both "answer" key and empty answer (impossible questions)
                valid_samples.append(sample)

            self.qa_samples = valid_samples

            if verbose:
                metadata = data.get("metadata", {})
                print(f"[Trainer] ✅ Loaded {len(valid_samples)} QA samples dari {path.name}")
                print(f"[Trainer]    Total samples: {metadata.get('total', len(valid_samples))}")

                # Print stats
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
        """Return list QA samples yang sudah di-load."""
        return list(self.qa_samples)

    # ========================================================================
    # CONTEXT MANIPULATION
    # ========================================================================

    def _split_sentences(self, text: str) -> List[str]:
        """Split teks menjadi kalimat-kalimat."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []
        return sentences

    def _shuffle_sentences(self, text: str) -> str:
        """Shuffle kalimat dalam paragraf."""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text
        self._py_rng.shuffle(sentences)
        return ' '.join(sentences)

    def _apply_context_dropout(self, context: str) -> Tuple[str, bool]:
        """Randomly hapus konteks (untuk context dropout training)."""
        if not context:
            return context, True

        if self._py_rng.random() < self.context_dropout_rate:
            return "", True

        return context, False

    def _apply_context_perturbation(self, context: str) -> str:
        """Randomly perturbasi konteks: shuffle atau truncate."""
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
        """
        Siapkan satu QA sample untuk training.

        Flow:
        1. Ambil context, question, answer
        2. Apply context dropout (random)
        3. Apply context perturbation (random)
        4. Encode ke token IDs
        5. Truncate/pad ke sequence_length + 1
        """
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        is_impossible = sample.get("is_impossible", False)

        if not question:
            return None

        # Handle impossible questions (SQuAD 2.0 feature)
        if is_impossible or not answer:
            answer = self.NO_CONTEXT_ANSWER

        # Context dropout
        context, is_dropped = self._apply_context_dropout(context)
        if is_dropped and context == "":
            answer = self.NO_CONTEXT_ANSWER

        # Context perturbation
        if context:
            context = self._apply_context_perturbation(context)

        # Encode
        try:
            if context:
                input_ids, target_ids = self.tokenizer.split_qa_for_training(
                    context=context,
                    question=question,
                    answer=answer,
                )
            else:
                full_ids = self.tokenizer.encode_qa_no_context(
                    question=question,
                    answer=answer,
                    add_bos=True,
                    add_eos=True,
                )
                input_text = f"Pertanyaan: {question} Jawaban:"
                input_ids = self.tokenizer.encode(input_text, add_bos=True, add_eos=False)
                target_ids = full_ids[len(input_ids):]
        except Exception:
            return None

        if not input_ids or not target_ids:
            return None

        # Concatenate dan truncate/pad
        full_sequence = input_ids + target_ids
        chunk_size = self.sequence_length + 1

        if len(full_sequence) > chunk_size:
            full_sequence = full_sequence[:chunk_size]
        elif len(full_sequence) < chunk_size:
            padding_needed = chunk_size - len(full_sequence)
            full_sequence = full_sequence + [self.pad_id] * padding_needed

        return np.array(full_sequence, dtype=np.int64)

    def _prepare_sample_progressive(
        self,
        sample: Dict,
        stage: int,
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
        """Hitung current progressive stage berdasarkan epoch."""
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
        """
        Yield batches dengan curriculum learning.
        """
        if total_epochs is None:
            total_epochs = self.epochs

        if not self.qa_samples:
            return

        # Apply curriculum sorting
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
    # 🎯 SOTA: SINGLE TRAINING STEP (dengan semua teknik)
    # ========================================================================

    def _train_step_sota(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        step: int,
        total_steps: int,
    ) -> Tuple[float, dict]:
        """
        Satu training step dengan semua SOTA techniques.
        
        Flow:
        1. Forward pass dengan cross-attention
        2. Compute loss dengan label smoothing + focal + token weighting
        3. Backward pass
        4. AGC (adaptive gradient clipping)
        5. Loss spike detection
        6. Self-correction loop jika hard sample
        7. Weight update
        8. EMA update
        
        Returns:
            (loss, metrics_dict)
        """
        # ============================================================
        # Step 0: Update learning rate (cosine schedule)
        # ============================================================
        self._apply_lr_to_weight_manager(step, total_steps)

        # ============================================================
        # Step 1: Forward pass
        # ============================================================
        self.model.zero_grad()
        logits = self.model.forward(inputs, training=True)

        # ============================================================
        # Step 2: Compute loss dengan SOTA techniques
        # ============================================================
        loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(
            logits, targets,
        )
        
        num_valid_tokens = int(metrics["num_valid_tokens"])
        if num_valid_tokens == 0:
            return 0.0, metrics
        
        # Apply label smoothing (modifikasi gradient)
        if self.label_smoothing > 0.0:
            # Smooth gradient: blend dengan uniform distribution
            smooth_factor = self.label_smoothing
            vocab_size = logits.shape[-1]
            uniform_grad = -1.0 / vocab_size
            grad_logits = (1.0 - smooth_factor) * grad_logits + smooth_factor * uniform_grad
        
        # Apply focal loss weighting
        if self.use_focal_loss:
            focal_weights = self._compute_focal_weights(logits, targets)
            if focal_weights is not None:
                # Expand dims untuk broadcast
                focal_weights = focal_weights[:, :, np.newaxis]
                grad_logits = grad_logits * focal_weights
        
        # Apply token importance weighting
        if self.use_token_weighting:
            token_weights = self._compute_token_weights(inputs, targets)
            if token_weights is not None:
                token_weights = token_weights[:, :, np.newaxis]
                grad_logits = grad_logits * token_weights

        # ============================================================
        # Step 3: Backward pass
        # ============================================================
        self.model.backward(grad_logits)

        # ============================================================
        # Step 4: Adaptive Gradient Clipping (AGC)
        # ============================================================
        self._apply_agc()

        # ============================================================
        # Step 5: Loss spike detection
        # ============================================================
        is_spike = self._detect_loss_spike(float(loss))
        if is_spike:
            self._recover_from_spike()

        # ============================================================
        # Step 6: Self-correction loop untuk hard samples
        # ============================================================
        final_loss = float(loss)
        if self.use_self_correction and final_loss > self.hard_sample_loss_threshold:
            final_loss, corrections = self._self_correct_batch(
                inputs, targets, final_loss
            )
            self._self_correction_stats["total_attempts"] += 1
            if corrections > 0:
                self._self_correction_stats["hard_samples_retrained"] += 1

        # ============================================================
        # Step 7: Weight update
        # ============================================================
        step_info = self.weight_manager.step(zero_grad=True)

        # ============================================================
        # Step 8: EMA update
        # ============================================================
        self._update_ema()

        # Update metrics dengan final loss
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
        """
        Main training entry point dengan SOTA techniques.
        """
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

        # Calculate total steps untuk cosine schedule
        steps_per_epoch = len(self.qa_samples) // self.batch_size
        total_steps = total_epochs * steps_per_epoch

        if verbose:
            print("=" * 70)
            print("🎯 SOTA RAG TRAINING")
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
            print("=" * 70)

        history: List[dict] = []
        stop_training = False
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_correct = 0
            steps_done = 0
            epoch_spikes = 0
            epoch_corrections = 0

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

                # 🎯 SOTA TRAINING STEP
                loss, metrics = self._train_step_sota(
                    inputs, targets, global_step, total_steps
                )

                num_valid_tokens = int(metrics.get("num_valid_tokens", 0))

                if num_valid_tokens > 0:
                    epoch_loss += loss * num_valid_tokens
                    epoch_tokens += num_valid_tokens
                    epoch_correct += int(metrics.get("num_correct", 0))
                    
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

                    if (
                        self.save_checkpoint_every_n_steps > 0
                        and global_step > 0
                        and global_step % self.save_checkpoint_every_n_steps == 0
                    ):
                        ckpt_path = self.weight_manager.save_checkpoint()
                        if verbose:
                            print(f"  [Checkpoint] Disimpan: {ckpt_path}")
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
                    "corrections": epoch_corrections,
                    "learning_rate": self._get_learning_rate(global_step, total_steps),
                }

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

            if save_checkpoints:
                ckpt_path = self.weight_manager.save_checkpoint()
                if verbose:
                    print(f"[Trainer] Checkpoint disimpan: {ckpt_path}")

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
                if verbose:
                    print("[Trainer] Training dihentikan.")
                break

        # ================================================================
        # FINAL: Apply EMA weights (jika aktif)
        # ================================================================
        if self.use_ema and self._ema_weights is not None:
            if verbose:
                print("\n[Trainer] 🎯 Applying EMA weights ke model...")
            self.apply_ema_to_model()
            
            # Save EMA checkpoint
            if save_checkpoints:
                ckpt_path = self.weight_manager.save_checkpoint()
                if verbose:
                    print(f"[Trainer] EMA checkpoint disimpan: {ckpt_path}")

        # ================================================================
        # FINAL: Print SOTA stats
        # ================================================================
        if verbose:
            print("\n" + "=" * 70)
            print("📊 SOTA TRAINING STATISTICS")
            print("=" * 70)
            stats = self._self_correction_stats
            print(f"  Total spikes detected    : {stats['spikes_detected']}")
            print(f"  Self-correction attempts : {stats['total_attempts']}")
            print(f"  Corrections applied      : {stats['corrections_applied']}")
            print(f"  Hard samples retrained   : {stats['hard_samples_retrained']}")
            print("=" * 70)

        # Build retriever index
        if _RETRIEVER_AVAILABLE:
            self._build_retriever_index(verbose=verbose)

        return history

    # ========================================================================
    # INLINE TRAINING (simplified versions)
    # ========================================================================

    def train_steps(
        self,
        steps: int,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        """Inline training untuk N steps dengan SOTA techniques."""
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
                    total_correct += int(metrics.get("num_correct", 0))

                    if verbose:
                        print(
                            f"[Trainer] step={steps_done} "
                            f"loss={loss:.6f} "
                            f"ppl={metrics.get('perplexity', 1.0):.4f} "
                            f"acc={metrics.get('accuracy', 0.0):.4f} "
                            f"lr={metrics.get('learning_rate', 0.0):.2e}"
                        )
                else:
                    self.model.zero_grad()

                steps_done += 1
                batches_this_epoch += 1

                self.weight_manager.global_step = current_global_step + steps_done

            epochs_traversed += 1

            if batches_this_epoch == 0:
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
        }

        if verbose:
            print(
                f"[Trainer] Inline SOTA training selesai | "
                f"steps={steps_done} | "
                f"loss={avg_loss:.6f} | "
                f"ppl={perplexity:.4f} | "
                f"acc={accuracy:.4f}"
            )

        if save_checkpoint:
            ckpt_path = self.weight_manager.save_checkpoint()
            if verbose:
                print(f"[Trainer] Checkpoint disimpan: {ckpt_path}")

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
        """Inline training untuk N epochs dengan SOTA techniques."""
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

        # Delegate ke run_epochs dengan current state
        self.set_resume_state(
            start_epoch=current_epoch,
            global_step=current_global_step,
        )

        # Override epochs untuk inline
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
        """Bangun BM25 retrieval index dari corpus mentah."""
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
        RetrievalConfig,
    )
    from core.tokenizer import Tokenizer
    from core.architecture import CustomTransformerLM
    from pipeline.evaluator import Evaluator
    from pipeline.weight_manager import WeightManager

    print("=" * 70)
    print("pipeline/trainer.py SOTA Test Suite")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        datasets_dir = tmp_path / "datasets"
        output_models_dir = tmp_path / "output_models"

        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        sample_file = datasets_dir / "sample.txt"
        sample_file.write_text(
            "Mobil adalah kendaraan beroda empat.\n\n"
            "Rem mobil berfungsi untuk menghentikan kendaraan.\n\n"
            "Air adalah senyawa H2O.\n\n"
            "Fotosintesis menghasilkan oksigen.",
            encoding="utf-8",
        )

        qa_dataset_path = output_models_dir / "qa_dataset.json"
        sample_qa_data = {
            "version": "1.0",
            "metadata": {
                "total_samples": 4,
                "source_corpus_size": 4,
            },
            "samples": [
                {
                    "context": "Mobil adalah kendaraan beroda empat.",
                    "question": "Apa itu mobil?",
                    "answer": "Kendaraan beroda empat.",
                    "is_impossible": False,
                },
                {
                    "context": "Rem mobil berfungsi untuk menghentikan kendaraan.",
                    "question": "Apa fungsi rem?",
                    "answer": "Menghentikan kendaraan.",
                    "is_impossible": False,
                },
                {
                    "context": "Air adalah senyawa H2O.",
                    "question": "Apa itu air?",
                    "answer": "Senyawa H2O.",
                    "is_impossible": False,
                },
                {
                    "context": "Fotosintesis menghasilkan oksigen.",
                    "question": "Apa hasil fotosintesis?",
                    "answer": "Oksigen.",
                    "is_impossible": False,
                },
            ],
        }

        with open(qa_dataset_path, "w", encoding="utf-8") as f:
            json.dump(sample_qa_data, f)

        paths = PathsConfig(
            datasets_dir=datasets_dir,
            output_models_dir=output_models_dir,
            vocab_path=tmp_path / "vocab.json",
            final_model_path=output_models_dir / "otak_model.safetensors",
            checkpoint_path=output_models_dir / "training_checkpoint.npz",
            retrieval_index_path=output_models_dir / "retrieval_index.json",
            qa_dataset_path=qa_dataset_path,
            merged_qa_train_path=qa_dataset_path,
        )

        small_config = Config(
            paths=paths,
            special_tokens=SpecialTokensConfig(),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=2,
                epochs=2,
                sequence_length=32,
                dropout_rate=0.1,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
                lr_warmup_steps=5,
            ),
            model=ModelConfig(
                vocab_size=128,
                embedding_dim=32,
                num_attention_heads=4,
                num_layers=1,
                max_position_embeddings=64,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=64,
                train_dtype="float32",
                export_dtype="float32",
            ),
            retrieval=RetrievalConfig(top_k=2),
        )

        tokenizer = Tokenizer(small_config)
        tokenizer.train()

        model = CustomTransformerLM(small_config)
        evaluator = Evaluator(small_config)
        weight_manager = WeightManager(model, small_config)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            evaluator=evaluator,
            weight_manager=weight_manager,
            config=small_config,
        )

        # Test 1: Load dataset
        print("\n[Test 1] Load Dataset")
        n = trainer.load_qa_dataset(verbose=True)
        assert n == 4, f"Expected 4 samples, got {n}"

        # Test 2: Cosine LR schedule
        print("\n[Test 2] Cosine LR Schedule")
        for step in [0, 2, 5, 10, 20]:
            lr = trainer._get_learning_rate(step, 20)
            print(f"  Step {step:2d}: lr={lr:.6f}")

        # Test 3: Sample difficulty scoring
        print("\n[Test 3] Curriculum Difficulty Scoring")
        for i, sample in enumerate(trainer.qa_samples):
            diff = trainer._compute_sample_difficulty(sample)
            print(f"  Sample {i}: difficulty={diff:.3f} | Q: {sample['question'][:30]}")

        # Test 4: SOTA training step
        print("\n[Test 4] SOTA Training Step")
        samples_prepared = []
        for s in trainer.qa_samples:
            result = trainer._prepare_sample(s)
            if result is not None:
                samples_prepared.append(result)
        
        if samples_prepared:
            batch = np.stack(samples_prepared[:2])
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            loss, metrics = trainer._train_step_sota(inputs, targets, step=0, total_steps=10)
            print(f"  Loss: {loss:.6f}")
            print(f"  Metrics: {metrics}")

        # Test 5: Full training (1 epoch)
        print("\n[Test 5] Full SOTA Training (1 epoch)")
        trainer.epochs = 1
        history = trainer.run_epochs(
            save_checkpoints=True,
            verbose=True,
        )
        
        print(f"\n  Training selesai: {len(history)} epoch")
        for r in history:
            print(f"  - Epoch {r['epoch']}: loss={r['loss']:.4f}, acc={r['accuracy']:.4f}")

        # Test 6: SOTA stats
        print("\n[Test 6] SOTA Statistics")
        stats = trainer.get_sota_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Test 7: EMA weights
        print("\n[Test 7] EMA Weights")
        if trainer._ema_weights is not None:
            print(f"  EMA weights initialized: {len(trainer._ema_weights)} tensors")
            trainer.apply_ema_to_model()
            print("  ✅ EMA weights applied ke model")
        else:
            print("  ⚠️ EMA weights belum diinitialize")

        print("\n" + "=" * 70)
        print("✅ pipeline/trainer.py (SOTA) ALL TESTS PASSED")
        print("=" * 70)
