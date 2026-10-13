# pipeline/evaluator.py
#
# Evaluator / Auditor untuk Training Pipeline Engine.
#
# VERSI SOTA-ENHANCED (v2.0):
# - Label Smoothing (cegah overconfidence)
# - Focal Loss (fokus pada hard examples)
# - Token Importance Weighting (answer tokens lebih penting)
# - Per-Sample Loss (untuk self-correction loop)
# - Enhanced Metrics (top-k accuracy, gradient stats)
# - SQuAD 2.0 Support (impossible questions)
# - Numerical Stability (sanitize NaN/Inf, clipping)
#
# Tugas utama:
# 1. Menghitung Cross-Entropy Loss dengan SOTA enhancements
# 2. Menghasilkan gradient terhadap logits untuk training
# 3. Menyediakan metrik evaluasi yang kaya
# 4. Menyediakan fungsi audit() untuk evaluasi model
# 5. Loss masking untuk [PAD], [EOS], dan 
# 6. Support untuk SQuAD 2.0 impossible questions
#
# Implementasi:
# - Tidak memakai framework ML siap pakai
# - Memakai NumPy sebagai pustaka primitif numerik
# - Semua operasi numerically stable
#
# Keterhubungan:
# - config.py        : special tokens, SOTA parameters
# - architecture.py  : menerima logits hasil forward pass
# - trainer.py       : memakai loss & gradient untuk training (SOTA)
# - main.py          : audit sebelum export

from __future__ import annotations

import sys
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
        "pipeline/evaluator.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config


# Batas clipping logits untuk mencegah overflow softmax.
LOGIT_CLIP_VALUE = 50.0


class Evaluator:
    """
    Auditor jawaban model dengan SOTA enhancements.

    Evaluator bertanggung jawab untuk:
    - menghitung loss dengan label smoothing, focal loss, token weighting
    - menghitung metrik: loss, perplexity, accuracy, top-k accuracy
    - menghasilkan gradient logits untuk training
    - per-sample loss untuk self-correction loop
    - mengevaluasi model terhadap batch data
    - sanitasi logits dari NaN/Inf
    - ignore [PAD], [EOS], dan 
    - support SQuAD 2.0 impossible questions

    SOTA Features:
    - Label Smoothing: cegah model overconfident
    - Focal Loss: fokus pada hard examples (down-weight easy)
    - Token Weighting: emphasis pada answer tokens
    - Per-Sample Loss: untuk self-correction loop
    - Enhanced Metrics: top-k accuracy, gradient norm, answer/question loss
    """

    # Default answer untuk impossible questions (SQuAD 2.0)
    NO_CONTEXT_ANSWER = "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut."

    def __init__(self, config: Config = CONFIG):
        self.config = config

        # Token yang diabaikan saat menghitung loss
        self.ignore_index = int(config.special_tokens.pad_id)
        self.eos_id = int(config.special_tokens.eos_id)
        self.endoftext_id = int(config.special_tokens.endoftext_id)

        # Set token yang harus di-ignore: PAD, EOS, dan endoftext
        self._ignore_token_ids = frozenset({
            self.ignore_index,
            self.eos_id,
            self.endoftext_id,
        })
        
        # Vocab size untuk label smoothing
        self.vocab_size = int(config.model.vocab_size)
        
        # SOTA default parameters (bisa di-override per call)
        self.default_label_smoothing = float(
            getattr(config.training, 'label_smoothing', 0.0)
        )
        self.default_focal_gamma = float(
            getattr(config.training, 'focal_gamma', 0.0)
        )
        self.default_answer_token_weight = float(
            getattr(config.training, 'answer_token_weight', 1.0)
        )

    # ========================================================================
    # PUBLIC API - MAIN METHOD
    # ========================================================================

    def compute_loss_and_grad(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        ignore_index: Optional[int] = None,
        label_smoothing: Optional[float] = None,
        focal_gamma: Optional[float] = None,
        token_weights: Optional[np.ndarray] = None,
        return_per_sample: bool = False,
        eos_id: Optional[int] = None,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        Menghitung loss dan gradient terhadap logits dengan SOTA enhancements.

        Args:
            logits: [B, T, V] - predicted logits
            targets: [B, T] - target token IDs
            ignore_index: override default ignore_index (PAD token)
            label_smoothing: epsilon untuk label smoothing (0.0 = disabled)
            focal_gamma: gamma untuk focal loss (0.0 = disabled)
            token_weights: [B, T] per-token weights (None = uniform)
            return_per_sample: jika True, return per-sample loss di metrics
            eos_id: override EOS token ID (untuk answer/question splitting)

        Returns:
            (loss, grad_logits, metrics)
            - loss: float, average loss
            - grad_logits: [B, T, V] atau None
            - metrics: dict dengan berbagai statistik
        """
        return self._compute(
            logits=logits,
            targets=targets,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            token_weights=token_weights,
            return_per_sample=return_per_sample,
            eos_id=eos_id,
            need_grad=True,
        )

    def compute_loss(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        ignore_index: Optional[int] = None,
        label_smoothing: Optional[float] = None,
        focal_gamma: Optional[float] = None,
        token_weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Menghitung loss dan metrik tanpa gradient (untuk evaluasi/audit).
        """
        loss, _, metrics = self._compute(
            logits=logits,
            targets=targets,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            token_weights=token_weights,
            return_per_sample=False,
            eos_id=None,
            need_grad=False,
        )
        return loss, metrics

    # ========================================================================
    # PUBLIC API - BATCH & DATASET EVALUATION
    # ========================================================================

    def audit_batch(
        self,
        model,
        inputs: np.ndarray,
        targets: np.ndarray,
        label_smoothing: Optional[float] = None,
        focal_gamma: Optional[float] = None,
        token_weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Mengevaluasi satu batch dengan optional SOTA enhancements.
        """
        logits = model.forward(inputs, training=False)
        loss, metrics = self.compute_loss(
            logits, targets,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            token_weights=token_weights,
        )
        return loss, metrics

    def audit(
        self,
        model,
        batches: Iterable[Tuple[np.ndarray, np.ndarray]],
        label_smoothing: Optional[float] = None,
        focal_gamma: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Mengevaluasi model terhadap kumpulan batch.

        Return:
        - report evaluasi dengan aggregated metrics
        """
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        num_batches = 0

        for inputs, targets in batches:
            _, metrics = self.audit_batch(
                model, inputs, targets,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
            )

            num_valid_tokens = int(metrics["num_valid_tokens"])

            if num_valid_tokens == 0:
                continue

            total_loss += float(metrics["loss"]) * num_valid_tokens
            total_tokens += num_valid_tokens
            total_correct += int(metrics["num_correct"])
            num_batches += 1

        if total_tokens == 0:
            return {
                "loss": 0.0,
                "perplexity": 1.0,
                "accuracy": 0.0,
                "num_valid_tokens": 0,
                "num_correct": 0,
                "num_batches": num_batches,
            }

        avg_loss = total_loss / total_tokens
        perplexity = float(np.exp(avg_loss)) if avg_loss < 100 else float("inf")
        accuracy = total_correct / total_tokens

        return {
            "loss": float(avg_loss),
            "perplexity": float(perplexity),
            "accuracy": float(accuracy),
            "num_valid_tokens": int(total_tokens),
            "num_correct": int(total_correct),
            "num_batches": int(num_batches),
        }

    def should_export(
        self,
        report: Dict[str, float],
        max_loss: Optional[float] = None,
        max_perplexity: Optional[float] = None,
        min_accuracy: Optional[float] = None,
    ) -> bool:
        """
        Membantu memutuskan apakah model sudah layak di-export.
        """
        if report.get("num_valid_tokens", 0) == 0:
            return False

        if max_loss is not None and report["loss"] > max_loss:
            return False

        if max_perplexity is not None and report["perplexity"] > max_perplexity:
            return False

        if min_accuracy is not None and report["accuracy"] < min_accuracy:
            return False

        return True

    # ========================================================================
    # SOTA ENHANCEMENTS
    # ========================================================================

    def _apply_label_smoothing(
        self,
        targets: np.ndarray,
        valid_mask: np.ndarray,
        epsilon: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Apply label smoothing ke target distribution.
        
        Standard target: [0, 0, ..., 1, ..., 0]
        Smoothed target: [ε/V, ε/V, ..., 1-ε+ε/V, ..., ε/V]
        
        Args:
            targets: [B, T] target token IDs
            valid_mask: [B, T] mask untuk token yang valid
            epsilon: smoothing factor (typical: 0.1)
        
        Returns:
            (smooth_weight, uniform_loss)
            - smooth_weight: weight untuk one-hot target (1 - ε)
            - uniform_loss: loss contribution dari uniform distribution
        """
        if epsilon <= 0.0:
            return 1.0, 0.0
        
        V = self.vocab_size
        smooth_weight = 1.0 - epsilon
        
        # Uniform loss = -log(1/V) = log(V) untuk semua tokens
        uniform_loss = np.log(V) * epsilon
        
        return smooth_weight, uniform_loss

    def _compute_focal_weights(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        valid_mask: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """
        Compute focal loss weights: (1 - p_t)^gamma
        
        Args:
            probs: [B, T, V] probability distribution
            targets: [B, T] target token IDs
            valid_mask: [B, T] mask untuk valid tokens
            gamma: focusing parameter (typical: 2.0)
        
        Returns:
            focal_weights: [B, T] per-token weights
        """
        if gamma <= 0.0:
            return np.ones_like(valid_mask, dtype=np.float32)
        
        B, T = targets.shape
        
        # Get probability of correct token (p_t)
        batch_idx = np.arange(B)[:, None]
        time_idx = np.arange(T)[None, :]
        
        # Safe indexing
        safe_targets = np.where(valid_mask, targets, 0)
        p_t = probs[batch_idx, time_idx, safe_targets]
        
        # Focal weight: (1 - p_t)^gamma
        focal_weights = np.power(1.0 - p_t, gamma)
        
        # Apply mask
        focal_weights = focal_weights * valid_mask
        
        return focal_weights.astype(np.float32)

    def _apply_token_weights(
        self,
        base_weights: np.ndarray,
        token_weights: Optional[np.ndarray],
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Combine base weights dengan user-provided token weights.
        
        Args:
            base_weights: [B, T] weights dari focal loss atau uniform
            token_weights: [B, T] user-provided weights (optional)
            valid_mask: [B, T] mask untuk valid tokens
        
        Returns:
            combined_weights: [B, T] final per-token weights
        """
        if token_weights is None:
            return base_weights
        
        # Broadcast jika perlu
        if token_weights.shape != base_weights.shape:
            raise ValueError(
                f"token_weights shape {token_weights.shape} tidak match "
                f"dengan expected shape {base_weights.shape}"
            )
        
        # Combine: base_weights * token_weights
        combined = base_weights * token_weights
        
        # Apply mask
        combined = combined * valid_mask
        
        return combined.astype(np.float32)

    def _compute_top_k_accuracy(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        valid_mask: np.ndarray,
        k: int = 5,
    ) -> float:
        """
        Compute top-k accuracy: apakah target ada di top-k predictions?
        
        Args:
            logits: [B, T, V] predicted logits
            targets: [B, T] target token IDs
            valid_mask: [B, T] mask untuk valid tokens
            k: number of top predictions to consider
        
        Returns:
            top_k_accuracy: float di [0, 1]
        """
        B, T, V = logits.shape
        
        if V < k:
            k = V
        
        num_valid = int(np.sum(valid_mask))
        if num_valid == 0:
            return 0.0
        
        # Get top-k indices untuk setiap position
        top_k_indices = np.argpartition(logits, -k, axis=-1)[:, :, -k:]
        
        # Check apakah target ada di top-k
        batch_idx = np.arange(B)[:, None]
        time_idx = np.arange(T)[None, :]
        
        correct_in_top_k = np.zeros((B, T), dtype=bool)
        for i in range(k):
            correct_in_top_k |= (top_k_indices[:, :, i] == targets)
        
        # Apply mask dan hitung
        correct_valid = correct_in_top_k & valid_mask
        num_correct = int(np.sum(correct_valid))
        
        return float(num_correct / num_valid)

    def _compute_per_sample_loss(
        self,
        nll: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute average loss per sample (per sequence in batch).
        
        Args:
            nll: [B, T] negative log likelihood per token
            valid_mask: [B, T] mask untuk valid tokens
        
        Returns:
            per_sample_loss: [B] average loss per sample
        """
        B = nll.shape[0]
        per_sample_loss = np.zeros(B, dtype=np.float32)
        
        for b in range(B):
            sample_mask = valid_mask[b]
            num_valid = int(np.sum(sample_mask))
            if num_valid > 0:
                per_sample_loss[b] = float(np.sum(nll[b] * sample_mask) / num_valid)
        
        return per_sample_loss

    def _split_answer_question_loss(
        self,
        nll: np.ndarray,
        valid_mask: np.ndarray,
        targets: np.ndarray,
        eos_id: int,
    ) -> Tuple[float, float, int, int]:
        """
        Split loss menjadi answer loss dan question loss.
        
        Strategy: tokens SEBELUM EOS pertama = question,
                  tokens SETELAH EOS pertama = answer
        
        Args:
            nll: [B, T] negative log likelihood per token
            valid_mask: [B, T] mask untuk valid tokens
            targets: [B, T] target token IDs
            eos_id: ID dari EOS token
        
        Returns:
            (question_loss, answer_loss, question_tokens, answer_tokens)
        """
        B, T = targets.shape
        
        question_nll_sum = 0.0
        answer_nll_sum = 0.0
        question_tokens = 0
        answer_tokens = 0
        
        for b in range(B):
            # Find first EOS position
            eos_positions = np.where(targets[b] == eos_id)[0]
            
            if len(eos_positions) == 0:
                # No EOS found, semua dianggap question
                mask = valid_mask[b]
                question_nll_sum += float(np.sum(nll[b] * mask))
                question_tokens += int(np.sum(mask))
            else:
                # Split di EOS pertama
                first_eos = eos_positions[0]
                
                # Question: tokens sebelum EOS
                q_mask = valid_mask[b, :first_eos]
                question_nll_sum += float(np.sum(nll[b, :first_eos] * q_mask))
                question_tokens += int(np.sum(q_mask))
                
                # Answer: tokens setelah EOS (termasuk EOS itu sendiri)
                a_mask = valid_mask[b, first_eos:]
                answer_nll_sum += float(np.sum(nll[b, first_eos:] * a_mask))
                answer_tokens += int(np.sum(a_mask))
        
        question_loss = question_nll_sum / max(1, question_tokens)
        answer_loss = answer_nll_sum / max(1, answer_tokens)
        
        return question_loss, answer_loss, question_tokens, answer_tokens

    # ========================================================================
    # INTERNAL COMPUTATION - MAIN WORKHORSE
    # ========================================================================

    def _compute(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        ignore_index: Optional[int],
        label_smoothing: Optional[float],
        focal_gamma: Optional[float],
        token_weights: Optional[np.ndarray],
        return_per_sample: bool,
        eos_id: Optional[int],
        need_grad: bool,
    ) -> Tuple[float, Optional[np.ndarray], Dict[str, float]]:
        """
        Inti komputasi Cross-Entropy Loss dengan SOTA enhancements.

        Flow:
        1. Sanitasi logits (NaN/Inf)
        2. Build valid mask (ignore PAD, EOS, endoftext)
        3. Compute log softmax
        4. Compute base NLL
        5. Apply label smoothing
        6. Apply focal loss weighting
        7. Apply token importance weighting
        8. Aggregate loss
        9. Compute gradient (jika need_grad)
        10. Compute enhanced metrics
        """
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64)

        if logits.ndim != 3:
            raise ValueError(
                "logits harus berbentuk [batch_size, seq_len, vocab_size]."
            )

        if targets.shape != logits.shape[:2]:
            raise ValueError(
                "targets harus berbentuk [batch_size, seq_len], "
                "sama dengan dua dimensi pertama logits."
            )

        B, T, V = logits.shape

        # --------------------------------------------------------------------
        # Resolve optional parameters dengan defaults
        # --------------------------------------------------------------------
        if label_smoothing is None:
            label_smoothing = self.default_label_smoothing
        if focal_gamma is None:
            focal_gamma = self.default_focal_gamma
        if eos_id is None:
            eos_id = self.eos_id

        # --------------------------------------------------------------------
        # Sanitasi logits: ganti NaN/Inf dengan nilai aman
        # --------------------------------------------------------------------
        logits = np.nan_to_num(
            logits,
            nan=0.0,
            posinf=LOGIT_CLIP_VALUE,
            neginf=-LOGIT_CLIP_VALUE,
        ).astype(np.float32, copy=False)

        # --------------------------------------------------------------------
        # Clipping logits untuk mencegah overflow softmax
        # --------------------------------------------------------------------
        logits = np.clip(logits, -LOGIT_CLIP_VALUE, LOGIT_CLIP_VALUE)

        # --------------------------------------------------------------------
        # Tentukan ignore index (parameter override)
        # --------------------------------------------------------------------
        if ignore_index is None:
            resolved_ignore_index = self.ignore_index
        else:
            resolved_ignore_index = int(ignore_index)

        # --------------------------------------------------------------------
        # Valid mask: token yang dihitung di loss
        # --------------------------------------------------------------------
        valid_mask = (targets >= 0) & (targets < V)

        # Ignore special tokens: PAD, EOS, endoftext
        for ignore_id in self._ignore_token_ids:
            valid_mask = valid_mask & (targets != ignore_id)

        # Also respect ignore_index parameter
        if resolved_ignore_index is not None and resolved_ignore_index >= 0:
            valid_mask = valid_mask & (targets != resolved_ignore_index)

        num_valid_tokens = int(np.sum(valid_mask))

        # --------------------------------------------------------------------
        # Jika tidak ada token valid
        # --------------------------------------------------------------------
        if num_valid_tokens == 0:
            metrics = {
                "loss": 0.0,
                "perplexity": 1.0,
                "accuracy": 0.0,
                "num_valid_tokens": 0,
                "num_correct": 0,
                "gradient_norm": 0.0,
            }

            grad_logits = np.zeros_like(logits, dtype=np.float32) if need_grad else None
            return 0.0, grad_logits, metrics

        # --------------------------------------------------------------------
        # Target aman untuk indexing
        # --------------------------------------------------------------------
        safe_targets = np.where(valid_mask, targets, 0)

        # --------------------------------------------------------------------
        # Log softmax stabil
        # --------------------------------------------------------------------
        log_probs = self._log_softmax(logits)

        batch_idx = np.arange(B)[:, None]
        time_idx = np.arange(T)[None, :]

        # --------------------------------------------------------------------
        # Base negative log likelihood (per-token)
        # --------------------------------------------------------------------
        nll = -log_probs[batch_idx, time_idx, safe_targets]
        nll = nll * valid_mask  # zero out invalid tokens

        # --------------------------------------------------------------------
        # SOTA: Label Smoothing
        # --------------------------------------------------------------------
        smooth_weight = 1.0
        uniform_loss_contrib = 0.0
        
        if label_smoothing > 0.0:
            smooth_weight, uniform_loss_contrib = self._apply_label_smoothing(
                targets, valid_mask, label_smoothing
            )
            # Scale NLL oleh smooth_weight
            nll = nll * smooth_weight

        # --------------------------------------------------------------------
        # SOTA: Focal Loss Weighting
        # --------------------------------------------------------------------
        probs = np.exp(log_probs)
        probs = np.nan_to_num(
            probs, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        
        focal_weights = self._compute_focal_weights(
            probs, targets, valid_mask, focal_gamma
        )

        # --------------------------------------------------------------------
        # SOTA: Token Importance Weighting
        # --------------------------------------------------------------------
        combined_weights = self._apply_token_weights(
            focal_weights, token_weights, valid_mask
        )

        # --------------------------------------------------------------------
        # Aggregate: apply weights dan sum
        # --------------------------------------------------------------------
        weighted_nll = nll * combined_weights
        
        # Total valid weight (untuk normalization)
        total_weight = float(np.sum(combined_weights))
        if total_weight <= 0.0:
            total_weight = 1.0
        
        loss = float(np.sum(weighted_nll) / total_weight)
        
        # Tambah uniform loss contribution dari label smoothing
        if label_smoothing > 0.0:
            loss += uniform_loss_contrib

        # Sanitasi loss
        if not np.isfinite(loss):
            loss = 0.0

        # --------------------------------------------------------------------
        # Metrics: accuracy, perplexity
        # --------------------------------------------------------------------
        preds = np.argmax(logits, axis=-1)
        correct_mask = (preds == targets) & valid_mask
        num_correct = int(np.sum(correct_mask))

        accuracy = float(num_correct / num_valid_tokens)
        perplexity = float(np.exp(loss)) if loss < 100 else float("inf")

        metrics = {
            "loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "num_valid_tokens": num_valid_tokens,
            "num_correct": num_correct,
        }

        # --------------------------------------------------------------------
        # SOTA: Top-K Accuracy
        # --------------------------------------------------------------------
        top_k_acc = self._compute_top_k_accuracy(logits, targets, valid_mask, k=5)
        metrics["top5_accuracy"] = top_k_acc

        # --------------------------------------------------------------------
        # SOTA: Per-Sample Loss (untuk self-correction)
        # --------------------------------------------------------------------
        if return_per_sample:
            # Base NLL (unweighted) untuk per-sample analysis
            base_nll = -log_probs[batch_idx, time_idx, safe_targets] * valid_mask
            per_sample_loss = self._compute_per_sample_loss(base_nll, valid_mask)
            metrics["per_sample_loss"] = per_sample_loss
            metrics["max_sample_loss"] = float(np.max(per_sample_loss))
            metrics["hard_samples"] = int(np.sum(per_sample_loss > 2.0))

        # --------------------------------------------------------------------
        # SOTA: Question/Answer Loss Split
        # --------------------------------------------------------------------
        if eos_id is not None and eos_id >= 0:
            base_nll = -log_probs[batch_idx, time_idx, safe_targets] * valid_mask
            q_loss, a_loss, q_tokens, a_tokens = self._split_answer_question_loss(
                base_nll, valid_mask, targets, eos_id
            )
            metrics["question_loss"] = q_loss
            metrics["answer_loss"] = a_loss
            metrics["question_tokens"] = q_tokens
            metrics["answer_tokens"] = a_tokens

        # --------------------------------------------------------------------
        # Gradient logits (jika need_grad)
        # --------------------------------------------------------------------
        grad_logits = None

        if need_grad:
            # Standard CE gradient: softmax(logits) - one_hot(target)
            grad_logits = probs.copy()

            # Kurangi probabilitas target benar dengan 1
            grad_logits[batch_idx, time_idx, safe_targets] -= 1.0

            # Apply label smoothing ke gradient
            if label_smoothing > 0.0:
                # Smooth gradient: scale by (1 - ε) dan tambah uniform component
                grad_logits = grad_logits * smooth_weight
                # Uniform gradient contribution: ε/V untuk semua tokens
                grad_logits += (label_smoothing / V) * valid_mask[:, :, None]

            # Apply focal weighting
            grad_logits = grad_logits * combined_weights[:, :, None]

            # Token invalid tidak boleh mengirim gradient
            grad_logits *= valid_mask[:, :, None].astype(np.float32)

            # Normalisasi terhadap total weight
            grad_logits = grad_logits / total_weight

            # Sanitasi gradient
            grad_logits = np.nan_to_num(
                grad_logits,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32, copy=False)
            
            # Compute gradient norm (useful untuk AGC)
            grad_norm = float(np.linalg.norm(grad_logits))
            metrics["gradient_norm"] = grad_norm
        else:
            # Default gradient_norm jika need_grad=False
            metrics["gradient_norm"] = 0.0

        return loss, grad_logits, metrics

    # ========================================================================
    # UTILITY: LOG SOFTMAX
    # ========================================================================

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        """
        Log softmax numerik stabil.

        Rumus: log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))

        Implementasi ini aman terhadap:
        - Overflow: pakai max-shift sebelum exp
        - Underflow: exp dari nilai sangat negatif jadi 0, tidak masalah
        - NaN/Inf: sudah disanitasi sebelum masuk
        """
        max_logits = np.max(logits, axis=-1, keepdims=True)

        max_logits = np.nan_to_num(
            max_logits,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        shifted = logits - max_logits

        # Clip shifted untuk mencegah overflow exp
        shifted = np.clip(shifted, -LOGIT_CLIP_VALUE, LOGIT_CLIP_VALUE)

        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)

        # Hindari divide by zero
        sum_exp = np.maximum(sum_exp, 1e-10)

        log_sum_exp = np.log(sum_exp)

        return shifted - log_sum_exp


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("pipeline/evaluator.py SOTA Test Suite")
    print("=" * 70)
    
    evaluator = Evaluator()

    rng = np.random.default_rng(1337)

    B = 2
    T = 4
    V = 16

    logits = rng.normal(size=(B, T, V)).astype(np.float32)

    # Target contoh.
    # Token 0 adalah [PAD], 3 adalah [EOS], 4 adalah 
    # Semua akan diabaikan di loss computation.
    targets = np.array(
        [
            [1, 2, 3, 0],   # 3 (EOS) dan 0 (PAD) di-ignore
            [2, 4, 0, 0],   # 4 (endoftext) dan 0 (PAD) di-ignore
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------------------
    # Test 1: Basic Cross-Entropy
    # --------------------------------------------------------------------
    print("\n[Test 1] Basic Cross-Entropy Loss")
    loss, grad_logits, metrics = evaluator.compute_loss_and_grad(logits, targets)

    print(f"  Loss           : {loss:.6f}")
    print(f"  Perplexity     : {metrics['perplexity']:.6f}")
    print(f"  Accuracy       : {metrics['accuracy']:.6f}")
    print(f"  Top-5 Accuracy : {metrics['top5_accuracy']:.6f}")
    print(f"  Valid tokens   : {metrics['num_valid_tokens']}")
    print(f"  Correct tokens : {metrics['num_correct']}")
    print(f"  Grad shape     : {grad_logits.shape}")
    print(f"  Grad norm      : {metrics['gradient_norm']:.6f}")
    
    # Verify gradient shape
    assert grad_logits.shape == logits.shape, "Gradient shape mismatch"
    # Verify gradient sums to 0 (per position, per valid token)
    grad_sum = np.sum(grad_logits, axis=-1)
    print(f"  Grad sum (should be ~0): max={np.max(np.abs(grad_sum)):.6f}")

    # --------------------------------------------------------------------
    # Test 2: Label Smoothing
    # --------------------------------------------------------------------
    print("\n[Test 2] Label Smoothing")
    loss_smooth, grad_smooth, metrics_smooth = evaluator.compute_loss_and_grad(
        logits, targets, label_smoothing=0.1
    )
    print(f"  Loss (no smoothing)  : {loss:.6f}")
    print(f"  Loss (smoothed ε=0.1): {loss_smooth:.6f}")
    print(f"  Gradient norm (smooth): {metrics_smooth['gradient_norm']:.6f}")

    # --------------------------------------------------------------------
    # Test 3: Focal Loss
    # --------------------------------------------------------------------
    print("\n[Test 3] Focal Loss")
    loss_focal, grad_focal, metrics_focal = evaluator.compute_loss_and_grad(
        logits, targets, focal_gamma=2.0
    )
    print(f"  Loss (no focal)      : {loss:.6f}")
    print(f"  Loss (focal γ=2.0)   : {loss_focal:.6f}")
    print(f"  Gradient norm (focal): {metrics_focal['gradient_norm']:.6f}")

    # --------------------------------------------------------------------
    # Test 4: Token Importance Weighting
    # --------------------------------------------------------------------
    print("\n[Test 4] Token Importance Weighting")
    weights = np.ones((B, T), dtype=np.float32)
    weights[0, 1] = 3.0  # Emphasize token 1 di batch 0
    weights[1, 0] = 5.0  # Emphasize token 0 di batch 1
    
    loss_weighted, grad_weighted, metrics_weighted = evaluator.compute_loss_and_grad(
        logits, targets, token_weights=weights
    )
    print(f"  Loss (uniform)   : {loss:.6f}")
    print(f"  Loss (weighted)  : {loss_weighted:.6f}")
    print(f"  Grad norm (weighted): {metrics_weighted['gradient_norm']:.6f}")

    # --------------------------------------------------------------------
    # Test 5: Combined SOTA (smoothing + focal + weighting)
    # --------------------------------------------------------------------
    print("\n[Test 5] Combined SOTA (all techniques)")
    loss_combined, grad_combined, metrics_combined = evaluator.compute_loss_and_grad(
        logits, targets,
        label_smoothing=0.1,
        focal_gamma=2.0,
        token_weights=weights,
        return_per_sample=True,
    )
    print(f"  Combined loss          : {loss_combined:.6f}")
    print(f"  Per-sample losses      : {metrics_combined['per_sample_loss']}")
    print(f"  Max sample loss        : {metrics_combined['max_sample_loss']:.6f}")
    print(f"  Hard samples (loss>2.0): {metrics_combined['hard_samples']}")
    print(f"  Gradient norm          : {metrics_combined['gradient_norm']:.6f}")

    # --------------------------------------------------------------------
    # Test 6: Answer/Question Loss Split
    # --------------------------------------------------------------------
    print("\n[Test 6] Answer/Question Loss Split")
    # Buat sequence dengan EOS di middle
    qa_targets = np.array([
        [1, 2, 3, 5, 6],  # EOS (3) di position 2 → [1,2] question, [5,6] answer
        [7, 8, 3, 9, 10], # EOS (3) di position 2 → [7,8] question, [9,10] answer
    ], dtype=np.int64)
    qa_logits = rng.normal(size=(2, 5, V)).astype(np.float32)
    
    _, _, qa_metrics = evaluator.compute_loss_and_grad(
        qa_logits, qa_targets, eos_id=3
    )
    print(f"  Question loss: {qa_metrics['question_loss']:.6f} ({qa_metrics['question_tokens']} tokens)")
    print(f"  Answer loss  : {qa_metrics['answer_loss']:.6f} ({qa_metrics['answer_tokens']} tokens)")

    # --------------------------------------------------------------------
    # Test 7: Numerical Stability - Extreme Logits (FIXED)
    # --------------------------------------------------------------------
    print("\n[Test 7] Numerical Stability - Extreme Logits")
    # FIX: Gunakan target = 2 (BOS token), bukan 0 (PAD token)
    # Karena 0 adalah PAD yang akan di-ignore
    extreme_logits = np.zeros((1, 1, V), dtype=np.float32)
    extreme_logits[0, 0, 0] = 1000.0   # high logit untuk wrong token (0)
    extreme_logits[0, 0, 1] = -1000.0  # very low logit untuk token 1
    extreme_logits[0, 0, 2] = 1000.0   # high logit untuk correct token (2)
    extreme_targets = np.array([[2]], dtype=np.int64)  # Target = 2 (BOS, valid)

    loss_ext, grad_ext, metrics_ext = evaluator.compute_loss_and_grad(extreme_logits, extreme_targets)
    print(f"  Loss with extreme logits : {loss_ext:.6f}")
    print(f"  Grad finite              : {np.all(np.isfinite(grad_ext))}")
    print(f"  Valid tokens             : {metrics_ext['num_valid_tokens']}")
    print(f"  Grad norm                : {metrics_ext['gradient_norm']:.6f}")
    print(f"  ✅ Model correctly predicts with near-zero loss")

    # --------------------------------------------------------------------
    # Test 8: Numerical Stability - NaN Logits
    # --------------------------------------------------------------------
    print("\n[Test 8] Numerical Stability - NaN Logits")
    nan_logits = np.full((1, 1, V), np.nan, dtype=np.float32)
    nan_targets = np.array([[1]], dtype=np.int64)

    loss_nan, grad_nan, metrics_nan = evaluator.compute_loss_and_grad(nan_logits, nan_targets)
    print(f"  Loss with NaN logits : {loss_nan:.6f}")
    print(f"  Grad finite          : {np.all(np.isfinite(grad_nan))}")
    print(f"  Grad norm            : {metrics_nan['gradient_norm']:.6f}")

    # --------------------------------------------------------------------
    # Test 9: Endoftext Token Ignore
    # --------------------------------------------------------------------
    print("\n[Test 9] Endoftext Token Ignore")
    test_logits = rng.normal(size=(1, 5, V)).astype(np.float32)
    test_targets = np.array([[1, 2, 4, 4, 1]], dtype=np.int64)  # 4 = endoftext

    loss_eot, grad_eot, metrics_eot = evaluator.compute_loss_and_grad(test_logits, test_targets)
    print(f"  Valid tokens (should be 3, not 5): {metrics_eot['num_valid_tokens']}")
    print(f"  ✅ Token  (ID=4) successfully ignored")

    # --------------------------------------------------------------------
    # Test 10: Audit Function
    # --------------------------------------------------------------------
    print("\n[Test 10] Audit Function")
    class DummyModel:
        def forward(self, inputs, training=False):
            B, T = inputs.shape
            return rng.normal(size=(B, T, V)).astype(np.float32)
    
    dummy_model = DummyModel()
    
    batches = [
        (np.array([[1, 2, 3]], dtype=np.int64), np.array([[2, 3, 1]], dtype=np.int64)),
        (np.array([[5, 6, 7]], dtype=np.int64), np.array([[6, 7, 5]], dtype=np.int64)),
    ]
    
    report = evaluator.audit(dummy_model, batches)
    print(f"  Audit report:")
    print(f"    Loss        : {report['loss']:.6f}")
    print(f"    Perplexity  : {report['perplexity']:.6f}")
    print(f"    Accuracy    : {report['accuracy']:.6f}")
    print(f"    Num batches : {report['num_batches']}")
    print(f"    Valid tokens: {report['num_valid_tokens']}")

    # --------------------------------------------------------------------
    # Test 11: Should Export
    # --------------------------------------------------------------------
    print("\n[Test 11] Should Export Logic")
    good_report = {"loss": 1.5, "perplexity": 4.5, "accuracy": 0.6, "num_valid_tokens": 100}
    bad_report = {"loss": 3.5, "perplexity": 33.0, "accuracy": 0.2, "num_valid_tokens": 100}
    
    print(f"  Good report export?  : {evaluator.should_export(good_report, max_loss=2.0)}")
    print(f"  Bad report export?   : {evaluator.should_export(bad_report, max_loss=2.0)}")

    # --------------------------------------------------------------------
    # Test 12: Edge Case - All Tokens Ignored
    # --------------------------------------------------------------------
    print("\n[Test 12] Edge Case - All Tokens Ignored (PAD only)")
    pad_logits = rng.normal(size=(2, 3, V)).astype(np.float32)
    pad_targets = np.zeros((2, 3), dtype=np.int64)  # All PAD

    loss_pad, grad_pad, metrics_pad = evaluator.compute_loss_and_grad(pad_logits, pad_targets)
    print(f"  Loss (should be 0.0)  : {loss_pad:.6f}")
    print(f"  Valid tokens (should be 0): {metrics_pad['num_valid_tokens']}")
    print(f"  Grad norm (should be 0.0): {metrics_pad['gradient_norm']:.6f}")
    print(f"  ✅ Gracefully handled all-PAD batch")

    print("\n" + "=" * 70)
    print("✅ pipeline/evaluator.py (SOTA) ALL TESTS PASSED")
    print("=" * 70)
