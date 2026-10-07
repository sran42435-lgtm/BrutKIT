# pipeline/evaluator.py
#
# Evaluator / Auditor untuk Training Pipeline Engine.
#
# Tugas utama:
# 1. Menghitung Cross-Entropy Loss.
# 2. Menghitung metrik evaluasi: loss, perplexity, accuracy.
# 3. Menghasilkan gradient terhadap logits untuk training.
# 4. Menyediakan fungsi audit() untuk mengevaluasi model pada batch data.
# 5. Menyediakan should_export() untuk membantu keputusan export model.
#
# Implementasi:
# - Tidak memakai framework ML siap pakai.
# - Memakai NumPy sebagai pustaka primitif numerik.
#
# Keterhubungan:
# - config.py        : special token untuk ignore index
# - architecture.py  : menerima logits hasil forward pass model
# - trainer.py       : memakai loss & gradient untuk training
# - main.py          : audit sebelum export

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
# Memastikan project root ada di sys.path, sehingga file di dalam folder
# pipeline/ tetap bisa meng-import config.py meskipun dijalankan langsung:
#   python pipeline/evaluator.py
# atau:
#   cd pipeline && python evaluator.py

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


class Evaluator:
    """
    Auditor jawaban model.

    Evaluator bertanggung jawab untuk:
    - menghitung loss
    - menghitung perplexity
    - menghitung accuracy
    - menghasilkan gradient logits saat training
    - mengevaluasi model terhadap batch data
    """

    def __init__(self, config: Config = CONFIG):
        self.config = config

        # Secara default token [PAD] diabaikan saat menghitung loss.
        self.ignore_index = int(config.special_tokens.pad_id)

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def compute_loss_and_grad(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        ignore_index: Optional[int] = None,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """
        Menghitung loss dan gradient terhadap logits.

        Input:
        - logits  : [B, T, V]
        - targets : [B, T]

        Output:
        - loss          : float
        - grad_logits   : [B, T, V]
        - metrics       : dict
        """
        return self._compute(
            logits=logits,
            targets=targets,
            ignore_index=ignore_index,
            need_grad=True,
        )

    def compute_loss(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        ignore_index: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Menghitung loss dan metrik tanpa gradient.
        Dipakai untuk evaluasi / audit.
        """
        loss, _, metrics = self._compute(
            logits=logits,
            targets=targets,
            ignore_index=ignore_index,
            need_grad=False,
        )
        return loss, metrics

    def audit_batch(
        self,
        model,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Mengevaluasi satu batch.

        Input:
        - inputs  : token ID [B, T]
        - targets : target token ID [B, T]

        Output:
        - loss
        - metrics
        """
        logits = model.forward(inputs, training=False)
        loss, metrics = self.compute_loss(logits, targets)
        return loss, metrics

    def audit(
        self,
        model,
        batches: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """
        Mengevaluasi model terhadap kumpulan batch.

        Parameter:
        - model   : instance CustomTransformerLM
        - batches : iterable berisi tuple (inputs, targets)

        Return:
        - report evaluasi
        """
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        num_batches = 0

        for inputs, targets in batches:
            _, metrics = self.audit_batch(model, inputs, targets)

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

        Contoh pemakaian:
            evaluator.should_export(
                report,
                max_loss=2.0,
                max_perplexity=8.0,
                min_accuracy=0.3,
            )
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
    # INTERNAL COMPUTATION
    # ========================================================================

    def _compute(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        ignore_index: Optional[int],
        need_grad: bool,
    ) -> Tuple[float, Optional[np.ndarray], Dict[str, float]]:
        """
        Inti komputasi Cross-Entropy Loss.

        Rumus:
            L = -1/N * Σ log(softmax(logits)[target])
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
        # Tentukan ignore index
        # --------------------------------------------------------------------
        if ignore_index is None:
            resolved_ignore_index = self.ignore_index
        else:
            resolved_ignore_index = int(ignore_index)

        # --------------------------------------------------------------------
        # Valid mask
        # --------------------------------------------------------------------
        valid_mask = (targets >= 0) & (targets < V)

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
        # Negative log likelihood
        # --------------------------------------------------------------------
        nll = -log_probs[batch_idx, time_idx, safe_targets]
        nll = nll * valid_mask

        loss = float(np.sum(nll) / num_valid_tokens)

        # --------------------------------------------------------------------
        # Probability untuk gradient dan metrik
        # --------------------------------------------------------------------
        probs = np.exp(log_probs)

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
        # Gradient logits
        # --------------------------------------------------------------------
        grad_logits = None

        if need_grad:
            # grad = softmax(logits) - one_hot(target)
            grad_logits = probs.copy()

            # Kurangi probabilitas target benar dengan 1
            grad_logits[batch_idx, time_idx, safe_targets] -= 1.0

            # Token invalid tidak boleh mengirim gradient
            grad_logits *= valid_mask[:, :, None].astype(np.float32)

            # Normalisasi terhadap jumlah token valid
            grad_logits = grad_logits / float(num_valid_tokens)

        return loss, grad_logits, metrics

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        """
        Log softmax numerik stabil.

        Rumus:
            log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        """
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        return shifted - log_sum_exp


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    evaluator = Evaluator()

    rng = np.random.default_rng(1337)

    B = 2
    T = 4
    V = 16

    logits = rng.normal(size=(B, T, V)).astype(np.float32)

    # Target contoh.
    # Token 0 adalah [PAD] pada config default, sehingga akan diabaikan.
    targets = np.array(
        [
            [1, 2, 3, 0],
            [2, 0, 0, 0],
        ],
        dtype=np.int64,
    )

    loss, grad_logits, metrics = evaluator.compute_loss_and_grad(logits, targets)

    print("pipeline/evaluator.py test OK")
    print(f"Loss           : {loss:.6f}")
    print(f"Perplexity     : {metrics['perplexity']:.6f}")
    print(f"Accuracy       : {metrics['accuracy']:.6f}")
    print(f"Valid tokens   : {metrics['num_valid_tokens']}")
    print(f"Correct tokens : {metrics['num_correct']}")
    print(f"Grad shape     : {grad_logits.shape}")
