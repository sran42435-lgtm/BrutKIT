# pipeline/trainer.py
#
# Trainer / Training Loop Engine untuk RAG-based Training Pipeline.
#
# Arsitektur: Single-Head + RAG (Retrieval-Augmented Generation)
# - Context-conditioned training
# - QA format: "Konteks: X Pertanyaan: Y Jawaban: Z"
# - Loss ↓ = Presisi ↑ (korelasi langsung, no domain mixing)
#
# Fitur Training:
# - Load QA dataset dari qa_dataset.json (hasil convert_dataset.py)
# - Context dropout (20% train tanpa konteks) - mitigasi hallucination
# - Context perturbation (30% shuffle kalimat) - cegah urutan memorization
# - Progressive training stages - difficulty meningkat bertahap
# - Multi-context samples (gabungan beberapa paragraf)
# - Resume dari checkpoint, early stopping, periodic checkpoint
# - Inline training dari playground
#
# Keterhubungan:
# - config.py              : hyperparameter training + RAG
# - core/tokenizer.py      : encode QA samples
# - core/architecture.py   : model (single LM head)
# - core/retriever.py      : build index setelah training (optional)
# - pipeline/evaluator.py  : menghitung loss dan gradient
# - pipeline/weight_manager.py : memperbarui bobot

from __future__ import annotations

import json
import random
import re
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
        "pipeline/trainer.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config

# Import retriever (optional, untuk build index setelah training)
try:
    from core.retriever import BM25Retriever, load_corpus_from_directory
    _RETRIEVER_AVAILABLE = True
except ImportError:
    _RETRIEVER_AVAILABLE = False


class Trainer:
    """
    Mesin latihan untuk RAG (Retrieval-Augmented Generation).

    Arsitektur:
    - Single LM head (no domain contamination)
    - Context-conditioned training
    - Loss ↓ = Presisi ↑ (korelasi langsung)

    Fitur:
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

        # RAG-specific training parameters
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

        # State untuk resume training
        self.start_epoch = 0
        self.start_global_step = 0

        # Token IDs untuk special tokens
        self.pad_id = int(config.special_tokens.pad_id)
        self.bos_id = int(config.special_tokens.bos_id)
        self.eos_id = int(config.special_tokens.eos_id)
        self.endoftext_id = int(config.special_tokens.endoftext_id)

        # QA Dataset (akan di-load saat training)
        self.qa_samples: List[Dict] = []

        # Random generators
        self._rng = np.random.default_rng(self.seed)
        self._py_rng = random.Random(self.seed)

        # Retriever (dibangun setelah training jika tersedia)
        self._retriever: Optional['BM25Retriever'] = None

        if self.batch_size <= 0:
            raise ValueError("batch_size harus lebih besar dari 0.")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length harus lebih besar dari 0.")

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
        """
        Load QA dataset dari file JSON.

        File ini dibuat oleh scripts/convert_dataset.py.

        Args:
            path: path ke qa_dataset.json (default dari config)
            verbose: print progress

        Returns:
            int: jumlah samples yang dimuat
        """
        if path is None:
            path = self.qa_dataset_path
        else:
            path = Path(path)

        if not path.exists():
            if verbose:
                print(f"[Trainer] ⚠️ QA dataset tidak ditemukan: {path}")
                print(f"[Trainer] Jalankan: python scripts/convert_dataset.py")
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            samples = data.get("samples", [])

            # Validasi struktur
            valid_samples = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                if "question" not in sample:
                    continue
                # Answer bisa kosong (untuk context dropout samples)
                valid_samples.append(sample)

            self.qa_samples = valid_samples

            if verbose:
                metadata = data.get("metadata", {})
                print(f"[Trainer] ✅ Loaded {len(valid_samples)} QA samples dari {path}")
                print(f"[Trainer]    Source corpus: {metadata.get('source_corpus_size', 'N/A')} paragraphs")
                print(f"[Trainer]    Augmentation factor: {metadata.get('augmentation_factor', 'N/A')}")

                # Print template distribution
                type_counts = metadata.get("template_types", {})
                if type_counts:
                    print(f"[Trainer]    Template distribution:")
                    for t_type, count in sorted(type_counts.items()):
                        print(f"      - {t_type:25s}: {count:4d}")

            return len(valid_samples)

        except Exception as exc:
            if verbose:
                print(f"[Trainer] ❌ Gagal load QA dataset: {exc}")
            return 0

    def get_qa_samples(self) -> List[Dict]:
        """Return list QA samples yang sudah di-load."""
        return list(self.qa_samples)

    # ========================================================================
    # CONTEXT MANIPULATION (Mitigasi Risiko)
    # ========================================================================

    def _split_sentences(self, text: str) -> List[str]:
        """Split teks menjadi kalimat-kalimat."""
        # Pattern: [.!?] diikuti spasi + huruf kapital, atau akhir string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text.strip()] if text.strip() else []
        return sentences

    def _shuffle_sentences(self, text: str) -> str:
        """
        Shuffle kalimat dalam paragraf.

        Digunakan untuk mitigasi risiko: model hafal urutan kata.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text
        self._py_rng.shuffle(sentences)
        return ' '.join(sentences)

    def _apply_context_dropout(self, context: str) -> Tuple[str, bool]:
        """
        Randomly hapus konteks (untuk context dropout training).

        Return:
        - (context, is_dropped): context yang mungkin kosong + flag apakah dropped
        """
        if not context:
            return context, True

        if self._py_rng.random() < self.context_dropout_rate:
            return "", True

        return context, False

    def _apply_context_perturbation(self, context: str) -> str:
        """
        Randomly perturbasi konteks:
        - Shuffle kalimat
        - Truncate sebagian
        """
        if not context:
            return context

        if self._py_rng.random() >= self.context_perturbation_rate:
            return context  # Tidak ada perturbasi

        # Pilih jenis perturbasi
        perturbation_type = self._py_rng.choice(["shuffle", "truncate"])

        if perturbation_type == "shuffle":
            return self._shuffle_sentences(context)

        elif perturbation_type == "truncate":
            sentences = self._split_sentences(context)
            if len(sentences) <= 1:
                # Truncate karakter
                mid = len(context) // 2
                return context[:mid].strip()
            # Ambil 50-80% kalimat
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
        4. Encode ke token IDs dengan format:
           "Konteks: {context} Pertanyaan: {question} Jawaban: {answer}"
        5. Truncate/ pad ke sequence_length + 1

        Return:
        - (input_ids, target_ids) numpy arrays dengan shape [sequence_length+1]
        - None jika sample tidak valid
        """
        context = sample.get("context", "")
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        if not question:
            return None

        # ================================================================
        # Step 1: Context Dropout (20% waktu tanpa konteks)
        # ================================================================
        context, is_dropped = self._apply_context_dropout(context)

        # Jika konteks di-drop, ganti answer dengan "tidak tahu"
        if is_dropped and context == "":
            answer = self.NO_CONTEXT_ANSWER

        # ================================================================
        # Step 2: Context Perturbation (shuffle/truncate)
        # ================================================================
        if context:
            context = self._apply_context_perturbation(context)

        # ================================================================
        # Step 3: Encode ke token IDs
        # ================================================================
        try:
            if context:
                # Dengan konteks: pakai split_qa_for_training
                input_ids, target_ids = self.tokenizer.split_qa_for_training(
                    context=context,
                    question=question,
                    answer=answer,
                )
            else:
                # Tanpa konteks: pakai encode_qa_no_context
                full_ids = self.tokenizer.encode_qa_no_context(
                    question=question,
                    answer=answer,
                    add_bos=True,
                    add_eos=True,
                )
                # Split di bagian "Jawaban:" - cari pattern
                input_text = f"Pertanyaan: {question} Jawaban:"
                input_ids = self.tokenizer.encode(input_text, add_bos=True, add_eos=False)
                target_ids = full_ids[len(input_ids):]
        except Exception as exc:
            # Skip sample yang error
            return None

        if not input_ids or not target_ids:
            return None

        # ================================================================
        # Step 4: Concatenate input + target, truncate/pad
        # ================================================================
        full_sequence = input_ids + target_ids

        chunk_size = self.sequence_length + 1

        if len(full_sequence) > chunk_size:
            # Truncate
            full_sequence = full_sequence[:chunk_size]
        elif len(full_sequence) < chunk_size:
            # Pad
            padding_needed = chunk_size - len(full_sequence)
            full_sequence = full_sequence + [self.pad_id] * padding_needed

        return np.array(full_sequence, dtype=np.int64)

    def _prepare_sample_progressive(
        self,
        sample: Dict,
        stage: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Siapkan sample dengan progressive difficulty.

        Stage 1 (epoch awal): single context, simple questions, no dropout
        Stage 2 (epoch tengah): multi context, complex questions, low dropout
        Stage 3 (epoch akhir): context dropout + noise injection, full difficulty

        Args:
            sample: QA sample
            stage: current stage (1, 2, atau 3)

        Return:
            (input_ids, target_ids) atau None
        """
        if stage == 1:
            # Stage 1: easy - no dropout, no perturbation
            # Temporarily disable dropout dan perturbation
            orig_dropout = self.context_dropout_rate
            orig_perturb = self.context_perturbation_rate
            self.context_dropout_rate = 0.0
            self.context_perturbation_rate = 0.0
            result = self._prepare_sample(sample)
            self.context_dropout_rate = orig_dropout
            self.context_perturbation_rate = orig_perturb
            return result

        elif stage == 2:
            # Stage 2: medium - low dropout (10%), low perturbation (15%)
            orig_dropout = self.context_dropout_rate
            orig_perturb = self.context_perturbation_rate
            self.context_dropout_rate = min(0.1, self.context_dropout_rate)
            self.context_perturbation_rate = min(0.15, self.context_perturbation_rate)
            result = self._prepare_sample(sample)
            self.context_dropout_rate = orig_dropout
            self.context_perturbation_rate = orig_perturb
            return result

        else:  # stage 3 or higher
            # Stage 3+: full difficulty dengan configured rates
            return self._prepare_sample(sample)

    # ========================================================================
    # BATCHING
    # ========================================================================

    def _get_current_stage(self, epoch: int, total_epochs: int) -> int:
        """
        Hitung current progressive stage berdasarkan epoch.

        Jika progressive_stages = 3 dan total_epochs = 30:
        - Epoch 0-9: Stage 1
        - Epoch 10-19: Stage 2
        - Epoch 20-29: Stage 3
        """
        if self.progressive_stages <= 0 or total_epochs <= 0:
            return 1  # Default ke stage 1

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
        Yield batches (inputs, targets) untuk satu epoch.

        Args:
            epoch: current epoch number (untuk progressive training)
            total_epochs: total epochs (untuk hitung stage)
            shuffle: shuffle samples

        Yields:
            (inputs, targets) tuple dengan shape [batch_size, sequence_length]
        """
        if total_epochs is None:
            total_epochs = self.epochs

        if not self.qa_samples:
            return

        # Shuffle samples
        if shuffle:
            samples = list(self.qa_samples)
            self._py_rng.shuffle(samples)
        else:
            samples = self.qa_samples

        # Get current stage
        stage = self._get_current_stage(epoch, total_epochs)

        # Prepare semua samples dulu
        prepared: List[np.ndarray] = []
        for sample in samples:
            result = self._prepare_sample_progressive(sample, stage)
            if result is not None:
                prepared.append(result)

        if not prepared:
            return

        # Yield batches
        for i in range(0, len(prepared), self.batch_size):
            chunk = prepared[i : i + self.batch_size]
            if not chunk:
                continue

            stacked = np.stack(chunk, axis=0)
            inputs = stacked[:, :-1]
            targets = stacked[:, 1:]

            yield inputs, targets

    # ========================================================================
    # PUBLIC API: FULL TRAINING
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
        Main training entry point.

        Flow:
        1. Load QA dataset (jika belum)
        2. Train untuk N epochs dengan progressive difficulty
        3. Build retriever index setelah training (jika tersedia)
        """
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        # Load QA dataset
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
            # Lanjutkan dari epoch yang sudah lewat
            new_total = start_epoch + total_epochs
            if verbose:
                print(
                    f"[Trainer] Semua {total_epochs} epoch config sudah selesai "
                    f"(checkpoint di epoch {start_epoch})."
                )
                print(
                    f"[Trainer] Melanjutkan dengan {total_epochs} epoch baru "
                    f"(epoch {start_epoch + 1} - {new_total})."
                )
            total_epochs = new_total

        if verbose:
            print("=" * 70)
            print("RAG CONTEXT-CONDITIONED TRAINING")
            print("=" * 70)
            print(f"  QA samples       : {len(self.qa_samples)}")
            print(f"  Total epochs     : {total_epochs}")
            print(f"  Batch size       : {self.batch_size}")
            print(f"  Sequence length  : {self.sequence_length}")
            print(f"  Progressive stages: {self.progressive_stages}")
            print(f"  Context dropout  : {self.context_dropout_rate:.1%}")
            print(f"  Context perturb  : {self.context_perturbation_rate:.1%}")
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

            # Reset RNG untuk epoch ini
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

                self.model.zero_grad()
                logits = self.model.forward(inputs, training=True)

                loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(
                    logits,
                    targets,
                )

                num_valid_tokens = int(metrics["num_valid_tokens"])

                if num_valid_tokens > 0:
                    self.model.backward(grad_logits)
                    step_info = self.weight_manager.step(zero_grad=True)

                    epoch_loss += float(loss) * num_valid_tokens
                    epoch_tokens += num_valid_tokens
                    epoch_correct += int(metrics["num_correct"])

                    if verbose and steps_done % self.LOG_EVERY_STEPS == 0:
                        print(
                            f"  [Stage {stage}] step={steps_done} "
                            f"global_step={global_step} "
                            f"loss={loss:.6f} "
                            f"ppl={metrics['perplexity']:.4f} "
                            f"acc={metrics['accuracy']:.4f} "
                            f"lr={step_info['learning_rate']:.2e}"
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
                    print(
                        "[Trainer] ⚠️ Tidak ada batch yang bisa diproses pada epoch ini. "
                        "Pastikan QA dataset sudah di-generate."
                    )

                epoch_report = {
                    "epoch": epoch + 1,
                    "stage": stage,
                    "loss": 0.0,
                    "perplexity": 1.0,
                    "accuracy": 0.0,
                    "num_valid_tokens": 0,
                    "num_correct": 0,
                    "steps": 0,
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
                }

                if verbose:
                    print(
                        f"[Trainer] Epoch {epoch + 1} selesai (Stage {stage}) | "
                        f"loss={avg_loss:.6f} | "
                        f"ppl={perplexity:.4f} | "
                        f"acc={accuracy:.4f} | "
                        f"tokens={epoch_tokens}"
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
                    if max_steps is not None and global_step >= int(max_steps):
                        print("[Trainer] Training dihentikan karena mencapai max_steps.")
                    else:
                        print("[Trainer] Training dihentikan.")
                break

        # ================================================================
        # Build retriever index setelah training selesai
        # ================================================================
        if _RETRIEVER_AVAILABLE:
            self._build_retriever_index(verbose=verbose)

        return history

    # ========================================================================
    # INLINE TRAINING BERBASIS STEP (dari playground)
    # ========================================================================

    def train_steps(
        self,
        steps: int,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        """Inline training untuk N steps dari playground."""
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        # Load QA dataset jika belum
        if not self.qa_samples:
            self.load_qa_dataset(verbose=verbose)

        if not self.qa_samples:
            if verbose:
                print("[Trainer] ❌ Tidak ada data untuk dilatih.")
            return []

        steps = max(1, int(steps))
        current_global_step = getattr(self.weight_manager, "global_step", 0)

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        steps_done = 0
        epochs_traversed = 0

        if verbose:
            print(f"[Trainer] Inline training dimulai: target {steps} step")

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

                self.model.zero_grad()
                logits = self.model.forward(inputs, training=True)

                loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(
                    logits, targets,
                )

                num_valid_tokens = int(metrics["num_valid_tokens"])

                if num_valid_tokens > 0:
                    self.model.backward(grad_logits)
                    step_info = self.weight_manager.step(zero_grad=True)

                    total_loss += float(loss) * num_valid_tokens
                    total_tokens += num_valid_tokens
                    total_correct += int(metrics["num_correct"])

                    if verbose:
                        print(
                            f"[Trainer] inline step={steps_done} "
                            f"loss={loss:.6f} "
                            f"ppl={metrics['perplexity']:.4f} "
                            f"acc={metrics['accuracy']:.4f} "
                            f"lr={step_info['learning_rate']:.2e}"
                        )
                else:
                    self.model.zero_grad()

                steps_done += 1
                batches_this_epoch += 1

                self.weight_manager.global_step = getattr(
                    self.weight_manager, "global_step", 0
                ) + 1

            epochs_traversed += 1

            if batches_this_epoch == 0:
                if verbose:
                    print("[Trainer] Tidak ada batch tersedia. Dataset mungkin kosong.")
                break

        if steps_done == 0:
            if verbose:
                print("[Trainer] Tidak ada batch untuk inline training.")
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
                f"[Trainer] Inline training selesai | "
                f"steps={steps_done} | "
                f"epochs_traversed={epochs_traversed} | "
                f"loss={avg_loss:.6f} | "
                f"ppl={perplexity:.4f} | "
                f"acc={accuracy:.4f}"
            )

        if save_checkpoint:
            ckpt_path = self.weight_manager.save_checkpoint()
            if verbose:
                print(f"[Trainer] Checkpoint disimpan: {ckpt_path}")

        # Build retriever index
        if _RETRIEVER_AVAILABLE:
            self._build_retriever_index(verbose=verbose)

        return [report]

    # ========================================================================
    # INLINE TRAINING BERBASIS EPOCH (dari playground)
    # ========================================================================

    def train_epochs_inline(
        self,
        epochs: int,
        min_steps: int = 0,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        """Inline training untuk N epochs dari playground."""
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        # Load QA dataset jika belum
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
                f"[Trainer] Inline epoch training dimulai: "
                f"target {epochs} epoch"
                + (f", minimal {min_steps} step" if min_steps > 0 else "")
            )
            print(f"[Trainer] QA samples: {len(self.qa_samples)}")

        history: List[dict] = []
        total_steps_done = 0

        for epoch_offset in range(epochs):
            epoch_num = current_epoch + epoch_offset
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_correct = 0
            steps_done = 0

            self._rng = np.random.default_rng(self.seed + epoch_num + 1)
            self._py_rng = random.Random(self.seed + epoch_num + 1)

            stage = self._get_current_stage(epoch_offset, epochs)

            if verbose:
                print(
                    f"[Trainer] Epoch {epoch_num + 1} dimulai "
                    f"(Stage {stage}/{self.progressive_stages})"
                )

            batch_iterator = self.iter_batches(
                epoch=epoch_offset,
                total_epochs=epochs,
                shuffle=True,
            )

            for inputs, targets in batch_iterator:
                self.model.zero_grad()
                logits = self.model.forward(inputs, training=True)

                loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(
                    logits, targets,
                )

                num_valid_tokens = int(metrics["num_valid_tokens"])

                if num_valid_tokens > 0:
                    self.model.backward(grad_logits)
                    step_info = self.weight_manager.step(zero_grad=True)

                    epoch_loss += float(loss) * num_valid_tokens
                    epoch_tokens += num_valid_tokens
                    epoch_correct += int(metrics["num_correct"])

                    if verbose and steps_done % self.LOG_EVERY_STEPS == 0:
                        print(
                            f"  [Stage {stage}] step={steps_done} "
                            f"loss={loss:.6f} "
                            f"ppl={metrics['perplexity']:.4f} "
                            f"acc={metrics['accuracy']:.4f} "
                            f"lr={step_info['learning_rate']:.2e}"
                        )
                else:
                    self.model.zero_grad()

                steps_done += 1
                total_steps_done += 1

                self.weight_manager.global_step = getattr(
                    self.weight_manager, "global_step", 0
                ) + 1

                # Check min_steps
                if min_steps > 0 and total_steps_done >= min_steps:
                    pass  # Lanjut sampai epoch selesai

            if steps_done == 0:
                if verbose:
                    print(
                        "[Trainer] ⚠️ Tidak ada batch pada epoch ini. "
                        "Dataset mungkin kosong."
                    )
                break

            avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0.0
            accuracy = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0.0
            perplexity = float(np.exp(avg_loss)) if avg_loss < 100 else float("inf")

            epoch_report = {
                "epoch": epoch_num + 1,
                "stage": stage,
                "loss": float(avg_loss),
                "perplexity": float(perplexity),
                "accuracy": float(accuracy),
                "num_valid_tokens": int(epoch_tokens),
                "num_correct": int(epoch_correct),
                "steps": int(steps_done),
            }

            if verbose:
                print(
                    f"[Trainer] Epoch {epoch_num + 1} selesai (Stage {stage}) | "
                    f"loss={avg_loss:.6f} | "
                    f"ppl={perplexity:.4f} | "
                    f"acc={accuracy:.4f} | "
                    f"tokens={epoch_tokens}"
                )

            history.append(epoch_report)

            self.weight_manager.last_epoch = epoch_num + 1

            if save_checkpoint:
                ckpt_path = self.weight_manager.save_checkpoint()
                if verbose:
                    print(f"[Trainer] Checkpoint disimpan: {ckpt_path}")

        if verbose:
            print(
                f"[Trainer] Inline epoch training selesai | "
                f"epochs={len(history)} | "
                f"total_steps={total_steps_done}"
            )

        # Build retriever index
        if _RETRIEVER_AVAILABLE:
            self._build_retriever_index(verbose=verbose)

        return history

    # ========================================================================
    # RETRIEVER INDEX BUILDING
    # ========================================================================

    def _build_retriever_index(self, verbose: bool = True) -> None:
        """
        Bangun BM25 retrieval index dari corpus mentah.

        Dipanggil setelah training selesai agar index siap untuk inference.
        """
        if not _RETRIEVER_AVAILABLE:
            if verbose:
                print("[Trainer] core/retriever.py tidak tersedia. Skip build index.")
            return

        if not self.datasets_dir.exists():
            if verbose:
                print(f"[Trainer] Datasets dir tidak ditemukan: {self.datasets_dir}")
            return

        if verbose:
            print()
            print("=" * 70)
            print("BUILDING RETRIEVAL INDEX")
            print("=" * 70)

        try:
            # Load corpus dari datasets/
            max_para_len = int(getattr(self.config.retrieval, 'max_paragraph_length', 500))
            corpus = load_corpus_from_directory(
                self.datasets_dir,
                max_paragraph_length=max_para_len,
                verbose=verbose,
            )

            if not corpus:
                if verbose:
                    print("[Trainer] ⚠️ Corpus kosong, tidak bisa build index.")
                return

            # Build retriever
            self._retriever = BM25Retriever(self.config)
            self._retriever.build_index(corpus, verbose=verbose)

            # Save index
            index_path = Path(self.config.paths.retrieval_index_path)
            self._retriever.save_index(index_path)

        except Exception as exc:
            if verbose:
                print(f"[Trainer] ❌ Gagal build retriever index: {exc}")

    def get_retriever(self) -> Optional['BM25Retriever']:
        """Return retriever instance (jika sudah dibangun)."""
        return self._retriever


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

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        datasets_dir = tmp_path / "datasets"
        output_models_dir = tmp_path / "output_models"

        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        # Buat sample corpus
        sample_file = datasets_dir / "sample.txt"
        sample_file.write_text(
            "Mobil adalah kendaraan beroda empat yang menggunakan mesin sebagai penggerak utama.\n\n"
            "Rem mobil berfungsi untuk mengurangi kecepatan dan menghentikan kendaraan.\n\n"
            "Air adalah senyawa kimia dengan rumus H2O yang penting bagi kehidupan.\n\n"
            "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan menggunakan cahaya matahari.",
            encoding="utf-8",
        )

        # Buat sample QA dataset (simulasi hasil convert_dataset.py)
        qa_dataset_path = output_models_dir / "qa_dataset.json"
        sample_qa_data = {
            "version": "1.0",
            "architecture": "rag-single-head",
            "metadata": {
                "total_samples": 4,
                "source_corpus_size": 4,
                "augmentation_factor": 1,
                "template_types": {"definition": 2, "explanation": 2},
            },
            "samples": [
                {
                    "context": "Mobil adalah kendaraan beroda empat yang menggunakan mesin sebagai penggerak utama.",
                    "question": "Apa itu mobil?",
                    "answer": "Mobil adalah kendaraan beroda empat yang menggunakan mesin sebagai penggerak utama.",
                    "source_file": "sample.txt",
                    "template_type": "definition",
                    "augmentation": "original",
                },
                {
                    "context": "Rem mobil berfungsi untuk mengurangi kecepatan dan menghentikan kendaraan.",
                    "question": "Apa fungsi rem mobil?",
                    "answer": "Rem mobil berfungsi untuk mengurangi kecepatan dan menghentikan kendaraan.",
                    "source_file": "sample.txt",
                    "template_type": "function",
                    "augmentation": "original",
                },
                {
                    "context": "Air adalah senyawa kimia dengan rumus H2O yang penting bagi kehidupan.",
                    "question": "Apa itu air?",
                    "answer": "Air adalah senyawa kimia dengan rumus H2O yang penting bagi kehidupan.",
                    "source_file": "sample.txt",
                    "template_type": "definition",
                    "augmentation": "original",
                },
                {
                    "context": "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan menggunakan cahaya matahari.",
                    "question": "Jelaskan tentang fotosintesis.",
                    "answer": "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan menggunakan cahaya matahari.",
                    "source_file": "sample.txt",
                    "template_type": "explanation",
                    "augmentation": "original",
                },
            ],
        }

        with open(qa_dataset_path, "w", encoding="utf-8") as f:
            json.dump(sample_qa_data, f, ensure_ascii=False, indent=2)

        paths = PathsConfig(
            datasets_dir=datasets_dir,
            output_models_dir=output_models_dir,
            vocab_path=tmp_path / "vocab.json",
            final_model_path=output_models_dir / "otak_model.safetensors",
            checkpoint_path=output_models_dir / "training_checkpoint.npz",
            retrieval_index_path=output_models_dir / "retrieval_index.json",
            qa_dataset_path=qa_dataset_path,
        )

        small_config = Config(
            paths=paths,
            special_tokens=SpecialTokensConfig(),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=2,
                epochs=2,
                sequence_length=32,
                dropout_rate=0.0,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
                # RAG parameters
                context_dropout_rate=0.2,
                context_perturbation_rate=0.3,
                progressive_stages=2,
                max_context_paragraphs=3,
                augmentation_factor=1,
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
                use_rope=True,
                use_weight_tying=True,
                attention_logit_cap=30.0,
                final_logit_cap=50.0,
                drop_path_rate=0.1,
            ),
            retrieval=RetrievalConfig(
                top_k=3,
                bm25_k1=1.5,
                bm25_b=0.75,
            ),
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

        # ================================================================
        # Test 1: Load QA Dataset
        # ================================================================
        print("=" * 70)
        print("Test 1: Load QA Dataset")
        print("=" * 70)

        n_loaded = trainer.load_qa_dataset(verbose=True)
        print(f"\nLoaded: {n_loaded} samples")
        print(f"First sample question: {trainer.qa_samples[0]['question']}")

        # ================================================================
        # Test 2: Context Manipulation
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 2: Context Manipulation")
        print("=" * 70)

        test_context = "Kalimat pertama. Kalimat kedua. Kalimat ketiga. Kalimat keempat."
        print(f"\nOriginal: {test_context}")

        shuffled = trainer._shuffle_sentences(test_context)
        print(f"Shuffled: {shuffled}")

        context, dropped = trainer._apply_context_dropout(test_context)
        print(f"After dropout (dropped={dropped}): '{context[:50]}...'")

        perturbed = trainer._apply_context_perturbation(test_context)
        print(f"Perturbed: {perturbed}")

        # ================================================================
        # Test 3: Sample Preparation
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 3: Sample Preparation")
        print("=" * 70)

        sample = trainer.qa_samples[0]
        result = trainer._prepare_sample(sample)
        if result is not None:
            full_seq = result
            print(f"Sequence shape: {full_seq.shape}")
            print(f"Sequence[:20]: {full_seq[:20]}")
            decoded = tokenizer.decode(full_seq.tolist(), skip_special_tokens=False)
            print(f"Decoded: {decoded[:150]}...")
        else:
            print("Failed to prepare sample")

        # ================================================================
        # Test 4: Progressive Stages
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 4: Progressive Stages")
        print("=" * 70)

        for epoch in range(20):
            stage = trainer._get_current_stage(epoch, 20)
            print(f"Epoch {epoch:2d} → Stage {stage}")

        # ================================================================
        # Test 5: Full Training (2 epochs)
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 5: Full Training (2 epochs)")
        print("=" * 70)

        history = trainer.run_epochs(
            save_checkpoints=True,
            verbose=True,
        )

        print(f"\nTotal epochs trained: {len(history)}")
        for report in history:
            print(
                f"  Epoch {report['epoch']} (Stage {report['stage']}): "
                f"loss={report['loss']:.4f}, ppl={report['perplexity']:.4f}, "
                f"acc={report['accuracy']:.4f}"
            )

        # ================================================================
        # Test 6: Retriever Index
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 6: Retriever Index")
        print("=" * 70)

        retriever = trainer.get_retriever()
        if retriever is not None:
            print(f"✅ Retriever tersedia: {retriever.total_docs} documents")

            # Test retrieval
            results = retriever.retrieve("apa itu mobil?", top_k=2)
            print(f"\nQuery: 'apa itu mobil?'")
            for i, (text, score) in enumerate(results, 1):
                print(f"  {i}. Score={score:.3f}: {text[:60]}...")
        else:
            print("⚠️ Retriever tidak tersedia (core/retriever.py tidak terinstall)")

        # ================================================================
        # Test 7: Inline Training
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 7: Inline Training (5 steps)")
        print("=" * 70)

        inline_reports = trainer.train_steps(
            steps=5,
            save_checkpoint=False,
            verbose=True,
        )

        if inline_reports:
            print(f"\nInline report: {inline_reports[0]}")

        print("\n" + "=" * 70)
        print("pipeline/trainer.py (RAG) ALL TESTS PASSED")
        print("=" * 70)
