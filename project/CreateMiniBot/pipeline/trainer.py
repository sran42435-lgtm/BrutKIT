# pipeline/trainer.py
#
# Trainer / Training Loop Engine untuk Training Pipeline Engine.
#
# Perubahan SOTA Batch 7 - Conditioned Training + Domain-Aware Batching:
# - Domain prefix ([MOBIL], [SAINS], dll.) DIPERTAHANKAN saat training
# - Model belajar mengenali tag domain sebagai anchor context
# - Domain-Aware Batching: setiap batch hanya berisi satu domain
# - Mencegah interferensi gradien lintas domain
# - Document Boundary dengan  tetap aktif
#
# Keterhubungan:
# - config.py              : hyperparameter training
# - core/tokenizer.py      : encode teks menjadi token ID
# - core/architecture.py   : model yang dilatih
# - pipeline/evaluator.py  : menghitung loss dan gradient
# - pipeline/weight_manager.py : memperbarui bobot

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
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


# Regex untuk mendeteksi domain prefix
DOMAIN_PREFIX_PATTERN = re.compile(r'^\[([A-Z]+)\]\s*')


class Trainer:
    """
    Mesin latihan utama.

    Fitur:
    - Resume dari epoch terakhir
    - Early stopping
    - Periodic checkpoint
    - Inline training dari playground (berbasis step & epoch)
    - Pembacaan berbasis blok paragraf
    - Conditioned Training (domain prefix dipertahankan)
    - Domain-Aware Batching (batch per domain)
    - Document Boundary dengan 
    """

    LOG_EVERY_STEPS = 10

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

        # State untuk resume training
        self.start_epoch = 0
        self.start_global_step = 0

        # Token IDs untuk Document Boundary
        self.pad_id = int(config.special_tokens.pad_id)
        self.bos_id = int(config.special_tokens.bos_id)
        self.eos_id = int(config.special_tokens.eos_id)
        self.endoftext_id = int(config.special_tokens.endoftext_id)

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
    # DOMAIN PREFIX HANDLING
    # ========================================================================

    def _extract_domain_from_text(self, text: str) -> str:
        """
        Mengekstrak domain dari teks.

        Contoh:
            "[MOBIL] Mobil adalah kendaraan." -> "MOBIL"
            "Teks tanpa prefix" -> "_GENERAL_"
        """
        match = DOMAIN_PREFIX_PATTERN.match(text)
        if match:
            return match.group(1)
        return "_GENERAL_"

    # ========================================================================
    # PUBLIC API: FULL TRAINING (MODE PIPELINE)
    # ========================================================================

    def run_epochs(
        self,
        max_steps: Optional[int] = None,
        save_checkpoints: bool = False,
        verbose: bool = True,
        start_epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> List[dict]:
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        if start_epoch is None:
            start_epoch = self.start_epoch

        if global_step is None:
            global_step = self.start_global_step

        if start_epoch >= self.epochs:
            total_epochs = start_epoch + self.epochs
            if verbose:
                print(
                    f"[Trainer] Semua {self.epochs} epoch config sudah selesai "
                    f"(checkpoint di epoch {start_epoch})."
                )
                print(
                    f"[Trainer] Melanjutkan dengan {self.epochs} epoch baru "
                    f"(epoch {start_epoch + 1} - {total_epochs})."
                )
        else:
            total_epochs = self.epochs

        history: List[dict] = []
        stop_training = False

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(start_epoch, total_epochs):
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_correct = 0
            steps_done = 0

            rng = np.random.default_rng(self.seed + epoch)

            if verbose:
                print(f"[Trainer] Epoch {epoch + 1}/{total_epochs} dimulai")

            batch_iterator = self.iter_batches(shuffle=True, rng=rng)

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
                            f"[Trainer] epoch={epoch + 1} "
                            f"step={steps_done} "
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
                            print(f"[Trainer] Periodic checkpoint disimpan: {ckpt_path}")
                else:
                    self.model.zero_grad()

                steps_done += 1
                global_step += 1

            if steps_done == 0:
                if verbose:
                    print(
                        "[Trainer] Tidak ada batch yang bisa diproses pada epoch ini. "
                        "Pastikan folder datasets/ berisi file .txt atau .json."
                    )

                epoch_report = {
                    "epoch": epoch + 1,
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
                    "loss": float(avg_loss),
                    "perplexity": float(perplexity),
                    "accuracy": float(accuracy),
                    "num_valid_tokens": int(epoch_tokens),
                    "num_correct": int(epoch_correct),
                    "steps": int(steps_done),
                }

                if verbose:
                    print(
                        f"[Trainer] Epoch {epoch + 1} selesai | "
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

        return history

    # ========================================================================
    # PUBLIC API: INLINE TRAINING BERBASIS STEP (dari playground)
    # ========================================================================

    def train_steps(
        self,
        steps: int,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        steps = max(1, int(steps))

        current_global_step = getattr(self.weight_manager, "global_step", 0)
        rng = np.random.default_rng(self.seed + current_global_step + 1)

        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        steps_done = 0
        epochs_traversed = 0

        if verbose:
            print(f"[Trainer] Inline training dimulai: target {steps} step")

        while steps_done < steps:
            batch_iterator = self.iter_batches(shuffle=True, rng=rng)
            batches_this_epoch = 0

            for inputs, targets in batch_iterator:
                if steps_done >= steps:
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

            rng = np.random.default_rng(
                self.seed + current_global_step + epochs_traversed
            )

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

        return [report]

    # ========================================================================
    # PUBLIC API: INLINE TRAINING BERBASIS EPOCH (dari playground)
    # ========================================================================

    def train_epochs_inline(
        self,
        epochs: int,
        min_steps: int = 0,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> List[dict]:
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        epochs = max(1, int(epochs))
        min_steps = max(0, int(min_steps))

        current_global_step = getattr(self.weight_manager, "global_step", 0)
        current_epoch = getattr(self.weight_manager, "last_epoch", 0)

        history: List[dict] = []
        total_steps_done = 0

        if verbose:
            print(
                f"[Trainer] Inline epoch training dimulai: "
                f"target {epochs} epoch"
                + (f", minimal {min_steps} step" if min_steps > 0 else "")
            )

        for epoch_offset in range(epochs):
            epoch_num = current_epoch + epoch_offset
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_correct = 0
            steps_done = 0

            rng = np.random.default_rng(self.seed + epoch_num + 1)

            if verbose:
                print(f"[Trainer] Epoch {epoch_num + 1} dimulai")

            batch_iterator = self.iter_batches(shuffle=True, rng=rng)

            for inputs, targets in batch_iterator:
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
                            f"[Trainer] epoch={epoch_num + 1} "
                            f"step={steps_done} "
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

            if steps_done == 0:
                if verbose:
                    print(
                        "[Trainer] Tidak ada batch pada epoch ini. "
                        "Dataset mungkin kosong."
                    )
                break

            avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0.0
            accuracy = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0.0
            perplexity = float(np.exp(avg_loss)) if avg_loss < 100 else float("inf")

            epoch_report = {
                "epoch": epoch_num + 1,
                "loss": float(avg_loss),
                "perplexity": float(perplexity),
                "accuracy": float(accuracy),
                "num_valid_tokens": int(epoch_tokens),
                "num_correct": int(epoch_correct),
                "steps": int(steps_done),
            }

            if verbose:
                print(
                    f"[Trainer] Epoch {epoch_num + 1} selesai | "
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

        return history

    # ========================================================================
    # BATCHING - CONDITIONED TRAINING + DOMAIN-AWARE BATCHING
    # ========================================================================

    def iter_batches(
        self,
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Menghasilkan batch (inputs, targets).

        CONDITIONED TRAINING + DOMAIN-AWARE BATCHING:
        - Domain prefix DIPERTAHANKAN (tidak dihapus)
        - Paragraf dikelompokkan berdasarkan domain
        - Setiap batch hanya berisi paragraf dari domain yang sama
        - Urutan batch di-shuffle (bukan paragraf individual)
        - Document Boundary dengan BOS/EOS/endoftext tetap aktif
        """
        if rng is None:
            rng = np.random.default_rng()

        # ================================================================
        # LOAD PARAGRAPHS PER DOMAIN
        # ================================================================
        paragraphs_by_domain: Dict[str, List[str]] = self._load_paragraphs_by_domain(rng=rng)

        if not paragraphs_by_domain:
            return

        # ================================================================
        # DOMAIN-AWARE BATCHING
        #
        # Strategi:
        # 1. Untuk setiap domain, buat batch-batch dari paragraf domain tersebut
        # 2. Kumpulkan semua batch dari semua domain
        # 3. Shuffle urutan batch (bukan paragraf individual)
        # 4. Yield batch satu per satu
        #
        # Hasil: Setiap batch hanya berisi satu domain, tapi urutan batch acak
        # ================================================================
        all_batches: List[Tuple[np.ndarray, np.ndarray]] = []

        for domain, paragraphs in paragraphs_by_domain.items():
            if not paragraphs:
                continue

            # Shuffle paragraf dalam domain ini
            if shuffle and len(paragraphs) > 1:
                rng.shuffle(paragraphs)

            # Generate examples untuk semua paragraf di domain ini
            domain_examples: List[np.ndarray] = []
            for paragraph in paragraphs:
                for example in self._paragraph_to_examples(paragraph):
                    domain_examples.append(example)

            if not domain_examples:
                continue

            # Buat batch-batch dari examples domain ini
            for i in range(0, len(domain_examples), self.batch_size):
                chunk = domain_examples[i : i + self.batch_size]

                if not chunk:
                    continue

                stacked = np.stack(chunk, axis=0).astype(np.int64)

                inputs = stacked[:, :-1]
                targets = stacked[:, 1:]

                all_batches.append((inputs, targets))

        if not all_batches:
            return

        # Shuffle urutan batch (domain-aware shuffle)
        if shuffle and len(all_batches) > 1:
            rng.shuffle(all_batches)

        # Yield batch satu per satu
        for batch in all_batches:
            yield batch

    def _paragraph_to_examples(self, paragraph: str) -> Iterable[np.ndarray]:
        """
        Mengubah satu blok paragraf menjadi contoh token terisolasi.

        CONDITIONED TRAINING + DOCUMENT BOUNDARY:
        - Domain prefix DIPERTAHANKAN (TIDAK dihapus)
        - Model belajar mengenali tag [MOBIL], [SAINS], dll sebagai anchor
        - Paragraf pendek: [BOS] + teks_dengan_prefix + [EOS] +  + [PAD]
        - Paragraf panjang: dipotong dengan BOS/EOS/endoftext di setiap potongan
        """
        # ================================================================
        # CONDITIONED TRAINING: TIDAK MENGHAPUS DOMAIN PREFIX
        # Biarkan teks asli dengan prefix [MOBIL], [SAINS], dll
        # Model akan belajar mengenali tag domain sebagai anchor context
        # ================================================================
        if not paragraph.strip():
            return

        chunk_size = self.sequence_length + 1

        # Encode paragraf tanpa BOS/EOS (kita tambahkan sendiri)
        # Prefix [MOBIL] tetap ikut di-encode
        ids = self.tokenizer.encode(paragraph, add_bos=False, add_eos=False)

        if not ids:
            return

        # ============================================================
        # KASUS 1: Paragraf muat dalam satu sequence
        # [BOS] + teks_dengan_prefix + [EOS] +  + [PAD]
        # ============================================================
        if len(ids) + 3 <= chunk_size:
            sequence = [self.bos_id] + ids + [self.eos_id, self.endoftext_id]
            padding_needed = chunk_size - len(sequence)
            sequence = sequence + [self.pad_id] * padding_needed
            yield np.array(sequence, dtype=np.int64)
            return

        # ============================================================
        # KASUS 2: Paragraf terlalu panjang
        # Setiap potongan: [BOS] + chunk + [EOS] +  + [PAD]
        # ============================================================
        content_size = chunk_size - 3  # Ruang untuk BOS, EOS, endoftext

        start = 0
        while start < len(ids):
            chunk = ids[start : start + content_size]

            sequence = [self.bos_id] + chunk + [self.eos_id, self.endoftext_id]

            if len(sequence) < chunk_size:
                padding_needed = chunk_size - len(sequence)
                sequence = sequence + [self.pad_id] * padding_needed

            yield np.array(sequence, dtype=np.int64)

            start += content_size

    # ========================================================================
    # DATASET READING - DOMAIN-AWARE LOADING
    # ========================================================================

    def _load_paragraphs_by_domain(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, List[str]]:
        """
        Membaca seluruh dataset dan mengelompokkan paragraf berdasarkan domain.

        Return:
        - Dict[str, List[str]]: domain -> list of paragraphs
          Contoh: {"MOBIL": [...], "SAINS": [...], "_GENERAL_": [...]}
        """
        if rng is None:
            rng = np.random.default_rng()

        if not self.datasets_dir.exists():
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            return {}

        file_paths: List[Path] = []

        for path in self.datasets_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".txt", ".json"}:
                file_paths.append(path)

        if not file_paths:
            return {}

        # Shuffle urutan file
        rng.shuffle(file_paths)

        # Kelompokkan paragraf berdasarkan domain
        paragraphs_by_domain: Dict[str, List[str]] = defaultdict(list)

        for path in file_paths:
            suffix = path.suffix.lower()

            if suffix == ".txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                paragraphs = content.split("\n\n")

                for para in paragraphs:
                    para = para.strip()
                    para = para.replace("\n", " ")
                    if para:
                        domain = self._extract_domain_from_text(para)
                        paragraphs_by_domain[domain].append(para)

            elif suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)

                    for text in self._extract_json_strings(data):
                        text = text.strip()
                        if text:
                            domain = self._extract_domain_from_text(text)
                            paragraphs_by_domain[domain].append(text)

                except json.JSONDecodeError:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            for text in self._extract_json_strings(obj):
                                text = text.strip()
                                if text:
                                    domain = self._extract_domain_from_text(text)
                                    paragraphs_by_domain[domain].append(text)

        return dict(paragraphs_by_domain)

    def _extract_json_strings(self, node) -> Iterable[str]:
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from self._extract_json_strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from self._extract_json_strings(value)


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

        # Dataset dengan domain prefix
        sample_file = datasets_dir / "sample.txt"
        sample_file.write_text(
            "[MOBIL] Mobil adalah kendaraan beroda empat. "
            "Mobil menggunakan mesin sebagai penggerak utama.\n\n"
            "[SAINS] Air adalah senyawa kimia dengan rumus H2O. "
            "Air merupakan kebutuhan mendasar bagi makhluk hidup.\n\n"
            "[MOBIL] Rem mobil berfungsi untuk menghentikan kendaraan. "
            "Keselamatan berkendara sangat penting.",
            encoding="utf-8",
        )

        paths = PathsConfig(
            datasets_dir=datasets_dir,
            output_models_dir=output_models_dir,
            vocab_path=tmp_path / "vocab.json",
            final_model_path=output_models_dir / "otak_model.safetensors",
            checkpoint_path=output_models_dir / "training_checkpoint.npz",
        )

        small_config = Config(
            paths=paths,
            special_tokens=SpecialTokensConfig(),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=2,
                epochs=1,
                sequence_length=16,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
            ),
            model=ModelConfig(
                vocab_size=128,
                embedding_dim=32,
                num_attention_heads=4,
                num_layers=1,
                dropout_rate=0.0,
                max_position_embeddings=32,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=64,
                train_dtype="float32",
                export_dtype="float32",
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

        # Test domain extraction
        print("Testing domain extraction...")
        test_texts = [
            "[MOBIL] Mobil adalah kendaraan.",
            "[SAINS] Air adalah H2O.",
            "Teks tanpa prefix.",
        ]
        for text in test_texts:
            domain = trainer._extract_domain_from_text(text)
            print(f"  '{text}' -> domain: {domain}")

        # Test paragraph loading by domain
        print("\nTesting paragraph loading by domain...")
        rng = np.random.default_rng(1337)
        paragraphs_by_domain = trainer._load_paragraphs_by_domain(rng=rng)
        print(f"  Domains found: {list(paragraphs_by_domain.keys())}")
        for domain, paragraphs in paragraphs_by_domain.items():
            print(f"  [{domain}]: {len(paragraphs)} paragraphs")
            for i, p in enumerate(paragraphs):
                print(f"    {i+1}. {p[:50]}...")

        # Test conditioned training (domain prefix NOT stripped)
        print("\nTesting conditioned training (domain prefix preserved)...")
        test_para = "[MOBIL] Mobil adalah kendaraan beroda empat."
        examples = list(trainer._paragraph_to_examples(test_para))
        print(f"  Paragraph: {test_para}")
        print(f"  Examples generated: {len(examples)}")
        if examples:
            ex = examples[0]
            tokens = tokenizer.get_token_pieces(ex.tolist())
            print(f"  First example tokens: {tokens[:10]}...")
            # Check if [MOBIL] or its tokens are in the example
            has_mobil = any("MOBIL" in t or "mobil" in t.lower() for t in tokens)
            print(f"  Domain prefix preserved in tokens: {has_mobil}")

        # Test domain-aware batching
        print("\nTesting domain-aware batching...")
        batch_count = 0
        for inputs, targets in trainer.iter_batches(shuffle=True, rng=rng):
            batch_count += 1
            if batch_count <= 3:
                print(f"  Batch {batch_count}: shape={inputs.shape}")
        print(f"  Total batches: {batch_count}")

        print("\npipeline/trainer.py test OK")
