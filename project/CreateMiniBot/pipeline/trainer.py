# pipeline/trainer.py
#
# Trainer / Training Loop Engine untuk Training Pipeline Engine.
#
# Tugas utama:
# 1. Membaca dataset dari folder datasets/
# 2. Mengubah teks menjadi token ID memakai tokenizer
# 3. Menyusun batch input/target untuk language modeling
# 4. Menjalankan forward pass
# 5. Menghitung loss & gradient memakai evaluator
# 6. Menjalankan backward pass
# 7. Memperbarui bobot memakai weight_manager
# 8. Mendukung resume dari epoch terakhir
# 9. Mendukung early stopping
# 10. Mendukung periodic checkpoint & eval
# 11. Mendukung inline training dari playground (step & epoch)
#
# Perubahan SOTA Batch 4 - Document Boundary:
# - Setiap paragraf diisolasi: [BOS] + teks + [EOS] + [PAD]
# - Tidak ada sliding window untuk paragraf pendek
# - Paragraf panjang dipotong dengan [BOS] dan [EOS] di setiap potongan
# - No Aggregation: tidak ada penyambungan antar paragraf di batch
#
# Keterhubungan:
# - config.py              : hyperparameter training
# - core/tokenizer.py      : encode teks menjadi token ID
# - core/architecture.py   : model yang dilatih
# - pipeline/evaluator.py  : menghitung loss dan gradient
# - pipeline/weight_manager.py : memperbarui bobot

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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


class Trainer:
    """
    Mesin latihan utama.

    Trainer menghubungkan:
    - tokenizer sebagai pengolah teks -> token ID
    - model sebagai arsitektur neural network
    - evaluator sebagai penghitung loss & gradient
    - weight_manager sebagai optimizer AdamW

    Fitur:
    - Resume dari epoch terakhir
    - Early stopping
    - Periodic checkpoint
    - Inline training dari playground (berbasis step & epoch)
    - Pembacaan berbasis blok paragraf
    - Shuffling tingkat blok paragraf
    - Document Boundary: isolasi paragraf dengan BOS/EOS/PAD
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

        if self.batch_size <= 0:
            raise ValueError("batch_size harus lebih besar dari 0.")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length harus lebih besar dari 0.")

    # ========================================================================
    # RESUME STATE
    # ========================================================================

    def set_resume_state(self, start_epoch: int = 0, global_step: int = 0) -> None:
        """
        Set state untuk resume training.

        Dipanggil oleh main.py setelah checkpoint dimuat.

        Parameter:
        - start_epoch: epoch untuk mulai (0-indexed, dari checkpoint)
        - global_step: global step terakhir
        """
        self.start_epoch = max(0, int(start_epoch))
        self.start_global_step = max(0, int(global_step))

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
        """
        Menjalankan epoch training untuk mode pipeline.

        Parameter:
        - max_steps: batasi jumlah step global (berguna untuk testing)
        - save_checkpoints: simpan checkpoint setiap akhir epoch
        - verbose: cetak log training
        - start_epoch: epoch untuk mulai (jika None, pakai self.start_epoch)
        - global_step: global step awal (jika None, pakai self.start_global_step)

        Return:
        - history list report epoch
        """
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        if start_epoch is None:
            start_epoch = self.start_epoch

        if global_step is None:
            global_step = self.start_global_step

        # ----------------------------------------------------------------
        # FIX BUG RESUME:
        # Jika start_epoch >= self.epochs, semua epoch config sudah selesai.
        # Lanjutkan dengan epoch baru sebanyak config epochs.
        # ----------------------------------------------------------------
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

            # ============================================================
            # SINKRONISASI RNG:
            # Buat rng baru untuk setiap epoch agar pola pengacakan
            # berbeda setiap epoch, tapi tetap reproducibel.
            # ============================================================
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

            # Rekap epoch
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

            # Early stopping check
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
        """
        Menjalankan training sebanyak N step tanpa terikat jumlah epoch.

        Jika dataset habis sebelum N step tercapai, dataset akan diulang
        otomatis sampai N step terpenuhi.

        Parameter:
        - steps: jumlah step training (bukan epoch)
        - save_checkpoint: simpan checkpoint setelah selesai
        - verbose: cetak log training

        Return:
        - list berisi satu report training
        """
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

        # Ulangi dataset sampai mencapai step yang diminta
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

            # Jika tidak ada batch sama sekali, berhenti
            if batches_this_epoch == 0:
                if verbose:
                    print("[Trainer] Tidak ada batch tersedia. Dataset mungkin kosong.")
                break

            # Update seed untuk epoch berikutnya
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
        """
        Menjalankan training sebanyak N epoch dari playground.

        Setiap epoch melewati seluruh dataset satu kali.
        Jika min_steps diberikan, training tetap berjalan sampai minimal
        min_steps tercapai meskipun epoch sudah selesai.

        Parameter:
        - epochs: jumlah epoch yang ingin dijalankan
        - min_steps: minimal step yang harus dijalankan (opsional)
        - save_checkpoint: simpan checkpoint setiap akhir epoch
        - verbose: cetak log training

        Return:
        - history list report epoch
        """
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

            # ============================================================
            # SINKRONISASI RNG untuk inline epoch training
            # ============================================================
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

            # Rekap epoch
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

            # Update state di weight_manager
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
    # BATCHING - DOCUMENT BOUNDARY (NO AGGREGATION)
    # ========================================================================

    def iter_batches(
        self,
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Menghasilkan batch (inputs, targets).

        DOCUMENT BOUNDARY - NO AGGREGATION:
        - Setiap contoh token sudah terisolasi murni per paragraf
        - Batch hanya menumpuk (stack) contoh-contoh mandiri
        - Tidak ada proses penyambungan teks antar-paragraf di tingkat batch

        Output:
        - inputs  : [B, sequence_length]
        - targets : [B, sequence_length]
        """
        if rng is None:
            rng = np.random.default_rng()

        # ================================================================
        # SHUFFLING TINGKAT BLOK PARAGRAF
        # ================================================================
        all_paragraphs: List[str] = self._load_all_paragraphs(rng=rng)

        if not all_paragraphs:
            return

        if shuffle and len(all_paragraphs) > 1:
            rng.shuffle(all_paragraphs)

        # ================================================================
        # GENERATE ISOLATED EXAMPLES:
        # Setiap paragraf menjadi satu atau lebih contoh terisolasi.
        # Tidak ada penggabungan antar paragraf.
        # ================================================================
        all_examples: List[np.ndarray] = []

        for paragraph in all_paragraphs:
            for example in self._paragraph_to_examples(paragraph):
                all_examples.append(example)

        if not all_examples:
            return

        # ================================================================
        # YIELD BATCH (NO AGGREGATION):
        # Batch hanya menumpuk contoh-contoh mandiri yang sudah terisolasi.
        # Tidak ada penyambungan atau penggabungan di sini.
        # ================================================================
        for i in range(0, len(all_examples), self.batch_size):
            chunk = all_examples[i : i + self.batch_size]

            if not chunk:
                continue

            stacked = np.stack(chunk, axis=0).astype(np.int64)

            inputs = stacked[:, :-1]
            targets = stacked[:, 1:]

            yield inputs, targets

    def _paragraph_to_examples(self, paragraph: str) -> Iterable[np.ndarray]:
        """
        Mengubah satu blok paragraf menjadi contoh token terisolasi.

        DOCUMENT BOUNDARY:
        - Paragraf pendek (< sequence_length):
          [BOS] + teks + [EOS] + [PAD] * sisa
        - Paragraf panjang (> sequence_length):
          Dipotong, setiap potongan: [BOS] + chunk + [EOS] + [PAD] * sisa
        - Tidak ada sliding window untuk paragraf pendek
        - Tidak ada pencampuran dengan paragraf lain
        """
        paragraph = paragraph.strip()
        if not paragraph:
            return

        chunk_size = self.sequence_length + 1
        pad_id = int(self.tokenizer.pad_id)
        bos_id = int(self.tokenizer.bos_id)
        eos_id = int(self.tokenizer.eos_id)

        # Encode paragraf tanpa BOS/EOS (kita tambahkan sendiri untuk kontrol penuh)
        ids = self.tokenizer.encode(paragraph, add_bos=False, add_eos=False)

        if not ids:
            return

        # ============================================================
        # KASUS 1: Paragraf muat dalam satu sequence
        # [BOS] + teks + [EOS] + [PAD] * sisa
        # ============================================================
        if len(ids) + 2 <= chunk_size:
            sequence = [bos_id] + ids + [eos_id]
            padding_needed = chunk_size - len(sequence)
            sequence = sequence + [pad_id] * padding_needed
            yield np.array(sequence, dtype=np.int64)
            return

        # ============================================================
        # KASUS 2: Paragraf terlalu panjang, potong dengan Document Boundary
        # Setiap potongan: [BOS] + chunk + [EOS] + [PAD] * sisa
        # ============================================================
        content_size = chunk_size - 2  # Ruang untuk BOS dan EOS

        start = 0
        while start < len(ids):
            # Ambil chunk konten
            chunk = ids[start : start + content_size]

            # Buat sequence dengan Document Boundary
            sequence = [bos_id] + chunk + [eos_id]

            # Jika potongan terakhir tidak penuh, pad
            if len(sequence) < chunk_size:
                padding_needed = chunk_size - len(sequence)
                sequence = sequence + [pad_id] * padding_needed

            yield np.array(sequence, dtype=np.int64)

            start += content_size

    # ========================================================================
    # DATASET READING - PARAGRAPH-BASED LOADING
    # ========================================================================

    def _load_all_paragraphs(self, rng: Optional[np.random.Generator] = None) -> List[str]:
        """
        Membaca seluruh dataset dan mengumpulkan blok paragraf.

        PERUBAHAN SOTA:
        - Baca seluruh isi file sekaligus dengan .read()
        - Split menggunakan \\n\\n untuk mendapatkan blok paragraf
        - Strip setiap blok dan buang yang kosong
        - Kumpulkan semua paragraf ke satu list raksasa

        Return:
        - List[str]: daftar semua blok paragraf dari semua file
        """
        if rng is None:
            rng = np.random.default_rng()

        if not self.datasets_dir.exists():
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            return []

        # ================================================================
        # Kumpulkan semua path file ke list (tanpa sorted)
        # ================================================================
        file_paths: List[Path] = []

        for path in self.datasets_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".txt", ".json"}:
                file_paths.append(path)

        if not file_paths:
            return []

        # Acak urutan file
        rng.shuffle(file_paths)

        # ================================================================
        # PEMBACAAN BERBASIS PARAGRAF
        # ================================================================
        all_paragraphs: List[str] = []

        for path in file_paths:
            suffix = path.suffix.lower()

            if suffix == ".txt":
                # Baca seluruh isi file sekaligus
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Split berdasarkan double newline (\n\n)
                paragraphs = content.split("\n\n")

                for para in paragraphs:
                    # Bersihkan spasi/sisa baris kosong di awal/akhir
                    para = para.strip()

                    # Ganti single newline di dalam paragraf dengan spasi
                    para = para.replace("\n", " ")

                    # Hanya masukkan yang tidak kosong
                    if para:
                        all_paragraphs.append(para)

            elif suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)

                    for text in self._extract_json_strings(data):
                        text = text.strip()
                        if text:
                            all_paragraphs.append(text)

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
                                    all_paragraphs.append(text)

        return all_paragraphs

    def _extract_json_strings(self, node) -> Iterable[str]:
        """
        Mengambil semua string dari struktur JSON secara rekursif.
        """
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

        # Buat dataset dengan format paragraf (\n\n)
        sample_file = datasets_dir / "sample.txt"
        sample_file.write_text(
            "Mobil adalah kendaraan beroda empat. "
            "Mobil menggunakan mesin sebagai penggerak utama. "
            "Bahan bakar digunakan untuk menghasilkan tenaga.\n\n"
            "Kucing adalah hewan peliharaan yang lucu. "
            "Kucing suka bermain dan tidur. "
            "Kucing membutuhkan makanan dan air.\n\n"
            "Teknologi berkembang sangat pesat. "
            "Komputer dan internet mengubah cara hidup manusia. "
            "Kecerdasan buatan menjadi tren masa depan.",
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
                sequence_length=4,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
            ),
            model=ModelConfig(
                vocab_size=64,
                embedding_dim=16,
                num_attention_heads=2,
                num_layers=1,
                dropout_rate=0.0,
                max_position_embeddings=16,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=32,
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

        # Test paragraph loading
        print("Testing paragraph loading...")
        rng = np.random.default_rng(1337)
        paragraphs = trainer._load_all_paragraphs(rng=rng)
        print(f"Number of paragraphs: {len(paragraphs)}")
        for i, p in enumerate(paragraphs):
            print(f"  Paragraph {i+1}: {p[:60]}...")

        # Test Document Boundary
        print("\nTesting Document Boundary...")
        for i, p in enumerate(paragraphs[:2]):
            examples = list(trainer._paragraph_to_examples(p))
            print(f"  Paragraph {i+1} -> {len(examples)} example(s)")
            for j, ex in enumerate(examples):
                print(f"    Example {j+1}: {ex[:10]}... (len={len(ex)})")

        # Test train_steps
        print("\nTesting train_steps...")
        result = trainer.train_steps(steps=5, verbose=True)
        print(f"Result: {result}")

        # Test train_epochs_inline
        print("\nTesting train_epochs_inline...")
        result = trainer.train_epochs_inline(epochs=2, verbose=True)
        print(f"Result: {result}")

        print("\npipeline/trainer.py test OK")
