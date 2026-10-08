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
    """

    LOG_EVERY_STEPS = 10
    SHUFFLE_BUFFER_BATCH_MULTIPLIER = 32
    MIN_SHUFFLE_BUFFER = 128

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

            # Jika min_steps sudah tercapai dan epoch sudah selesai,
            # tetap lanjutkan epoch berikutnya sampai epochs terpenuhi.
            # min_steps hanya menjamin minimal step, tidak memotong epoch.

        if verbose:
            print(
                f"[Trainer] Inline epoch training selesai | "
                f"epochs={len(history)} | "
                f"total_steps={total_steps_done}"
            )

        return history

    # ========================================================================
    # BATCHING
    # ========================================================================

    def iter_batches(
        self,
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Menghasilkan batch (inputs, targets).

        Output:
        - inputs  : [B, sequence_length]
        - targets : [B, sequence_length]
        """
        if rng is None:
            rng = np.random.default_rng()

        buffer: List[np.ndarray] = []
        buffer_limit = max(
            self.batch_size * self.SHUFFLE_BUFFER_BATCH_MULTIPLIER,
            self.MIN_SHUFFLE_BUFFER,
        )

        for example in self._iter_examples():
            buffer.append(example)

            if len(buffer) >= buffer_limit:
                yield from self._yield_batches(buffer, shuffle, rng)
                buffer.clear()

        if buffer:
            yield from self._yield_batches(buffer, shuffle, rng)
            buffer.clear()

    def _yield_batches(
        self,
        buffer: List[np.ndarray],
        shuffle: bool,
        rng: np.random.Generator,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Mengubah buffer contoh menjadi batch.
        """
        if shuffle:
            rng.shuffle(buffer)

        for i in range(0, len(buffer), self.batch_size):
            chunk = buffer[i : i + self.batch_size]

            if not chunk:
                continue

            stacked = np.stack(chunk, axis=0).astype(np.int64)

            inputs = stacked[:, :-1]
            targets = stacked[:, 1:]

            yield inputs, targets

    def _iter_examples(self) -> Iterable[np.ndarray]:
        """
        Menghasilkan contoh token sepanjang sequence_length + 1.
        """
        chunk_size = self.sequence_length + 1
        current: List[int] = []
        pad_id = int(self.tokenizer.pad_id)

        for text in self._iter_text_sequences():
            text = text.strip()

            if not text:
                continue

            ids = self.tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
            )

            if not ids:
                continue

            current.extend(ids)

            start = 0

            while len(current) - start >= chunk_size:
                yield np.array(current[start : start + chunk_size], dtype=np.int64)
                start += chunk_size

            if start > 0:
                current = current[start:]

        if len(current) > 1:
            padded = current + [pad_id] * (chunk_size - len(current))
            yield np.array(padded, dtype=np.int64)

    # ========================================================================
    # DATASET READING
    # ========================================================================

    def _iter_text_sequences(self) -> Iterable[str]:
        """
        Membaca urutan teks dari folder datasets/.
        """
        if not self.datasets_dir.exists():
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            return

        for path in sorted(self.datasets_dir.iterdir()):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()

            if suffix == ".txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        yield line

            elif suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)

                    for text in self._extract_json_strings(data):
                        yield text

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
                                yield text

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

        sample_file = datasets_dir / "sample.txt"
        sample_file.write_text(
            "\n".join(
                [
                    "mobil memiliki mesin",
                    "mobil memakai bahan bakar",
                    "kecepatan mobil tergantung mesin",
                    "apa itu mobil",
                    "mesin mobil perlu dirawat",
                ]
            ),
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

        # Test train_steps
        print("Testing train_steps...")
        result = trainer.train_steps(steps=5, verbose=True)
        print(f"Result: {result}")

        # Test train_epochs_inline
        print("\nTesting train_epochs_inline...")
        result = trainer.train_epochs_inline(epochs=2, verbose=True)
        print(f"Result: {result}")

        print("\npipeline/trainer.py test OK")
