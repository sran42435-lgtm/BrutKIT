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
# Memastikan project root ada di sys.path, sehingga file di dalam folder
# pipeline/ tetap bisa meng-import config.py meskipun dijalankan langsung:
#   python pipeline/trainer.py
# atau:
#   cd pipeline && python trainer.py

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

        if self.batch_size <= 0:
            raise ValueError("batch_size harus lebih besar dari 0.")

        if self.sequence_length <= 0:
            raise ValueError("sequence_length harus lebih besar dari 0.")

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def run_epochs(
        self,
        max_steps: Optional[int] = None,
        save_checkpoints: bool = False,
        verbose: bool = True,
    ) -> List[dict]:
        """
        Menjalankan seluruh epoch training.

        Parameter:
        - max_steps: batasi jumlah step global (berguna untuk testing)
        - save_checkpoints: simpan checkpoint setiap akhir epoch
        - verbose: cetak log training

        Return:
        - history list report epoch
        """
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        history: List[dict] = []
        global_step = 0
        stop_training = False

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_correct = 0
            steps_done = 0

            rng = np.random.default_rng(self.seed + epoch)

            if verbose:
                print(f"[Trainer] Epoch {epoch + 1}/{self.epochs} dimulai")

            batch_iterator = self.iter_batches(shuffle=True, rng=rng)

            for inputs, targets in batch_iterator:
                if max_steps is not None and global_step >= int(max_steps):
                    stop_training = True
                    break

                # ------------------------------------------------------------
                # 1. Kosongkan gradient lama
                # ------------------------------------------------------------
                self.model.zero_grad()

                # ------------------------------------------------------------
                # 2. Forward pass
                # ------------------------------------------------------------
                logits = self.model.forward(inputs, training=True)

                # ------------------------------------------------------------
                # 3. Hitung loss dan gradient terhadap logits
                # ------------------------------------------------------------
                loss, grad_logits, metrics = self.evaluator.compute_loss_and_grad(
                    logits,
                    targets,
                )

                num_valid_tokens = int(metrics["num_valid_tokens"])

                # ------------------------------------------------------------
                # 4. Jika ada token valid, backward + update bobot
                # ------------------------------------------------------------
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
                            f"grad_norm={step_info['grad_norm']:.6f}"
                        )
                else:
                    # Tidak ada token valid, misalnya semua target adalah [PAD].
                    self.model.zero_grad()

                steps_done += 1
                global_step += 1

            # ----------------------------------------------------------------
            # Rekap epoch
            # ----------------------------------------------------------------
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

            if save_checkpoints:
                ckpt_path = self.weight_manager.save_checkpoint()
                if verbose:
                    print(f"[Trainer] Checkpoint disimpan: {ckpt_path}")

            if stop_training:
                if verbose:
                    print("[Trainer] Training dihentikan karena mencapai max_steps.")
                break

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

        Contoh dengan panjang sequence_length + 1 nanti dipotong menjadi:
        - input  = contoh[:-1]
        - target = contoh[1:]
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

        # Sisa token terakhir, jika masih cukup bermakna, padding hingga
        # mencapai chunk_size.
        if len(current) > 1:
            padded = current + [pad_id] * (chunk_size - len(current))
            yield np.array(padded, dtype=np.int64)

    # ========================================================================
    # DATASET READING
    # ========================================================================

    def _iter_text_sequences(self) -> Iterable[str]:
        """
        Membaca urutan teks dari folder datasets/.

        Format yang didukung:
        - .txt
        - .json
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
                    # Fallback: anggap sebagai JSON Lines
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

        history = trainer.run_epochs(
            max_steps=2,
            save_checkpoints=True,
            verbose=True,
        )

        print("pipeline/trainer.py test OK")
        print(f"History: {history}")
