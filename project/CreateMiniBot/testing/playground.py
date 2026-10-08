# testing/playground.py
#
# Playground / Inference Testing Engine.
#
# Perintah interactive (slash commands):
#   /help                  : tampilkan bantuan
#   /training              : latih model (step default dari config)
#   /training 100          : latih model 100 step
#   /training --steps 100  : latih model 100 step
#   /training --epochs 10  : latih model 10 epoch
#   /training --epochs 10 --steps 50 : latih 10 epoch, minimal 50 step
#   /status                : tampilkan status model
#   /exit                  : keluar
#
# Semua input tanpa prefix / dianggap sebagai prompt inference.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

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
        "testing/playground.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config


class Playground:
    """
    Tempat testing pertanyaan / inference.

    Perintah interactive (slash commands):
        /help                  : tampilkan bantuan
        /training              : latih model (step default)
        /training 100          : latih model 100 step
        /training --steps 100  : latih model 100 step
        /training --epochs 10  : latih model 10 epoch
        /training --epochs 10 --steps 50 : latih 10 epoch, minimal 50 step
        /status                : tampilkan status model
        /exit                  : keluar
    """

    DEFAULT_MAX_NEW_TOKENS = 48
    DEFAULT_TEMPERATURE = 0.75
    DEFAULT_TOP_K = 30
    DEFAULT_TOP_P = 0.92

    DEFAULT_REPETITION_PENALTY = 1.25
    DEFAULT_REPETITION_WINDOW = 64

    DEFAULT_MIN_NEW_TOKENS = 8

    # Default step untuk /training tanpa argumen
    DEFAULT_INLINE_TRAINING_STEPS = 10

    def __init__(self, model, tokenizer, config: Config = CONFIG, trainer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = trainer

        self.rng = np.random.default_rng(int(config.training.seed))

        self.max_context = int(config.model.max_position_embeddings)
        self.pad_id = int(config.special_tokens.pad_id)
        self.unk_id = int(config.special_tokens.unk_id)
        self.bos_id = int(config.special_tokens.bos_id)
        self.eos_id = int(config.special_tokens.eos_id)

        self._forbidden_cache: Optional[List[int]] = None

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def test_prompt(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        repetition_window: int = DEFAULT_REPETITION_WINDOW,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        verbose: bool = True,
    ) -> str:
        """
        Uji satu prompt dan cetak hasilnya.
        """
        answer, generated_ids = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            min_new_tokens=min_new_tokens,
        )

        if verbose:
            print("-" * 60)
            print("Prompt  :", prompt)
            print("Jawaban :", answer if answer else "(kosong)")
            print("Tokens  :", len(generated_ids))
            print("-" * 60)

        return answer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        repetition_window: int = DEFAULT_REPETITION_WINDOW,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    ) -> tuple[str, List[int]]:
        """
        Membuat jawaban/lanjutan teks dari prompt.
        """
        self._ensure_tokenizer_ready()

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )

        if not prompt_ids:
            prompt_ids = [self.bos_id]

        current_ids = list(prompt_ids)
        forbidden_ids = self._get_forbidden_ids()

        max_new_tokens = max(0, int(max_new_tokens))
        min_new_tokens = max(0, int(min_new_tokens))

        for step in range(max_new_tokens):
            context = current_ids[-self.max_context :]
            input_ids = np.array([context], dtype=np.int64)

            logits = self.model.forward(input_ids, training=False)
            next_logits = np.asarray(logits[0, -1, :], dtype=np.float32)

            step_forbidden = forbidden_ids

            if step < min_new_tokens:
                step_forbidden = list(forbidden_ids)
                if self.eos_id not in step_forbidden:
                    step_forbidden.append(self.eos_id)

            generated_so_far = current_ids[len(prompt_ids) :]

            if repetition_window is not None and int(repetition_window) > 0:
                previous_ids = generated_so_far[-int(repetition_window) :]
            else:
                previous_ids = generated_so_far

            next_token = self._sample(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_ids=step_forbidden,
                previous_ids=previous_ids,
                repetition_penalty=repetition_penalty,
            )

            if next_token == self.eos_id:
                break

            current_ids.append(int(next_token))

        generated_ids = current_ids[len(prompt_ids) :]

        answer_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return answer_text, generated_ids

    def interactive(
        self,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        repetition_window: int = DEFAULT_REPETITION_WINDOW,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    ) -> None:
        """
        Mode CLI interaktif.

        Perintah (slash commands):
            /help                  : tampilkan bantuan
            /training              : latih model (step default)
            /training 100          : latih model 100 step
            /training --steps 100  : latih model 100 step
            /training --epochs 10  : latih model 10 epoch
            /training --epochs 10 --steps 50 : 10 epoch, minimal 50 step
            /status                : tampilkan status model
            /exit                  : keluar
        """
        print("=" * 60)
        print("Testing Playground - Inference Engine")
        print("=" * 60)
        print("Ketik prompt untuk inference.")
        print("Ketik /help untuk daftar perintah.")
        print("=" * 60)
        print(f"temperature        : {temperature}")
        print(f"top_k              : {top_k}")
        print(f"top_p              : {top_p}")
        print(f"repetition_penalty : {repetition_penalty}")
        print(f"min_new_tokens     : {min_new_tokens}")
        print(f"max_new_tokens     : {max_new_tokens}")
        print("=" * 60)

        while True:
            try:
                user_input = input("prompt> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            stripped = user_input.strip()

            if not stripped:
                continue

            # ================================================================
            # SLASH COMMANDS
            # ================================================================
            if stripped.startswith("/"):
                should_exit = self._handle_command(
                    command=stripped,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    repetition_window=repetition_window,
                    min_new_tokens=min_new_tokens,
                )

                if should_exit:
                    break

                continue

            # ================================================================
            # PROMPT INFERENCE
            # ================================================================
            answer, generated_ids = self.generate(
                prompt=stripped,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                min_new_tokens=min_new_tokens,
            )

            print("Jawaban :", answer if answer else "(kosong)")
            print(f"[{len(generated_ids)} token dihasilkan]")
            print("-" * 60)

    # ========================================================================
    # COMMAND HANDLER
    # ========================================================================

    def _handle_command(
        self,
        command: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        repetition_window: int = DEFAULT_REPETITION_WINDOW,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    ) -> bool:
        """
        Memproses slash command.

        Return:
        - True jika harus exit, False jika lanjut
        """
        parts = command.strip().split()
        cmd = parts[0].lower()

        # ----------------------------------------------------------------
        # /exit, /quit, /keluar
        # ----------------------------------------------------------------
        if cmd in {"/exit", "/quit", "/keluar"}:
            print("[playground] Keluar dari playground.")
            return True

        # ----------------------------------------------------------------
        # /help
        # ----------------------------------------------------------------
        if cmd in {"/help", "/bantuan", "/h", "/?"}:
            self._print_help()
            return False

        # ----------------------------------------------------------------
        # /training, /train, /latih
        # ----------------------------------------------------------------
        if cmd in {"/training", "/train", "/latih"}:
            self._handle_training_command(parts)
            return False

        # ----------------------------------------------------------------
        # /status
        # ----------------------------------------------------------------
        if cmd in {"/status", "/info"}:
            self._print_status()
            return False

        # ----------------------------------------------------------------
        # Perintah tidak dikenal
        # ----------------------------------------------------------------
        print(f"[playground] Perintah tidak dikenal: {cmd}")
        print("[playground] Ketik /help untuk daftar perintah.")
        return False

    # ========================================================================
    # TRAINING COMMAND HANDLER
    # ========================================================================

    def _handle_training_command(self, parts: List[str]) -> None:
        """
        Memproses perintah /training dengan berbagai format argumen.

        Format yang didukung:
            /training                          → step default dari config
            /training 100                      → 100 step
            /training --steps 100              → 100 step
            /training --s 100                  → 100 step
            /training --epochs 10              → 10 epoch
            /training --e 10                   → 10 epoch
            /training --epochs 10 --steps 50   → 10 epoch, minimal 50 step
            /training --e 10 --s 50            → 10 epoch, minimal 50 step
        """
        if self.trainer is None:
            print("[playground] Trainer tidak tersedia.")
            print("[playground] Pastikan playground dipanggil dengan parameter trainer.")
            return

        parsed = self._parse_training_args(parts)

        steps = parsed.get("steps")
        epochs = parsed.get("epochs")

        print("-" * 60)

        if epochs is not None:
            # MODE EPOCH
            min_steps = steps if steps is not None else 0

            print(
                f"[playground] Training mode EPOCH: {epochs} epoch"
                + (f", minimal {min_steps} step" if min_steps > 0 else "")
            )
            print("-" * 60)

            try:
                if hasattr(self.trainer, "train_epochs_inline"):
                    history = self.trainer.train_epochs_inline(
                        epochs=epochs,
                        min_steps=min_steps,
                        save_checkpoint=True,
                        verbose=True,
                    )
                else:
                    # Fallback: gunakan run_epochs
                    history = self.trainer.run_epochs(
                        max_steps=None,
                        save_checkpoints=True,
                        verbose=True,
                    )

                if history:
                    last = history[-1]
                    print("-" * 60)
                    print("[playground] Training epoch selesai.")
                    print(f"[playground] Epoch terakhir : {last.get('epoch', 0)}")
                    print(f"[playground] Loss           : {last.get('loss', 0.0):.6f}")
                    print(f"[playground] Perplexity     : {last.get('perplexity', 0.0):.4f}")
                    print(f"[playground] Accuracy       : {last.get('accuracy', 0.0):.4f}")
                    print("-" * 60)
                else:
                    print("[playground] Training tidak menghasilkan history.")
                    print("[playground] Pastikan folder datasets/ berisi data.")

            except Exception as exc:
                print(f"[playground] Training gagal: {exc}")

        else:
            # MODE STEP
            if steps is None:
                steps = self.DEFAULT_INLINE_TRAINING_STEPS

            print(f"[playground] Training mode STEP: {steps} step")
            print("-" * 60)

            try:
                if hasattr(self.trainer, "train_steps"):
                    history = self.trainer.train_steps(
                        steps=steps,
                        save_checkpoint=True,
                        verbose=True,
                    )
                else:
                    history = self.trainer.run_epochs(
                        max_steps=steps,
                        save_checkpoints=True,
                        verbose=True,
                    )

                if history:
                    report = history[-1]
                    print("-" * 60)
                    print("[playground] Training step selesai.")
                    print(f"[playground] Steps          : {report.get('steps', 0)}")
                    print(f"[playground] Loss           : {report.get('loss', 0.0):.6f}")
                    print(f"[playground] Perplexity     : {report.get('perplexity', 0.0):.4f}")
                    print(f"[playground] Accuracy       : {report.get('accuracy', 0.0):.4f}")
                    epochs_trav = report.get("epochs_traversed", 0)
                    if epochs_trav:
                        print(f"[playground] Epochs traversed : {epochs_trav}")
                    print("-" * 60)
                else:
                    print("[playground] Training tidak menghasilkan update.")
                    print("[playground] Pastikan folder datasets/ berisi data.")

            except Exception as exc:
                print(f"[playground] Training gagal: {exc}")

    def _parse_training_args(self, parts: List[str]) -> Dict[str, Optional[int]]:
        """
        Parse argumen perintah /training.

        Return:
            {"steps": int|None, "epochs": int|None}
        """
        result: Dict[str, Optional[int]] = {"steps": None, "epochs": None}

        i = 1  # skip /training
        while i < len(parts):
            arg = parts[i].lower()

            # --steps, --s, -s
            if arg in {"--steps", "--s", "-s"}:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    result["steps"] = int(parts[i + 1])
                    i += 2
                else:
                    i += 1

            # --epochs, --e, -e
            elif arg in {"--epochs", "--e", "-e"}:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    result["epochs"] = int(parts[i + 1])
                    i += 2
                else:
                    i += 1

            # Angka langsung = step (backward compatible)
            elif arg.isdigit():
                result["steps"] = int(arg)
                i += 1

            else:
                i += 1

        return result

    # ========================================================================
    # STATUS
    # ========================================================================

    def _print_status(self) -> None:
        """
        Tampilkan status model.
        """
        print("=" * 60)
        print("Status Model")
        print("=" * 60)

        print(f"  Parameter count : {self.model.parameter_count():,}")
        print(f"  Vocab size      : {self.tokenizer.vocab_size}")
        print(f"  Max context     : {self.max_context}")
        print(f"  Trainer         : {'Tersedia' if self.trainer is not None else 'Tidak tersedia'}")

        if self.trainer is not None and hasattr(self.trainer, "weight_manager"):
            wm = self.trainer.weight_manager
            print(f"  Optimizer step  : {wm.step_count}")
            print(f"  Global step     : {getattr(wm, 'global_step', 0)}")
            print(f"  Last epoch      : {getattr(wm, 'last_epoch', 0)}")
            print(f"  Learning rate   : {wm.learning_rate:.2e}")

        print("=" * 60)

    # ========================================================================
    # HELP
    # ========================================================================

    def _print_help(self) -> None:
        """
        Tampilkan bantuan perintah interactive.
        """
        print("=" * 60)
        print("Bantuan Playground")
        print("=" * 60)
        print()
        print("Ketik prompt apa saja untuk inference.")
        print("Contoh: apa itu mobil")
        print()
        print("Perintah (slash commands):")
        print()
        print("  /help                  : tampilkan bantuan ini")
        print()
        print("  /training              : latih model (step default)")
        print("  /training 100          : latih model 100 step")
        print("  /training --steps 100  : latih model 100 step")
        print("  /training --epochs 10  : latih model 10 epoch")
        print("  /training --epochs 10 --steps 50")
        print("                         : latih 10 epoch, minimal 50 step")
        print()
        print("  /status                : tampilkan status model")
        print("  /exit                  : keluar dari playground")
        print("  /quit                  : keluar dari playground")
        print("  /keluar                : keluar dari playground")
        print()
        print("Semua input tanpa prefix / dianggap sebagai prompt inference.")
        print("=" * 60)

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _ensure_tokenizer_ready(self) -> None:
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

    def _get_forbidden_ids(self) -> List[int]:
        """
        Token yang tidak boleh dihasilkan saat sampling.
        """
        if self._forbidden_cache is not None:
            return self._forbidden_cache

        forbidden = {
            self.pad_id,
            self.unk_id,
            self.bos_id,
        }

        if self.tokenizer.is_ready:
            for token, token_id in self.tokenizer.token_to_id.items():
                if token.startswith("[UNUSED_"):
                    forbidden.add(int(token_id))

        self._forbidden_cache = sorted(forbidden)
        return self._forbidden_cache

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        forbidden_ids: Optional[List[int]] = None,
        previous_ids: Optional[List[int]] = None,
        repetition_penalty: float = 1.0,
    ) -> int:
        """
        Sampling token berikutnya.
        """
        logits = np.asarray(logits, dtype=np.float32).copy()

        if logits.size == 0:
            return self.eos_id

        float_max = np.finfo(np.float32).max / 4.0

        raw_logits = np.nan_to_num(
            logits,
            nan=-float_max,
            posinf=float_max,
            neginf=-float_max,
        )

        raw_argmax = int(np.argmax(raw_logits))

        work_logits = raw_logits.copy()

        # Repetition penalty
        if repetition_penalty is not None:
            penalty = float(repetition_penalty)

            if penalty > 0.0 and penalty != 1.0 and previous_ids:
                prev_tokens = sorted(set(int(t) for t in previous_ids))
                prev_arr = np.array(prev_tokens, dtype=np.int64)

                prev_arr = prev_arr[
                    (prev_arr >= 0) & (prev_arr < work_logits.size)
                ]

                if prev_arr.size > 0:
                    selected_logits = work_logits[prev_arr]

                    work_logits[prev_arr] = np.where(
                        selected_logits > 0,
                        selected_logits / penalty,
                        selected_logits * penalty,
                    )

        # Forbidden tokens
        valid_mask = np.ones(raw_logits.size, dtype=bool)
        valid_forbidden: List[int] = []

        if forbidden_ids:
            valid_forbidden = [
                idx for idx in forbidden_ids
                if 0 <= idx < raw_logits.size
            ]

            if valid_forbidden:
                valid_mask[valid_forbidden] = False

        if not np.any(valid_mask):
            fallback_mask = np.ones(raw_logits.size, dtype=bool)

            if valid_forbidden:
                fallback_mask[valid_forbidden] = False

            if np.any(fallback_mask):
                valid_mask = fallback_mask
            else:
                valid_mask[raw_argmax] = True

        # Greedy jika temperature sangat kecil
        if temperature is None or float(temperature) <= 1e-8:
            greedy_logits = np.where(valid_mask, work_logits, -np.inf)
            return int(np.argmax(greedy_logits))

        # Temperature scaling
        work_logits = work_logits / float(temperature)

        # Top-K
        if top_k is not None and int(top_k) > 0:
            top_k_value = int(top_k)

            if top_k_value == 1:
                topk_logits = np.where(valid_mask, work_logits, -np.inf)
                return int(np.argmax(topk_logits))

            num_valid = int(np.sum(valid_mask))

            if num_valid > top_k_value:
                masked_logits = np.where(valid_mask, work_logits, -np.inf)
                threshold = np.partition(masked_logits, -top_k_value)[-top_k_value]
                valid_mask = valid_mask & (work_logits >= threshold)

        # Top-P / Nucleus Sampling
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            masked_logits = np.where(valid_mask, work_logits, -np.inf)
            probs = self._softmax(masked_logits)

            if np.any(probs > 0):
                sorted_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumulative_probs = np.cumsum(sorted_probs)

                cutoff = int(np.searchsorted(cumulative_probs, float(top_p)) + 1)
                cutoff = max(1, min(cutoff, len(sorted_idx)))

                allowed_idx = sorted_idx[:cutoff]

                nucleus_mask = np.zeros_like(valid_mask, dtype=bool)
                nucleus_mask[allowed_idx] = True

                valid_mask = valid_mask & nucleus_mask

                if not np.any(valid_mask) and len(allowed_idx) > 0:
                    valid_mask[allowed_idx[0]] = True

        # Sampling akhir
        final_logits = np.where(valid_mask, work_logits, -np.inf)
        probs = self._softmax(final_logits)

        probs = np.nan_to_num(
            probs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        total_prob = float(np.sum(probs))

        if not np.isfinite(total_prob) or total_prob <= 0.0:
            fallback_logits = np.where(valid_mask, raw_logits, -np.inf)
            return int(np.argmax(fallback_logits))

        probs = probs / total_prob

        try:
            return int(self.rng.choice(raw_logits.size, p=probs))
        except ValueError:
            fallback_logits = np.where(valid_mask, raw_logits, -np.inf)
            return int(np.argmax(fallback_logits))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """
        Softmax numerik stabil.
        """
        max_logit = np.max(logits)

        if not np.isfinite(max_logit):
            return np.zeros_like(logits, dtype=np.float32)

        shifted = logits - max_logit
        exp_logits = np.exp(shifted)
        denom = np.sum(exp_logits)

        if not np.isfinite(denom) or denom <= 0.0:
            return np.zeros_like(logits, dtype=np.float32)

        return exp_logits / denom


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    import argparse
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Jalankan mode CLI interaktif.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="mobil adalah",
        help="Prompt untuk mode non-interaktif.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Jumlah token maksimum yang dihasilkan.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=4,
        help="Jumlah token minimum sebelum EOS diperbolehkan.",
    )

    args = parser.parse_args()

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
                    "mobil adalah kendaraan",
                    "mobil memiliki mesin",
                    "mobil memakai bahan bakar",
                    "mesin mobil menghasilkan tenaga",
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
                sequence_length=12,
                dropout_rate=0.0,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
            ),
            model=ModelConfig(
                vocab_size=96,
                embedding_dim=32,
                num_attention_heads=2,
                num_layers=1,
                dropout_rate=0.0,
                max_position_embeddings=64,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=64,
                train_dtype="float32",
                export_dtype="float32",
            ),
        )

        tokenizer = Tokenizer(small_config)
        tokenizer.train()

        model = CustomTransformerLM(small_config)

        playground = Playground(
            model=model,
            tokenizer=tokenizer,
            config=small_config,
            trainer=None,
        )

        if args.interactive:
            playground.interactive(
                max_new_tokens=args.max_new_tokens,
            )
        else:
            playground.test_prompt(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )

        print("testing/playground.py test OK")
