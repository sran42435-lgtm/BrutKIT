# testing/playground.py
#
# Playground / Inference Testing Engine.
#
# Tugas utama:
# 1. Menerima prompt teks.
# 2. Encode prompt menjadi token ID.
# 3. Jalankan inference ke model.
# 4. Sampling token berikutnya dengan temperature, top-k, dan top-p.
# 5. Decode token ID menjadi teks jawaban.
# 6. Menyediakan CLI interaktif.
#
# Keterhubungan:
# - config.py            : konfigurasi global
# - core/tokenizer.py    : encode/decode teks
# - core/architecture.py : model inference

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
# Memastikan project root ada di sys.path, sehingga file di dalam folder
# testing/ tetap bisa meng-import config.py meskipun dijalankan langsung:
#   python testing/playground.py
# atau:
#   cd testing && python playground.py

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

    Contoh pemakaian:

        playground = Playground(model, tokenizer, CONFIG)
        playground.test_prompt("Apa itu mobil?")

    Atau mode interaktif:

        playground.interactive()
    """

    DEFAULT_MAX_NEW_TOKENS = 32
    DEFAULT_TEMPERATURE = 0.8
    DEFAULT_TOP_K = 40
    DEFAULT_TOP_P = 0.9

    def __init__(self, model, tokenizer, config: Config = CONFIG):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

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
    ) -> tuple[str, List[int]]:
        """
        Membuat jawaban/lanjutan teks dari prompt.

        Return:
        - answer_text  : teks hasil generation
        - generated_ids : daftar token ID yang dihasilkan (tanpa prompt awal)
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

        for _ in range(max_new_tokens):
            # Jaga konteks agar tidak melebihi max_position_embeddings
            context = current_ids[-self.max_context :]

            input_ids = np.array([context], dtype=np.int64)

            logits = self.model.forward(input_ids, training=False)

            # Ambil logits dari token terakhir
            next_logits = np.asarray(logits[0, -1, :], dtype=np.float32)

            next_token = self._sample(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_ids=forbidden_ids,
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
    ) -> None:
        """
        Mode CLI interaktif.
        """
        print("=" * 60)
        print("Testing Playground - Inference Engine")
        print("Ketik prompt lalu tekan Enter.")
        print("Ketik 'exit', 'quit', atau 'keluar' untuk berhenti.")
        print("=" * 60)

        while True:
            try:
                prompt = input("prompt> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            prompt_stripped = prompt.strip()

            if not prompt_stripped:
                continue

            if prompt_stripped.lower() in {"exit", "quit", "keluar"}:
                break

            answer, generated_ids = self.generate(
                prompt=prompt_stripped,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            print("Jawaban :", answer if answer else "(kosong)")
            print(f"[{len(generated_ids)} token dihasilkan]")
            print("-" * 60)

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _ensure_tokenizer_ready(self) -> None:
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

    def _get_forbidden_ids(self) -> List[int]:
        """
        Token yang tidak boleh dihasilkan saat sampling:
        - [PAD]
        - [UNK]
        - [BOS]
        - token [UNUSED_*]

        [EOS] tetap diperbolehkan sebagai tanda berhenti.
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
    ) -> int:
        """
        Sampling token berikutnya.

        Urutan:
        1. Bersihkan logits.
        2. Blokir token terlarang.
        3. Terapkan temperature.
        4. Terapkan top-k.
        5. Terapkan top-p.
        6. Sample dari distribusi probabilitas.
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

        valid_mask = np.ones(raw_logits.size, dtype=bool)

        if forbidden_ids:
            valid_forbidden = [
                idx for idx in forbidden_ids
                if 0 <= idx < raw_logits.size
            ]

            if valid_forbidden:
                valid_mask[valid_forbidden] = False

        # Jika semua token terblokir, buka fallback ke token dengan logit tertinggi.
        if not np.any(valid_mask):
            valid_mask[raw_argmax] = True

        work_logits = raw_logits.copy()

        # --------------------------------------------------------------------
        # Greedy jika temperature sangat kecil / nol
        # --------------------------------------------------------------------
        if temperature is None or float(temperature) <= 1e-8:
            greedy_logits = np.where(valid_mask, work_logits, -np.inf)
            return int(np.argmax(greedy_logits))

        # --------------------------------------------------------------------
        # Temperature scaling
        # --------------------------------------------------------------------
        work_logits = work_logits / float(temperature)

        # --------------------------------------------------------------------
        # Top-K
        # --------------------------------------------------------------------
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

        # --------------------------------------------------------------------
        # Top-P / Nucleus Sampling
        # --------------------------------------------------------------------
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

                # Jika tidak ada token valid setelah top-p, paksa ambil
                # token teratas dari nucleus.
                if not np.any(valid_mask) and len(allowed_idx) > 0:
                    valid_mask[allowed_idx[0]] = True

        # --------------------------------------------------------------------
        # Sampling akhir
        # --------------------------------------------------------------------
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
        default="Apa itu mobil?",
        help="Prompt untuk mode non-interaktif.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12,
        help="Jumlah token maksimum yang dihasilkan.",
    )
    args = parser.parse_args()

    # ========================================================================
    # Demo kecil untuk menguji playground.
    #
    # Ini memakai folder sementara dan model kecil, sehingga tidak mengganggu
    # dataset utama atau konfigurasi utama.
    # ========================================================================

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
                sequence_length=8,
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
                max_position_embeddings=32,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=32,
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
        )

        if args.interactive:
            playground.interactive(
                max_new_tokens=args.max_new_tokens,
            )
        else:
            playground.test_prompt(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
                top_k=20,
                top_p=0.9,
                verbose=True,
            )

        print("testing/playground.py test OK")
