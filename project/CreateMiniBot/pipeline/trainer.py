# pipeline/trainer.py
#
# Trainer / Training Loop Engine untuk Training Pipeline Engine.
#
# VERSI MOE-INTEGRATED + BUG FIX:
# SOTA features (tetap dipertahankan):
# - Conditioned Training: domain prefix ([MOBIL], [SAINS]) dipertahankan
# - Domain-Aware Batching: batch hanya berisi satu domain
# - Document Boundary dengan 
# - Resume dari epoch terakhir
# - Early stopping & periodic checkpoint
#
# MoE features:
# - Auto domain detection dari filenames datasets/*.txt
# - Per-domain training dengan shared backbone
# - Integrasi dengan core/router.py untuk keyword extraction
# - Build semantic cache setelah training
# - Save keywords + embeddings ke output folder
# - Support mode "per_domain" dan "mixed"
#
# BUG FIX (PENTING):
# - Router sekarang dibangun juga setelah train_epochs_inline()
# - Router sekarang dibangun juga setelah train_steps()
# - Sebelumnya router HANYA dibangun di run_epochs()/_run_moe_training()
# - Ini menyebabkan inline training dari playground tidak menghasilkan router
#
# Keterhubungan:
# - config.py              : hyperparameter training + MoE
# - core/tokenizer.py      : encode teks menjadi token ID
# - core/architecture.py   : model (shared backbone + LM heads)
# - core/router.py         : keyword extraction + semantic cache
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

# Import router untuk MoE integration
try:
    from core.router import DomainRouter
    _ROUTER_AVAILABLE = True
except ImportError:
    _ROUTER_AVAILABLE = False


# Regex untuk mendeteksi domain prefix
DOMAIN_PREFIX_PATTERN = re.compile(r'^\[([A-Z]+)\]\s*')


class Trainer:
    """
    Mesin latihan utama dengan dukungan MoE (Mixture of Experts).

    Fitur SOTA:
    - Resume dari epoch terakhir
    - Early stopping
    - Periodic checkpoint
    - Inline training dari playground (berbasis step & epoch)
    - Pembacaan berbasis blok paragraf
    - Conditioned Training (domain prefix dipertahankan)
    - Domain-Aware Batching (batch per domain)
    - Document Boundary dengan 

    Fitur MoE:
    - Auto domain detection dari filenames
    - Per-domain training dengan shared backbone
    - Keyword extraction via core/router.py
    - Semantic cache untuk inference-time retrieval
    - Save all assets ke output folder
    
    BUG FIX:
    - Router dibangun otomatis setelah SEMUA mode training selesai
      (run_epochs, train_epochs_inline, train_steps)
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

        # MoE config
        self.use_moe = bool(getattr(config.model, 'use_moe', True))
        self.moe_train_mode = getattr(config.training, 'moe_train_mode', 'per_domain')
        self.moe_epochs_per_domain = int(getattr(config.training, 'moe_epochs_per_domain', 30))
        self.moe_rebuild_cache = bool(
            getattr(config.training, 'moe_rebuild_cache_after_training', True)
        )

        # Router instance
        self._router: Optional['DomainRouter'] = None
        self._paragraphs_by_domain: Dict[str, List[str]] = {}

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
    # ROUTER ACCESS
    # ========================================================================

    def get_router(self) -> Optional['DomainRouter']:
        """
        Return router instance (jika sudah dibangun).

        Router dibangun otomatis setelah training selesai.
        """
        return self._router

    def get_paragraphs_by_domain(self) -> Dict[str, List[str]]:
        """
        Return dict paragraf per domain (dari dataset loading terakhir).
        """
        return dict(self._paragraphs_by_domain)

    # ========================================================================
    # DOMAIN PREFIX HANDLING
    # ========================================================================

    def _extract_domain_from_text(self, text: str) -> Optional[str]:
        """
        Mengekstrak domain dari prefix teks [DOMAIN].

        Return:
        - Nama domain jika ada prefix
        - None jika tidak ada prefix
        """
        match = DOMAIN_PREFIX_PATTERN.match(text)
        if match:
            return match.group(1)
        return None

    def _extract_domain_from_filename(self, filepath: Path) -> str:
        """
        Extract domain name dari filename.

        Contoh:
            "datasets/mobil.txt" → "MOBIL"
            "datasets/sains_data.json" → "SAINS_DATA"
        """
        return filepath.stem.upper().replace("-", "_").replace(" ", "_")

    # ========================================================================
    # MOE: DOMAIN INITIALIZATION
    # ========================================================================

    def _init_model_domains(
        self,
        paragraphs_by_domain: Dict[str, List[str]],
        verbose: bool = True,
    ) -> None:
        """
        Inisialisasi LM heads di model untuk setiap domain yang terdeteksi.

        Hanya dipanggil jika use_moe=True.
        """
        if not self.use_moe:
            return

        if not hasattr(self.model, 'init_domains'):
            if verbose:
                print("[Trainer] Model tidak mendukung MoE (tidak ada init_domains method).")
            return

        domains = list(paragraphs_by_domain.keys())

        # Filter out "_GENERAL_" karena sudah ada sebagai fallback di model
        domains = [d for d in domains if d != "_GENERAL_"]

        if not domains:
            if verbose:
                print("[Trainer] Tidak ada domain spesifik yang terdeteksi.")
            return

        # Inisialisasi semua domain
        self.model.init_domains(domains)

        if verbose:
            print(f"[Trainer] Inisialisasi {len(domains)} domain LM heads:")
            for domain in domains:
                n_paras = len(paragraphs_by_domain[domain])
                print(f"  - {domain}: {n_paras} paragraphs")

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
        Main training entry point.

        Jika use_moe=True dan moe_train_mode="per_domain":
            → Panggil _run_moe_training()

        Jika tidak:
            → Panggil _run_mixed_training() (traditional)
        """
        # Pre-load paragraphs untuk MoE
        rng_init = np.random.default_rng(self.seed)
        self._paragraphs_by_domain = self._load_paragraphs_by_domain(rng=rng_init)

        if self.use_moe and self.moe_train_mode == "per_domain":
            return self._run_moe_training(
                max_steps=max_steps,
                save_checkpoints=save_checkpoints,
                verbose=verbose,
                start_epoch=start_epoch,
                global_step=global_step,
            )
        else:
            return self._run_mixed_training(
                max_steps=max_steps,
                save_checkpoints=save_checkpoints,
                verbose=verbose,
                start_epoch=start_epoch,
                global_step=global_step,
            )

    # ========================================================================
    # MOE TRAINING LOOP
    # ========================================================================

    def _run_moe_training(
        self,
        max_steps: Optional[int] = None,
        save_checkpoints: bool = False,
        verbose: bool = True,
        start_epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> List[dict]:
        """
        Per-domain training loop untuk MoE.

        Flow:
        1. Auto-detect domains dari dataset
        2. Init LM heads untuk setiap domain
        3. Untuk setiap domain:
           a. Set active domain di model
           b. Train selama moe_epochs_per_domain epochs
           c. Track loss per domain
        4. Setelah semua domain selesai:
           a. Build keywords (via router)
           b. Build semantic cache (via router)
           c. Save semua assets ke output folder
        """
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        if start_epoch is None:
            start_epoch = self.start_epoch
        if global_step is None:
            global_step = self.start_global_step

        if not self._paragraphs_by_domain:
            if verbose:
                print("[Trainer] Dataset kosong. Tidak ada yang bisa dilatih.")
            return []

        # ================================================================
        # STEP 1: Inisialisasi LM heads untuk setiap domain
        # ================================================================
        if verbose:
            print("=" * 70)
            print("MoE TRAINING MODE")
            print("=" * 70)

        self._init_model_domains(self._paragraphs_by_domain, verbose=verbose)

        # Daftar domain yang akan dilatih (exclude _GENERAL_)
        training_domains = [
            d for d in self._paragraphs_by_domain.keys()
            if d != "_GENERAL_"
        ]

        if not training_domains:
            if verbose:
                print("[Trainer] Tidak ada domain spesifik untuk dilatih.")
            return []

        # ================================================================
        # STEP 2: Per-domain training
        # ================================================================
        all_history: List[dict] = []

        for domain_idx, domain in enumerate(training_domains):
            if verbose:
                print()
                print("=" * 70)
                print(f"DOMAIN [{domain}] ({domain_idx + 1}/{len(training_domains)})")
                print("=" * 70)

            # Set active domain di model
            if hasattr(self.model, 'set_active_domain'):
                self.model.set_active_domain(domain)

            # Train untuk domain ini
            domain_history = self._train_single_domain(
                domain=domain,
                paragraphs=self._paragraphs_by_domain[domain],
                max_steps=max_steps,
                save_checkpoints=save_checkpoints,
                verbose=verbose,
                start_epoch=start_epoch,
                global_step=global_step,
            )

            # Update global_step untuk domain berikutnya
            if domain_history:
                global_step += sum(r.get('steps', 0) for r in domain_history)
                all_history.extend(domain_history)

        # ================================================================
        # STEP 3: Train GENERAL domain (dengan semua data campuran)
        # ================================================================
        if "_GENERAL_" in self._paragraphs_by_domain or self.config.model.moe_always_create_general_head:
            if verbose:
                print()
                print("=" * 70)
                print("DOMAIN [GENERAL] (fallback head)")
                print("=" * 70)

            # Kumpulkan semua paragraf untuk GENERAL training
            all_paragraphs = []
            for paras in self._paragraphs_by_domain.values():
                all_paragraphs.extend(paras)

            if all_paragraphs:
                if hasattr(self.model, 'set_active_domain'):
                    self.model.set_active_domain("GENERAL")

                general_history = self._train_single_domain(
                    domain="GENERAL",
                    paragraphs=all_paragraphs,
                    max_steps=max_steps,
                    save_checkpoints=save_checkpoints,
                    verbose=verbose,
                    start_epoch=start_epoch,
                    global_step=global_step,
                    epochs_override=self.moe_epochs_per_domain // len(training_domains) if training_domains else self.moe_epochs_per_domain,
                )

                if general_history:
                    global_step += sum(r.get('steps', 0) for r in general_history)
                    all_history.extend(general_history)

        # ================================================================
        # STEP 4: Build router assets (keywords + semantic cache)
        # ================================================================
        if self.moe_rebuild_cache and _ROUTER_AVAILABLE:
            self._build_router_assets(verbose=verbose)
            self._save_router_assets(verbose=verbose)

        if verbose:
            print()
            print("=" * 70)
            print("MoE TRAINING SELESAI")
            print(f"Total domains dilatih: {len(training_domains) + 1}")  # +1 untuk GENERAL
            print(f"Total epochs: {len(all_history)}")
            print("=" * 70)

        return all_history

    def _train_single_domain(
        self,
        domain: str,
        paragraphs: List[str],
        max_steps: Optional[int],
        save_checkpoints: bool,
        verbose: bool,
        start_epoch: int,
        global_step: int,
        epochs_override: Optional[int] = None,
    ) -> List[dict]:
        """
        Train satu domain untuk beberapa epochs.

        Return:
        - List of epoch reports
        """
        if not paragraphs:
            return []

        num_epochs = epochs_override if epochs_override is not None else self.moe_epochs_per_domain
        history: List[dict] = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_correct = 0
            steps_done = 0

            rng = np.random.default_rng(self.seed + epoch + hash(domain) % 10000)

            if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
                print(f"[{domain}] Epoch {epoch + 1}/{num_epochs} dimulai")

            # Generate batches untuk domain ini saja
            for inputs, targets in self._iter_domain_batches(paragraphs, shuffle=True, rng=rng):
                if max_steps is not None and global_step >= int(max_steps):
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

                    epoch_loss += float(loss) * num_valid_tokens
                    epoch_tokens += num_valid_tokens
                    epoch_correct += int(metrics["num_correct"])

                    if verbose and steps_done % self.LOG_EVERY_STEPS == 0:
                        print(
                            f"[{domain}] epoch={epoch + 1} "
                            f"step={steps_done} "
                            f"global_step={global_step} "
                            f"loss={loss:.6f} "
                            f"ppl={metrics['perplexity']:.4f} "
                            f"acc={metrics['accuracy']:.4f} "
                            f"lr={step_info['learning_rate']:.2e}"
                        )
                else:
                    self.model.zero_grad()

                steps_done += 1
                global_step += 1

            # Rekap epoch
            if steps_done > 0:
                avg_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0.0
                accuracy = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0.0
                perplexity = float(np.exp(avg_loss)) if avg_loss < 100 else float("inf")

                epoch_report = {
                    "epoch": epoch + 1,
                    "domain": domain,
                    "loss": float(avg_loss),
                    "perplexity": float(perplexity),
                    "accuracy": float(accuracy),
                    "num_valid_tokens": int(epoch_tokens),
                    "num_correct": int(epoch_correct),
                    "steps": int(steps_done),
                }

                history.append(epoch_report)

                if verbose:
                    print(
                        f"[{domain}] Epoch {epoch + 1} selesai | "
                        f"loss={avg_loss:.6f} | "
                        f"ppl={perplexity:.4f} | "
                        f"acc={accuracy:.4f} | "
                        f"tokens={epoch_tokens}"
                    )

                # Save checkpoint
                self.weight_manager.last_epoch = epoch + 1
                self.weight_manager.global_step = global_step

                if save_checkpoints:
                    ckpt_path = self.weight_manager.save_checkpoint()
                    if verbose:
                        print(f"[{domain}] Checkpoint disimpan: {ckpt_path}")

        return history

    def _iter_domain_batches(
        self,
        paragraphs: List[str],
        shuffle: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield batches dari list paragraf (satu domain saja).
        """
        if rng is None:
            rng = np.random.default_rng()

        if shuffle and len(paragraphs) > 1:
            rng.shuffle(paragraphs)

        # Generate examples
        examples: List[np.ndarray] = []
        for para in paragraphs:
            for ex in self._paragraph_to_examples(para):
                examples.append(ex)

        if not examples:
            return

        # Yield batches
        for i in range(0, len(examples), self.batch_size):
            chunk = examples[i : i + self.batch_size]
            if not chunk:
                continue

            stacked = np.stack(chunk, axis=0).astype(np.int64)
            inputs = stacked[:, :-1]
            targets = stacked[:, 1:]

            yield inputs, targets

    # ========================================================================
    # MIXED TRAINING LOOP (TRADITIONAL - NON-MOE)
    # ========================================================================

    def _run_mixed_training(
        self,
        max_steps: Optional[int] = None,
        save_checkpoints: bool = False,
        verbose: bool = True,
        start_epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> List[dict]:
        """
        Traditional training loop (non-MoE).

        Semua domain dicampur dalam satu epoch, dengan domain-aware batching.
        """
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
    # INLINE TRAINING BERBASIS STEP (dari playground)
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

        # ================================================================
        # BUG FIX: Pre-load paragraphs sebelum training
        # Sebelumnya ini tidak dilakukan, menyebabkan router tidak dibangun
        # ================================================================
        if self.use_moe and not self._paragraphs_by_domain:
            rng_init = np.random.default_rng(self.seed)
            self._paragraphs_by_domain = self._load_paragraphs_by_domain(rng=rng_init)
            
            # Inisialisasi LM heads jika belum ada
            if self._paragraphs_by_domain:
                self._init_model_domains(self._paragraphs_by_domain, verbose=verbose)

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

        # ================================================================
        # BUG FIX: Build router setelah step training selesai
        # Sebelumnya router TIDAK dibangun di train_steps()
        # ================================================================
        if self.use_moe and self.moe_rebuild_cache and _ROUTER_AVAILABLE:
            if not self._paragraphs_by_domain:
                rng_init = np.random.default_rng(self.seed)
                self._paragraphs_by_domain = self._load_paragraphs_by_domain(rng=rng_init)
            
            if self._paragraphs_by_domain:
                self._build_router_assets(verbose=verbose)
                self._save_router_assets(verbose=verbose)

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
        if not self.tokenizer.is_ready:
            self.tokenizer.load()

        epochs = max(1, int(epochs))
        min_steps = max(0, int(min_steps))

        # ================================================================
        # BUG FIX: Pre-load paragraphs sebelum training
        # Sebelumnya ini tidak dilakukan, menyebabkan router tidak dibangun
        # ================================================================
        if self.use_moe and not self._paragraphs_by_domain:
            rng_init = np.random.default_rng(self.seed)
            self._paragraphs_by_domain = self._load_paragraphs_by_domain(rng=rng_init)
            
            # Inisialisasi LM heads jika belum ada
            if self._paragraphs_by_domain:
                self._init_model_domains(self._paragraphs_by_domain, verbose=verbose)

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

        # ================================================================
        # BUG FIX: Build router setelah inline epoch training selesai
        # Sebelumnya router TIDAK dibangun di train_epochs_inline()
        # ================================================================
        if self.use_moe and self.moe_rebuild_cache and _ROUTER_AVAILABLE:
            if not self._paragraphs_by_domain:
                rng_init = np.random.default_rng(self.seed)
                self._paragraphs_by_domain = self._load_paragraphs_by_domain(rng=rng_init)
            
            if self._paragraphs_by_domain:
                self._build_router_assets(verbose=verbose)
                self._save_router_assets(verbose=verbose)

        return history

    # ========================================================================
    # ROUTER ASSETS: BUILD & SAVE
    # ========================================================================

    def _build_router_assets(self, verbose: bool = True) -> None:
        """
        Bangun keywords dan semantic cache menggunakan DomainRouter.

        Dipanggil setelah training selesai.
        """
        if not _ROUTER_AVAILABLE:
            if verbose:
                print("[Trainer] core/router.py tidak tersedia. Skip build router assets.")
            return

        if not self._paragraphs_by_domain:
            if verbose:
                print("[Trainer] Tidak ada paragraf untuk build router assets.")
            return

        if verbose:
            print()
            print("=" * 70)
            print("BUILDING ROUTER ASSETS")
            print("=" * 70)

        # Create router instance
        self._router = DomainRouter(self.config)

        # Build keywords
        if verbose:
            print("[Trainer] Building keywords...")
        self._router.build_keywords(self._paragraphs_by_domain, verbose=verbose)

        # Build semantic cache
        if verbose:
            print("[Trainer] Building semantic cache...")
        self._router.build_semantic_cache(
            model=self.model,
            tokenizer=self.tokenizer,
            corpus_by_domain=self._paragraphs_by_domain,
            verbose=verbose,
        )

    def _save_router_assets(self, verbose: bool = True) -> None:
        """
        Save keywords JSON dan semantic cache NPZ ke output folder.
        """
        if self._router is None:
            if verbose:
                print("[Trainer] Router belum dibangun. Skip save.")
            return

        output_dir = Path(self.config.paths.output_models_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keywords_path = output_dir / "domain_keywords.json"
        cache_path = output_dir / "semantic_cache.npz"

        if verbose:
            print()
            print("[Trainer] Saving router assets:")
            print(f"  Keywords: {keywords_path}")
            print(f"  Cache   : {cache_path}")

        try:
            self._router.save_keywords(keywords_path)
            self._router.save_cache(cache_path)

            if verbose:
                print("[Trainer] Router assets berhasil disimpan.")
        except Exception as exc:
            if verbose:
                print(f"[Trainer] Gagal menyimpan router assets: {exc}")

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

        # Reload paragraphs jika belum ada
        if not self._paragraphs_by_domain:
            self._paragraphs_by_domain = self._load_paragraphs_by_domain(rng=rng)

        if not self._paragraphs_by_domain:
            return

        # Domain-Aware Batching
        all_batches: List[Tuple[np.ndarray, np.ndarray]] = []

        for domain, paragraphs in self._paragraphs_by_domain.items():
            if not paragraphs:
                continue

            if shuffle and len(paragraphs) > 1:
                rng.shuffle(paragraphs)

            domain_examples: List[np.ndarray] = []
            for paragraph in paragraphs:
                for example in self._paragraph_to_examples(paragraph):
                    domain_examples.append(example)

            if not domain_examples:
                continue

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

        if shuffle and len(all_batches) > 1:
            rng.shuffle(all_batches)

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
        if not paragraph.strip():
            return

        chunk_size = self.sequence_length + 1

        # Encode paragraf tanpa BOS/EOS (kita tambahkan sendiri)
        ids = self.tokenizer.encode(paragraph, add_bos=False, add_eos=False)

        if not ids:
            return

        # KASUS 1: Paragraf muat dalam satu sequence
        if len(ids) + 3 <= chunk_size:
            sequence = [self.bos_id] + ids + [self.eos_id, self.endoftext_id]
            padding_needed = chunk_size - len(sequence)
            sequence = sequence + [self.pad_id] * padding_needed
            yield np.array(sequence, dtype=np.int64)
            return

        # KASUS 2: Paragraf terlalu panjang
        content_size = chunk_size - 3

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

        Domain detection strategy:
        1. Cek prefix [DOMAIN] di setiap paragraf
        2. Jika tidak ada prefix, gunakan nama file sebagai domain

        Return:
        - Dict[str, List[str]]: domain -> list of paragraphs
          Contoh: {"MOBIL": [...], "SAINS": [...], "FILSAFAT": [...]}
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

        rng.shuffle(file_paths)

        paragraphs_by_domain: Dict[str, List[str]] = defaultdict(list)

        for path in file_paths:
            # Domain dari filename (sebagai fallback)
            fallback_domain = self._extract_domain_from_filename(path)
            suffix = path.suffix.lower()

            if suffix == ".txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                paragraphs = content.split("\n\n")

                for para in paragraphs:
                    para = para.strip()
                    para = para.replace("\n", " ")
                    if para:
                        # Cek prefix di teks, fallback ke nama file
                        domain = self._extract_domain_from_text(para) or fallback_domain
                        paragraphs_by_domain[domain].append(para)

            elif suffix == ".json":
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        data = json.load(f)

                    for text in self._extract_json_strings(data):
                        text = text.strip()
                        if text:
                            domain = self._extract_domain_from_text(text) or fallback_domain
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
                                    domain = self._extract_domain_from_text(text) or fallback_domain
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

        # Dataset dengan domain prefix (akan auto-detect)
        mobil_file = datasets_dir / "mobil.txt"
        mobil_file.write_text(
            "[MOBIL] Mobil adalah kendaraan beroda empat. "
            "Mobil menggunakan mesin sebagai penggerak utama.\n\n"
            "[MOBIL] Rem mobil berfungsi untuk menghentikan kendaraan. "
            "Keselamatan berkendara sangat penting.",
            encoding="utf-8",
        )

        sains_file = datasets_dir / "sains.txt"
        sains_file.write_text(
            "[SAINS] Air adalah senyawa kimia dengan rumus H2O. "
            "Air merupakan kebutuhan mendasar bagi makhluk hidup.\n\n"
            "[SAINS] Fotosintesis adalah proses pembuatan makanan oleh tumbuhan.",
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
                dropout_rate=0.0,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
                moe_keyword_method="hybrid",
                moe_keywords_per_domain=10,
                moe_train_mode="per_domain",
                moe_epochs_per_domain=2,
                moe_rebuild_cache_after_training=True,
            ),
            model=ModelConfig(
                vocab_size=128,
                embedding_dim=32,
                num_attention_heads=4,
                num_layers=1,
                max_position_embeddings=32,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=64,
                train_dtype="float32",
                export_dtype="float32",
                use_moe=True,
                moe_auto_routing=True,
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
        # Test 1: Domain extraction dari text dan filename
        # ================================================================
        print("=" * 60)
        print("Test 1: Domain Extraction")
        print("=" * 60)

        test_texts = [
            "[MOBIL] Mobil adalah kendaraan.",
            "[SAINS] Air adalah H2O.",
            "Teks tanpa prefix.",
        ]
        for text in test_texts:
            domain = trainer._extract_domain_from_text(text)
            print(f"  Text: '{text}'")
            print(f"  Domain from text: {domain}")
            print()

        test_paths = [
            Path("datasets/mobil.txt"),
            Path("datasets/sains_data.json"),
            Path("datasets/filsafat-klasik.txt"),
        ]
        for p in test_paths:
            domain = trainer._extract_domain_from_filename(p)
            print(f"  File: {p.name} → domain: {domain}")

        # ================================================================
        # Test 2: Paragraph loading
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 2: Paragraph Loading by Domain")
        print("=" * 60)

        rng = np.random.default_rng(1337)
        paragraphs_by_domain = trainer._load_paragraphs_by_domain(rng=rng)
        print(f"Domains found: {list(paragraphs_by_domain.keys())}")
        for domain, paras in paragraphs_by_domain.items():
            print(f"  [{domain}]: {len(paras)} paragraphs")

        # ================================================================
        # Test 3: MoE Training
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 3: MoE Training (per-domain)")
        print("=" * 60)

        history = trainer.run_epochs(
            save_checkpoints=True,
            verbose=True,
        )

        print(f"\nTotal epoch reports: {len(history)}")
        for r in history[-3:]:
            print(f"  Epoch {r['epoch']} [{r.get('domain', 'MIXED')}]: loss={r['loss']:.4f}")

        # ================================================================
        # Test 4: Router assets
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 4: Router Assets")
        print("=" * 60)

        router = trainer.get_router()
        if router is not None:
            print(f"Router available: YES")
            print(f"Domains with keywords: {list(router.domain_keywords.keys())}")
            print(f"Domains with embeddings: {list(router.search_engine.domain_embeddings.keys())}")

            # Test routing
            test_prompts = [
                "apa itu fotosintesis?",
                "bagaimana cara kerja rem?",
                "jelaskan tentang etika",
            ]
            for prompt in test_prompts:
                domain, conf, _ = router.route(prompt)
                print(f"  '{prompt}' → {domain} ({conf:.2%})")
        else:
            print("Router available: NO (core/router.py tidak terinstall)")

        # ================================================================
        # Test 5: Check saved files
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 5: Saved Files")
        print("=" * 60)

        keywords_path = output_models_dir / "domain_keywords.json"
        cache_path = output_models_dir / "semantic_cache.npz"

        print(f"Keywords file exists: {keywords_path.exists()}")
        print(f"Cache file exists: {cache_path.exists()}")

        if keywords_path.exists():
            with open(keywords_path, "r", encoding="utf-8") as f:
                kw_data = json.load(f)
            print(f"Keywords domains: {list(kw_data.get('domain_keywords', {}).keys())}")

        # ================================================================
        # Test 6: BUG FIX - Inline training builds router
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 6: Inline Training Router Build (BUG FIX)")
        print("=" * 60)
        
        # Reset router
        trainer._router = None
        
        # Jalankan inline training
        inline_history = trainer.train_epochs_inline(
            epochs=1,
            save_checkpoint=False,
            verbose=True,
        )
        
        # Cek apakah router sekarang terbangun
        router_after = trainer.get_router()
        if router_after is not None:
            print(f"\n✅ BUG FIXED: Router terbangun setelah train_epochs_inline()")
            print(f"  Domains: {list(router_after.domain_keywords.keys())}")
        else:
            print(f"\n❌ BUG STILL EXISTS: Router tidak terbangun")
        
        # Cek file tersimpan
        print(f"\nKeywords file exists: {keywords_path.exists()}")
        print(f"Cache file exists: {cache_path.exists()}")

        print("\n" + "=" * 60)
        print("pipeline/trainer.py (MoE + BUG FIX) test OK")
        print("=" * 60)
