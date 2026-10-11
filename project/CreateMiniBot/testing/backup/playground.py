# testing/playground.py
#
# Playground / Inference Testing Engine dengan MoE Auto-Routing.
#
# Fitur Utama:
# - Anti Semantic Drift: temperature rendah, top-k/top-p ketat
# - MoE Auto-Routing: otomatis pilih LM head berdasarkan domain
# - Semantic Context Injection: inject paragraf relevan ke prompt
# - Domain-Aware Inference: manual prefix via /domain
# - Stop generation saat  atau [EOS]
# - Debug mode: tampilkan token IDs dan pieces
#
# Perintah interactive (slash commands):
#   /help                  : tampilkan bantuan
#   /training              : latih model (step default)
#   /training 100          : latih model 100 step
#   /training --steps 100  : latih model 100 step
#   /training --epochs 10  : latih model 10 epoch
#   /training --epochs 10 --steps 50 : latih 10 epoch, minimal 50 step
#   /status                : tampilkan status model
#   /domain MOBIL          : set domain prefix manual
#   /domain                : hapus domain prefix manual
#   /domains               : list semua domain yang tersedia (MoE)
#   /auto [on|off]         : toggle auto-routing MoE (BARU)
#   /force DOMAIN          : force domain untuk testing (BARU)
#   /force                 : hapus force domain (BARU)
#   /exit                  : keluar
#
# Semua input tanpa prefix / dianggap sebagai prompt inference.

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Import DomainRouter untuk MoE auto-routing (optional)
try:
    from core.router import DomainRouter
    _ROUTER_AVAILABLE = True
except ImportError:
    DomainRouter = None
    _ROUTER_AVAILABLE = False


# Regex untuk mendeteksi domain prefix
DOMAIN_PREFIX_PATTERN = re.compile(r'^\[([A-Z]+)\]\s*')


class Playground:
    """
    Tempat testing pertanyaan / inference dengan MoE Auto-Routing.

    Fitur Anti Semantic Drift:
    - Temperature rendah (0.3) untuk determinisme
    - Top-K/Top-P ketat untuk filtering
    - Stop generation saat 

    Fitur MoE (BARU):
    - Auto-routing: otomatis pilih LM head berdasarkan keyword di prompt
    - Semantic context injection: inject paragraf relevan ke prompt
    - Domain info: /domains untuk list domain, /force untuk testing

    Perintah interactive (slash commands):
        /help                  : tampilkan bantuan
        /training              : latih model (step default)
        /training 100          : latih model 100 step
        /training --steps 100  : latih model 100 step
        /training --epochs 10  : latih model 10 epoch
        /training --epochs 10 --steps 50 : latih 10 epoch, minimal 50 step
        /status                : tampilkan status model
        /domain MOBIL          : set domain prefix manual
        /domain                : hapus domain prefix manual
        /domains               : list semua domain yang tersedia
        /auto [on|off]         : toggle auto-routing MoE
        /force DOMAIN          : force domain untuk testing
        /force                 : hapus force domain
        /exit                  : keluar
    """

    # ================================================================
    # DEFAULT PARAMETERS - DIOPTIMALKAN UNTUK MODEL 17M
    # ================================================================
    DEFAULT_MAX_NEW_TOKENS = 64
    
    # Temperature rendah = lebih deterministik, kurang "liar"
    DEFAULT_TEMPERATURE = 0.3
    
    # Top-K ketat = hanya pertimbangkan 20 token teratas
    DEFAULT_TOP_K = 20
    
    # Top-P ketat = nucleus sampling lebih fokus
    DEFAULT_TOP_P = 0.85

    DEFAULT_REPETITION_PENALTY = 1.3
    DEFAULT_REPETITION_WINDOW = 64

    DEFAULT_MIN_NEW_TOKENS = 4

    # Default step untuk /training tanpa argumen
    DEFAULT_INLINE_TRAINING_STEPS = 10

    # Domain fallback jika router tidak bisa menentukan
    FALLBACK_DOMAIN = "GENERAL"

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
        self.endoftext_id = int(config.special_tokens.endoftext_id)

        self._forbidden_cache: Optional[List[int]] = None
        
        # Domain prefix manual untuk domain-aware inference
        self._default_domain: Optional[str] = None
        
        # ============================================================
        # MoE STATE (BARU)
        # ============================================================
        self._router: Optional[object] = None  # DomainRouter instance
        self._auto_routing: bool = bool(getattr(config.model, 'moe_auto_routing', True))
        self._force_domain: Optional[str] = None
        self._use_moe: bool = bool(getattr(config.model, 'use_moe', False))
        self._semantic_top_k: int = int(getattr(config.model, 'moe_semantic_top_k', 2))
        
        # Routing stats untuk debugging
        self._routing_stats: Dict[str, int] = {}
        
        # Auto-setup router jika tersedia
        self._setup_router()

    # ========================================================================
    # MoE SETUP
    # ========================================================================

    def _setup_router(self) -> None:
        """
        Setup DomainRouter dari trainer (jika tersedia).
        
        Urutan pencarian:
        1. Dari trainer.get_router() (jika sudah di-build saat training)
        2. Dari cache file di output folder (jika ada)
        """
        if not _ROUTER_AVAILABLE:
            return
        
        if not self._use_moe:
            return
        
        # ============================================================
        # Opsi 1: Dari trainer
        # ============================================================
        if self.trainer is not None:
            router = None
            if hasattr(self.trainer, "get_router"):
                router = self.trainer.get_router()
            
            if router is not None:
                self._router = router
                print(f"[playground] MoE router dimuat dari trainer.")
                print(f"[playground] Domains tersedia: {list(router.domain_keywords.keys())}")
                return
        
        # ============================================================
        # Opsi 2: Dari cache file
        # ============================================================
        try:
            router = DomainRouter(self.config)
            
            keywords_path = Path(self.config.paths.output_models_dir) / "domain_keywords.json"
            cache_path = Path(self.config.paths.output_models_dir) / "semantic_cache.npz"
            
            keywords_loaded = router.load_keywords(keywords_path)
            cache_loaded = router.load_cache(cache_path)
            
            if keywords_loaded:
                self._router = router
                domains = list(router.domain_keywords.keys())
                print(f"[playground] MoE router dimuat dari cache.")
                print(f"[playground] Domains tersedia: {domains}")
                if cache_loaded:
                    print(f"[playground] Semantic cache dimuat: {len(router.search_engine.domain_embeddings)} domain")
            else:
                print(f"[playground] MoE router tidak tersedia (keywords belum dibangun).")
                print(f"[playground] Jalankan training dulu untuk membangun router.")
        except Exception as exc:
            print(f"[playground] Warning: Gagal setup router: {exc}")

    # ========================================================================
    # MoE ROUTING (BARU)
    # ========================================================================

    def _route_prompt(self, prompt: str) -> Tuple[str, float]:
        """
        Route prompt ke domain menggunakan router.
        
        Return:
        - (domain, confidence)
        - Jika router tidak tersedia, return (None, 0.0)
        """
        # Force domain punya prioritas tertinggi
        if self._force_domain is not None:
            return (self._force_domain, 1.0)
        
        # Manual domain prefix (untuk backward compat)
        if self._default_domain is not None:
            return (self._default_domain, 0.9)
        
        # Auto routing via router
        if self._router is not None and self._auto_routing:
            available_domains = None
            if hasattr(self.model, "list_domains"):
                available_domains = self.model.list_domains()
            
            domain, confidence, _ = self._router.route(
                prompt,
                available_domains=available_domains,
            )
            
            # Track stats
            self._routing_stats[domain] = self._routing_stats.get(domain, 0) + 1
            
            return (domain, confidence)
        
        return (None, 0.0)

    def _inject_semantic_context(self, prompt: str, domain: Optional[str]) -> Tuple[str, List[str]]:
        """
        Inject paragraf relevan ke prompt via semantic search.
        
        Return:
        - (augmented_prompt, context_paragraphs)
        """
        # Skip jika tidak ada router atau semantic search disabled
        if self._router is None:
            return (prompt, [])
        
        if self._semantic_top_k <= 0:
            return (prompt, [])
        
        if not hasattr(self._router, "search_engine"):
            return (prompt, [])
        
        if not self._router.search_engine.domain_embeddings:
            return (prompt, [])
        
        # Skip semantic search jika prompt sudah sangat panjang
        # (untuk menghemat context window)
        if len(prompt.split()) > 20:
            return (prompt, [])
        
        try:
            paragraphs, similarities = self._router.semantic_search(
                prompt=prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                domain=domain,
                top_k=self._semantic_top_k,
            )
            
            if not paragraphs:
                return (prompt, [])
            
            # Format context injection
            context_lines = []
            for para, sim in zip(paragraphs, similarities):
                # Truncate paragraf jika terlalu panjang
                para_truncated = para[:200] + "..." if len(para) > 200 else para
                context_lines.append(f"- {para_truncated}")
            
            context_text = "\n".join(context_lines)
            
            # Format: context dulu, lalu prompt
            augmented = f"Berdasarkan informasi berikut:\n{context_text}\n\nJawab pertanyaan: {prompt}"
            
            return (augmented, paragraphs)
            
        except Exception as exc:
            # Fallback: prompt asli tanpa context
            return (prompt, [])

    def _set_model_domain(self, domain: Optional[str]) -> None:
        """
        Set active domain di model (untuk MoE LM head selection).
        """
        if domain is None:
            return
        
        if not hasattr(self.model, "set_active_domain"):
            return
        
        try:
            self.model.set_active_domain(domain)
        except Exception:
            pass

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
        debug_tokens: bool = False,
    ) -> str:
        """
        Uji satu prompt dan cetak hasilnya.
        """
        answer, generated_ids, token_pieces, routing_info = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            min_new_tokens=min_new_tokens,
            return_debug=True,
        )

        if verbose:
            print("-" * 60)
            print("Prompt  :", prompt)
            
            # ============================================================
            # MoE ROUTING INFO
            # ============================================================
            if routing_info.get("domain"):
                domain = routing_info["domain"]
                confidence = routing_info.get("confidence", 0.0)
                print(f"Domain  : [{domain}] (confidence: {confidence:.2%})")
                
                if routing_info.get("semantic_context"):
                    n_ctx = len(routing_info["semantic_context"])
                    print(f"Context : {n_ctx} paragraf relevan di-inject")

            # ============================================================
            # DEBUG MODE: Tampilkan Token IDs dan Token Pieces
            # ============================================================
            if debug_tokens:
                print()
                print("[DEBUG] Token IDs    :")
                for i in range(0, len(generated_ids), 10):
                    chunk = generated_ids[i : i + 10]
                    print(f"         {i+1:>3}-{i+len(chunk):<3} : {chunk}")

                print()
                print("[DEBUG] Token Pieces :")
                for i in range(0, len(token_pieces), 10):
                    chunk = token_pieces[i : i + 10]
                    formatted = [f"'{p}'" for p in chunk]
                    print(f"         {i+1:>3}-{i+len(chunk):<3} : {formatted}")

                print()

            print("Jawaban :", answer if answer else "(kosong)")
            print(f"Tokens  : {len(generated_ids)}")
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
        return_debug: bool = False,
    ) -> Tuple[str, List[int], List[str], Dict]:
        """
        Membuat jawaban/lanjutan teks dari prompt.

        MoE INFERENCE FLOW:
        1. Route prompt ke domain (auto atau manual)
        2. Set active LM head di model
        3. Inject semantic context (jika tersedia)
        4. Generate dengan sampling parameters
        5. Stop saat EOS atau 
        """
        self._ensure_tokenizer_ready()

        # ================================================================
        # MoE ROUTING
        # ================================================================
        domain, confidence = self._route_prompt(prompt)
        
        # Set active domain di model (untuk MoE LM head)
        if domain is not None:
            self._set_model_domain(domain)

        # ================================================================
        # SEMANTIC CONTEXT INJECTION
        # ================================================================
        augmented_prompt = prompt
        semantic_context: List[str] = []
        
        if self._use_moe and domain is not None and self._auto_routing:
            augmented_prompt, semantic_context = self._inject_semantic_context(
                prompt, domain
            )

        # ================================================================
        # DOMAIN PREFIX (untuk backward compat)
        # ================================================================
        actual_prompt = augmented_prompt
        if self._default_domain is not None and not semantic_context:
            # Hanya tambahkan prefix jika tidak ada semantic context
            # (untuk menghindari duplikasi)
            if not DOMAIN_PREFIX_PATTERN.match(prompt.strip()):
                actual_prompt = f"[{self._default_domain}] {augmented_prompt}"

        prompt_ids = self.tokenizer.encode(
            actual_prompt,
            add_bos=True,
            add_eos=False,
        )

        if not prompt_ids:
            prompt_ids = [self.bos_id]

        current_ids = list(prompt_ids)
        forbidden_ids = self._get_forbidden_ids()

        max_new_tokens = max(0, int(max_new_tokens))
        min_new_tokens = max(0, int(min_new_tokens))

        # ================================================================
        # GENERATION LOOP
        # ================================================================
        for step in range(max_new_tokens):
            context = current_ids[-self.max_context :]
            input_ids = np.array([context], dtype=np.int64)

            logits = self.model.forward(input_ids, training=False)
            next_logits = np.asarray(logits[0, -1, :], dtype=np.float32)

            step_forbidden = forbidden_ids

            # Jika belum mencapai min_new_tokens, larang EOS dan endoftext
            if step < min_new_tokens:
                step_forbidden = list(forbidden_ids)
                if self.eos_id not in step_forbidden:
                    step_forbidden.append(self.eos_id)
                if self.endoftext_id not in step_forbidden:
                    step_forbidden.append(self.endoftext_id)

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

            # ============================================================
            # STOP CONDITIONS
            # ============================================================
            # Stop jika EOS
            if next_token == self.eos_id:
                break
            
            # Stop jika endoftext (document boundary)
            if next_token == self.endoftext_id:
                break

            current_ids.append(int(next_token))

        generated_ids = current_ids[len(prompt_ids) :]

        # ================================================================
        # DECODE: Token ID → Token String → Final Text
        # ================================================================
        token_pieces = self._get_token_pieces(generated_ids)

        answer_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # ================================================================
        # ROUTING INFO (untuk debug)
        # ================================================================
        routing_info = {
            "domain": domain,
            "confidence": confidence,
            "semantic_context": semantic_context,
            "auto_routing": self._auto_routing,
        }

        if return_debug:
            return answer_text, generated_ids, token_pieces, routing_info

        return answer_text, generated_ids, token_pieces, routing_info

    def _get_token_pieces(self, token_ids: List[int]) -> List[str]:
        """
        Ubah daftar token ID menjadi daftar token string (subword).
        """
        pieces = []
        id_to_token = getattr(self.tokenizer, "id_to_token", None)

        if id_to_token is None:
            for tid in token_ids:
                pieces.append(f"<{tid}>")
            return pieces

        for tid in token_ids:
            token_str = id_to_token.get(tid, f"<{tid}>")
            pieces.append(token_str)

        return pieces

    def interactive(
        self,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        repetition_window: int = DEFAULT_REPETITION_WINDOW,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        debug_tokens: bool = False,
    ) -> None:
        """
        Mode CLI interaktif.
        """
        print("=" * 60)
        print("Testing Playground - Inference Engine")
        print("=" * 60)
        print("Ketik prompt untuk inference.")
        print("Ketik /help untuk daftar perintah.")

        if debug_tokens:
            print(">>> DEBUG MODE AKTIF: Token IDs & Pieces akan ditampilkan <<<")
        
        if self._use_moe:
            print(f">>> MoE AKTIF: auto-routing = {self._auto_routing} <<<")

        print("=" * 60)
        print(f"temperature        : {temperature}")
        print(f"top_k              : {top_k}")
        print(f"top_p              : {top_p}")
        print(f"repetition_penalty : {repetition_penalty}")
        print(f"min_new_tokens     : {min_new_tokens}")
        print(f"max_new_tokens     : {max_new_tokens}")
        print(f"debug_tokens       : {debug_tokens}")
        print(f"use_moe            : {self._use_moe}")
        print(f"auto_routing       : {self._auto_routing}")
        if self._default_domain:
            print(f"default_domain     : [{self._default_domain}]")
        if self._force_domain:
            print(f"force_domain       : [{self._force_domain}]")
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
            answer, generated_ids, token_pieces, routing_info = self.generate(
                prompt=stripped,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                min_new_tokens=min_new_tokens,
                return_debug=True,
            )

            # ============================================================
            # MoE ROUTING INFO
            # ============================================================
            if routing_info.get("domain"):
                domain = routing_info["domain"]
                confidence = routing_info.get("confidence", 0.0)
                print(f"[Router] Domain: [{domain}] (confidence: {confidence:.2%})")
                
                if routing_info.get("semantic_context"):
                    n_ctx = len(routing_info["semantic_context"])
                    print(f"[Router] Context: {n_ctx} paragraf relevan di-inject")

            # ============================================================
            # DEBUG MODE
            # ============================================================
            if debug_tokens:
                print()
                print("[DEBUG] Token IDs    :")
                for i in range(0, len(generated_ids), 10):
                    chunk = generated_ids[i : i + 10]
                    print(f"         {i+1:>3}-{i+len(chunk):<3} : {chunk}")

                print()
                print("[DEBUG] Token Pieces :")
                for i in range(0, len(token_pieces), 10):
                    chunk = token_pieces[i : i + 10]
                    formatted = [f"'{p}'" for p in chunk]
                    print(f"         {i+1:>3}-{i+len(chunk):<3} : {formatted}")

                print()

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
        # /domain [NAMA] - Set domain prefix manual
        # ----------------------------------------------------------------
        if cmd == "/domain":
            self._handle_domain_command(parts)
            return False

        # ----------------------------------------------------------------
        # /domains - List semua domain yang tersedia (BARU)
        # ----------------------------------------------------------------
        if cmd == "/domains":
            self._handle_domains_command()
            return False

        # ----------------------------------------------------------------
        # /auto [on|off] - Toggle auto-routing MoE (BARU)
        # ----------------------------------------------------------------
        if cmd == "/auto":
            self._handle_auto_command(parts)
            return False

        # ----------------------------------------------------------------
        # /force DOMAIN - Force domain untuk testing (BARU)
        # ----------------------------------------------------------------
        if cmd == "/force":
            self._handle_force_command(parts)
            return False

        # ----------------------------------------------------------------
        # Perintah tidak dikenal
        # ----------------------------------------------------------------
        print(f"[playground] Perintah tidak dikenal: {cmd}")
        print("[playground] Ketik /help untuk daftar perintah.")
        return False

    # ========================================================================
    # DOMAIN COMMAND HANDLER
    # ========================================================================

    def _handle_domain_command(self, parts: List[str]) -> None:
        """
        Memproses perintah /domain.

        Format:
            /domain          → hapus domain default
            /domain MOBIL    → set domain default ke MOBIL
        """
        if len(parts) < 2:
            # Hapus domain default
            if self._default_domain:
                print(f"[playground] Domain default dihapus (sebelumnya: [{self._default_domain}])")
                self._default_domain = None
            else:
                print("[playground] Tidak ada domain default yang aktif.")
                print("[playground] Gunakan: /domain MOBIL, /domain SAINS, /domain FILSAFAT, dll.")
                print("[playground] Atau gunakan /domains untuk lihat domain yang tersedia.")
            return

        # Set domain baru
        domain = parts[1].upper().strip()
        
        # Validasi: hanya huruf kapital, angka, dan underscore
        if not re.match(r'^[A-Z0-9_]+$', domain):
            print(f"[playground] Domain tidak valid: {domain}")
            print("[playground] Domain harus huruf kapital/angka/underscore (contoh: MOBIL, SAINS)")
            return

        self._default_domain = domain
        print(f"[playground] Domain default diset ke: [{domain}]")
        print(f"[playground] Semua prompt akan otomatis di-prefix dengan [{domain}]")
        print(f"[playground] Gunakan /domain untuk menghapus domain default.")

    def _handle_domains_command(self) -> None:
        """
        Memproses perintah /domains.
        
        List semua domain yang tersedia (dari model dan router).
        """
        print("=" * 60)
        print("Daftar Domain yang Tersedia")
        print("=" * 60)
        
        # Dari model (LM heads)
        model_domains = []
        if hasattr(self.model, "list_domains"):
            try:
                model_domains = self.model.list_domains()
            except Exception:
                model_domains = []
        
        if model_domains:
            print(f"\nModel LM heads ({len(model_domains)}):")
            for domain in model_domains:
                marker = " ← aktif" if self._force_domain == domain else ""
                marker += " ← default" if self._default_domain == domain else ""
                print(f"  [{domain}]{marker}")
        else:
            print("\nModel LM heads: (tidak ada / MoE tidak aktif)")
        
        # Dari router (keywords)
        if self._router is not None:
            router_domains = list(self._router.domain_keywords.keys())
            if router_domains:
                print(f"\nRouter keywords ({len(router_domains)}):")
                for domain in router_domains:
                    n_keywords = len(self._router.domain_keywords.get(domain, []))
                    print(f"  [{domain}]: {n_keywords} keywords")
            else:
                print("\nRouter keywords: (belum dibangun)")
        else:
            print("\nRouter: (tidak tersedia)")
            print("  Jalankan training dulu untuk membangun router.")
        
        # Routing stats
        if self._routing_stats:
            print(f"\nRouting statistics (sesi ini):")
            for domain, count in sorted(
                self._routing_stats.items(),
                key=lambda x: -x[1]
            ):
                print(f"  [{domain}]: {count}x")
        
        print("=" * 60)

    def _handle_auto_command(self, parts: List[str]) -> None:
        """
        Memproses perintah /auto.
        
        Format:
            /auto          → tampilkan status auto-routing
            /auto on       → aktifkan auto-routing
            /auto off      → nonaktifkan auto-routing
        """
        if len(parts) < 2:
            # Tampilkan status
            status = "AKTIF" if self._auto_routing else "NONAKTIF"
            print(f"[playground] Auto-routing: {status}")
            print(f"[playground] Gunakan: /auto on atau /auto off")
            return
        
        mode = parts[1].lower().strip()
        
        if mode in {"on", "1", "true", "aktif"}:
            self._auto_routing = True
            print("[playground] Auto-routing MoE: AKTIF")
            print("[playground] Router akan otomatis memilih LM head berdasarkan prompt.")
        elif mode in {"off", "0", "false", "nonaktif", "mati"}:
            self._auto_routing = False
            print("[playground] Auto-routing MoE: NONAKTIF")
            print("[playground] Gunakan /domain untuk set domain manual.")
        else:
            print(f"[playground] Mode tidak valid: {mode}")
            print("[playground] Gunakan: /auto on atau /auto off")

    def _handle_force_command(self, parts: List[str]) -> None:
        """
        Memproses perintah /force.
        
        Format:
            /force          → hapus force domain
            /force MOBIL    → force domain ke MOBIL (bypass router)
        """
        if len(parts) < 2:
            # Hapus force domain
            if self._force_domain:
                print(f"[playground] Force domain dihapus (sebelumnya: [{self._force_domain}])")
                self._force_domain = None
            else:
                print("[playground] Tidak ada force domain yang aktif.")
            return
        
        # Set force domain
        domain = parts[1].upper().strip()
        
        # Validasi
        if not re.match(r'^[A-Z0-9_]+$', domain):
            print(f"[playground] Domain tidak valid: {domain}")
            return
        
        # Cek apakah domain ada di model
        if hasattr(self.model, "has_domain"):
            try:
                if not self.model.has_domain(domain):
                    print(f"[playground] Warning: Domain [{domain}] tidak ada di model.")
                    print(f"[playground] Domain yang tersedia: {self.model.list_domains()}")
            except Exception:
                pass
        
        self._force_domain = domain
        self._set_model_domain(domain)
        print(f"[playground] Force domain diset ke: [{domain}]")
        print(f"[playground] Router akan di-bypass, semua prompt pakai LM head [{domain}].")
        print(f"[playground] Gunakan /force untuk menghapus force domain.")

    # ========================================================================
    # TRAINING COMMAND HANDLER
    # ========================================================================

    def _handle_training_command(self, parts: List[str]) -> None:
        """
        Memproses perintah /training dengan berbagai format argumen.
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
                    
                    # Reload router setelah training (karena mungkin sudah di-rebuild)
                    self._setup_router()
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
        """Parse argumen perintah /training."""
        result: Dict[str, Optional[int]] = {"steps": None, "epochs": None}

        i = 1
        while i < len(parts):
            arg = parts[i].lower()

            if arg in {"--steps", "--s", "-s"}:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    result["steps"] = int(parts[i + 1])
                    i += 2
                else:
                    i += 1

            elif arg in {"--epochs", "--e", "-e"}:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    result["epochs"] = int(parts[i + 1])
                    i += 2
                else:
                    i += 1

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
        """Tampilkan status model."""
        print("=" * 60)
        print("Status Model")
        print("=" * 60)

        print(f"  Parameter count : {self.model.parameter_count():,}")
        print(f"  Vocab size      : {self.tokenizer.vocab_size}")
        print(f"  Max context     : {self.max_context}")
        print(f"  Trainer         : {'Tersedia' if self.trainer is not None else 'Tidak tersedia'}")
        print(f"  Use MoE         : {self._use_moe}")
        print(f"  Auto-routing    : {self._auto_routing}")
        print(f"  Domain default  : {f'[{self._default_domain}]' if self._default_domain else 'Tidak diset'}")
        print(f"  Force domain    : {f'[{self._force_domain}]' if self._force_domain else 'Tidak diset'}")
        print(f"  Router          : {'Tersedia' if self._router is not None else 'Tidak tersedia'}")
        
        # LM heads info
        if hasattr(self.model, "list_domains"):
            try:
                domains = self.model.list_domains()
                print(f"  LM heads        : {len(domains)} domain")
                for domain in domains:
                    active_marker = " ← aktif" if self._force_domain == domain else ""
                    print(f"    [{domain}]{active_marker}")
            except Exception:
                pass

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
        """Tampilkan bantuan perintah interactive."""
        print("=" * 60)
        print("Bantuan Playground")
        print("=" * 60)
        print()
        print("Ketik prompt apa saja untuk inference.")
        print("Contoh: apa itu mobil")
        print("        [SAINS] jelaskan fotosintesis")
        print()
        print("Perintah (slash commands):")
        print()
        print("  /help                  : tampilkan bantuan ini")
        print()
        print("  MoE Commands (BARU):")
        print("  /domains               : list semua domain yang tersedia")
        print("  /auto                  : tampilkan status auto-routing")
        print("  /auto on               : aktifkan auto-routing")
        print("  /auto off              : nonaktifkan auto-routing")
        print("  /force MOBIL           : force domain ke MOBIL (bypass router)")
        print("  /force                 : hapus force domain")
        print()
        print("  Domain Commands (manual):")
        print("  /domain MOBIL          : set domain default ke MOBIL")
        print("  /domain SAINS          : set domain default ke SAINS")
        print("  /domain                : hapus domain default")
        print()
        print("  Training Commands:")
        print("  /training              : latih model (step default)")
        print("  /training 100          : latih model 100 step")
        print("  /training --steps 100  : latih model 100 step")
        print("  /training --epochs 10  : latih model 10 epoch")
        print("  /training --epochs 10 --steps 50")
        print("                         : latih 10 epoch, minimal 50 step")
        print()
        print("  Lainnya:")
        print("  /status                : tampilkan status model")
        print("  /exit                  : keluar dari playground")
        print("  /quit                  : keluar dari playground")
        print("  /keluar                : keluar dari playground")
        print()
        print("TIPS MoE INFERENCE:")
        print("  1. Pastikan training sudah dilakukan (router auto-built)")
        print("  2. Auto-routing aktif by default (/auto on)")
        print("  3. Ketik prompt biasa: 'fotosintesis adalah'")
        print("  4. Router otomatis pilih LM head SAINS")
        print("  5. Gunakan /domains untuk lihat domain yang tersedia")
        print("  6. Gunakan /force MOBIL untuk test domain spesifik")
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
        
        ANTI SEMANTIC DRIFT:
        - endoftext_id ditambahkan agar model tidak generate pemisah dokumen
        """
        if self._forbidden_cache is not None:
            return self._forbidden_cache

        forbidden = {
            self.pad_id,
            self.unk_id,
            self.bos_id,
            self.endoftext_id,  # ← PENTING: jangan generate endoftext
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
        """Softmax numerik stabil."""
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
        default=32,
        help="Jumlah token maksimum yang dihasilkan.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=4,
        help="Jumlah token minimum sebelum EOS diperbolehkan.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature sampling (lebih rendah = lebih deterministik).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K sampling (lebih kecil = lebih fokus).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-P sampling (lebih kecil = lebih fokus).",
    )
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Tampilkan Token IDs dan Token Pieces secara detail.",
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
                    "[MOBIL] Mobil adalah kendaraan beroda empat.",
                    "[MOBIL] Mobil menggunakan mesin sebagai penggerak.",
                    "[SAINS] Air adalah senyawa H2O.",
                    "[SAINS] Fotosintesis menghasilkan oksigen.",
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

        # CATATAN: dropout_rate ada di TrainingConfig, BUKAN ModelConfig
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
                use_moe=True,  # Aktifkan MoE untuk test
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
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                debug_tokens=args.debug_tokens,
            )
        else:
            playground.test_prompt(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                debug_tokens=args.debug_tokens,
            )

        print("testing/playground.py test OK")

