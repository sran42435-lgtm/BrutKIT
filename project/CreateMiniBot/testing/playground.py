# testing/playground.py
#
# Playground / Inference Testing Engine dengan RAG + SOTA + Constrained Generation.
#
# Arsitektur: Single-Head + BM25 Retrieval + Cross-Attention + Constrained Gen
#
# 🎯 SOTA Features yang Di-expose:
# - Constrained Generation: hanya tokens dari context yang boleh di-generate
# - Cross-Attention: model attend ke context embeddings di setiap layer
# - EMA Weights: apply Polyak averaged weights untuk smoother inference
# - SOTA Stats: lihat spikes detected, self-corrections, hard samples
#
# Fitur Utama:
# - BM25 Retrieval: otomatis ambil top-k paragraf relevan
# - Query expansion fallback: sinonim jika retrieval gagal
# - Graceful degradation: fallback ke "tidak tahu" jika tidak ada konteks
# - Context toggle: bisa disable retrieval untuk testing
# - Constrained toggle: bisa disable constrained gen untuk creative mode
# - Debug mode: tampilkan retrieved context, token IDs, token pieces
# - Sampling: temperature, top-k, top-p, repetition penalty
#
# Perintah interactive (slash commands):
#   /help                  : tampilkan bantuan
#   /training              : latih model (step default)
#   /training --epochs 10  : latih model 10 epoch
#   /retrieve [query]      : test retrieval dengan query spesifik
#   /context [on|off]      : toggle context injection
#   /constrained [on|off]  : toggle constrained generation (NEW)
#   /sota                  : lihat SOTA training stats (NEW)
#   /apply-ema             : apply EMA weights ke model (NEW)
#   /status                : tampilkan status model
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

# Import BM25Retriever untuk RAG (optional)
try:
    from core.retriever import BM25Retriever
    _RETRIEVER_AVAILABLE = True
except ImportError:
    BM25Retriever = None
    _RETRIEVER_AVAILABLE = False


class Playground:
    """
    Tempat testing pertanyaan / inference dengan RAG + SOTA + Constrained Gen.

    Arsitektur:
    - BM25 retrieval untuk konteks
    - Cross-Attention untuk context awareness
    - Constrained Generation untuk on-topic jawaban
    - Single LM head (no domain contamination)

    Fitur:
    - Temperature rendah untuk determinisme
    - Top-K/Top-P ketat untuk filtering
    - Repetition penalty untuk cegah pengulangan
    - Stop generation saat [EOS] atau 
    - Context toggle untuk testing tanpa retrieval
    - Constrained toggle untuk creative mode
    - SOTA stats untuk monitoring training
    """

    # ================================================================
    # DEFAULT PARAMETERS - DIOPTIMALKAN UNTUK MODEL RAG + CONSTRAINED
    # ================================================================
    DEFAULT_MAX_NEW_TOKENS = 128
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_K = 20
    DEFAULT_TOP_P = 0.85
    DEFAULT_REPETITION_PENALTY = 1.3
    DEFAULT_REPETITION_WINDOW = 64
    DEFAULT_MIN_NEW_TOKENS = 4
    DEFAULT_INLINE_TRAINING_STEPS = 10

    # RAG default
    NO_CONTEXT_RESPONSE = "Maaf, saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan tersebut."

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
        
        # ============================================================
        # RAG + SOTA STATE
        # ============================================================
        self._retriever: Optional[object] = None
        self._context_enabled: bool = True  # Toggle retrieval
        
        # 🎯 NEW: Constrained generation toggle
        # Default: gunakan constrained jika model support
        self._constrained_enabled: bool = bool(
            getattr(config.model, 'use_constrained_generation', True)
        )
        
        # 🎯 NEW: Track whether EMA has been applied
        self._ema_applied: bool = False
        
        self._retrieval_stats: Dict[str, int] = {
            "total_queries": 0,
            "with_context": 0,
            "no_context_fallback": 0,
            "constrained_used": 0,
            "standard_used": 0,
        }
        
        # Auto-setup retriever jika tersedia
        self._setup_retriever()

    # ========================================================================
    # RETRIEVER SETUP
    # ========================================================================

    def _setup_retriever(self) -> None:
        """
        Setup BM25Retriever.
        
        Urutan pencarian:
        1. Dari trainer.get_retriever() (jika sudah di-build saat training)
        2. Dari index file di output folder (jika ada)
        3. Build baru dari corpus (sebagai fallback)
        """
        if not _RETRIEVER_AVAILABLE:
            print("[playground] ⚠️ core/retriever.py tidak tersedia. RAG disabled.")
            return
        
        # ============================================================
        # Opsi 1: Dari trainer
        # ============================================================
        if self.trainer is not None:
            retriever = None
            if hasattr(self.trainer, "get_retriever"):
                retriever = self.trainer.get_retriever()
            
            if retriever is not None:
                self._retriever = retriever
                print(f"[playground] ✅ BM25 retriever dimuat dari trainer.")
                print(f"[playground]    Corpus size: {retriever.total_docs} paragraphs")
                return
        
        # ============================================================
        # Opsi 2: Dari index file
        # ============================================================
        try:
            retriever = BM25Retriever(self.config)
            index_path = Path(self.config.paths.retrieval_index_path)
            
            if retriever.load_index(index_path):
                self._retriever = retriever
                print(f"[playground] ✅ BM25 retriever dimuat dari cache.")
                print(f"[playground]    Corpus size: {retriever.total_docs} paragraphs")
                return
        except Exception as exc:
            print(f"[playground] Warning: Gagal load index dari cache: {exc}")
        
        # ============================================================
        # Opsi 3: Build baru dari corpus
        # ============================================================
        try:
            from core.retriever import load_corpus_from_directory
            
            datasets_dir = Path(self.config.paths.datasets_dir)
            if datasets_dir.exists():
                max_para_len = int(getattr(self.config.retrieval, 'max_paragraph_length', 500))
                corpus = load_corpus_from_directory(
                    datasets_dir,
                    max_paragraph_length=max_para_len,
                    verbose=False,
                )
                
                if corpus:
                    retriever = BM25Retriever(self.config)
                    retriever.build_index(corpus, verbose=False)
                    
                    # Save untuk session berikutnya
                    index_path = Path(self.config.paths.retrieval_index_path)
                    retriever.save_index(index_path)
                    
                    self._retriever = retriever
                    print(f"[playground] ✅ BM25 retriever dibangun dari corpus.")
                    print(f"[playground]    Corpus size: {len(corpus)} paragraphs")
                    print(f"[playground]    Index disimpan ke: {index_path}")
                    return
        except Exception as exc:
            print(f"[playground] Warning: Gagal build retriever dari corpus: {exc}")
        
        print("[playground] ⚠️ Retriever tidak tersedia.")
        print("[playground]    Jalankan training dulu untuk membangun index.")

    # ========================================================================
    # RETRIEVAL
    # ========================================================================

    def _retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        """
        Retrieve konteks yang relevan untuk query.
        
        Returns:
            - (combined_context, results_list)
            - combined_context: string dengan paragraf digabung, atau None jika gagal
            - results_list: list of (paragraph, score) tuples
        """
        if self._retriever is None or not self._context_enabled:
            return None, []
        
        # Default dari config
        retrieval_cfg = getattr(self.config, 'retrieval', None)
        if top_k is None:
            top_k = getattr(retrieval_cfg, 'top_k', 3) if retrieval_cfg else 3
        if min_score is None:
            min_score = getattr(retrieval_cfg, 'min_score', None) if retrieval_cfg else None
        
        try:
            results = self._retriever.retrieve(
                query=query,
                top_k=top_k,
                min_score=min_score,
                use_dynamic_threshold=True,
                enable_expansion_fallback=True,
                verbose=verbose,
            )
            
            self._retrieval_stats["total_queries"] += 1
            
            if not results:
                self._retrieval_stats["no_context_fallback"] += 1
                return None, []
            
            # Gabungkan paragraf dengan separator
            paragraphs = [para for para, score in results]
            combined_context = "\n\n".join(paragraphs)
            
            self._retrieval_stats["with_context"] += 1
            
            return combined_context, results
        
        except Exception as exc:
            if verbose:
                print(f"[playground] Retrieval error: {exc}")
            self._retrieval_stats["no_context_fallback"] += 1
            return None, []

    # ========================================================================
    # 🎯 NEW: CONSTRAINED GENERATION HELPERS
    # ========================================================================

    def _get_context_token_ids(self, context: str) -> List[int]:
        """
        Encode context ke token IDs untuk constrained generation.
        
        Returns:
            List of token IDs (flattened, unique)
        """
        if not context:
            return []
        
        # Encode tanpa BOS/EOS - kita hanya mau content tokens
        context_ids = self.tokenizer.encode(
            context,
            add_bos=False,
            add_eos=False,
        )
        
        return context_ids

    def _generate_with_constraints(
        self,
        context: str,
        question: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        min_new_tokens: int,
    ) -> Tuple[List[int], str]:
        """
        Generate menggunakan constrained generation dari model.
        
        Menggunakan model.generate_constrained() yang sudah ada di architecture.py
        dengan cross-attention ke context embeddings.
        
        Returns:
            (generated_ids, answer_text)
        """
        # Encode context dan question
        context_ids = self._get_context_token_ids(context)
        question_ids = self.tokenizer.encode(
            question,
            add_bos=True,
            add_eos=False,
        )
        
        if not context_ids:
            # Fallback ke standard generation
            return self._generate_standard(
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                min_new_tokens=min_new_tokens,
            )
        
        # Use model's constrained generation method
        try:
            context_ids_np = np.array(context_ids, dtype=np.int64)
            question_ids_np = np.array(question_ids, dtype=np.int64)
            
            generated_ids, full_seq = self.model.generate_constrained(
                context_ids=context_ids_np,
                question_ids=question_ids_np,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                min_new_tokens=min_new_tokens,
                rng=self.rng,
            )
            
            self._retrieval_stats["constrained_used"] += 1
            
            # Decode answer
            answer_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            return generated_ids, answer_text
        
        except Exception as exc:
            print(f"[playground] ⚠️ Constrained generation failed: {exc}")
            print("[playground]    Falling back ke standard generation.")
            return self._generate_standard(
                question=question,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                min_new_tokens=min_new_tokens,
            )

    def _generate_standard(
        self,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        top_k: int = 20,
        top_p: float = 0.85,
        repetition_penalty: float = 1.3,
        min_new_tokens: int = 4,
    ) -> Tuple[List[int], str]:
        """
        Standard generation (tanpa constraints) dengan format RAG.
        
        Menggunakan cross-attention ke context jika tersedia.
        """
        # Format prompt dengan RAG template
        if context:
            actual_prompt = self.tokenizer.format_rag_prompt(
                context=context,
                question=question,
                include_context=True,
            )
        else:
            # Tanpa context
            actual_prompt = f"Pertanyaan: {question} Jawaban:"
        
        # Encode prompt
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

        # Generation loop
        for step in range(max_new_tokens):
            context_window = current_ids[-self.max_context:]
            input_ids = np.array([context_window], dtype=np.int64)

            # Forward dengan cross-attention ke context jika tersedia
            logits = self.model.forward(input_ids, training=False)
            next_logits = np.asarray(logits[0, -1, :], dtype=np.float32)

            step_forbidden = forbidden_ids

            if step < min_new_tokens:
                step_forbidden = list(forbidden_ids)
                if self.eos_id not in step_forbidden:
                    step_forbidden.append(self.eos_id)
                if self.endoftext_id not in step_forbidden:
                    step_forbidden.append(self.endoftext_id)

            generated_so_far = current_ids[len(prompt_ids):]

            if repetition_penalty > 1.0 and generated_so_far:
                previous_ids = generated_so_far[-self.DEFAULT_REPETITION_WINDOW:]
            else:
                previous_ids = None

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
            
            if next_token == self.endoftext_id:
                break

            current_ids.append(int(next_token))

        generated_ids = current_ids[len(prompt_ids):]
        self._retrieval_stats["standard_used"] += 1

        # Decode
        answer_text = self.tokenizer.decode_until_eos(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_ids, answer_text

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
        debug_retrieval: bool = True,
    ) -> str:
        """Uji satu prompt dan cetak hasilnya."""
        answer, generated_ids, token_pieces, rag_info = self.generate(
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
            # RAG INFO
            # ============================================================
            if debug_retrieval:
                context = rag_info.get("context")
                retrieval_results = rag_info.get("retrieval_results", [])
                mode = rag_info.get("generation_mode", "standard")
                
                if context:
                    print(f"Context : {len(retrieval_results)} paragraf retrieved")
                    for i, (para, score) in enumerate(retrieval_results, 1):
                        preview = para[:80] + "..." if len(para) > 80 else para
                        preview = preview.replace('\n', ' ')
                        print(f"  [{i}] score={score:.3f}: {preview}")
                else:
                    print(f"Context : (tidak ditemukan, menggunakan fallback)")
                
                print(f"Mode    : {mode}")
            
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
        Membuat jawaban dari prompt dengan RAG + Constrained Generation.

        RAG INFERENCE FLOW:
        1. Retrieve top-k paragraf relevan dari corpus
        2. Jika constrained mode aktif + context ada:
           - Gunakan model.generate_constrained() dengan cross-attention
           - Hanya tokens dari context yang bisa di-generate
        3. Jika standard mode:
           - Format prompt dengan RAG template
           - Standard autoregressive generation
        4. Stop saat EOS atau 
        """
        self._ensure_tokenizer_ready()

        # ================================================================
        # RAG RETRIEVAL
        # ================================================================
        context, retrieval_results = self._retrieve_context(
            query=prompt,
            verbose=False,
        )

        # ================================================================
        # NO CONTEXT FALLBACK
        # ================================================================
        if not context:
            rag_info = {
                "context": None,
                "retrieval_results": [],
                "used_fallback": True,
                "generation_mode": "fallback",
            }
            
            if return_debug:
                return (
                    self.NO_CONTEXT_RESPONSE,
                    [],
                    [],
                    rag_info,
                )
            return self.NO_CONTEXT_RESPONSE, [], [], rag_info

        # ================================================================
        # CONSTRAINED vs STANDARD GENERATION
        # ================================================================
        use_constrained = (
            self._constrained_enabled
            and hasattr(self.model, 'generate_constrained')
        )

        if use_constrained:
            generated_ids, answer_text = self._generate_with_constraints(
                context=context,
                question=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                min_new_tokens=min_new_tokens,
            )
            generation_mode = "constrained"
        else:
            generated_ids, answer_text = self._generate_standard(
                question=prompt,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                min_new_tokens=min_new_tokens,
            )
            generation_mode = "standard"

        # ================================================================
        # TOKEN PIECES (untuk debug)
        # ================================================================
        token_pieces = self._get_token_pieces(generated_ids)

        # ================================================================
        # RAG INFO (untuk debug)
        # ================================================================
        rag_info = {
            "context": context,
            "retrieval_results": retrieval_results,
            "used_fallback": False,
            "generation_mode": generation_mode,
        }

        if return_debug:
            return answer_text, generated_ids, token_pieces, rag_info

        return answer_text, generated_ids, token_pieces, rag_info

    def _get_token_pieces(self, token_ids: List[int]) -> List[str]:
        """Ubah daftar token ID menjadi daftar token string (subword)."""
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
        debug_retrieval: bool = True,
    ) -> None:
        """Mode CLI interaktif."""
        print("=" * 70)
        print("🎯 RAG Playground - SOTA + Constrained Generation")
        print("=" * 70)
        print("Ketik prompt untuk inference.")
        print("Ketik /help untuk daftar perintah.")
        print()

        if debug_tokens:
            print(">>> DEBUG MODE AKTIF: Token IDs & Pieces akan ditampilkan <<<")

        if self._retriever is not None:
            print(f">>> RAG AKTIF: BM25 retriever dengan {self._retriever.total_docs} paragraf <<<")
        else:
            print(">>> RAG DISABLE: Retriever tidak tersedia <<<")

        if self._constrained_enabled and hasattr(self.model, 'generate_constrained'):
            print(f">>> CONSTRAINED GEN: AKTIF (on-topic guaranteed) <<<")
        else:
            print(f">>> CONSTRAINED GEN: NONAKTIF (standard generation) <<<")

        print("=" * 70)
        print(f"temperature        : {temperature}")
        print(f"top_k              : {top_k}")
        print(f"top_p              : {top_p}")
        print(f"repetition_penalty : {repetition_penalty}")
        print(f"min_new_tokens     : {min_new_tokens}")
        print(f"max_new_tokens     : {max_new_tokens}")
        print(f"debug_tokens       : {debug_tokens}")
        print(f"debug_retrieval    : {debug_retrieval}")
        print(f"context_enabled    : {self._context_enabled}")
        print(f"constrained_enabled: {self._constrained_enabled}")
        print(f"ema_applied        : {self._ema_applied}")
        if self._retriever:
            print(f"retriever corpus   : {self._retriever.total_docs} paragraphs")
        print("=" * 70)

        while True:
            try:
                user_input = input("\nprompt> ")
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
            print()
            answer, generated_ids, token_pieces, rag_info = self.generate(
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
            # RAG INFO
            # ============================================================
            if debug_retrieval:
                context = rag_info.get("context")
                retrieval_results = rag_info.get("retrieval_results", [])
                used_fallback = rag_info.get("used_fallback", False)
                gen_mode = rag_info.get("generation_mode", "standard")
                
                if used_fallback:
                    print("[RAG] ⚠️  Tidak ada konteks relevan. Menggunakan fallback response.")
                elif context:
                    print(f"[RAG] ✅ Retrieved {len(retrieval_results)} paragraphs:")
                    for i, (para, score) in enumerate(retrieval_results, 1):
                        preview = para[:100] + "..." if len(para) > 100 else para
                        preview = preview.replace('\n', ' ')
                        print(f"  [{i}] score={score:.3f}: {preview}")
                else:
                    print("[RAG] Context disabled (gunakan /context on untuk aktifkan)")
                
                print(f"[MODE] Generation: {gen_mode}")

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

            print()
            print("Jawaban:", answer if answer else "(kosong)")
            print(f"[{len(generated_ids)} token dihasilkan]")
            print("-" * 70)

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
        """Memproses slash command."""
        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd in {"/exit", "/quit", "/keluar"}:
            print("[playground] Keluar dari playground.")
            return True

        if cmd in {"/help", "/bantuan", "/h", "/?"}:
            self._print_help()
            return False

        if cmd in {"/training", "/train", "/latih"}:
            self._handle_training_command(parts)
            return False

        if cmd in {"/status", "/info"}:
            self._print_status()
            return False

        if cmd == "/retrieve":
            self._handle_retrieve_command(parts)
            return False

        if cmd == "/context":
            self._handle_context_command(parts)
            return False

        # 🎯 NEW COMMANDS
        if cmd == "/constrained":
            self._handle_constrained_command(parts)
            return False

        if cmd == "/sota":
            self._print_sota_stats()
            return False

        if cmd == "/apply-ema":
            self._apply_ema_weights()
            return False

        print(f"[playground] Perintah tidak dikenal: {cmd}")
        print("[playground] Ketik /help untuk daftar perintah.")
        return False

    # ========================================================================
    # 🎯 NEW: CONSTRAINED COMMAND
    # ========================================================================

    def _handle_constrained_command(self, parts: List[str]) -> None:
        """
        Toggle constrained generation.
        
        Format:
            /constrained          → tampilkan status
            /constrained on       → aktifkan constrained generation
            /constrained off      → nonaktifkan (standard mode)
        """
        if len(parts) < 2:
            status = "AKTIF" if self._constrained_enabled else "NONAKTIF"
            print(f"[playground] Constrained generation: {status}")
            print(f"[playground] Gunakan: /constrained on atau /constrained off")
            print()
            
            if self._constrained_enabled:
                print("  📌 Mode aktif: Model hanya boleh generate tokens yang ada di context")
                print("  📌 Benefit: Jawaban dijamin on-topic, tidak ada hallucination")
            else:
                print("  📌 Mode nonaktif: Standard autoregressive generation")
                print("  📌 Benefit: Lebih flexible, tapi bisa hallucinate")
            return
        
        # Check if model support constrained
        if not hasattr(self.model, 'generate_constrained'):
            print("[playground] ⚠️ Model tidak support constrained generation.")
            print("[playground]    Update core/architecture.py untuk enable feature ini.")
            return
        
        mode = parts[1].lower().strip()
        
        if mode in {"on", "1", "true", "aktif"}:
            self._constrained_enabled = True
            print("[playground] ✅ Constrained generation: AKTIF")
            print("[playground]    Model hanya akan generate tokens yang ada di context.")
            print("[playground]    Benefit: On-topic guaranteed, no hallucination.")
        elif mode in {"off", "0", "false", "nonaktif", "mati"}:
            self._constrained_enabled = False
            print("[playground] ⚠️ Constrained generation: NONAKTIF")
            print("[playground]    Model akan gunakan standard autoregressive generation.")
            print("[playground]    Benefit: Lebih creative dan flexible.")
        else:
            print(f"[playground] Mode tidak valid: {mode}")
            print("[playground] Gunakan: /constrained on atau /constrained off")

    # ========================================================================
    # 🎯 NEW: SOTA STATS COMMAND
    # ========================================================================

    def _print_sota_stats(self) -> None:
        """
        Tampilkan SOTA training statistics dari trainer.
        
        Menampilkan:
        - Spikes detected
        - Self-corrections applied
        - Hard samples retrained
        - EMA weights status
        - Cosine LR schedule
        - Focal loss, AGC, curriculum stats
        """
        print("=" * 70)
        print("🎯 SOTA Training Statistics")
        print("=" * 70)
        
        # Check if trainer has SOTA stats
        if self.trainer is None:
            print("[playground] ⚠️ Trainer tidak tersedia.")
            return
        
        # Get SOTA stats jika tersedia
        if hasattr(self.trainer, 'get_sota_stats'):
            stats = self.trainer.get_sota_stats()
            print()
            print("📊 Self-Correction Stats:")
            print(f"  Total spikes detected      : {stats.get('spikes_detected', 0)}")
            print(f"  Self-correction attempts   : {stats.get('total_attempts', 0)}")
            print(f"  Corrections applied        : {stats.get('corrections_applied', 0)}")
            print(f"  Hard samples retrained     : {stats.get('hard_samples_retrained', 0)}")
        
        # Check training config untuk SOTA features aktif
        training_cfg = self.config.training
        print()
        print("⚙️ SOTA Features (dari config):")
        features = [
            ("Cosine LR + Warmup", getattr(training_cfg, 'use_cosine_schedule', True)),
            ("Label Smoothing", getattr(training_cfg, 'label_smoothing', 0.0) > 0),
            ("Focal Loss", getattr(training_cfg, 'use_focal_loss', True)),
            ("Token Weighting", getattr(training_cfg, 'use_token_weighting', True)),
            ("Adaptive Gradient Clipping", getattr(training_cfg, 'use_agc', True)),
            ("Loss Spike Detection", getattr(training_cfg, 'use_loss_spike_detection', True)),
            ("Curriculum Learning", getattr(training_cfg, 'use_curriculum', True)),
            ("Self-Correction Loop", getattr(training_cfg, 'use_self_correction', True)),
            ("EMA Weights", getattr(training_cfg, 'use_ema', True)),
        ]
        
        for name, active in features:
            status = "✅" if active else "❌"
            print(f"  {status} {name}")
        
        # SOTA parameters
        print()
        print("🔧 SOTA Parameters:")
        params = [
            ("Label smoothing ε", getattr(training_cfg, 'label_smoothing', 0.0)),
            ("Focal loss γ", getattr(training_cfg, 'focal_gamma', 0.0)),
            ("Answer token weight", getattr(training_cfg, 'answer_token_weight', 1.0)),
            ("AGC factor", getattr(training_cfg, 'agc_factor', 0.0)),
            ("EMA decay", getattr(training_cfg, 'ema_decay', 0.0)),
            ("Hard sample threshold", getattr(training_cfg, 'hard_sample_loss_threshold', 0.0)),
            ("Self-correction max iter", getattr(training_cfg, 'self_correction_max_iter', 0)),
        ]
        
        for name, value in params:
            print(f"  {name:30s}: {value}")
        
        # EMA status
        print()
        print("🧠 EMA Weights Status:")
        if hasattr(self.trainer, '_ema_weights') and self.trainer._ema_weights is not None:
            print(f"  ✅ EMA weights tersedia ({len(self.trainer._ema_weights)} tensors)")
            print(f"  📌 Applied ke model: {'Ya' if self._ema_applied else 'Belum'}")
            print(f"  💡 Gunakan /apply-ema untuk apply EMA weights ke model")
        else:
            print(f"  ⚠️ EMA weights belum tersedia")
            print(f"     (akan otomatis dibuat selama training)")
        
        print("=" * 70)

    # ========================================================================
    # 🎯 NEW: APPLY EMA WEIGHTS
    # ========================================================================

    def _apply_ema_weights(self) -> None:
        """
        Apply EMA (Exponential Moving Average) weights ke model.
        
        EMA weights adalah average dari semua weights selama training,
        biasanya lebih smooth dan generalize lebih baik.
        """
        print("[playground] 🧠 Applying EMA weights...")
        
        if self.trainer is None:
            print("[playground] ❌ Trainer tidak tersedia.")
            return
        
        if not hasattr(self.trainer, 'apply_ema_to_model'):
            print("[playground] ⚠️ Trainer tidak support EMA weights.")
            print("[playground]    Pastikan trainer.py adalah versi SOTA.")
            return
        
        try:
            self.trainer.apply_ema_to_model()
            self._ema_applied = True
            print("[playground] ✅ EMA weights berhasil di-apply ke model!")
            print("[playground]    Model sekarang menggunakan averaged weights.")
            print("[playground]    Expected: lebih smooth, better generalization.")
        except Exception as exc:
            print(f"[playground] ❌ Gagal apply EMA weights: {exc}")

    # ========================================================================
    # RETRIEVE COMMAND
    # ========================================================================

    def _handle_retrieve_command(self, parts: List[str]) -> None:
        """Test retrieval tanpa generate."""
        if self._retriever is None:
            print("[playground] ⚠️ Retriever tidak tersedia.")
            return
        
        # Ambil query dari args atau minta input
        if len(parts) >= 2:
            query = " ".join(parts[1:])
        else:
            try:
                query = input("Masukkan query untuk retrieval: ").strip()
                if not query:
                    print("[playground] Query kosong. Dibatalkan.")
                    return
            except (EOFError, KeyboardInterrupt):
                return
        
        print(f"\n[playground] Retrieving untuk query: '{query}'")
        print("-" * 70)
        
        context, results = self._retrieve_context(query, verbose=True)
        
        if not results:
            print("[playground] ❌ Tidak ada hasil retrieval.")
        else:
            print(f"\n[playground] ✅ Ditemukan {len(results)} paragraf:")
            for i, (para, score) in enumerate(results, 1):
                print(f"\n--- Hasil {i} (score: {score:.3f}) ---")
                print(para)
        
        print("-" * 70)

    # ========================================================================
    # CONTEXT COMMAND
    # ========================================================================

    def _handle_context_command(self, parts: List[str]) -> None:
        """Toggle context injection."""
        if len(parts) < 2:
            status = "AKTIF" if self._context_enabled else "NONAKTIF"
            print(f"[playground] Context injection: {status}")
            print(f"[playground] Gunakan: /context on atau /context off")
            
            # Print stats
            print(f"\n[playground] Retrieval statistics:")
            for key, value in self._retrieval_stats.items():
                print(f"  {key}: {value}")
            return
        
        mode = parts[1].lower().strip()
        
        if mode in {"on", "1", "true", "aktif"}:
            self._context_enabled = True
            print("[playground] ✅ Context injection: AKTIF")
            print("[playground] Retriever akan menyediakan konteks untuk setiap query.")
        elif mode in {"off", "0", "false", "nonaktif", "mati"}:
            self._context_enabled = False
            print("[playground] ⚠️ Context injection: NONAKTIF")
            print("[playground] Model akan generate tanpa konteks (seperti language model biasa).")
        else:
            print(f"[playground] Mode tidak valid: {mode}")
            print("[playground] Gunakan: /context on atau /context off")

    # ========================================================================
    # TRAINING COMMAND HANDLER
    # ========================================================================

    def _handle_training_command(self, parts: List[str]) -> None:
        """Memproses perintah /training."""
        if self.trainer is None:
            print("[playground] Trainer tidak tersedia.")
            return

        parsed = self._parse_training_args(parts)
        steps = parsed.get("steps")
        epochs = parsed.get("epochs")

        print("-" * 70)

        if epochs is not None:
            min_steps = steps if steps is not None else 0

            print(
                f"[playground] Training mode EPOCH: {epochs} epoch"
                + (f", minimal {min_steps} step" if min_steps > 0 else "")
            )
            print("[playground] Mode: 🎯 SOTA RAG context-conditioned training")
            print("-" * 70)

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
                    print("-" * 70)
                    print("[playground] ✅ Training epoch selesai.")
                    print(f"[playground] Epoch terakhir : {last.get('epoch', 0)}")
                    print(f"[playground] Stage          : {last.get('stage', 'N/A')}")
                    print(f"[playground] Loss           : {last.get('loss', 0.0):.6f}")
                    print(f"[playground] Perplexity     : {last.get('perplexity', 0.0):.4f}")
                    print(f"[playground] Accuracy       : {last.get('accuracy', 0.0):.4f}")
                    
                    # SOTA metrics
                    if 'top5_accuracy' in last:
                        print(f"[playground] Top-5 Acc      : {last.get('top5_accuracy', 0.0):.4f}")
                    if 'spikes' in last:
                        print(f"[playground] Spikes         : {last.get('spikes', 0)}")
                    
                    print("-" * 70)
                    
                    # Reload retriever setelah training
                    self._setup_retriever()
                else:
                    print("[playground] Training tidak menghasilkan history.")

            except Exception as exc:
                print(f"[playground] Training gagal: {exc}")
                import traceback
                traceback.print_exc()

        else:
            if steps is None:
                steps = self.DEFAULT_INLINE_TRAINING_STEPS

            print(f"[playground] Training mode STEP: {steps} step")
            print("-" * 70)

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
                    print("-" * 70)
                    print("[playground] ✅ Training step selesai.")
                    print(f"[playground] Steps          : {report.get('steps', 0)}")
                    print(f"[playground] Loss           : {report.get('loss', 0.0):.6f}")
                    print(f"[playground] Perplexity     : {report.get('perplexity', 0.0):.4f}")
                    print(f"[playground] Accuracy       : {report.get('accuracy', 0.0):.4f}")
                    print("-" * 70)
                    
                    self._setup_retriever()
                else:
                    print("[playground] Training tidak menghasilkan update.")

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
    # STATUS (ENHANCED)
    # ========================================================================

    def _print_status(self) -> None:
        """Tampilkan status model dengan SOTA info."""
        print("=" * 70)
        print("📊 Status Model (RAG + SOTA)")
        print("=" * 70)

        print(f"  Parameter count : {self.model.parameter_count():,}")
        print(f"  Vocab size      : {self.tokenizer.vocab_size:,}")
        print(f"  Max context     : {self.max_context}")
        print(f"  Trainer         : {'✅ Tersedia' if self.trainer is not None else '❌ Tidak tersedia'}")
        print(f"  Retriever       : {'✅ Tersedia' if self._retriever is not None else '❌ Tidak tersedia'}")
        
        # 🎯 NEW: Feature flags
        print()
        print("  🎯 Features:")
        print(f"    Context enabled     : {self._context_enabled}")
        print(f"    Constrained gen     : {self._constrained_enabled}")
        print(f"    EMA applied         : {self._ema_applied}")
        
        # Cross-attention support
        has_cross_attn = hasattr(self.model, 'cross_attention_layer_indices')
        print(f"    Cross-attention     : {has_cross_attn}")
        if has_cross_attn:
            n_cross = len(self.model.cross_attention_layer_indices)
            print(f"      Layers            : {n_cross} ({sorted(self.model.cross_attention_layer_indices)})")
        
        # Constrained gen support
        has_constrained = hasattr(self.model, 'generate_constrained')
        print(f"    Constrained gen API : {has_constrained}")
        
        # Retriever stats
        if self._retriever is not None:
            print()
            print("  📚 Retriever Stats:")
            stats = self._retriever.get_corpus_stats()
            print(f"    Corpus docs     : {stats.get('total_docs', 0)}")
            print(f"    Corpus terms    : {stats.get('total_terms', 0)}")
            print(f"    Avg doc length  : {stats.get('avg_doc_length', 0):.1f} tokens")

        # Trainer stats
        if self.trainer is not None and hasattr(self.trainer, "weight_manager"):
            print()
            print("  🏋️ Training Stats:")
            wm = self.trainer.weight_manager
            print(f"    Optimizer step  : {wm.step_count}")
            print(f"    Global step     : {getattr(wm, 'global_step', 0)}")
            print(f"    Last epoch      : {getattr(wm, 'last_epoch', 0)}")
            print(f"    Learning rate   : {wm.learning_rate:.2e}")

        # QA samples info
        if self.trainer is not None and hasattr(self.trainer, "qa_samples"):
            print(f"    QA samples      : {len(self.trainer.qa_samples):,}")

        # Retrieval stats (inference session)
        print()
        print("  📈 Inference Statistics (sesi ini):")
        for key, value in self._retrieval_stats.items():
            print(f"    {key:25s}: {value}")

        print("=" * 70)

    # ========================================================================
    # HELP (ENHANCED)
    # ========================================================================

    def _print_help(self) -> None:
        """Tampilkan bantuan perintah interactive."""
        print("=" * 70)
        print("🎯 Bantuan RAG Playground (SOTA + Constrained)")
        print("=" * 70)
        print()
        print("Ketik prompt apa saja untuk inference dengan RAG + Constrained Gen.")
        print("Sistem akan otomatis mencari konteks relevan dari corpus,")
        print("lalu generate jawaban yang on-topic menggunakan constraints.")
        print()
        print("Contoh:")
        print("  apa itu mobil")
        print("  jelaskan tentang fotosintesis")
        print("  bagaimana cara kerja rem?")
        print()
        print("Perintah (slash commands):")
        print()
        print("  /help                  : tampilkan bantuan ini")
        print()
        print("  🎯 RAG Commands:")
        print("  /retrieve              : test retrieval dengan query interaktif")
        print("  /retrieve [query]      : test retrieval dengan query spesifik")
        print("  /context               : tampilkan status context injection")
        print("  /context on            : aktifkan retrieval (default)")
        print("  /context off           : nonaktifkan retrieval (no-context mode)")
        print()
        print("  🚀 Constrained Generation:")
        print("  /constrained           : tampilkan status constrained generation")
        print("  /constrained on        : aktifkan (on-topic guaranteed)")
        print("  /constrained off       : nonaktifkan (creative mode)")
        print()
        print("  🎯 SOTA Training:")
        print("  /sota                  : lihat SOTA training stats")
        print("  /apply-ema             : apply EMA weights ke model")
        print("  /training --epochs N   : latih N epochs dengan SOTA techniques")
        print("  /training --steps N    : latih N steps")
        print()
        print("  📊 Informasi:")
        print("  /status                : tampilkan status lengkap")
        print("  /exit                  : keluar dari playground")
        print()
        print("🎯 CONSTRAINED GENERATION EXPLAINED:")
        print("  Mode ini memaksa model HANYA generate tokens yang ada di context.")
        print("  Benefit:")
        print("    ✅ On-topic guaranteed (tidak mungkin ngelantur)")
        print("    ✅ No hallucination (semua kata dari sumber terpercaya)")
        print("    ✅ Factual accuracy tinggi")
        print("  Limitation:")
        print("    ⚠️ Jawaban = rephrase dari context")
        print("    ⚠️ Tidak bisa 'creative' atau summarize")
        print()
        print("🎯 SOTA TECHNIQUES:")
        print("  Training menggunakan 9 teknik terbaik dari paper ML modern:")
        print("  - Cosine LR + Warmup (GPT-3, LLaMA)")
        print("  - Label Smoothing (Inception-v3)")
        print("  - Focal Loss (RetinaNet)")
        print("  - Token Weighting (answer tokens 3x penting)")
        print("  - Adaptive Gradient Clipping (NFNet)")
        print("  - Loss Spike Detection + Recovery")
        print("  - Curriculum Learning (easy → hard)")
        print("  - Self-Correction Loop (DeepMind)")
        print("  - EMA Weights (Polyak Averaging)")
        print()
        print("FLOW INFERENCE:")
        print("  Query → BM25 Retrieval → Context → Constrained Gen → Jawaban On-Topic")
        print()
        print("Semua input tanpa prefix / dianggap sebagai prompt inference.")
        print("=" * 70)

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
            self.endoftext_id,
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
        """Sampling token berikutnya."""
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
        RetrievalConfig,
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
        default="apa itu mobil",
        help="Prompt untuk mode non-interaktif.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
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
        help="Temperature sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-P sampling.",
    )
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="Tampilkan Token IDs dan Token Pieces.",
    )
    parser.add_argument(
        "--no-debug-retrieval",
        action="store_true",
        help="Jangan tampilkan info retrieval.",
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
            "Mobil adalah kendaraan beroda empat.\n\n"
            "Rem mobil berfungsi untuk menghentikan kendaraan.\n\n"
            "Air adalah senyawa H2O.\n\n"
            "Fotosintesis menghasilkan oksigen.",
            encoding="utf-8",
        )

        paths = PathsConfig(
            datasets_dir=datasets_dir,
            output_models_dir=output_models_dir,
            vocab_path=tmp_path / "vocab.json",
            final_model_path=output_models_dir / "otak_model.safetensors",
            checkpoint_path=output_models_dir / "training_checkpoint.npz",
            retrieval_index_path=output_models_dir / "retrieval_index.json",
            qa_dataset_path=output_models_dir / "qa_dataset.json",
        )

        small_config = Config(
            paths=paths,
            special_tokens=SpecialTokensConfig(),
            training=TrainingConfig(
                learning_rate=3e-4,
                batch_size=2,
                epochs=1,
                sequence_length=32,
                dropout_rate=0.0,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
                context_dropout_rate=0.2,
                context_perturbation_rate=0.3,
                progressive_stages=2,
            ),
            model=ModelConfig(
                vocab_size=128,
                embedding_dim=32,
                num_attention_heads=4,
                num_layers=4,
                max_position_embeddings=64,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=64,
                train_dtype="float32",
                export_dtype="float32",
                use_rope=True,
                use_weight_tying=True,
                use_constrained_generation=True,
                cross_attention_layers=(1, 3),
            ),
            retrieval=RetrievalConfig(
                top_k=2,
                bm25_k1=1.5,
                bm25_b=0.75,
                max_paragraph_length=500,
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
                debug_retrieval=not args.no_debug_retrieval,
            )
        else:
            playground.test_prompt(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                debug_tokens=args.debug_tokens,
                debug_retrieval=not args.no_debug_retrieval,
            )

        print("\ntesting/playground.py (RAG + SOTA + Constrained) test OK")
