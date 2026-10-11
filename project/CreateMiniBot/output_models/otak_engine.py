#!/usr/bin/env python3
# core/engine_standalone.py
#
# ============================================================================
# OTAK MINI - STANDALONE INFERENCE ENGINE
# ============================================================================
#
# Self-contained inference engine untuk deployment model CreateMiniBot.
# File ini TIDAK bergantung pada modul project lain - hanya butuh NumPy.
#
# Fitur:
# - Load model dari safetensors (format custom)
# - MoE auto-routing (keyword-based domain detection)
# - Semantic context injection (cosine similarity search)
# - Multiple LM heads (satu per domain)
# - Interactive CLI untuk testing
#
# Dependencies:
# - NumPy (satu-satunya dependency eksternal)
# - Python 3.8+
#
# Cara pakai:
#   # Interactive mode
#   python otak_engine.py --model otak_model.safetensors
#
#   # Single prompt
#   python otak_engine.py --model otak_model.safetensors --prompt "apa itu fotosintesis?"
#
#   # Sebagai module
#   from otak_engine import MiniEngine
#   engine = MiniEngine("otak_model.safetensors")
#   answer = engine.generate("apa itu fotosintesis?")
#
# Struktur file safetensors yang diharapkan:
#   [8 byte header length]
#   [JSON header dengan tensor metadata]
#   [raw tensor bytes]
#
# Metadata yang dibaca:
#   - tokenizer       : JSON string dengan vocab BPE
#   - moe_keywords    : JSON string dengan keywords per domain
#   - moe_paragraphs  : JSON string dengan paragraf per domain
#   - use_moe         : flag MoE aktif
#
# Tensors yang dibaca:
#   - embed_tokens.weight
#   - layers.{i}.input_layernorm.weight
#   - layers.{i}.self_attn.{q,k,v,o}_proj.weight
#   - layers.{i}.post_attention_layernorm.weight
#   - layers.{i}.mlp.{gate,up,down}_proj.weight
#   - norm.weight
#   - lm_heads.{DOMAIN}.weight
#   - __semantic_embeddings__.{DOMAIN}
#
# ============================================================================

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy diperlukan. Install dengan: pip install numpy")
    sys.exit(1)


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def _bf16_bytes_to_float32(raw: bytes) -> np.ndarray:
    """Konversi raw bfloat16 -> float32."""
    u16 = np.frombuffer(raw, dtype=np.dtype("<u2"))
    u32 = u16.astype(np.dtype("<u4")) << 16
    f32 = u32.view(np.dtype("<f4"))
    return f32.astype(np.float32)


def _unpack_int4(raw: bytes, shape: Tuple[int, ...], scale: float) -> np.ndarray:
    """Membuka kembali int4 packed menjadi float32."""
    numel = int(np.prod(shape)) if len(shape) > 0 else 1
    if numel == 0:
        return np.empty(shape, dtype=np.float32)

    buf = np.frombuffer(raw, dtype=np.uint8)
    low = (buf & 0x0F).astype(np.int8)
    high = ((buf >> 4) & 0x0F).astype(np.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16

    q = np.empty(buf.size * 2, dtype=np.int8)
    q[0::2] = low
    q[1::2] = high
    q = q[:numel]

    return q.astype(np.float32) * float(scale)


def load_safetensors(filepath: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Memuat file model biner .safetensors-like.

    Return:
    - state_dict: dict nama tensor -> numpy array
    - metadata: dict metadata string
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {filepath}")

    with open(filepath, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("File model tidak valid: header terlalu pendek.")

        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        tensor_bytes = f.read()

    header = json.loads(header_bytes.decode("utf-8"))
    metadata = header.get("__metadata__", {})

    # Load quantization info jika ada
    quant_raw = metadata.get("quantization", "{}")
    try:
        quant_meta = json.loads(quant_raw) if isinstance(quant_raw, str) else {}
    except Exception:
        quant_meta = {}

    state_dict: Dict[str, np.ndarray] = {}

    for name, meta in header.items():
        if name == "__metadata__":
            continue

        dtype = meta.get("dtype")
        shape = tuple(meta.get("shape", []))
        start, end = meta.get("data_offsets", [0, 0])
        raw = tensor_bytes[start:end]

        if dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.dtype("<f4")).astype(np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.dtype("<f2")).astype(np.float32)
        elif dtype == "BF16":
            arr = _bf16_bytes_to_float32(raw)
        elif dtype == "I8":
            q = np.frombuffer(raw, dtype=np.int8)
            scale = float(quant_meta.get(name, {}).get("scale", 1.0))
            arr = q.astype(np.float32) * scale
        elif dtype == "I4":
            scale = float(quant_meta.get(name, {}).get("scale", 1.0))
            arr = _unpack_int4(raw, shape, scale)
        else:
            raise ValueError(f"Dtype tensor tidak dikenal: {dtype}")

        state_dict[name] = arr.reshape(shape).astype(np.float32, copy=False)

    return state_dict, metadata


# ============================================================================
# NUMERIC HELPERS
# ============================================================================

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax numerik stabil."""
    x = np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.nan_to_num(x_max, nan=0.0, posinf=50.0, neginf=-50.0)
    shifted = np.clip(x - x_max, -50.0, 50.0)
    exp_x = np.exp(shifted)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1e-10)
    return exp_x / sum_exp


def _sanitize(x: np.ndarray, clip: float = 50.0) -> np.ndarray:
    """Sanitasi array dari NaN/Inf."""
    return np.nan_to_num(
        x, nan=0.0, posinf=clip, neginf=-clip
    ).astype(np.float32, copy=False)


def _softcap(x: np.ndarray, cap: float) -> np.ndarray:
    """Soft-capping berbasis tanh."""
    return cap * np.tanh(x / cap).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity antara dua vektor."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.shape != b.shape:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# RoPE HELPERS
# ============================================================================

def _build_rope_cache(max_position: int, head_dim: int, base: float = 10000.0):
    """Bangun cache cos dan sin untuk RoPE."""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_position, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    cos = np.cos(angles)
    sin = np.sin(angles)
    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)
    return cos.astype(np.float32), sin.astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotasi separuh dimensi terakhir."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_rotary_emb(q, k, cos, sin):
    """Terapkan RoPE pada Q dan K."""
    cos_expanded = cos[np.newaxis, np.newaxis, :, :]
    sin_expanded = sin[np.newaxis, np.newaxis, :, :]
    q_rot = q * cos_expanded + _rotate_half(q) * sin_expanded
    k_rot = k * cos_expanded + _rotate_half(k) * sin_expanded
    return q_rot.astype(np.float32), k_rot.astype(np.float32)


# ============================================================================
# MINI TOKENIZER
# ============================================================================

class MiniTokenizer:
    """
    Tokenizer BPE sederhana untuk inference.
    
    Dibangun dari metadata tokenizer di safetensors.
    """

    SPACE_MARKER = "▁"

    def __init__(self, tokenizer_payload: dict):
        self.token_to_id: Dict[str, int] = {
            str(k): int(v) for k, v in tokenizer_payload.get("token_to_id", {}).items()
        }
        self.id_to_token: Dict[int, str] = {
            v: k for k, v in self.token_to_id.items()
        }
        
        # Load merges untuk BPE
        self.merges: List[Tuple[str, str]] = []
        for pair in tokenizer_payload.get("merges", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                self.merges.append((str(pair[0]), str(pair[1])))
        
        self.merge_ranks: Dict[Tuple[str, str], int] = {
            pair: idx for idx, pair in enumerate(self.merges)
        }
        
        # Special tokens
        special = tokenizer_payload.get("special_tokens", {})
        self.pad_token = special.get("pad", "[PAD]")
        self.unk_token = special.get("unk", "[UNK]")
        self.bos_token = special.get("bos", "[BOS]")
        self.eos_token = special.get("eos", "[EOS]")
        self.endoftext_token = special.get("endoftext", "<|endoftext|>")
        
        self.pad_id = self.token_to_id.get(self.pad_token, 0)
        self.unk_id = self.token_to_id.get(self.unk_token, 1)
        self.bos_id = self.token_to_id.get(self.bos_token, 2)
        self.eos_id = self.token_to_id.get(self.eos_token, 3)
        self.endoftext_id = self.token_to_id.get(self.endoftext_token, 4)
        
        self._special_set = frozenset({
            self.pad_token, self.unk_token, self.bos_token,
            self.eos_token, self.endoftext_token,
        })

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _normalize_text(self, text: str) -> str:
        import re
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_valid_char(self, ch: str) -> bool:
        if ch in (" ", "\n", "\r", "\t"):
            return False
        return ch.isprintable()

    def _apply_bpe(self, tokens: List[str]) -> List[str]:
        """Terapkan BPE merges."""
        if len(tokens) <= 1 or not self.merge_ranks:
            return tokens

        current_tokens = list(tokens)

        while len(current_tokens) > 1:
            best_pair: Optional[Tuple[str, str]] = None
            best_rank: Optional[int] = None

            for i in range(len(current_tokens) - 1):
                a = current_tokens[i]
                b = current_tokens[i + 1]

                if a in self._special_set or b in self._special_set:
                    continue

                pair = (a, b)
                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_token = best_pair[0] + best_pair[1]
            merged_tokens: List[str] = []
            i = 0

            while i < len(current_tokens):
                if (
                    i < len(current_tokens) - 1
                    and current_tokens[i] == best_pair[0]
                    and current_tokens[i + 1] == best_pair[1]
                ):
                    merged_tokens.append(new_token)
                    i += 2
                else:
                    merged_tokens.append(current_tokens[i])
                    i += 1

            current_tokens = merged_tokens

        return current_tokens

    def _encode_text_part(self, text: str) -> List[str]:
        """Encode bagian teks biasa."""
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        words = normalized.split(" ")
        all_tokens: List[str] = []

        for idx, word in enumerate(words):
            base_tokens: List[str] = []

            if idx > 0:
                base_tokens.append(self.SPACE_MARKER)

            for ch in word:
                if self._is_valid_char(ch):
                    base_tokens.append(ch)

            if not base_tokens or base_tokens == [self.SPACE_MARKER]:
                continue

            base_tokens = [
                token if token in self.token_to_id else self.unk_token
                for token in base_tokens
            ]

            all_tokens.extend(self._apply_bpe(base_tokens))

        return all_tokens

    def encode_to_tokens(self, text: str) -> List[str]:
        """Encode teks ke token strings dengan deteksi endoftext."""
        text = str(text)
        if not text.strip():
            return []

        # Guard untuk endoftext kosong
        if not self.endoftext_token:
            return self._encode_text_part(text)

        try:
            parts = text.split(self.endoftext_token)
        except ValueError:
            return self._encode_text_part(text)

        all_tokens: List[str] = []
        for idx, part in enumerate(parts):
            part_tokens = self._encode_text_part(part)
            all_tokens.extend(part_tokens)
            if idx < len(parts) - 1:
                all_tokens.append(self.endoftext_token)

        return all_tokens

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode teks ke token IDs."""
        tokens = self.encode_to_tokens(text)
        ids: List[int] = [
            self.token_to_id.get(token, self.unk_id) for token in tokens
        ]

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs ke teks."""
        import re
        pieces: List[str] = []

        for token_id in token_ids:
            try:
                tid = int(token_id)
            except (TypeError, ValueError):
                continue

            token = self.id_to_token.get(tid)
            if token is None:
                continue

            if skip_special_tokens:
                if token in self._special_set:
                    continue
                if token.startswith("[UNUSED_"):
                    continue

            pieces.append(token)

        text = "".join(pieces)
        text = text.replace(self.SPACE_MARKER, " ")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def get_token_pieces(self, token_ids: List[int]) -> List[str]:
        """Return token strings untuk debugging."""
        pieces = []
        for tid in token_ids:
            token = self.id_to_token.get(int(tid), f"<{tid}>")
            pieces.append(token)
        return pieces


# ============================================================================
# MINI ROUTER (MoE)
# ============================================================================

class MiniRouter:
    """
    Router sederhana untuk MoE auto-routing.
    
    Dibangun dari metadata moe_keywords dan semantic embeddings di safetensors.
    """

    def __init__(
        self,
        keywords: Dict[str, List[str]],
        paragraphs: Dict[str, List[str]],
        embeddings: Dict[str, np.ndarray],
        confidence_threshold: float = 0.3,
    ):
        self.keywords = keywords
        self.paragraphs = paragraphs
        self.embeddings = embeddings
        self.confidence_threshold = confidence_threshold

    def score_prompt(self, prompt: str) -> Dict[str, float]:
        """Hitung score setiap domain untuk prompt."""
        import re
        # Tokenisasi sederhana
        text = prompt.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        prompt_words = set(text.split())

        if not prompt_words:
            return {}

        scores = {}
        for domain, keywords in self.keywords.items():
            score = 0.0
            for word in prompt_words:
                if word in keywords:
                    # Weight menurun berdasarkan posisi di list keywords
                    idx = keywords.index(word)
                    weight = 1.0 - (idx / (2 * len(keywords)))
                    score += weight
            scores[domain] = score

        return scores

    def route(self, prompt: str) -> Tuple[str, float]:
        """
        Route prompt ke domain.
        
        Return:
        - (domain, confidence)
        - Jika tidak ada match, return ("GENERAL", 0.0)
        """
        if not self.keywords:
            return ("GENERAL", 0.0)

        scores = self.score_prompt(prompt)
        if not scores:
            return ("GENERAL", 0.0)

        # Normalize
        max_score = max(scores.values())
        if max_score <= 0:
            return ("GENERAL", 0.0)

        normalized = {d: s / max_score for d, s in scores.items()}
        best_domain = max(normalized, key=normalized.get)
        best_score = normalized[best_domain]

        # Fallback jika confidence terlalu rendah
        if best_score < self.confidence_threshold:
            return ("GENERAL", best_score)

        return (best_domain, best_score)

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        domain: Optional[str] = None,
        top_k: int = 2,
    ) -> Tuple[List[str], List[float]]:
        """
        Cari paragraf paling mirip dengan query embedding.
        
        Return:
        - (paragraphs, similarities)
        """
        if not self.embeddings:
            return [], []

        # Tentukan domain yang dicari
        if domain is not None and domain in self.embeddings:
            search_domains = [domain]
        else:
            search_domains = list(self.embeddings.keys())

        if not search_domains:
            return [], []

        # Kumpulkan semua embeddings dan paragraphs
        all_embeddings = []
        all_paragraphs = []

        for d in search_domains:
            if d not in self.embeddings:
                continue
            all_embeddings.append(self.embeddings[d])
            all_paragraphs.extend(self.paragraphs.get(d, []))

        if not all_embeddings:
            return [], []

        # Pastikan jumlah embeddings dan paragraphs match
        corpus = np.concatenate(all_embeddings, axis=0)
        if corpus.shape[0] != len(all_paragraphs):
            # Truncate ke minimum
            min_len = min(corpus.shape[0], len(all_paragraphs))
            corpus = corpus[:min_len]
            all_paragraphs = all_paragraphs[:min_len]

        if corpus.shape[0] == 0:
            return [], []

        # Cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-10:
            return [], []
        query_normalized = query_embedding / query_norm

        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        corpus_norms = np.where(corpus_norms < 1e-10, 1.0, corpus_norms)
        corpus_normalized = corpus / corpus_norms

        similarities = corpus_normalized @ query_normalized

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_paragraphs = [all_paragraphs[i] for i in top_indices]
        top_sims = [float(similarities[i]) for i in top_indices]

        return top_paragraphs, top_sims


# ============================================================================
# MINI MODEL
# ============================================================================

class MiniModel:
    """
    Model Transformer mini untuk inference.
    
    Arsitektur:
    - Embedding
    - N x TransformerBlock (RMSNorm + Attention + RMSNorm + SwiGLU FFN)
    - Final RMSNorm
    - Multiple LM heads (MoE) atau single LM head
    """

    def __init__(
        self,
        state_dict: Dict[str, np.ndarray],
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        max_position: int,
        ffn_hidden_dim: int,
        use_rope: bool = True,
        attention_logit_cap: float = 30.0,
        final_logit_cap: float = 50.0,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_position = max_position
        self.ffn_hidden_dim = ffn_hidden_dim
        self.use_rope = use_rope
        self.attention_logit_cap = attention_logit_cap
        self.final_logit_cap = final_logit_cap
        self.head_dim = embedding_dim // num_heads

        # Load weights
        self.embed_tokens = state_dict.get("embed_tokens.weight")
        if self.embed_tokens is None:
            raise ValueError("embed_tokens.weight tidak ditemukan di state_dict")

        self.final_norm_weight = state_dict.get("norm.weight")
        if self.final_norm_weight is None:
            raise ValueError("norm.weight tidak ditemukan di state_dict")

        # Load layer weights
        self.layers = []
        for i in range(num_layers):
            prefix = f"layers.{i}."
            layer = {
                "input_norm": state_dict.get(f"{prefix}input_layernorm.weight"),
                "q_proj": state_dict.get(f"{prefix}self_attn.q_proj.weight"),
                "k_proj": state_dict.get(f"{prefix}self_attn.k_proj.weight"),
                "v_proj": state_dict.get(f"{prefix}self_attn.v_proj.weight"),
                "o_proj": state_dict.get(f"{prefix}self_attn.o_proj.weight"),
                "post_norm": state_dict.get(f"{prefix}post_attention_layernorm.weight"),
                "gate_proj": state_dict.get(f"{prefix}mlp.gate_proj.weight"),
                "up_proj": state_dict.get(f"{prefix}mlp.up_proj.weight"),
                "down_proj": state_dict.get(f"{prefix}mlp.down_proj.weight"),
            }
            
            # Validasi semua weights ada
            for name, weight in layer.items():
                if weight is None:
                    raise ValueError(f"{prefix}{name} tidak ditemukan di state_dict")
            
            self.layers.append(layer)

        # Load LM heads (MoE atau single)
        self.lm_heads: Dict[str, np.ndarray] = {}
        for name, arr in state_dict.items():
            if name.startswith("lm_heads.") and name.endswith(".weight"):
                domain = name[len("lm_heads."):-len(".weight")]
                self.lm_heads[domain] = arr

        # Fallback: jika tidak ada lm_heads, gunakan embed_tokens (weight tying)
        if not self.lm_heads:
            self.lm_heads["GENERAL"] = self.embed_tokens

        # Active domain
        self.active_domain = "GENERAL" if "GENERAL" in self.lm_heads else list(self.lm_heads.keys())[0]

        # RoPE cache
        if self.use_rope:
            self.rope_cos, self.rope_sin = _build_rope_cache(
                max_position, self.head_dim
            )

    def list_domains(self) -> List[str]:
        """List semua domain yang tersedia."""
        return list(self.lm_heads.keys())

    def has_domain(self, domain: str) -> bool:
        """Cek apakah domain ada."""
        return domain in self.lm_heads

    def set_active_domain(self, domain: str) -> None:
        """Set active domain untuk LM head."""
        if domain in self.lm_heads:
            self.active_domain = domain
        else:
            # Fallback ke GENERAL atau first available
            if "GENERAL" in self.lm_heads:
                self.active_domain = "GENERAL"
            elif self.lm_heads:
                self.active_domain = list(self.lm_heads.keys())[0]

    def _rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """RMSNorm forward."""
        x = _sanitize(x)
        ms = np.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(ms + eps)
        y = x * inv_rms * weight
        return _sanitize(y)

    def _attention_forward(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Multi-head self-attention dengan RoPE dan causal mask."""
        B, T, D = x.shape

        Q = x @ layer["q_proj"].T
        K = x @ layer["k_proj"].T
        V = x @ layer["v_proj"].T

        Q = _sanitize(Q)
        K = _sanitize(K)
        V = _sanitize(V)

        Qh = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE
        if self.use_rope:
            Qh, Kh = _apply_rotary_emb(Qh, Kh, self.rope_cos[:T], self.rope_sin[:T])

        # Attention scores
        scores = Qh @ Kh.transpose(0, 1, 3, 2)
        scores = scores / math.sqrt(self.head_dim)

        # Soft-cap
        scores = _softcap(scores, self.attention_logit_cap)

        # Causal mask
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

        probs = _softmax(scores, axis=-1)
        probs = _sanitize(probs)

        context = probs @ Vh
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        context = _sanitize(context)

        output = context @ layer["o_proj"].T
        return _sanitize(output)

    def _ffn_forward(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """SwiGLU FFN forward."""
        gate = x @ layer["gate_proj"].T
        up = x @ layer["up_proj"].T

        gate = np.clip(gate, -50.0, 50.0)
        up = np.clip(up, -50.0, 50.0)

        sig = 1.0 / (1.0 + np.exp(-gate))
        silu = gate * sig
        silu = _sanitize(silu)

        h = silu * up
        h = _sanitize(h)

        out = h @ layer["down_proj"].T
        return _sanitize(out)

    def get_hidden_states(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Extract hidden states dari final layer (sebelum LM head).
        
        Untuk semantic search.
        
        Return: [B, D] mean pooling
        """
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]

        B, T = ids.shape
        T = min(T, self.max_position)
        ids = ids[:, :T]

        # Clamp IDs
        ids = np.clip(ids, 0, self.vocab_size - 1)

        x = self.embed_tokens[ids]
        x = _sanitize(x)

        for layer in self.layers:
            # Pre-LN + Attention + residual
            h = self._rms_norm(x, layer["input_norm"])
            h = self._attention_forward(h, layer)
            x = x + h

            # Pre-LN + FFN + residual
            h = self._rms_norm(x, layer["post_norm"])
            h = self._ffn_forward(h, layer)
            x = x + h

        # Final norm
        final_hidden = self._rms_norm(x, self.final_norm_weight)

        # Mean pooling
        return np.mean(final_hidden, axis=1).astype(np.float32)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass inference.
        
        Return: logits [B, T, vocab_size]
        """
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]

        B, T = ids.shape

        if T > self.max_position:
            ids = ids[:, -self.max_position:]
            T = self.max_position

        # Clamp IDs
        ids = np.clip(ids, 0, self.vocab_size - 1)

        x = self.embed_tokens[ids]
        x = _sanitize(x)

        for layer in self.layers:
            # Pre-LN + Attention + residual
            h = self._rms_norm(x, layer["input_norm"])
            h = self._attention_forward(h, layer)
            x = x + h

            # Pre-LN + FFN + residual
            h = self._rms_norm(x, layer["post_norm"])
            h = self._ffn_forward(h, layer)
            x = x + h

        # Final norm
        final_hidden = self._rms_norm(x, self.final_norm_weight)

        # LM head (MoE)
        active_head = self.lm_heads[self.active_domain]
        logits = final_hidden @ active_head.T

        # Soft-cap
        logits = _softcap(logits, self.final_logit_cap)
        logits = _sanitize(logits)

        return logits


# ============================================================================
# MINI ENGINE (Main API)
# ============================================================================

class MiniEngine:
    """
    Main API untuk inference standalone.
    
    Contoh pakai:
        engine = MiniEngine("otak_model.safetensors")
        answer = engine.generate("apa itu fotosintesis?")
        print(answer)
    """

    # Default sampling parameters (dioptimalkan untuk model kecil)
    DEFAULT_MAX_NEW_TOKENS = 64
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_K = 20
    DEFAULT_TOP_P = 0.85
    DEFAULT_REPETITION_PENALTY = 1.3
    DEFAULT_MIN_NEW_TOKENS = 4

    def __init__(
        self,
        model_path: Path,
        auto_routing: bool = True,
        semantic_top_k: int = 2,
        verbose: bool = True,
    ):
        self.model_path = Path(model_path)
        self.auto_routing = auto_routing
        self.semantic_top_k = semantic_top_k
        self.verbose = verbose

        # Load model
        if verbose:
            print(f"[Engine] Loading model dari: {model_path}")

        state_dict, metadata = load_safetensors(model_path)

        # Parse metadata
        self.use_moe = metadata.get("use_moe", "false").lower() == "true"
        vocab_size = int(metadata.get("vocab_size", 4096))
        embedding_dim = int(metadata.get("hidden_size", 512))
        num_heads = int(metadata.get("num_attention_heads", 16))
        num_layers = int(metadata.get("num_layers", 6))
        max_position = int(metadata.get("context_length", 96))

        # Fallback ffn_hidden_dim (biasanya 4x embedding_dim untuk SwiGLU)
        ffn_hidden_dim = int(metadata.get("ffn_hidden_dim", embedding_dim * 4))

        # Load tokenizer
        tokenizer_raw = metadata.get("tokenizer")
        if tokenizer_raw:
            try:
                tokenizer_payload = json.loads(tokenizer_raw)
                self.tokenizer = MiniTokenizer(tokenizer_payload)
                if verbose:
                    print(f"[Engine] Tokenizer dimuat: {self.tokenizer.vocab_size} tokens")
            except Exception as exc:
                raise ValueError(f"Gagal parse tokenizer metadata: {exc}")
        else:
            raise ValueError("Tokenizer metadata tidak ditemukan di model file")

        # Load model
        self.model = MiniModel(
            state_dict=state_dict,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_position=max_position,
            ffn_hidden_dim=ffn_hidden_dim,
        )

        if verbose:
            print(f"[Engine] Model dimuat: {num_layers} layers, {embedding_dim} dim")
            print(f"[Engine] LM heads: {self.model.list_domains()}")

        # Load MoE assets
        self.router: Optional[MiniRouter] = None
        if self.use_moe:
            self._setup_router(metadata, state_dict)

        # RNG untuk sampling
        self.rng = np.random.default_rng(1337)

    def _setup_router(self, metadata: dict, state_dict: dict) -> None:
        """Setup MoE router dari metadata dan state_dict."""
        # Load keywords
        keywords = {}
        moe_keywords_raw = metadata.get("moe_keywords")
        if moe_keywords_raw:
            try:
                keywords = json.loads(moe_keywords_raw)
            except Exception:
                pass

        # Load paragraphs
        paragraphs = {}
        moe_paragraphs_raw = metadata.get("moe_paragraphs")
        if moe_paragraphs_raw:
            try:
                paragraphs = json.loads(moe_paragraphs_raw)
            except Exception:
                pass

        # Load semantic embeddings dari tensors
        embeddings = {}
        for name, arr in state_dict.items():
            if name.startswith("__semantic_embeddings__."):
                domain = name[len("__semantic_embeddings__."):]
                embeddings[domain] = arr

        if keywords:
            self.router = MiniRouter(
                keywords=keywords,
                paragraphs=paragraphs,
                embeddings=embeddings,
            )
            if self.verbose:
                print(f"[Engine] MoE router dimuat: {len(keywords)} domain")
                if embeddings:
                    print(f"[Engine] Semantic cache: {len(embeddings)} domain")
        else:
            if self.verbose:
                print("[Engine] MoE aktif tapi keywords tidak ditemukan.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        inject_context: bool = True,
    ) -> str:
        """
        Generate jawaban untuk prompt.
        
        Flow:
        1. Route prompt ke domain (jika MoE aktif)
        2. Set active LM head
        3. Inject semantic context (jika tersedia)
        4. Generate dengan sampling
        """
        # ================================================================
        # MoE ROUTING
        # ================================================================
        domain = None
        if self.use_moe and self.router is not None and self.auto_routing:
            domain, confidence = self.router.route(prompt)
            self.model.set_active_domain(domain)

        # ================================================================
        # SEMANTIC CONTEXT INJECTION
        # ================================================================
        actual_prompt = prompt
        if (
            inject_context
            and self.use_moe
            and self.router is not None
            and domain is not None
            and self.semantic_top_k > 0
        ):
            try:
                # Encode prompt untuk embedding
                prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
                if prompt_ids:
                    input_ids = np.array([prompt_ids], dtype=np.int64)
                    query_emb = self.model.get_hidden_states(input_ids)[0]

                    paragraphs, sims = self.router.semantic_search(
                        query_emb, domain=domain, top_k=self.semantic_top_k
                    )

                    if paragraphs:
                        context_lines = []
                        for para in paragraphs:
                            para_truncated = para[:200] + "..." if len(para) > 200 else para
                            context_lines.append(f"- {para_truncated}")

                        context_text = "\n".join(context_lines)
                        actual_prompt = f"Berdasarkan informasi berikut:\n{context_text}\n\nJawab pertanyaan: {prompt}"
            except Exception:
                # Fallback ke prompt asli
                actual_prompt = prompt

        # ================================================================
        # ENCODE PROMPT
        # ================================================================
        prompt_ids = self.tokenizer.encode(actual_prompt, add_bos=True, add_eos=False)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]

        current_ids = list(prompt_ids)
        forbidden_ids = self._get_forbidden_ids()

        # ================================================================
        # GENERATION LOOP
        # ================================================================
        for step in range(max_new_tokens):
            context = current_ids[-self.model.max_position:]
            input_ids = np.array([context], dtype=np.int64)

            logits = self.model.forward(input_ids)
            next_logits = logits[0, -1, :].astype(np.float32)

            step_forbidden = forbidden_ids
            if step < min_new_tokens:
                step_forbidden = list(forbidden_ids)
                if self.tokenizer.eos_id not in step_forbidden:
                    step_forbidden.append(self.tokenizer.eos_id)
                if self.tokenizer.endoftext_id not in step_forbidden:
                    step_forbidden.append(self.tokenizer.endoftext_id)

            generated_so_far = current_ids[len(prompt_ids):]
            previous_ids = generated_so_far[-64:] if len(generated_so_far) > 64 else generated_so_far

            next_token = self._sample(
                logits=next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_ids=step_forbidden,
                previous_ids=previous_ids,
                repetition_penalty=repetition_penalty,
            )

            # Stop conditions
            if next_token == self.tokenizer.eos_id:
                break
            if next_token == self.tokenizer.endoftext_id:
                break

            current_ids.append(int(next_token))

        # ================================================================
        # DECODE
        # ================================================================
        generated_ids = current_ids[len(prompt_ids):]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return answer

    def _get_forbidden_ids(self) -> List[int]:
        """Token yang tidak boleh di-generate."""
        forbidden = {
            self.tokenizer.pad_id,
            self.tokenizer.unk_id,
            self.tokenizer.bos_id,
            self.tokenizer.endoftext_id,
        }

        # Tambah UNUSED tokens
        for token, tid in self.tokenizer.token_to_id.items():
            if token.startswith("[UNUSED_"):
                forbidden.add(int(tid))

        return sorted(forbidden)

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
            return self.tokenizer.eos_id

        float_max = np.finfo(np.float32).max / 4.0
        raw_logits = np.nan_to_num(
            logits, nan=-float_max, posinf=float_max, neginf=-float_max
        )

        work_logits = raw_logits.copy()

        # Repetition penalty
        if repetition_penalty is not None and repetition_penalty != 1.0 and previous_ids:
            prev_tokens = sorted(set(int(t) for t in previous_ids))
            prev_arr = np.array(prev_tokens, dtype=np.int64)
            prev_arr = prev_arr[(prev_arr >= 0) & (prev_arr < work_logits.size)]

            if prev_arr.size > 0:
                selected_logits = work_logits[prev_arr]
                work_logits[prev_arr] = np.where(
                    selected_logits > 0,
                    selected_logits / repetition_penalty,
                    selected_logits * repetition_penalty,
                )

        # Forbidden tokens
        valid_mask = np.ones(raw_logits.size, dtype=bool)
        if forbidden_ids:
            valid_forbidden = [idx for idx in forbidden_ids if 0 <= idx < raw_logits.size]
            if valid_forbidden:
                valid_mask[valid_forbidden] = False

        if not np.any(valid_mask):
            valid_mask = np.ones(raw_logits.size, dtype=bool)

        # Greedy jika temperature sangat kecil
        if temperature is None or temperature <= 1e-8:
            greedy_logits = np.where(valid_mask, work_logits, -np.inf)
            return int(np.argmax(greedy_logits))

        # Temperature
        work_logits = work_logits / float(temperature)

        # Top-K
        if top_k is not None and top_k > 0:
            num_valid = int(np.sum(valid_mask))
            if num_valid > top_k:
                masked_logits = np.where(valid_mask, work_logits, -np.inf)
                threshold = np.partition(masked_logits, -top_k)[-top_k]
                valid_mask = valid_mask & (work_logits >= threshold)

        # Top-P
        if top_p is not None and 0.0 < top_p < 1.0:
            masked_logits = np.where(valid_mask, work_logits, -np.inf)
            probs = _softmax(masked_logits)

            if np.any(probs > 0):
                sorted_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumulative_probs = np.cumsum(sorted_probs)

                cutoff = int(np.searchsorted(cumulative_probs, top_p) + 1)
                cutoff = max(1, min(cutoff, len(sorted_idx)))

                allowed_idx = sorted_idx[:cutoff]
                nucleus_mask = np.zeros_like(valid_mask, dtype=bool)
                nucleus_mask[allowed_idx] = True
                valid_mask = valid_mask & nucleus_mask

                if not np.any(valid_mask) and len(allowed_idx) > 0:
                    valid_mask[allowed_idx[0]] = True

        # Sampling
        final_logits = np.where(valid_mask, work_logits, -np.inf)
        probs = _softmax(final_logits)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

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


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_cli(engine: MiniEngine) -> None:
    """Interactive command line interface."""
    print("=" * 60)
    print("OTAK MINI - Standalone Inference Engine")
    print("=" * 60)
    print("Ketik prompt untuk inference.")
    print("Ketik /help untuk bantuan, /exit untuk keluar.")
    
    if engine.use_moe:
        print(f"MoE aktif: {len(engine.model.list_domains())} domain tersedia")
        if engine.router:
            print(f"Auto-routing: {engine.auto_routing}")
    
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

        # Commands
        if stripped.startswith("/"):
            cmd = stripped.lower().split()[0]

            if cmd in {"/exit", "/quit", "/keluar"}:
                print("Keluar.")
                break

            elif cmd in {"/help", "/h", "/?"}:
                print("\nPerintah:")
                print("  /help      : tampilkan bantuan ini")
                print("  /domains   : list domain yang tersedia")
                print("  /auto on   : aktifkan auto-routing")
                print("  /auto off  : nonaktifkan auto-routing")
                print("  /exit      : keluar")
                print()

            elif cmd == "/domains":
                domains = engine.model.list_domains()
                print(f"\nDomain tersedia ({len(domains)}):")
                for domain in domains:
                    marker = " ← aktif" if domain == engine.model.active_domain else ""
                    print(f"  [{domain}]{marker}")
                print()

            elif cmd == "/auto":
                parts = stripped.split()
                if len(parts) >= 2:
                    mode = parts[1].lower()
                    if mode in {"on", "1", "true"}:
                        engine.auto_routing = True
                        print("Auto-routing: AKTIF")
                    elif mode in {"off", "0", "false"}:
                        engine.auto_routing = False
                        print("Auto-routing: NONAKTIF")
                else:
                    status = "AKTIF" if engine.auto_routing else "NONAKTIF"
                    print(f"Auto-routing: {status}")
                print()

            else:
                print(f"Perintah tidak dikenal: {cmd}")
                print("Ketik /help untuk bantuan.")
                print()

            continue

        # Inference
        try:
            answer = engine.generate(stripped)
            print(f"Jawaban: {answer}")
            print("-" * 60)
        except Exception as exc:
            print(f"Error: {exc}")
            print("-" * 60)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OTAK MINI - Standalone Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  # Interactive mode
  python otak_engine.py --model otak_model.safetensors

  # Single prompt
  python otak_engine.py --model otak_model.safetensors --prompt "apa itu fotosintesis?"

  # Disable auto-routing
  python otak_engine.py --model otak_model.safetensors --no-auto-routing
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path ke file model .safetensors",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt untuk single-shot inference (jika tidak diberikan, masuk interactive mode)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Jumlah token maksimum yang dihasilkan",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature sampling (lebih rendah = lebih deterministik)",
    )
    parser.add_argument(
        "--no-auto-routing",
        action="store_true",
        help="Nonaktifkan auto-routing MoE",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Nonaktifkan semantic context injection",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode quiet (tidak cetak info loading)",
    )

    args = parser.parse_args()

    try:
        engine = MiniEngine(
            model_path=args.model,
            auto_routing=not args.no_auto_routing,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"ERROR: Gagal load model: {exc}")
        sys.exit(1)

    if args.prompt:
        # Single-shot mode
        try:
            answer = engine.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                inject_context=not args.no_context,
            )
            print(f"\nPrompt : {args.prompt}")
            print(f"Jawaban: {answer}")
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_cli(engine)


if __name__ == "__main__":
    main()
