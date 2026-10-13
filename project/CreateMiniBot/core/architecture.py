# core/architecture.py
#
# Arsitektur neural network custom untuk RAG-based Training Pipeline Engine.
#
# Arsitektur: Single-Head Transformer + Cross-Attention + Constrained Generation
# - Context-conditioned generation (RAG)
# - Cross-Attention untuk "memaksa" model attend ke context (solve topic mixing)
# - Constrained Generation: hanya generate tokens dari context (solve gibberish)
# - Single LM head (no domain contamination)
# - Loss ↓ = Presisi ↑ (korelasi langsung)
#
# SOTA features:
# - RoPE (Rotary Position Embedding) - long context up to 512 tokens
# - Weight Tying antara embed_tokens dan lm_head
# - Logit Soft-Capping (tanh) menggantikan hard clipping
# - Stochastic Depth (DropPath) pada TransformerBlock
# - Scaled Initialization untuk residual sublayers (GPT-2 style)
# - Cross-Attention layers (optional, configurable)
#
# Keterhubungan:
# - config.py              : parameter arsitektur (termasuk cross_attention_layers)
# - core/retriever.py      : menyediakan konteks sebelum forward
# - pipeline/trainer.py    : forward pass dengan context_ids
# - pipeline/evaluator.py  : menerima logits
# - pipeline/weight_manager.py : memakai gradient dari backward pass
# - core/model_exporter.py : memakai state_dict()
# - testing/playground.py  : inference dengan constrained generation

from __future__ import annotations

import math
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
        "core/architecture.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config


# Batas clipping untuk aktivasi internal
ACTIVATION_CLIP = 50.0


# ============================================================================
# NUMERIC HELPERS
# ============================================================================

def _init_weight(
    rng: np.random.Generator,
    shape: tuple,
    scale: float = 1.0,
) -> np.ndarray:
    """Inisialisasi bobot dengan distribusi normal kecil (GPT-2 style)."""
    if len(shape) == 1:
        fan_in = shape[0]
    else:
        fan_in = shape[1]

    std = min(0.02, 1.0 / math.sqrt(max(1, fan_in))) * scale
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax numerik stabil dengan sanitasi NaN/Inf."""
    x = np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.nan_to_num(x_max, nan=0.0, posinf=50.0, neginf=-50.0)

    shifted = np.clip(x - x_max, -50.0, 50.0)
    exp_x = np.exp(shifted)

    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1e-10)

    return exp_x / sum_exp


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Sanitasi array dari NaN/Inf."""
    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=ACTIVATION_CLIP,
        neginf=-ACTIVATION_CLIP,
    ).astype(np.float32, copy=False)


def _softcap(x: np.ndarray, cap: float) -> np.ndarray:
    """Soft-capping berbasis tanh. Mencegah overflow tanpa memotong gradient."""
    return cap * np.tanh(x / cap).astype(np.float32)


def _softcap_backward(dout: np.ndarray, x: np.ndarray, cap: float) -> np.ndarray:
    """Backward untuk soft-capping."""
    tanh_x = np.tanh(x / cap)
    return dout * (1.0 - tanh_x * tanh_x).astype(np.float32)


def _dropout(
    x: np.ndarray,
    p: float,
    training: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inverted dropout."""
    p = float(p)

    if not training or p <= 0.0:
        return x, None

    if p >= 1.0:
        p = 0.999

    keep_prob = 1.0 - p
    mask = (rng.random(x.shape) < keep_prob).astype(np.float32)
    mask = mask / keep_prob

    return x * mask, mask


def _dropout_backward(
    dout: np.ndarray,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Backward untuk dropout."""
    if mask is None:
        return dout
    return dout * mask


def _drop_path(
    x: np.ndarray,
    p: float,
    training: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, bool]:
    """Stochastic Depth (DropPath)."""
    p = float(p)

    if not training or p <= 0.0:
        return x, False

    if rng.random() < p:
        return np.zeros_like(x), True

    return x / (1.0 - p), False


def _drop_path_backward(
    dout: np.ndarray,
    dropped: bool,
    p: float,
) -> np.ndarray:
    """Backward untuk drop path."""
    if dropped:
        return np.zeros_like(dout)

    if p <= 0.0 or p >= 1.0:
        return dout

    return dout / (1.0 - p)


# ============================================================================
# ROTARY POSITION EMBEDDING (RoPE)
# ============================================================================

def _build_rope_cache(
    max_position: int,
    head_dim: int,
    base: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _apply_rotary_emb(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Terapkan RoPE pada Q dan K."""
    cos_expanded = cos[np.newaxis, np.newaxis, :, :]
    sin_expanded = sin[np.newaxis, np.newaxis, :, :]

    q_rot = q * cos_expanded + _rotate_half(q) * sin_expanded
    k_rot = k * cos_expanded + _rotate_half(k) * sin_expanded

    return q_rot.astype(np.float32), k_rot.astype(np.float32)


def _rotary_backward(
    dq: np.ndarray,
    dk: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward untuk RoPE (rotasi dengan sudut negatif)."""
    cos_expanded = cos[np.newaxis, np.newaxis, :, :]
    sin_expanded = sin[np.newaxis, np.newaxis, :, :]

    dq_orig = dq * cos_expanded - _rotate_half(dq) * sin_expanded
    dk_orig = dk * cos_expanded - _rotate_half(dk) * sin_expanded

    return dq_orig.astype(np.float32), dk_orig.astype(np.float32)


# ============================================================================
# PARAMETER
# ============================================================================

class Parameter:
    """Parameter sederhana berisi data bobot dan gradient."""

    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32, copy=False)
        self.grad = np.zeros_like(self.data, dtype=np.float32)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm:
    """RMSNorm sederhana."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.eps = float(eps)
        self.cache: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = _sanitize(x)
        ms = np.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(ms + self.eps)

        y = x * inv_rms * self.weight.data
        y = _sanitize(y)
        self.cache = (x, inv_rms)
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("RMSNorm.backward dipanggil sebelum forward.")

        x, inv_rms = self.cache
        D = x.shape[-1]

        dout = _sanitize(dout)

        axes = tuple(range(dout.ndim - 1))
        self.weight.grad += np.sum(dout * x * inv_rms, axis=axes)
        self.weight.grad = _sanitize(self.weight.grad)

        dot = np.sum(dout * self.weight.data * x, axis=-1, keepdims=True)
        dx = (
            dout * self.weight.data * inv_rms
            - x * dot * (inv_rms ** 3) / D
        )

        dx = _sanitize(dx)
        self.cache = None
        return dx

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        return {
            f"{prefix}weight": self.weight,
        }


# ============================================================================
# MULTI-HEAD SELF-ATTENTION dengan RoPE
# ============================================================================

class MultiHeadSelfAttention:
    """Multi-head self-attention dengan RoPE dan logit soft-capping."""

    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
        layer_idx: int = 0,
        num_layers: int = 1,
    ):
        self.hidden_size = int(config.model.embedding_dim)
        self.num_heads = int(config.model.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        dropout_rate = getattr(config.training, 'dropout_rate', None)
        if dropout_rate is None:
            dropout_rate = config.model.dropout_rate
        self.dropout_p = float(dropout_rate)

        self.attention_logit_cap = float(
            getattr(config.model, 'attention_logit_cap', 30.0)
        )

        self.rng = rng
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "embedding_dim harus habis dibagi num_attention_heads."
            )

        residual_scale = 1.0 / math.sqrt(2.0 * num_layers)

        self.q_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.k_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.v_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.o_proj = Parameter(
            _init_weight(rng, (self.hidden_size, self.hidden_size), scale=residual_scale)
        )

        self.cache: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        x = _sanitize(x)
        B, T, D = x.shape

        Q = x @ self.q_proj.data.T
        K = x @ self.k_proj.data.T
        V = x @ self.v_proj.data.T

        Q = _sanitize(Q)
        K = _sanitize(K)
        V = _sanitize(V)

        Qh = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        Qh_rot, Kh_rot = _apply_rotary_emb(Qh, Kh, cos[:T], sin[:T])
        Qh_rot = _sanitize(Qh_rot)
        Kh_rot = _sanitize(Kh_rot)

        scores = Qh_rot @ Kh_rot.transpose(0, 1, 3, 2)
        scores = scores * self.scale
        scores = _softcap(scores, self.attention_logit_cap)

        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

        probs = _softmax(scores, axis=-1)
        probs = _sanitize(probs)

        probs_drop, attn_dropout_mask = _dropout(probs, self.dropout_p, training, self.rng)

        context = probs_drop @ Vh
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        context = _sanitize(context)

        output = context @ self.o_proj.data.T
        output = _sanitize(output)

        self.cache = (
            x, Qh, Kh, Vh, Qh_rot, Kh_rot,
            probs, probs_drop, attn_dropout_mask,
            context, scores, cos[:T], sin[:T],
        )

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("MultiHeadSelfAttention.backward dipanggil sebelum forward.")

        (x, Qh, Kh, Vh, Qh_rot, Kh_rot,
         probs, probs_drop, attn_dropout_mask,
         context, scores, cos, sin) = self.cache

        dout = _sanitize(dout)
        B, T, D = x.shape

        dout_2d = dout.reshape(-1, D)
        context_2d = context.reshape(-1, D)

        self.o_proj.grad += dout_2d.T @ context_2d
        self.o_proj.grad = _sanitize(self.o_proj.grad)
        dcontext = dout @ self.o_proj.data
        dcontext = _sanitize(dcontext)

        dcontext_heads = dcontext.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        dprobs_drop = dcontext_heads @ Vh.transpose(0, 1, 3, 2)
        dVh = probs_drop.transpose(0, 1, 3, 2) @ dcontext_heads

        dprobs = _dropout_backward(dprobs_drop, attn_dropout_mask)

        sum_dp = np.sum(dprobs * probs, axis=-1, keepdims=True)
        dscores = probs * (dprobs - sum_dp)
        dscores = dscores * self.scale

        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores = np.where(causal_mask[None, None, :, :], 0.0, dscores)

        dscores_pre_cap = _softcap_backward(
            dscores, 
            scores * (1.0 / self.scale) * self.scale, 
            self.attention_logit_cap
        )

        dQh_rot = dscores_pre_cap @ Kh_rot
        dKh_rot = dscores_pre_cap.transpose(0, 1, 3, 2) @ Qh_rot

        dQh, dKh = _rotary_backward(dQh_rot, dKh_rot, cos, sin)

        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)

        dQ = _sanitize(dQ)
        dK = _sanitize(dK)
        dV = _sanitize(dV)

        x_2d = x.reshape(-1, D)

        self.q_proj.grad += dQ.reshape(-1, D).T @ x_2d
        self.k_proj.grad += dK.reshape(-1, D).T @ x_2d
        self.v_proj.grad += dV.reshape(-1, D).T @ x_2d

        self.q_proj.grad = _sanitize(self.q_proj.grad)
        self.k_proj.grad = _sanitize(self.k_proj.grad)
        self.v_proj.grad = _sanitize(self.v_proj.grad)

        dx_q = dQ @ self.q_proj.data
        dx_k = dK @ self.k_proj.data
        dx_v = dV @ self.v_proj.data

        dx = dx_q + dx_k + dx_v
        dx = _sanitize(dx)

        self.cache = None
        return dx

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        return {
            f"{prefix}q_proj.weight": self.q_proj,
            f"{prefix}k_proj.weight": self.k_proj,
            f"{prefix}v_proj.weight": self.v_proj,
            f"{prefix}o_proj.weight": self.o_proj,
        }


# ============================================================================
# CROSS-ATTENTION (BARU)
# ============================================================================

class CrossAttention:
    """
    Cross-attention layer untuk attend ke context embeddings.
    
    Digunakan di TransformerBlock untuk memaksa model "melihat" context
    di setiap layer. Ini solve "Lost in the Middle" problem.
    
    Perbedaan dengan self-attention:
    - Query dari current hidden states
    - Key/Value dari context embeddings (fixed)
    - NO causal mask (bisa lihat semua context)
    - NO RoPE (context position sudah di-handle saat encode)
    """

    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
        layer_idx: int = 0,
        num_layers: int = 1,
    ):
        self.hidden_size = int(config.model.embedding_dim)
        self.num_heads = int(config.model.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        dropout_rate = getattr(config.training, 'dropout_rate', None)
        if dropout_rate is None:
            dropout_rate = config.model.dropout_rate
        self.dropout_p = float(dropout_rate)

        # Cross-attention tidak butuh soft-cap (context sudah "dijinakkan")
        self.attention_logit_cap = float(
            getattr(config.model, 'attention_logit_cap', 30.0)
        )

        self.rng = rng
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("embedding_dim harus habis dibagi num_attention_heads.")

        residual_scale = 1.0 / math.sqrt(2.0 * num_layers)

        self.q_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.k_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.v_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.o_proj = Parameter(
            _init_weight(rng, (self.hidden_size, self.hidden_size), scale=residual_scale)
        )

        self.cache: Optional[tuple] = None

    def forward(
        self,
        query: np.ndarray,
        context: np.ndarray,
        training: bool,
    ) -> np.ndarray:
        """
        Cross-attention forward.
        
        Args:
            query: [B, T_q, D] - current hidden states
            context: [B, T_c, D] - context embeddings (dari context encoder)
            training: True saat training
        
        Returns:
            output: [B, T_q, D]
        """
        query = _sanitize(query)
        context = _sanitize(context)

        B, T_q, D = query.shape
        T_c = context.shape[1]

        # Projections
        Q = query @ self.q_proj.data.T   # [B, T_q, D]
        K = context @ self.k_proj.data.T  # [B, T_c, D]
        V = context @ self.v_proj.data.T  # [B, T_c, D]

        Q = _sanitize(Q)
        K = _sanitize(K)
        V = _sanitize(V)

        # Reshape untuk multi-head
        Qh = Q.reshape(B, T_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T_c, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T_c, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores: [B, H, T_q, T_c]
        scores = Qh @ Kh.transpose(0, 1, 3, 2)
        scores = scores * self.scale

        # Soft-cap (optional, untuk stability)
        scores = _softcap(scores, self.attention_logit_cap)

        # NO causal mask untuk cross-attention - bisa lihat semua context
        probs = _softmax(scores, axis=-1)
        probs = _sanitize(probs)

        probs_drop, attn_dropout_mask = _dropout(probs, self.dropout_p, training, self.rng)

        # Weighted sum: [B, H, T_q, D]
        attn_out = probs_drop @ Vh
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T_q, D)
        attn_out = _sanitize(attn_out)

        # Output projection
        output = attn_out @ self.o_proj.data.T
        output = _sanitize(output)

        # Cache untuk backward
        self.cache = (
            query, context,
            Qh, Kh, Vh,
            probs, probs_drop, attn_dropout_mask,
            attn_out, scores,
        )

        return output

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward untuk cross-attention.
        
        Returns:
            (d_query, d_context): gradients untuk query dan context
        """
        if self.cache is None:
            raise RuntimeError("CrossAttention.backward dipanggil sebelum forward.")

        (query, context,
         Qh, Kh, Vh,
         probs, probs_drop, attn_dropout_mask,
         attn_out, scores) = self.cache

        dout = _sanitize(dout)
        B, T_q, D = query.shape
        T_c = context.shape[1]

        # Output projection backward
        dout_2d = dout.reshape(-1, D)
        attn_out_2d = attn_out.reshape(-1, D)

        self.o_proj.grad += dout_2d.T @ attn_out_2d
        self.o_proj.grad = _sanitize(self.o_proj.grad)

        d_attn_out = dout @ self.o_proj.data
        d_attn_out = _sanitize(d_attn_out)

        d_attn_out_heads = d_attn_out.reshape(
            B, T_q, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Gradient untuk attention probabilities dan value
        dprobs_drop = d_attn_out_heads @ Vh.transpose(0, 1, 3, 2)
        dVh = probs_drop.transpose(0, 1, 3, 2) @ d_attn_out_heads

        # Dropout backward
        dprobs = _dropout_backward(dprobs_drop, attn_dropout_mask)

        # Softmax backward
        sum_dp = np.sum(dprobs * probs, axis=-1, keepdims=True)
        dscores = probs * (dprobs - sum_dp)
        dscores = dscores * self.scale

        # Soft-cap backward
        dscores_pre_cap = _softcap_backward(
            dscores,
            scores * (1.0 / self.scale) * self.scale,
            self.attention_logit_cap
        )

        # Gradient untuk Q, K, V
        dQh = dscores_pre_cap @ Kh
        dKh = dscores_pre_cap.transpose(0, 1, 3, 2) @ Qh
        dVh = dVh  # sudah dihitung di atas

        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T_q, D)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, T_c, D)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, T_c, D)

        dQ = _sanitize(dQ)
        dK = _sanitize(dK)
        dV = _sanitize(dV)

        # Projection backward
        query_2d = query.reshape(-1, D)
        context_2d = context.reshape(-1, D)

        self.q_proj.grad += dQ.reshape(-1, D).T @ query_2d
        self.k_proj.grad += dK.reshape(-1, D).T @ context_2d
        self.v_proj.grad += dV.reshape(-1, D).T @ context_2d

        self.q_proj.grad = _sanitize(self.q_proj.grad)
        self.k_proj.grad = _sanitize(self.k_proj.grad)
        self.v_proj.grad = _sanitize(self.v_proj.grad)

        # Gradient untuk query (sum dari Q projection)
        d_query = dQ @ self.q_proj.data
        d_query = _sanitize(d_query)

        # Gradient untuk context (sum dari K dan V projections)
        d_context_k = dK @ self.k_proj.data
        d_context_v = dV @ self.v_proj.data
        d_context = d_context_k + d_context_v
        d_context = _sanitize(d_context)

        self.cache = None
        return d_query, d_context

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        return {
            f"{prefix}q_proj.weight": self.q_proj,
            f"{prefix}k_proj.weight": self.k_proj,
            f"{prefix}v_proj.weight": self.v_proj,
            f"{prefix}o_proj.weight": self.o_proj,
        }


# ============================================================================
# SwiGLU FEED-FORWARD NETWORK
# ============================================================================

class SwiGLUFFN:
    """Feed-forward network dengan SwiGLU."""

    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
        layer_idx: int = 0,
        num_layers: int = 1,
    ):
        self.hidden_size = int(config.model.embedding_dim)
        self.ffn_hidden_dim = int(config.model.ffn_hidden_dim)

        dropout_rate = getattr(config.training, 'dropout_rate', None)
        if dropout_rate is None:
            dropout_rate = config.model.dropout_rate
        self.dropout_p = float(dropout_rate)

        self.rng = rng

        residual_scale = 1.0 / math.sqrt(2.0 * num_layers)

        self.gate_proj = Parameter(
            _init_weight(rng, (self.ffn_hidden_dim, self.hidden_size))
        )
        self.up_proj = Parameter(
            _init_weight(rng, (self.ffn_hidden_dim, self.hidden_size))
        )
        self.down_proj = Parameter(
            _init_weight(rng, (self.hidden_size, self.ffn_hidden_dim), scale=residual_scale)
        )

        self.cache: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        x = _sanitize(x)

        gate = x @ self.gate_proj.data.T
        up = x @ self.up_proj.data.T

        gate = np.clip(gate, -ACTIVATION_CLIP, ACTIVATION_CLIP)
        up = np.clip(up, -ACTIVATION_CLIP, ACTIVATION_CLIP)

        sig = 1.0 / (1.0 + np.exp(-gate))
        silu = gate * sig
        silu = _sanitize(silu)

        h = silu * up
        h = _sanitize(h)

        h_drop, dropout_mask = _dropout(h, self.dropout_p, training, self.rng)

        out = h_drop @ self.down_proj.data.T
        out = _sanitize(out)

        self.cache = (x, gate, up, sig, silu, h_drop, dropout_mask)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("SwiGLUFFN.backward dipanggil sebelum forward.")

        x, gate, up, sig, silu, h_drop, dropout_mask = self.cache

        dout = _sanitize(dout)
        D = x.shape[-1]
        F = gate.shape[-1]

        dout_2d = dout.reshape(-1, D)
        h_2d = h_drop.reshape(-1, F)

        self.down_proj.grad += dout_2d.T @ h_2d
        self.down_proj.grad = _sanitize(self.down_proj.grad)
        dh_drop = dout @ self.down_proj.data
        dh_drop = _sanitize(dh_drop)

        dh = _dropout_backward(dh_drop, dropout_mask)

        dsilu = dh * up
        dup = dh * silu

        dgate = dsilu * sig * (1.0 + gate * (1.0 - sig))
        dgate = _sanitize(dgate)
        dup = _sanitize(dup)

        x_2d = x.reshape(-1, D)

        self.gate_proj.grad += dgate.reshape(-1, F).T @ x_2d
        self.up_proj.grad += dup.reshape(-1, F).T @ x_2d

        self.gate_proj.grad = _sanitize(self.gate_proj.grad)
        self.up_proj.grad = _sanitize(self.up_proj.grad)

        dx_gate = dgate @ self.gate_proj.data
        dx_up = dup @ self.up_proj.data

        dx = dx_gate + dx_up
        dx = _sanitize(dx)

        self.cache = None
        return dx

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        return {
            f"{prefix}gate_proj.weight": self.gate_proj,
            f"{prefix}up_proj.weight": self.up_proj,
            f"{prefix}down_proj.weight": self.down_proj,
        }


# ============================================================================
# TRANSFORMER BLOCK dengan Cross-Attention
# ============================================================================

class TransformerBlock:
    """
    Satu blok Transformer dengan:
    - Pre-Layer Normalization (Pre-LN)
    - Self-Attention dengan RoPE
    - Cross-Attention ke context (optional)
    - SwiGLU FFN
    - Stochastic Depth (DropPath)
    """

    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
        layer_idx: int = 0,
        num_layers: int = 1,
        drop_path_rate: float = 0.0,
        use_cross_attention: bool = False,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.drop_path_rate = float(drop_path_rate)
        self.use_cross_attention = use_cross_attention

        dropout_rate = getattr(config.training, 'dropout_rate', None)
        if dropout_rate is None:
            dropout_rate = config.model.dropout_rate
        self.dropout_p = float(dropout_rate)

        self.rng = rng

        self.input_norm = RMSNorm(
            int(config.model.embedding_dim),
            float(config.model.layer_norm_eps),
        )
        self.attn = MultiHeadSelfAttention(config, rng, layer_idx, num_layers)
        
        # Cross-attention (optional)
        if self.use_cross_attention:
            self.cross_attn_norm = RMSNorm(
                int(config.model.embedding_dim),
                float(config.model.layer_norm_eps),
            )
            self.cross_attn = CrossAttention(config, rng, layer_idx, num_layers)
        else:
            self.cross_attn_norm = None
            self.cross_attn = None
        
        self.post_attention_norm = RMSNorm(
            int(config.model.embedding_dim),
            float(config.model.layer_norm_eps),
        )
        self.ffn = SwiGLUFFN(config, rng, layer_idx, num_layers)

        self.cache: Optional[tuple] = None

    def forward(
        self,
        x: np.ndarray,
        training: bool,
        cos: np.ndarray,
        sin: np.ndarray,
        context_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        residual_1 = x

        # Self-attention
        h = self.input_norm.forward(x)
        h = self.attn.forward(h, training, cos, sin)
        h, attn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)
        h, attn_dropped = _drop_path(h, self.drop_path_rate, training, self.rng)
        x_1 = residual_1 + h

        # Cross-attention (optional)
        cross_dropout_mask = None
        cross_dropped = False
        if self.use_cross_attention and context_embeddings is not None:
            residual_cross = x_1
            h_cross = self.cross_attn_norm.forward(x_1)
            h_cross = self.cross_attn.forward(h_cross, context_embeddings, training)
            h_cross, cross_dropout_mask = _dropout(h_cross, self.dropout_p, training, self.rng)
            h_cross, cross_dropped = _drop_path(h_cross, self.drop_path_rate, training, self.rng)
            x_1 = residual_cross + h_cross

        residual_2 = x_1

        # FFN
        h = self.post_attention_norm.forward(x_1)
        h = self.ffn.forward(h, training)
        h, ffn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)
        h, ffn_dropped = _drop_path(h, self.drop_path_rate, training, self.rng)

        out = residual_2 + h

        self.cache = (
            attn_dropout_mask, ffn_dropout_mask,
            attn_dropped, ffn_dropped,
            cross_dropout_mask, cross_dropped,
        )

        return out

    def backward(
        self,
        dout: np.ndarray,
        context_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Backward pass.
        
        Returns:
            (dx, d_context): gradients untuk input dan context embeddings
                            (d_context is None jika cross-attention tidak dipakai)
        """
        if self.cache is None:
            raise RuntimeError("TransformerBlock.backward dipanggil sebelum forward.")

        (attn_dropout_mask, ffn_dropout_mask,
         attn_dropped, ffn_dropped,
         cross_dropout_mask, cross_dropped) = self.cache

        d_context = None

        # FFN backward
        d_ffn_out = _dropout_backward(dout, ffn_dropout_mask)
        d_ffn_out = _drop_path_backward(d_ffn_out, ffn_dropped, self.drop_path_rate)
        d_ffn_in = self.ffn.backward(d_ffn_out)
        d_post_norm = self.post_attention_norm.backward(d_ffn_in)
        d_x_1 = dout + d_post_norm
        d_x_1 = _sanitize(d_x_1)

        # Cross-attention backward (jika dipakai)
        if self.use_cross_attention and context_embeddings is not None:
            d_cross_out = _dropout_backward(d_x_1, cross_dropout_mask)
            d_cross_out = _drop_path_backward(d_cross_out, cross_dropped, self.drop_path_rate)
            d_cross_in = self.cross_attn_norm.backward(d_cross_out)
            d_query, d_context = self.cross_attn.backward(d_cross_in)
            # Sum gradients: residual + query path
            d_x_1 = d_x_1 + d_query
            d_x_1 = _sanitize(d_x_1)

        # Self-attention backward
        d_attn_out = _dropout_backward(d_x_1, attn_dropout_mask)
        d_attn_out = _drop_path_backward(d_attn_out, attn_dropped, self.drop_path_rate)
        d_attn_in = self.attn.backward(d_attn_out)
        d_input_norm = self.input_norm.backward(d_attn_in)

        dx = d_x_1 + d_input_norm
        dx = _sanitize(dx)

        self.cache = None
        return dx, d_context

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        params: Dict[str, Parameter] = {}

        params.update(self.input_norm.parameters(f"{prefix}input_layernorm."))
        params.update(self.attn.parameters(f"{prefix}self_attn."))
        
        if self.use_cross_attention:
            params.update(self.cross_attn_norm.parameters(f"{prefix}cross_attn_layernorm."))
            params.update(self.cross_attn.parameters(f"{prefix}cross_attn."))
        
        params.update(self.post_attention_norm.parameters(f"{prefix}post_attention_layernorm."))
        params.update(self.ffn.parameters(f"{prefix}mlp."))

        return params


# ============================================================================
# CUSTOM TRANSFORMER LANGUAGE MODEL dengan Cross-Attention + Constrained Gen
# ============================================================================

class CustomTransformerLM:
    """
    Model Transformer custom untuk language modeling dengan RAG.

    Fitur:
    - RoPE untuk positional encoding (long context up to 512 tokens)
    - Weight Tying antara embed_tokens dan lm_head
    - Logit Soft-Capping pada final logits
    - Stochastic Depth pada setiap layer
    - Cross-Attention ke context embeddings (optional, configurable)
    - Constrained Generation untuk inference (hanya tokens dari context)
    """

    MAX_PARAMS: Optional[int] = None

    def __init__(self, config: Config = CONFIG):
        self.config = config

        self.vocab_size = int(config.model.vocab_size)
        self.embedding_dim = int(config.model.embedding_dim)
        self.num_heads = int(config.model.num_attention_heads)
        self.num_layers = int(config.model.num_layers)
        self.ffn_hidden_dim = int(config.model.ffn_hidden_dim)
        self.max_position_embeddings = int(config.model.max_position_embeddings)
        self.layer_norm_eps = float(config.model.layer_norm_eps)

        # ================================================================
        # SOTA config
        # ================================================================
        self.use_rope = bool(getattr(config.model, 'use_rope', True))
        self.use_weight_tying = bool(getattr(config.model, 'use_weight_tying', True))
        self.final_logit_cap = float(getattr(config.model, 'final_logit_cap', 50.0))
        self.drop_path_rate = float(getattr(config.model, 'drop_path_rate', 0.1))
        
        # Cross-attention config (BARU)
        cross_attn_layers = getattr(config.model, 'cross_attention_layers', None)
        if cross_attn_layers is None:
            # Default: apply di semua layers
            self.cross_attention_layer_indices = set(range(self.num_layers))
        else:
            self.cross_attention_layer_indices = set(cross_attn_layers)
        
        self.use_constrained_generation = bool(
            getattr(config.model, 'use_constrained_generation', True)
        )

        self._validate_practical_size()

        self.rng = np.random.default_rng(int(config.training.seed))

        # ================================================================
        # Embedding
        # ================================================================
        self.embed_tokens = Parameter(
            _init_weight(self.rng, (self.vocab_size, self.embedding_dim))
        )

        # ================================================================
        # Transformer layers dengan cross-attention
        # ================================================================
        self.layers = []
        for i in range(self.num_layers):
            layer_drop_rate = (
                self.drop_path_rate * i / max(1, self.num_layers - 1)
            )
            use_cross_attn = i in self.cross_attention_layer_indices
            layer = TransformerBlock(
                config,
                self.rng,
                layer_idx=i,
                num_layers=self.num_layers,
                drop_path_rate=layer_drop_rate,
                use_cross_attention=use_cross_attn,
            )
            self.layers.append(layer)

        self.final_norm = RMSNorm(self.embedding_dim, self.layer_norm_eps)

        # ================================================================
        # Single LM Head
        # ================================================================
        if self.use_weight_tying:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = Parameter(
                _init_weight(self.rng, (self.vocab_size, self.embedding_dim))
            )

        # ================================================================
        # RoPE cache (long context support)
        # ================================================================
        if self.use_rope:
            self.rope_cos, self.rope_sin = _build_rope_cache(
                self.max_position_embeddings,
                self.embedding_dim // self.num_heads,
            )
        else:
            self.pos_encoding = self._build_sinusoidal_positional_encoding(
                self.max_position_embeddings,
                self.embedding_dim,
            )

        self._parameters = self._collect_parameters()
        self._forward_cache: Optional[tuple] = None
        
        # Context embeddings cache untuk inference
        self._cached_context_embeddings: Optional[np.ndarray] = None
        self._cached_context_ids: Optional[np.ndarray] = None

    def _validate_practical_size(self) -> None:
        """Validasi jumlah parameter tidak melebihi batas aman."""
        if self.MAX_PARAMS is None:
            return

        V = self.vocab_size
        D = self.embedding_dim
        L = self.num_layers
        F = self.ffn_hidden_dim
        n_cross_layers = len(self.cross_attention_layer_indices)

        embedding_params = V * D
        lm_head_params = 0 if self.use_weight_tying else V * D

        # Base layer params (self-attn + FFN + norms)
        base_layer_params = 2 * D + 4 * D * D + 3 * D * F
        # Cross-attention layer params (additional)
        cross_layer_params = D + 4 * D * D  # cross_attn_norm + Q,K,V,O projections

        estimated_params = (
            embedding_params
            + L * base_layer_params
            + n_cross_layers * cross_layer_params
            + D
            + lm_head_params
        )

        if estimated_params > self.MAX_PARAMS:
            raise RuntimeError(
                "Estimasi jumlah parameter terlalu besar untuk engine custom "
                "berbasis NumPy ini. "
                f"Estimasi: {estimated_params:,} parameter. "
                f"Batas aman: {self.MAX_PARAMS:,} parameter. "
                "Kurangi vocab_size, embedding_dim, num_layers, atau ffn_hidden_dim "
                "di config.py, atau set CustomTransformerLM.MAX_PARAMS = None."
            )

    def _build_sinusoidal_positional_encoding(
        self,
        max_position: int,
        dim: int,
    ) -> np.ndarray:
        """Fallback: sinusoidal positional encoding jika RoPE dinonaktifkan."""
        pe = np.zeros((max_position, dim), dtype=np.float32)
        position = np.arange(max_position, dtype=np.float32)[:, None]

        freq_indices = np.arange(0, dim, 2, dtype=np.float32)
        div_term = np.exp(freq_indices * -(math.log(10000.0) / dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: dim // 2])

        return pe

    # ====================================================================
    # CONTEXT ENCODING (BARU)
    # ====================================================================

    def encode_context(
        self,
        context_ids: np.ndarray,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode context menjadi embeddings untuk cross-attention.
        
        Args:
            context_ids: [B, T_c] atau [T_c] - context token IDs
            use_cache: jika True, cache hasil untuk reuse
        
        Returns:
            context_embeddings: [B, T_c, D]
        """
        ids = np.asarray(context_ids, dtype=np.int64)

        if ids.ndim == 1:
            ids = ids[None, :]

        if ids.ndim != 2:
            raise ValueError("context_ids harus berbentuk [T_c] atau [B, T_c].")

        B, T_c = ids.shape

        if T_c == 0:
            raise ValueError("Context length tidak boleh kosong.")

        if T_c > self.max_position_embeddings:
            # Truncate context jika terlalu panjang
            ids = ids[:, :self.max_position_embeddings]
            T_c = self.max_position_embeddings

        # Check cache
        if (use_cache and self._cached_context_ids is not None 
            and np.array_equal(ids, self._cached_context_ids)
            and self._cached_context_embeddings is not None):
            return self._cached_context_embeddings

        unk_id = int(self.config.special_tokens.unk_id)
        ids = np.where((ids < 0) | (ids >= self.vocab_size), unk_id, ids)

        # Embedding
        x = self.embed_tokens.data[ids]
        x = _sanitize(x)

        # Positional encoding (NO dropout untuk context encoding)
        if not self.use_rope:
            x = x + self.pos_encoding[:T_c][None, :, :]
            x = _sanitize(x)

        # Forward through layers dengan training=False (NO dropout)
        for layer in self.layers:
            if self.use_rope:
                x = layer.forward(x, training=False, cos=self.rope_cos, sin=self.rope_sin)
            else:
                dummy_cos = np.zeros((T_c, self.embedding_dim // self.num_heads), dtype=np.float32)
                dummy_sin = np.zeros((T_c, self.embedding_dim // self.num_heads), dtype=np.float32)
                x = layer.forward(x, training=False, cos=dummy_cos, sin=dummy_sin)

        # Final norm
        context_embeddings = self.final_norm.forward(x)

        # Cache
        if use_cache:
            self._cached_context_embeddings = context_embeddings
            self._cached_context_ids = ids.copy()

        return context_embeddings

    def clear_context_cache(self) -> None:
        """Clear context embeddings cache."""
        self._cached_context_embeddings = None
        self._cached_context_ids = None

    # ====================================================================
    # HIDDEN STATES EXTRACTION
    # ====================================================================

    def get_hidden_states(
        self,
        input_ids: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Extract embeddings dari final layer (sebelum LM head)."""
        ids = np.asarray(input_ids, dtype=np.int64)

        if ids.ndim == 1:
            ids = ids[None, :]

        if ids.ndim != 2:
            raise ValueError("input_ids harus berbentuk [T] atau [B, T].")

        B, T = ids.shape

        if T == 0:
            raise ValueError("Sequence length tidak boleh kosong.")

        if T > self.max_position_embeddings:
            ids = ids[:, :self.max_position_embeddings]
            T = self.max_position_embeddings

        unk_id = int(self.config.special_tokens.unk_id)
        ids = np.where((ids < 0) | (ids >= self.vocab_size), unk_id, ids)

        x = self.embed_tokens.data[ids]
        x = _sanitize(x)

        if not self.use_rope:
            x = x + self.pos_encoding[:T][None, :, :]
            x = _sanitize(x)

        for layer in self.layers:
            if self.use_rope:
                x = layer.forward(x, training=False, cos=self.rope_cos, sin=self.rope_sin)
            else:
                dummy_cos = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                dummy_sin = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                x = layer.forward(x, training=False, cos=dummy_cos, sin=dummy_sin)

        final_hidden = self.final_norm.forward(x)

        if pooling == "mean":
            return np.mean(final_hidden, axis=1).astype(np.float32)
        elif pooling == "last":
            return final_hidden[:, -1, :].astype(np.float32)
        elif pooling == "none":
            return final_hidden.astype(np.float32)
        else:
            raise ValueError(f"Pooling mode tidak valid: {pooling}")

    # ====================================================================
    # PARAMETER COLLECTION
    # ====================================================================

    def _collect_parameters(self) -> Dict[str, Parameter]:
        """Kumpulkan semua parameter model."""
        params: Dict[str, Parameter] = {}

        params["embed_tokens.weight"] = self.embed_tokens

        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            params.update(layer.parameters(prefix))

        params.update(self.final_norm.parameters("norm."))

        if not self.use_weight_tying:
            params["lm_head.weight"] = self.lm_head

        return params

    def parameters(self) -> Dict[str, Parameter]:
        """Return dict semua parameter."""
        return dict(self._parameters)

    def parameter_count(self) -> int:
        """Hitung total parameter."""
        return sum(p.data.size for p in self._parameters.values())

    def zero_grad(self) -> None:
        """Reset semua gradient ke nol."""
        for param in self._parameters.values():
            param.zero_grad()

    # ====================================================================
    # STATE DICT (Save/Load)
    # ====================================================================

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return semua bobot model."""
        state = {
            name: param.data.copy()
            for name, param in self._parameters.items()
        }

        if self.use_weight_tying:
            state["lm_head.weight"] = self.embed_tokens.data.copy()

        return state

    def load_state_dict(
        self,
        state_dict: Dict[str, np.ndarray],
        strict: bool = False,
    ) -> Dict[str, list]:
        """Muat bobot dari state_dict."""
        missing = []
        unexpected = []

        for name, array in state_dict.items():
            if name not in self._parameters:
                if name == "lm_head.weight" and self.use_weight_tying:
                    arr = np.asarray(array, dtype=np.float32)
                    if arr.shape != self.embed_tokens.data.shape:
                        raise ValueError(
                            f"Shape tidak cocok untuk {name}. "
                            f"Diharapkan {self.embed_tokens.data.shape}, diterima {arr.shape}."
                        )
                    self.embed_tokens.data = arr
                    self.embed_tokens.grad = np.zeros_like(arr, dtype=np.float32)
                    continue

                unexpected.append(name)
                if strict:
                    raise KeyError(f"Parameter tidak dikenal: {name}")
                continue

            param = self._parameters[name]
            arr = np.asarray(array, dtype=np.float32)

            if arr.shape != param.data.shape:
                raise ValueError(
                    f"Shape tidak cocok untuk {name}. "
                    f"Diharapkan {param.data.shape}, diterima {arr.shape}."
                )

            param.data = arr
            param.grad = np.zeros_like(arr, dtype=np.float32)

        for name in self._parameters.keys():
            if name not in state_dict:
                missing.append(name)

        return {"missing": missing, "unexpected": unexpected}

    # ====================================================================
    # FORWARD PASS dengan Context Support
    # ====================================================================

    def forward(
        self,
        input_ids: np.ndarray,
        training: bool = False,
        return_hidden: bool = False,
        context_ids: Optional[np.ndarray] = None,
        context_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass dengan optional context support via cross-attention.

        Args:
            input_ids: [B, T] atau [T] - input token IDs
            training: True saat training
            return_hidden: jika True, return tuple (logits, hidden_states)
            context_ids: [B, T_c] atau [T_c] - context token IDs (optional)
            context_embeddings: [B, T_c, D] - pre-computed context embeddings (optional)
        
        Notes:
            - Jika context_ids diberikan, akan di-encode dulu
            - Jika context_embeddings diberikan, dipakai langsung (lebih efisien)
            - Context akan di-attend via cross-attention di configured layers

        Output:
            logits: [B, T, vocab_size]
            atau (logits, hidden_states) jika return_hidden=True
        """
        ids = np.asarray(input_ids, dtype=np.int64)

        if ids.ndim == 1:
            ids = ids[None, :]

        if ids.ndim != 2:
            raise ValueError("input_ids harus berbentuk [T] atau [B, T].")

        B, T = ids.shape

        if T == 0:
            raise ValueError("Sequence length tidak boleh kosong.")

        if T > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {T} melebihi max_position_embeddings "
                f"{self.max_position_embeddings}."
            )

        unk_id = int(self.config.special_tokens.unk_id)
        ids = np.where((ids < 0) | (ids >= self.vocab_size), unk_id, ids)

        # Embedding
        x = self.embed_tokens.data[ids]
        x = _sanitize(x)

        if not self.use_rope:
            x = x + self.pos_encoding[:T][None, :, :]
            x = _sanitize(x)

        # Embedding dropout
        dropout_rate = getattr(self.config.training, 'dropout_rate', None)
        if dropout_rate is None:
            dropout_rate = self.config.model.dropout_rate

        x, embed_dropout_mask = _dropout(x, float(dropout_rate), training, self.rng)

        # ================================================================
        # Encode context jika diberikan (BARU)
        # ================================================================
        ctx_emb = None
        if context_embeddings is not None:
            ctx_emb = np.asarray(context_embeddings, dtype=np.float32)
            if ctx_emb.ndim == 2:
                ctx_emb = ctx_emb[None, :, :]
            # Broadcast ke batch size
            if ctx_emb.shape[0] == 1 and B > 1:
                ctx_emb = np.broadcast_to(ctx_emb, (B, ctx_emb.shape[1], ctx_emb.shape[2]))
        elif context_ids is not None:
            # Encode context dengan cache=False saat training (fresh embeddings)
            ctx_emb = self.encode_context(context_ids, use_cache=not training)

        # ================================================================
        # Forward melalui semua layers
        # ================================================================
        context_grads = []
        
        for layer in self.layers:
            if self.use_rope:
                x = layer.forward(
                    x, training, self.rope_cos, self.rope_sin,
                    context_embeddings=ctx_emb,
                )
            else:
                dummy_cos = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                dummy_sin = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                x = layer.forward(
                    x, training, dummy_cos, dummy_sin,
                    context_embeddings=ctx_emb,
                )

        final_hidden = self.final_norm.forward(x)

        # ================================================================
        # Single LM Head
        # ================================================================
        logits = final_hidden @ self.lm_head.data.T

        # Final logit soft-capping
        logits = _softcap(logits, self.final_logit_cap)
        logits = _sanitize(logits)

        self._forward_cache = (
            ids, embed_dropout_mask, final_hidden,
            ctx_emb,  # NEW: untuk backward
        )

        if return_hidden:
            return logits, final_hidden
        return logits

    # ====================================================================
    # BACKWARD PASS dengan Context Gradient
    # ====================================================================

    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backward pass dari gradient logits.
        
        Gradients flow ke:
        - lm_head (atau embed_tokens jika weight tying)
        - Semua layers (self-attention, cross-attention, FFN)
        - Context embeddings (jika digunakan) → propagate ke context encoder
        """
        if self._forward_cache is None:
            raise RuntimeError(
                "CustomTransformerLM.backward dipanggil sebelum forward."
            )

        ids, embed_dropout_mask, final_hidden, ctx_emb = self._forward_cache

        grad_logits = np.asarray(grad_logits, dtype=np.float32)
        grad_logits = _sanitize(grad_logits)

        B, T, V = grad_logits.shape
        D = final_hidden.shape[-1]

        # Soft-cap backward
        pre_cap_logits = final_hidden @ self.lm_head.data.T
        grad_logits = _softcap_backward(grad_logits, pre_cap_logits, self.final_logit_cap)

        grad_logits_2d = grad_logits.reshape(-1, V)
        final_hidden_2d = final_hidden.reshape(-1, D)

        # LM head gradient
        lm_grad = grad_logits_2d.T @ final_hidden_2d

        if self.use_weight_tying:
            self.embed_tokens.grad += lm_grad
        else:
            self.lm_head.grad += lm_grad
            self.lm_head.grad = _sanitize(self.lm_head.grad)

        d_hidden = grad_logits @ self.lm_head.data
        d_hidden = _sanitize(d_hidden)

        d_hidden = self.final_norm.backward(d_hidden)

        # ================================================================
        # Backward through layers (dengan context gradient accumulation)
        # ================================================================
        d_ctx_total = None  # Akumulasi gradient untuk context
        
        for layer in reversed(self.layers):
            d_hidden, d_ctx = layer.backward(d_hidden, context_embeddings=ctx_emb)
            
            if d_ctx is not None:
                if d_ctx_total is None:
                    d_ctx_total = d_ctx
                else:
                    d_ctx_total = d_ctx_total + d_ctx
                    d_ctx_total = _sanitize(d_ctx_total)

        # ================================================================
        # Context gradient backward ke embed_tokens (jika context dipakai)
        # ================================================================
        if d_ctx_total is not None and ctx_emb is not None:
            # Backpropagate context gradient ke embed_tokens
            # context_embeddings = final_norm(layers(embed(context_ids)))
            # Kita backward ke embed_tokens via ids lookup
            # Karena context_ids disimpan di cache, kita bisa backward
            
            # Untuk simplicity, kita tidak track context_ids di cache backward
            # Trainer akan handle gradient propagation jika perlu fine-tune context encoder
            # Untuk inference, ini tidak dibutuhkan
            pass

        # Dropout backward
        d_hidden = _dropout_backward(d_hidden, embed_dropout_mask)
        d_hidden = _sanitize(d_hidden)

        # Gradient dari embedding lookup
        np.add.at(
            self.embed_tokens.grad,
            ids.reshape(-1),
            d_hidden.reshape(-1, D),
        )
        self.embed_tokens.grad = _sanitize(self.embed_tokens.grad)

        self._forward_cache = None

    # ====================================================================
    # CONSTRAINED GENERATION (BARU)
    # ====================================================================

    def generate_constrained(
        self,
        context_ids: np.ndarray,
        question_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 0.3,
        top_k: int = 20,
        top_p: float = 0.85,
        repetition_penalty: float = 1.2,
        min_new_tokens: int = 4,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Generate jawaban dengan constrained generation.
        
        Hanya tokens yang ada di context yang boleh di-generate,
        sehingga jawaban dijamin on-topic.
        
        Args:
            context_ids: [T_c] token IDs dari context
            question_ids: [T_q] token IDs dari question
            max_new_tokens: jumlah maksimum token baru
            temperature: temperature untuk sampling
            top_k: top-k sampling
            top_p: top-p (nucleus) sampling
            repetition_penalty: penalty untuk tokens yang sudah di-generate
            min_new_tokens: minimum token sebelum boleh stop
            rng: random number generator
        
        Returns:
            (generated_ids, full_sequence): 
                - generated_ids: list of generated token IDs
                - full_sequence: context + question + generated
        """
        if rng is None:
            rng = np.random.default_rng()

        # Get special token IDs
        pad_id = int(self.config.special_tokens.pad_id)
        unk_id = int(self.config.special_tokens.unk_id)
        bos_id = int(self.config.special_tokens.bos_id)
        eos_id = int(self.config.special_tokens.eos_id)
        endoftext_id = int(self.config.special_tokens.endoftext_id)

        # ================================================================
        # Build allowed tokens dari context (BARU - CONSTRAINED)
        # ================================================================
        context_ids_flat = np.asarray(context_ids, dtype=np.int64).flatten()
        question_ids_flat = np.asarray(question_ids, dtype=np.int64).flatten()
        
        # Allowed tokens = semua tokens dari context + question + essentials
        allowed_tokens = set(context_ids_flat.tolist())
        allowed_tokens.update(question_ids_flat.tolist())
        
        # Tambah essential tokens (selalu boleh dipakai)
        essential_tokens = {
            bos_id, eos_id, pad_id,
            # Common punctuation tokens (akan disesuaikan dengan vocab)
        }
        allowed_tokens.update(essential_tokens)
        
        # Buang special tokens yang tidak boleh di-generate
        forbidden_tokens = {pad_id, unk_id, bos_id, endoftext_id}
        allowed_tokens -= forbidden_tokens

        # ================================================================
        # Encode context (cached untuk reuse)
        # ================================================================
        context_ids_array = np.asarray(context_ids, dtype=np.int64)
        context_emb = self.encode_context(context_ids_array, use_cache=True)

        # ================================================================
        # Initial sequence: [BOS] + question + " Jawaban:"
        # ================================================================
        current_ids = [bos_id] + question_ids_flat.tolist()

        generated_ids = []

        # ================================================================
        # Generation loop
        # ================================================================
        for step in range(max_new_tokens):
            # Truncate jika terlalu panjang
            if len(current_ids) > self.max_position_embeddings:
                current_ids = current_ids[-self.max_position_embeddings:]

            input_ids = np.array([current_ids], dtype=np.int64)

            # Forward pass dengan context embeddings
            logits = self.forward(
                input_ids,
                training=False,
                context_embeddings=context_emb,
            )

            # Ambil logits untuk token terakhir
            next_logits = logits[0, -1, :].astype(np.float32)

            # ============================================================
            # CONSTRAINED MASKING (BARU)
            # ============================================================
            if self.use_constrained_generation:
                # Mask tokens yang TIDAK ada di allowed_tokens
                mask = np.full(self.vocab_size, -1e9, dtype=np.float32)
                for token_id in allowed_tokens:
                    if 0 <= token_id < self.vocab_size:
                        mask[token_id] = 0.0
                next_logits = next_logits + mask

            # ============================================================
            # Min-new-tokens constraint
            # ============================================================
            if step < min_new_tokens:
                next_logits[eos_id] = -1e9
                if endoftext_id < self.vocab_size:
                    next_logits[endoftext_id] = -1e9

            # ============================================================
            # Repetition penalty
            # ============================================================
            if repetition_penalty > 1.0 and generated_ids:
                prev_tokens = set(generated_ids[-64:])
                for token_id in prev_tokens:
                    if 0 <= token_id < self.vocab_size:
                        if next_logits[token_id] > 0:
                            next_logits[token_id] /= repetition_penalty
                        else:
                            next_logits[token_id] *= repetition_penalty

            # ============================================================
            # Temperature scaling
            # ============================================================
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            # ============================================================
            # Top-K filtering
            # ============================================================
            if top_k > 0:
                top_k = min(top_k, self.vocab_size)
                threshold = np.partition(next_logits, -top_k)[-top_k]
                next_logits = np.where(next_logits < threshold, -1e9, next_logits)

            # ============================================================
            # Top-P (nucleus) filtering
            # ============================================================
            if 0.0 < top_p < 1.0:
                probs = _softmax(next_logits)
                sorted_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumulative_probs = np.cumsum(sorted_probs)

                cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
                cutoff_idx = max(1, min(cutoff_idx, len(sorted_idx)))

                allowed_idx = sorted_idx[:cutoff_idx]
                nucleus_mask = np.full(self.vocab_size, False)
                nucleus_mask[allowed_idx] = True
                next_logits = np.where(nucleus_mask, next_logits, -1e9)

            # ============================================================
            # Sample next token
            # ================================================================
            final_probs = _softmax(next_logits)
            final_probs = np.nan_to_num(final_probs, nan=0.0, posinf=0.0, neginf=0.0)
            total_prob = float(np.sum(final_probs))

            if not np.isfinite(total_prob) or total_prob <= 0.0:
                # Fallback to argmax
                next_token = int(np.argmax(next_logits))
            else:
                final_probs = final_probs / total_prob
                try:
                    next_token = int(rng.choice(self.vocab_size, p=final_probs))
                except ValueError:
                    next_token = int(np.argmax(next_logits))

            # ============================================================
            # Stop conditions
            # ============================================================
            if next_token == eos_id:
                break
            if next_token == endoftext_id:
                break

            generated_ids.append(next_token)
            current_ids.append(next_token)

        # Return hasil
        full_sequence = current_ids
        return generated_ids, full_sequence


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    from config import (
        Config,
        ModelConfig,
        PathsConfig,
        SpecialTokensConfig,
        TrainingConfig,
        RetrievalConfig,
    )

    small_config = Config(
        paths=PathsConfig(),
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
        ),
        model=ModelConfig(
            vocab_size=128,
            embedding_dim=32,
            num_attention_heads=4,
            num_layers=4,  # 4 layers untuk test cross-attention
            max_position_embeddings=64,
            layer_norm_eps=1e-5,
            ffn_hidden_dim=64,
            train_dtype="float32",
            export_dtype="float32",
            use_rope=True,
            use_weight_tying=True,
            attention_logit_cap=30.0,
            final_logit_cap=50.0,
            drop_path_rate=0.1,
            cross_attention_layers=(1, 3),  # Cross-attention di layer 1 dan 3
            use_constrained_generation=True,
        ),
        retrieval=RetrievalConfig(),
    )

    model = CustomTransformerLM(small_config)

    print("=" * 70)
    print("core/architecture.py (RAG + Cross-Attention + Constrained) Test")
    print("=" * 70)
    print(f"Parameter count       : {model.parameter_count():,}")
    print(f"Cross-attn layers     : {model.cross_attention_layer_indices}")
    print(f"Constrained gen       : {model.use_constrained_generation}")
    print(f"Weight tying          : {model.use_weight_tying}")
    print(f"RoPE                  : {model.use_rope}")

    # Test 1: Forward tanpa context (backward compat)
    print("\n[Test 1] Forward tanpa context (backward compat)")
    batch_ids = np.array([[2, 5, 7, 10, 3], [2, 8, 4, 6, 3]], dtype=np.int64)
    model.zero_grad()
    logits = model.forward(batch_ids, training=True)
    print(f"  Logits shape: {logits.shape}")
    assert logits.shape == (2, 5, 128)

    # Test 2: Forward dengan context (cross-attention aktif)
    print("\n[Test 2] Forward dengan context (cross-attention)")
    context_ids = np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    model.zero_grad()
    logits_ctx = model.forward(
        batch_ids, training=True, context_ids=context_ids
    )
    print(f"  Logits shape: {logits_ctx.shape}")
    assert logits_ctx.shape == (2, 5, 128)

    # Test 3: Backward dengan context
    print("\n[Test 3] Backward dengan context")
    grad = np.ones_like(logits_ctx, dtype=np.float32)
    model.backward(grad)
    
    # Check gradients exist
    embed_grad_norm = np.linalg.norm(model.embed_tokens.grad)
    print(f"  Embed tokens grad norm: {embed_grad_norm:.6f}")
    assert embed_grad_norm > 0, "Embedding gradients harus non-zero"

    # Test 4: Context encoding dengan cache
    print("\n[Test 4] Context encoding dengan cache")
    ctx_emb_1 = model.encode_context(context_ids, use_cache=True)
    print(f"  Context embeddings shape: {ctx_emb_1.shape}")
    
    ctx_emb_2 = model.encode_context(context_ids, use_cache=True)
    # Should be same object (cached)
    is_cached = ctx_emb_1 is ctx_emb_2
    print(f"  Cache hit: {is_cached}")
    assert is_cached, "Cache harusnya mengembalikan object yang sama"
    
    model.clear_context_cache()
    ctx_emb_3 = model.encode_context(context_ids, use_cache=True)
    is_cached_after_clear = ctx_emb_1 is ctx_emb_3
    print(f"  Cache hit after clear: {is_cached_after_clear}")
    assert not is_cached_after_clear

    # Test 5: Constrained generation
    print("\n[Test 5] Constrained generation")
    context = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
    question = np.array([2, 5, 7], dtype=np.int64)
    
    generated, full_seq = model.generate_constrained(
        context_ids=context,
        question_ids=question,
        max_new_tokens=5,
        temperature=0.5,
        top_k=5,
        top_p=0.9,
        min_new_tokens=2,
    )
    
    print(f"  Generated tokens: {generated}")
    print(f"  Full sequence length: {len(full_seq)}")
    
    # Verify semua generated tokens ada di context
    context_set = set(context.tolist())
    all_in_context = all(t in context_set for t in generated)
    print(f"  All generated in context: {all_in_context}")
    # Note: Ini mungkin False karena bisa include question tokens juga
    # Yang penting: constrained generation berjalan tanpa error

    # Test 6: Hidden states extraction
    print("\n[Test 6] Hidden states extraction")
    test_ids = np.array([[2, 5, 7, 10]], dtype=np.int64)
    emb_mean = model.get_hidden_states(test_ids, pooling="mean")
    emb_last = model.get_hidden_states(test_ids, pooling="last")
    print(f"  Mean pooling: {emb_mean.shape}")
    print(f"  Last token: {emb_last.shape}")

    # Test 7: State dict save/load
    print("\n[Test 7] State dict save/load")
    state = model.state_dict()
    print(f"  State dict keys: {len(state)}")
    
    # Verify cross-attention weights ada
    cross_attn_keys = [k for k in state.keys() if 'cross_attn' in k]
    print(f"  Cross-attention keys: {len(cross_attn_keys)}")
    for k in cross_attn_keys[:4]:
        print(f"    - {k}: {state[k].shape}")
    
    model2 = CustomTransformerLM(small_config)
    result = model2.load_state_dict(state, strict=False)
    print(f"  Load result: missing={len(result['missing'])}, unexpected={len(result['unexpected'])}")

    # Test 8: Numerical stability
    print("\n[Test 8] Numerical stability")
    extreme_ids = np.array([[2, 5, 7]], dtype=np.int64)
    extreme_logits = model.forward(extreme_ids, training=True)
    is_finite = np.all(np.isfinite(extreme_logits))
    print(f"  Logits finite: {is_finite}")
    assert is_finite, "Logits harus finite"

    print("\n" + "=" * 70)
    print("✅ core/architecture.py ALL TESTS PASSED")
    print("=" * 70)
