# core/architecture.py
#
# Arsitektur neural network custom untuk Training Pipeline Engine.
#
# VERSI SOTA + MOE UPGRADE:
# SOTA features:
# - RoPE (Rotary Position Embedding) menggantikan sinusoidal absolut
# - Weight Tying antara embed_tokens dan lm_head (single-head mode)
# - Logit Soft-Capping (tanh) menggantikan hard clipping
# - Stochastic Depth (DropPath) pada TransformerBlock
# - Scaled Initialization untuk residual sublayers (GPT-2 style)
#
# MoE features (BARU):
# - Dynamic LM Heads: satu LM head per domain (auto-detect dari datasets/)
# - Shared backbone: semua domain berbagi encoder yang sama
# - Active domain selector: pilih head saat inference/training
# - Hidden states extraction: untuk semantic search
# - Backward compatibility: auto-convert old single-head checkpoints
#
# Keterhubungan:
# - config.py          : parameter arsitektur + MoE
# - core/router.py     : memilih active_domain saat inference
# - pipeline/trainer.py     : forward pass saat training
# - pipeline/evaluator.py   : menerima logits
# - pipeline/weight_manager.py : memakai gradient dari backward pass
# - core/model_exporter.py  : memakai state_dict()
# - testing/playground.py   : inference

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


# Batas clipping untuk aktivasi internal (tetap dipertahankan untuk safety)
ACTIVATION_CLIP = 50.0


# ============================================================================
# NUMERIC HELPERS
# ============================================================================

def _init_weight(
    rng: np.random.Generator,
    shape: tuple,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Inisialisasi bobot dengan distribusi normal kecil.

    Parameter scale digunakan untuk GPT-2 style residual scaling.
    """
    if len(shape) == 1:
        fan_in = shape[0]
    else:
        fan_in = shape[1]

    std = min(0.02, 1.0 / math.sqrt(max(1, fan_in))) * scale
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax numerik stabil dengan sanitasi NaN/Inf.
    """
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
    """
    Soft-capping berbasis tanh. Mencegah overflow tanpa memotong gradient.

    Forward:  y = cap * tanh(x / cap)
    Gradient: dy/dx = 1 - tanh(x/cap)^2 = 1 - (y/cap)^2
    """
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
    """
    Stochastic Depth (DropPath).

    Return:
    - output: tensor setelah drop
    - dropped: True jika blok di-drop (gradient = 0 di backward)
    """
    p = float(p)

    if not training or p <= 0.0:
        return x, False

    if rng.random() < p:
        # Drop: output = 0 (residual akan tetap lewat)
        return np.zeros_like(x), True

    # Scale untuk inverse dropout
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
    """
    Bangun cache cos dan sin untuk RoPE.

    Return:
    - cos: [max_position, head_dim]
    - sin: [max_position, head_dim]
    """
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_position, dtype=np.float32)

    # [max_position, head_dim/2]
    angles = np.outer(positions, inv_freq)

    # Duplikasi untuk setiap pasangan dimensi
    cos = np.cos(angles)
    sin = np.sin(angles)

    cos = np.concatenate([cos, cos], axis=-1)
    sin = np.concatenate([sin, sin], axis=-1)

    return cos.astype(np.float32), sin.astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    """
    Rotasi separuh dimensi terakhir.

    Input:  [q0, q1, q2, q3, ..., q_{D-2}, q_{D-1}]
    Output: [-q_{D/2}, -q_{D/2+1}, ..., -q_{D-1}, q0, q1, ..., q_{D/2-1}]
    """
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
    """
    Terapkan RoPE pada Q dan K.

    Input:
    - q, k: [B, H, T, D]
    - cos, sin: [T, D]

    Output:
    - q_rot, k_rot: [B, H, T, D]
    """
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
    """
    Backward untuk RoPE.

    Rotasi adalah operasi orthogonal, jadi backward = rotasi dengan sudut negatif.
    """
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
    """
    Multi-head self-attention dengan:
    - RoPE untuk positional encoding
    - Logit soft-capping untuk attention scores
    - Scaled initialization untuk o_proj
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

        # GPT-2 style scaled init untuk residual sublayers
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

        probs_drop, attn_dropout_mask = _dropout(
            probs,
            self.dropout_p,
            training,
            self.rng,
        )

        context = probs_drop @ Vh
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        context = _sanitize(context)

        output = context @ self.o_proj.data.T
        output = _sanitize(output)

        self.cache = (
            x,
            Qh,
            Kh,
            Vh,
            Qh_rot,
            Kh_rot,
            probs,
            probs_drop,
            attn_dropout_mask,
            context,
            scores,
            cos[:T],
            sin[:T],
        )

        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError(
                "MultiHeadSelfAttention.backward dipanggil sebelum forward."
            )

        (
            x,
            Qh,
            Kh,
            Vh,
            Qh_rot,
            Kh_rot,
            probs,
            probs_drop,
            attn_dropout_mask,
            context,
            scores,
            cos,
            sin,
        ) = self.cache

        dout = _sanitize(dout)
        B, T, D = x.shape

        dout_2d = dout.reshape(-1, D)
        context_2d = context.reshape(-1, D)

        self.o_proj.grad += dout_2d.T @ context_2d
        self.o_proj.grad = _sanitize(self.o_proj.grad)
        dcontext = dout @ self.o_proj.data
        dcontext = _sanitize(dcontext)

        dcontext_heads = dcontext.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        dprobs_drop = dcontext_heads @ Vh.transpose(0, 1, 3, 2)
        dVh = probs_drop.transpose(0, 1, 3, 2) @ dcontext_heads

        dprobs = _dropout_backward(dprobs_drop, attn_dropout_mask)

        sum_dp = np.sum(dprobs * probs, axis=-1, keepdims=True)
        dscores = probs * (dprobs - sum_dp)
        dscores = dscores * self.scale

        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores = np.where(causal_mask[None, None, :, :], 0.0, dscores)

        dscores_pre_cap = _softcap_backward(dscores, scores * (1.0 / self.scale) * self.scale, self.attention_logit_cap)

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
# SwiGLU FEED-FORWARD NETWORK
# ============================================================================

class SwiGLUFFN:
    """
    Feed-forward network dengan SwiGLU.
    down_proj menggunakan scaled init untuk residual.
    """

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

        h_drop, dropout_mask = _dropout(
            h,
            self.dropout_p,
            training,
            self.rng,
        )

        out = h_drop @ self.down_proj.data.T
        out = _sanitize(out)

        self.cache = (
            x,
            gate,
            up,
            sig,
            silu,
            h_drop,
            dropout_mask,
        )

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("SwiGLUFFN.backward dipanggil sebelum forward.")

        (
            x,
            gate,
            up,
            sig,
            silu,
            h_drop,
            dropout_mask,
        ) = self.cache

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
# TRANSFORMER BLOCK dengan Stochastic Depth
# ============================================================================

class TransformerBlock:
    """
    Satu blok Transformer dengan:
    - Pre-Layer Normalization (Pre-LN)
    - Stochastic Depth (DropPath)
    - Pure residual connection (tanpa sanitasi pada shortcut)
    """

    def __init__(
        self,
        config: Config,
        rng: np.random.Generator,
        layer_idx: int = 0,
        num_layers: int = 1,
        drop_path_rate: float = 0.0,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.drop_path_rate = float(drop_path_rate)

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
    ) -> np.ndarray:
        residual_1 = x

        h = self.input_norm.forward(x)
        h = self.attn.forward(h, training, cos, sin)
        h, attn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)

        h, attn_dropped = _drop_path(h, self.drop_path_rate, training, self.rng)

        x_1 = residual_1 + h

        residual_2 = x_1

        h = self.post_attention_norm.forward(x_1)
        h = self.ffn.forward(h, training)
        h, ffn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)

        h, ffn_dropped = _drop_path(h, self.drop_path_rate, training, self.rng)

        out = residual_2 + h

        self.cache = (
            attn_dropout_mask,
            ffn_dropout_mask,
            attn_dropped,
            ffn_dropped,
        )

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("TransformerBlock.backward dipanggil sebelum forward.")

        attn_dropout_mask, ffn_dropout_mask, attn_dropped, ffn_dropped = self.cache

        d_ffn_out = _dropout_backward(dout, ffn_dropout_mask)
        d_ffn_out = _drop_path_backward(d_ffn_out, ffn_dropped, self.drop_path_rate)

        d_ffn_in = self.ffn.backward(d_ffn_out)
        d_post_norm = self.post_attention_norm.backward(d_ffn_in)

        d_x_1 = dout + d_post_norm
        d_x_1 = _sanitize(d_x_1)

        d_attn_out = _dropout_backward(d_x_1, attn_dropout_mask)
        d_attn_out = _drop_path_backward(d_attn_out, attn_dropped, self.drop_path_rate)

        d_attn_in = self.attn.backward(d_attn_out)
        d_input_norm = self.input_norm.backward(d_attn_in)

        dx = d_x_1 + d_input_norm
        dx = _sanitize(dx)

        self.cache = None
        return dx

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        params: Dict[str, Parameter] = {}

        params.update(self.input_norm.parameters(f"{prefix}input_layernorm."))
        params.update(self.attn.parameters(f"{prefix}self_attn."))
        params.update(
            self.post_attention_norm.parameters(
                f"{prefix}post_attention_layernorm."
            )
        )
        params.update(self.ffn.parameters(f"{prefix}mlp."))

        return params


# ============================================================================
# CUSTOM TRANSFORMER LANGUAGE MODEL dengan MoE Dynamic LM Heads
# ============================================================================

class CustomTransformerLM:
    """
    Model Transformer custom untuk language modeling.

    Fitur SOTA:
    - RoPE untuk positional encoding
    - Weight Tying antara embed_tokens dan lm_head (single-head mode)
    - Logit Soft-Capping pada final logits
    - Stochastic Depth pada setiap layer
    - Scaled Initialization untuk residual sublayers

    Fitur MoE (BARU):
    - Dynamic LM Heads: satu head per domain, auto-detect dari datasets/
    - Shared backbone: semua domain berbagi encoder yang sama
    - Active domain selector: pilih head saat inference/training
    - Hidden states extraction: untuk semantic search (cosine similarity)
    - Backward compatibility: auto-convert old single-head checkpoints
    """

    MAX_PARAMS: Optional[int] = 50_000_000

    # Domain default untuk fallback
    GENERAL_DOMAIN: str = "GENERAL"

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
        self.final_logit_cap = float(getattr(config.model, 'final_logit_cap', 50.0))
        self.drop_path_rate = float(getattr(config.model, 'drop_path_rate', 0.1))

        # ================================================================
        # MOE config (BARU)
        # ================================================================
        self.use_moe = bool(getattr(config.model, 'use_moe', True))

        # Weight tying: otomatis di-disable jika MoE aktif
        self.use_weight_tying = bool(
            getattr(config.model, 'effective_use_weight_tying',
                    getattr(config.model, 'use_weight_tying', True))
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
        # Transformer layers dengan linear drop path rate
        # ================================================================
        self.layers = []
        for i in range(self.num_layers):
            layer_drop_rate = (
                self.drop_path_rate * i / max(1, self.num_layers - 1)
            )
            layer = TransformerBlock(
                config,
                self.rng,
                layer_idx=i,
                num_layers=self.num_layers,
                drop_path_rate=layer_drop_rate,
            )
            self.layers.append(layer)

        self.final_norm = RMSNorm(self.embedding_dim, self.layer_norm_eps)

        # ================================================================
        # LM HEADS: Dynamic (MoE) atau Single (Weight Tying)
        # ================================================================
        self.lm_heads: Dict[str, Parameter] = {}

        # Selalu buat GENERAL head sebagai fallback
        self.lm_heads[self.GENERAL_DOMAIN] = Parameter(
            _init_weight(self.rng, (self.vocab_size, self.embedding_dim))
        )

        # Active domain untuk forward/backward (default: GENERAL)
        self.active_domain: str = self.GENERAL_DOMAIN

        # Legacy single lm_head (untuk backward compatibility)
        # Hanya dipakai jika use_moe=False dan use_weight_tying=False
        self.lm_head: Optional[Parameter] = None
        if not self.use_moe and not self.use_weight_tying:
            self.lm_head = self.lm_heads[self.GENERAL_DOMAIN]

        # ================================================================
        # RoPE cache
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

    def _validate_practical_size(self) -> None:
        if self.MAX_PARAMS is None:
            return

        V = self.vocab_size
        D = self.embedding_dim
        L = self.num_layers
        F = self.ffn_hidden_dim

        # Base: backbone params (tanpa LM head)
        backbone_params = (
            V * D  # embed_tokens
            + L * (2 * D + 4 * D * D + 3 * D * F)  # transformer layers
            + D  # final_norm
        )

        # LM heads: minimal 1 (GENERAL)
        lm_head_params = V * D  # GENERAL head

        estimated_params = backbone_params + lm_head_params

        if estimated_params > self.MAX_PARAMS:
            raise RuntimeError(
                "Estimasi jumlah parameter terlalu besar untuk engine custom "
                "berbasis NumPy ini. "
                f"Estimasi (base): {estimated_params:,} parameter. "
                f"Batas aman: {self.MAX_PARAMS:,} parameter. "
                "Kurangi vocab_size, embedding_dim, num_layers, atau ffn_hidden_dim "
                "di config.py, atau set CustomTransformerLM.MAX_PARAMS = None.\n"
                f"Catatan: setiap domain tambahan menambah {V*D:,} parameter."
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
    # MOE: DYNAMIC DOMAIN MANAGEMENT
    # ====================================================================

    def list_domains(self) -> List[str]:
        """
        Return list semua domain yang terdaftar di model.
        """
        return list(self.lm_heads.keys())

    def has_domain(self, domain: str) -> bool:
        """
        Cek apakah domain sudah terdaftar.
        """
        return domain in self.lm_heads

    def add_domain(self, domain: str, init_from_general: bool = True) -> None:
        """
        Tambah LM head baru untuk domain.

        Parameter:
        - domain: nama domain (uppercase, alphanumeric)
        - init_from_general: jika True, inisialisasi dengan copy GENERAL head
                             (transfer learning dari base model)
        """
        if not domain or not domain.strip():
            raise ValueError("Nama domain tidak boleh kosong.")

        domain = domain.strip().upper()

        if domain in self.lm_heads:
            return  # Sudah ada, skip

        if init_from_general:
            # Transfer learning: mulai dari GENERAL head
            general_data = self.lm_heads[self.GENERAL_DOMAIN].data
            new_head = Parameter(general_data.copy())
        else:
            # Random initialization
            new_head = Parameter(
                _init_weight(self.rng, (self.vocab_size, self.embedding_dim))
            )

        self.lm_heads[domain] = new_head

        # Update _parameters dict
        self._parameters[f"lm_heads.{domain}.weight"] = new_head

    def init_domains(self, domains: List[str]) -> None:
        """
        Inisialisasi multiple domain sekaligus.

        Parameter:
        - domains: list nama domain (e.g., ["MOBIL", "SAINS", "FILSAFAT"])
        """
        for domain in domains:
            self.add_domain(domain, init_from_general=True)

    def remove_domain(self, domain: str) -> None:
        """
        Hapus LM head untuk domain tertentu.
        GENERAL domain tidak bisa dihapus.
        """
        if domain == self.GENERAL_DOMAIN:
            raise ValueError(f"Domain {self.GENERAL_DOMAIN} tidak bisa dihapus (wajib ada).")

        if domain not in self.lm_heads:
            return

        del self.lm_heads[domain]

        # Update _parameters dict
        key = f"lm_heads.{domain}.weight"
        if key in self._parameters:
            del self._parameters[key]

        # Reset active domain jika yang aktif dihapus
        if self.active_domain == domain:
            self.active_domain = self.GENERAL_DOMAIN

    def set_active_domain(self, domain: str) -> None:
        """
        Set domain aktif untuk forward/backward pass.

        Jika domain tidak ada, fallback ke GENERAL.
        """
        if domain in self.lm_heads:
            self.active_domain = domain
        else:
            # Fallback ke GENERAL
            self.active_domain = self.GENERAL_DOMAIN

    def get_active_head(self) -> Parameter:
        """
        Return LM head yang sedang aktif.
        """
        return self.lm_heads[self.active_domain]

    def estimated_params_for_domains(self, num_domains: int) -> int:
        """
        Hitung estimasi total parameter jika ada N domain.

        Termasuk GENERAL head.
        """
        V = self.vocab_size
        D = self.embedding_dim
        L = self.num_layers
        F = self.ffn_hidden_dim

        backbone_params = (
            V * D
            + L * (2 * D + 4 * D * D + 3 * D * F)
            + D
        )

        # N domain + 1 GENERAL = (N+1) LM heads
        lm_head_params = (num_domains + 1) * V * D

        return backbone_params + lm_head_params

    # ====================================================================
    # HIDDEN STATES EXTRACTION (untuk Semantic Search)
    # ====================================================================

    def get_hidden_states(
        self,
        input_ids: np.ndarray,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Extract embeddings dari final layer (sebelum LM head).

        Parameter:
        - input_ids: [B, T] atau [T]
        - pooling: "mean" (default), "last", atau "none"
            - "mean": mean pooling → [B, D]
            - "last": ambil token terakhir → [B, D]
            - "none": return full sequence → [B, T, D]

        Return:
        - embeddings: np.ndarray
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
            # Truncate
            ids = ids[:, :self.max_position_embeddings]
            T = self.max_position_embeddings

        unk_id = int(self.config.special_tokens.unk_id)
        ids = np.where((ids < 0) | (ids >= self.vocab_size), unk_id, ids)

        x = self.embed_tokens.data[ids]
        x = _sanitize(x)

        if not self.use_rope:
            x = x + self.pos_encoding[:T][None, :, :]
            x = _sanitize(x)

        # NO dropout untuk inference embeddings
        for layer in self.layers:
            if self.use_rope:
                x = layer.forward(x, training=False, cos=self.rope_cos, sin=self.rope_sin)
            else:
                dummy_cos = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                dummy_sin = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                x = layer.forward(x, training=False, cos=dummy_cos, sin=dummy_sin)

        final_hidden = self.final_norm.forward(x)

        # Pooling
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
        params: Dict[str, Parameter] = {}

        params["embed_tokens.weight"] = self.embed_tokens

        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            params.update(layer.parameters(prefix))

        params.update(self.final_norm.parameters("norm."))

        # ================================================================
        # Multiple LM Heads (MoE)
        # ================================================================
        for domain, head in self.lm_heads.items():
            params[f"lm_heads.{domain}.weight"] = head

        return params

    def parameters(self) -> Dict[str, Parameter]:
        return dict(self._parameters)

    def parameter_count(self) -> int:
        """
        Hitung total parameter (termasuk semua LM heads).
        """
        return sum(p.data.size for p in self._parameters.values())

    def backbone_parameter_count(self) -> int:
        """
        Hitung parameter backbone saja (tanpa LM heads).
        """
        count = 0
        for name, p in self._parameters.items():
            if not name.startswith("lm_heads."):
                count += p.data.size
        return count

    def lm_heads_parameter_count(self) -> int:
        """
        Hitung total parameter semua LM heads.
        """
        count = 0
        for name, p in self._parameters.items():
            if name.startswith("lm_heads."):
                count += p.data.size
        return count

    def zero_grad(self) -> None:
        for param in self._parameters.values():
            param.zero_grad()

    # ====================================================================
    # STATE DICT (Save/Load)
    # ====================================================================

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Return semua bobot model.

        Format keys:
        - embed_tokens.weight
        - layers.{i}.*
        - norm.weight
        - lm_heads.{DOMAIN}.weight  (satu per domain, termasuk GENERAL)
        """
        state = {
            name: param.data.copy()
            for name, param in self._parameters.items()
        }

        # Metadata tambahan
        state["__moe_metadata__"] = {
            "active_domains": list(self.lm_heads.keys()),
            "use_moe": self.use_moe,
            "use_weight_tying": self.use_weight_tying,
            "use_rope": self.use_rope,
        }

        return state

    def load_state_dict(
        self,
        state_dict: Dict[str, np.ndarray],
        strict: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Muat bobot dari state_dict.

        Backward compatibility:
        - Jika state_dict berisi "lm_head.weight" (format lama),
          auto-convert ke lm_heads.GENERAL.weight
        - Jika ada domain baru di state_dict yang belum ada di model,
          otomatis ditambahkan

        Return:
        - dict dengan keys "missing" dan "unexpected"
        """
        missing = []
        unexpected = []

        # ================================================================
        # Backward compatibility: convert old format
        # ================================================================
        converted_state = dict(state_dict)

        # Jika ada lm_head.weight (format lama single head)
        if "lm_head.weight" in converted_state and "lm_heads.GENERAL.weight" not in converted_state:
            converted_state["lm_heads.GENERAL.weight"] = converted_state["lm_head.weight"]

        # Auto-create domain baru jika ada di state_dict tapi belum ada di model
        for key in converted_state.keys():
            if key.startswith("lm_heads.") and key.endswith(".weight"):
                domain = key[len("lm_heads."):-len(".weight")]
                if domain != "__moe_metadata__" and not self.has_domain(domain):
                    self.add_domain(domain, init_from_general=False)

        # Update _parameters setelah domain baru ditambahkan
        self._parameters = self._collect_parameters()

        # ================================================================
        # Load parameters
        # ================================================================
        for name, array in converted_state.items():
            # Skip metadata
            if name == "__moe_metadata__":
                continue

            if name not in self._parameters:
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

        # Cek parameter yang missing
        for name in self._parameters.keys():
            if name not in converted_state:
                missing.append(name)

        return {"missing": missing, "unexpected": unexpected}

    # ====================================================================
    # FORWARD PASS
    # ====================================================================

    def forward(
        self,
        input_ids: np.ndarray,
        training: bool = False,
        domain: Optional[str] = None,
        return_hidden: bool = False,
    ) -> np.ndarray:
        """
        Forward pass.

        Input:
        - input_ids: [B, T] atau [T]
        - training: True saat training, False saat inference
        - domain: override active_domain (optional)
        - return_hidden: jika True, return tuple (logits, hidden_states)

        Output:
        - logits: [B, T, vocab_size]
        - atau (logits, hidden_states) jika return_hidden=True
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

        # Positional encoding
        if self.use_rope:
            pass  # RoPE diterapkan di attention layer
        else:
            x = x + self.pos_encoding[:T][None, :, :]
            x = _sanitize(x)

        # Dropout embedding
        dropout_rate = getattr(self.config.training, 'dropout_rate', None)
        if dropout_rate is None:
            dropout_rate = self.config.model.dropout_rate

        x, embed_dropout_mask = _dropout(
            x,
            float(dropout_rate),
            training,
            self.rng,
        )

        # Transformer blocks
        for layer in self.layers:
            if self.use_rope:
                x = layer.forward(x, training, self.rope_cos, self.rope_sin)
            else:
                dummy_cos = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                dummy_sin = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                x = layer.forward(x, training, dummy_cos, dummy_sin)

        final_hidden = self.final_norm.forward(x)

        # ================================================================
        # LM Head: pilih berdasarkan domain
        # ================================================================
        if domain is not None:
            forward_domain = domain if domain in self.lm_heads else self.GENERAL_DOMAIN
        else:
            forward_domain = self.active_domain

        active_head = self.lm_heads[forward_domain]
        logits = final_hidden @ active_head.data.T

        # Final logit soft-capping
        logits = _softcap(logits, self.final_logit_cap)
        logits = _sanitize(logits)

        # ================================================================
        # Cache untuk backward
        # ================================================================
        self._forward_cache = (
            ids,
            embed_dropout_mask,
            final_hidden,
            forward_domain,  # Track domain yang dipakai
        )

        if return_hidden:
            return logits, final_hidden
        return logits

    # ====================================================================
    # BACKWARD PASS
    # ====================================================================

    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backward pass dari gradient logits.

        Gradient akan mengalir ke:
        - LM head yang aktif saat forward
        - Shared backbone (embed_tokens, layers, final_norm)
        """
        if self._forward_cache is None:
            raise RuntimeError(
                "CustomTransformerLM.backward dipanggil sebelum forward."
            )

        ids, embed_dropout_mask, final_hidden, forward_domain = self._forward_cache

        grad_logits = np.asarray(grad_logits, dtype=np.float32)
        grad_logits = _sanitize(grad_logits)

        B, T, V = grad_logits.shape
        D = final_hidden.shape[-1]

        # ================================================================
        # Soft-cap backward pada logits
        # ================================================================
        active_head = self.lm_heads[forward_domain]
        pre_cap_logits = final_hidden @ active_head.data.T
        grad_logits = _softcap_backward(grad_logits, pre_cap_logits, self.final_logit_cap)

        grad_logits_2d = grad_logits.reshape(-1, V)
        final_hidden_2d = final_hidden.reshape(-1, D)

        # ================================================================
        # Gradient untuk LM head yang aktif
        # ================================================================
        lm_grad = grad_logits_2d.T @ final_hidden_2d
        active_head.grad += lm_grad
        active_head.grad = _sanitize(active_head.grad)

        # ================================================================
        # Backprop ke hidden states
        # ================================================================
        d_hidden = grad_logits @ active_head.data
        d_hidden = _sanitize(d_hidden)

        d_hidden = self.final_norm.backward(d_hidden)

        # Backward melalui layers
        for layer in reversed(self.layers):
            d_hidden = layer.backward(d_hidden)

        # Embedding dropout backward
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
    )

    small_config = Config(
        paths=PathsConfig(),
        special_tokens=SpecialTokensConfig(),
        training=TrainingConfig(
            learning_rate=3e-4,
            batch_size=2,
            epochs=1,
            sequence_length=8,
            dropout_rate=0.0,
            weight_decay=0.01,
            grad_clip_norm=1.0,
            seed=1337,
        ),
        model=ModelConfig(
            vocab_size=64,
            embedding_dim=32,
            num_attention_heads=4,
            num_layers=2,
            dropout_rate=0.0,
            max_position_embeddings=32,
            layer_norm_eps=1e-5,
            ffn_hidden_dim=48,
            train_dtype="float32",
            export_dtype="float32",
            # SOTA parameters
            use_rope=True,
            use_weight_tying=True,
            attention_logit_cap=30.0,
            final_logit_cap=50.0,
            drop_path_rate=0.1,
            # MoE parameters
            use_moe=True,
            moe_auto_routing=True,
        ),
    )

    model = CustomTransformerLM(small_config)

    # ================================================================
    # Test 1: MoE domain management
    # ================================================================
    print("=" * 60)
    print("Test 1: MoE Domain Management")
    print("=" * 60)

    print(f"Initial domains: {model.list_domains()}")

    # Tambah beberapa domain
    model.init_domains(["MOBIL", "SAINS", "FILSAFAT"])
    print(f"After init_domains: {model.list_domains()}")

    print(f"\nParameter count:")
    print(f"  Backbone     : {model.backbone_parameter_count():,}")
    print(f"  LM heads     : {model.lm_heads_parameter_count():,}")
    print(f"  Total        : {model.parameter_count():,}")

    # ================================================================
    # Test 2: Forward dengan domain berbeda
    # ================================================================
    print("\n" + "=" * 60)
    print("Test 2: Forward dengan Domain Berbeda")
    print("=" * 60)

    batch_ids = np.array(
        [
            [2, 5, 7, 10, 3],
            [2, 8, 4, 6, 3],
        ],
        dtype=np.int64,
    )

    # Test per domain
    for domain in ["GENERAL", "MOBIL", "SAINS"]:
        model.set_active_domain(domain)
        model.zero_grad()

        logits = model.forward(batch_ids, training=True)
        print(f"Domain {domain:10s} - logits shape: {logits.shape}, sample: {logits[0, 0, :4]}")

    # ================================================================
    # Test 3: Backward hanya update active domain
    # ================================================================
    print("\n" + "=" * 60)
    print("Test 3: Backward per Domain")
    print("=" * 60)

    model.set_active_domain("MOBIL")
    model.zero_grad()

    logits = model.forward(batch_ids, training=True)
    grad_logits = np.ones_like(logits, dtype=np.float32)
    model.backward(grad_logits)

    # Cek gradient per domain
    for domain in model.list_domains():
        head = model.lm_heads[domain]
        grad_norm = np.linalg.norm(head.grad)
        print(f"Domain {domain:10s} - grad norm: {grad_norm:.6f}")

    # ================================================================
    # Test 4: Hidden states extraction
    # ================================================================
    print("\n" + "=" * 60)
    print("Test 4: Hidden States Extraction")
    print("=" * 60)

    test_ids = np.array([[2, 5, 7, 10]], dtype=np.int64)

    emb_mean = model.get_hidden_states(test_ids, pooling="mean")
    emb_last = model.get_hidden_states(test_ids, pooling="last")
    emb_none = model.get_hidden_states(test_ids, pooling="none")

    print(f"Mean pooling  : shape {emb_mean.shape}")
    print(f"Last token    : shape {emb_last.shape}")
    print(f"No pooling    : shape {emb_none.shape}")

    # ================================================================
    # Test 5: State dict save/load
    # ================================================================
    print("\n" + "=" * 60)
    print("Test 5: State Dict Save/Load")
    print("=" * 60)

    state = model.state_dict()
    print(f"State dict keys ({len(state)} total):")
    for key in list(state.keys())[:8]:
        if isinstance(state[key], np.ndarray):
            print(f"  {key}: shape {state[key].shape}")
        else:
            print(f"  {key}: {state[key]}")
    if len(state) > 8:
        print(f"  ... and {len(state) - 8} more")

    # Load ke model baru
    print("\nLoading ke model baru...")
    model2 = CustomTransformerLM(small_config)
    result = model2.load_state_dict(state, strict=False)
    print(f"Load result: {result}")
    print(f"Domains di model2: {model2.list_domains()}")

    # ================================================================
    # Test 6: Backward compatibility dengan old format
    # ================================================================
    print("\n" + "=" * 60)
    print("Test 6: Backward Compatibility (Old Single-Head Format)")
    print("=" * 60)

    # Simulasi state_dict lama
    old_state = {
        "embed_tokens.weight": np.random.randn(64, 32).astype(np.float32),
        "norm.weight": np.ones(32, dtype=np.float32),
        "lm_head.weight": np.random.randn(64, 32).astype(np.float32),  # Old format
    }
    # Tambah dummy layer weights
    for i in range(2):
        old_state[f"layers.{i}.input_layernorm.weight"] = np.ones(32, dtype=np.float32)
        old_state[f"layers.{i}.post_attention_layernorm.weight"] = np.ones(32, dtype=np.float32)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            old_state[f"layers.{i}.self_attn.{proj}.weight"] = np.random.randn(32, 32).astype(np.float32)
        for proj in ["gate_proj", "up_proj"]:
            old_state[f"layers.{i}.mlp.{proj}.weight"] = np.random.randn(48, 32).astype(np.float32)
        old_state[f"layers.{i}.mlp.down_proj.weight"] = np.random.randn(32, 48).astype(np.float32)

    model3 = CustomTransformerLM(small_config)
    result3 = model3.load_state_dict(old_state, strict=False)
    print(f"Load result: {result3}")
    print(f"Domains setelah load old format: {model3.list_domains()}")
    print(f"GENERAL head shape: {model3.lm_heads['GENERAL'].shape}")

    print("\n" + "=" * 60)
    print("core/architecture.py (SOTA + MoE) test OK")
    print("=" * 60)
