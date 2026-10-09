# core/architecture.py
#
# Arsitektur neural network custom untuk Training Pipeline Engine.
# VERSI SOTA UPGRADE:
# - RoPE (Rotary Position Embedding) menggantikan sinusoidal absolut
# - Weight Tying antara embed_tokens dan lm_head
# - Logit Soft-Capping (tanh) menggantikan hard clipping
# - Stochastic Depth (DropPath) pada TransformerBlock
# - Scaled Initialization untuk residual sublayers (GPT-2 style)
#
# Keterhubungan:
# - config.py          : parameter arsitektur
# - pipeline/trainer.py     : forward pass saat training
# - pipeline/evaluator.py   : menerima logits
# - pipeline/weight_manager.py : memakai gradient dari backward pass
# - core/model_exporter.py  : memakai state_dict()
# - testing/playground.py   : inference

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    # Reshape cos dan sin: [1, 1, T, D]
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
    cos(-θ) = cos(θ), sin(-θ) = -sin(θ)

    Untuk q: q_rot = q*cos + rotate_half(q)*sin
    dq = dq_rot*cos - rotate_half(dq_rot)*sin
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

        # Logit soft-capping untuk attention scores
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
        # o_proj menggunakan scaled init karena merupakan residual sublayer
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

        # ================================================================
        # RoPE: rotasi pada Q dan K
        # ================================================================
        Qh_rot, Kh_rot = _apply_rotary_emb(Qh, Kh, cos[:T], sin[:T])
        Qh_rot = _sanitize(Qh_rot)
        Kh_rot = _sanitize(Kh_rot)

        # scores: [B, H, T, T]
        scores = Qh_rot @ Kh_rot.transpose(0, 1, 3, 2)
        scores = scores * self.scale

        # ================================================================
        # Soft-capping pada attention scores
        # ================================================================
        scores = _softcap(scores, self.attention_logit_cap)

        # Causal mask: token tidak boleh melihat masa depan
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

        # ================================================================
        # Output projection backward
        # ================================================================
        dout_2d = dout.reshape(-1, D)
        context_2d = context.reshape(-1, D)

        self.o_proj.grad += dout_2d.T @ context_2d
        self.o_proj.grad = _sanitize(self.o_proj.grad)
        dcontext = dout @ self.o_proj.data
        dcontext = _sanitize(dcontext)

        dcontext_heads = dcontext.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # ================================================================
        # Gradient terhadap attention probabilities dan value
        # ================================================================
        dprobs_drop = dcontext_heads @ Vh.transpose(0, 1, 3, 2)
        dVh = probs_drop.transpose(0, 1, 3, 2) @ dcontext_heads

        # Dropout backward pada attention probabilities
        dprobs = _dropout_backward(dprobs_drop, attn_dropout_mask)

        # Softmax backward
        sum_dp = np.sum(dprobs * probs, axis=-1, keepdims=True)
        dscores = probs * (dprobs - sum_dp)
        dscores = dscores * self.scale

        # Masked position tidak boleh mengirim gradient
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores = np.where(causal_mask[None, None, :, :], 0.0, dscores)

        # ================================================================
        # Soft-cap backward pada scores
        # ================================================================
        # scores sebelum softcap: (scores / scale) = Qh_rot @ Kh_rot.T
        # Kita perlu dscores sebelum softcap
        dscores_pre_cap = _softcap_backward(dscores, scores * (1.0 / self.scale) * self.scale, self.attention_logit_cap)

        # ================================================================
        # Gradient terhadap Q_rot dan K_rot
        # ================================================================
        dQh_rot = dscores_pre_cap @ Kh_rot
        dKh_rot = dscores_pre_cap.transpose(0, 1, 3, 2) @ Qh_rot

        # ================================================================
        # RoPE backward
        # ================================================================
        dQh, dKh = _rotary_backward(dQh_rot, dKh_rot, cos, sin)

        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)

        dQ = _sanitize(dQ)
        dK = _sanitize(dK)
        dV = _sanitize(dV)

        # ================================================================
        # Projection backward untuk Q, K, V
        # ================================================================
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

        # GPT-2 style scaled init untuk residual sublayers
        residual_scale = 1.0 / math.sqrt(2.0 * num_layers)

        self.gate_proj = Parameter(
            _init_weight(rng, (self.ffn_hidden_dim, self.hidden_size))
        )
        self.up_proj = Parameter(
            _init_weight(rng, (self.ffn_hidden_dim, self.hidden_size))
        )
        # down_proj menggunakan scaled init
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

        # Down projection
        dout_2d = dout.reshape(-1, D)
        h_2d = h_drop.reshape(-1, F)

        self.down_proj.grad += dout_2d.T @ h_2d
        self.down_proj.grad = _sanitize(self.down_proj.grad)
        dh_drop = dout @ self.down_proj.data
        dh_drop = _sanitize(dh_drop)

        # Dropout backward
        dh = _dropout_backward(dh_drop, dropout_mask)

        # SwiGLU backward
        dsilu = dh * up
        dup = dh * silu

        dgate = dsilu * sig * (1.0 + gate * (1.0 - sig))
        dgate = _sanitize(dgate)
        dup = _sanitize(dup)

        # Gate dan Up projection
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
        residual_1 = x  # Pure residual - NO sanitization

        h = self.input_norm.forward(x)
        h = self.attn.forward(h, training, cos, sin)
        h, attn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)

        # ================================================================
        # Stochastic Depth (DropPath) pada attention branch
        # ================================================================
        h, attn_dropped = _drop_path(h, self.drop_path_rate, training, self.rng)

        # Residual connection (pure, no sanitization)
        x_1 = residual_1 + h

        residual_2 = x_1  # Pure residual - NO sanitization

        h = self.post_attention_norm.forward(x_1)
        h = self.ffn.forward(h, training)
        h, ffn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)

        # ================================================================
        # Stochastic Depth (DropPath) pada FFN branch
        # ================================================================
        h, ffn_dropped = _drop_path(h, self.drop_path_rate, training, self.rng)

        # Residual connection (pure, no sanitization)
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

        # ================================================================
        # Branch FFN + residual kedua
        # ================================================================
        # Gradient dari residual = dout (pure flow)
        d_ffn_out = _dropout_backward(dout, ffn_dropout_mask)
        d_ffn_out = _drop_path_backward(d_ffn_out, ffn_dropped, self.drop_path_rate)

        d_ffn_in = self.ffn.backward(d_ffn_out)
        d_post_norm = self.post_attention_norm.backward(d_ffn_in)

        # Gradient ke x_1 = dout (residual) + d_post_norm (FFN branch)
        d_x_1 = dout + d_post_norm
        d_x_1 = _sanitize(d_x_1)

        # ================================================================
        # Branch Attention + residual pertama
        # ================================================================
        d_attn_out = _dropout_backward(d_x_1, attn_dropout_mask)
        d_attn_out = _drop_path_backward(d_attn_out, attn_dropped, self.drop_path_rate)

        d_attn_in = self.attn.backward(d_attn_out)
        d_input_norm = self.input_norm.backward(d_attn_in)

        # Gradient ke x = d_x_1 (residual) + d_input_norm (attention branch)
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
# CUSTOM TRANSFORMER LANGUAGE MODEL dengan Weight Tying
# ============================================================================

class CustomTransformerLM:
    """
    Model Transformer custom untuk language modeling.

    Fitur SOTA:
    - RoPE untuk positional encoding
    - Weight Tying antara embed_tokens dan lm_head
    - Logit Soft-Capping pada final logits
    - Stochastic Depth pada setiap layer
    - Scaled Initialization untuk residual sublayers
    """

    MAX_PARAMS: Optional[int] = 50_000_000

    def __init__(self, config: Config = CONFIG):
        self.config = config

        self.vocab_size = int(config.model.vocab_size)
        self.embedding_dim = int(config.model.embedding_dim)
        self.num_heads = int(config.model.num_attention_heads)
        self.num_layers = int(config.model.num_layers)
        self.ffn_hidden_dim = int(config.model.ffn_hidden_dim)
        self.max_position_embeddings = int(config.model.max_position_embeddings)
        self.layer_norm_eps = float(config.model.layer_norm_eps)

        # SOTA config
        self.use_weight_tying = bool(getattr(config.model, 'use_weight_tying', True))
        self.use_rope = bool(getattr(config.model, 'use_rope', True))
        self.final_logit_cap = float(getattr(config.model, 'final_logit_cap', 50.0))
        self.drop_path_rate = float(getattr(config.model, 'drop_path_rate', 0.1))

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
            # Linear ramp dari 0 di layer pertama ke drop_path_rate di layer terakhir
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
        # LM Head dengan Weight Tying
        # ================================================================
        if self.use_weight_tying:
            # Share reference - lm_head adalah embed_tokens yang sama
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = Parameter(
                _init_weight(self.rng, (self.vocab_size, self.embedding_dim))
            )

        # ================================================================
        # RoPE cache (hanya dibangun jika RoPE aktif)
        # ================================================================
        if self.use_rope:
            self.rope_cos, self.rope_sin = _build_rope_cache(
                self.max_position_embeddings,
                self.embedding_dim // self.num_heads,
            )
        else:
            # Fallback: sinusoidal positional encoding (legacy mode)
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

        # Dengan weight tying, lm_head tidak dihitung terpisah
        embedding_params = V * D
        lm_head_params = 0 if self.use_weight_tying else V * D

        estimated_params = (
            embedding_params
            + L * (
                2 * D
                + 4 * D * D
                + 3 * D * F
            )
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

    def _collect_parameters(self) -> Dict[str, Parameter]:
        params: Dict[str, Parameter] = {}

        params["embed_tokens.weight"] = self.embed_tokens

        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            params.update(layer.parameters(prefix))

        params.update(self.final_norm.parameters("norm."))

        # ================================================================
        # Weight Tying: lm_head TIDAK ditambahkan jika sharing
        # ================================================================
        if not self.use_weight_tying:
            params["lm_head.weight"] = self.lm_head

        return params

    def parameters(self) -> Dict[str, Parameter]:
        return dict(self._parameters)

    def parameter_count(self) -> int:
        return sum(p.data.size for p in self._parameters.values())

    def zero_grad(self) -> None:
        for param in self._parameters.values():
            param.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        state = {
            name: param.data.copy()
            for name, param in self._parameters.items()
        }

        # ================================================================
        # Weight Tying: simpan lm_head sebagai reference ke embed_tokens
        # ================================================================
        if self.use_weight_tying:
            state["lm_head.weight"] = self.embed_tokens.data.copy()

        return state

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        for name, array in state_dict.items():
            if name not in self._parameters:
                # Weight tying: lm_head.weight mungkin ada di state_dict tapi
                # tidak di _parameters karena sharing
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

                raise KeyError(f"Parameter tidak dikenal: {name}")

            param = self._parameters[name]
            arr = np.asarray(array, dtype=np.float32)

            if arr.shape != param.data.shape:
                raise ValueError(
                    f"Shape tidak cocok untuk {name}. "
                    f"Diharapkan {param.data.shape}, diterima {arr.shape}."
                )

            param.data = arr
            param.grad = np.zeros_like(arr, dtype=np.float32)

    def forward(
        self,
        input_ids: np.ndarray,
        training: bool = False,
    ) -> np.ndarray:
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

        x = self.embed_tokens.data[ids]
        x = _sanitize(x)

        # ================================================================
        # Positional encoding: RoPE atau sinusoidal (fallback)
        # ================================================================
        if self.use_rope:
            # RoPE diterapkan di setiap attention layer, bukan di embedding
            pass
        else:
            # Legacy sinusoidal PE
            x = x + self.pos_encoding[:T][None, :, :]
            x = _sanitize(x)

        # Embedding dropout
        dropout_rate = getattr(self.config.training, 'dropout_rate', None)
        if dropout_rate is None:
            dropout_rate = self.config.model.dropout_rate

        x, embed_dropout_mask = _dropout(
            x,
            float(dropout_rate),
            training,
            self.rng,
        )

        # ================================================================
        # Forward melalui semua layer
        # ================================================================
        for layer in self.layers:
            if self.use_rope:
                x = layer.forward(x, training, self.rope_cos, self.rope_sin)
            else:
                # Fallback: pass dummy cos/sin (tidak dipakai)
                dummy_cos = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                dummy_sin = np.zeros((T, self.embedding_dim // self.num_heads), dtype=np.float32)
                x = layer.forward(x, training, dummy_cos, dummy_sin)

        final_hidden = self.final_norm.forward(x)

        # ================================================================
        # LM Head dengan Weight Tying
        # ================================================================
        logits = final_hidden @ self.embed_tokens.data.T

        # ================================================================
        # Final logit soft-capping
        # ================================================================
        logits = _softcap(logits, self.final_logit_cap)
        logits = _sanitize(logits)

        self._forward_cache = (
            ids,
            embed_dropout_mask,
            final_hidden,
        )

        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        if self._forward_cache is None:
            raise RuntimeError(
                "CustomTransformerLM.backward dipanggil sebelum forward."
            )

        ids, embed_dropout_mask, final_hidden = self._forward_cache

        grad_logits = np.asarray(grad_logits, dtype=np.float32)
        grad_logits = _sanitize(grad_logits)

        B, T, V = grad_logits.shape
        D = final_hidden.shape[-1]

        # ================================================================
        # Soft-cap backward pada logits
        # ================================================================
        # logits = cap * tanh(final_hidden @ embed_tokens.T / cap)
        # Kita perlu gradient terhadap pre-cap logits
        pre_cap_logits = final_hidden @ self.embed_tokens.data.T
        grad_logits = _softcap_backward(grad_logits, pre_cap_logits, self.final_logit_cap)

        grad_logits_2d = grad_logits.reshape(-1, V)
        final_hidden_2d = final_hidden.reshape(-1, D)

        # ================================================================
        # Gradient untuk lm_head (yang sama dengan embed_tokens jika weight tying)
        # ================================================================
        # grad_logits = grad @ (embed_tokens.T)
        # d(embed_tokens) dari logits = grad.T @ final_hidden
        lm_grad = grad_logits_2d.T @ final_hidden_2d

        if self.use_weight_tying:
            # lm_head = embed_tokens, accumulate gradient langsung ke embed_tokens
            self.embed_tokens.grad += lm_grad
        else:
            self.lm_head.grad += lm_grad
            self.lm_head.grad = _sanitize(self.lm_head.grad)

        d_hidden = grad_logits @ self.embed_tokens.data
        d_hidden = _sanitize(d_hidden)

        d_hidden = self.final_norm.backward(d_hidden)

        for layer in reversed(self.layers):
            if self.use_rope:
                d_hidden = layer.backward(d_hidden)
            else:
                d_hidden = layer.backward(d_hidden)

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
        ),
    )

    model = CustomTransformerLM(small_config)

    batch_ids = np.array(
        [
            [2, 5, 7, 10, 3],
            [2, 8, 4, 6, 3],
        ],
        dtype=np.int64,
    )

    model.zero_grad()

    logits = model.forward(batch_ids, training=True)

    grad_logits = np.ones_like(logits, dtype=np.float32)

    model.backward(grad_logits)

    print("core/architecture.py (SOTA) test OK")
    print(f"Parameter count   : {model.parameter_count():,}")
    print(f"Logits shape      : {logits.shape}")
    print(f"Sample logits     : {logits[0, 0, :8]}")
    print(f"Weight tying      : {model.use_weight_tying}")
    print(f"RoPE              : {model.use_rope}")
    print(f"Final logit cap   : {model.final_logit_cap}")
    print(f"Drop path rate    : {model.drop_path_rate}")

    # Test sanitasi dengan input ekstrem
    print("\nTest sanitasi dengan input ekstrem...")
    extreme_ids = np.array([[2, 5, 7]], dtype=np.int64)
    extreme_logits = model.forward(extreme_ids, training=True)
    print(f"Logits finite: {np.all(np.isfinite(extreme_logits))}")

    # Test weight tying
    if model.use_weight_tying:
        print(f"\nWeight tying check: embed_tokens is lm_head: {model.embed_tokens is model.lm_head}")
