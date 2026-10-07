# core/architecture.py
#
# Arsitektur neural network custom untuk Training Pipeline Engine.
#
# Implementasi ini:
# - Tidak memakai framework ML siap pakai seperti PyTorch/TensorFlow.
# - Memakai NumPy hanya sebagai pustaka primitif numerik.
# - Menulis forward pass dan backward pass secara manual.
#
# Komponen:
# - Embedding
# - Sinusoidal Positional Encoding
# - RMSNorm
# - Multi-Head Self-Attention dengan causal mask
# - SwiGLU Feed-Forward Network
# - Dropout
# - LM Head
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
from typing import Dict, Optional

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
# Memastikan project root ada di sys.path, sehingga file di dalam folder core/
# tetap bisa meng-import config.py meskipun dijalankan langsung:
#   python core/architecture.py
# atau:
#   cd core && python architecture.py

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


# ============================================================================
# NUMERIC HELPERS
# ============================================================================

def _init_weight(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """
    Inisialisasi bobot dengan distribusi normal kecil.

    Standar inisialisasi:
    - std = min(0.02, 1 / sqrt(fan_in))
    """
    if len(shape) == 1:
        fan_in = shape[0]
    else:
        fan_in = shape[1]

    std = min(0.02, 1.0 / math.sqrt(max(1, fan_in)))
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax numerik stabil.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _dropout(
    x: np.ndarray,
    p: float,
    training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Inverted dropout.

    Return:
    - output
    - mask (berisi scaling 1 / keep_prob) atau None jika dropout tidak aktif
    """
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
    """
    Backward untuk dropout.
    """
    if mask is None:
        return dout
    return dout * mask


# ============================================================================
# PARAMETER
# ============================================================================

class Parameter:
    """
    Parameter sederhana berisi data bobot dan gradient.
    """

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
    """
    RMSNorm sederhana.

    Rumus:
        y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.eps = float(eps)
        self.cache: Optional[tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        ms = np.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(ms + self.eps)

        y = x * inv_rms * self.weight.data
        self.cache = (x, inv_rms)
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("RMSNorm.backward dipanggil sebelum forward.")

        x, inv_rms = self.cache
        D = x.shape[-1]

        # Gradient untuk weight
        axes = tuple(range(dout.ndim - 1))
        self.weight.grad += np.sum(dout * x * inv_rms, axis=axes)

        # Gradient untuk input
        dot = np.sum(dout * self.weight.data * x, axis=-1, keepdims=True)
        dx = (
            dout * self.weight.data * inv_rms
            - x * dot * (inv_rms ** 3) / D
        )

        self.cache = None
        return dx

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        return {
            f"{prefix}weight": self.weight,
        }


# ============================================================================
# MULTI-HEAD SELF-ATTENTION
# ============================================================================

class MultiHeadSelfAttention:
    """
    Multi-head self-attention dengan causal mask.

    Rumus inti:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, config: Config, rng: np.random.Generator):
        self.hidden_size = int(config.model.embedding_dim)
        self.num_heads = int(config.model.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = float(config.model.dropout_rate)
        self.rng = rng

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "embedding_dim harus habis dibagi num_attention_heads."
            )

        self.q_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.k_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.v_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))
        self.o_proj = Parameter(_init_weight(rng, (self.hidden_size, self.hidden_size)))

        self.cache: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        B, T, D = x.shape

        Q = x @ self.q_proj.data.T
        K = x @ self.k_proj.data.T
        V = x @ self.v_proj.data.T

        Qh = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Kh = K.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # scores: [B, H, T, T]
        scores = Qh @ Kh.transpose(0, 1, 3, 2)
        scores = scores * self.scale

        # Causal mask: token tidak boleh melihat masa depan
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

        probs = _softmax(scores, axis=-1)

        probs_drop, attn_dropout_mask = _dropout(
            probs,
            self.dropout_p,
            training,
            self.rng,
        )

        context = probs_drop @ Vh
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)

        output = context @ self.o_proj.data.T

        self.cache = (
            x,
            Qh,
            Kh,
            Vh,
            probs,
            probs_drop,
            attn_dropout_mask,
            context,
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
            probs,
            probs_drop,
            attn_dropout_mask,
            context,
        ) = self.cache

        B, T, D = x.shape

        # ------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------
        dout_2d = dout.reshape(-1, D)
        context_2d = context.reshape(-1, D)

        self.o_proj.grad += dout_2d.T @ context_2d
        dcontext = dout @ self.o_proj.data

        dcontext_heads = dcontext.reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # ------------------------------------------------------------
        # Gradient terhadap attention probabilities dan value
        # ------------------------------------------------------------
        dprobs_drop = dcontext_heads @ Vh.transpose(0, 1, 3, 2)
        dVh = probs_drop.transpose(0, 1, 3, 2) @ dcontext_heads

        # ------------------------------------------------------------
        # Dropout backward pada attention probabilities
        # ------------------------------------------------------------
        dprobs = _dropout_backward(dprobs_drop, attn_dropout_mask)

        # ------------------------------------------------------------
        # Softmax backward
        # ------------------------------------------------------------
        sum_dp = np.sum(dprobs * probs, axis=-1, keepdims=True)
        dscores = probs * (dprobs - sum_dp)

        # Forward memakai scale, maka gradient ke raw scores juga diskalakan
        dscores = dscores * self.scale

        # Masked position tidak boleh mengirim gradient
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        dscores = np.where(causal_mask[None, None, :, :], 0.0, dscores)

        # ------------------------------------------------------------
        # Gradient terhadap Q, K
        # ------------------------------------------------------------
        dQh = dscores @ Kh
        dKh = dscores.transpose(0, 1, 3, 2) @ Qh

        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)

        # ------------------------------------------------------------
        # Projection backward untuk Q, K, V
        # ------------------------------------------------------------
        x_2d = x.reshape(-1, D)

        self.q_proj.grad += dQ.reshape(-1, D).T @ x_2d
        self.k_proj.grad += dK.reshape(-1, D).T @ x_2d
        self.v_proj.grad += dV.reshape(-1, D).T @ x_2d

        dx_q = dQ @ self.q_proj.data
        dx_k = dK @ self.k_proj.data
        dx_v = dV @ self.v_proj.data

        dx = dx_q + dx_k + dx_v

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

    Struktur:
        gate = x @ W_gate.T
        up   = x @ W_up.T
        h    = silu(gate) * up
        out  = h @ W_down.T
    """

    def __init__(self, config: Config, rng: np.random.Generator):
        self.hidden_size = int(config.model.embedding_dim)
        self.ffn_hidden_dim = int(config.model.ffn_hidden_dim)
        self.dropout_p = float(config.model.dropout_rate)
        self.rng = rng

        self.gate_proj = Parameter(
            _init_weight(rng, (self.ffn_hidden_dim, self.hidden_size))
        )
        self.up_proj = Parameter(
            _init_weight(rng, (self.ffn_hidden_dim, self.hidden_size))
        )
        self.down_proj = Parameter(
            _init_weight(rng, (self.hidden_size, self.ffn_hidden_dim))
        )

        self.cache: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        gate = x @ self.gate_proj.data.T
        up = x @ self.up_proj.data.T

        sig = 1.0 / (1.0 + np.exp(-gate))
        silu = gate * sig

        h = silu * up

        h_drop, dropout_mask = _dropout(
            h,
            self.dropout_p,
            training,
            self.rng,
        )

        out = h_drop @ self.down_proj.data.T

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

        D = x.shape[-1]
        F = gate.shape[-1]

        # ------------------------------------------------------------
        # Down projection
        # ------------------------------------------------------------
        dout_2d = dout.reshape(-1, D)
        h_2d = h_drop.reshape(-1, F)

        self.down_proj.grad += dout_2d.T @ h_2d
        dh_drop = dout @ self.down_proj.data

        # ------------------------------------------------------------
        # Dropout backward
        # ------------------------------------------------------------
        dh = _dropout_backward(dh_drop, dropout_mask)

        # ------------------------------------------------------------
        # SwiGLU backward
        # ------------------------------------------------------------
        dsilu = dh * up
        dup = dh * silu

        # d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        dgate = dsilu * sig * (1.0 + gate * (1.0 - sig))

        # ------------------------------------------------------------
        # Gate dan Up projection
        # ------------------------------------------------------------
        x_2d = x.reshape(-1, D)

        self.gate_proj.grad += dgate.reshape(-1, F).T @ x_2d
        self.up_proj.grad += dup.reshape(-1, F).T @ x_2d

        dx_gate = dgate @ self.gate_proj.data
        dx_up = dup @ self.up_proj.data

        dx = dx_gate + dx_up

        self.cache = None
        return dx

    def parameters(self, prefix: str) -> Dict[str, Parameter]:
        return {
            f"{prefix}gate_proj.weight": self.gate_proj,
            f"{prefix}up_proj.weight": self.up_proj,
            f"{prefix}down_proj.weight": self.down_proj,
        }


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock:
    """
    Satu blok Transformer:

        x = x + Dropout(Attention(RMSNorm(x)))
        x = x + Dropout(FFN(RMSNorm(x)))
    """

    def __init__(self, config: Config, rng: np.random.Generator):
        self.config = config
        self.dropout_p = float(config.model.dropout_rate)
        self.rng = rng

        self.input_norm = RMSNorm(
            int(config.model.embedding_dim),
            float(config.model.layer_norm_eps),
        )
        self.attn = MultiHeadSelfAttention(config, rng)
        self.post_attention_norm = RMSNorm(
            int(config.model.embedding_dim),
            float(config.model.layer_norm_eps),
        )
        self.ffn = SwiGLUFFN(config, rng)

        self.cache: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool) -> np.ndarray:
        residual_1 = x

        h = self.input_norm.forward(x)
        h = self.attn.forward(h, training)
        h, attn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)

        x_1 = residual_1 + h

        residual_2 = x_1

        h = self.post_attention_norm.forward(x_1)
        h = self.ffn.forward(h, training)
        h, ffn_dropout_mask = _dropout(h, self.dropout_p, training, self.rng)

        out = residual_2 + h

        self.cache = (
            attn_dropout_mask,
            ffn_dropout_mask,
        )

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.cache is None:
            raise RuntimeError("TransformerBlock.backward dipanggil sebelum forward.")

        attn_dropout_mask, ffn_dropout_mask = self.cache

        # ------------------------------------------------------------
        # Branch FFN + residual kedua
        # ------------------------------------------------------------
        d_ffn_out = _dropout_backward(dout, ffn_dropout_mask)
        d_ffn_in = self.ffn.backward(d_ffn_out)
        d_post_norm = self.post_attention_norm.backward(d_ffn_in)

        # Gradient terhadap x_1 datang dari residual langsung + branch FFN
        d_x_1 = dout + d_post_norm

        # ------------------------------------------------------------
        # Branch Attention + residual pertama
        # ------------------------------------------------------------
        d_attn_out = _dropout_backward(d_x_1, attn_dropout_mask)
        d_attn_in = self.attn.backward(d_attn_out)
        d_input_norm = self.input_norm.backward(d_attn_in)

        # Gradient terhadap residual pertama datang dari residual langsung
        # + branch attention
        dx = d_x_1 + d_input_norm

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
# CUSTOM TRANSFORMER LANGUAGE MODEL
# ============================================================================

class CustomTransformerLM:
    """
    Model Transformer custom untuk language modeling.

    Struktur state_dict:
    - embed_tokens.weight
    - layers.{i}.input_layernorm.weight
    - layers.{i}.self_attn.q_proj.weight
    - layers.{i}.self_attn.k_proj.weight
    - layers.{i}.self_attn.v_proj.weight
    - layers.{i}.self_attn.o_proj.weight
    - layers.{i}.post_attention_layernorm.weight
    - layers.{i}.mlp.gate_proj.weight
    - layers.{i}.mlp.up_proj.weight
    - layers.{i}.mlp.down_proj.weight
    - norm.weight
    - lm_head.weight
    """

    # Batas aman untuk implementasi pure-custom berbasis NumPy.
    # Set None jika ingin menonaktifkan pemeriksaan ini.
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

        self._validate_practical_size()

        self.rng = np.random.default_rng(int(config.training.seed))

        self.embed_tokens = Parameter(
            _init_weight(self.rng, (self.vocab_size, self.embedding_dim))
        )

        self.layers = [
            TransformerBlock(config, self.rng)
            for _ in range(self.num_layers)
        ]

        self.final_norm = RMSNorm(self.embedding_dim, self.layer_norm_eps)

        self.lm_head = Parameter(
            _init_weight(self.rng, (self.vocab_size, self.embedding_dim))
        )

        self.pos_encoding = self._build_sinusoidal_positional_encoding(
            self.max_position_embeddings,
            self.embedding_dim,
        )

        self._parameters = self._collect_parameters()
        self._forward_cache: Optional[tuple] = None

    # ========================================================================
    # INIT / VALIDATION
    # ========================================================================

    def _validate_practical_size(self) -> None:
        """
        Implementasi custom ini ditujukan untuk eksperimen dan pembelajaran.
        Untuk konfigurasi sangat besar, sebaiknya gunakan backend GPU/framework
        numerik yang lebih serius.
        """
        if self.MAX_PARAMS is None:
            return

        V = self.vocab_size
        D = self.embedding_dim
        L = self.num_layers
        F = self.ffn_hidden_dim

        estimated_params = (
            V * D
            + L * (
                2 * D
                + 4 * D * D
                + 3 * D * F
            )
            + D
            + V * D
        )

        if estimated_params > self.MAX_PARAMS:
            raise RuntimeError(
                "Estimasi jumlah parameter terlalu besar untuk engine custom "
                "berbasis NumPy ini. "
                f"Estimasi: {estimated_params:,} parameter. "
                f"Batas aman: {self.MAX_PARAMS:,} parameter. "
                "Kurangi vocab_size, embedding_dim, num_layers, atau ffn_hidden_dim "
                "di config.py, atau set CustomTransformerLM.MAX_PARAMS = None "
                "untuk menonaktifkan pemeriksaan ini."
            )

    def _build_sinusoidal_positional_encoding(
        self,
        max_position: int,
        dim: int,
    ) -> np.ndarray:
        """
        Positional Encoding sinusoidal.

        Rumus:
            PE(pos, 2i)   = sin(pos / 10000^(2i/d))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        """
        pe = np.zeros((max_position, dim), dtype=np.float32)
        position = np.arange(max_position, dtype=np.float32)[:, None]

        freq_indices = np.arange(0, dim, 2, dtype=np.float32)
        div_term = np.exp(freq_indices * -(math.log(10000.0) / dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: dim // 2])

        return pe

    # ========================================================================
    # PARAMETER MANAGEMENT
    # ========================================================================

    def _collect_parameters(self) -> Dict[str, Parameter]:
        params: Dict[str, Parameter] = {}

        params["embed_tokens.weight"] = self.embed_tokens

        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            params.update(layer.parameters(prefix))

        params.update(self.final_norm.parameters("norm."))
        params["lm_head.weight"] = self.lm_head

        return params

    def parameters(self) -> Dict[str, Parameter]:
        """
        Mengembalikan dictionary nama parameter -> objek Parameter.
        """
        return dict(self._parameters)

    def parameter_count(self) -> int:
        """
        Menghitung jumlah total parameter.
        """
        return sum(p.data.size for p in self._parameters.values())

    def zero_grad(self) -> None:
        """
        Mengosongkan seluruh gradient.
        Dipanggil sebelum satu langkah training.
        """
        for param in self._parameters.values():
            param.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Mengembalikan salinan seluruh bobot model.
        Dipakai oleh model_exporter.py dan weight_manager.py.
        """
        return {
            name: param.data.copy()
            for name, param in self._parameters.items()
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Memuat bobot dari state_dict.
        """
        for name, array in state_dict.items():
            if name not in self._parameters:
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

    # ========================================================================
    # FORWARD
    # ========================================================================

    def forward(
        self,
        input_ids: np.ndarray,
        training: bool = False,
    ) -> np.ndarray:
        """
        Forward pass.

        Input:
        - input_ids: [B, T] atau [T]

        Output:
        - logits: [B, T, vocab_size]
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

        # Token ID di luar vocab dipetakan ke [UNK]
        unk_id = int(self.config.special_tokens.unk_id)
        ids = np.where((ids < 0) | (ids >= self.vocab_size), unk_id, ids)

        # Embedding
        x = self.embed_tokens.data[ids]  # [B, T, D]

        # Positional encoding
        x = x + self.pos_encoding[:T][None, :, :]

        # Dropout embedding
        x, embed_dropout_mask = _dropout(
            x,
            float(self.config.model.dropout_rate),
            training,
            self.rng,
        )

        # Transformer blocks
        for layer in self.layers:
            x = layer.forward(x, training)

        # Final norm
        final_hidden = self.final_norm.forward(x)

        # LM head
        logits = final_hidden @ self.lm_head.data.T

        self._forward_cache = (
            ids,
            embed_dropout_mask,
            final_hidden,
        )

        return logits

    # ========================================================================
    # BACKWARD
    # ========================================================================

    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backward pass dari gradient logits.

        Input:
        - grad_logits: [B, T, vocab_size]

        Method ini mengisi .grad pada seluruh Parameter.
        """
        if self._forward_cache is None:
            raise RuntimeError(
                "CustomTransformerLM.backward dipanggil sebelum forward."
            )

        ids, embed_dropout_mask, final_hidden = self._forward_cache

        grad_logits = np.asarray(grad_logits, dtype=np.float32)

        B, T, V = grad_logits.shape
        D = final_hidden.shape[-1]

        # ------------------------------------------------------------
        # LM head backward
        # ------------------------------------------------------------
        grad_logits_2d = grad_logits.reshape(-1, V)
        final_hidden_2d = final_hidden.reshape(-1, D)

        self.lm_head.grad += grad_logits_2d.T @ final_hidden_2d
        d_hidden = grad_logits @ self.lm_head.data

        # ------------------------------------------------------------
        # Final norm backward
        # ------------------------------------------------------------
        d_hidden = self.final_norm.backward(d_hidden)

        # ------------------------------------------------------------
        # Transformer blocks backward
        # ------------------------------------------------------------
        for layer in reversed(self.layers):
            d_hidden = layer.backward(d_hidden)

        # ------------------------------------------------------------
        # Embedding dropout backward
        # ------------------------------------------------------------
        d_hidden = _dropout_backward(d_hidden, embed_dropout_mask)

        # ------------------------------------------------------------
        # Embedding backward
        # ------------------------------------------------------------
        d_embed = d_hidden
        np.add.at(
            self.embed_tokens.grad,
            ids.reshape(-1),
            d_embed.reshape(-1, D),
        )

        self._forward_cache = None


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================

if __name__ == "__main__":
    # Uji arsitektur dengan konfigurasi kecil agar cepat dijalankan.
    # Ini tidak mengubah config.py utama.

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
            weight_decay=0.01,
            grad_clip_norm=1.0,
            seed=1337,
        ),
        model=ModelConfig(
            vocab_size=64,
            embedding_dim=32,
            num_attention_heads=4,
            num_layers=1,
            dropout_rate=0.0,
            max_position_embeddings=32,
            layer_norm_eps=1e-5,
            ffn_hidden_dim=48,
            train_dtype="float32",
            export_dtype="float32",
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

    # Gradient dummy untuk menguji backward pass.
    # Nantinya gradient asli berasal dari evaluator.py (Cross-Entropy Loss).
    grad_logits = np.ones_like(logits, dtype=np.float32)

    model.backward(grad_logits)

    print("core/architecture.py test OK")
    print(f"Parameter count : {model.parameter_count():,}")
    print(f"Logits shape    : {logits.shape}")
    print(f"Sample logits   : {logits[0, 0, :8]}")
