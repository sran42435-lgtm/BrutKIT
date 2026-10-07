# core/model_exporter.py
#
# Model Exporter / Generator file model biner.
#
# Tugas utama:
# 1. Mengambil seluruh state_dict model.
# 2. Melakukan casting / quantization presisi tensor.
# 3. Menyusun header metadata.
# 4. Menulis file model biner tunggal (.safetensors-like).
# 5. Menyertakan vocab/tokenizer di dalam metadata.
# 6. Menyediakan fungsi load kembali.
#
# Implementasi:
# - Tidak memakai pustaka safetensors eksternal.
# - Menulis format safetensors-like secara manual.
# - Memakai NumPy sebagai pustaka primitif numerik.
#
# Keterhubungan:
# - config.py            : path output model dan konfigurasi arsitektur
# - core/architecture.py : model yang akan di-export
# - core/tokenizer.py    : vocab/tokenizer untuk disertakan ke file model

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# ============================================================================
# FIX IMPORT PATH
# ============================================================================
# Memastikan project root ada di sys.path, sehingga file di dalam folder core/
# tetap bisa meng-import config.py meskipun dijalankan langsung:
#   python core/model_exporter.py
# atau:
#   cd core && python model_exporter.py

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
        "core/model_exporter.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config


# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def _float32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """
    Konversi float32 -> bfloat16 sebagai raw bytes.

    bfloat16 memakai 16-bit tertinggi dari float32.
    """
    arr32 = np.ascontiguousarray(arr, dtype=np.dtype("<f4"))
    u32 = arr32.view(np.dtype("<u4"))
    u16 = (u32 >> 16).astype(np.dtype("<u2"))
    return u16.tobytes()


def _bf16_bytes_to_float32(raw: bytes) -> np.ndarray:
    """
    Konversi raw bfloat16 -> float32.
    """
    u16 = np.frombuffer(raw, dtype=np.dtype("<u2"))
    u32 = u16.astype(np.dtype("<u4")) << 16
    f32 = u32.view(np.dtype("<f4"))
    return f32.astype(np.float32)


def _quantize_int8(arr: np.ndarray) -> Tuple[bytes, float]:
    """
    Quantization simetris float32 -> int8.

    q = round(x / scale)
    x ≈ q * scale
    """
    if arr.size == 0:
        return b"", 1.0

    amax = float(np.max(np.abs(arr)))

    if amax <= 0.0 or not np.isfinite(amax):
        scale = 1.0
    else:
        scale = amax / 127.0

    q = np.round(arr / scale).clip(-127, 127).astype(np.int8)
    return q.tobytes(), float(scale)


def _quantize_int4(arr: np.ndarray) -> Tuple[bytes, float]:
    """
    Quantization simetris float32 -> int4.

    Nilai int4 disimpan dalam rentang [-8, 7].
    Dua nilai int4 dikemas dalam satu byte.
    """
    if arr.size == 0:
        return b"", 1.0

    amax = float(np.max(np.abs(arr)))

    if amax <= 0.0 or not np.isfinite(amax):
        scale = 1.0
    else:
        scale = amax / 7.0

    q = np.round(arr / scale).clip(-8, 7).astype(np.int8).ravel()

    if q.size == 0:
        return b"", float(scale)

    # Padding jika jumlah elemen ganjil
    if q.size % 2 == 1:
        q = np.pad(q, (0, 1), mode="constant", constant_values=0)

    low = (q[0::2] & 0x0F).astype(np.uint8)
    high = (q[1::2] & 0x0F).astype(np.uint8)

    packed = low | (high << 4)

    return packed.tobytes(), float(scale)


def _unpack_int4(
    raw: bytes,
    shape: Tuple[int, ...],
    scale: float,
) -> np.ndarray:
    """
    Membuka kembali int4 packed menjadi float32.
    """
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


def _serialize_tensor(
    arr: np.ndarray,
    export_dtype: str,
    quant_info: Dict[str, dict],
    tensor_name: str,
) -> Tuple[bytes, str, list]:
    """
    Serialisasi satu tensor menjadi raw bytes.

    Return:
    - raw bytes
    - dtype string
    - shape list
    """
    arr = np.asarray(arr, dtype=np.float32)
    shape = list(arr.shape)

    if export_dtype == "float32":
        raw = np.ascontiguousarray(arr, dtype=np.dtype("<f4")).tobytes()
        return raw, "F32", shape

    if export_dtype == "float16":
        raw = np.ascontiguousarray(arr, dtype=np.dtype("<f2")).tobytes()
        return raw, "F16", shape

    if export_dtype == "bfloat16":
        raw = _float32_to_bf16_bytes(arr)
        return raw, "BF16", shape

    if export_dtype == "int8":
        raw, scale = _quantize_int8(arr)
        quant_info[tensor_name] = {
            "scale": scale,
            "original_dtype": "float32",
        }
        return raw, "I8", shape

    if export_dtype == "int4":
        raw, scale = _quantize_int4(arr)
        quant_info[tensor_name] = {
            "scale": scale,
            "original_dtype": "float32",
        }
        return raw, "I4", shape

    # Fallback aman
    raw = np.ascontiguousarray(arr, dtype=np.dtype("<f2")).tobytes()
    return raw, "F16", shape


def _build_tokenizer_payload(tokenizer, config: Config) -> Optional[dict]:
    """
    Mengambil payload tokenizer/vocab untuk dimasukkan ke metadata file model.
    """
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "is_ready") and not tokenizer.is_ready:
                tokenizer.load()
        except Exception:
            pass

        if hasattr(tokenizer, "token_to_id") and hasattr(tokenizer, "merges"):
            return {
                "version": "1.0",
                "model": "custom-bpe",
                "special_tokens": {
                    "pad": config.special_tokens.pad,
                    "unk": config.special_tokens.unk,
                    "bos": config.special_tokens.bos,
                    "eos": config.special_tokens.eos,
                },
                "token_to_id": dict(tokenizer.token_to_id),
                "merges": [
                    [a, b] for a, b in getattr(tokenizer, "merges", [])
                ],
            }

    vocab_path = Path(config.paths.vocab_path)

    if vocab_path.exists():
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


# ============================================================================
# MAIN EXPORT FUNCTION
# ============================================================================

def export_to_safetensors(
    model,
    tokenizer=None,
    config: Config = CONFIG,
    output_filepath: Optional[Path] = None,
    export_dtype: Optional[str] = None,
    include_vocab: bool = True,
) -> Path:
    """
    Mengemas seluruh bobot model menjadi satu file biner tunggal.

    Struktur file:
        [8 byte header length]
        [JSON header]
        [raw tensor bytes]
    """
    if output_filepath is None:
        output_filepath = Path(config.paths.final_model_path)
    else:
        output_filepath = Path(output_filepath)

    if output_filepath.suffix == "":
        output_filepath = output_filepath.with_suffix(".safetensors")

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    if export_dtype is None:
        export_dtype = str(config.model.export_dtype)

    export_dtype = export_dtype.lower()

    state_dict = model.state_dict()

    tensor_entries: Dict[str, dict] = {}
    tensor_bytes = bytearray()
    offset = 0
    quant_info: Dict[str, dict] = {}

    for name, arr in state_dict.items():
        raw, dtype_str, shape = _serialize_tensor(
            arr=arr,
            export_dtype=export_dtype,
            quant_info=quant_info,
            tensor_name=name,
        )

        tensor_entries[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }

        tensor_bytes.extend(raw)
        offset += len(raw)

    tensor_sha256 = hashlib.sha256(bytes(tensor_bytes)).hexdigest()

    metadata: Dict[str, str] = {
        "format": "pt",
        "producer": "CreateMiniBot-custom-exporter",
        "architecture": "CustomTransformerLM",
        "hidden_size": str(config.model.embedding_dim),
        "num_attention_heads": str(config.model.num_attention_heads),
        "num_layers": str(config.model.num_layers),
        "vocab_size": str(config.model.vocab_size),
        "context_length": str(config.model.max_position_embeddings),
        "export_dtype": export_dtype,
        "tensor_sha256": tensor_sha256,
    }

    if include_vocab:
        tokenizer_payload = _build_tokenizer_payload(tokenizer, config)

        if tokenizer_payload is not None:
            metadata["tokenizer"] = json.dumps(
                tokenizer_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    if quant_info:
        metadata["quantization"] = json.dumps(
            quant_info,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    header = dict(tensor_entries)
    header["__metadata__"] = metadata

    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    with open(output_filepath, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(tensor_bytes)

    return output_filepath


# ============================================================================
# LOAD FUNCTION
# ============================================================================

def load_safetensors(
    filepath: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Memuat file model biner .safetensors-like.

    Return:
    - state_dict
    - metadata
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        header_len_bytes = f.read(8)

        if len(header_len_bytes) < 8:
            raise ValueError("File model tidak valid: header length terlalu pendek.")

        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        tensor_bytes = f.read()

    header = json.loads(header_bytes.decode("utf-8"))
    metadata = header.get("__metadata__", {})

    quant_raw = metadata.get("quantization", "{}")

    if isinstance(quant_raw, str):
        try:
            quant_meta = json.loads(quant_raw)
        except Exception:
            quant_meta = {}
    else:
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


def load_into_model(model, filepath: Path) -> Dict[str, str]:
    """
    Memuat state_dict dari file model ke dalam model.
    """
    state_dict, metadata = load_safetensors(filepath)
    model.load_state_dict(state_dict)
    return metadata


# ============================================================================
# EXPORTER CLASS
# ============================================================================

class ModelExporter:
    """
    Wrapper exporter agar mudah dipanggil dari main.py.
    """

    def __init__(self, config: Config = CONFIG):
        self.config = config
        self.default_output_path = Path(config.paths.final_model_path)

    def save(
        self,
        model,
        tokenizer=None,
        output_path: Optional[Path] = None,
        export_dtype: Optional[str] = None,
        include_vocab: bool = True,
    ) -> Path:
        """
        Simpan model ke file biner tunggal.
        """
        return export_to_safetensors(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            output_filepath=output_path,
            export_dtype=export_dtype,
            include_vocab=include_vocab,
        )

    # Alias agar kompatibel dengan penamaan:
    #   model_exporter.export(...)
    export = save


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
                export_dtype="float16",
            ),
        )

        tokenizer = Tokenizer(small_config)
        tokenizer.train()

        model = CustomTransformerLM(small_config)

        exporter = ModelExporter(small_config)

        model_path = exporter.save(
            model=model,
            tokenizer=tokenizer,
            include_vocab=True,
        )

        file_size = model_path.stat().st_size

        print("core/model_exporter.py test OK")
        print(f"Model file      : {model_path}")
        print(f"File size       : {file_size:,} bytes")

        # Load kembali
        state_dict, metadata = load_safetensors(model_path)

        print(f"Tensors loaded  : {len(state_dict)}")
        print(f"Architecture    : {metadata.get('architecture')}")
        print(f"Export dtype    : {metadata.get('export_dtype')}")
        print(f"Tensor SHA-256  : {metadata.get('tensor_sha256')}")

        # Pastikan state_dict bisa dimasukkan ke model baru
        new_model = CustomTransformerLM(small_config)
        new_model.load_state_dict(state_dict)

        print("Model load_state_dict OK")
