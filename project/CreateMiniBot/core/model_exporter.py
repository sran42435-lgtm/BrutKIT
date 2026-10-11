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
# 7. Bundle MoE assets (keywords, semantic embeddings, paragraphs)
# 8. Copy engine_standalone.py ke output folder (untuk deployment)
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
# - core/router.py       : MoE assets yang akan di-bundle

from __future__ import annotations

import hashlib
import json
import shutil
import struct
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
        "core/model_exporter.py membutuhkan NumPy sebagai pustaka primitif numerik. "
        "Silakan pasang NumPy terlebih dahulu dengan: pip install numpy"
    ) from exc

from config import CONFIG, Config


# Prefix untuk tensor semantic cache (disimpan sebagai tensor khusus)
SEMANTIC_CACHE_TENSOR_PREFIX = "__semantic_embeddings__."


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
    
    Termasuk special token: pad, unk, bos, eos, endoftext.
    """
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "is_ready") and not tokenizer.is_ready:
                tokenizer.load()
        except Exception:
            pass

        if hasattr(tokenizer, "token_to_id") and hasattr(tokenizer, "merges"):
            # Ambil endoftext dengan fallback
            endoftext_token = getattr(tokenizer, "endoftext_token", None)
            if not endoftext_token:
                endoftext_token = getattr(config.special_tokens, "endoftext", "")
            if not endoftext_token:
                endoftext_token = '<' + '|' + 'endoftext' + '|' + '>'
            
            return {
                "version": "1.0",
                "model": "custom-bpe",
                "special_tokens": {
                    "pad": config.special_tokens.pad,
                    "unk": config.special_tokens.unk,
                    "bos": config.special_tokens.bos,
                    "eos": config.special_tokens.eos,
                    "endoftext": endoftext_token,
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
# MOE ASSETS LOADER
# ============================================================================

def _load_moe_assets(config: Config) -> Dict:
    """
    Load MoE assets dari output folder (hasil training).
    
    Assets yang dicari:
    - domain_keywords.json → keywords per domain
    - semantic_cache.npz → embeddings + paragraphs per domain
    
    Return:
    - dict dengan keys: keywords, paragraphs, embeddings
    """
    output_dir = Path(config.paths.output_models_dir)
    
    assets: Dict = {
        "keywords": {},
        "paragraphs": {},
        "embeddings": {},
    }
    
    # ================================================================
    # Load domain_keywords.json
    # ================================================================
    keywords_path = output_dir / "domain_keywords.json"
    if keywords_path.exists():
        try:
            with open(keywords_path, "r", encoding="utf-8") as f:
                kw_data = json.load(f)
            
            if isinstance(kw_data, dict):
                # Bisa struktur baru (dengan "domain_keywords" key) atau lama (langsung dict)
                if "domain_keywords" in kw_data:
                    assets["keywords"] = kw_data["domain_keywords"]
                else:
                    assets["keywords"] = kw_data
        except Exception as exc:
            print(f"[Exporter] Warning: Gagal load domain_keywords.json: {exc}")
    
    # ================================================================
    # Load semantic_cache.npz
    # ================================================================
    cache_path = output_dir / "semantic_cache.npz"
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                # Load paragraphs dari JSON
                if "__paragraphs_json__" in data.files:
                    try:
                        paragraphs_str = str(data["__paragraphs_json__"][0])
                        paragraphs_data = json.loads(paragraphs_str)
                        if isinstance(paragraphs_data, dict):
                            assets["paragraphs"] = paragraphs_data
                    except Exception:
                        pass
                
                # Load embeddings per domain
                for key in data.files:
                    if key == "__paragraphs_json__":
                        continue
                    if key.endswith("__embeddings"):
                        # Reconstruct domain name dari key
                        # Format: {safe_domain}__embeddings
                        domain = key[:-len("__embeddings")]
                        # Unsanitize domain name (ganti _ kembali ke aslinya)
                        assets["embeddings"][domain] = data[key]
        except Exception as exc:
            print(f"[Exporter] Warning: Gagal load semantic_cache.npz: {exc}")
    
    return assets


def _copy_engine_standalone(config: Config, output_dir: Path) -> Optional[Path]:
    """
    Copy engine_standalone.py ke output folder untuk deployment.
    
    Return:
    - Path ke file yang di-copy, atau None jika gagal
    """
    # Cari engine_standalone.py di core/
    engine_src = Path(__file__).resolve().parent / "engine_standalone.py"
    
    if not engine_src.exists():
        return None
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        engine_dst = output_dir / "otak_engine.py"
        shutil.copy2(engine_src, engine_dst)
        return engine_dst
    except Exception as exc:
        print(f"[Exporter] Warning: Gagal copy engine_standalone.py: {exc}")
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
    include_moe_assets: bool = True,
) -> Path:
    """
    Mengemas seluruh bobot model menjadi satu file biner tunggal.
    
    Jika include_moe_assets=True dan model menggunakan MoE:
    - Bundle domain keywords ke metadata
    - Bundle semantic embeddings sebagai tensor khusus
    - Bundle paragraphs sebagai JSON di metadata
    - Copy engine_standalone.py ke output folder

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

    # ================================================================
    # Serialize model weights
    # ================================================================
    for name, arr in state_dict.items():
        # Skip metadata internal keys
        if name == "__moe_metadata__":
            continue
        
        # Skip non-array entries
        if not isinstance(arr, np.ndarray):
            continue

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

    # ================================================================
    # Build metadata
    # ================================================================
    use_moe = bool(getattr(config.model, 'use_moe', False))
    
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
        # MoE flags
        "use_moe": "true" if use_moe else "false",
    }

    # Add tokenizer/vocab
    if include_vocab:
        tokenizer_payload = _build_tokenizer_payload(tokenizer, config)

        if tokenizer_payload is not None:
            metadata["tokenizer"] = json.dumps(
                tokenizer_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    # Add quantization info
    if quant_info:
        metadata["quantization"] = json.dumps(
            quant_info,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # ================================================================
    # Bundle MoE assets
    # ================================================================
    moe_domains: list = []
    
    if use_moe and include_moe_assets:
        moe_assets = _load_moe_assets(config)
        
        # Bundle keywords ke metadata
        if moe_assets["keywords"]:
            moe_domains = list(moe_assets["keywords"].keys())
            metadata["moe_domains"] = json.dumps(moe_domains)
            metadata["moe_keywords"] = json.dumps(
                moe_assets["keywords"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            print(f"[Exporter] Bundled keywords untuk {len(moe_domains)} domain: {moe_domains}")
        
        # Bundle paragraphs ke metadata
        if moe_assets["paragraphs"]:
            metadata["moe_paragraphs"] = json.dumps(
                moe_assets["paragraphs"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            total_paras = sum(len(p) for p in moe_assets["paragraphs"].values())
            print(f"[Exporter] Bundled {total_paras} paragraphs untuk semantic search")
        
        # Bundle embeddings sebagai tensor khusus
        for domain, emb in moe_assets.get("embeddings", {}).items():
            if not isinstance(emb, np.ndarray):
                continue
            
            tensor_name = f"{SEMANTIC_CACHE_TENSOR_PREFIX}{domain}"
            
            raw, dtype_str, shape = _serialize_tensor(
                arr=emb,
                export_dtype=export_dtype,
                quant_info=quant_info,
                tensor_name=tensor_name,
            )
            
            tensor_entries[tensor_name] = {
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [offset, offset + len(raw)],
            }
            
            tensor_bytes.extend(raw)
            offset += len(raw)
            
            print(f"[Exporter] Bundled semantic embeddings untuk domain {domain}: shape {emb.shape}")
        
        # Copy engine_standalone.py ke output folder
        engine_path = _copy_engine_standalone(config, output_filepath.parent)
        if engine_path:
            metadata["engine_standalone"] = engine_path.name
            print(f"[Exporter] Copied engine: {engine_path.name}")
    
    # Recompute SHA-256 setelah semua tensor (termasuk semantic cache) ditambahkan
    tensor_sha256 = hashlib.sha256(bytes(tensor_bytes)).hexdigest()
    metadata["tensor_sha256"] = tensor_sha256

    # ================================================================
    # Build header dan tulis file
    # ================================================================
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
    - state_dict (hanya bobot model, tanpa semantic cache tensors)
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


def extract_moe_assets(filepath: Path) -> Dict:
    """
    Extract MoE assets dari safetensors file.
    
    Berguna untuk deployment: load keywords, paragraphs, dan semantic
    embeddings tanpa harus load model.
    
    Return:
    - dict dengan keys: keywords, paragraphs, embeddings, domains, use_moe
    """
    filepath = Path(filepath)
    
    assets: Dict = {
        "keywords": {},
        "paragraphs": {},
        "embeddings": {},
        "domains": [],
        "use_moe": False,
    }
    
    # Load full file
    with open(filepath, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return assets
        
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
        tensor_bytes = f.read()
    
    header = json.loads(header_bytes.decode("utf-8"))
    metadata = header.get("__metadata__", {})
    
    # Check MoE flags
    assets["use_moe"] = metadata.get("use_moe", "false").lower() == "true"
    
    if not assets["use_moe"]:
        return assets
    
    # Extract keywords
    moe_keywords = metadata.get("moe_keywords")
    if moe_keywords:
        try:
            assets["keywords"] = json.loads(moe_keywords)
        except Exception:
            pass
    
    # Extract domains
    moe_domains = metadata.get("moe_domains")
    if moe_domains:
        try:
            assets["domains"] = json.loads(moe_domains)
        except Exception:
            pass
    
    # Extract paragraphs
    moe_paragraphs = metadata.get("moe_paragraphs")
    if moe_paragraphs:
        try:
            assets["paragraphs"] = json.loads(moe_paragraphs)
        except Exception:
            pass
    
    # Extract semantic embeddings dari tensors
    quant_raw = metadata.get("quantization", "{}")
    try:
        quant_meta = json.loads(quant_raw) if isinstance(quant_raw, str) else {}
    except Exception:
        quant_meta = {}
    
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not name.startswith(SEMANTIC_CACHE_TENSOR_PREFIX):
            continue
        
        domain = name[len(SEMANTIC_CACHE_TENSOR_PREFIX):]
        
        dtype = meta.get("dtype")
        shape = tuple(meta.get("shape", []))
        start, end = meta.get("data_offsets", [0, 0])
        raw = tensor_bytes[start:end]
        
        # Deserialize tensor
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
            continue
        
        assets["embeddings"][domain] = arr.reshape(shape).astype(np.float32, copy=False)
    
    return assets


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
        include_moe_assets: bool = True,
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
            include_moe_assets=include_moe_assets,
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
                    "[MOBIL] mobil memiliki mesin",
                    "[MOBIL] mobil memakai bahan bakar",
                    "[SAINS] air adalah H2O",
                    "[SAINS] fotosintesis menghasilkan oksigen",
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
                dropout_rate=0.0,
                weight_decay=0.01,
                grad_clip_norm=1.0,
                seed=1337,
            ),
            model=ModelConfig(
                vocab_size=64,
                embedding_dim=16,
                num_attention_heads=2,
                num_layers=1,
                max_position_embeddings=32,
                layer_norm_eps=1e-5,
                ffn_hidden_dim=32,
                train_dtype="float32",
                export_dtype="float16",
                use_moe=True,
            ),
        )

        tokenizer = Tokenizer(small_config)
        tokenizer.train()

        model = CustomTransformerLM(small_config)
        
        # Inisialisasi beberapa domain untuk test MoE
        model.init_domains(["MOBIL", "SAINS"])

        exporter = ModelExporter(small_config)

        # ================================================================
        # Test 1: Export tanpa MoE assets (default)
        # ================================================================
        print("=" * 60)
        print("Test 1: Export Model (tanpa MoE assets)")
        print("=" * 60)
        
        model_path = exporter.save(
            model=model,
            tokenizer=tokenizer,
            include_vocab=True,
            include_moe_assets=False,  # Skip MoE assets
        )

        file_size = model_path.stat().st_size

        print(f"Model file      : {model_path}")
        print(f"File size       : {file_size:,} bytes")

        state_dict, metadata = load_safetensors(model_path)

        print(f"Tensors loaded  : {len(state_dict)}")
        print(f"Architecture    : {metadata.get('architecture')}")
        print(f"Export dtype    : {metadata.get('export_dtype')}")
        print(f"Use MoE         : {metadata.get('use_moe')}")
        print(f"Tensor SHA-256  : {metadata.get('tensor_sha256')}")

        # Pastikan state_dict bisa dimasukkan ke model baru
        new_model = CustomTransformerLM(small_config)
        result = new_model.load_state_dict(state_dict)
        print(f"Model load OK: {result}")

        # Cek LM heads
        print(f"\nLM heads di model baru: {new_model.list_domains()}")

        # ================================================================
        # Test 2: Simulasi MoE assets
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 2: Simulasi MoE Assets Bundle")
        print("=" * 60)
        
        # Buat dummy domain_keywords.json
        keywords_path = output_models_dir / "domain_keywords.json"
        keywords_data = {
            "domain_keywords": {
                "MOBIL": ["mobil", "mesin", "roda", "ban", "bensin"],
                "SAINS": ["air", "fotosintesis", "atom", "molekul"],
            },
        }
        with open(keywords_path, "w", encoding="utf-8") as f:
            json.dump(keywords_data, f)
        print(f"Created: {keywords_path}")
        
        # Buat dummy semantic_cache.npz
        cache_path = output_models_dir / "semantic_cache.npz"
        dummy_embeddings = {
            "MOBIL__embeddings": np.random.randn(5, 16).astype(np.float32),
            "SAINS__embeddings": np.random.randn(4, 16).astype(np.float32),
            "__paragraphs_json__": np.array(
                [json.dumps({
                    "MOBIL": ["paragraf 1", "paragraf 2"],
                    "SAINS": ["paragraf 3"],
                })],
                dtype=object,
            ),
        }
        np.savez_compressed(cache_path, **dummy_embeddings)
        print(f"Created: {cache_path}")

        # ================================================================
        # Test 3: Export dengan MoE assets
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 3: Export Model (dengan MoE assets)")
        print("=" * 60)
        
        model_path_moe = exporter.save(
            model=model,
            tokenizer=tokenizer,
            include_vocab=True,
            include_moe_assets=True,
        )

        file_size_moe = model_path_moe.stat().st_size
        print(f"\nModel file      : {model_path_moe}")
        print(f"File size       : {file_size_moe:,} bytes (sebelum: {file_size:,})")
        print(f"Size increase   : {file_size_moe - file_size:,} bytes")

        # ================================================================
        # Test 4: Extract MoE assets dari safetensors
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 4: Extract MoE Assets")
        print("=" * 60)
        
        moe_assets = extract_moe_assets(model_path_moe)
        
        print(f"Use MoE         : {moe_assets['use_moe']}")
        print(f"Domains         : {moe_assets['domains']}")
        print(f"Keywords count  : {sum(len(v) for v in moe_assets['keywords'].values())}")
        print(f"Paragraphs count: {sum(len(v) for v in moe_assets['paragraphs'].values())}")
        print(f"Embeddings:")
        for domain, emb in moe_assets['embeddings'].items():
            print(f"  {domain}: shape {emb.shape}, dtype {emb.dtype}")

        # ================================================================
        # Test 5: Verifikasi model bisa di-load kembali
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 5: Verify Model Reload")
        print("=" * 60)
        
        state_dict_moe, metadata_moe = load_safetensors(model_path_moe)
        
        # Filter out semantic cache tensors untuk load ke model
        model_state = {
            k: v for k, v in state_dict_moe.items()
            if not k.startswith(SEMANTIC_CACHE_TENSOR_PREFIX)
        }
        
        new_model_moe = CustomTransformerLM(small_config)
        result_moe = new_model_moe.load_state_dict(model_state)
        print(f"Model load OK: missing={len(result_moe['missing'])}, unexpected={len(result_moe['unexpected'])}")
        print(f"LM heads di model baru: {new_model_moe.list_domains()}")

        # ================================================================
        # Test 6: Inspect metadata
        # ================================================================
        print("\n" + "=" * 60)
        print("Test 6: Metadata Inspection")
        print("=" * 60)
        
        print("Metadata keys:")
        for key in sorted(metadata_moe.keys()):
            value = metadata_moe[key]
            if len(str(value)) > 100:
                value = str(value)[:100] + "..."
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("core/model_exporter.py (MoE) test OK")
        print("=" * 60)
