# core/model_exporter.py
#
# Model Exporter / Generator file model biner.
#
# Arsitektur: RAG (Retrieval-Augmented Generation) - Single Head
#
# Tugas utama:
# 1. Mengambil seluruh state_dict model.
# 2. Melakukan casting / quantization presisi tensor.
# 3. Menyusun header metadata.
# 4. Menulis file model biner tunggal (.safetensors-like).
# 5. Menyertakan vocab/tokenizer di dalam metadata.
# 6. Bundle retrieval index sebagai file terpisah.
# 7. Copy engine_standalone.py untuk deployment.
# 8. Menyediakan fungsi load kembali.
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
# - core/retriever.py    : retrieval index yang akan di-bundle

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
    Termasuk RAG templates.
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
                "architecture": "rag-single-head",
                "special_tokens": {
                    "pad": config.special_tokens.pad,
                    "unk": config.special_tokens.unk,
                    "bos": config.special_tokens.bos,
                    "eos": config.special_tokens.eos,
                    "endoftext": endoftext_token,
                },
                "rag_templates": {
                    "qa_train": getattr(tokenizer, "QA_TRAIN_TEMPLATE", 
                        "Konteks: {context} Pertanyaan: {question} Jawaban: {answer}"),
                    "qa_inference": getattr(tokenizer, "QA_INFERENCE_TEMPLATE",
                        "Konteks: {context} Pertanyaan: {question} Jawaban:"),
                    "qa_no_context": getattr(tokenizer, "QA_NO_CONTEXT_TEMPLATE",
                        "Pertanyaan: {question} Jawaban:"),
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
# RAG ASSETS HANDLING
# ============================================================================

def _copy_retrieval_index(config: Config, output_dir: Path, verbose: bool = True) -> Optional[Path]:
    """
    Copy retrieval index ke output folder untuk deployment.
    
    Index disimpan sebagai file terpisah (bukan di dalam safetensors)
    karena format JSON inverted index lebih mudah di-update dan load.
    
    Return:
    - Path ke file index, atau None jika tidak ada
    """
    index_src = Path(config.paths.retrieval_index_path)
    
    if not index_src.exists():
        if verbose:
            print(f"[Exporter] ⚠️ Retrieval index tidak ditemukan: {index_src}")
            print(f"[Exporter]    Jalankan training dulu untuk membangun index.")
        return None
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        index_dst = output_dir / "retrieval_index.json"
        shutil.copy2(index_src, index_dst)
        
        if verbose:
            file_size = index_dst.stat().st_size
            print(f"[Exporter] ✅ Copied retrieval index: {index_dst.name} ({file_size:,} bytes)")
        
        return index_dst
    except Exception as exc:
        if verbose:
            print(f"[Exporter] ⚠️ Gagal copy retrieval index: {exc}")
        return None


def _copy_qa_dataset(config: Config, output_dir: Path, verbose: bool = True) -> Optional[Path]:
    """
    Copy QA dataset ke output folder (optional, untuk reference).
    
    Return:
    - Path ke file dataset, atau None jika tidak ada
    """
    dataset_src = Path(config.paths.qa_dataset_path)
    
    if not dataset_src.exists():
        return None
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_dst = output_dir / "qa_dataset.json"
        shutil.copy2(dataset_src, dataset_dst)
        
        if verbose:
            file_size = dataset_dst.stat().st_size
            print(f"[Exporter] ✅ Copied QA dataset: {dataset_dst.name} ({file_size:,} bytes)")
        
        return dataset_dst
    except Exception as exc:
        if verbose:
            print(f"[Exporter] ⚠️ Gagal copy QA dataset: {exc}")
        return None


def _copy_engine_standalone(config: Config, output_dir: Path, verbose: bool = True) -> Optional[Path]:
    """
    Copy engine_standalone.py ke output folder untuk deployment.
    
    Return:
    - Path ke file yang di-copy, atau None jika gagal
    """
    engine_src = Path(__file__).resolve().parent / "engine_standalone.py"
    
    if not engine_src.exists():
        if verbose:
            print(f"[Exporter] ⚠️ engine_standalone.py tidak ditemukan: {engine_src}")
        return None
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        engine_dst = output_dir / "otak_engine.py"
        shutil.copy2(engine_src, engine_dst)
        
        if verbose:
            print(f"[Exporter] ✅ Copied standalone engine: {engine_dst.name}")
        
        return engine_dst
    except Exception as exc:
        if verbose:
            print(f"[Exporter] ⚠️ Gagal copy engine_standalone.py: {exc}")
        return None


def _read_index_stats(index_path: Path) -> Dict:
    """
    Baca statistik dari retrieval index untuk metadata.
    
    Return:
    - dict dengan stats (total_docs, total_terms, etc)
    """
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        stats = data.get("stats", {})
        return {
            "total_docs": stats.get("total_docs", 0),
            "total_terms": stats.get("total_terms", 0),
            "version": data.get("version", "unknown"),
        }
    except Exception:
        return {}


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
    include_rag_assets: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Mengemas seluruh bobot model menjadi satu file biner tunggal.
    
    Untuk arsitektur RAG:
    - Model weights disimpan di safetensors
    - Retrieval index disimpan sebagai file terpisah
    - QA dataset disimpan sebagai file terpisah (opsional)
    - Standalone engine di-copy untuk deployment

    Struktur file output:
        otak_model.safetensors    ← model weights + metadata
        retrieval_index.json      ← BM25 index (terpisah)
        qa_dataset.json           ← QA pairs (terpisah, opsional)
        otak_engine.py            ← standalone engine

    Args:
        model: CustomTransformerLM instance
        tokenizer: Tokenizer instance (opsional)
        config: Config object
        output_filepath: path output (default dari config)
        export_dtype: dtype untuk export (default dari config)
        include_vocab: sertakan vocab di metadata
        include_rag_assets: copy retrieval index + engine
        verbose: print progress
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

    if verbose:
        print()
        print("=" * 70)
        print("EXPORT MODEL (RAG Architecture)")
        print("=" * 70)

    state_dict = model.state_dict()

    tensor_entries: Dict[str, dict] = {}
    tensor_bytes = bytearray()
    offset = 0
    quant_info: Dict[str, dict] = {}

    # ================================================================
    # Serialize model weights
    # ================================================================
    if verbose:
        print(f"[Exporter] Serializing {len(state_dict)} tensors...")
    
    for name, arr in state_dict.items():
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

    if verbose:
        print(f"[Exporter]   Tensors serialized: {len(tensor_entries)}")
        print(f"[Exporter]   Raw bytes: {len(tensor_bytes):,}")

    tensor_sha256 = hashlib.sha256(bytes(tensor_bytes)).hexdigest()

    # ================================================================
    # Build metadata
    # ================================================================
    metadata: Dict[str, str] = {
        "format": "pt",
        "producer": "CreateMiniBot-RAG-exporter",
        "architecture": "rag-single-head",
        "model_type": "CustomTransformerLM",
        "hidden_size": str(config.model.embedding_dim),
        "num_attention_heads": str(config.model.num_attention_heads),
        "num_layers": str(config.model.num_layers),
        "ffn_hidden_dim": str(config.model.ffn_hidden_dim),
        "vocab_size": str(config.model.vocab_size),
        "context_length": str(config.model.max_position_embeddings),
        "export_dtype": export_dtype,
        "tensor_sha256": tensor_sha256,
        # RAG-specific flags
        "use_rope": "true" if getattr(config.model, 'use_rope', True) else "false",
        "use_weight_tying": "true" if getattr(config.model, 'use_weight_tying', True) else "false",
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
            if verbose:
                print(f"[Exporter] ✅ Tokenizer bundled ({tokenizer_payload.get('version', '?')} )")

    # Add quantization info
    if quant_info:
        metadata["quantization"] = json.dumps(
            quant_info,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if verbose:
            print(f"[Exporter] Quantization info: {len(quant_info)} tensors")

    # ================================================================
    # Bundle RAG assets
    # ================================================================
    index_path = None
    dataset_path = None
    engine_path = None
    
    if include_rag_assets:
        if verbose:
            print(f"[Exporter] Copying RAG assets...")
        
        output_dir = output_filepath.parent
        
        # Copy retrieval index
        index_path = _copy_retrieval_index(config, output_dir, verbose=verbose)
        
        # Add index stats to metadata
        if index_path:
            index_stats = _read_index_stats(index_path)
            if index_stats:
                metadata["retrieval_index_stats"] = json.dumps(index_stats)
                metadata["has_retrieval_index"] = "true"
        
        # Copy QA dataset (optional reference)
        dataset_path = _copy_qa_dataset(config, output_dir, verbose=False)
        if dataset_path:
            metadata["has_qa_dataset"] = "true"
        
        # Copy standalone engine
        engine_path = _copy_engine_standalone(config, output_dir, verbose=verbose)
        if engine_path:
            metadata["engine_standalone"] = engine_path.name

    # Recompute SHA-256 (final)
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

    file_size = output_filepath.stat().st_size

    if verbose:
        print()
        print(f"[Exporter] ✅ Model exported:")
        print(f"   Path      : {output_filepath}")
        print(f"   File size : {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        print(f"   SHA-256   : {tensor_sha256[:16]}...")
        print()
        print(f"[Exporter] Output directory: {output_filepath.parent}")
        for f in sorted(output_filepath.parent.iterdir()):
            if f.is_file():
                size = f.stat().st_size
                marker = " ← model" if f == output_filepath else ""
                marker = " ← index" if f.name == "retrieval_index.json" else marker
                marker = " ← engine" if f.name == "otak_engine.py" else marker
                marker = " ← dataset" if f.name == "qa_dataset.json" else marker
                print(f"   {f.name:40s} {size:>12,} bytes{marker}")
        print("=" * 70)

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
    - state_dict: dict nama tensor → numpy array
    - metadata: dict metadata string
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
    
    Return:
    - metadata dict
    """
    state_dict, metadata = load_safetensors(filepath)
    model.load_state_dict(state_dict)
    return metadata


def inspect_model_file(filepath: Path) -> Dict:
    """
    Inspect safetensors file tanpa load ke model.
    
    Berguna untuk debugging atau verifikasi.
    
    Return:
    - dict dengan info: metadata, tensor_names, tensor_count, file_size, etc
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {"error": f"File tidak ditemukan: {filepath}"}
    
    file_size = filepath.stat().st_size
    
    with open(filepath, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return {"error": "File corrupt: header terlalu pendek"}
        
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f.read(header_len)
    
    header = json.loads(header_bytes.decode("utf-8"))
    metadata = header.get("__metadata__", {})
    
    tensor_names = [k for k in header.keys() if k != "__metadata__"]
    
    # Parse tokenizer info
    tokenizer_info = None
    if "tokenizer" in metadata:
        try:
            tokenizer_info = json.loads(metadata["tokenizer"])
        except Exception:
            pass
    
    # Parse retrieval index stats
    index_stats = None
    if "retrieval_index_stats" in metadata:
        try:
            index_stats = json.loads(metadata["retrieval_index_stats"])
        except Exception:
            pass
    
    return {
        "file_path": str(filepath),
        "file_size": file_size,
        "architecture": metadata.get("architecture", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
        "producer": metadata.get("producer", "unknown"),
        "export_dtype": metadata.get("export_dtype", "unknown"),
        "tensor_count": len(tensor_names),
        "tensor_names": tensor_names,
        "vocab_size": int(metadata.get("vocab_size", 0)),
        "hidden_size": int(metadata.get("hidden_size", 0)),
        "num_layers": int(metadata.get("num_layers", 0)),
        "num_attention_heads": int(metadata.get("num_attention_heads", 0)),
        "context_length": int(metadata.get("context_length", 0)),
        "use_rope": metadata.get("use_rope", "false") == "true",
        "use_weight_tying": metadata.get("use_weight_tying", "false") == "true",
        "has_tokenizer": "tokenizer" in metadata,
        "tokenizer_info": tokenizer_info,
        "has_retrieval_index": metadata.get("has_retrieval_index", "false") == "true",
        "retrieval_index_stats": index_stats,
        "has_engine_standalone": "engine_standalone" in metadata,
        "engine_standalone_name": metadata.get("engine_standalone"),
        "tensor_sha256": metadata.get("tensor_sha256", "unknown"),
        "metadata_keys": list(metadata.keys()),
    }


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
        include_rag_assets: bool = True,
        verbose: bool = True,
    ) -> Path:
        """
        Simpan model ke file biner tunggal + RAG assets.
        
        Args:
            model: CustomTransformerLM instance
            tokenizer: Tokenizer instance (opsional)
            output_path: path output (default dari config)
            export_dtype: dtype untuk export
            include_vocab: sertakan vocab di metadata
            include_rag_assets: copy retrieval index + engine
            verbose: print progress
        
        Return:
            Path ke file model yang dibuat
        """
        return export_to_safetensors(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            output_filepath=output_path,
            export_dtype=export_dtype,
            include_vocab=include_vocab,
            include_rag_assets=include_rag_assets,
            verbose=verbose,
        )

    # Alias untuk backward compatibility
    export = save

    def inspect(self, filepath: Optional[Path] = None) -> Dict:
        """
        Inspect model file tanpa load.
        
        Args:
            filepath: path ke safetensors file (default dari config)
        
        Return:
            Dict dengan info model
        """
        if filepath is None:
            filepath = self.default_output_path
        
        return inspect_model_file(filepath)


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
        RetrievalConfig,
    )
    from core.tokenizer import Tokenizer
    from core.architecture import CustomTransformerLM

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        datasets_dir = tmp_path / "datasets"
        output_models_dir = tmp_path / "output_models"

        datasets_dir.mkdir(parents=True, exist_ok=True)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        # Sample corpus
        sample_file = datasets_dir / "sample.txt"
        sample_file.write_text(
            "Mobil adalah kendaraan beroda empat yang menggunakan mesin sebagai penggerak utama.\n\n"
            "Rem mobil berfungsi untuk mengurangi kecepatan dan menghentikan kendaraan.\n\n"
            "Air adalah senyawa kimia dengan rumus H2O yang penting bagi kehidupan.\n\n"
            "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan menggunakan cahaya matahari.",
            encoding="utf-8",
        )

        # Buat dummy retrieval index
        index_path = output_models_dir / "retrieval_index.json"
        dummy_index = {
            "version": "1.0",
            "config": {"k1": 1.5, "b": 0.75},
            "corpus": [
                "Mobil adalah kendaraan beroda empat.",
                "Rem mobil berfungsi untuk menghentikan.",
                "Air adalah senyawa H2O.",
                "Fotosintesis menghasilkan oksigen.",
            ],
            "doc_lengths": [6, 6, 5, 5],
            "avg_doc_length": 5.5,
            "inverted_index": {
                "mobil": {"0": 1, "1": 1},
                "kendaraan": {"0": 1},
                "rem": {"1": 1},
                "air": {"2": 1},
                "h2o": {"2": 1},
                "fotosintesis": {"3": 1},
            },
            "idf_cache": {
                "mobil": 0.5,
                "kendaraan": 1.0,
                "rem": 1.0,
                "air": 1.0,
                "h2o": 1.0,
                "fotosintesis": 1.0,
            },
            "stats": {
                "total_docs": 4,
                "total_terms": 6,
            },
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(dummy_index, f, indent=2)

        # Buat dummy QA dataset
        qa_path = output_models_dir / "qa_dataset.json"
        dummy_qa = {
            "version": "1.0",
            "architecture": "rag-single-head",
            "metadata": {
                "total_samples": 2,
                "source_corpus_size": 4,
                "augmentation_factor": 1,
                "template_types": {"definition": 2},
            },
            "samples": [
                {
                    "context": "Mobil adalah kendaraan beroda empat.",
                    "question": "Apa itu mobil?",
                    "answer": "Mobil adalah kendaraan beroda empat.",
                    "source_file": "sample.txt",
                    "template_type": "definition",
                    "augmentation": "original",
                },
                {
                    "context": "Air adalah senyawa H2O.",
                    "question": "Apa itu air?",
                    "answer": "Air adalah senyawa H2O.",
                    "source_file": "sample.txt",
                    "template_type": "definition",
                    "augmentation": "original",
                },
            ],
        }
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(dummy_qa, f, indent=2)

        paths = PathsConfig(
            datasets_dir=datasets_dir,
            output_models_dir=output_models_dir,
            vocab_path=tmp_path / "vocab.json",
            final_model_path=output_models_dir / "otak_model.safetensors",
            checkpoint_path=output_models_dir / "training_checkpoint.npz",
            retrieval_index_path=index_path,
            qa_dataset_path=qa_path,
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
                export_dtype="float16",
                use_rope=True,
                use_weight_tying=True,
                attention_logit_cap=30.0,
                final_logit_cap=50.0,
                drop_path_rate=0.1,
            ),
            retrieval=RetrievalConfig(
                top_k=3,
                bm25_k1=1.5,
                bm25_b=0.75,
            ),
        )

        tokenizer = Tokenizer(small_config)
        tokenizer.train()

        model = CustomTransformerLM(small_config)

        exporter = ModelExporter(small_config)

        # ================================================================
        # Test 1: Export tanpa RAG assets
        # ================================================================
        print("=" * 70)
        print("Test 1: Export Model (tanpa RAG assets)")
        print("=" * 70)
        
        model_path = exporter.save(
            model=model,
            tokenizer=tokenizer,
            include_vocab=True,
            include_rag_assets=False,
            verbose=True,
        )

        file_size = model_path.stat().st_size
        print(f"\nModel file: {model_path}")
        print(f"File size: {file_size:,} bytes")

        # ================================================================
        # Test 2: Export dengan RAG assets (lengkap)
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 2: Export Model (dengan RAG assets)")
        print("=" * 70)
        
        model_path_rag = exporter.save(
            model=model,
            tokenizer=tokenizer,
            include_vocab=True,
            include_rag_assets=True,
            verbose=True,
        )

        # ================================================================
        # Test 3: Load & verify
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 3: Load & Verify")
        print("=" * 70)
        
        state_dict, metadata = load_safetensors(model_path_rag)
        
        print(f"Tensors loaded: {len(state_dict)}")
        print(f"Architecture: {metadata.get('architecture')}")
        print(f"Export dtype: {metadata.get('export_dtype')}")
        print(f"Has tokenizer: {'tokenizer' in metadata}")
        print(f"Has retrieval index: {metadata.get('has_retrieval_index', 'false')}")
        print(f"Tensor SHA-256: {metadata.get('tensor_sha256', '')[:16]}...")
        
        # Verify model load
        new_model = CustomTransformerLM(small_config)
        result = new_model.load_state_dict(state_dict)
        print(f"\nModel load result:")
        print(f"  Missing: {len(result['missing'])}")
        print(f"  Unexpected: {len(result['unexpected'])}")

        # ================================================================
        # Test 4: Inspect
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 4: Inspect Model File")
        print("=" * 70)
        
        info = exporter.inspect(model_path_rag)
        
        print(f"File: {info['file_path']}")
        print(f"Size: {info['file_size']:,} bytes")
        print(f"Architecture: {info['architecture']}")
        print(f"Model type: {info['model_type']}")
        print(f"Vocab size: {info['vocab_size']}")
        print(f"Hidden size: {info['hidden_size']}")
        print(f"Num layers: {info['num_layers']}")
        print(f"Context length: {info['context_length']}")
        print(f"Use RoPE: {info['use_rope']}")
        print(f"Weight tying: {info['use_weight_tying']}")
        print(f"Tensor count: {info['tensor_count']}")
        print(f"Has tokenizer: {info['has_tokenizer']}")
        print(f"Has retrieval index: {info['has_retrieval_index']}")
        if info['retrieval_index_stats']:
            print(f"  Index stats: {info['retrieval_index_stats']}")
        print(f"Has engine standalone: {info['has_engine_standalone']}")

        # ================================================================
        # Test 5: Verify output files
        # ================================================================
        print("\n" + "=" * 70)
        print("Test 5: Verify Output Files")
        print("=" * 70)
        
        expected_files = [
            "otak_model.safetensors",
            "retrieval_index.json",
            "qa_dataset.json",
        ]
        
        for fname in expected_files:
            fpath = output_models_dir / fname
            exists = fpath.exists()
            size = fpath.stat().st_size if exists else 0
            status = "✅" if exists else "❌"
            print(f"  {status} {fname:30s} {size:>10,} bytes")

        print("\n" + "=" * 70)
        print("core/model_exporter.py (RAG) ALL TESTS PASSED")
        print("=" * 70)
