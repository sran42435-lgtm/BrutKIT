"""
compression.py
Lossless Exact Payload Engine.
"""

import hashlib
import logging
import uuid
from typing import Union

import zstandard as zstd

from config import CONFIG
from models import CompressedPayload

logger = logging.getLogger(__name__)

class CompressionError(Exception):
    pass

class DecompressionError(Exception):
    pass

class IntegrityError(Exception):
    pass


class PayloadCompressor:
    def __init__(self):
        self.compression_level = CONFIG.compression.COMPRESSION_LEVEL
        self.algorithm = CONFIG.compression.COMPRESSION_ALGORITHM
        self.checksum_algo = CONFIG.retrieval.CHECKSUM_ALGORITHM

    def _calculate_checksum(self, data: bytes) -> str:
        if self.checksum_algo == "sha256":
            return hashlib.sha256(data).hexdigest()
        raise ValueError(f"Algoritma checksum tidak didukung: {self.checksum_algo}")

    def compress(self, raw_data: Union[str, bytes]) -> CompressedPayload:
        if isinstance(raw_data, str):
            data_bytes = raw_data.encode('utf-8')
        else:
            data_bytes = raw_data

        if not data_bytes:
            raise CompressionError("Tidak bisa mengompresi data kosong.")

        try:
            checksum = self._calculate_checksum(data_bytes)
            cctx = zstd.ZstdCompressor(level=self.compression_level)
            compressed_blob = cctx.compress(data_bytes)
            
            return CompressedPayload(
                payload_id=str(uuid.uuid4()),
                compressed_blob=compressed_blob,
                checksum=checksum,
                compression_algorithm=self.algorithm,
                original_size=len(data_bytes),
                compressed_size=len(compressed_blob)
            )
            
        except Exception as e:
            logger.error(f"Kegagalan kompresi: {e}")
            raise CompressionError(f"Gagal mengompresi payload: {e}")

    def decompress(self, payload: CompressedPayload) -> bytes:
        if payload.compression_algorithm != self.algorithm:
            raise DecompressionError(
                f"Mismatch algoritma: Diharapkan {self.algorithm}, "
                f"ditemukan {payload.compression_algorithm}"
            )

        try:
            dctx = zstd.ZstdDecompressor()
            raw_data = dctx.decompress(payload.compressed_blob)
            
        except zstd.ZstdError as e:
            # [FIX] Zstd memiliki internal checksum. Jika data korup (bit rot), 
            # zstd akan melempar ZstdError SEBELUM kita sempat menghitung SHA-256.
            # Secara arsitektural, ini adalah pelanggaran integritas.
            logger.error(f"Kegagalan dekompresi (ZstdError): {e}")
            raise IntegrityError(f"Data binary korup terdeteksi oleh Zstd: {e}")
        except Exception as e:
            logger.error(f"Kegagalan dekompresi (Unexpected): {e}")
            raise DecompressionError(f"Error tak terduga saat dekompresi: {e}")

        actual_checksum = self._calculate_checksum(raw_data)
        if actual_checksum != payload.checksum:
            logger.critical(
                f"PELANGGARAN INTEGRITAS: Checksum tidak cocok! "
                f"Expected: {payload.checksum}, Actual: {actual_checksum}"
            )
            raise IntegrityError("Data korup terdeteksi. Checksum SHA-256 tidak cocok.")

        return raw_data

    def verify_integrity_only(self, payload: CompressedPayload) -> bool:
        try:
            self.decompress(payload)
            return True
        except (DecompressionError, IntegrityError):
            return False
