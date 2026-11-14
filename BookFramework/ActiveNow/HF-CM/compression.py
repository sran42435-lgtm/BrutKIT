"""
compression.py
Lossless Exact Payload Engine.

Fungsi: Menjamin memori besar disimpan hemat (compressed) namun dapat 
direkonstruksi 100% identik (lossless) dengan verifikasi integritas ketat.

PRINSIP UTAMA: FAIL-CLOSED
1. Jika kompresi gagal -> Batalkan penyimpanan.
2. Jika dekompresi gagal -> Anggap memori hilang/rusak, jangan menebak.
3. Jika checksum tidak cocok -> Tolak data (IntegrityError).
"""

import hashlib
import logging
import uuid
from typing import Union

import zstandard as zstd

from config import CONFIG
from models import CompressedPayload

logger = logging.getLogger(__name__)

# ==========================================
# CUSTOM EXCEPTIONS (Fail-Closed Triggers)
# ==========================================
class CompressionError(Exception):
    """Dipicu ketika proses kompresi gagal."""
    pass

class DecompressionError(Exception):
    """Dipicu ketika data tidak bisa didekompresi (format salah/rusak)."""
    pass

class IntegrityError(Exception):
    """Dipicu ketika checksum tidak cocok (data korup/berubah)."""
    pass


class PayloadCompressor:
    """
    Engine untuk kompresi dan dekompresi payload memori.
    Menggunakan Zstandard untuk rasio kompresi tinggi dan kecepatan dekompresi cepat.
    """

    # [NEW] Versi kompresi untuk tracking perubahan algoritma di masa depan
    COMPRESSION_VERSION = "1.0"

    def __init__(self):
        self.compression_level = CONFIG.compression.COMPRESSION_LEVEL
        self.algorithm = CONFIG.compression.COMPRESSION_ALGORITHM
        self.checksum_algo = CONFIG.retrieval.CHECKSUM_ALGORITHM

    def _calculate_checksum(self, data: bytes) -> str:
        """Menghitung hash SHA-256 dari data mentah."""
        if self.checksum_algo == "sha256":
            return hashlib.sha256(data).hexdigest()
        raise ValueError(f"Algoritma checksum tidak didukung: {self.checksum_algo}")

    def compress(self, raw_data: Union[str, bytes]) -> CompressedPayload:
        """
        Mengompresi data mentah menjadi binary payload yang aman.
        
        Returns:
            CompressedPayload: Object Pydantic yang berisi blob, checksum, dan metadata.
            
        Raises:
            CompressionError: Jika proses kompresi gagal.
        """
        # Normalisasi input ke bytes
        if isinstance(raw_data, str):
            data_bytes = raw_data.encode('utf-8')
        else:
            data_bytes = raw_data

        if not data_bytes:
            raise CompressionError("Tidak bisa mengompresi data kosong.")

        try:
            # 1. Hitung Checksum DATA ASLI (sebelum dikompresi)
            # Ini penting untuk memverifikasi bahwa hasil dekompresi nanti 100% identik.
            checksum = self._calculate_checksum(data_bytes)
            
            # 2. Kompresi
            cctx = zstd.ZstdCompressor(level=self.compression_level)
            compressed_blob = cctx.compress(data_bytes)
            
            # 3. Bungkus dalam Model Data (Immutable)
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
        """
        Mendekompresi payload dan MEMVERIFIKASI integritasnya.
        
        Returns:
            bytes: Data asli yang telah diverifikasi.
            
        Raises:
            DecompressionError: Jika format binary tidak valid.
            IntegrityError: Jika data berubah (checksum mismatch).
        """
        # Validasi Algoritma
        if payload.compression_algorithm != self.algorithm:
            # Mencegah dekompresi liar dengan algoritma yang tidak dikenal
            raise DecompressionError(
                f"Mismatch algoritma: Diharapkan {self.algorithm}, "
                f"ditemukan {payload.compression_algorithm}"
            )

        try:
            dctx = zstd.ZstdDecompressor()
            # Dekompresi blob
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

        # VERIFIKASI INTEGRITAS (Fail-Closed Gate)
        actual_checksum = self._calculate_checksum(raw_data)
        if actual_checksum != payload.checksum:
            logger.critical(
                f"PELANGGARAN INTEGRITAS: Checksum tidak cocok! "
                f"Expected: {payload.checksum}, Actual: {actual_checksum}"
            )
            raise IntegrityError("Data korup terdeteksi. Checksum SHA-256 tidak cocok.")

        return raw_data

    def verify_integrity_only(self, payload: CompressedPayload) -> bool:
        """
        Melakukan dekompresi senyap hanya untuk memvalidasi checksum.
        Berguna untuk health check database tanpa membebani memori dengan data besar.
        """
        try:
            self.decompress(payload)
            return True
        except (DecompressionError, IntegrityError):
            return False
