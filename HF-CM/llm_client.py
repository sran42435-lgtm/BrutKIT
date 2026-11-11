"""
llm_client.py
Adapter Model API & Resilience (Layer 10).

Fungsi:
1. Mengurung seluruh interaksi dengan Model AI eksternal (Stateless API Call).
2. Menyediakan antarmuka (Interface) yang seragam agar sistem tidak terikat pada satu vendor.
3. Menangani Retry & Backoff Strategy jika terjadi Timeout atau Rate Limit.
4. Menyediakan FakeLLMClient untuk keperluan testing otomatis tanpa koneksi internet/biaya API.

PRINSIP:
- Stateless: Klien tidak menyimpan riwayat percakapan. Riwayat ada di Mesin Memori.
- Fail-Open for Conversation: Jika model gagal, lempar error yang jelas agar Engine 
  bisa mengambil keputusan (misal: fallback ke respons sistem), jangan crash diam-diam.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from config import CONFIG
from models import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================
class LLMClientError(Exception):
    """Base error untuk semua kegagalan LLM."""
    pass

class ModelTimeoutError(LLMClientError):
    """Dipicu ketika API tidak merespons dalam batas waktu."""
    pass

class ModelUnavailableError(LLMClientError):
    """Dipicu ketika API_down atau rate_limit setelah semua retry habis."""
    pass


# ==========================================
# ABSTRACT BASE CLASS (Kontrak Interface)
# ==========================================
class BaseLLMClient(ABC):
    """
    Interface wajib untuk semua adapter Model API.
    Setiap provider (OpenAI, Anthropic, Ollama) harus mengimplementasikan kelas ini.
    """
    
    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Mengirim konteks dan prompt ke Model API, lalu mengembalikan respons teks.
        """
        pass


# ==========================================
# 1. FAKE LLM CLIENT (Untuk PoC & Testing)
# ==========================================
class FakeLLMClient(BaseLLMClient):
    """
    Klien simulasi untuk Integration Testing.
    Tidak memanggil API eksternal. Ia membaca ModelRequest dan merakit respons 
    deterministik untuk MEMBUKTIKAN bahwa injeksi memori persis berhasil dilakukan.
    """
    
    def generate(self, request: ModelRequest) -> ModelResponse:
        logger.info("FakeLLMClient: Memproses request secara lokal (Simulasi).")
        
        # Simulasi pemrosesan (menghitung token fiktif)
        prompt_tokens = len(request.system_context) // 4 + len(request.past_memory) // 4 + len(request.current_user_message) // 4
        
        # Logika Echo untuk membuktikan Context Injection berhasil
        if request.past_memory and "<past_memories>" in request.past_memory:
            # AI "pura-pura" membaca memori dan merangkumnya
            response_text = (
                f"[FAKE AI RESPONSE] Berdasarkan catatan masa lalu yang Anda berikan, "
                f"saya melihat ada memori persis yang diinjeksikan. "
                f"Menjawab pertanyaan terbaru Anda: '{request.current_user_message}'"
            )
        else:
            # Tidak ada memori yang diinjeksikan (Topik baru / Fail-closed retrieval)
            response_text = (
                f"[FAKE AI RESPONSE] Tidak ada memori masa lalu yang relevan ditemukan. "
                f"Saya akan menjawab berdasarkan pengetahuan umum untuk: '{request.current_user_message}'"
            )
            
        return ModelResponse(
            response_text=response_text,
            tokens_used=prompt_tokens + 50 # Estimasi output tokens
        )


# ==========================================
# 2. GENERIC API CLIENT (Skeleton untuk Provider Nyata)
# ==========================================
class GenericAPIClient(BaseLLMClient):
    """
    Skeleton untuk koneksi ke API nyata (OpenAI, Anthropic, dll).
    Menggunakan Retry & Backoff Strategy berdasarkan CONFIG.llm.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = CONFIG.llm.MAX_RETRIES
        self.timeout = CONFIG.llm.REQUEST_TIMEOUT_SECONDS
        self.model_name = CONFIG.llm.MODEL_NAME

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Mengirim request ke API eksternal dengan mekanisme Retry Exponential Backoff.
        """
        payload = self._build_payload(request)
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"API Call Attempt {attempt + 1}/{self.max_retries + 1}")
                
                # --- PLACEHOLDER FOR ACTUAL HTTP REQUEST ---
                # Di implementasi nyata, gunakan `requests` atau `httpx` di sini.
                # response = httpx.post(self.base_url, json=payload, headers=self._get_headers(), timeout=self.timeout)
                # response.raise_for_status()
                # return self._parse_response(response.json())
                
                raise NotImplementedError(
                    "HTTP Request asli belum diimplementasikan. "
                    "Gunakan FakeLLMClient untuk PoC, atau sambungkan SDK provider di sini."
                )
                # -------------------------------------------

            except TimeoutError:
                logger.warning(f"Timeout pada attempt {attempt + 1}.")
                if attempt == self.max_retries:
                    raise ModelTimeoutError(f"Model API timeout setelah {self.max_retries} percobaan.")
            
            except Exception as e:
                logger.error(f"Error API pada attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    raise ModelUnavailableError(f"Model API tidak tersedia: {e}")
                    
            # Exponential Backoff (Tunggu sebelum retry: 1s, 2s, 4s...)
            if attempt < self.max_retries:
                sleep_time = 2 ** attempt
                logger.info(f"Menunggu {sleep_time} detik sebelum retry...")
                time.sleep(sleep_time)

    def _build_payload(self, request: ModelRequest) -> dict:
        """Menyusun payload JSON sesuai format standar Chat Completion API."""
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": request.system_context},
                {"role": "system", "content": f"Context from Memory:\n{request.past_memory}" if request.past_memory else ""},
                {"role": "user", "content": request.current_user_message}
            ],
            "temperature": CONFIG.llm.TEMPERATURE,
            "max_tokens": CONFIG.llm.MAX_OUTPUT_TOKENS
        }

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
