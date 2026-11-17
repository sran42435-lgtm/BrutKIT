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

[BATCH C - ASSOCIATION EXPANSION]
LLM sekarang mengembalikan DUA output terpisah:
    1. response_text     : jawaban visible untuk user (tidak terkontaminasi metadata)
    2. internal_metadata : structured identity fragment dari LLM (ResponseMetadata)

Kontrak: metadata TIDAK BOLEH digabung sebagai visible text. Lapisan terpisah
antara user-facing content dan internal identity.

FakeLLMClient menghasilkan metadata deterministik berdasarkan input,
sehingga integration test dapat memverifikasi kontrak pipeline metadata
tanpa perlu LLM asli.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

from config import CONFIG
from models import ModelRequest, ModelResponse, ResponseMetadata

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
        Mengirim konteks dan prompt ke Model API, lalu mengembalikan respons teks
        + metadata internal (jika tersedia).
        """
        pass


# ==========================================
# METADATA EXTRACTION HELPER (shared)
# ==========================================

def _extract_topic_from_memory(past_memory: str) -> str:
    """
    Ekstrak topic_title dari XML tag di past_memory.

    Contoh input:
        <past_memories topic="Bisnis Kopi">...</past_memories>

    Returns:
        topic_title (e.g. "Bisnis Kopi") atau "" jika tidak ada.
    """
    if not past_memory:
        return ""
    match = re.search(r'topic="([^"]+)"', past_memory)
    return match.group(1) if match else ""


def _extract_first_fragment_key(past_memory: str) -> str:
    """
    Ekstrak fragment_key pertama dari hierarchy framing di past_memory.

    Contoh input (dari context_builder):
        SECTION: Modal Bisnis Kopi
        DEPTH: 1
        [Waktu: ...]
        EXACT MEMORY:
        ...

    Returns:
        fragment_key (e.g. "Modal Bisnis Kopi") atau "" jika tidak ada.
    """
    if not past_memory:
        return ""
    # Cari label SECTION / SUBSECTION / OPENING yang diikuti nama
    patterns = [
        r'SECTION:\s*([^\n]+)',
        r'SUBSECTION:\s*([^\n]+)',
        r'OPENING:\s*([^\n]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, past_memory)
        if match:
            return match.group(1).strip()
    return ""


# ==========================================
# 1. FAKE LLM CLIENT (Untuk PoC & Testing)
# ==========================================
class FakeLLMClient(BaseLLMClient):
    """
    Klien simulasi untuk Integration Testing.
    Tidak memanggil API eksternal. Ia membaca ModelRequest dan merakit respons 
    deterministik untuk MEMBUKTIKAN bahwa injeksi memori persis berhasil dilakukan.
    
    [FIX ARSITEKTURAL] FakeLLM kini mengekstrak topic_title dari XML tag di past_memory
    dan menyebatkannya ke dalam respons. Ini mensimulasikan LLM asli yang context-aware,
    sehingga Turn Analysis (Layer 11) dapat mendeteksi overlap dari respons AI.

    [BATCH C] FakeLLM juga mengembalikan internal_metadata (ResponseMetadata)
    yang deterministik berdasarkan input. Metadata TIDAK tercampur dengan
    response_text, sesuai kontrak pemisahan layer.

    Metadata deterministik yang dihasilkan:
        - fragment_key     : dari SECTION framing di past_memory, atau dari topic
        - helper_description: kalimat ringkas dari topic_context
        - primary_topic_hint: topic_context (nama TopicKey yang di-inject)
        - confidence       : 0.8 (tinggi, simulasi "yakin")
        - metadata_version : "1" (current)

    Ini cukup untuk menguji kontrak pipeline: parser menerima, validator lolos,
    resolver memilih USE_METADATA. Provider asli nanti bisa menghasilkan
    metadata yang lebih kaya.
    """
    
    def generate(self, request: ModelRequest) -> ModelResponse:
        logger.info("FakeLLMClient: Memproses request secara lokal (Simulasi).")
        
        # Simulasi pemrosesan (menghitung token fiktif)
        prompt_tokens = len(request.system_context) // 4 + len(request.past_memory) // 4 + len(request.current_user_message) // 4
        
        # Ekstrak context dari past_memory untuk dipakai di respons + metadata
        topic_context = _extract_topic_from_memory(request.past_memory) or "masa lalu"
        fragment_key_from_mem = _extract_first_fragment_key(request.past_memory)

        # Logika Echo untuk membuktikan Context Injection berhasil
        if request.past_memory and "<past_memories" in request.past_memory:
            # AI "pura-pura" membaca memori dan merangkumnya dengan menyebut topik
            response_text = (
                f"[FAKE AI RESPONSE] Berdasarkan catatan {topic_context} yang Anda berikan, "
                f"saya melihat ada memori persis yang diinjeksikan. "
                f"Menjawab pertanyaan terbaru Anda: '{request.current_user_message}'"
            )
        else:
            # Tidak ada memori yang diinjeksikan (Topik baru / Fail-closed retrieval)
            response_text = (
                f"[FAKE AI RESPONSE] Tidak ada memori masa lalu yang relevan ditemukan. "
                f"Saya akan menjawab berdasarkan pengetahuan umum untuk: '{request.current_user_message}'"
            )

        # [BATCH C] Bangun metadata deterministik
        internal_metadata = self._build_fake_metadata(
            request=request,
            topic_context=topic_context,
            fragment_key_from_mem=fragment_key_from_mem,
        )
            
        return ModelResponse(
            response_text=response_text,
            tokens_used=prompt_tokens + 50,  # Estimasi output tokens
            internal_metadata=internal_metadata,
        )

    def _build_fake_metadata(
        self,
        request: ModelRequest,
        topic_context: str,
        fragment_key_from_mem: str,
    ) -> ResponseMetadata:
        """
        Bangun ResponseMetadata deterministik untuk testing kontrak pipeline.

        Strategi:
            - Jika ada memory inject + fragment framing → pakai fragment_key
            - Jika ada memory inject tanpa framing → pakai topic_context sebagai key
            - Jika tidak ada memory → fragment_key dari user message (singkat)

        Helper description dibuat ringkas dari topic_context.
        Positive clues diambil dari topic_context (tokenisasi sederhana).
        Confidence tinggi (0.8) agar lolos MIN_METADATA_CONFIDENCE (0.45).
        """
        # 1. Tentukan fragment_key
        if fragment_key_from_mem:
            fragment_key = fragment_key_from_mem
        elif topic_context and topic_context != "masa lalu":
            fragment_key = topic_context
        else:
            # Fallback: ambil 3 kata pertama dari user message
            words = re.findall(r'\b\w+\b', request.current_user_message.lower())
            fragment_key = " ".join(words[:3]).capitalize() if words else "Topik Baru"

        # Clamp fragment_key length
        max_key_len = CONFIG.response_metadata.MAX_FRAGMENT_KEY_LENGTH
        if len(fragment_key) > max_key_len:
            fragment_key = fragment_key[:max_key_len].rstrip()

        # 2. Helper description (ringkas)
        if topic_context and topic_context != "masa lalu":
            helper_description = f"Pembahasan terkait {topic_context}."
        else:
            helper_description = "Pembahasan umum."

        max_helper_len = CONFIG.response_metadata.MAX_HELPER_DESCRIPTION_LENGTH
        if len(helper_description) > max_helper_len:
            helper_description = helper_description[:max_helper_len].rstrip()

        # 3. Primary topic hint
        primary_topic_hint = topic_context if topic_context != "masa lalu" else ""

        # 4. Positive clues: token dari topic_context (3 token pertama bermakna)
        positive_clues: list = []
        if topic_context and topic_context != "masa lalu":
            tokens = [
                t for t in re.findall(r'\b\w+\b', topic_context.lower())
                if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH
            ]
            positive_clues = tokens[:3]

        # 5. Negative clues: none dari fake (user input tidak diparse di sini)
        negative_clues: list = []

        # 6. Related topic/fragment hints: empty (fake tidak punya knowledge graph)
        related_topic_hints: list = []
        related_fragment_hints: list = []

        # 7. Confidence tinggi (simulasi LLM yang yakin)
        confidence = 0.80

        return ResponseMetadata(
            fragment_key=fragment_key,
            helper_description=helper_description,
            primary_topic_hint=primary_topic_hint,
            positive_clues=positive_clues,
            negative_clues=negative_clues,
            related_topic_hints=related_topic_hints,
            related_fragment_hints=related_fragment_hints,
            confidence=confidence,
            metadata_version=CONFIG.response_metadata.CURRENT_METADATA_VERSION,
        )


# ==========================================
# 2. GENERIC API CLIENT (Skeleton untuk Provider Nyata)
# ==========================================
class GenericAPIClient(BaseLLMClient):
    """
    Skeleton untuk koneksi ke API nyata (OpenAI, Anthropic, dll).
    Menggunakan Retry & Backoff Strategy berdasarkan CONFIG.llm.

    [BATCH C] Skeleton ini juga menunjukkan bagaimana metadata diekstrak
    dari respons provider asli. Implementasi nyata tinggal mengganti
    placeholder HTTP request dan menyesuaikan parser metadata sesuai
    format provider (misal: OpenAI function calling, Anthropic tool use,
    atau JSON-mode terstruktur).
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
                # raw = response.json()
                #
                # return self._parse_response(raw)
                
                raise NotImplementedError(
                    "HTTP Request asli belum diimplementasikan. "
                    "Gunakan FakeLLMClient untuk PoC, atau sambungkan SDK provider di sini."
                )
                # -------------------------------------------

            except TimeoutError:
                logger.warning(f"Timeout pada attempt {attempt + 1}.")
                if attempt == self.max_retries:
                    raise ModelTimeoutError(f"Model API timeout setelah {self.max_retries} percobaan.")
            
            except NotImplementedError:
                # Re-raise NotImplementedError agar test tahu ini skeleton
                raise
            
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

    def _parse_response(self, raw: Dict[str, Any]) -> ModelResponse:
        """
        [BATCH C] Parse raw response JSON dari provider menjadi ModelResponse.

        Skeleton ini menunjukkan pola ekstraksi metadata:
            1. Ambil response_text dari field standar provider
            2. Coba parse internal_metadata dari field terstruktur (jika ada)
            3. Jika metadata tidak tersedia/malformed → internal_metadata=None

        Provider-specific implementations harus override method ini sesuai
        format mereka:
            - OpenAI     : function calling / JSON mode
            - Anthropic  : tool_use blocks
            - Ollama     : custom structured output
            - Local LLM  : raw JSON dalam text response
        """
        # Default: ekstrak dari struktur generik
        # (provider asli akan override dengan pola mereka sendiri)
        response_text = ""
        internal_metadata = None

        # Pola umum: choices[0].message.content
        choices = raw.get("choices") or []
        if choices and isinstance(choices, list):
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, str):
                response_text = content

        # Coba ekstrak metadata jika provider mendukung
        # (format spesifik tergantung provider - ini contoh generik)
        raw_metadata = raw.get("internal_metadata") or raw.get("metadata")
        if raw_metadata and isinstance(raw_metadata, dict):
            internal_metadata = self._extract_metadata_from_response(raw_metadata)

        tokens_used = raw.get("usage", {}).get("total_tokens", 0)
        if not isinstance(tokens_used, int):
            try:
                tokens_used = int(tokens_used)
            except (TypeError, ValueError):
                tokens_used = 0

        return ModelResponse(
            response_text=response_text,
            tokens_used=tokens_used,
            internal_metadata=internal_metadata,
        )

    def _extract_metadata_from_response(
        self,
        raw_metadata: Dict[str, Any],
    ) -> Optional[ResponseMetadata]:
        """
        [BATCH C] Konversi raw metadata dict → ResponseMetadata.

        Method ini reusable: dipanggil dari _parse_response untuk provider
        yang mengembalikan metadata sebagai field terpisah, atau dapat
        dipanggil manual dari implementasi provider spesifik.

        Returns None jika parsing gagal (defensive: jangan crash karena
        metadata yang tidak sempurna).
        """
        try:
            return ResponseMetadata(
                fragment_key=str(raw_metadata.get("fragment_key", "") or "")[:CONFIG.response_metadata.MAX_FRAGMENT_KEY_LENGTH],
                helper_description=str(raw_metadata.get("helper_description", "") or "")[:CONFIG.response_metadata.MAX_HELPER_DESCRIPTION_LENGTH],
                primary_topic_hint=str(raw_metadata.get("primary_topic_hint", "") or "")[:CONFIG.response_metadata.MAX_TOPIC_HINT_LENGTH],
                positive_clues=list(raw_metadata.get("positive_clues", []) or [])[:CONFIG.response_metadata.MAX_POSITIVE_CLUES],
                negative_clues=list(raw_metadata.get("negative_clues", []) or [])[:CONFIG.response_metadata.MAX_NEGATIVE_CLUES],
                related_topic_hints=list(raw_metadata.get("related_topic_hints", []) or [])[:CONFIG.response_metadata.MAX_RELATED_TOPIC_HINTS],
                related_fragment_hints=list(raw_metadata.get("related_fragment_hints", []) or [])[:CONFIG.response_metadata.MAX_RELATED_FRAGMENT_HINTS],
                confidence=float(raw_metadata.get("confidence", 0.0) or 0.0),
                metadata_version=str(raw_metadata.get("metadata_version", CONFIG.response_metadata.CURRENT_METADATA_VERSION) or CONFIG.response_metadata.CURRENT_METADATA_VERSION),
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"Gagal parse metadata dari respons LLM: {e}")
            return None
