"""
fragment_identity.py
Fragment Identity Extraction (Mekanisme 3 — Prioritas 1).

Tanggung jawab:
- Ekstrak identitas fragment dari USER INPUT + AI RESPONSE secara combined
- Hasilkan fragment_key + helper_description yang representatif
- Hitung identity_confidence (seberapa solid identitas fragment)

Prinsip kunci:
  Fragment Identity = Combined Evidence (User + AI), bukan User Alone.
  User "Modalnya berapa?" + AI "Untuk bisnis kopi, modal awal..."
  -> fragment_key: "Modal Bisnis Kopi"
  -> helper: "Pembahasan modal awal untuk bisnis kopi"

Kenapa tidak hanya user input:
  - Input user sering berupa probe pendek (short probe)
  - AI response biasanya memperjelas konteks (mengisi yang user tidak sebut)
  - Combined text = representasi fragment yang sebenarnya disimpan di payload

Kenapa tidak hanya AI response:
  - User input tetap sinyal utama intent
  - AI bisa hallucinate topik yang tidak diminta
  - User tokens diberi bobot primary, AI tokens sebagai enricher
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict

from config import CONFIG
from models import TopicKey, SignalProfile, TokenQuality

logger = logging.getLogger(__name__)


@dataclass
class FragmentIdentity:
    """Hasil ekstraksi identitas fragment."""
    fragment_key: str = ""
    helper_description: str = ""
    representative_tokens: List[str] = field(default_factory=list)
    meaningful_tokens: List[str] = field(default_factory=list)
    identity_confidence: float = 0.0
    user_token_count: int = 0
    ai_token_count: int = 0
    combined_token_count: int = 0
    reason: str = ""


class FragmentIdentityExtractor:
    """
    Ekstrak identitas fragment dari user input + AI response.

    Dipanggil SEBELUM payload dikompres (urutan wajib).
    Hasilnya dipakai untuk:
      - fragment_key / fragment_helper_description di FragmentMetadata
      - placement decision (parent matching)
      - helper propagation (kontribusi ke parent)
    """

    # Token yang TIDAK boleh masuk ke fragment_key / helper_description.
    # Diambil dari CONFIG.token_quality.LOW_INFORMATION_TOKENS + beberapa
    # meta-token AI yang bukan konten topik.
    _AI_META_MARKERS: frozenset = frozenset({
        "response", "simulasi", "simulasi untuk", "berdasarkan", "catatan",
        "yang anda berikan", "saya melihat", "ada memori", "persis yang",
        "diinjeksikan", "menjawab", "pertanyaan terbaru",
    })

    def __init__(self):
        self._min_token_length = CONFIG.signal.MIN_TOKEN_LENGTH
        self._hq = CONFIG.hierarchy
        self._hp = CONFIG.helper_propagation

    def _tokenize(self, text: str) -> List[str]:
        """Tokenisasi sederhana dengan filter panjang minimum."""
        if not text:
            return []
        raw = re.findall(r'\b\w+\b', text.lower())
        return [t for t in raw if len(t) >= self._min_token_length]

    def _filter_low_info(
        self,
        tokens: List[str],
        quality_map: Optional[Dict[str, TokenQuality]] = None,
    ) -> List[str]:
        """Buang token low-information."""
        low_info_set = CONFIG.token_quality.LOW_INFORMATION_TOKENS
        result = []
        for t in tokens:
            if t in low_info_set:
                continue
            if quality_map and quality_map.get(t) == TokenQuality.LOW_INFORMATION:
                continue
            # Filter AI meta markers (substring check)
            if t in self._AI_META_MARKERS:
                continue
            result.append(t)
        return result

    def _build_fragment_key(self, meaningful_tokens: List[str]) -> str:
        """
        Bangun fragment_key dari token bermakna.

        Batas: max FRAGMENT_KEY_MAX_TOKENS token.
        Format: Capitalized Words.
        """
        if not meaningful_tokens:
            return ""
        key_tokens = meaningful_tokens[:self._hq.FRAGMENT_KEY_MAX_TOKENS]
        # Deduplicate sambil pertahankan urutan
        seen: Set[str] = set()
        deduped: List[str] = []
        for t in key_tokens:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return " ".join(t.capitalize() for t in deduped)

    def _build_helper_description(self, meaningful_tokens: List[str]) -> str:
        """
        Bangun helper_description ringkas.

        Batas: MAX_HELPER_CONTRIBUTION_TOKENS token, MAX_FRAGMENT_HELPER_LENGTH char.
        """
        if not meaningful_tokens:
            return ""
        helper_tokens = meaningful_tokens[:self._hp.MAX_HELPER_CONTRIBUTION_TOKENS]
        seen: Set[str] = set()
        deduped: List[str] = []
        for t in helper_tokens:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        helper = " ".join(deduped)
        if len(helper) > self._hq.MAX_FRAGMENT_HELPER_LENGTH:
            helper = helper[:self._hq.MAX_FRAGMENT_HELPER_LENGTH].rstrip()
        return helper

    def _compute_confidence(
        self,
        user_count: int,
        ai_count: int,
        combined_count: int,
    ) -> float:
        """
        Identity confidence = seberapa solid kombinasi user + AI.

        Formula:
          base = combined_count / MAX_HELPER_CONTRIBUTION_TOKENS (capped 1.0)
          bonus = 0.1 jika kedua sisi (user & AI) berkontribusi
          clamp ke [0.0, 1.0]

        Artinya: fragment dengan evidence dari KEDUA sisi lebih dipercaya
        daripada fragment yang hanya didukung satu sisi (mis. user probe
        saja tanpa konfirmasi AI, atau AI hallucinate tanpa user signal).
        """
        max_tokens = max(1, self._hp.MAX_HELPER_CONTRIBUTION_TOKENS)
        base = min(1.0, combined_count / max_tokens)
        two_sided_bonus = 0.1 if (user_count > 0 and ai_count > 0) else 0.0
        return min(1.0, base + two_sided_bonus)

    def extract(
        self,
        user_text: str,
        assistant_text: str,
        topic_key: Optional[TopicKey] = None,
        signal_profile: Optional[SignalProfile] = None,
    ) -> FragmentIdentity:
        """
        Ekstrak identitas fragment dari user_text + assistant_text.

        Alur:
            1. Tokenize user text (primary evidence)
            2. Tokenize AI text (enricher)
            3. Filter low-information dari keduanya
            4. Merge dengan urutan: user tokens dulu, AI tokens yang baru
            5. Bangun fragment_key + helper_description dari merged list
            6. Hitung confidence

        Jika topic_key tersedia, tokens yang muncul di topic_key.token_profile
        diangkat ke depan (prioritas) karena itu identitas topik aktif.
        """
        result = FragmentIdentity()

        # Ambil quality_map dari signal_profile jika ada.
        quality_map: Optional[Dict[str, TokenQuality]] = None
        if signal_profile is not None and signal_profile.token_quality:
            quality_map = signal_profile.token_quality

        # 1. Tokenize
        user_tokens_raw = self._tokenize(user_text)
        ai_tokens_raw = self._tokenize(assistant_text)

        # 2. Filter low-info
        user_tokens = self._filter_low_info(user_tokens_raw, quality_map)
        ai_tokens = self._filter_low_info(ai_tokens_raw, quality_map)

        result.user_token_count = len(user_tokens)
        result.ai_token_count = len(ai_tokens)

        # 3. Topic-aware reordering (jika topic_key tersedia).
        # Token yang ada di topic_key.profile mendapat prioritas.
        topic_tokens: Set[str] = set()
        if topic_key is not None and topic_key.token_profile:
            topic_tokens = set(topic_key.token_profile.keys())

        # 4. Merge: user first (preserving order), then new AI tokens.
        merged: List[str] = []
        seen: Set[str] = set()
        for pool in (user_tokens, ai_tokens):
            for t in pool:
                if t not in seen:
                    seen.add(t)
                    merged.append(t)

        # 5. Boost topic tokens ke depan (stable sort: pertahankan urutan asal).
        if topic_tokens:
            def key_fn(t: str) -> int:
                return 0 if t in topic_tokens else 1
            merged.sort(key=key_fn)

        result.meaningful_tokens = merged
        result.combined_token_count = len(merged)

        # 6. Ambil subset untuk representasi (representative_tokens).
        result.representative_tokens = merged[:5]

        # 7. Bangun fragment_key + helper_description
        result.fragment_key = self._build_fragment_key(merged)
        result.helper_description = self._build_helper_description(merged)

        # 8. Confidence
        result.identity_confidence = self._compute_confidence(
            result.user_token_count,
            result.ai_token_count,
            result.combined_token_count,
        )

        # 9. Reason (untuk observability)
        parts = []
        if result.user_token_count > 0:
            parts.append(f"user={result.user_token_count}")
        if result.ai_token_count > 0:
            parts.append(f"ai={result.ai_token_count}")
        parts.append(f"combined={result.combined_token_count}")
        parts.append(f"conf={result.identity_confidence:.2f}")
        result.reason = " | ".join(parts)

        return result
