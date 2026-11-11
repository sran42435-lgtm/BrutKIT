"""
retrieval.py
Mesin Asosiatif, Temporal, & Ambiguity Gate (Layer 3, 4, 5, 6).

Fungsi:
1. Menerjemahkan bahasa manusia menjadi sinyal (Token, Waktu, Co-occurrence).
2. Mengaktivasi Topic Key berdasarkan bukti multi-lapis (Title, Desc, History).
3. Menilai ambiguitas (Ambiguity Gate) sebelum memilih memori.
4. Memfilter fragment berdasarkan constraint waktu (Temporal Resolution).

PRINSIP:
- Fail-Closed: Jika bukti lemah, jangan memilih (Return NO_MATCH).
- No Guessing: Jika selisih skor tipis, wajib flag AMBIGUOUS.
- Temporal as Evidence: Waktu adalah penguat bukti, bukan pengganti identifikasi topik.
"""

import re
import logging
import calendar
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, Field

from config import CONFIG
from models import (
    SignalProfile, TopicKey, CandidateScore, FragmentMetadata, 
    DecisionType, PendingStatus
)

logger = logging.getLogger(__name__)

# ==========================================
# INTERNAL MODELS (Retrieval Output)
# ==========================================
class RetrievalResult(BaseModel):
    """Hasil akhir dari mesin retrieval sebelum diproses Engine."""
    decision_type: DecisionType
    candidates: List[CandidateScore] = Field(default_factory=list)
    selected_key_id: Optional[str] = None
    selected_fragment_ids: List[str] = Field(default_factory=list)
    temporal_constraint: Optional[str] = None
    analysis_notes: str = ""

# ==========================================
# 1. SIGNAL EXTRACTION (LAYER 3)
# ==========================================
class SignalExtractor:
    """Mengubah teks mentah menjadi profil sinyal terstruktur."""
    
    # Pola dasar untuk tokenisasi (alfanumerik + underscore)
    _TOKEN_PATTERN = re.compile(r'\b\w+\b', re.UNICODE)
    
    # Kata kunci temporal (Indonesian context untuk PoC)
    _TEMPORAL_KEYWORDS = {
        "kemarin": "YESTERDAY",
        "hari ini": "TODAY",
        "besok": "TOMORROW",
        "minggu lalu": "LAST_WEEK",
        "bulan lalu": "LAST_MONTH",
        "tadi": "EARLIER_TODAY",
        "pagi": "MORNING",
        "siang": "AFTERNOON",
        "sore": "EVENING",
        "malam": "NIGHT"
    }

    def extract(self, text: str, reference_time: datetime) -> SignalProfile:
        """Mengekstrak token dan sinyal temporal dari teks."""
        if not text or not text.strip():
            return SignalProfile(confidence=0.0)

        # 1. Tokenization & Normalization
        raw_tokens = self._TOKEN_PATTERN.findall(text.lower())
        # Filter token terlalu pendek (misal: "di", "ke", "yg")
        normalized_tokens = [t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
        
        # 2. Temporal Detection
        temporal_signals = []
        text_lower = text.lower()
        for keyword, signal_type in self._TEMPORAL_KEYWORDS.items():
            if keyword in text_lower:
                temporal_signals.append(signal_type)

        # 3. Unique Terms (untuk matching)
        unique_terms = list(set(normalized_tokens))

        # Confidence sederhana: semakin banyak token unik, semakin tinggi (sampai batas)
        confidence = min(1.0, len(unique_terms) / 10.0)

        return SignalProfile(
            tokens=raw_tokens,
            normalized_tokens=normalized_tokens,
            unique_terms=unique_terms,
            temporal_signals=temporal_signals,
            confidence=confidence
        )

# ==========================================
# 2. TEMPORAL RESOLVER (LAYER 6)
# ==========================================
class TemporalResolver:
    """Menerjemahkan sinyal temporal menjadi rentang waktu (datetime range)."""

    def resolve(self, signals: List[str], ref_time: datetime) -> Optional[Tuple[datetime, datetime]]:
        """
        Mengembalikan (start_time, end_time) atau None jika tidak ada constraint.
        Menggunakan logika kalender (bukan sekadar -24 jam) untuk akurasi.
        """
        if not signals:
            return None

        # Prioritas sinyal (jika ada banyak, ambil yang paling spesifik)
        # Urutan prioritas: TIME > DAY > WEEK > MONTH
        if "YESTERDAY" in signals:
            target = ref_time - timedelta(days=1)
            return self._get_day_range(target)
            
        elif "TODAY" in signals or "EARLIER_TODAY" in signals:
            return self._get_day_range(ref_time)
            
        elif "LAST_WEEK" in signals:
            # Minggu lalu (Senin - Minggu)
            target = ref_time - timedelta(weeks=1)
            return self._get_week_range(target)
            
        elif "LAST_MONTH" in signals:
            # Bulan lalu
            target_month = ref_time.month - 1
            target_year = ref_time.year
            if target_month == 0:
                target_month = 12
                target_year -= 1
            return self._get_month_range(target_year, target_month)

        return None

    def _get_day_range(self, dt: datetime) -> Tuple[datetime, datetime]:
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    def _get_week_range(self, dt: datetime) -> Tuple[datetime, datetime]:
        # Asumsi minggu mulai Senin (0)
        start = dt - timedelta(days=dt.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return start, end

    def _get_month_range(self, year: int, month: int) -> Tuple[datetime, datetime]:
        start = datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end = datetime(year, month, last_day, 23, 59, 59)
        return start, end

# ==========================================
# 3. TOPIC KEY SCORER (LAYER 4)
# ==========================================
class TopicKeyScorer:
    """Menghitung skor relevansi Topic Key terhadap sinyal input."""

    def score(
        self, 
        signal: SignalProfile, 
        key: TopicKey, 
        is_active_topic: bool,
        continuity_score: float
    ) -> CandidateScore:
        """
        Menghitung skor komposit.
        Rumus: (Token * W) + (Desc * W) + (CoOcc * W) + (Cont * W)
        """
        if not signal.unique_terms:
            return CandidateScore(key_id=key.key_id, title=key.title, final_score=0.0)

        # A. Token Score (Query Recall)
        # [FIX] Mengubah dari Jaccard (intersection/union) menjadi Query Recall (intersection/query_len).
        # Ini mencegah query pendek dihukum secara tidak adil oleh union yang besar.
        key_tokens = set(key.token_profile.keys())
        signal_tokens = set(signal.unique_terms)
        intersection = signal_tokens.intersection(key_tokens)
        token_score = len(intersection) / len(signal_tokens) if signal_tokens else 0.0

        # B. Description Score
        # Cek apakah ada token sinyal yang muncul di deskripsi (bantuan semantik)
        desc_lower = key.helper_description.lower()
        desc_matches = sum(1 for t in signal_tokens if t in desc_lower)
        # Normalisasi: jika ada minimal 1 kata kunci utama cocok, skor tinggi
        description_score = 1.0 if desc_matches > 0 else 0.0

        # C. Co-occurrence Score
        # Cek profil co-occurrence (kata yang sering muncul bersama topik ini)
        cooc_score = 0.0
        if key.co_occurrence_profile:
            total_weight = sum(key.co_occurrence_profile.values())
            matched_weight = sum(key.co_occurrence_profile.get(t, 0) for t in signal_tokens)
            if total_weight > 0:
                cooc_score = matched_weight / total_weight

        # D. Continuity Score
        # Bonus jika ini adalah topik yang sedang aktif dan stabil
        cont_score = continuity_score if is_active_topic else 0.0

        # E. Hitung Final Score (Weighted Sum)
        w = CONFIG.activation
        final = (
            (token_score * w.TOKEN_WEIGHT) +
            (description_score * w.DESCRIPTION_WEIGHT) +
            (cooc_score * w.CO_OCCURRENCE_WEIGHT) +
            (cont_score * w.CONTINUITY_WEIGHT)
        )

        return CandidateScore(
            key_id=key.key_id,
            title=key.title,
            token_score=token_score,
            description_score=description_score,
            co_occurrence_score=cooc_score,
            continuity_score=cont_score,
            final_score=final
        )

# ==========================================
# 4. RETRIEVAL ENGINE (ORCHESTRATOR)
# ==========================================
class RetrievalEngine:
    """
    Mesin utama yang menggabungkan Signal, Scoring, dan Decision Gate.
    """

    def __init__(self):
        self.extractor = SignalExtractor()
        self.temporal_resolver = TemporalResolver()
        self.scorer = TopicKeyScorer()

    def retrieve(
        self,
        input_text: str,
        keys: List[TopicKey],
        fragments: List[FragmentMetadata],
        reference_time: datetime,
        active_topic_id: Optional[str] = None,
        continuity_score: float = 0.0
    ) -> RetrievalResult:
        """
        Alur Retrieval Lengkap:
        1. Extract Signal
        2. Score Keys
        3. Check Ambiguity
        4. Temporal Disambiguation (jika perlu)
        5. Select Fragments
        """
        
        # 1. Extract Signal
        signal = self.extractor.extract(input_text, reference_time)
        if signal.confidence < CONFIG.signal.MIN_SIGNAL_STRENGTH:
            return RetrievalResult(
                decision_type=DecisionType.NO_MATCH,
                analysis_notes="Sinyal terlalu lemah / noise."
            )

        # 2. Score Candidates
        candidates = []
        for key in keys:
            is_active = (key.key_id == active_topic_id)
            score = self.scorer.score(signal, key, is_active, continuity_score)
            if score.final_score >= CONFIG.decision.MINIMUM_RETRIEVAL_SCORE:
                candidates.append(score)

        if not candidates:
            return RetrievalResult(
                decision_type=DecisionType.NO_MATCH,
                analysis_notes="Tidak ada Topic Key yang melewati batas minimum."
            )

        # 3. Sort & Analyze Ambiguity
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        top_candidate = candidates[0]
        second_candidate = candidates[1] if len(candidates) > 1 else None
        
        decision_type = DecisionType.CLEAR
        selected_key_id = top_candidate.key_id
        notes = f"Top: {top_candidate.title} ({top_candidate.final_score:.2f})"

        if second_candidate:
            margin = top_candidate.final_score - second_candidate.final_score
            if margin < CONFIG.decision.AMBIGUITY_MARGIN:
                decision_type = DecisionType.AMBIGUOUS
                selected_key_id = None # Belum boleh dipilih
                notes = f"Ambigu: {top_candidate.title} vs {second_candidate.title} (Diff: {margin:.2f})"

        # 4. Temporal Disambiguation (Layer 6)
        # Jika AMBIGU tapi ada sinyal waktu, coba gunakan waktu untuk memecah kebuntuan
        temporal_range = self.temporal_resolver.resolve(signal.temporal_signals, reference_time)
        
        if decision_type == DecisionType.AMBIGUOUS and temporal_range:
            # Cek kandidat mana yang memiliki fragment di rentang waktu tersebut
            resolved_key_id = self._try_temporal_resolution(
                candidates, fragments, temporal_range
            )
            if resolved_key_id:
                decision_type = DecisionType.CLEAR
                selected_key_id = resolved_key_id
                notes += " | Resolved by Temporal Evidence."

        # 5. Fragment Selection (Jika CLEAR)
        selected_fragments = []
        if decision_type == DecisionType.CLEAR and selected_key_id:
            selected_fragments = self._select_fragments(
                selected_key_id, fragments, temporal_range
            )

        return RetrievalResult(
            decision_type=decision_type,
            candidates=candidates,
            selected_key_id=selected_key_id,
            selected_fragment_ids=selected_fragments,
            temporal_constraint=str(temporal_range) if temporal_range else None,
            analysis_notes=notes
        )

    def _try_temporal_resolution(
        self, 
        candidates: List[CandidateScore], 
        fragments: List[FragmentMetadata],
        time_range: Tuple[datetime, datetime]
    ) -> Optional[str]:
        """
        Mencoba menyelesaikan ambiguitas dengan memeriksa fragment mana 
        yang cocok dengan constraint waktu.
        """
        start_t, end_t = time_range
        matches_per_key = {}

        for cand in candidates:
            # Cari fragment milik key ini
            key_frags = [f for f in fragments if f.topic_key_id == cand.key_id]
            match_count = 0
            for f in key_frags:
                if f.temporal_index and (start_t <= f.temporal_index <= end_t):
                    match_count += 1
            matches_per_key[cand.key_id] = match_count

        # Cari key dengan match terbanyak
        if not matches_per_key:
            return None

        best_key = max(matches_per_key, key=matches_per_key.get)
        best_count = matches_per_key[best_key]

        # Hanya anggap resolved jika ada minimal 1 fragment yang cocok secara waktu
        # DAN key lain tidak memiliki jumlah match yang sama (unik)
        if best_count > 0:
            # Cek apakah ada pesaing dengan jumlah match yang sama
            competitors = [k for k, v in matches_per_key.items() if v == best_count and k != best_key]
            if not competitors:
                return best_key
        
        return None

    def _select_fragments(
        self, 
        key_id: str, 
        fragments: List[FragmentMetadata],
        time_range: Optional[Tuple[datetime, datetime]]
    ) -> List[str]:
        """Memilih ID fragment terbaik untuk key yang terpilih."""
        key_frags = [f for f in fragments if f.topic_key_id == key_id]
        
        # Filter Temporal jika ada
        if time_range:
            start_t, end_t = time_range
            key_frags = [
                f for f in key_frags 
                if f.temporal_index and (start_t <= f.temporal_index <= end_t)
            ]
        
        # Jika filter waktu terlalu ketat dan hasil kosong, fallback ke fragment terbaru?
        # Aturan Fail-Closed: Jika user minta "kemarin" dan tidak ada data "kemarin",
        # jangan berikan data "hari ini" secara diam-diam. Return kosong.
        if time_range and not key_frags:
            return []

        # Sort by time (terbaru dulu) atau relevansi
        key_frags.sort(key=lambda x: x.created_at, reverse=True)
        
        # Ambil maksimal N fragment
        limit = CONFIG.retrieval.MAX_RETRIEVED_FRAGMENTS
        return [f.fragment_id for f in key_frags[:limit]]
