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
    # [NEW] Ditambahkan sesuai Spesifikasi Penguatan untuk output klarifikasi informatif
    clarification_message: str = ""

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
        "tahun lalu": "LAST_YEAR", # [NEW] Review Kedua
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

        # [NEW] 2b. Specific Hour Detection (pukul 15:00, jam 3 sore, 15.30)
        hour_patterns = [
            r'pukul\s+(\d{1,2})(?::(\d{2}))?',
            r'jam\s+(\d{1,2})(?::(\d{2}))?',
            r'(\d{1,2})\.(\d{2})'
        ]
        for pattern in hour_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                hour = int(match.group(1))
                minute = int(match.group(2)) if match.group(2) else 0
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    if ("sore" in text_lower or "malam" in text_lower) and hour < 12: 
                        hour += 12
                    temporal_signals.append(f"HOUR_{hour:02d}_{minute:02d}")

        # 3. Unique Terms (untuk matching)
        unique_terms = list(set(normalized_tokens))
        
        # [NEW] 4. Co-occurrence Signals (Pasangan token berurutan untuk bukti asosiatif)
        co_occurrence_signals = []
        for i in range(len(normalized_tokens) - 1):
            pair = f"{normalized_tokens[i]}↔{normalized_tokens[i+1]}"
            co_occurrence_signals.append(pair)
            
        # [NEW] 5. Topic Signals (Untuk PoC leksikal, sama dengan unique terms)
        topic_signals = unique_terms[:]

        # Confidence sederhana: semakin banyak token unik, semakin tinggi (sampai batas)
        confidence = min(1.0, len(unique_terms) / 10.0)

        return SignalProfile(
            raw_tokens=raw_tokens,
            tokens=raw_tokens,
            normalized_tokens=normalized_tokens,
            unique_terms=unique_terms,
            unique_tokens=unique_terms,
            topic_signals=topic_signals,
            co_occurrence_signals=co_occurrence_signals,
            temporal_signals=temporal_signals,
            novel_tokens=[], # Akan dihitung di Engine berdasarkan global keys
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

        tz_info = ref_time.tzinfo
        start, end = None, None

        # Prioritas sinyal (jika ada banyak, ambil yang paling spesifik)
        # Urutan prioritas: TIME > DAY > WEEK > MONTH > YEAR
        if "YESTERDAY" in signals:
            target = ref_time - timedelta(days=1)
            start, end = self._get_day_range(target, tz_info)
            
        elif "TODAY" in signals or "EARLIER_TODAY" in signals:
            start, end = self._get_day_range(ref_time, tz_info)
            
        elif "LAST_WEEK" in signals:
            # Minggu lalu (Senin - Minggu)
            target = ref_time - timedelta(weeks=1)
            start, end = self._get_week_range(target, tz_info)
            
        elif "LAST_MONTH" in signals:
            # Bulan lalu
            target_month = ref_time.month - 1
            target_year = ref_time.year
            if target_month == 0:
                target_month = 12
                target_year -= 1
            start, end = self._get_month_range(target_year, target_month, tz_info)

        elif "LAST_YEAR" in signals:
            # [NEW] Tahun lalu
            target_year = ref_time.year - 1
            start, end = self._get_year_range(target_year, tz_info)

        if start and end:
            # [NEW] Specific Hour Narrowing
            for sig in signals:
                if sig.startswith("HOUR_"):
                    parts = sig.split("_")
                    target_hour = int(parts[1])
                    tolerance = CONFIG.temporal.TEMPORAL_HOUR_TOLERANCE
                    
                    start_h = max(0, target_hour - tolerance)
                    end_h = min(23, target_hour + tolerance)
                    
                    start = start.replace(hour=start_h, minute=0, second=0, microsecond=0)
                    end = end.replace(hour=end_h, minute=59, second=59, microsecond=999999)
                    
            return start, end

        return None

    def _get_day_range(self, dt: datetime, tz_info) -> Tuple[datetime, datetime]:
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz_info)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=tz_info)
        return start, end

    def _get_week_range(self, dt: datetime, tz_info) -> Tuple[datetime, datetime]:
        # Asumsi minggu mulai Senin (0)
        start = dt - timedelta(days=dt.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz_info)
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return start, end

    def _get_month_range(self, year: int, month: int, tz_info) -> Tuple[datetime, datetime]:
        # [FIX] Bug timezone-aware datetime
        start = datetime(year, month, 1, tzinfo=tz_info)
        last_day = calendar.monthrange(year, month)[1]
        end = datetime(year, month, last_day, 23, 59, 59, tzinfo=tz_info)
        return start, end

    def _get_year_range(self, year: int, tz_info) -> Tuple[datetime, datetime]:
        # [NEW] Tahun lalu range
        start = datetime(year, 1, 1, tzinfo=tz_info)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=tz_info)
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
        continuity_score: float,
        cross_session_score: float = 0.0, # [NEW] Bukti lintas sesi (Layer 16/Graph)
        global_token_counts: Optional[dict] = None # [NEW] Untuk Unique Token Evidence
    ) -> CandidateScore:
        """
        Menghitung skor komposit.
        Rumus: S = (T * W) + (D * W) + (C * W) + (Cont * W) + (Temp * W) + (U * W) + (X * W)
        """
        if not signal.unique_terms:
            return CandidateScore(key_id=key.key_id, title=key.title, final_score=0.0)

        signal_tokens = set(signal.unique_terms)

        # A. Token Score (Query Recall)
        key_tokens = set(key.token_profile.keys())
        intersection = signal_tokens.intersection(key_tokens)
        token_score = len(intersection) / len(signal_tokens) if signal_tokens else 0.0

        # B. Description Score
        # [FIX] Hitung rasio irisan token deskripsi agar tidak bias pada kata umum seperti "kopi"
        desc_tokens = set(re.findall(r'\b\w+\b', key.helper_description.lower()))
        desc_tokens = {t for t in desc_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
        if desc_tokens:
            desc_intersection = signal_tokens.intersection(desc_tokens)
            description_score = len(desc_intersection) / len(desc_tokens)
        else:
            description_score = 0.0

        # C. Co-occurrence Score
        cooc_score = 0.0
        if key.co_occurrence_profile:
            matched_pairs = 0
            total_pairs = len(signal.co_occurrence_signals)
            for pair in signal.co_occurrence_signals:
                if pair in key.co_occurrence_profile:
                    matched_pairs += 1
            
            if total_pairs > 0:
                cooc_score = matched_pairs / total_pairs
            else:
                total_weight = sum(key.co_occurrence_profile.values())
                matched_weight = sum(key.co_occurrence_profile.get(t, 0) for t in signal_tokens)
                if total_weight > 0:
                    cooc_score = matched_weight / total_weight

        # D. Continuity Score
        cont_score = continuity_score if is_active_topic else 0.0
        
        # E. Temporal Score (Indikator sederhana adanya sinyal waktu)
        temporal_score = 1.0 if signal.temporal_signals else 0.0
        
        # [NEW] U. Unique Token Evidence (IDF-like rarity)
        unique_score = 0.0
        if global_token_counts and signal_tokens:
            rarity_sum = 0.0
            for t in signal_tokens:
                global_freq = global_token_counts.get(t, 0)
                # Inverse frequency: rarer tokens get higher score
                rarity_sum += 1.0 / (global_freq + 1.0)
            unique_score = min(1.0, rarity_sum / len(signal_tokens))
        
        # [NEW] X. Cross-session Evidence
        x_score = cross_session_score

        # F. Hitung Final Score (Weighted Sum)
        w = CONFIG.activation
        final = (
            (token_score * w.TOKEN_WEIGHT) +
            (description_score * w.DESCRIPTION_WEIGHT) +
            (cooc_score * w.CO_OCCURRENCE_WEIGHT) +
            (cont_score * w.CONTINUITY_WEIGHT) +
            (temporal_score * w.TEMPORAL_WEIGHT) +
            (unique_score * w.UNIQUE_TOKEN_WEIGHT) +
            (x_score * w.CROSS_SESSION_WEIGHT)
        )

        return CandidateScore(
            key_id=key.key_id,
            title=key.title,
            token_score=token_score,
            description_score=description_score,
            co_occurrence_score=cooc_score,
            continuity_score=cont_score,
            temporal_score=temporal_score,
            unique_token_score=unique_score,
            cross_session_score=x_score,
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
        continuity_score: float = 0.0,
        global_token_counts: Optional[dict] = None, # [NEW] Untuk Unique Token Evidence
        cross_session_evidence: Optional[dict] = None # [NEW] topic_id -> score
    ) -> RetrievalResult:
        """
        Alur Retrieval Lengkap:
        1. Extract Signal
        2. Score Keys
        3. Check Ambiguity
        4. Temporal Disambiguation (jika perlu)
        5. Select Fragments (Fragment-Level Ranking)
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
        key_map = {k.key_id: k for k in keys} # [NEW] Untuk deskripsi klarifikasi
        
        if cross_session_evidence is None:
            cross_session_evidence = {}

        for key in keys:
            is_active = (key.key_id == active_topic_id)
            x_score = cross_session_evidence.get(key.key_id, 0.0)
            score = self.scorer.score(
                signal, key, is_active, continuity_score, x_score, global_token_counts
            )
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
        clarification_message = ""

        # [FIX] Penegakan MIN_TOPIC_ACTIVATION_SCORE (Mencegah AMBIGUOUS pada skor yang terlalu lemah)
        if top_candidate.final_score < CONFIG.decision.MIN_TOPIC_ACTIVATION_SCORE:
            decision_type = DecisionType.NO_MATCH
            selected_key_id = None
            notes = "Skor kandidat tertinggi terlalu lemah (di bawah MIN_TOPIC_ACTIVATION_SCORE)."
            clarification_message = ""
        elif second_candidate:
            margin = top_candidate.final_score - second_candidate.final_score
            if margin < CONFIG.decision.AMBIGUITY_MARGIN:
                decision_type = DecisionType.AMBIGUOUS
                selected_key_id = None # Belum boleh dipilih
                notes = f"Ambigu: {top_candidate.title} vs {second_candidate.title} (Diff: {margin:.2f})"
                
                # [NEW] 6. Informative Clarification Output
                msg_parts = ["Saya menemukan beberapa topik yang mungkin Anda maksud:\n"]
                for i, cand in enumerate(candidates[:3], 1):
                    desc = key_map[cand.key_id].helper_description if cand.key_id in key_map else "pembahasan terkait."
                    msg_parts.append(f"{i}. {cand.title} — {desc}")
                msg_parts.append("\nYang mana yang ingin Anda lanjutkan?")
                clarification_message = "\n".join(msg_parts)

        # 4. Temporal Disambiguation (Layer 6)
        temporal_range = self.temporal_resolver.resolve(signal.temporal_signals, reference_time)
        
        if decision_type == DecisionType.AMBIGUOUS and temporal_range:
            resolved_key_id = self._try_temporal_resolution(
                candidates, fragments, temporal_range
            )
            if resolved_key_id:
                decision_type = DecisionType.CLEAR
                selected_key_id = resolved_key_id
                notes += " | Resolved by Temporal Evidence."
                clarification_message = "" # Reset karena sudah resolved

        # 5. Fragment Selection (Jika CLEAR)
        selected_fragments = []
        if decision_type == DecisionType.CLEAR and selected_key_id:
            selected_fragments = self._select_fragments(
                selected_key_id, fragments, temporal_range, signal, reference_time
            )

        return RetrievalResult(
            decision_type=decision_type,
            candidates=candidates,
            selected_key_id=selected_key_id,
            selected_fragment_ids=selected_fragments,
            temporal_constraint=str(temporal_range) if temporal_range else None,
            analysis_notes=notes,
            clarification_message=clarification_message
        )

    def _try_temporal_resolution(
        self, 
        candidates: List[CandidateScore], 
        fragments: List[FragmentMetadata],
        time_range: Tuple[datetime, datetime]
    ) -> Optional[str]:
        """
        [CRITICAL FIX] Temporal Quality Scoring.
        Tidak lagi hanya menghitung jumlah fragment (count-based).
        Menggunakan base score kandidat + temporal match sebagai quality score.
        """
        start_t, end_t = time_range
        ref_tz = start_t.tzinfo # [FIX] Ambil timezone dari range
        
        key_scores = {}

        for cand in candidates:
            # Cari fragment milik key ini
            key_frags = [f for f in fragments if f.topic_key_id == cand.key_id]
            match_count = 0
            for f in key_frags:
                if f.temporal_index:
                    t_idx = f.temporal_index
                    # [FIX] Normalisasi timezone (SQLite mengembalikan naive datetime)
                    if t_idx.tzinfo is None:
                        t_idx = t_idx.replace(tzinfo=ref_tz)
                    if start_t <= t_idx <= end_t:
                        match_count += 1
            
            # Quality Score: Base final score + bonus jika ada fragment yang cocok secara waktu
            # Ini memastikan key dengan base score tinggi tidak kalah hanya karena key lain punya lebih banyak fragment acak di waktu tersebut.
            quality_score = cand.final_score + (match_count * 0.5)
            key_scores[cand.key_id] = (quality_score, match_count)

        if not key_scores:
            return None

        # Urutkan berdasarkan quality score tertinggi
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1][0], reverse=True)
        best_key, (best_q_score, best_count) = sorted_keys[0]
        
        # Hanya anggap resolved jika ada minimal 1 fragment yang cocok secara waktu
        if best_count > 0:
            if len(sorted_keys) > 1:
                second_best_key, (second_q_score, second_count) = sorted_keys[1]
                # Jika quality score terlalu dekat, tetap anggap ambigu (Temporal Tie)
                if abs(best_q_score - second_q_score) < CONFIG.decision.AMBIGUITY_MARGIN and second_count > 0:
                    return None
            return best_key
        
        return None

    def _select_fragments(
        self, 
        key_id: str, 
        fragments: List[FragmentMetadata],
        time_range: Optional[Tuple[datetime, datetime]],
        signal: SignalProfile, # [NEW] Untuk fragment relevance scoring
        reference_time: datetime # [NEW] Untuk recency scoring
    ) -> List[str]:
        """
        [CRITICAL FIX] True Fragment-Level Ranking (Review Kedua).
        Memilih ID fragment terbaik berdasarkan relevansi query-to-fragment yang sesungguhnya,
        membandingkan signal tokens dengan fragment's own token_profile.
        """
        key_frags = [f for f in fragments if f.topic_key_id == key_id]
        
        # Filter Temporal ketat jika ada constraint waktu
        if time_range:
            start_t, end_t = time_range
            ref_tz = start_t.tzinfo
            filtered_frags = []
            for f in key_frags:
                if f.temporal_index:
                    t_idx = f.temporal_index
                    # [FIX] Normalisasi timezone
                    if t_idx.tzinfo is None:
                        t_idx = t_idx.replace(tzinfo=ref_tz)
                    if start_t <= t_idx <= end_t:
                        filtered_frags.append(f)
            key_frags = filtered_frags
        
        # Aturan Fail-Closed: Jika user minta "kemarin" dan tidak ada data "kemarin", return kosong.
        if time_range and not key_frags:
            return []

        # Fragment-Level Ranking Calculation
        ranked_frags = []
        w_frag = CONFIG.fragment_ranking
        signal_tokens = set(signal.unique_terms)
        
        for f in key_frags:
            # 1. TRUE Token Relevance (Query vs Fragment Token Profile)
            frag_tokens = set(f.token_profile.keys()) if f.token_profile else set()
            frag_token_score = len(signal_tokens.intersection(frag_tokens)) / len(signal_tokens) if signal_tokens else 0.0
            
            # 2. TRUE Co-occurrence Relevance (Query vs Fragment Co-occurrence Profile)
            frag_cooc_score = 0.0
            if f.co_occurrence_profile and signal.co_occurrence_signals:
                matched_pairs = sum(1 for p in signal.co_occurrence_signals if p in f.co_occurrence_profile)
                frag_cooc_score = matched_pairs / len(signal.co_occurrence_signals)
            
            # 3. Temporal Match (Sudah difilter di atas, tapi bisa diberi bobot tambahan jika exact match)
            frag_temporal_score = 1.0 if time_range else 0.5
            
            # 4. Recency (Decay berdasarkan umur fragment)
            frag_recency_score = 0.0
            if reference_time and f.created_at:
                created_at = f.created_at
                # [FIX] Normalisasi timezone
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=reference_time.tzinfo)
                # Normalisasi age: newer = higher score (decay 30 hari)
                age_hours = max(0, (reference_time - created_at).total_seconds() / 3600.0)
                frag_recency_score = max(0.0, 1.0 - (age_hours / 720.0)) 
                
            relevance = (
                (frag_token_score * w_frag.FRAGMENT_TOKEN_WEIGHT) +
                (frag_cooc_score * w_frag.FRAGMENT_COOC_WEIGHT) +
                (frag_temporal_score * w_frag.FRAGMENT_TEMPORAL_WEIGHT) +
                (frag_recency_score * w_frag.FRAGMENT_RECENCY_WEIGHT)
            )
            ranked_frags.append((f.fragment_id, relevance))
            
        # Sort by relevance descending
        ranked_frags.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil maksimal N fragment
        limit = CONFIG.retrieval.MAX_RETRIEVED_FRAGMENTS
        return [fid for fid, score in ranked_frags[:limit]]
