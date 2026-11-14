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

[ASSOCIATIVE EVIDENCE LOCKING]
- Skor tinggi TIDAK otomatis berarti bukti kuat.
- Beberapa bukti yang saling mendukung > satu kecocokan besar.
- Bukti kontradiktif menurunkan kepercayaan.
- Dua dimensi: RELEVANCE (final_score) dan EVIDENCE STRENGTH (confidence).
"""

import re
import math
import logging
import calendar
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, Field

from config import CONFIG
from models import (
    SignalProfile, TopicKey, CandidateScore, FragmentMetadata, 
    DecisionType, PendingStatus, TokenQuality, EvidenceProfile
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
    # [ASSOCIATIVE EVIDENCE LOCKING] Metadata bukti per-fragment terpilih.
    # Dipakai Context Builder untuk mengurutkan berdasarkan kualitas bukti,
    # bukan sekadar "memori terbaru".
    selected_fragment_scores: Dict[str, dict] = Field(default_factory=dict)
    # Profil sinyal完整 agar Engine dapat memvalidasi mutasi memori.
    signal_profile: Optional[SignalProfile] = None
    # Ringkasan bukti kandidat terpilih (untuk transparansi / debugging).
    evidence_summary: str = ""

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

    # [ASSOCIATIVE EVIDENCE LOCKING] Marker scaffolding sistem, BUKAN konten user.
    # Jika tidak di-strip, kata "clarification" menjadi sinyal palsu dan
    # mendilusi token score pada jalur klarifikasi.
    _SCAFFOLDING_PATTERN = re.compile(r'\[clarification\s*:?', re.IGNORECASE)

    def _strip_scaffolding(self, text: str) -> str:
        """Membuang penanda internal sistem sebelum tokenisasi."""
        cleaned = self._SCAFFOLDING_PATTERN.sub(" ", text)
        return cleaned.replace("]", " ")

    def _classify_token(self, token: str) -> TokenQuality:
        """
        [ASSOCIATIVE EVIDENCE LOCKING] Menentukan tier kualitas token.

        "bagaimana" != "kopi" != "rp50"

        Catatan: tier RARE_SIGNAL ditentukan di Scorer (butuh Document Frequency
        global). Di extractor, token dikelompokkan berdasarkan bentuk & fungsi.
        """
        tq = CONFIG.token_quality
        if token in tq.LOW_INFORMATION_TOKENS:
            return TokenQuality.LOW_INFORMATION
        if re.match(tq.NUMERIC_IDENTIFIER_PATTERN, token):
            return TokenQuality.NUMERIC_IDENTIFIER
        if len(token) >= tq.MIN_TOPIC_SIGNAL_LENGTH:
            return TokenQuality.TOPIC_SIGNAL
        return TokenQuality.NORMAL

    def token_weight(self, token: str, quality_map: Dict[str, TokenQuality]) -> float:
        """Bobot kualitas token. Dipakai Scorer agar numerator/denominator konsisten."""
        tq = CONFIG.token_quality
        quality = quality_map.get(token, TokenQuality.NORMAL)
        if quality == TokenQuality.LOW_INFORMATION:
            return tq.LOW_INFORMATION_WEIGHT
        if quality == TokenQuality.NUMERIC_IDENTIFIER:
            return tq.NUMERIC_IDENTIFIER_WEIGHT
        if quality == TokenQuality.RARE_SIGNAL:
            return tq.RARE_SIGNAL_WEIGHT
        if quality == TokenQuality.TOPIC_SIGNAL:
            return tq.TOPIC_SIGNAL_WEIGHT
        return tq.NORMAL_WEIGHT

    def extract(self, text: str, reference_time: datetime) -> SignalProfile:
        """Mengekstrak token dan sinyal temporal dari teks."""
        if not text or not text.strip():
            return SignalProfile(confidence=0.0)

        # [ASSOCIATIVE EVIDENCE LOCKING] Strip marker sistem dulu.
        clean_text = self._strip_scaffolding(text)

        # 1. Tokenization & Normalization
        raw_tokens = self._TOKEN_PATTERN.findall(clean_text.lower())
        # Filter token terlalu pendek (misal: "di", "ke", "yg")
        normalized_tokens = [t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
        
        # 2. Temporal Detection
        temporal_signals = []
        text_lower = clean_text.lower()
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

        # [ASSOCIATIVE EVIDENCE LOCKING] 4b. WINDOW CO-OCCURRENCE (Level 2).
        # "bisnis kecil di bidang kopi" harus tetap memperkuat bisnis↔kopi
        # walaupun tidak berdampingan langsung.
        window_signals, pair_strength = self._build_window_co_occurrence(normalized_tokens)

        # [ASSOCIATIVE EVIDENCE LOCKING] 5. Kualitas token per-token.
        token_quality = {t: self._classify_token(t) for t in unique_terms}
        numeric_tokens = [t for t, q in token_quality.items() if q == TokenQuality.NUMERIC_IDENTIFIER]
        low_info_tokens = [t for t, q in token_quality.items() if q == TokenQuality.LOW_INFORMATION]

        # [NEW] 5. Topic Signals (Untuk PoC leksikal, sama dengan unique terms)
        # [UPDATED] Topic signal kini hanya token yang BUKAN low-information.
        topic_signals = [t for t in unique_terms if token_quality[t] != TokenQuality.LOW_INFORMATION]
        if not topic_signals:
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
            confidence=confidence,
            token_quality=token_quality,
            numeric_tokens=numeric_tokens,
            rare_tokens=[], # Diisi Scorer/Engine setelah DF global diketahui
            low_information_tokens=low_info_tokens,
            window_co_occurrence_signals=window_signals,
            window_pair_strength=pair_strength
        )

    def _build_window_co_occurrence(self, tokens: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        [ASSOCIATIVE EVIDENCE LOCKING] Co-occurrence berbasis window dengan decay.

        distance=1 -> bobot penuh (ADJACENT_PAIR_BONUS)
        distance=d -> DECAY ** (d-1)

        Pasangan di bawah CO_OCCURRENCE_MIN_STRENGTH dibuang.
        """
        cc = CONFIG.co_occurrence
        signals: List[str] = []
        strengths: Dict[str, float] = {}

        for i, token_a in enumerate(tokens):
            window_end = min(len(tokens), i + cc.CO_OCCURRENCE_WINDOW_SIZE + 1)
            for j in range(i + 1, window_end):
                token_b = tokens[j]
                if token_a == token_b:
                    continue
                distance = j - i
                if distance == 1:
                    strength = cc.ADJACENT_PAIR_BONUS
                else:
                    strength = cc.ADJACENT_PAIR_BONUS * (cc.CO_OCCURRENCE_DECAY ** (distance - 1))
                if strength < cc.CO_OCCURRENCE_MIN_STRENGTH:
                    continue
                # Canonical ordering agar "a↔b" dan "b↔a" dianggap pasangan sama.
                left, right = sorted([token_a, token_b])
                pair = f"{left}↔{right}"
                if pair not in strengths:
                    signals.append(pair)
                    strengths[pair] = strength
                else:
                    strengths[pair] = max(strengths[pair], strength)

        return signals, strengths

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
        global_token_counts: Optional[dict] = None, # [NEW] Untuk Unique Token Evidence
        total_topic_keys: int = 0 # [ASSOCIATIVE EVIDENCE LOCKING] Untuk rarity log(TotalKeys/DF)
    ) -> CandidateScore:
        """
        Menghitung skor komposit.
        Rumus: S = (T * W) + (D * W) + (C * W) + (Cont * W) + (Temp * W) + (U * W) + (X * W)

        [ASSOCIATIVE EVIDENCE LOCKING]
        Selain final_score (Dimensi 1: RELEVANCE), dihitung juga:
        - support_count      : berapa JENIS bukti yang benar-benar mendukung
        - evidence_strength  : Dimensi 2, rata-rata nilai bukti pendukung
        - confidence         : evidence_strength setelah safety cap
        Penalti kontradiksi TIDAK dihitung di sini (butuh pandangan antar-kandidat),
        melainkan di RetrievalEngine._apply_contradiction_penalties().
        """
        if not signal.unique_terms:
            return CandidateScore(key_id=key.key_id, title=key.title, final_score=0.0)

        signal_tokens = set(signal.unique_terms)
        extractor = SignalExtractor()
        quality_map = signal.token_quality or {}

        # Bobot kualitas token (denominator konsisten untuk seluruh perhitungan)
        token_weights = {t: extractor.token_weight(t, quality_map) for t in signal_tokens}
        total_weight = sum(token_weights.values()) or 1.0

        # A. Token Score (Quality-Weighted Query Recall)
        # [UPDATED] Token low-information tidak lagi bernilai setara topik/identifier.
        key_tokens = set(key.token_profile.keys())
        intersection = signal_tokens.intersection(key_tokens)
        matched_weight = sum(token_weights[t] for t in intersection)
        token_score = matched_weight / total_weight if total_weight else 0.0

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
        # [UPDATED] Memakai window co-occurrence (Level 2) dengan bobot decay,
        # sehingga "bisnis kecil di bidang kopi" tetap cocok dengan bisnis↔kopi.
        cooc_score = 0.0
        if key.co_occurrence_profile:
            window_pairs = signal.window_pair_strength or {}
            if window_pairs:
                total_pair_strength = sum(window_pairs.values())
                matched_pair_strength = 0.0
                for pair, strength in window_pairs.items():
                    if pair in key.co_occurrence_profile:
                        matched_pair_strength += strength
                    else:
                        # Toleransi arah pasangan (profil lama mungkin tidak canonical)
                        parts = pair.split("↔")
                        if len(parts) == 2:
                            reversed_pair = f"{parts[1]}↔{parts[0]}"
                            if reversed_pair in key.co_occurrence_profile:
                                matched_pair_strength += strength
                if total_pair_strength > 0:
                    cooc_score = min(1.0, matched_pair_strength / total_pair_strength)
            else:
                matched_pairs = 0
                total_pairs = len(signal.co_occurrence_signals)
                for pair in signal.co_occurrence_signals:
                    if pair in key.co_occurrence_profile:
                        matched_pairs += 1
                
                if total_pairs > 0:
                    cooc_score = matched_pairs / total_pairs
                else:
                    total_w = sum(key.co_occurrence_profile.values())
                    matched_w = sum(key.co_occurrence_profile.get(t, 0) for t in signal_tokens)
                    if total_w > 0:
                        cooc_score = matched_w / total_w

        # D. Continuity Score
        cont_score = continuity_score if is_active_topic else 0.0
        
        # E. Temporal Score (Indikator sederhana adanya sinyal waktu)
        temporal_score = 1.0 if signal.temporal_signals else 0.0
        
        # [NEW] U. Unique Token Evidence (IDF-like rarity)
        # [ASSOCIATIVE EVIDENCE LOCKING] Rumus dinaikkan menjadi:
        #   rarity(t) = log((TotalKeys+1)/(DF(t)+1)) / log(TotalKeys+1)
        # dan HANYA dihitung atas token yang benar-benar cocok dengan key ini.
        # Sebelumnya token tak dikenal memberi skor tinggi ke SEMUA kandidat
        # (noise, bukan evidence identitas).
        unique_score = 0.0
        if global_token_counts and intersection:
            total_keys = max(total_topic_keys, len(global_token_counts) and total_topic_keys or 1, 1)
            denom = math.log(total_keys + 1.0)
            rarity_weight_sum = 0.0
            rarity_accum = 0.0
            for t in intersection:
                df = global_token_counts.get(t, 0)
                rarity = math.log((total_keys + 1.0) / (df + 1.0)) / denom if denom > 0 else 0.0
                rarity = max(0.0, min(1.0, rarity))
                w = token_weights[t]
                rarity_accum += rarity * w
                rarity_weight_sum += w
            if rarity_weight_sum > 0:
                unique_score = rarity_accum / rarity_weight_sum
        
        # [NEW] X. Cross-session Evidence
        x_score = cross_session_score

        # F. Hitung Final Score (Weighted Sum)
        w = CONFIG.activation
        contributions = {
            "token": token_score * w.TOKEN_WEIGHT,
            "description": description_score * w.DESCRIPTION_WEIGHT,
            "co_occurrence": cooc_score * w.CO_OCCURRENCE_WEIGHT,
            "continuity": cont_score * w.CONTINUITY_WEIGHT,
            "temporal": temporal_score * w.TEMPORAL_WEIGHT,
            "unique_token": unique_score * w.UNIQUE_TOKEN_WEIGHT,
            "graph": x_score * w.CROSS_SESSION_WEIGHT,
        }
        final = sum(contributions.values())

        # ==========================================
        # [ASSOCIATIVE EVIDENCE LOCKING] DIMENSI 2: EVIDENCE STRENGTH
        # ==========================================
        ev = CONFIG.evidence
        evidence_values = {
            "token": token_score,
            "description": description_score,
            "co_occurrence": cooc_score,
            "unique_token": unique_score,
            "temporal": temporal_score,
            "continuity": cont_score,
            "graph": x_score,
        }
        supporting_names = [name for name, val in evidence_values.items() if val >= ev.EVIDENCE_SUPPORT_THRESHOLD]
        support_count = len(supporting_names)

        # evidence_strength = rata-rata nilai bukti PENDUKUNG.
        # Sengaja TIDAK memakai final_score, karena final_score terdilusi oleh
        # 7 bobot sehingga continuity-only yang sah terlihat "lemah".
        if support_count > 0:
            evidence_strength = sum(evidence_values[n] for n in supporting_names) / support_count
        else:
            evidence_strength = 0.0

        # Deteksi dominasi satu bukti terhadap total positive evidence.
        total_contribution = sum(contributions.values())
        max_contribution = max(contributions.values()) if contributions else 0.0
        is_dominated = False
        if total_contribution > 0:
            is_dominated = (max_contribution / total_contribution) >= ev.EVIDENCE_DOMINANCE_RATIO

        # Safety cap: satu skor besar tidak boleh menghasilkan kepercayaan palsu.
        safety = CONFIG.candidate_safety
        confidence = evidence_strength
        if support_count <= 1:
            confidence = min(confidence, safety.SINGLE_SIGNAL_CONFIDENCE_CAP)
        elif support_count == 2:
            confidence = min(confidence, safety.DUAL_SIGNAL_CONFIDENCE_CAP)

        evidence_profile = EvidenceProfile(
            token_score=token_score,
            description_score=description_score,
            co_occurrence_score=cooc_score,
            unique_token_score=unique_score,
            temporal_score=temporal_score,
            continuity_score=cont_score,
            graph_score=x_score,
            supporting_evidence_count=support_count,
            contradiction_count=0,
            supporting_evidence_names=supporting_names,
            contradiction_notes=[],
            evidence_strength=evidence_strength,
            is_single_signal_dominated=is_dominated
        )

        notes = []
        if support_count <= 1:
            notes.append(f"single-signal capped at {safety.SINGLE_SIGNAL_CONFIDENCE_CAP:.2f}")
        if is_dominated:
            notes.append("single-evidence dominated")

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
            final_score=final,
            graph_score=x_score,
            evidence_strength=evidence_strength,
            support_count=support_count,
            contradiction_score=0.0,
            confidence=confidence,
            evidence_profile=evidence_profile,
            evidence_notes=notes
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

        [ASSOCIATIVE EVIDENCE LOCKING]
        2b. Contradiction Penalty (relatif antar-kandidat)
        3b. Evidence Validation Gate (support / contradiction / confidence)
        """
        
        # 1. Extract Signal
        signal = self.extractor.extract(input_text, reference_time)
        if signal.confidence < CONFIG.signal.MIN_SIGNAL_STRENGTH:
            return RetrievalResult(
                decision_type=DecisionType.NO_MATCH,
                analysis_notes="Sinyal terlalu lemah / noise.",
                signal_profile=signal
            )

        # 2. Score Candidates
        candidates = []
        key_map = {k.key_id: k for k in keys} # [NEW] Untuk deskripsi klarifikasi
        
        if cross_session_evidence is None:
            cross_session_evidence = {}

        # [ASSOCIATIVE EVIDENCE LOCKING] Total Topic Key untuk rarity global.
        total_topic_keys = len(keys)

        for key in keys:
            is_active = (key.key_id == active_topic_id)
            x_score = cross_session_evidence.get(key.key_id, 0.0)
            score = self.scorer.score(
                signal, key, is_active, continuity_score, x_score, global_token_counts,
                total_topic_keys=total_topic_keys
            )
            if score.final_score >= CONFIG.decision.MINIMUM_RETRIEVAL_SCORE:
                candidates.append(score)

        if not candidates:
            return RetrievalResult(
                decision_type=DecisionType.NO_MATCH,
                analysis_notes="Tidak ada Topic Key yang melewati batas minimum.",
                signal_profile=signal
            )

        # 2b. [ASSOCIATIVE EVIDENCE LOCKING] CONTRADICTION PENALTY (relatif).
        # Kontradiksi = "ada bukti kuat bahwa kandidat ini BUKAN yang dimaksud".
        # Karena itu penalti hanya dipicu bila ada kandidat lain yang
        # menjelaskan intent dengan lebih baik. Tanpa pembanding, deskripsi
        # pendek tidak boleh dianggap kontradiksi.
        temporal_range = self.temporal_resolver.resolve(signal.temporal_signals, reference_time)
        self._apply_contradiction_penalties(candidates, fragments, temporal_range)

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
        if decision_type == DecisionType.AMBIGUOUS and temporal_range:
            resolved_key_id = self._try_temporal_resolution(
                candidates, fragments, temporal_range
            )
            if resolved_key_id:
                decision_type = DecisionType.CLEAR
                selected_key_id = resolved_key_id
                notes += " | Resolved by Temporal Evidence."
                clarification_message = "" # Reset karena sudah resolved
                top_candidate = next((c for c in candidates if c.key_id == resolved_key_id), top_candidate)

        # 4b. [ASSOCIATIVE EVIDENCE LOCKING] EVIDENCE VALIDATION GATE.
        # Hanya diterapkan pada jalur CLEAR agar semantics AMBIGUOUS tetap sama.
        if decision_type == DecisionType.CLEAR:
            gate_decision, gate_reason = self._validate_evidence(top_candidate)
            if gate_decision is not None:
                decision_type = gate_decision
                selected_key_id = None
                notes = gate_reason
                clarification_message = self._build_evidence_clarification(gate_decision, candidates, key_map)

        # 5. Fragment Selection (Jika CLEAR)
        selected_fragments = []
        fragment_scores: Dict[str, dict] = {}
        if decision_type == DecisionType.CLEAR and selected_key_id:
            selected_fragments, fragment_scores = self._select_fragments(
                selected_key_id, fragments, temporal_range, signal, reference_time
            )

        evidence_summary = self._build_evidence_summary(top_candidate)

        return RetrievalResult(
            decision_type=decision_type,
            candidates=candidates,
            selected_key_id=selected_key_id,
            selected_fragment_ids=selected_fragments,
            temporal_constraint=str(temporal_range) if temporal_range else None,
            analysis_notes=notes,
            clarification_message=clarification_message,
            selected_fragment_scores=fragment_scores,
            signal_profile=signal,
            evidence_summary=evidence_summary
        )

    # ==========================================
    # [ASSOCIATIVE EVIDENCE LOCKING] HELPERS
    # ==========================================

    def _apply_contradiction_penalties(
        self,
        candidates: List[CandidateScore],
        fragments: List[FragmentMetadata],
        temporal_range: Optional[Tuple[datetime, datetime]]
    ) -> None:
        """
        Menghitung POSITIVE EVIDENCE - CONTRADICTION PENALTY.

        Dua jenis kontradiksi:
        1. DESCRIPTION: token cocok (mendukung) TAPI deskripsi hampir tidak cocok,
           SEMENTARA ada kandidat lain yang deskripsinya menjelaskan lebih baik.
           Contoh: "lanjutkan bisnis kopi" vs "Kedai kopi terdekat".
        2. TEMPORAL: user menyebut waktu spesifik, kandidat punya fragment tetapi
           TIDAK SATUPUN berada di rentang itu, sementara kandidat lain punya.
        """
        ev = CONFIG.evidence
        if not candidates:
            return

        best_description = max(c.description_score for c in candidates)
        has_better_explainer = best_description >= ev.EVIDENCE_SUPPORT_THRESHOLD

        # Cek apakah ada kandidat yang cocok secara temporal.
        temporal_matches: Dict[str, int] = {}
        temporal_totals: Dict[str, int] = {}
        if temporal_range:
            start_t, end_t = temporal_range
            ref_tz = start_t.tzinfo
            for cand in candidates:
                key_frags = [f for f in fragments if f.topic_key_id == cand.key_id]
                temporal_totals[cand.key_id] = len(key_frags)
                count = 0
                for f in key_frags:
                    if f.temporal_index:
                        t_idx = f.temporal_index
                        if t_idx.tzinfo is None:
                            t_idx = t_idx.replace(tzinfo=ref_tz)
                        if start_t <= t_idx <= end_t:
                            count += 1
                temporal_matches[cand.key_id] = count
        any_temporal_match = any(v > 0 for v in temporal_matches.values())

        for cand in candidates:
            penalty = 0.0
            notes: List[str] = []

            # 1. DESCRIPTION CONTRADICTION
            if (has_better_explainer
                    and cand.token_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
                    and cand.description_score < ev.DESCRIPTION_CONTRADICTION_THRESHOLD
                    and cand.description_score < best_description):
                penalty += ev.DESCRIPTION_CONTRADICTION_PENALTY
                notes.append(
                    f"description contradiction (-{ev.DESCRIPTION_CONTRADICTION_PENALTY:.2f}): "
                    f"token cocok tetapi intent deskripsi berbeda"
                )

            # 2. TEMPORAL CONTRADICTION
            if (temporal_range is not None
                    and any_temporal_match
                    and temporal_totals.get(cand.key_id, 0) > 0
                    and temporal_matches.get(cand.key_id, 0) == 0):
                penalty += ev.TEMPORAL_CONTRADICTION_PENALTY
                notes.append(
                    f"temporal contradiction (-{ev.TEMPORAL_CONTRADICTION_PENALTY:.2f}): "
                    f"tidak ada fragment di rentang waktu yang diminta"
                )

            if penalty > 0:
                cand.contradiction_score = min(1.0, penalty)
                cand.final_score = max(0.0, cand.final_score - penalty)
                cand.confidence = max(0.0, cand.confidence - penalty)
                cand.evidence_notes.extend(notes)
                if cand.evidence_profile:
                    cand.evidence_profile.contradiction_count = len(notes)
                    cand.evidence_profile.contradiction_notes = notes

    def _validate_evidence(self, candidate: CandidateScore) -> Tuple[Optional[DecisionType], str]:
        """
        EVIDENCE VALIDATION.

        Membedakan tiga kondisi yang TIDAK sama:
        - INSUFFICIENT_EVIDENCE  : tidak ada kandidat yang cukup terbukti
        - CONTRADICTORY_EVIDENCE : ada kecocokan tetapi bukti saling bertentangan
        - AMBIGUOUS              : ditangani terpisah (margin antar kandidat)
        """
        ev = CONFIG.evidence

        # 1. Kontradiksi tinggi memblokir walaupun skor akhir tinggi.
        if candidate.contradiction_score > ev.MAX_CONTRADICTION_SCORE:
            return (
                DecisionType.CONTRADICTORY_EVIDENCE,
                f"Bukti kontradiktif pada '{candidate.title}' "
                f"(contradiction={candidate.contradiction_score:.2f} > {ev.MAX_CONTRADICTION_SCORE:.2f})."
            )

        # 2. Support count minimum.
        if candidate.support_count < ev.MIN_SUPPORTING_EVIDENCE:
            continuity_exemption = (
                ev.CONTINUITY_EXEMPT_FROM_MIN_SUPPORT
                and candidate.continuity_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_exemption:
                return (
                    DecisionType.INSUFFICIENT_EVIDENCE,
                    f"Bukti pendukung tidak cukup pada '{candidate.title}' "
                    f"(support={candidate.support_count} < {ev.MIN_SUPPORTING_EVIDENCE})."
                )

        # 3. Confidence minimum setelah safety cap.
        if candidate.confidence < ev.MIN_ASSOCIATIVE_CONFIDENCE:
            return (
                DecisionType.INSUFFICIENT_EVIDENCE,
                f"Confidence asosiatif tidak cukup pada '{candidate.title}' "
                f"(confidence={candidate.confidence:.2f} < {ev.MIN_ASSOCIATIVE_CONFIDENCE:.2f})."
            )

        return (None, "")

    def _build_evidence_clarification(
        self,
        decision_type: DecisionType,
        candidates: List[CandidateScore],
        key_map: Dict[str, TopicKey]
    ) -> str:
        """Pesan klarifikasi yang menjelaskan MENGAPA sistem menolak memilih."""
        if decision_type == DecisionType.CONTRADICTORY_EVIDENCE:
            return (
                "Ada kecocokan kata, tetapi bukti saling bertentangan dengan topik yang saya miliki. "
                "Bisa jelaskan lebih spesifik topik mana yang Anda maksud?"
            )
        if decision_type == DecisionType.INSUFFICIENT_EVIDENCE:
            parts = ["Bukti yang saya miliki belum cukup untuk memilih satu topik."]
            if candidates:
                parts.append("Kandidat terdekat:")
                for i, cand in enumerate(candidates[:3], 1):
                    desc = key_map[cand.key_id].helper_description if cand.key_id in key_map else "pembahasan terkait."
                    parts.append(f"{i}. {cand.title} — {desc}")
            parts.append("\nBisa tambahkan detail yang lebih spesifik?")
            return "\n".join(parts)
        return ""

    def _build_evidence_summary(self, candidate: Optional[CandidateScore]) -> str:
        """Ringkasan bukti untuk audit: 'KENAPA kandidat ini dipilih'."""
        if candidate is None:
            return ""
        lines = [
            f"TOP CANDIDATE: {candidate.title}",
            f"  Relevance (final_score) : {candidate.final_score:.3f}",
            f"  Evidence Strength       : {candidate.evidence_strength:.3f}",
            f"  Confidence (capped)     : {candidate.confidence:.3f}",
            f"  Support Count           : {candidate.support_count}",
            f"  Contradiction           : {candidate.contradiction_score:.3f}",
            "  Evidence:",
            f"    Token          {candidate.token_score:.3f}",
            f"    Description    {candidate.description_score:.3f}",
            f"    Co-occurrence  {candidate.co_occurrence_score:.3f}",
            f"    Unique Token   {candidate.unique_token_score:.3f}",
            f"    Temporal       {candidate.temporal_score:.3f}",
            f"    Continuity     {candidate.continuity_score:.3f}",
            f"    Graph          {candidate.graph_score:.3f}",
        ]
        if candidate.evidence_profile:
            lines.append(f"  Supporting: {', '.join(candidate.evidence_profile.supporting_evidence_names) or '(none)'}")
        if candidate.evidence_notes:
            lines.append(f"  Notes: {'; '.join(candidate.evidence_notes)}")
        return "\n".join(lines)

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
    ) -> Tuple[List[str], Dict[str, dict]]:
        """
        [CRITICAL FIX] True Fragment-Level Ranking (Review Kedua).
        Memilih ID fragment terbaik berdasarkan relevansi query-to-fragment yang sesungguhnya,
        membandingkan signal tokens dengan fragment's own token_profile.

        [ASSOCIATIVE EVIDENCE LOCKING] Mengembalikan JUGA metadata bukti per-fragment
        agar Context Builder dapat mengurutkan berdasarkan kualitas bukti,
        bukan sekadar recency.
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
            return [], {}

        # Fragment-Level Ranking Calculation
        ranked_frags = []
        fragment_scores: Dict[str, dict] = {}
        w_frag = CONFIG.fragment_ranking
        signal_tokens = set(signal.unique_terms)
        extractor = SignalExtractor()
        quality_map = signal.token_quality or {}
        token_weights = {t: extractor.token_weight(t, quality_map) for t in signal_tokens}
        total_weight = sum(token_weights.values()) or 1.0
        
        for f in key_frags:
            # 1. TRUE Token Relevance (Query vs Fragment Token Profile)
            # [UPDATED] Quality-weighted agar token low-information tidak mendominasi.
            frag_tokens = set(f.token_profile.keys()) if f.token_profile else set()
            matched = signal_tokens.intersection(frag_tokens)
            frag_token_score = sum(token_weights[t] for t in matched) / total_weight if total_weight else 0.0
            
            # 2. TRUE Co-occurrence Relevance (Query vs Fragment Co-occurrence Profile)
            # [UPDATED] Memakai window pairs dengan bobot decay.
            frag_cooc_score = 0.0
            if f.co_occurrence_profile:
                window_pairs = signal.window_pair_strength or {}
                if window_pairs:
                    total_pair_strength = sum(window_pairs.values())
                    matched_strength = 0.0
                    for pair, strength in window_pairs.items():
                        if pair in f.co_occurrence_profile:
                            matched_strength += strength
                        else:
                            parts = pair.split("↔")
                            if len(parts) == 2 and f"{parts[1]}↔{parts[0]}" in f.co_occurrence_profile:
                                matched_strength += strength
                    if total_pair_strength > 0:
                        frag_cooc_score = min(1.0, matched_strength / total_pair_strength)
                elif signal.co_occurrence_signals:
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
            fragment_scores[f.fragment_id] = {
                "relevance_score": relevance,
                "token_score": frag_token_score,
                "co_occurrence_score": frag_cooc_score,
                "temporal_score": frag_temporal_score,
                "recency_score": frag_recency_score,
                "graph_score": f.relationship_strength,
                "exact_temporal_match": bool(time_range),
            }
            
        # Sort by relevance descending
        ranked_frags.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil maksimal N fragment
        limit = CONFIG.retrieval.MAX_RETRIEVED_FRAGMENTS
        selected = [fid for fid, score in ranked_frags[:limit]]
        selected_scores = {fid: fragment_scores[fid] for fid in selected}
        return selected, selected_scores
