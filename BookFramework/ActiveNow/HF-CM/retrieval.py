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

[REASONING PURITY]
- final_score (relevance kandidat) TIDAK BOLEH disamakan dengan
  turn_continuity_score (seberapa kuat turn ini melanjutkan state aktif).
- 3 sinyal lexical (token, unique, co-occurrence) = 1 family, bukan 3 evidence.
- Filter temporal ≠ ranking temporal; fragment di dalam range tetap dibedakan
  berdasarkan kedekatan waktu (distance-based decay).
- graph_score fragment dihitung dari edge nyata (CONTINUES / CROSS_SESSION_LINK),
  bukan dari relationship_strength metadata.

[HIERARCHICAL FRAGMENT EXPANSION]
- M1: Negation scope-aware + combined clarification evidence + candidate narrowing.
- M2: Continuation intent detection (translasi state ada di engine).
- M3/M5: Fragment-key & helper scoring untuk retrieval dan hierarchy placement.
"""

import re
import math
import logging
import calendar
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Set
from pydantic import BaseModel, Field

from config import CONFIG
from models import (
    SignalProfile, TopicKey, CandidateScore, FragmentMetadata, 
    DecisionType, PendingStatus, TokenQuality, EvidenceProfile, EvidenceFamily,
    NegationScope, ClarificationAccumulation, FragmentCandidateScore
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
    # Profil sinyal lengkap agar Engine dapat memvalidasi mutasi memori.
    signal_profile: Optional[SignalProfile] = None
    # Ringkasan bukti kandidat terpilih (untuk transparansi / debugging).
    evidence_summary: str = ""
    # [REASONING PURITY] Continuity score MURNI dari active state.
    # Berbeda dari final_score kandidat tertinggi. Dipakai oleh
    # memory_engine untuk relationship_strength persistence.
    turn_continuity_score: float = 0.0
    # Sumber bukti continuity (untuk audit: "overlap", "history", "fragments").
    continuity_evidence_sources: List[str] = Field(default_factory=list)
    # [MEKANISME 1] Kandidat yang dieliminasi oleh negative evidence.
    excluded_candidate_keys: List[str] = Field(default_factory=list)
    # Kandidat yang gugur karena narrowing (bukan karena exclusion).
    narrowed_out_keys: List[str] = Field(default_factory=list)
    # [MEKANISME 2] Intent melanjutkan terdeteksi pada input ini.
    # Retrieval HANYA melaporkan sinyal; translasi NO_MATCH →
    # UNRESOLVED_CONTINUATION dilakukan memory_engine.
    continuation_intent: bool = False
    # [MEKANISME 1] Explicit cancellation / topic switch terdeteksi.
    cancellation_intent: bool = False

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

    # [MEKANISME 1] Marker negation SCOPE-AWARE.
    #
    # "bukan cpp"  → cpp dieliminasi
    # "cpp bukan?" → TIDAK dieliminasi (marker di belakang, scope kosong)
    #
    # CATATAN PENTING: "tidak" SENGAJA tidak dimasukkan.
    # "Tidak, saya sekarang mau belajar C++" adalah penolakan terhadap
    # pertanyaan, bukan penolakan terhadap topik C++. Memasukkan "tidak"
    # akan menghasilkan false exclusion pada alur topic-switch yang sah.
    _NEGATION_MARKERS: Tuple[str, ...] = ("bukan", "jangan", "kecuali", "tanpa")

    # Pembatas scope negation: tanda baca & konjungsi.
    _SCOPE_STOP_PATTERN = re.compile(
        r'[,.;:!?\n]|\bdan\b|\batau\b|\btapi\b|\byang\b|\buntuk\b',
        re.IGNORECASE
    )

    # [MEKANISME 1] Identifier simbolik yang tidak ditangkap \b\w+\b.
    _SYMBOL_IDENTITY_PATTERN = re.compile(r"[a-z0-9_]+(?:\+\+|\+|#)+", re.IGNORECASE)

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

    # ==========================================
    # [MEKANISME 1] IDENTITY NORMALIZATION
    # ==========================================

    def normalize_identity(self, term: str) -> str:
        """
        Normalisasi identifier agar "cpp" dan "c++" dianggap sama.

        Tanpa ini, exclusion "bukan cpp" tidak akan pernah kena kandidat
        yang token-nya "c++" (karena \\b\\w+\\b pada "c++" hanya menangkap "c").
        """
        t = (term or "").lower().strip()
        if t in ("c++", "cpp"):
            return "cpp"
        if t in ("c#", "csharp"):
            return "csharp"
        if t in ("f#", "fsharp"):
            return "fsharp"
        return t

    def extract_identity_terms(self, text: str) -> Set[str]:
        """
        Kumpulkan term identitas dari teks, INCLUDING identifier simbolik.

        Dipakai untuk exclusion matching terhadap title/helper kandidat.
        """
        if not text:
            return set()
        lower = text.lower()
        terms: Set[str] = set()
        for t in self._TOKEN_PATTERN.findall(lower):
            if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH:
                terms.add(self.normalize_identity(t))
        for t in self._SYMBOL_IDENTITY_PATTERN.findall(lower):
            terms.add(self.normalize_identity(t))
        return terms

    # ==========================================
    # [MEKANISME 1] NEGATION SCOPE EXTRACTION
    # ==========================================

    def _extract_negation_scopes(
        self,
        clean_text: str,
    ) -> Tuple[List[NegationScope], List[str]]:
        """
        Ekstrak negative evidence DENGAN SCOPE.

        Aturan:
            1. Marker harus muncul SEBELUM target (scope = teks setelah marker).
            2. Scope berhenti di tanda baca / konjungsi pembatas.
            3. Hanya token bermakna (bukan low-information) yang jadi target.
            4. Jumlah target dibatasi NEGATION_SCOPE_TOKEN_WINDOW.

        Contoh:
            "yang penting bukan topik cpp" → targets {topik, cpp}
            "cpp bukan?"                   → scope kosong → tidak ada target
            "bukan cpp, tapi kopi"         → scope berhenti di koma → {cpp}
        """
        cc = CONFIG.clarification
        text_lower = clean_text.lower()
        scopes: List[NegationScope] = []
        excluded: Set[str] = set()

        for marker in self._NEGATION_MARKERS:
            pattern = rf'\b{re.escape(marker)}\b'
            for m in re.finditer(pattern, text_lower):
                if len(scopes) >= cc.MAX_NEGATION_SCOPES:
                    break

                tail = text_lower[m.end():]
                stop = self._SCOPE_STOP_PATTERN.search(tail)
                scope_text = tail[:stop.start()] if stop else tail

                targets: List[str] = []
                for t in self._TOKEN_PATTERN.findall(scope_text):
                    if len(t) < CONFIG.signal.MIN_TOKEN_LENGTH:
                        continue
                    if self._classify_token(t) == TokenQuality.LOW_INFORMATION:
                        continue
                    targets.append(t)
                    if len(targets) >= cc.NEGATION_SCOPE_TOKEN_WINDOW:
                        break

                # Identifier simbolik dalam scope (mis. "bukan c++").
                for t in self._SYMBOL_IDENTITY_PATTERN.findall(scope_text):
                    if len(targets) >= cc.NEGATION_SCOPE_TOKEN_WINDOW:
                        break
                    if t not in targets:
                        targets.append(t)

                if not targets:
                    # Marker di akhir kalimat / diikuti tanda baca → bukan exclusion.
                    continue

                scope_end = m.end() + (stop.start() if stop else len(tail))
                scopes.append(NegationScope(
                    marker=marker,
                    target_tokens=targets,
                    marker_position=m.start(),
                    scope_end_position=scope_end,
                    raw_fragment=clean_text[m.start():scope_end].strip(),
                ))
                for t in targets:
                    excluded.add(self.normalize_identity(t))

        return scopes, sorted(excluded)

    # ==========================================
    # [MEKANISME 2] CONTINUATION INTENT DETECTION
    # ==========================================

    def _detect_continuation_intent(
        self,
        clean_text: str,
    ) -> Tuple[bool, List[str]]:
        """
        Deteksi apakah user meminta MELANJUTKAN memori, bukan membuka topik baru.

        "lanjutkan topik ini" / "yang kemarin" / "pembahasan tadi"

        Retrieval hanya melaporkan sinyal ini. Keputusan
        NO_MATCH → UNRESOLVED_CONTINUATION ada di memory_engine,
        sehingga kontrak retrieval tetap stabil.
        """
        cfg = CONFIG.continuation
        lower = clean_text.lower()
        found: List[str] = []

        for phrase in cfg.CONTINUATION_PHRASE_MARKERS:
            if phrase in lower:
                found.append(phrase)

        tokens = set(self._TOKEN_PATTERN.findall(lower))
        for marker in cfg.CONTINUATION_TOKEN_MARKERS:
            if marker in tokens:
                found.append(marker)

        unique_found = sorted(set(found))
        return (len(unique_found) > 0), unique_found

    # ==========================================
    # [MEKANISME 1] CANCELLATION / EXPLICIT SWITCH
    # ==========================================

    def _detect_cancellation_intent(
        self,
        clean_text: str,
    ) -> Tuple[bool, List[str]]:
        """
        Deteksi pembatalan eksplisit: "ganti topik", "lupakan", "batal".

        CATATAN: "saja" TIDAK dijadikan marker cancellation.
        "Pilih saja." adalah respons ambigu terhadap klarifikasi,
        bukan pembatalan. Memasukkan "saja" akan merusak alur
        clarification-round yang sah.
        """
        cfg = CONFIG.continuation
        lower = clean_text.lower()
        found: List[str] = []

        for phrase in cfg.CANCELLATION_PHRASE_MARKERS:
            if phrase in lower:
                found.append(phrase)

        tokens = set(self._TOKEN_PATTERN.findall(lower))
        for marker in cfg.CANCELLATION_TOKEN_MARKERS:
            if marker in tokens:
                found.append(marker)

        unique_found = sorted(set(found))
        return (len(unique_found) > 0), unique_found

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

        # [MEKANISME 1] Negative evidence dengan scope.
        negation_scopes, excluded_tokens = self._extract_negation_scopes(clean_text)

        # [MEKANISME 2] Continuation intent.
        continuation_intent, continuation_markers = self._detect_continuation_intent(clean_text)

        # [MEKANISME 1] Cancellation / explicit switch.
        cancellation_intent, cancellation_markers = self._detect_cancellation_intent(clean_text)

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
            window_pair_strength=pair_strength,
            excluded_tokens=excluded_tokens,
            negation_scopes=negation_scopes,
            continuation_intent=continuation_intent,
            continuation_markers=continuation_markers,
            cancellation_intent=cancellation_intent,
            cancellation_markers=cancellation_markers,
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

    # [REASONING PURITY] Pemetaan sinyal → keluarga bukti.
    # 3 sinyal lexical = 1 family (LEXICAL), bukan 3 evidence independen.
    _SIGNAL_FAMILY: Dict[str, EvidenceFamily] = {
        "token": EvidenceFamily.LEXICAL,
        "unique_token": EvidenceFamily.LEXICAL,
        "co_occurrence": EvidenceFamily.LEXICAL,
        "description": EvidenceFamily.CONTEXTUAL,
        "temporal": EvidenceFamily.CONTEXTUAL,
        "continuity": EvidenceFamily.CONTEXTUAL,
        "graph": EvidenceFamily.RELATIONAL,
    }

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

        [REASONING PURITY]
        - family_support_count : berapa KELUARGA bukti berbeda yang aktif
        - turn_continuity_score: skor continuity murni untuk persistence

        [MEKANISME 3] Aggregated helper (warisan descendant) ikut memperkuat
        bukti description. TIDAK menambah bobot ke-8 — ia masuk melalui
        term description yang sudah ada, sebagai max(own, aggregated).
        Karena TopicKey lama punya aggregated "" → nilai identik dengan sebelumnya.
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

        # [MEKANISME 3] Aggregated Helper Score (warisan descendant).
        # Terpisah dari description_score agar terlihat mana bukti own vs warisan.
        aggregated_helper_score = 0.0
        agg_text = key.aggregated_helper_description or ""
        if agg_text:
            agg_tokens = set(re.findall(r'\b\w+\b', agg_text.lower()))
            agg_tokens = {t for t in agg_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
            if agg_tokens:
                aggregated_helper_score = len(
                    signal_tokens.intersection(agg_tokens)
                ) / len(agg_tokens)

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

        # [MEKANISME 3] Effective description = max(own, aggregated).
        # Ini memasukkan hierarchical helper evidence TANPA menambah bobot baru.
        effective_description = max(description_score, aggregated_helper_score)

        # F. Hitung Final Score (Weighted Sum)
        w = CONFIG.activation
        contributions = {
            "token": token_score * w.TOKEN_WEIGHT,
            "description": effective_description * w.DESCRIPTION_WEIGHT,
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
            "description": effective_description,
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

        # ==========================================
        # [REASONING PURITY] EVIDENCE FAMILY GROUPING
        # ==========================================
        # Kelompokkan sinyal aktif ke dalam 3 keluarga bukti.
        # 3 sinyal lexical (token + unique + co-occurrence) = 1 family.
        family_set: set = set()
        signal_to_family: Dict[str, EvidenceFamily] = {}
        for name in supporting_names:
            family = self._SIGNAL_FAMILY.get(name)
            if family is not None:
                family_set.add(family)
                signal_to_family[name] = family
        evidence_families = sorted(family_set, key=lambda f: f.value)
        family_support_count = len(evidence_families)

        # Dominasi family: satu family menyumbang mayoritas strength.
        family_strengths: Dict[EvidenceFamily, float] = {f: 0.0 for f in EvidenceFamily}
        for name in supporting_names:
            family = self._SIGNAL_FAMILY.get(name)
            if family is not None:
                family_strengths[family] += evidence_values[name]
        total_family_strength = sum(family_strengths.values())
        is_single_family_dominated = False
        if total_family_strength > 0:
            max_family_strength = max(family_strengths.values())
            is_single_family_dominated = (
                max_family_strength / total_family_strength
            ) >= CONFIG.evidence_family.SINGLE_FAMILY_DOMINANCE_RATIO

        # Safety cap: satu skor besar tidak boleh menghasilkan kepercayaan palsu.
        safety = CONFIG.candidate_safety
        ef = CONFIG.evidence_family
        confidence = evidence_strength
        if support_count <= 1:
            confidence = min(confidence, safety.SINGLE_SIGNAL_CONFIDENCE_CAP)
        elif support_count == 2:
            confidence = min(confidence, safety.DUAL_SIGNAL_CONFIDENCE_CAP)
        # [REASONING PURITY] Family cap — jika hanya 1 keluarga bukti aktif,
        # confidence tidak boleh melewati SINGLE_FAMILY_CONFIDENCE_CAP.
        # Exemption: continuity (CONTEXTUAL) adalah bukti sah untuk continuation.
        if family_support_count <= 1:
            continuity_exempt = (
                ef.CONTINUITY_EXEMPT_FROM_FAMILY_CHECK
                and cont_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_exempt:
                confidence = min(confidence, ef.SINGLE_FAMILY_CONFIDENCE_CAP)

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
            is_single_signal_dominated=is_dominated,
            active_evidence_families=evidence_families,
            evidence_family_count=family_support_count,
            is_single_family_dominated=is_single_family_dominated,
            signal_to_family=signal_to_family,
        )

        notes = []
        if support_count <= 1:
            notes.append(f"single-signal capped at {safety.SINGLE_SIGNAL_CONFIDENCE_CAP:.2f}")
        if is_dominated:
            notes.append("single-evidence dominated")
        if family_support_count <= 1 and not (
            ef.CONTINUITY_EXEMPT_FROM_FAMILY_CHECK and cont_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
        ):
            notes.append(
                f"single-family capped at {ef.SINGLE_FAMILY_CONFIDENCE_CAP:.2f} "
                f"(family={evidence_families[0].value if evidence_families else 'none'})"
            )
        if aggregated_helper_score > description_score:
            notes.append(
                f"aggregated helper evidence used ({aggregated_helper_score:.2f} "
                f"> own {description_score:.2f})"
            )

        # [REASONING PURITY] turn_continuity_score = skor continuity murni untuk
        # persistence (relationship_strength fragment).
        # Untuk kandidat yang IS active topic: sama dengan continuity_score.
        # Untuk yang BUKAN: 0 (bukan kelanjutan state aktif).
        # memory_engine.py akan menghitung ulang dengan formula lebih kaya
        # (overlap + history + fragments - novelty).
        turn_continuity_score = cont_score

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
            evidence_notes=notes,
            family_support_count=family_support_count,
            evidence_families=evidence_families,
            is_single_family_dominated=is_single_family_dominated,
            turn_continuity_score=turn_continuity_score,
            aggregated_helper_score=aggregated_helper_score,
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
        cross_session_evidence: Optional[dict] = None, # [NEW] topic_id -> score
        fragment_graph_scores: Optional[Dict[str, float]] = None, # [REASONING PURITY] fragment_id -> graph score
        clarification: Optional[ClarificationAccumulation] = None, # [MEKANISME 1] accumulated evidence
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

        [REASONING PURITY]
        5b. fragment_graph_scores dipakai untuk fragment-level graph evidence
            (bukan relationship_strength metadata sebagai proxy)
        5c. temporal_distance_score dihitung berbasis kedekatan waktu, bukan binary

        [MEKANISME 1] Jika `clarification` diberikan:
            - sinyal diekstrak dari COMBINED text (original + seluruh history)
            - exclusion penalty diterapkan pada kandidat yang ditolak user
            - kandidat lemah DIELIMINASI dari shortlist (narrowing)
            Jika None → perilaku identik dengan versi sebelumnya.
        """
        
        # 1. Extract Signal
        # [MEKANISME 1] Sumber sinyal: combined clarification jika ada.
        source_text = input_text
        if clarification is not None and clarification.combined_text:
            source_text = clarification.combined_text

        signal = self.extractor.extract(source_text, reference_time)

        # [MEKANISME 1] Gabungkan exclusion dari accumulated context.
        if clarification is not None and clarification.excluded_tokens:
            merged = set(signal.excluded_tokens)
            for t in clarification.excluded_tokens:
                merged.add(self.extractor.normalize_identity(t))
            signal.excluded_tokens = sorted(merged)

        if signal.confidence < CONFIG.signal.MIN_SIGNAL_STRENGTH:
            return RetrievalResult(
                decision_type=DecisionType.NO_MATCH,
                analysis_notes="Sinyal terlalu lemah / noise.",
                signal_profile=signal,
                turn_continuity_score=continuity_score if active_topic_id else 0.0,
                continuation_intent=signal.continuation_intent,
                cancellation_intent=signal.cancellation_intent,
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
                signal_profile=signal,
                turn_continuity_score=continuity_score if active_topic_id else 0.0,
                continuation_intent=signal.continuation_intent,
                cancellation_intent=signal.cancellation_intent,
            )

        # [MEKANISME 1] Blend combined-context score dengan last-answer score.
        # Hanya aktif jika ada history klarifikasi.
        if clarification is not None and clarification.clarification_history:
            self._blend_clarification_scores(
                candidates=candidates,
                keys=keys,
                clarification=clarification,
                reference_time=reference_time,
                active_topic_id=active_topic_id,
                continuity_score=continuity_score,
                cross_session_evidence=cross_session_evidence,
                global_token_counts=global_token_counts,
                total_topic_keys=total_topic_keys,
            )

        # 2b. [ASSOCIATIVE EVIDENCE LOCKING] CONTRADICTION PENALTY (relatif).
        # Kontradiksi = "ada bukti kuat bahwa kandidat ini BUKAN yang dimaksud".
        # Karena itu penalti hanya dipicu bila ada kandidat lain yang
        # menjelaskan intent dengan lebih baik. Tanpa pembanding, deskripsi
        # pendek tidak boleh dianggap kontradiksi.
        temporal_range = self.temporal_resolver.resolve(signal.temporal_signals, reference_time)
        self._apply_contradiction_penalties(candidates, fragments, temporal_range)

        # [MEKANISME 1] EXCLUSION PENALTY dari negative evidence user.
        self._apply_exclusion_penalties(candidates, key_map, signal)

        # 3. Sort & Analyze Ambiguity
        candidates.sort(key=lambda x: x.final_score, reverse=True)

        # [MEKANISME 1] CANDIDATE NARROWING.
        # Kandidat yang sudah lemah / dieliminasi tidak boleh muncul lagi
        # di round klarifikasi berikutnya.
        narrowed_out_keys: List[str] = []
        excluded_keys: List[str] = [c.key_id for c in candidates if c.is_excluded]
        if clarification is not None:
            before = [c.key_id for c in candidates]
            candidates = self._narrow_candidates(candidates, clarification)
            after = {c.key_id for c in candidates}
            narrowed_out_keys = [k for k in before if k not in after]
            if not candidates:
                return RetrievalResult(
                    decision_type=DecisionType.NO_MATCH,
                    candidates=[],
                    analysis_notes="Semua kandidat tereliminasi oleh klarifikasi.",
                    signal_profile=signal,
                    excluded_candidate_keys=excluded_keys,
                    narrowed_out_keys=narrowed_out_keys,
                    continuation_intent=signal.continuation_intent,
                    cancellation_intent=signal.cancellation_intent,
                )

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
                # [MEKANISME 1] Shortlist dibatasi MAX_AMBIGUITY_CANDIDATES dan
                # memakai aggregated helper agar pertanyaan makin spesifik.
                msg_parts = ["Saya menemukan beberapa topik yang mungkin Anda maksud:\n"]
                max_show = CONFIG.hierarchy.MAX_AMBIGUITY_CANDIDATES
                for i, cand in enumerate(candidates[:max_show], 1):
                    k = key_map.get(cand.key_id)
                    if k is None:
                        desc = "pembahasan terkait."
                    else:
                        desc = k.aggregated_helper_description or k.helper_description
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
                selected_key_id, fragments, temporal_range, signal, reference_time,
                fragment_graph_scores=fragment_graph_scores,
            )

        evidence_summary = self._build_evidence_summary(top_candidate)

        # [REASONING PURITY] turn_continuity_score untuk persistence.
        # Diambil dari kandidat yang terpilih. Jika tidak ada kandidat terpilih,
        # fallback ke continuity_score dari active state (history window).
        if selected_key_id:
            selected_candidate = next(
                (c for c in candidates if c.key_id == selected_key_id), None
            )
            chosen_turn_continuity = (
                selected_candidate.turn_continuity_score if selected_candidate else 0.0
            )
        else:
            chosen_turn_continuity = continuity_score if active_topic_id else 0.0

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
            evidence_summary=evidence_summary,
            turn_continuity_score=chosen_turn_continuity,
            excluded_candidate_keys=excluded_keys,
            narrowed_out_keys=narrowed_out_keys,
            continuation_intent=signal.continuation_intent,
            cancellation_intent=signal.cancellation_intent,
        )

    # ==========================================
    # [MEKANISME 1] CLARIFICATION ACCUMULATION HELPERS
    # ==========================================

    def _blend_clarification_scores(
        self,
        candidates: List[CandidateScore],
        keys: List[TopicKey],
        clarification: ClarificationAccumulation,
        reference_time: datetime,
        active_topic_id: Optional[str],
        continuity_score: float,
        cross_session_evidence: Dict[str, float],
        global_token_counts: Optional[dict],
        total_topic_keys: int,
    ) -> None:
        """
        Blend skor COMBINED context dengan skor JAWABAN TERAKHIR saja.

        Kenapa perlu:
            "yang rencana kopi" sendiri terlalu miskin bukti.
            Combined context (original + history) adalah sumber utama,
            jawaban terakhir melengkapi.

        Rumus:
            final = W * combined_score + (1 - W) * last_answer_score
            W = CLARIFICATION_COMBINED_INPUT_WEIGHT (0.70)

        Hanya dipanggil jika ada history — jalur tanpa klarifikasi tidak berubah.
        """
        weight = CONFIG.clarification.CLARIFICATION_COMBINED_INPUT_WEIGHT
        last_answer = clarification.clarification_history[-1]
        last_signal = self.extractor.extract(last_answer, reference_time)
        if not last_signal.unique_terms:
            return

        key_map = {k.key_id: k for k in keys}
        for cand in candidates:
            key = key_map.get(cand.key_id)
            if key is None:
                continue
            last_score = self.scorer.score(
                last_signal, key,
                key.key_id == active_topic_id,
                continuity_score,
                cross_session_evidence.get(key.key_id, 0.0),
                global_token_counts,
                total_topic_keys=total_topic_keys,
            )
            combined_final = cand.final_score
            blended = (weight * combined_final) + ((1.0 - weight) * last_score.final_score)
            cand.final_score = blended
            cand.evidence_notes.append(
                f"clarification blend: combined={combined_final:.2f} "
                f"last={last_score.final_score:.2f} w={weight:.2f} → {blended:.2f}"
            )

    def _candidate_identity_terms(self, key: TopicKey) -> Set[str]:
        """
        Term identitas kandidat untuk exclusion matching.

        Menggabungkan token_profile + title + helper + aggregated helper,
        DENGAN symbol-aware extraction dan alias normalization, sehingga
        "bukan cpp" bisakena kandidat berjudul "C++ Kasir Kopi".
        """
        terms: Set[str] = set()
        for t in (key.token_profile or {}).keys():
            terms.add(self.extractor.normalize_identity(t))
        terms |= self.extractor.extract_identity_terms(key.title)
        terms |= self.extractor.extract_identity_terms(key.helper_description)
        terms |= self.extractor.extract_identity_terms(key.aggregated_helper_description)
        return terms

    def _apply_exclusion_penalties(
        self,
        candidates: List[CandidateScore],
        key_map: Dict[str, TopicKey],
        signal: SignalProfile,
    ) -> None:
        """
        [MEKANISME 1] Negative evidence → candidate elimination.

        "yang penting bukan topik cpp" → kandidat C++ dipenalti & ditandai
        is_excluded, sehingga tidak muncul lagi di shortlist klarifikasi.
        """
        if not signal.excluded_tokens:
            return

        weight = CONFIG.clarification.CLARIFICATION_NEGATION_WEIGHT
        excluded_norm = {
            self.extractor.normalize_identity(t) for t in signal.excluded_tokens
        }

        for cand in candidates:
            key = key_map.get(cand.key_id)
            if key is None:
                continue
            terms = self._candidate_identity_terms(key)
            hits = sorted(excluded_norm.intersection(terms))
            if not hits:
                continue
            cand.exclusion_score = weight
            cand.excluded_matched_tokens = hits
            cand.is_excluded = True
            cand.final_score = max(0.0, cand.final_score - weight)
            cand.confidence = max(0.0, cand.confidence - weight)
            cand.evidence_notes.append(
                f"exclusion (-{weight:.2f}): user menolak {hits}"
            )
            if cand.evidence_profile:
                cand.evidence_profile.contradiction_count += 1
                cand.evidence_profile.contradiction_notes.append(
                    f"negative evidence: {hits}"
                )

    def _narrow_candidates(
        self,
        candidates: List[CandidateScore],
        clarification: ClarificationAccumulation,
    ) -> List[CandidateScore]:
        """
        [MEKANISME 1] Persempit shortlist kandidat.

        Dua aturan retensi (harus lolos KEDUANYA):
            1. final_score >= MIN_CANDIDATE_RETENTION_SCORE (0.35)
            2. final_score >= top_score * CANDIDATE_RETENTION_RATIO (0.60)

        Contoh: A=0.81, B=0.79, C=0.20
            floor = max(0.35, 0.81*0.60=0.486) = 0.486
            → C gugur. Round berikutnya hanya A & B.

        Kandidat yang sudah di-exclude user selalu gugur.
        Kandidat yang sudah di excluded_candidate_keys juga tidak boleh kembali.
        """
        if not candidates:
            return []

        cc = CONFIG.clarification
        permanently_excluded = set(clarification.excluded_candidate_keys or [])

        eligible = [
            c for c in candidates
            if not c.is_excluded and c.key_id not in permanently_excluded
        ]
        if not eligible:
            return []

        top_score = eligible[0].final_score
        floor = max(
            cc.MIN_CANDIDATE_RETENTION_SCORE,
            top_score * cc.CANDIDATE_RETENTION_RATIO,
        )
        kept = [c for c in eligible if c.final_score >= floor]

        # Jangan sampai shortlist kosong total jika masih ada kandidat sah.
        if not kept:
            kept = eligible[:1]

        max_show = CONFIG.hierarchy.MAX_AMBIGUITY_CANDIDATES
        return kept[:max_show]

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

        [REASONING PURITY] Juga memvalidasi evidence family count.
        """
        ev = CONFIG.evidence
        ef = CONFIG.evidence_family

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

        # 2b. [REASONING PURITY] Family count minimum.
        # Dengan exemption untuk continuity (CONTEXTUAL family).
        if candidate.family_support_count < ef.MIN_EVIDENCE_FAMILIES:
            continuity_family_exempt = (
                ef.CONTINUITY_EXEMPT_FROM_FAMILY_CHECK
                and EvidenceFamily.CONTEXTUAL in candidate.evidence_families
                and candidate.continuity_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_family_exempt:
                families_str = ",".join(f.value for f in candidate.evidence_families) or "none"
                return (
                    DecisionType.INSUFFICIENT_EVIDENCE,
                    f"Keragaman bukti tidak cukup pada '{candidate.title}' "
                    f"(families={candidate.family_support_count}, active=[{families_str}])."
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
            f"  Family Count            : {candidate.family_support_count}",
            f"  Families                : {[f.value for f in candidate.evidence_families]}",
            f"  Contradiction           : {candidate.contradiction_score:.3f}",
            f"  Turn Continuity (pure)  : {candidate.turn_continuity_score:.3f}",
            "  Evidence:",
            f"    Token          {candidate.token_score:.3f}",
            f"    Description    {candidate.description_score:.3f}",
            f"    Co-occurrence  {candidate.co_occurrence_score:.3f}",
            f"    Unique Token   {candidate.unique_token_score:.3f}",
            f"    Temporal       {candidate.temporal_score:.3f}",
            f"    Continuity     {candidate.continuity_score:.3f}",
            f"    Graph          {candidate.graph_score:.3f}",
            f"    Aggregated Hlp {candidate.aggregated_helper_score:.3f}",
        ]
        if candidate.exclusion_score > 0:
            lines.append(
                f"  EXCLUDED (-{candidate.exclusion_score:.2f}) "
                f"matched={candidate.excluded_matched_tokens}"
            )
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

    # ==========================================
    # [REASONING PURITY] TEMPORAL DISTANCE QUALITY
    # ==========================================

    def _resolve_requested_time(
        self,
        signal: SignalProfile,
        temporal_range: Optional[Tuple[datetime, datetime]],
    ) -> Optional[datetime]:
        """
        [FIX — Batch C] Resolve WAKTU YANG DIMINTA USER, bukan midpoint range.

        Bug sebelumnya: midpoint dari range "kemarin pukul 15:00"
            start = 13:00:00.000000
            end   = 17:59:59.999999
            midpoint = 15:29:59.999999   ← BUKAN 15:00!
        Akibatnya fragment 14:59 dihitung berjarak 31 menit (seharusnya 1 menit).

        Sekarang:
            - Jika query menyebut jam spesifik (HOUR_15_00) → target = jam itu
              pada tanggal awal range.
            - Jika tidak (query "kemarin" saja) → midpoint range, karena seluruh
              hari memang range yang diminta dan presisi tidak ditekankan.
        """
        if temporal_range is None:
            return None

        start_t, end_t = temporal_range

        # Cari sinyal jam spesifik
        for sig in (signal.temporal_signals or []):
            if not sig.startswith("HOUR_"):
                continue
            parts = sig.split("_")
            if len(parts) < 2:
                continue
            try:
                target_hour = int(parts[1])
                target_minute = int(parts[2]) if len(parts) > 2 else 0
            except (ValueError, IndexError):
                continue
            if 0 <= target_hour <= 23 and 0 <= target_minute <= 59:
                return start_t.replace(
                    hour=target_hour, minute=target_minute,
                    second=0, microsecond=0
                )

        # Tidak ada jam spesifik → midpoint range (presisi hari)
        total_seconds = (end_t - start_t).total_seconds()
        return start_t + timedelta(seconds=total_seconds / 2)

    def _compute_temporal_distance_score(
        self,
        fragment_time: datetime,
        temporal_range: Optional[Tuple[datetime, datetime]],
        has_specific_hour: bool,
        requested_time: Optional[datetime] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Menghitung skor kualitas temporal berdasarkan KEDEKATAN fragment
        dengan waktu yang diminta. Bukan sekadar "masuk range atau tidak".

        Returns:
            (temporal_distance_score, distance_minutes)

        [FIX — Batch C] Dua perbaikan:
        1. Target waktu diambil dari `requested_time` (jam yang diminta user),
           bukan midpoint range yang bergeser karena microsecond=999999.
        2. TEMPORAL_HOUR_PRECISION_WEIGHT dipakai sebagai pengali DECAY,
           bukan pengali SKOR. Sebelumnya boost multiplicative
           (score × 1.5 lalu clamp ke 1.0) membuat fragment dekat dan jauh
           sama-sama mendapat 1.0 → diskriminasi hilang total.

        Sekarang presisi tinggi = decay lebih tajam = diskriminasi lebih baik:
            query "pukul 15:00", fragment 14:59 → distance 1 min  → EXACT 1.0
            query "pukul 15:00", fragment 16:30 → distance 90 min → 0.775
        """
        tc = CONFIG.temporal

        if temporal_range is None:
            # Tidak ada constraint waktu → baseline score.
            return (0.5, None)

        start_t, end_t = temporal_range

        # Normalisasi timezone fragment
        if fragment_time.tzinfo is None and start_t.tzinfo is not None:
            fragment_time = fragment_time.replace(tzinfo=start_t.tzinfo)

        # [FIX] Target = waktu yang diminta user (jika ada), else midpoint range.
        if requested_time is not None:
            target_time = requested_time
            if target_time.tzinfo is None and start_t.tzinfo is not None:
                target_time = target_time.replace(tzinfo=start_t.tzinfo)
        else:
            total_seconds = (end_t - start_t).total_seconds()
            target_time = start_t + timedelta(seconds=total_seconds / 2)

        # Distance dari waktu yang diminta
        distance_seconds = abs((fragment_time - target_time).total_seconds())
        distance_hours = distance_seconds / 3600.0
        distance_minutes = distance_seconds / 60.0

        # Exact match: fragment di dalam 5 menit dari waktu yang diminta
        if distance_minutes <= 5.0:
            return (tc.TEMPORAL_EXACT_MATCH_SCORE, distance_minutes)

        # [FIX] Precision weight mengalikan DECAY, bukan SKOR.
        # Query dengan jam spesifik → presisi lebih ditekankan → decay lebih tajam.
        precision_weight = (
            tc.TEMPORAL_HOUR_PRECISION_WEIGHT if has_specific_hour
            else tc.TEMPORAL_DAY_PRECISION_WEIGHT
        )
        decay = tc.TEMPORAL_DISTANCE_DECAY * distance_hours * precision_weight

        # Floor di 0.1 agar fragment yang masuk range tidak drop ke 0
        score = max(0.1, 1.0 - decay)

        return (score, distance_minutes)

    def _select_fragments(
        self, 
        key_id: str, 
        fragments: List[FragmentMetadata],
        time_range: Optional[Tuple[datetime, datetime]],
        signal: SignalProfile, # [NEW] Untuk fragment relevance scoring
        reference_time: datetime, # [NEW] Untuk recency scoring
        fragment_graph_scores: Optional[Dict[str, float]] = None, # [REASONING PURITY]
    ) -> Tuple[List[str], Dict[str, dict]]:
        """
        [CRITICAL FIX] True Fragment-Level Ranking (Review Kedua).
        Memilih ID fragment terbaik berdasarkan relevansi query-to-fragment yang sesungguhnya,
        membandingkan signal tokens dengan fragment's own token_profile.

        [ASSOCIATIVE EVIDENCE LOCKING] Mengembalikan JUGA metadata bukti per-fragment
        agar Context Builder dapat mengurutkan berdasarkan kualitas bukti,
        bukan sekadar recency.

        [REASONING PURITY]
        - graph_score dihitung dari fragment_graph_scores (edge nyata),
          BUKAN dari relationship_strength metadata.
        - temporal_score dihitung berbasis distance (decay per jam),
          BUKAN binary 1.0/0.5.

        [MEKANISME 3] fragment_key & helper (own + aggregated) ikut menjadi
        bukti identitas fragment, sehingga "Modal Bisnis Kopi" bisa dibedakan
        dari "Lokasi Bisnis Kopi" meski berada di TopicKey yang sama.
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

        # [REASONING PURITY] Cek apakah query menyebut jam spesifik.
        has_specific_hour = any(
            sig.startswith("HOUR_") for sig in (signal.temporal_signals or [])
        )

        # [FIX — Batch C] Resolve waktu yang diminta SEKALI untuk seluruh fragment,
        # agar semua fragment dibandingkan terhadap target yang sama dan benar.
        requested_time = self._resolve_requested_time(signal, time_range)

        graph_scores_map = fragment_graph_scores or {}
        
        for f in key_frags:
            # 1. TRUE Token Relevance (Query vs Fragment Token Profile)
            # [UPDATED] Quality-weighted agar token low-information tidak mendominasi.
            frag_tokens = set(f.token_profile.keys()) if f.token_profile else set()
            matched = signal_tokens.intersection(frag_tokens)
            frag_token_score = sum(token_weights[t] for t in matched) / total_weight if total_weight else 0.0

            # [MEKANISME 3] Fragment identity evidence (key + own helper + aggregated).
            identity_score = self._fragment_identity_score(f, signal_tokens)
            # Identitas bagian adalah sinyal lexical terkuat → boleh menggantikan
            # token_score jika lebih kuat. Tidak menambah bobot baru.
            effective_lexical = max(frag_token_score, identity_score)
            
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
            
            # 3. [REASONING PURITY] Temporal Distance Score (BUKAN binary)
            # Fragment di dalam range dibedakan berdasarkan kedekatan waktu.
            frag_temporal_score = 0.5  # baseline jika tidak ada constraint
            temporal_distance_minutes: Optional[float] = None
            if f.temporal_index is not None:
                frag_temporal_score, temporal_distance_minutes = self._compute_temporal_distance_score(
                    f.temporal_index, time_range, has_specific_hour,
                    requested_time=requested_time,
                )
            
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

            # 5. [REASONING PURITY] Fragment Graph Score (dari edge nyata)
            # Jika fragment_graph_scores disediakan oleh memory_engine, pakai itu.
            # Fallback ke relationship_strength hanya jika graph scores tidak ada.
            if fragment_graph_scores is not None:
                frag_graph_score = graph_scores_map.get(f.fragment_id, 0.0)
            else:
                frag_graph_score = f.relationship_strength or 0.0
            # Floor dari config
            gc = CONFIG.graph
            if frag_graph_score < gc.FRAGMENT_GRAPH_MIN_SCORE:
                frag_graph_score = 0.0
                
            relevance = (
                (effective_lexical * w_frag.FRAGMENT_TOKEN_WEIGHT) +
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
                # [REASONING PURITY] graph_score dari edge nyata
                "graph_score": frag_graph_score,
                "exact_temporal_match": (
                    time_range is not None
                    and temporal_distance_minutes is not None
                    and temporal_distance_minutes <= 5.0
                ),
                "temporal_distance_minutes": temporal_distance_minutes,
                "temporal_distance_score": frag_temporal_score,
                # [REASONING PURITY] Untuk RetrievedMemory
                "fragment_graph_score": frag_graph_score,
                # [MEKANISME 3] Identity evidence untuk audit & context framing.
                "fragment_identity_score": identity_score,
                "fragment_key": f.fragment_key,
                "fragment_helper_description": f.fragment_helper_description,
                "parent_fragment_id": f.parent_fragment_id,
                "depth": f.depth,
                "is_composite": f.is_composite,
                "source_fragment_ids": list(f.source_fragment_ids or []),
            }
            
        # Sort by relevance descending
        ranked_frags.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil maksimal N fragment
        limit = CONFIG.retrieval.MAX_RETRIEVED_FRAGMENTS
        selected = [fid for fid, score in ranked_frags[:limit]]
        selected_scores = {fid: fragment_scores[fid] for fid in selected}
        return selected, selected_scores

    # ==========================================
    # [MEKANISME 3 + 5] FRAGMENT IDENTITY & PLACEMENT SCORING
    # ==========================================

    def _fragment_identity_score(
        self,
        fragment: FragmentMetadata,
        signal_tokens: Set[str],
    ) -> float:
        """
        Skor identitas fragment dari fragment_key + own helper + aggregated helper.

        Ini yang membedakan "Modal Bisnis Kopi" dari "Lokasi Bisnis Kopi"
        meski keduanya berada dalam TopicKey yang sama.

        Mengembalikan nilai 0..1 (rasio token sinyal yang terjelaskan).
        """
        if not signal_tokens:
            return 0.0

        pools: List[Set[str]] = []
        for text in (
            fragment.fragment_key,
            fragment.fragment_helper_description,
            fragment.aggregated_helper_description,
        ):
            if not text:
                continue
            toks = set(re.findall(r'\b\w+\b', text.lower()))
            toks = {t for t in toks if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
            if toks:
                pools.append(toks)

        if not pools:
            return 0.0

        best = 0.0
        for pool in pools:
            score = len(signal_tokens.intersection(pool)) / len(pool)
            if score > best:
                best = score
        return min(1.0, best)

    def score_fragment_candidates(
        self,
        signal: SignalProfile,
        fragments: List[FragmentMetadata],
        reference_time: datetime,
        temporal_range: Optional[Tuple[datetime, datetime]] = None,
        fragment_graph_scores: Optional[Dict[str, float]] = None,
    ) -> List[FragmentCandidateScore]:
        """
        [MEKANISME 3] Fragment-level retrieval scoring dengan identitas bagian.

        Dipakai ketika TopicKey sudah jelas tetapi bagian belum, atau ketika
        query menyebut subtopik spesifik ("kopi yang modalnya murah").

        Berbeda dari _select_fragments: method ini mengembalikan objek
        FragmentCandidateScore lengkap dengan fragment_key & helper,
        sehingga engine bisa menavigasi hierarchy.
        """
        if not fragments or not signal.unique_terms:
            return []

        signal_tokens = set(signal.unique_terms)
        quality_map = signal.token_quality or {}
        token_weights = {
            t: self.extractor.token_weight(t, quality_map) for t in signal_tokens
        }
        total_weight = sum(token_weights.values()) or 1.0
        w = CONFIG.fragment_ranking
        graph_map = fragment_graph_scores or {}
        has_specific_hour = any(
            s.startswith("HOUR_") for s in (signal.temporal_signals or [])
        )
        requested_time = self._resolve_requested_time(signal, temporal_range)

        results: List[FragmentCandidateScore] = []
        for f in fragments:
            frag_tokens = set(f.token_profile.keys()) if f.token_profile else set()
            matched = signal_tokens.intersection(frag_tokens)
            token_score = sum(token_weights[t] for t in matched) / total_weight

            key_score = 0.0
            if f.fragment_key:
                kt = set(re.findall(r'\b\w+\b', f.fragment_key.lower()))
                kt = {t for t in kt if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
                if kt:
                    key_score = len(signal_tokens.intersection(kt)) / len(kt)

            helper_score = 0.0
            if f.fragment_helper_description:
                ht = set(re.findall(r'\b\w+\b', f.fragment_helper_description.lower()))
                ht = {t for t in ht if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
                if ht:
                    helper_score = len(signal_tokens.intersection(ht)) / len(ht)

            agg_score = 0.0
            if f.aggregated_helper_description:
                at = set(re.findall(r'\b\w+\b', f.aggregated_helper_description.lower()))
                at = {t for t in at if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
                if at:
                    agg_score = len(signal_tokens.intersection(at)) / len(at)

            cooc_score = 0.0
            if f.co_occurrence_profile and signal.window_pair_strength:
                total_pair = sum(signal.window_pair_strength.values())
                hit = 0.0
                for pair, strength in signal.window_pair_strength.items():
                    if pair in f.co_occurrence_profile:
                        hit += strength
                    else:
                        parts = pair.split("↔")
                        if len(parts) == 2 and f"{parts[1]}↔{parts[0]}" in f.co_occurrence_profile:
                            hit += strength
                if total_pair > 0:
                    cooc_score = min(1.0, hit / total_pair)

            temporal_score = 0.5
            if f.temporal_index is not None:
                temporal_score, _ = self._compute_temporal_distance_score(
                    f.temporal_index, temporal_range, has_specific_hour,
                    requested_time=requested_time,
                )

            recency_score = 0.0
            if f.created_at:
                created = f.created_at
                if created.tzinfo is None and reference_time.tzinfo is not None:
                    created = created.replace(tzinfo=reference_time.tzinfo)
                age_hours = max(0.0, (reference_time - created).total_seconds() / 3600.0)
                recency_score = max(0.0, 1.0 - (age_hours / 720.0))

            graph_score = graph_map.get(f.fragment_id, 0.0)

            identity = max(key_score, helper_score, agg_score)
            effective_lexical = max(token_score, identity)
            final = (
                effective_lexical * w.FRAGMENT_TOKEN_WEIGHT
                + cooc_score * w.FRAGMENT_COOC_WEIGHT
                + temporal_score * w.FRAGMENT_TEMPORAL_WEIGHT
                + recency_score * w.FRAGMENT_RECENCY_WEIGHT
            )

            supporting: List[str] = []
            ev_thr = CONFIG.evidence.EVIDENCE_SUPPORT_THRESHOLD
            if token_score >= ev_thr:
                supporting.append("token")
            if key_score >= ev_thr:
                supporting.append("fragment_key")
            if helper_score >= ev_thr:
                supporting.append("fragment_helper")
            if agg_score >= ev_thr:
                supporting.append("aggregated_helper")
            if cooc_score >= ev_thr:
                supporting.append("co_occurrence")
            if graph_score >= ev_thr:
                supporting.append("graph")

            results.append(FragmentCandidateScore(
                fragment_id=f.fragment_id,
                fragment_key=f.fragment_key,
                topic_key_id=f.topic_key_id,
                depth=f.depth,
                parent_fragment_id=f.parent_fragment_id,
                fragment_key_score=key_score,
                helper_score=helper_score,
                aggregated_helper_score=agg_score,
                token_score=token_score,
                co_occurrence_score=cooc_score,
                temporal_score=temporal_score,
                graph_score=graph_score,
                recency_score=recency_score,
                final_score=final,
                supporting_evidence_names=supporting,
            ))

        results.sort(key=lambda c: c.final_score, reverse=True)
        return results

    def score_parent_candidates(
        self,
        signal: SignalProfile,
        candidate_parents: List[FragmentMetadata],
    ) -> List[FragmentCandidateScore]:
        """
        [MEKANISME 5] Placement scoring: fragment baru → parent mana yang tepat.

        [KUNCI] TEMPORAL & RECENCY SENGAJA DIKECUALIKAN.
        Kondisi uji 20 menuntut parent dipilih berdasarkan EVIDENCE,
        bukan sekadar fragment terbaru. Jika recency ikut dinilai,
        fragment terakhir akan selalu menang tanpa dasar semantik.

        Bobot diambil dari FragmentRankingConstants lalu DINORMALISASI
        atas bobot yang benar-benar dipakai (tidak ada angka ajaib):
            identity → FRAGMENT_TOKEN_WEIGHT
            token    → FRAGMENT_COOC_WEIGHT
            cooc     → FRAGMENT_COOC_WEIGHT
        """
        if not candidate_parents or not signal.unique_terms:
            return []

        signal_tokens = set(signal.unique_terms)
        quality_map = signal.token_quality or {}
        token_weights = {
            t: self.extractor.token_weight(t, quality_map) for t in signal_tokens
        }
        total_weight = sum(token_weights.values()) or 1.0

        w = CONFIG.fragment_ranking
        w_identity = w.FRAGMENT_TOKEN_WEIGHT
        w_token = w.FRAGMENT_COOC_WEIGHT
        w_cooc = w.FRAGMENT_COOC_WEIGHT
        w_sum = (w_identity + w_token + w_cooc) or 1.0

        results: List[FragmentCandidateScore] = []
        for f in candidate_parents:
            frag_tokens = set(f.token_profile.keys()) if f.token_profile else set()
            matched = signal_tokens.intersection(frag_tokens)
            token_score = sum(token_weights[t] for t in matched) / total_weight

            identity = self._fragment_identity_score(f, signal_tokens)

            cooc_score = 0.0
            if f.co_occurrence_profile and signal.window_pair_strength:
                total_pair = sum(signal.window_pair_strength.values())
                hit = 0.0
                for pair, strength in signal.window_pair_strength.items():
                    if pair in f.co_occurrence_profile:
                        hit += strength
                    else:
                        parts = pair.split("↔")
                        if len(parts) == 2 and f"{parts[1]}↔{parts[0]}" in f.co_occurrence_profile:
                            hit += strength
                if total_pair > 0:
                    cooc_score = min(1.0, hit / total_pair)

            final = (
                identity * w_identity
                + token_score * w_token
                + cooc_score * w_cooc
            ) / w_sum

            supporting: List[str] = []
            ev_thr = CONFIG.evidence.EVIDENCE_SUPPORT_THRESHOLD
            if identity >= ev_thr:
                supporting.append("identity")
            if token_score >= ev_thr:
                supporting.append("token")
            if cooc_score >= ev_thr:
                supporting.append("co_occurrence")

            results.append(FragmentCandidateScore(
                fragment_id=f.fragment_id,
                fragment_key=f.fragment_key,
                topic_key_id=f.topic_key_id,
                depth=f.depth,
                parent_fragment_id=f.parent_fragment_id,
                fragment_key_score=identity,
                helper_score=identity,
                aggregated_helper_score=identity,
                token_score=token_score,
                co_occurrence_score=cooc_score,
                # Temporal & recency Dinolkan secara eksplisit: tidak dipakai.
                temporal_score=0.0,
                graph_score=0.0,
                recency_score=0.0,
                final_score=final,
                supporting_evidence_names=supporting,
                notes=["placement scoring: temporal & recency excluded by design"],
            ))

        results.sort(key=lambda c: c.final_score, reverse=True)
        return results
