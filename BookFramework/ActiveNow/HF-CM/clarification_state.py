"""
clarification_state.py
Recursive Ambiguity Context Accumulation (Mekanisme 1).

Tanggung jawab:
- Membangun akumulasi klarifikasi (original + history)
- Mengekstrak negative evidence (scope-aware)
- Memvalidasi relevansi input pendek
- Mengelola candidate scope (retained vs permanently excluded)
- Mendeteksi explicit cancellation
- Mem-persist state klarifikasi

Prinsip:
  SHORT INPUT != AUTOMATICALLY RELEVANT
  DROPPED CANDIDATE = PERMANENTLY EXCLUDED (tidak boleh kembali)
  NEGATION IS SCOPE-AWARE ("bukan X" != "X bukan?")
"""

import re
import logging
from datetime import datetime
from typing import List, Optional, Dict

from config import CONFIG
from models import (
    InputEnvelope, PendingState, PendingStatus, TopicKey,
    CandidateScore, ClarificationAccumulation, TokenQuality,
)

logger = logging.getLogger(__name__)

# ==========================================
# MODULE-LEVEL CONSTANTS
# ==========================================

# Single source of truth: indikator domain baru yang kuat.
# Digunakan oleh check_pending_relevance dan _analyze_turn (via import).
STRONG_NOVELTY_INDICATORS: frozenset = frozenset({
    "pointer", "reference", "coding", "program", "algoritma",
    "c++", "c#", "python", "java", "astronomi", "hukum", "matematika",
})

# Pola identifier simbolik yang tidak ditangkap \b\w+\b.
# "C++" -> ["c++"], bukan ["c"].
SYMBOL_IDENTITY_PATTERN = re.compile(r"[a-z0-9_]+(?:\+\+|\+|#)+", re.IGNORECASE)


# ==========================================
# CLARIFICATION STATE MANAGER
# ==========================================

class ClarificationStateManager:
    """
    Mengelola seluruh lifecycle ambiguity dan klarifikasi.
    
    memory_engine.py mendelegasikan semua logika state management ke sini,
    dan hanya menangani orchestration (panggil method, bangun result).
    
    Dependency: storage (untuk read/write PendingState dan TopicKey).
    """

    def __init__(self, storage):
        self.storage = storage

    # ==========================================
    # TOKEN EXTRACTION HELPERS
    # ==========================================

    def extract_input_tokens(self, envelope: InputEnvelope) -> set:
        """Single source of truth untuk tokenisasi input user."""
        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        return {t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}

    def extract_symbol_identity_tokens(self, raw_text: str) -> set:
        """
        Ekstrak identity token INCLUDING identifier simbolik.
        
        Masalah: "C++" melalui re.findall(r'\b\w+\b', ...) hanya menghasilkan
        ["c"] (1 karakter), lalu terfilter MIN_TOKEN_LENGTH=3 -> himpunan kosong.
        
        Fungsi ini menangkap identifier simbolik secara utuh ("c++", "c#", "f#")
        plus token alfanumerik pendek, sehingga evaluasi bukti tetap bisa jalan.
        """
        text = raw_text.lower()
        tokens = set(SYMBOL_IDENTITY_PATTERN.findall(text))
        tokens |= {t for t in re.findall(r'\b\w+\b', text) if t}
        return tokens

    def get_input_token_quality(self, envelope: InputEnvelope) -> Dict[str, TokenQuality]:
        """Helper: klasifikasi token kualitas (re-use SignalExtractor logic)."""
        from retrieval import SignalExtractor
        extractor = SignalExtractor()
        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        unique_tokens = [t for t in set(raw_tokens) if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
        return {t: extractor._classify_token(t) for t in unique_tokens}

    # ==========================================
    # PENDING EVIDENCE COLLECTION
    # ==========================================

    def collect_pending_evidence_tokens(
        self,
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> set:
        """
        Single source of truth untuk bukti milik Pending State.
        Menggabungkan TITLE + TOKEN PROFILE + HELPER DESCRIPTION + AGGREGATED HELPER
        dari seluruh candidate keys.
        """
        if all_keys is None:
            all_keys = self.storage.get_all_topic_keys()

        pending_tokens = set()
        for k in all_keys:
            if k.key_id in pending.candidate_keys:
                pending_tokens.update(k.token_profile.keys())
                title_words = re.findall(r'\b\w+\b', k.title.lower())
                pending_tokens.update(
                    w for w in title_words if len(w) >= CONFIG.signal.MIN_TOKEN_LENGTH
                )
                desc_words = re.findall(r'\b\w+\b', k.helper_description.lower())
                pending_tokens.update(
                    w for w in desc_words if len(w) >= CONFIG.signal.MIN_TOKEN_LENGTH
                )
                agg_words = re.findall(
                    r'\b\w+\b', (k.aggregated_helper_description or "").lower()
                )
                pending_tokens.update(
                    w for w in agg_words if len(w) >= CONFIG.signal.MIN_TOKEN_LENGTH
                )
        return pending_tokens

    # ==========================================
    # PENDING RELEVANCE CHECK
    # ==========================================

    def score_pending_relevance(
        self,
        envelope: InputEnvelope,
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> float:
        """
        Skor komposit relevansi pending untuk RANKING antar pending.
        Bukan untuk gating (gating dilakukan oleh check_pending_relevance).
        
        Komponen:
        - overlap_ratio : proporsi token input yang terjelaskan oleh bukti pending
        - overlap_bonus : penguat absolut agar pending dengan bukti lebih kaya menang
        """
        new_tokens = self.extract_input_tokens(envelope)
        if not new_tokens or len(new_tokens) < 3:
            return 0.5

        pending_tokens = self.collect_pending_evidence_tokens(pending, all_keys=all_keys)
        overlap = new_tokens.intersection(pending_tokens)
        overlap_ratio = len(overlap) / len(new_tokens)
        overlap_bonus = min(len(overlap), 5) * 0.01
        return overlap_ratio + overlap_bonus

    def detect_explicit_selection(self, raw_text: str, pending: PendingState) -> bool:
        """
        Deteksi apakah input pendek adalah pilihan eksplisit
        terhadap salah satu kandidat pending.

        Contoh pola yang dikenali:
            "nomor 1", "pilihan 2", "opsi 3", "1" (angka saja)
            "yang pertama", "yang kedua", "yang terakhir"
            "pertama", "kedua"

        Mengembalikan True jika index yang dipilih valid (1..N kandidat).
        """
        n_candidates = len(pending.candidate_keys)
        if n_candidates == 0:
            return False

        text_lower = raw_text.lower().strip()

        number_patterns = [
            r'\bnomor\s+(\d+)\b',
            r'\bpilihan\s+(\d+)\b',
            r'\boption\s+(\d+)\b',
            r'\boptions?\s+(\d+)\b',
            r'\bopsi\s+(\d+)\b',
            r'^\s*(\d+)\s*$',
        ]
        for pattern in number_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    idx = int(match.group(1))
                    if 1 <= idx <= n_candidates:
                        return True
                except ValueError:
                    continue

        ordinal_map = {
            "pertama": 1, "kesatu": 1, "kedua": 2, "ketiga": 3,
            "keempat": 4, "kelima": 5, "keenam": 6, "ketujuh": 7,
            "terakhir": n_candidates, "last": n_candidates,
            "first": 1, "second": 2, "third": 3,
        }
        for word, idx in ordinal_map.items():
            if re.search(rf'\b{re.escape(word)}\b', text_lower):
                if 1 <= idx <= n_candidates:
                    return True

        cardinal_map = {
            "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        }
        for word, idx in cardinal_map.items():
            if re.search(rf'\b{re.escape(word)}\b', text_lower):
                if 1 <= idx <= n_candidates:
                    return True

        return False

    def check_pending_relevance(
        self,
        envelope: InputEnvelope,
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> bool:
        """
        SHORT INPUT != AUTOMATICALLY RELEVANT.
        
        Urutan evaluasi input pendek:
            0. Tokenisasi kosong -> cek identity simbolik
            1. Benar-benar kosong -> RELEVANT (edge case)
            2. Semua low-information -> RELEVANT ("terserah", "mana?")
            3. Explicit selection -> RELEVANT ("nomor 1", "yang pertama")
            4. Overlap candidate evidence -> RELEVANT ("yang bisnis", "kopi")
            5. Strong novelty -> NOT RELEVANT ("C++", "pointer")
            6. Tanpa bukti -> NOT RELEVANT (jangan paksa)
        
        Urutan evaluasi input panjang:
            - overlap >= 2 tokens -> RELEVANT
            - overlap == 0 dan >= 3 tokens baru -> NOT RELEVANT (topic switch)
            - fallback: overlap_ratio >= SWITCH_THRESHOLD
        """
        new_tokens = self.extract_input_tokens(envelope)
        short_threshold = CONFIG.fast_memory.SHORT_PENDING_INPUT_THRESHOLD
        min_match = CONFIG.fast_memory.SHORT_PENDING_MIN_MATCH

        # ==========================================
        # JALUR INPUT PENDEK
        # ==========================================
        if not new_tokens or len(new_tokens) < short_threshold:
            # 0. Tokenisasi kosong: bedakan "benar-benar kosong"
            # dari "berisi identifier simbolik yang tidak lolos tokenisasi".
            if not new_tokens:
                raw_stripped = envelope.raw_text.strip()

                # 1. Benar-benar kosong (whitespace saja) -> edge case, tetap relevan.
                if not raw_stripped:
                    return True

                # 3. Explicit selection tetap diprioritaskan (mis. "1", "2").
                if self.detect_explicit_selection(raw_stripped, pending):
                    return True

                # Ekstrak identity simbolik + token pendek dari raw_text.
                identity_tokens = self.extract_symbol_identity_tokens(raw_stripped)

                from retrieval import SignalExtractor
                extractor = SignalExtractor()
                meaningful_identity = [
                    t for t in identity_tokens
                    if extractor._classify_token(t) != TokenQuality.LOW_INFORMATION
                ]

                # 2. Semua identity low-information (mis. "ya", "ok") -> relevan.
                if not meaningful_identity:
                    return True

                # 5. Strong novelty (mis. "c++", "c#") -> BUKAN jawaban pending.
                if any(t in STRONG_NOVELTY_INDICATORS for t in meaningful_identity):
                    return False

                # 4. Overlap dengan bukti candidate -> relevan.
                pending_tokens = self.collect_pending_evidence_tokens(pending, all_keys=all_keys)
                if set(meaningful_identity).intersection(pending_tokens):
                    return True

                # 6. Tidak ada bukti apapun -> jangan paksa relevan.
                return False

            # 2. Semua token low-information -> RELEVANT (weak fallback)
            quality_map = self.get_input_token_quality(envelope)
            meaningful = [
                t for t in new_tokens
                if quality_map.get(t, TokenQuality.NORMAL) != TokenQuality.LOW_INFORMATION
            ]
            if not meaningful:
                return True

            # 3. Explicit candidate selection -> RELEVANT
            if self.detect_explicit_selection(envelope.raw_text, pending):
                return True

            # 4. Token overlap dengan candidate evidence -> RELEVANT
            pending_tokens = self.collect_pending_evidence_tokens(pending, all_keys=all_keys)
            overlap = set(meaningful).intersection(pending_tokens)
            if len(overlap) >= min_match:
                return True

            # 5. Strong novelty indicator -> NOT RELEVANT
            has_strong_novelty = any(t in STRONG_NOVELTY_INDICATORS for t in meaningful)
            if has_strong_novelty:
                return False

            # 6. Tidak ada bukti apapun -> NOT RELEVANT
            return False

        # ==========================================
        # JALUR INPUT PANJANG
        # ==========================================
        pending_tokens = self.collect_pending_evidence_tokens(pending, all_keys=all_keys)
        overlap = new_tokens.intersection(pending_tokens)

        if len(overlap) >= 2:
            return True

        if len(overlap) == 0 and len(new_tokens) >= 3:
            return False

        overlap_ratio = len(overlap) / len(new_tokens) if new_tokens else 0.0
        return overlap_ratio >= CONFIG.fast_memory.PENDING_SWITCH_THRESHOLD

    def pause_pending(self, pending: PendingState):
        """Mengubah status Pending menjadi PAUSED."""
        pending.status = PendingStatus.PAUSED
        pending.last_updated_at = datetime.now()
        with self.storage.transaction() as tx:
            self.storage.save_pending_state(tx, pending)

    # ==========================================
    # EXPLICIT CANCELLATION DETECTION
    # ==========================================

    def detect_explicit_cancellation(self, envelope: InputEnvelope) -> bool:
        """
        Deteksi pembatalan / perpindahan topik eksplisit.
        
        "ganti topik", "lupakan", "batal", "bahas yang lain"
        
        Input seperti ini TIDAK boleh dipaksa menjadi jawaban ambiguity.
        """
        from retrieval import SignalExtractor
        extractor = SignalExtractor()
        signal = extractor.extract(envelope.raw_text, envelope.timestamp)
        return bool(signal.cancellation_intent)

    # ==========================================
    # CLARIFICATION ACCUMULATION
    # ==========================================

    def build_accumulation(
        self,
        envelope: InputEnvelope,
        pending: PendingState,
    ) -> ClarificationAccumulation:
        """
        Bangun paket evidence klarifikasi AKUMULATIF.
        
        SUMBER KEBENARAN = original_input + clarification_history.
        combined_text adalah DERIVED CACHE yang boleh diregenerasi.
        
        Round 0: original
        Round 1: original + answer1
        Round 2: original + answer1 + answer2
        
        Retrieval tidak boleh hanya membaca jawaban terakhir, karena
        "yang rencana kopi" sendiri terlalu miskin bukti.
        """
        cc = CONFIG.clarification

        history = list(pending.clarification_history or [])
        history.append(envelope.raw_text)
        if len(history) > cc.MAX_CLARIFICATION_HISTORY_ENTRIES:
            history = history[-cc.MAX_CLARIFICATION_HISTORY_ENTRIES:]

        parts = [pending.original_input] + history
        combined = " ".join(p for p in parts if p)
        if len(combined) > cc.CLARIFICATION_HISTORY_MAX_CHARS:
            tail_budget = (
                cc.CLARIFICATION_HISTORY_MAX_CHARS - len(pending.original_input) - 1
            )
            tail = " ".join(history)
            if tail_budget > 0:
                tail = tail[-tail_budget:]
                combined = f"{pending.original_input} {tail}"
            else:
                combined = pending.original_input[:cc.CLARIFICATION_HISTORY_MAX_CHARS]

        from retrieval import SignalExtractor
        extractor = SignalExtractor()
        signal = extractor.extract(combined, envelope.timestamp)

        excluded_tokens = sorted(
            set(pending.excluded_tokens or []) | set(signal.excluded_tokens or [])
        )

        return ClarificationAccumulation(
            original_input=pending.original_input,
            clarification_history=history,
            combined_text=combined,
            excluded_tokens=excluded_tokens,
            excluded_candidate_keys=list(pending.excluded_candidate_keys or []),
            negation_scopes=list(signal.negation_scopes or []),
            round_number=(pending.clarification_round or 1),
            retained_candidate_keys=list(pending.candidate_keys or []),
        )

    # ==========================================
    # STATE PERSISTENCE
    # ==========================================

    def persist_state(
        self,
        pending: PendingState,
        clarification: Optional[ClarificationAccumulation],
        result,  # RetrievalResult
        envelope: InputEnvelope,
        question: str,
    ) -> None:
        """
        Pending BERKEMBANG, bukan dibuat ulang.
        
        [P3 FIX — Persistent Candidate Scope]
        Kandidat yang telah gugur (dropped by narrowing) ditambahkan ke
        excluded_candidate_keys secara PERMANEN. Mereka tidak boleh bebas
        kembali pada putaran berikutnya.
        
        Perbedaan:
        - DROPPED: skor terlalu rendah, tidak lolos narrowing
        - EXCLUDED: user secara eksplisit menolak ("bukan X")
        
        Keduanya masuk excluded_candidate_keys dan tidak akan muncul lagi.
        """
        if clarification is not None:
            pending.clarification_history = list(clarification.clarification_history)
            pending.combined_clarification_text = clarification.combined_text
            pending.excluded_tokens = list(clarification.excluded_tokens)

        # Collect all excluded (from user negation)
        excluded = set(pending.excluded_candidate_keys or [])
        excluded.update(result.excluded_candidate_keys or [])

        # [P3 FIX] Also exclude candidates that were dropped by narrowing.
        # Old candidates not in new candidates are permanently excluded.
        old_keys = set(pending.candidate_keys or [])
        new_keys = {c.key_id for c in result.candidates}
        dropped = old_keys - new_keys
        excluded.update(dropped)

        pending.excluded_candidate_keys = sorted(excluded)

        # Update candidate shortlist
        new_key_list = [c.key_id for c in result.candidates]
        if new_key_list:
            pending.candidate_keys = new_key_list
            pending.candidate_scores = {
                c.key_id: c.final_score for c in result.candidates
            }
            hist = list(pending.candidate_history or [])
            hist.append(new_key_list)
            pending.candidate_history = hist

        pending.assistant_question = question
        pending.last_updated_at = envelope.timestamp
