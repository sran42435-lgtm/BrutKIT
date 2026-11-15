"""
trend_navigation.py
Trend Navigation (Mekanisme 5 — bagian trend).

Tanggung jawab:
- Menentukan trend candidates untuk continuation tanpa target spesifik
- Mode A: TopicKey trend (ketika topik belum jelas)
- Mode B: Fragment trend (ketika topik jelas tapi bagian belum)
- Depth-aware expansion (jangan expand di MAX_DEPTH)
- Temporal trend berbasis ACTIVITY DENSITY (Prioritas 4 dari review)

Prinsip kunci:
    Trend = NAVIGATION EVIDENCE, bukan identitas final.
    Banyak child (branch richness) = SINYAL UTAMA.
    Recency & temporal = MODIFIER, bukan penentu utama.

Prioritas 4 (dari review):
    SEBELUM:
        "ada fragment dalam range?" -> temporal = 1.0
        "tidak ada?"                -> temporal = 0.0
    SESUDAH:
        "berapa banyak fragment dalam range?" -> normalized density
        Topic A: 10 fragment kemarin -> temporal ~0.5
        Topic B: 1 fragment kemarin  -> temporal ~0.05

    Ini membuat "lanjutkan topik kemarin" benar-benar memilih topik
    yang paling aktif kemarin, bukan topik apa saja yang kebetulan
    punya minimal 1 fragment.

Formula trend_score (sinkron dengan storage.py):
    trend_score = (
        CHILD_COUNT_WEIGHT * normalized_child_count
        + RECENCY_WEIGHT * recency_score
        + TEMPORAL_WEIGHT * temporal_density
    )

    dengan normalized_child_count = min(count, CAP) / CAP

Flow dalam memory_engine:
    _resolve_trend_candidates(envelope, ...) ->
        self.trend_navigator.resolve_trend_candidates(...)
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

from config import CONFIG
from models import (
    EngineDecision, DecisionType, InputEnvelope,
    SessionActiveState, TrendCandidate, TrendNodeType,
)
from retrieval import RetrievalResult
from storage import MemoryStorage

logger = logging.getLogger(__name__)


class TrendNavigator:
    """
    Manajer trend navigation (Mekanisme 5).

    Dependency:
        - storage: untuk query trending topics / fragments / children
        - retrieval_engine: untuk temporal_resolver (resolve temporal signals)

    Kontrak:
        - resolve_trend_candidates() adalah PURE function (tidak mengubah state).
        - Returns list TrendCandidate, bisa kosong.
        - Trend hanya relevan saat:
            * continuation intent terdeteksi
            * decision type AMBIGUOUS / UNRESOLVED_CONTINUATION / INSUFFICIENT_EVIDENCE
    """

    def __init__(self, storage: MemoryStorage, retrieval_engine):
        self.storage = storage
        self.retrieval = retrieval_engine

    # ==========================================
    # ENTRY POINT
    # ==========================================

    def resolve_trend_candidates(
        self,
        decision: EngineDecision,
        retrieval: RetrievalResult,
        envelope: InputEnvelope,
        session_state: Optional[SessionActiveState],
    ) -> List[TrendCandidate]:
        """
        Tentukan trend candidates untuk navigasi.

        Dipanggil oleh memory_engine saat:
            - continuation_intent = True
            - decision_type adalah salah satu dari {AMBIGUOUS, UNRESOLVED_CONTINUATION,
              INSUFFICIENT_EVIDENCE}

        Mode A (topik belum jelas):
            Ambil top-N TopicKey, recompute temporal dengan activity density.

        Mode B (topik jelas, bagian belum):
            Ambil top-N fragments di depth tertentu, recompute temporal.

        Returns list kosong jika tidak ada continuation intent, decision sudah CLEAR,
        atau depth limit tercapai.
        """
        # Guard: hanya aktif untuk continuation intent
        if not retrieval.continuation_intent:
            return []

        # Guard: hanya untuk decision types tertentu
        if decision.decision_type not in (
            DecisionType.AMBIGUOUS,
            DecisionType.UNRESOLVED_CONTINUATION,
            DecisionType.INSUFFICIENT_EVIDENCE,
        ):
            return []

        # Resolve temporal range dari signal
        temporal_range = self._resolve_temporal_range(retrieval, envelope)

        # Mode B: topik sudah jelas -> fragment trend di depth tertentu
        if retrieval.selected_key_id:
            depth = self._compute_fragment_trend_depth(session_state, retrieval)
            if depth > CONFIG.hierarchy.MAX_FRAGMENT_DEPTH:
                return []
            candidates = self.storage.get_trending_fragments(
                topic_key_id=retrieval.selected_key_id,
                depth=depth,
                reference_time=envelope.timestamp,
                temporal_range=temporal_range,
            )
            if temporal_range is not None:
                candidates = self._recompute_fragment_temporal_density(
                    candidates, temporal_range
                )
            return candidates

        # Mode A: topik belum jelas -> TopicKey trend
        candidates = self.storage.get_trending_topic_keys(
            reference_time=envelope.timestamp,
            temporal_range=temporal_range,
        )
        if temporal_range is not None:
            candidates = self._recompute_topic_temporal_density(
                candidates, temporal_range
            )
        return candidates

    # ==========================================
    # TEMPORAL RANGE RESOLUTION
    # ==========================================

    def _resolve_temporal_range(
        self,
        retrieval: RetrievalResult,
        envelope: InputEnvelope,
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Resolve temporal range dari signal profile.

        Returns None jika tidak ada temporal signal.
        """
        if retrieval.signal_profile is None:
            return None
        return self.retrieval.temporal_resolver.resolve(
            retrieval.signal_profile.temporal_signals, envelope.timestamp
        )

    # ==========================================
    # DEPTH COMPUTATION (Mode B)
    # ==========================================

    def _compute_fragment_trend_depth(
        self,
        session_state: Optional[SessionActiveState],
        retrieval: RetrievalResult,
    ) -> int:
        """
        Hitung depth untuk fragment trend (Mode B).

        Aturan:
            - Default: NEW_BRANCH_DEPTH (biasanya 1 = section utama)
            - Jika session_state aktif dan topicKey-nya sama dengan selected:
              depth = active_depth + 1 (coba ekspansi ke level berikut)
            - Clamp: depth <= MAX_FRAGMENT_DEPTH
        """
        depth = CONFIG.hierarchy.NEW_BRANCH_DEPTH
        if (session_state is not None
                and session_state.active_topic_key_id == retrieval.selected_key_id):
            depth = min(
                session_state.active_depth + 1,
                CONFIG.hierarchy.MAX_FRAGMENT_DEPTH,
            )
        return depth

    # ==========================================
    # TEMPORAL DENSITY RECOMPUTATION
    # ==========================================

    def _recompute_topic_temporal_density(
        self,
        candidates: List[TrendCandidate],
        temporal_range: Tuple[datetime, datetime],
    ) -> List[TrendCandidate]:
        """
        [Prioritas 4] Recompute temporal_score untuk setiap TopicKey candidate
        berdasarkan ACTIVITY DENSITY (bukan binary).

        Alur:
            1. Untuk setiap candidate, fetch fragment milik TopicKey tersebut.
            2. Hitung fragment yang masuk temporal_range.
            3. Normalize count ke [0, 1] via TREND_COUNT_NORMALIZATION_CAP.
            4. Recompute trend_score dengan temporal_score baru.
            5. Re-sort candidates.

        Kenapa tidak pakai temporal_score dari storage:
            Storage menghitung temporal = 1.0 jika > 0 fragment match,
            0.0 jika tidak. Ini kehilangan informasi "berapa banyak".
        """
        if not candidates:
            return candidates

        tc = CONFIG.trend
        start_t, end_t = temporal_range
        # SQLite temporal_index naive — strip tzinfo untuk perbandingan.
        start_naive = start_t.replace(tzinfo=None)
        end_naive = end_t.replace(tzinfo=None)

        recomputed: List[TrendCandidate] = []
        for cand in candidates:
            if cand.node_type != TrendNodeType.TOPIC_KEY:
                recomputed.append(cand)
                continue

            # Fetch fragments milik TopicKey ini
            frags = self.storage.get_fragment_metadata_by_key(cand.node_id)

            # Hitung berapa fragment yang masuk temporal_range
            in_range_count = 0
            for f in frags:
                if f.temporal_index is None:
                    continue
                ti = f.temporal_index
                if ti.tzinfo is not None:
                    ti = ti.replace(tzinfo=None)
                if start_naive <= ti <= end_naive:
                    in_range_count += 1

            # Normalize ke [0, 1] dengan cap
            cap = max(1, tc.TREND_COUNT_NORMALIZATION_CAP)
            new_temporal = min(1.0, in_range_count / cap)

            # Recompute trend_score
            new_trend = self._compute_trend_score(
                child_count=cand.child_count,
                fragment_count=cand.fragment_count,
                recency_score=cand.recency_score,
                temporal_score=new_temporal,
            )

            # Rebuild candidate dengan skor baru
            recomputed.append(cand.model_copy(update={
                "temporal_score": new_temporal,
                "trend_score": new_trend,
            }))

        # Re-sort berdasarkan trend_score baru
        recomputed.sort(key=lambda c: c.trend_score, reverse=True)

        # Re-limit sesuai TREND_TOPIC_LIMIT
        return recomputed[:tc.TREND_TOPIC_LIMIT]

    def _recompute_fragment_temporal_density(
        self,
        candidates: List[TrendCandidate],
        temporal_range: Tuple[datetime, datetime],
    ) -> List[TrendCandidate]:
        """
        [Prioritas 4] Recompute temporal_score untuk fragment candidates
        berdasarkan ACTIVITY DENSITY (child count dalam range).

        Alur:
            1. Untuk setiap candidate fragment, fetch children.
            2. Hitung children yang masuk temporal_range.
            3. Normalize count ke [0, 1].
            4. Recompute trend_score.
            5. Re-sort.
        """
        if not candidates:
            return candidates

        tc = CONFIG.trend
        start_t, end_t = temporal_range
        start_naive = start_t.replace(tzinfo=None)
        end_naive = end_t.replace(tzinfo=None)

        recomputed: List[TrendCandidate] = []
        for cand in candidates:
            if cand.node_type != TrendNodeType.FRAGMENT:
                recomputed.append(cand)
                continue

            # Fetch children dari fragment ini
            children = self.storage.get_child_fragments(cand.node_id)

            # Hitung children yang masuk range
            in_range_count = 0
            for child in children:
                if child.temporal_index is None:
                    continue
                ti = child.temporal_index
                if ti.tzinfo is not None:
                    ti = ti.replace(tzinfo=None)
                if start_naive <= ti <= end_naive:
                    in_range_count += 1

            # Normalize
            cap = max(1, tc.TREND_COUNT_NORMALIZATION_CAP)
            new_temporal = min(1.0, in_range_count / cap)

            new_trend = self._compute_trend_score(
                child_count=cand.child_count,
                fragment_count=cand.fragment_count,
                recency_score=cand.recency_score,
                temporal_score=new_temporal,
            )

            recomputed.append(cand.model_copy(update={
                "temporal_score": new_temporal,
                "trend_score": new_trend,
            }))

        recomputed.sort(key=lambda c: c.trend_score, reverse=True)
        return recomputed[:tc.TREND_FRAGMENT_LIMIT]

    # ==========================================
    # TREND SCORE COMPUTATION
    # ==========================================

    def _compute_trend_score(
        self,
        child_count: int,
        fragment_count: int,
        recency_score: float,
        temporal_score: float,
    ) -> float:
        """
        Recompute trend_score dengan formula yang sama seperti storage.

        Sinkron dengan storage._compute_trend_score agar tidak terjadi
        inkonsistensi ranking.

        Formula:
            count_signal = child_count + fragment_count
            normalized = min(count_signal, CAP) / CAP
            trend = (
                CHILD_COUNT_WEIGHT * normalized
                + RECENCY_WEIGHT * recency
                + TEMPORAL_WEIGHT * temporal
            )
        """
        tc = CONFIG.trend
        count_signal = child_count + fragment_count
        cap = max(1, tc.TREND_COUNT_NORMALIZATION_CAP)
        normalized = min(count_signal, cap) / float(cap)
        return (
            tc.TREND_CHILD_COUNT_WEIGHT * normalized
            + tc.TREND_RECENCY_WEIGHT * max(0.0, min(1.0, recency_score))
            + tc.TREND_TEMPORAL_WEIGHT * max(0.0, min(1.0, temporal_score))
        )
