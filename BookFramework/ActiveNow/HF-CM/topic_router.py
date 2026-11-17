"""
topic_router.py
Local-First Topic Routing (Batch D).

Tanggung jawab:
    Memutuskan apakah input baru melanjutkan TopicKey aktif (LOCAL MATCH)
    atau harus mencari topik baru via global search.

Prinsip kunci (dari dokumentasi arsitek):
    "cek dulu TopicKey terakhir/state terakhir; jika cocok jangan langsung
     mencari global."

    Alur:
        NEW INPUT
            ↓
        SESSION ACTIVE STATE
            ↓
        active_topic_key_id ada?
            ├── tidak → GLOBAL ROUTING
            └── ada → LOCAL TOPIC EVALUATION
                        ↓
                    score >= ACTIVE_TOPIC_MATCH_THRESHOLD?
                        ├── ya → ACTIVE_LOCAL_MATCH
                        └── tidak → GLOBAL SEARCH
                                        ↓
                                    ranking global
                                        ↓
                                    GLOBAL_MATCH / NEW_TOPIC / UNRESOLVED

Kontrak:
    - Stateless: tidak memegang storage, tidak mengubah state.
    - Caller (memory_engine) resolve TopicKey object dan inject ke route().
    - Continuation intent TIDAK ditangani di sini; itu tugas
      memory_engine._apply_continuation_protection() setelah routing.
    - Cancellation intent langsung short-circuit ke NEW_TOPIC.

Threshold (dari config.AssociationConstants):
    - ACTIVE_TOPIC_MATCH_THRESHOLD = 0.45 (local match minimum)
    - LOCAL_VS_GLOBAL_MARGIN = 0.03 (margin vs global best)

Route types (dari models.TopicRouteType):
    - ACTIVE_LOCAL_MATCH : TopicKey aktif cocok, local-first menang
    - GLOBAL_MATCH       : Topik ditemukan via pencarian global
    - NEW_TOPIC          : Tidak ada yang cocok, topik baru
    - UNRESOLVED         : Retrieval tidak yakin (ambiguous / insufficient)
"""

import logging
import re
from typing import Optional, List, Tuple

from config import CONFIG
from models import (
    InputEnvelope,
    SessionActiveState,
    TopicKey,
    TopicRouteResult,
    TopicRouteType,
)
from retrieval import RetrievalResult, DecisionType


logger = logging.getLogger(__name__)


class TopicRouter:
    """
    Local-first topic router.

    Dipanggil oleh memory_engine SEBELUM decision gate, sebagai lapisan
    observability tambahan untuk menjawab:
        "Mengapa sistem memilih topik itu?"

    Dependency:
        - Tidak ada (pure decision function).

    Kontrak:
        - Method route() adalah PURE function.
        - Tidak mengubah storage, tidak mengubah state.
        - Returns TopicRouteResult dengan reason untuk audit.
    """

    def __init__(self):
        self._cfg = CONFIG.association
        self._signal_cfg = CONFIG.signal

    # ==========================================
    # ENTRY POINT
    # ==========================================

    def route(
        self,
        envelope: InputEnvelope,
        session_state: Optional[SessionActiveState],
        active_topic_key: Optional[TopicKey],
        retrieval_result: RetrievalResult,
    ) -> TopicRouteResult:
        """
        Tentukan route topik untuk input baru.

        Args:
            envelope: InputEnvelope dari user.
            session_state: SessionActiveState saat ini (bisa None).
            active_topic_key: TopicKey object yang sedang aktif (bisa None).
                              Caller harus resolve dari session_state.active_topic_key_id.
            retrieval_result: Hasil retrieval global (sudah dijalankan caller).

        Returns:
            TopicRouteResult dengan route_type dan observability fields.

        Decision flow:
            1. Cancellation intent? → NEW_TOPIC (short-circuit)
            2. Active state exists?
               - No → extract route from retrieval
               - Yes → compute local score
                       - score >= threshold? → ACTIVE_LOCAL_MATCH
                       - score < threshold? → compare with global best
                           - local margin OK? → ACTIVE_LOCAL_MATCH
                           - else → extract route from retrieval
            3. Fallback → extract route from retrieval
        """
        # Extract signal (reuse retrieval's signal_profile jika ada)
        signal = retrieval_result.signal_profile

        # ==========================================
        # 1. CANCELLATION INTENT SHORT-CIRCUIT
        # ==========================================
        # Jika user eksplisit membatalkan ("ganti topik"),
        # jangan paksa active state. Langsung NEW_TOPIC.
        if signal is not None and signal.cancellation_intent:
            return TopicRouteResult(
                route_type=TopicRouteType.NEW_TOPIC,
                selected_topic_key_id=None,
                active_topic_score=0.0,
                local_attempted=False,
                local_matched=False,
                global_search_performed=False,
                reason="cancellation intent detected -> force new topic",
                global_candidates=[],
            )

        # ==========================================
        # 2. LOCAL STATE EVALUATION
        # ==========================================
        has_active_state = (
            session_state is not None
            and session_state.active_topic_key_id is not None
            and active_topic_key is not None
        )

        local_attempted = has_active_state
        local_matched = False
        active_score = 0.0

        if has_active_state:
            active_score = self._compute_local_score(
                envelope=envelope,
                active_key=active_topic_key,
                signal=signal,
            )

            # Primary check: absolute threshold
            if active_score >= self._cfg.ACTIVE_TOPIC_MATCH_THRESHOLD:
                local_matched = True
                return TopicRouteResult(
                    route_type=TopicRouteType.ACTIVE_LOCAL_MATCH,
                    selected_topic_key_id=session_state.active_topic_key_id,
                    active_topic_score=active_score,
                    local_attempted=True,
                    local_matched=True,
                    global_search_performed=False,
                    reason=(
                        f"local score {active_score:.3f} >= "
                        f"ACTIVE_TOPIC_MATCH_THRESHOLD "
                        f"{self._cfg.ACTIVE_TOPIC_MATCH_THRESHOLD:.2f}"
                    ),
                    global_candidates=[],
                )

            # Secondary check: margin vs global best
            # (jika local score rendah tapi masih lebih tinggi dari global best
            #  dengan margin, tetap local win)
            global_best_score = self._get_global_best_score(retrieval_result)
            margin = active_score - global_best_score

            if margin >= self._cfg.LOCAL_VS_GLOBAL_MARGIN and active_score > 0.0:
                local_matched = True
                return TopicRouteResult(
                    route_type=TopicRouteType.ACTIVE_LOCAL_MATCH,
                    selected_topic_key_id=session_state.active_topic_key_id,
                    active_topic_score=active_score,
                    local_attempted=True,
                    local_matched=True,
                    global_search_performed=True,
                    reason=(
                        f"local score {active_score:.3f} vs global best "
                        f"{global_best_score:.3f} (margin {margin:.3f} >= "
                        f"{self._cfg.LOCAL_VS_GLOBAL_MARGIN:.2f})"
                    ),
                    global_candidates=self._extract_global_candidates(
                        retrieval_result
                    ),
                )

        # ==========================================
        # 3. FALLBACK TO GLOBAL ROUTE
        # ==========================================
        route_type, selected_key_id, global_candidates = (
            self._extract_route_from_retrieval(retrieval_result)
        )

        # Build reason string
        if local_attempted:
            reason = (
                f"local score {active_score:.3f} < "
                f"ACTIVE_TOPIC_MATCH_THRESHOLD {self._cfg.ACTIVE_TOPIC_MATCH_THRESHOLD:.2f}; "
                f"fell back to global search"
            )
        else:
            reason = "no active state -> global search"

        return TopicRouteResult(
            route_type=route_type,
            selected_topic_key_id=selected_key_id,
            active_topic_score=active_score,
            local_attempted=local_attempted,
            local_matched=False,
            global_search_performed=True,
            reason=reason,
            global_candidates=global_candidates,
        )

    # ==========================================
    # LOCAL SCORE COMPUTATION
    # ==========================================

    def _compute_local_score(
        self,
        envelope: InputEnvelope,
        active_key: TopicKey,
        signal,
    ) -> float:
        """
        Hitung skor kecocokan antara input user dan active TopicKey.

        Formula:
            base = token overlap ratio (|intersection| / |user_tokens|)
            + helper bonus (jika overlap dengan aggregated helper)
            + cooc bonus (jika ada co-occurrence match)

        Clamp ke [0.0, 1.0].

        Returns 0.0 jika tidak ada signal / token.
        """
        if active_key is None:
            return 0.0

        # Extract user tokens (meaningful)
        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        user_tokens = [
            t for t in raw_tokens if len(t) >= self._signal_cfg.MIN_TOKEN_LENGTH
        ]

        if not user_tokens:
            return 0.0

        user_set = set(user_tokens)

        # Token overlap dengan TopicKey token_profile
        topic_tokens = set(active_key.token_profile.keys()) if active_key.token_profile else set()
        intersection = user_set.intersection(topic_tokens)

        if topic_tokens:
            base_score = len(intersection) / max(len(user_set), 1)
        else:
            base_score = 0.0

        # Bonus: overlap dengan helper description (own atau aggregated)
        helper_bonus = 0.0
        helper_text = (
            (active_key.aggregated_helper_description or "")
            + " "
            + (active_key.helper_description or "")
        ).lower()
        if helper_text.strip():
            helper_tokens = {
                t for t in re.findall(r'\b\w+\b', helper_text)
                if len(t) >= self._signal_cfg.MIN_TOKEN_LENGTH
            }
            if helper_tokens:
                helper_overlap = user_set.intersection(helper_tokens)
                helper_bonus = 0.15 * (len(helper_overlap) / max(len(helper_tokens), 1))

        # Bonus: co-occurrence match
        cooc_bonus = 0.0
        if signal is not None and signal.window_pair_strength:
            active_cooc = active_key.co_occurrence_profile or {}
            matched_pairs = sum(
                strength
                for pair, strength in signal.window_pair_strength.items()
                if pair in active_cooc
            )
            total_pairs = sum(signal.window_pair_strength.values()) or 1.0
            cooc_bonus = 0.10 * (matched_pairs / total_pairs)

        # Continuity bonus (jika active state sudah stabil)
        continuity_bonus = 0.0
        if signal is not None and getattr(signal, "continuation_intent", False):
            continuity_bonus = 0.05

        final = base_score + helper_bonus + cooc_bonus + continuity_bonus
        return max(0.0, min(1.0, final))

    # ==========================================
    # GLOBAL ROUTE EXTRACTION
    # ==========================================

    def _get_global_best_score(self, retrieval_result: RetrievalResult) -> float:
        """Ambil final_score dari kandidat global terbaik."""
        if not retrieval_result.candidates:
            return 0.0
        return retrieval_result.candidates[0].final_score

    def _extract_global_candidates(
        self,
        retrieval_result: RetrievalResult,
        limit: int = 3,
    ) -> List[str]:
        """Ambil top-N key_id dari retrieval candidates untuk observability."""
        if not retrieval_result.candidates:
            return []
        return [c.key_id for c in retrieval_result.candidates[:limit]]

    def _extract_route_from_retrieval(
        self,
        retrieval_result: RetrievalResult,
    ) -> Tuple[TopicRouteType, Optional[str], List[str]]:
        """
        Terjemahkan RetrievalResult.decision_type ke TopicRouteType.

        Mapping:
            CLEAR                → GLOBAL_MATCH + selected_key_id
            AMBIGUOUS            → UNRESOLVED (retrieval tidak yakin)
            INSUFFICIENT_EVIDENCE→ UNRESOLVED
            CONTRADICTORY_EVIDENCE → UNRESOLVED
            NO_MATCH             → NEW_TOPIC
            UNRESOLVED_CONTINUATION → UNRESOLVED
            INSUFFICIENT         → UNRESOLVED

        Returns:
            (route_type, selected_key_id, global_candidates)
        """
        decision = retrieval_result.decision_type
        global_candidates = self._extract_global_candidates(retrieval_result)

        if decision == DecisionType.CLEAR:
            return (
                TopicRouteType.GLOBAL_MATCH,
                retrieval_result.selected_key_id,
                global_candidates,
            )

        if decision == DecisionType.NO_MATCH:
            return (
                TopicRouteType.NEW_TOPIC,
                None,
                global_candidates,
            )

        # Semua decision "tidak yakin" → UNRESOLVED
        if decision in (
            DecisionType.AMBIGUOUS,
            DecisionType.INSUFFICIENT,
            DecisionType.INSUFFICIENT_EVIDENCE,
            DecisionType.CONTRADICTORY_EVIDENCE,
            DecisionType.UNRESOLVED_CONTINUATION,
        ):
            return (
                TopicRouteType.UNRESOLVED,
                None,
                global_candidates,
            )

        # Fallback: decision tidak dikenal
        logger.warning(f"TopicRouter: unknown decision_type={decision}, treat as UNRESOLVED")
        return (
            TopicRouteType.UNRESOLVED,
            None,
            global_candidates,
        )
