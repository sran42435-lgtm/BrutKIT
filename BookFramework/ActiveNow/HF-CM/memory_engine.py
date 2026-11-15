"""
memory_engine.py
Global Orchestrator & State Machine (Layer 16).

[ASSOCIATIVE EVIDENCE LOCKING]
EVIDENCE BEFORE MUTATION:
    ANALYZE -> SCORE -> VALIDATE EVIDENCE -> DECIDE -> BARU UPDATE MEMORY

[REASONING PURITY]
- final_score != turn_continuity_score != relationship_strength
- Short pending input divalidasi, bukan otomatis dianggap relevan.
- graph_score fragment dihitung dari edge nyata.

[HIERARCHICAL FRAGMENT EXPANSION - 5 MEKANISME]
M1  Recursive Ambiguity Context Accumulation
M2  Continuation Intent Protection
M3  Hierarchical Fragment Identity & Helper Propagation
M4  Cross-Key Composite Fragment
M5  Tree + Depth + Trend

URUTAN WAJIB:
    EXTRACT METADATA (dari exact text) -> COMPRESS -> PERSIST
    -> COMMIT -> PLACE -> PROPAGATE

[REFACTOR BATCH 1]
Logika Mekanisme 1 (Clarification State Management) dipindahkan ke
clarification_state.py. memory_engine.py sekarang mendelegasikan semua
operasi pending/clarification ke ClarificationStateManager.

[REFACTOR BATCH 2]
Fragment Identity Extraction dipindahkan ke fragment_identity.py.
Identitas fragment sekarang dianalisis dari USER + AI RESPONSE (bukan
hanya user input), menghasilkan fragment_key + helper_description
yang lebih representatif.

[REFACTOR BATCH 3]
Fragment Placement dipindahkan ke fragment_hierarchy.py.
Parent selection sekarang memakai FragmentIdentity (Batch 2) sebagai
PRIMARY signal, dengan retrieval.signal_profile sebagai fallback.

[REFACTOR BATCH 4]
Helper Propagation dipindahkan ke helper_propagation.py.
Propagasi child -> parent -> root -> TopicKey dikelola oleh
HelperPropagationManager.

[REFACTOR BATCH 5]
Composite Detection dipindahkan ke composite_memory.py.
Analisis multi-source composite dikelola oleh CompositeMemoryAnalyzer.
Forwarding method _detect_composite_sources dipertahankan untuk
backward compatibility dengan test suite.

[REFACTOR BATCH 6]
Trend Navigation dipindahkan ke trend_navigation.py.
TrendNavigator mengelola Mode A (TopicKey trend) dan Mode B (Fragment
trend), dengan perbaikan Prioritas 4: temporal score berbasis activity
density (bukan binary match).
"""

import logging
import re
import uuid
import statistics
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Set

from config import CONFIG
from models import (
    InputEnvelope, EngineResult, EngineDecision, DecisionType,
    MemoryClassification, PendingState, PendingStatus, TopicKey,
    FragmentMetadata, CompressedPayload, MemoryGraphEdge, RelationshipType,
    CandidateScore, TurnAnalysisResult, ReconstructionStatus,
    RetrievedMemory, TokenQuality, EvidenceFamily,
    SessionActiveState, FragmentPlacementResult, FragmentPlacementKind,
    HelperPropagationResult, CompositeSourceAnalysis, FragmentSourceContribution,
    TrendCandidate, TrendNodeType, ClarificationAccumulation, FragmentCandidateScore
)
from storage import MemoryStorage
from retrieval import RetrievalEngine, RetrievalResult
from compression import PayloadCompressor, DecompressionError, IntegrityError
from clarification_state import ClarificationStateManager, STRONG_NOVELTY_INDICATORS
from fragment_identity import FragmentIdentityExtractor, FragmentIdentity
from fragment_hierarchy import FragmentHierarchyManager
from helper_propagation import HelperPropagationManager
from composite_memory import CompositeMemoryAnalyzer
from trend_navigation import TrendNavigator

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Otak utama sistem memori asosiatif."""

    _SIGNAL_FAMILY: Dict[str, EvidenceFamily] = {
        "token": EvidenceFamily.LEXICAL,
        "unique_token": EvidenceFamily.LEXICAL,
        "co_occurrence": EvidenceFamily.LEXICAL,
        "description": EvidenceFamily.CONTEXTUAL,
        "temporal": EvidenceFamily.CONTEXTUAL,
        "continuity": EvidenceFamily.CONTEXTUAL,
        "graph": EvidenceFamily.RELATIONAL,
    }

    def __init__(self, storage: MemoryStorage, retrieval: RetrievalEngine,
                 compressor: PayloadCompressor, context_builder=None, llm_client=None):
        self.storage = storage
        self.retrieval = retrieval
        self.compressor = compressor
        self.context_builder = context_builder
        self.llm_client = llm_client
        # [REFACTOR BATCH 1] ClarificationStateManager
        self.clarification_state = ClarificationStateManager(storage)
        # [REFACTOR BATCH 2] FragmentIdentityExtractor
        self.identity_extractor = FragmentIdentityExtractor()
        # [REFACTOR BATCH 3] FragmentHierarchyManager
        self.hierarchy_manager = FragmentHierarchyManager(storage)
        # [REFACTOR BATCH 4] HelperPropagationManager
        self.helper_propagator = HelperPropagationManager(storage)
        # [REFACTOR BATCH 5] CompositeMemoryAnalyzer
        self.composite_analyzer = CompositeMemoryAnalyzer(storage)
        # [REFACTOR BATCH 6] TrendNavigator (dengan temporal density recomputation)
        self.trend_navigator = TrendNavigator(storage, retrieval)

    def process_input(self, envelope: InputEnvelope) -> EngineResult:
        """Main Entry Point. Memproses satu turn percakapan secara utuh."""
        session_id = envelope.session_id
        ref_time = envelope.timestamp

        self.storage.ensure_session_exists(session_id, ref_time)
        session_state = self.storage.get_session_state(session_id)

        active_pending = self.storage.get_active_pending_state(session_id)

        if active_pending:
            created_at = active_pending.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=ref_time.tzinfo)
            age_seconds = (ref_time - created_at).total_seconds()
            if age_seconds > CONFIG.fast_memory.PENDING_TTL_SECONDS:
                active_pending.status = PendingStatus.EXPIRED
                with self.storage.transaction() as tx:
                    self.storage.save_pending_state(tx, active_pending)
                active_pending = None
                logger.info("PENDING STATE: Expired karena TTL.")

        context_text = envelope.raw_text
        is_cross_session_pending = False
        clarification_ctx: Optional[ClarificationAccumulation] = None

        explicit_cancel = self.clarification_state.detect_explicit_cancellation(envelope)
        if explicit_cancel and active_pending:
            self.clarification_state.pause_pending(active_pending)
            active_pending = None

        if active_pending:
            is_relevant = self.clarification_state.check_pending_relevance(envelope, active_pending)
            if is_relevant:
                clarification_ctx = self.clarification_state.build_accumulation(
                    envelope, active_pending
                )
                context_text = clarification_ctx.combined_text
            else:
                self.clarification_state.pause_pending(active_pending)
                active_pending = None
        elif not explicit_cancel:
            global_pendings = self.storage.get_global_active_pendings()
            keys_for_pending = self.storage.get_all_topic_keys()
            best_pending = None
            best_score = -1.0

            for gp in global_pendings:
                if gp.session_id == session_id:
                    continue
                if not self.clarification_state.check_pending_relevance(
                    envelope, gp, all_keys=keys_for_pending
                ):
                    continue
                score = self.clarification_state.score_pending_relevance(
                    envelope, gp, all_keys=keys_for_pending
                )
                if score > best_score:
                    best_score = score
                    best_pending = gp

            if best_pending:
                active_pending = best_pending
                is_cross_session_pending = True
                clarification_ctx = self.clarification_state.build_accumulation(
                    envelope, best_pending
                )
                context_text = clarification_ctx.combined_text

        all_keys = self.storage.get_all_topic_keys()
        active_topic_id, continuity_score = self._calculate_active_state(session_id)
        relevant_fragments = self._get_relevant_fragments(all_keys)
        global_token_counts = self.storage.get_global_token_counts()
        graph_evidence = self._build_graph_evidence(all_keys, relevant_fragments, active_topic_id)
        fragment_graph_scores = self._compute_fragment_graph_scores(relevant_fragments)

        retrieval_result = self.retrieval.retrieve(
            input_text=context_text,
            keys=all_keys,
            fragments=relevant_fragments,
            reference_time=ref_time,
            active_topic_id=active_topic_id,
            continuity_score=continuity_score,
            global_token_counts=global_token_counts,
            cross_session_evidence=graph_evidence,
            fragment_graph_scores=fragment_graph_scores,
            clarification=clarification_ctx,
        )

        decision = self._make_decision(retrieval_result, active_pending)
        decision = self._apply_continuation_protection(decision, retrieval_result)

        # [REFACTOR BATCH 6] Delegate ke TrendNavigator
        trend_candidates = self._resolve_trend_candidates(
            decision=decision,
            retrieval=retrieval_result,
            envelope=envelope,
            session_state=session_state,
        )
        if trend_candidates:
            decision.trend_candidates = trend_candidates

        _unresolved_states = {
            DecisionType.AMBIGUOUS,
            DecisionType.INSUFFICIENT_EVIDENCE,
            DecisionType.CONTRADICTORY_EVIDENCE,
            DecisionType.UNRESOLVED_CONTINUATION,
        }
        if active_pending and (
            decision.decision_type == DecisionType.CLEAR
            or (is_cross_session_pending and decision.decision_type not in _unresolved_states)
        ):
            active_pending.status = PendingStatus.RESOLVED
            active_pending.last_updated_at = ref_time
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, active_pending)

        if decision.decision_type == DecisionType.AMBIGUOUS:
            question = retrieval_result.clarification_message or (
                "Maaf, saya kurang yakin. Maksud Anda topik yang mana?"
            )
            return self._handle_ambiguity(
                envelope, retrieval_result, active_pending, question,
                clarification=clarification_ctx,
                trend_candidates=trend_candidates,
            )

        if decision.decision_type in (
            DecisionType.INSUFFICIENT_EVIDENCE, DecisionType.CONTRADICTORY_EVIDENCE
        ):
            question = retrieval_result.clarification_message or decision.message or (
                "Bukti yang saya miliki belum cukup untuk memilih satu topik. Bisa diperjelas?"
            )
            return self._handle_ambiguity(
                envelope, retrieval_result, active_pending, question,
                decision_type=decision.decision_type,
                clarification=clarification_ctx,
                trend_candidates=trend_candidates,
            )

        if decision.decision_type == DecisionType.UNRESOLVED_CONTINUATION:
            return self._handle_unresolved_continuation(
                envelope, retrieval_result, active_pending,
                trend_candidates=trend_candidates,
                clarification=clarification_ctx,
            )

        exact_memories = []
        retrieved_memories: List[RetrievedMemory] = []
        reconstruction_status = ReconstructionStatus.NO_MEMORY_REQUESTED

        if decision.selected_fragment_ids:
            exact_memories, reconstruction_status = self._reconstruct_memories(
                decision.selected_fragment_ids
            )
            retrieved_memories = self._attach_evidence_metadata(exact_memories, retrieval_result)

        if self.context_builder and self.llm_client:
            active_topic_title = None
            if retrieval_result.selected_key_id:
                key_obj = self.storage.get_topic_key(retrieval_result.selected_key_id)
                active_topic_title = key_obj.title if key_obj else None

            model_request = self.context_builder.build_context(
                current_user_input=envelope.raw_text,
                exact_memories=retrieved_memories if retrieved_memories else exact_memories,
                active_topic_title=active_topic_title
            )
            try:
                model_response = self.llm_client.generate(model_request)
                assistant_response = model_response.response_text
            except Exception as e:
                logger.error(f"LLM Generation gagal: {e}")
                assistant_response = self._simulate_generation(
                    envelope.raw_text, exact_memories, decision
                )
        else:
            assistant_response = self._simulate_generation(
                envelope.raw_text, exact_memories, decision
            )

        analysis = self._analyze_turn(envelope, assistant_response, retrieval_result, active_topic_id)

        if analysis.unresolved_continuation:
            return EngineResult(
                assistant_response=assistant_response,
                decision=decision,
                retrieved_memory_ids=decision.selected_fragment_ids,
                active_topic_key_id=analysis.active_topic_key_id,
                persistence_status="BLOCKED_UNRESOLVED_CONTINUATION",
                reconstruction_status=reconstruction_status,
                retrieved_memories=retrieved_memories,
                evidence_summary=retrieval_result.evidence_summary,
                candidates=retrieval_result.candidates,
                turn_continuity_score=analysis.turn_continuity_score,
                continuity_evidence_sources=analysis.continuity_evidence_sources,
                continuation_intent=True,
                active_fragment_id=analysis.active_fragment_id,
                active_depth=analysis.active_depth,
            )

        composite_analysis = self.composite_analyzer.detect_composite_sources(
            response=assistant_response,
            retrieved_memories=retrieved_memories,
            primary_topic_key_id=analysis.active_topic_key_id,
        )

        placement: Optional[FragmentPlacementResult] = None
        propagation: Optional[HelperPropagationResult] = None
        created_fragment_id: Optional[str] = None

        if analysis.classification != MemoryClassification.DISCARD:
            created_fragment_id, placement, propagation = self._persist_memory(
                envelope,
                assistant_response,
                analysis,
                retrieval_result,
                active_topic_id,
                composite_analysis=composite_analysis,
            )

        self._update_session_state(
            session_id=session_id,
            topic_key_id=analysis.active_topic_key_id,
            fragment_id=created_fragment_id,
            placement=placement,
            ref_time=ref_time,
        )

        reasoning_summary = self._build_reasoning_purity_summary(analysis, retrieval_result)

        return EngineResult(
            assistant_response=assistant_response,
            decision=decision,
            retrieved_memory_ids=decision.selected_fragment_ids,
            active_topic_key_id=analysis.active_topic_key_id,
            persistence_status="SUCCESS",
            reconstruction_status=reconstruction_status,
            retrieved_memories=retrieved_memories,
            evidence_summary=retrieval_result.evidence_summary,
            candidates=retrieval_result.candidates,
            reasoning_purity_summary=reasoning_summary,
            turn_continuity_score=analysis.turn_continuity_score,
            continuity_evidence_sources=analysis.continuity_evidence_sources,
            active_fragment_id=created_fragment_id or analysis.active_fragment_id,
            active_depth=placement.depth if placement else analysis.active_depth,
            created_fragment_id=created_fragment_id,
            fragment_placement=placement,
            helper_propagation=propagation,
            composite_analysis=composite_analysis,
            trend_candidates=trend_candidates,
            clarification_round=(clarification_ctx.round_number if clarification_ctx else 0),
            clarification_history=(
                clarification_ctx.clarification_history if clarification_ctx else []
            ),
            excluded_candidate_keys=retrieval_result.excluded_candidate_keys,
            continuation_intent=retrieval_result.continuation_intent,
        )

    # ==========================================
    # FORWARDING METHODS (Backward Compat untuk Test)
    # ==========================================

    def _extract_input_tokens(self, envelope: InputEnvelope) -> set:
        """Forwarding ke ClarificationStateManager.extract_input_tokens."""
        return self.clarification_state.extract_input_tokens(envelope)

    def _extract_symbol_identity_tokens(self, raw_text: str) -> set:
        """Forwarding ke ClarificationStateManager.extract_symbol_identity_tokens."""
        return self.clarification_state.extract_symbol_identity_tokens(raw_text)

    def _get_input_token_quality(self, envelope: InputEnvelope) -> Dict[str, TokenQuality]:
        """Forwarding ke ClarificationStateManager.get_input_token_quality."""
        return self.clarification_state.get_input_token_quality(envelope)

    def _collect_pending_evidence_tokens(
        self,
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> set:
        """Forwarding ke ClarificationStateManager.collect_pending_evidence_tokens."""
        return self.clarification_state.collect_pending_evidence_tokens(pending, all_keys)

    def _score_pending_relevance(
        self,
        envelope: InputEnvelope,
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> float:
        """Forwarding ke ClarificationStateManager.score_pending_relevance."""
        return self.clarification_state.score_pending_relevance(envelope, pending, all_keys)

    def _detect_explicit_selection(self, raw_text: str, pending: PendingState) -> bool:
        """Forwarding ke ClarificationStateManager.detect_explicit_selection."""
        return self.clarification_state.detect_explicit_selection(raw_text, pending)

    def _check_pending_relevance(
        self,
        envelope: InputEnvelope,
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> bool:
        """Forwarding ke ClarificationStateManager.check_pending_relevance."""
        return self.clarification_state.check_pending_relevance(envelope, pending, all_keys)

    def _pause_pending(self, pending: PendingState):
        """Forwarding ke ClarificationStateManager.pause_pending."""
        return self.clarification_state.pause_pending(pending)

    def _detect_explicit_cancellation(self, envelope: InputEnvelope) -> bool:
        """Forwarding ke ClarificationStateManager.detect_explicit_cancellation."""
        return self.clarification_state.detect_explicit_cancellation(envelope)

    def _build_clarification_accumulation(
        self,
        envelope: InputEnvelope,
        pending: PendingState,
    ) -> ClarificationAccumulation:
        """Forwarding ke ClarificationStateManager.build_accumulation."""
        return self.clarification_state.build_accumulation(envelope, pending)

    def _persist_clarification_state(
        self,
        pending: PendingState,
        clarification: Optional[ClarificationAccumulation],
        result: RetrievalResult,
        envelope: InputEnvelope,
        question: str,
    ) -> None:
        """Forwarding ke ClarificationStateManager.persist_state."""
        return self.clarification_state.persist_state(
            pending, clarification, result, envelope, question
        )

    # ==========================================
    # CONTINUATION PROTECTION (M2)
    # ==========================================

    def _apply_continuation_protection(
        self,
        decision: EngineDecision,
        retrieval: RetrievalResult,
    ) -> EngineDecision:
        """[M2] NO_MATCH + continuation intent != NEW_TOPIC."""
        cfg = CONFIG.continuation
        if not cfg.BLOCK_NEW_TOPIC_ON_CONTINUATION:
            return decision
        if not retrieval.continuation_intent:
            return decision
        if decision.decision_type not in (
            DecisionType.NO_MATCH,
            DecisionType.INSUFFICIENT,
            DecisionType.INSUFFICIENT_EVIDENCE,
        ):
            return decision

        top_score = retrieval.candidates[0].final_score if retrieval.candidates else 0.0
        if top_score >= cfg.MIN_CONTINUATION_CANDIDATE_SCORE:
            return decision

        decision.decision_type = DecisionType.UNRESOLVED_CONTINUATION
        decision.continuation_intent = True
        decision.message = (
            "Anda meminta melanjutkan pembahasan sebelumnya, tetapi saya belum "
            "menemukan topik yang cukup cocok. Bisa sebutkan topik atau bagiannya?"
        )
        decision.decision_reason = (
            f"continuation intent detected; top_score={top_score:.3f}"
        )
        return decision

    def _handle_unresolved_continuation(
        self,
        envelope: InputEnvelope,
        result: RetrievalResult,
        existing: Optional[PendingState],
        trend_candidates: Optional[List[TrendCandidate]] = None,
        clarification: Optional[ClarificationAccumulation] = None,
    ) -> EngineResult:
        """[M2] Respons ini TIDAK dipersist sebagai fragment TopicKey baru."""
        trend_candidates = trend_candidates or []
        if trend_candidates:
            lines = [
                "Saya belum yakin pembahasan mana yang Anda maksud.",
                "Beberapa topik yang sedang aktif:",
            ]
            for i, t in enumerate(trend_candidates, 1):
                helper = t.helper_description or "pembahasan terkait."
                lines.append(f"{i}. {t.label} — {helper}")
            lines.append("\nYang mana yang ingin dilanjutkan?")
            message = "\n".join(lines)
        else:
            message = (
                "Saya belum menemukan pembahasan yang cocok untuk dilanjutkan. "
                "Bisa sebutkan topik atau bagian yang Anda maksud?"
            )

        pending = existing
        if pending is not None and pending.status == PendingStatus.ACTIVE:
            pending.clarification_round = (pending.clarification_round or 1) + 1
            self.clarification_state.persist_state(
                pending, clarification, result, envelope, message
            )
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, pending)
        else:
            pending = PendingState(
                pending_id=str(uuid.uuid4()),
                session_id=envelope.session_id,
                original_input=envelope.raw_text,
                candidate_keys=[
                    t.node_id for t in trend_candidates
                    if t.node_type == TrendNodeType.TOPIC_KEY
                ],
                candidate_scores={
                    t.node_id: t.trend_score for t in trend_candidates
                    if t.node_type == TrendNodeType.TOPIC_KEY
                },
                assistant_question=message,
                created_at=envelope.timestamp,
                status=PendingStatus.ACTIVE,
                origin_session_id=envelope.session_id,
                clarification_history=(
                    list(clarification.clarification_history) if clarification else []
                ),
                combined_clarification_text=(
                    clarification.combined_text if clarification else envelope.raw_text
                ),
            )
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, pending)

        return EngineResult(
            assistant_response=message,
            decision=EngineDecision(
                decision_type=DecisionType.UNRESOLVED_CONTINUATION,
                message=message,
                decision_reason="continuation intent unresolved; persistence blocked",
                continuation_intent=True,
                trend_candidates=trend_candidates,
                excluded_candidate_keys=result.excluded_candidate_keys,
            ),
            persistence_status="BLOCKED_UNRESOLVED_CONTINUATION",
            evidence_summary=result.evidence_summary,
            candidates=result.candidates,
            retrieved_memories=[],
            trend_candidates=trend_candidates,
            continuation_intent=True,
            clarification_round=(pending.clarification_round if pending else 0),
            clarification_history=(list(pending.clarification_history) if pending else []),
            excluded_candidate_keys=result.excluded_candidate_keys,
        )

    # ==========================================
    # TREND NAVIGATION (M5) — BATCH 6: Delegate to TrendNavigator
    # ==========================================

    def _resolve_trend_candidates(
        self,
        decision: EngineDecision,
        retrieval: RetrievalResult,
        envelope: InputEnvelope,
        session_state: Optional[SessionActiveState],
    ) -> List[TrendCandidate]:
        """
        [BATCH 6] Forwarding ke TrendNavigator.

        TrendNavigator melakukan:
            1. Resolve temporal_range dari signal.
            2. Mode A: TopicKey trend (jika topik belum jelas).
            3. Mode B: Fragment trend (jika topik jelas, bagian belum).
            4. Recompute temporal_score dengan activity density (Prioritas 4).
            5. Re-sort dan re-limit candidates.
        """
        return self.trend_navigator.resolve_trend_candidates(
            decision=decision,
            retrieval=retrieval,
            envelope=envelope,
            session_state=session_state,
        )

    # ==========================================
    # SESSION STATE & CONTINUITY
    # ==========================================

    def _update_session_state(
        self,
        session_id: str,
        topic_key_id: Optional[str],
        fragment_id: Optional[str],
        placement: Optional[FragmentPlacementResult],
        ref_time: datetime,
    ) -> None:
        """[M3] Simpan Current Conversational State."""
        if topic_key_id is None:
            return
        state = SessionActiveState(
            session_id=session_id,
            active_topic_key_id=topic_key_id,
            active_fragment_id=fragment_id,
            active_depth=placement.depth if placement else 0,
            last_updated=ref_time,
        )
        try:
            self.storage.save_session_state(state)
        except Exception as e:
            logger.debug(f"Gagal menyimpan session state: {e}")

    def _calculate_active_state(self, session_id: str) -> Tuple[Optional[str], float]:
        """Real Continuity Window menggunakan Median."""
        recent_frags = self.storage.get_recent_fragments(
            session_id, limit=CONFIG.continuity.CONTINUITY_WINDOW_SIZE
        )
        if not recent_frags:
            return None, 0.0

        latest_key_id = recent_frags[0].topic_key_id
        match_count = sum(1 for f in recent_frags if f.topic_key_id == latest_key_id)
        freq_ratio = match_count / len(recent_frags)

        strength_scores = [
            f.relationship_strength for f in recent_frags
            if f.topic_key_id == latest_key_id and f.relationship_strength > 0
        ]

        if strength_scores:
            median_score = statistics.median(strength_scores)
            stability_ratio = max(median_score, freq_ratio * 0.8)
        else:
            stability_ratio = freq_ratio

        if stability_ratio >= CONFIG.continuity.MEDIAN_CONTINUITY_THRESHOLD:
            return latest_key_id, stability_ratio
        return latest_key_id, stability_ratio * 0.5

    def _get_relevant_fragments(self, keys: List[TopicKey]) -> List[FragmentMetadata]:
        """Mengambil metadata fragment untuk kunci-kunci yang ada."""
        fragments = []
        for k in keys:
            fragments.extend(self.storage.get_fragment_metadata_by_key(k.key_id))
        return fragments

    # ==========================================
    # GRAPH EVIDENCE
    # ==========================================

    def _build_graph_evidence(
        self,
        keys: List[TopicKey],
        fragments: List[FragmentMetadata],
        active_topic_id: Optional[str]
    ) -> Dict[str, float]:
        """Graph = EVIDENCE, bukan absolute truth."""
        evidence: Dict[str, float] = {}
        if not active_topic_id:
            return evidence

        reference_ids = [
            f.fragment_id for f in fragments if f.topic_key_id == active_topic_id
        ]
        if not reference_ids:
            return evidence
        reference_ids = reference_ids[:CONFIG.graph.MAX_EDGES_PER_NODE]

        for key in keys:
            key_frag_ids = [
                f.fragment_id for f in fragments if f.topic_key_id == key.key_id
            ]
            if not key_frag_ids:
                continue
            scores = self.storage.get_fragment_graph_evidence(key_frag_ids, reference_ids)
            if scores:
                best = max(scores.values())
                if best > 0:
                    evidence[key.key_id] = best
        return evidence

    def _compute_fragment_graph_scores(
        self,
        fragments: List[FragmentMetadata]
    ) -> Dict[str, float]:
        """Graph score per fragment dari edge NYATA."""
        if not fragments:
            return {}
        frag_ids = [f.fragment_id for f in fragments]
        raw_scores = self.storage.get_fragment_graph_evidence(
            frag_ids, reference_fragment_ids=frag_ids
        )
        return {fid: score for fid, score in raw_scores.items() if score > 0}

    def _attach_evidence_metadata(
        self,
        exact_memories: List[dict],
        retrieval: RetrievalResult
    ) -> List[RetrievedMemory]:
        """RetrievedMemory dengan metadata bukti + hierarchy framing."""
        score_map = retrieval.selected_fragment_scores or {}
        results: List[RetrievedMemory] = []
        for mem in exact_memories:
            fid = mem.get('fragment_id', '')
            meta = score_map.get(fid, {})
            frag_meta = self.storage.get_fragment(fid)
            topic_title = ""
            if frag_meta is not None:
                key_obj = self.storage.get_topic_key(frag_meta.topic_key_id)
                topic_title = key_obj.title if key_obj else ""
            results.append(RetrievedMemory(
                fragment_id=fid,
                text=mem.get('text', ''),
                timestamp=mem.get('timestamp', ''),
                topic_key_id=mem.get('topic_key_id', ''),
                relevance_score=meta.get('relevance_score', mem.get('relevance_score', 0.0)),
                temporal_score=meta.get('temporal_score', 0.0),
                graph_score=meta.get('graph_score', 0.0),
                evidence_strength=meta.get('relevance_score', 0.0),
                exact_temporal_match=meta.get('exact_temporal_match', False),
                fragment_graph_score=meta.get('fragment_graph_score', 0.0),
                graph_edge_count=1 if meta.get('fragment_graph_score', 0.0) > 0 else 0,
                graph_path_types=[],
                temporal_distance_score=meta.get('temporal_distance_score', 0.0),
                temporal_distance_minutes=meta.get('temporal_distance_minutes'),
                topic_key_title=topic_title,
                fragment_key=(frag_meta.fragment_key if frag_meta else ""),
                fragment_helper_description=(
                    frag_meta.fragment_helper_description if frag_meta else ""
                ),
                parent_fragment_id=(frag_meta.parent_fragment_id if frag_meta else None),
                depth=(frag_meta.depth if frag_meta else 0),
                source_fragment_ids=(
                    list(frag_meta.source_fragment_ids) if frag_meta else []
                ),
                is_composite=(frag_meta.is_composite if frag_meta else False),
            ))
        return results

    # ==========================================
    # CO-OCCURRENCE & MUTATION GATE
    # ==========================================

    def _build_window_pairs(self, tokens: List[str]) -> Dict[str, int]:
        """Co-occurrence berbasis window untuk PERSISTENCE."""
        cc = CONFIG.co_occurrence
        pairs: Dict[str, int] = {}
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
                    strength = cc.ADJACENT_PAIR_BONUS * (
                        cc.CO_OCCURRENCE_DECAY ** (distance - 1)
                    )
                if strength < cc.CO_OCCURRENCE_MIN_STRENGTH:
                    continue
                left, right = sorted([token_a, token_b])
                pair = f"{left}↔{right}"
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _compute_mutation_permissions(
        self,
        envelope: InputEnvelope,
        retrieval: RetrievalResult,
        candidate: Optional[CandidateScore],
        classification: MemoryClassification
    ) -> Tuple[bool, bool, str, float, int, float]:
        """EVIDENCE BEFORE MUTATION."""
        mut = CONFIG.mutation

        if classification == MemoryClassification.NEW_TOPIC or candidate is None:
            return (True, True, "new topic / no candidate", 0.0, 0, 0.0)

        evidence_strength = candidate.evidence_strength
        support_count = candidate.support_count
        contradiction = candidate.contradiction_score

        if contradiction > mut.MUTATION_CONTRADICTION_BLOCK:
            return (False, False, "blocked: contradiction",
                    evidence_strength, support_count, contradiction)

        if evidence_strength < mut.MIN_EVIDENCE_TO_MUTATE_TOPIC:
            return (False, False, "blocked: weak evidence",
                    evidence_strength, support_count, contradiction)

        if support_count < mut.MIN_SUPPORT_TO_MUTATE_TOPIC:
            continuity_only_ok = (
                mut.ALLOW_CONTINUITY_ONLY_MUTATION
                and candidate.continuity_score >= CONFIG.evidence.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_only_ok:
                return (False, False, "blocked: low support",
                        evidence_strength, support_count, contradiction)

        signal = retrieval.signal_profile
        quality_map = (signal.token_quality if signal else {}) or {}
        user_tokens = [
            t for t in re.findall(r'\b\w+\b', envelope.raw_text.lower())
            if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH
        ]
        meaningful_tokens = [
            t for t in user_tokens
            if quality_map.get(t, TokenQuality.NORMAL) != TokenQuality.LOW_INFORMATION
        ]
        if user_tokens and not meaningful_tokens:
            return (True, False, "only low-information tokens",
                    evidence_strength, support_count, contradiction)

        desc_allowed = evidence_strength >= mut.MIN_EVIDENCE_TO_UPDATE_DESCRIPTION
        reason = "allowed" if desc_allowed else "allowed: description held"
        return (True, desc_allowed, reason, evidence_strength, support_count, contradiction)

    # ==========================================
    # CONTINUITY PURITY
    # ==========================================

    def _compute_turn_continuity(
        self,
        envelope: InputEnvelope,
        active_key: Optional[TopicKey],
        top_candidate: Optional[CandidateScore],
        history_ratio: float,
    ) -> Tuple[float, List[str]]:
        """CONTINUITY SCORE MURNI. Bukan relevance kandidat."""
        cc = CONFIG.continuity
        ev = CONFIG.evidence

        if active_key is None or top_candidate is None:
            return 0.0, []

        user_tokens = [
            t for t in re.findall(r'\b\w+\b', envelope.raw_text.lower())
            if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH
        ]
        if not user_tokens:
            overlap_ratio = 0.0
        else:
            active_tokens = set(active_key.token_profile.keys())
            overlap_count = sum(1 for t in user_tokens if t in active_tokens)
            overlap_ratio = overlap_count / len(user_tokens)

        hist = history_ratio if history_ratio else 0.0

        active_frag_rel = 0.0
        try:
            recent_frags = self.storage.get_fragment_metadata_by_key(active_key.key_id)
            session_frags = [
                f for f in recent_frags if f.session_id == envelope.session_id
            ]
            if session_frags:
                session_frags_sorted = sorted(
                    session_frags, key=lambda x: x.created_at, reverse=True
                )
                active_frag_rel = session_frags_sorted[0].relationship_strength or 0.0
        except Exception as e:
            logger.debug(f"Gagal mengambil active fragment relationship: {e}")

        novelty_ratio = 1.0 - overlap_ratio
        novelty_penalty = 0.0
        if novelty_ratio > CONFIG.topic_switch.NOVELTY_THRESHOLD:
            novelty_penalty = cc.CONTINUITY_NOVELTY_PENALTY * novelty_ratio

        contradiction_penalty = 0.0
        if top_candidate.contradiction_score > ev.MAX_CONTRADICTION_SCORE:
            contradiction_penalty = (
                cc.CONTINUITY_CONTRADICTION_PENALTY * top_candidate.contradiction_score
            )

        raw = (
            cc.ACTIVE_TOPIC_OVERLAP_WEIGHT * overlap_ratio
            + cc.CONTINUITY_HISTORY_WEIGHT * hist
            + cc.CONTINUITY_FRAGMENT_WEIGHT * active_frag_rel
            - novelty_penalty
            - contradiction_penalty
        )
        continuity = max(0.0, min(1.0, raw))

        sources = []
        if overlap_ratio >= 0.15:
            sources.append("overlap")
        if hist >= 0.5:
            sources.append("history")
        if active_frag_rel >= 0.1:
            sources.append("fragments")
        if novelty_penalty > 0:
            sources.append(f"-novelty({novelty_penalty:.2f})")
        if contradiction_penalty > 0:
            sources.append(f"-contradiction({contradiction_penalty:.2f})")

        return continuity, sources

    def _build_reasoning_purity_summary(
        self,
        analysis: TurnAnalysisResult,
        retrieval: RetrievalResult
    ) -> str:
        """Ringkasan kemurnian reasoning untuk observability."""
        top = retrieval.candidates[0] if retrieval.candidates else None
        if top is None:
            return ""
        lines = [
            f"PURITY CHECK: {analysis.classification.value}",
            f"  final_score      : {top.final_score:.3f}",
            f"  turn_continuity  : {analysis.turn_continuity_score:.3f}",
            f"  delta            : {abs(top.final_score - analysis.turn_continuity_score):.3f}",
            f"  families         : {[f.value for f in top.evidence_families]}",
            f"  family_count     : {top.family_support_count}",
            f"  cont_sources     : {analysis.continuity_evidence_sources or '(none)'}",
        ]
        return "\n".join(lines)

    # ==========================================
    # DECISION GATE
    # ==========================================

    def _make_decision(
        self,
        result: RetrievalResult,
        pending: Optional[PendingState]
    ) -> EngineDecision:
        """Menerjemahkan hasil retrieval menjadi keputusan mesin."""
        if result.decision_type == DecisionType.AMBIGUOUS:
            top = result.candidates[0] if result.candidates else None
            return EngineDecision(
                decision_type=DecisionType.AMBIGUOUS,
                message=result.clarification_message or "Ambiguitas terdeteksi.",
                decision_reason=result.analysis_notes,
                family_support_count=top.family_support_count if top else 0,
                evidence_families=top.evidence_families if top else [],
                excluded_candidate_keys=result.excluded_candidate_keys,
                continuation_intent=result.continuation_intent,
            )

        if result.decision_type in (
            DecisionType.INSUFFICIENT_EVIDENCE, DecisionType.CONTRADICTORY_EVIDENCE
        ):
            top = result.candidates[0] if result.candidates else None
            return EngineDecision(
                decision_type=result.decision_type,
                message=result.clarification_message or result.analysis_notes,
                confidence=top.confidence if top else 0.0,
                support_count=top.support_count if top else 0,
                evidence_strength=top.evidence_strength if top else 0.0,
                contradiction_score=top.contradiction_score if top else 0.0,
                decision_reason=result.analysis_notes,
                family_support_count=top.family_support_count if top else 0,
                evidence_families=top.evidence_families if top else [],
                excluded_candidate_keys=result.excluded_candidate_keys,
                continuation_intent=result.continuation_intent,
            )

        if result.decision_type == DecisionType.CLEAR:
            selected = next(
                (c for c in result.candidates if c.key_id == result.selected_key_id),
                result.candidates[0] if result.candidates else None
            )
            gate_decision, gate_reason = self._validate_candidate_evidence(selected)
            if gate_decision is not None:
                return EngineDecision(
                    decision_type=gate_decision,
                    message=gate_reason,
                    confidence=selected.confidence if selected else 0.0,
                    support_count=selected.support_count if selected else 0,
                    evidence_strength=selected.evidence_strength if selected else 0.0,
                    contradiction_score=selected.contradiction_score if selected else 0.0,
                    decision_reason=gate_reason,
                    family_support_count=selected.family_support_count if selected else 0,
                    evidence_families=selected.evidence_families if selected else [],
                    excluded_candidate_keys=result.excluded_candidate_keys,
                    continuation_intent=result.continuation_intent,
                )

            return EngineDecision(
                decision_type=DecisionType.CLEAR,
                selected_key_id=result.selected_key_id,
                selected_fragment_ids=result.selected_fragment_ids,
                message=f"Topik terpilih: {result.selected_key_id}",
                confidence=selected.confidence if selected else 0.0,
                support_count=selected.support_count if selected else 0,
                evidence_strength=selected.evidence_strength if selected else 0.0,
                contradiction_score=selected.contradiction_score if selected else 0.0,
                decision_reason=result.analysis_notes,
                family_support_count=selected.family_support_count if selected else 0,
                evidence_families=selected.evidence_families if selected else [],
                excluded_candidate_keys=result.excluded_candidate_keys,
                continuation_intent=result.continuation_intent,
            )

        return EngineDecision(
            decision_type=DecisionType.NO_MATCH,
            message="Tidak ada memori relevan ditemukan.",
            decision_reason=result.analysis_notes,
            excluded_candidate_keys=result.excluded_candidate_keys,
            continuation_intent=result.continuation_intent,
        )

    def _infer_families_from_candidate(
        self,
        candidate: CandidateScore
    ) -> Tuple[int, List[EvidenceFamily], bool]:
        """Infer evidence families. Robust terhadap candidate manual."""
        if candidate.family_support_count > 0:
            return (
                candidate.family_support_count,
                list(candidate.evidence_families),
                candidate.is_single_family_dominated,
            )

        if candidate.evidence_profile is None:
            return (0, [], False)

        profile = candidate.evidence_profile
        supporting_names = profile.supporting_evidence_names or []
        if not supporting_names:
            return (0, [], False)

        family_set: set = set()
        signal_to_family = profile.signal_to_family or {}
        for name in supporting_names:
            if name in signal_to_family:
                family_set.add(signal_to_family[name])
            else:
                family = self._SIGNAL_FAMILY.get(name)
                if family is not None:
                    family_set.add(family)

        evidence_families = sorted(family_set, key=lambda f: f.value)
        return (len(evidence_families), evidence_families, profile.is_single_signal_dominated)

    def _validate_candidate_evidence(
        self,
        candidate: Optional[CandidateScore]
    ) -> Tuple[Optional[DecisionType], str]:
        """EVIDENCE VALIDATION lapis kedua di Engine."""
        if candidate is None:
            return (None, "")

        ev = CONFIG.evidence
        ef = CONFIG.evidence_family

        if candidate.contradiction_score > ev.MAX_CONTRADICTION_SCORE:
            return (DecisionType.CONTRADICTORY_EVIDENCE,
                    f"Bukti kontradiktif pada '{candidate.title}'.")

        if candidate.support_count < ev.MIN_SUPPORTING_EVIDENCE:
            continuity_exemption = (
                ev.CONTINUITY_EXEMPT_FROM_MIN_SUPPORT
                and candidate.continuity_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_exemption:
                return (DecisionType.INSUFFICIENT_EVIDENCE,
                        f"Bukti pendukung tidak cukup pada '{candidate.title}' "
                        f"(support={candidate.support_count}).")

        family_count, families, _ = self._infer_families_from_candidate(candidate)
        if family_count < ef.MIN_EVIDENCE_FAMILIES:
            exempt = (
                ef.CONTINUITY_EXEMPT_FROM_FAMILY_CHECK
                and EvidenceFamily.CONTEXTUAL in families
                and candidate.continuity_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not exempt:
                names = ",".join(f.value for f in families) or "none"
                return (DecisionType.INSUFFICIENT_EVIDENCE,
                        f"Keragaman bukti tidak cukup pada '{candidate.title}' "
                        f"(families={family_count}, active=[{names}]).")

        if candidate.confidence < ev.MIN_ASSOCIATIVE_CONFIDENCE:
            return (DecisionType.INSUFFICIENT_EVIDENCE,
                    f"Confidence asosiatif tidak cukup pada '{candidate.title}' "
                    f"(confidence={candidate.confidence:.2f}).")

        return (None, "")

    # ==========================================
    # AMBIGUITY LIFECYCLE
    # ==========================================

    def _handle_ambiguity(
        self,
        envelope: InputEnvelope,
        result: RetrievalResult,
        existing: Optional[PendingState],
        question: str,
        decision_type: DecisionType = DecisionType.AMBIGUOUS,
        clarification: Optional[ClarificationAccumulation] = None,
        trend_candidates: Optional[List[TrendCandidate]] = None,
    ) -> EngineResult:
        """Clarification Round Lifecycle. Pending BERKEMBANG."""
        top = result.candidates[0] if result.candidates else None
        trend_candidates = trend_candidates or []
        decision_kwargs = dict(
            confidence=top.confidence if top else 0.0,
            support_count=top.support_count if top else 0,
            evidence_strength=top.evidence_strength if top else 0.0,
            contradiction_score=top.contradiction_score if top else 0.0,
            decision_reason=result.analysis_notes,
            family_support_count=top.family_support_count if top else 0,
            evidence_families=top.evidence_families if top else [],
            excluded_candidate_keys=result.excluded_candidate_keys,
            continuation_intent=result.continuation_intent,
            trend_candidates=trend_candidates,
        )

        if existing and existing.status == PendingStatus.ACTIVE:
            round_num = existing.clarification_round + 1
            self.clarification_state.persist_state(
                existing, clarification, result, envelope, question
            )

            if round_num > CONFIG.fast_memory.MAX_CLARIFICATION_ROUNDS:
                existing.status = PendingStatus.EXPIRED
                existing.last_updated_at = envelope.timestamp
                with self.storage.transaction() as tx:
                    self.storage.save_pending_state(tx, existing)
                fallback_msg = (
                    "Saya masih belum yakin, tapi mari kita lanjutkan. "
                    "Silakan berikan detail yang lebih spesifik jika Anda ingin "
                    "membahas topik tertentu."
                )
                return EngineResult(
                    assistant_response=fallback_msg,
                    decision=EngineDecision(
                        decision_type=DecisionType.INSUFFICIENT, message=fallback_msg
                    ),
                    persistence_status="PENDING_EXPIRED",
                    evidence_summary=result.evidence_summary,
                    candidates=result.candidates,
                    clarification_round=existing.clarification_round,
                    clarification_history=list(existing.clarification_history or []),
                    excluded_candidate_keys=list(existing.excluded_candidate_keys or []),
                    continuation_intent=result.continuation_intent,
                )

            existing.clarification_round = round_num
            existing.last_updated_at = envelope.timestamp
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, existing)
            return EngineResult(
                assistant_response=question,
                decision=EngineDecision(
                    decision_type=decision_type, message=question, **decision_kwargs
                ),
                persistence_status="PENDING_UPDATED",
                evidence_summary=result.evidence_summary,
                candidates=result.candidates,
                trend_candidates=trend_candidates,
                clarification_round=round_num,
                clarification_history=list(existing.clarification_history or []),
                excluded_candidate_keys=list(existing.excluded_candidate_keys or []),
                continuation_intent=result.continuation_intent,
            )

        max_show = CONFIG.hierarchy.MAX_AMBIGUITY_CANDIDATES
        shortlist = [c.key_id for c in result.candidates[:max_show]]
        new_pending = PendingState(
            pending_id=str(uuid.uuid4()),
            session_id=envelope.session_id,
            original_input=envelope.raw_text,
            candidate_keys=shortlist,
            candidate_scores={c.key_id: c.final_score for c in result.candidates},
            assistant_question=question,
            created_at=envelope.timestamp,
            status=PendingStatus.ACTIVE,
            origin_session_id=envelope.session_id,
            clarification_history=[],
            combined_clarification_text=envelope.raw_text,
            excluded_tokens=list(
                result.signal_profile.excluded_tokens if result.signal_profile else []
            ),
            excluded_candidate_keys=list(result.excluded_candidate_keys or []),
            candidate_history=[shortlist],
        )

        with self.storage.transaction() as tx:
            self.storage.save_pending_state(tx, new_pending)

        return EngineResult(
            assistant_response=question,
            decision=EngineDecision(
                decision_type=decision_type, message=question, **decision_kwargs
            ),
            persistence_status="PENDING_CREATED",
            evidence_summary=result.evidence_summary,
            candidates=result.candidates,
            trend_candidates=trend_candidates,
            clarification_round=1,
            clarification_history=[],
            excluded_candidate_keys=list(new_pending.excluded_candidate_keys or []),
            continuation_intent=result.continuation_intent,
        )

    # ==========================================
    # RECONSTRUCTION & GENERATION
    # ==========================================

    def _reconstruct_memories(
        self,
        fragment_ids: List[str]
    ) -> Tuple[List[dict], ReconstructionStatus]:
        """Exact Reconstruction dengan Observability."""
        memories = []
        failed_count = 0

        for fid in fragment_ids:
            frag_meta = self.storage.get_fragment(fid)
            if not frag_meta:
                failed_count += 1
                continue

            payload = self.storage.get_exact_payload(frag_meta.payload_reference)
            if not payload:
                failed_count += 1
                continue

            try:
                exact_bytes = self.compressor.decompress(payload)
                exact_text = exact_bytes.decode('utf-8')
                memories.append({
                    'fragment_id': fid,
                    'text': exact_text,
                    'timestamp': frag_meta.created_at.isoformat(),
                    'topic_key_id': frag_meta.topic_key_id,
                    'relevance_score': getattr(frag_meta, 'relevance_score', 0.0)
                })
            except (DecompressionError, IntegrityError) as e:
                logger.error(f"Gagal rekonstruksi fragment {fid}: {e}")
                failed_count += 1
                continue

        total = len(fragment_ids)
        if total == 0:
            status = ReconstructionStatus.NO_MEMORY_REQUESTED
        elif failed_count == total:
            status = ReconstructionStatus.FAILED
        elif failed_count > 0:
            status = ReconstructionStatus.PARTIAL_SUCCESS
        else:
            status = ReconstructionStatus.FULL_SUCCESS

        return memories, status

    def _simulate_generation(
        self,
        user_input: str,
        memories: List[dict],
        decision: EngineDecision
    ) -> str:
        """Placeholder untuk LLM Generation (Fallback)."""
        if decision.decision_type == DecisionType.AMBIGUOUS:
            return decision.message or "Mohon maaf, bisa diperjelas maksud Anda?"
        return f"Response simulasi untuk: {user_input}"

    # ==========================================
    # FRAGMENT IDENTITY (M3) — BATCH 2: Combined User+AI
    # ==========================================

    def _extract_fragment_identity(
        self,
        envelope: InputEnvelope,
        assistant_response: str,
        retrieval: RetrievalResult,
        topic_key: Optional[TopicKey] = None,
    ) -> FragmentIdentity:
        """
        [M3 — Batch 2] Ekstrak identitas fragment dari USER + AI RESPONSE.
        """
        return self.identity_extractor.extract(
            user_text=envelope.raw_text,
            assistant_text=assistant_response,
            topic_key=topic_key,
            signal_profile=retrieval.signal_profile,
        )

    def _extract_fragment_identity_legacy(
        self,
        raw_text: str,
        meaningful_tokens: List[str],
    ) -> Tuple[str, str]:
        """
        [LEGACY] Versi lama yang hanya menggunakan user tokens.
        Dipertahankan untuk backward compat dengan test.
        """
        hc = CONFIG.hierarchy
        hp = CONFIG.helper_propagation

        if not meaningful_tokens:
            return "", ""

        key_tokens = meaningful_tokens[:hc.FRAGMENT_KEY_MAX_TOKENS]
        fragment_key = " ".join(t.capitalize() for t in key_tokens)

        helper_tokens = meaningful_tokens[:hp.MAX_HELPER_CONTRIBUTION_TOKENS]
        helper = " ".join(helper_tokens)
        if len(helper) > hc.MAX_FRAGMENT_HELPER_LENGTH:
            helper = helper[:hc.MAX_FRAGMENT_HELPER_LENGTH].rstrip()

        return fragment_key, helper

    # ==========================================
    # FRAGMENT PLACEMENT (M5) — BATCH 3: Delegate to Hierarchy Manager
    # ==========================================

    def _determine_fragment_placement(
        self,
        topic_key_id: str,
        retrieval: RetrievalResult,
        is_new_topic: bool,
        fragment_identity: Optional[FragmentIdentity] = None,
    ) -> FragmentPlacementResult:
        """[M5 — Batch 3] Delegate ke FragmentHierarchyManager."""
        return self.hierarchy_manager.determine_placement(
            topic_key_id=topic_key_id,
            retrieval=retrieval,
            is_new_topic=is_new_topic,
            fragment_identity=fragment_identity,
        )

    # ==========================================
    # HELPER PROPAGATION (M3) — BATCH 4
    # ==========================================
    # Method _helper_similarity, _aggregate_helper, _propagate_helper
    # dipindahkan ke helper_propagation.py.

    # ==========================================
    # COMPOSITE DETECTION (M4) — BATCH 5
    # ==========================================

    def _detect_composite_sources(
        self,
        response: str,
        retrieved_memories: List[RetrievedMemory],
        primary_topic_key_id: Optional[str],
    ) -> CompositeSourceAnalysis:
        """
        [BACKWARD COMPAT] Forwarding ke CompositeMemoryAnalyzer.
        Dipakai oleh test yang memanggil engine._detect_composite_sources langsung.
        """
        return self.composite_analyzer.detect_composite_sources(
            response=response,
            retrieved_memories=retrieved_memories,
            primary_topic_key_id=primary_topic_key_id,
        )

    # ==========================================
    # TURN ANALYSIS
    # ==========================================

    def _analyze_turn(
        self,
        envelope: InputEnvelope,
        response: str,
        retrieval: RetrievalResult,
        current_active_id: Optional[str]
    ) -> TurnAnalysisResult:
        """Topic Switch + Evidence Before Mutation + Continuity Purity."""
        if retrieval.selected_key_id:
            if retrieval.selected_key_id == current_active_id:
                top_candidate = next(
                    (c for c in retrieval.candidates if c.key_id == current_active_id),
                    None
                )
                if top_candidate:
                    user_tokens = [
                        t for t in re.findall(r'\b\w+\b', envelope.raw_text.lower())
                        if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH
                    ]
                    combined_text = f"{envelope.raw_text} {response}"
                    combined_tokens = [
                        t for t in re.findall(r'\b\w+\b', combined_text.lower())
                        if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH
                    ]

                    all_keys = self.storage.get_all_topic_keys()
                    active_key = next(
                        (k for k in all_keys if k.key_id == current_active_id), None
                    )

                    if active_key and user_tokens:
                        active_tokens = set(active_key.token_profile.keys())
                        user_overlap_count = sum(
                            1 for t in user_tokens if t in active_tokens
                        )
                        user_overlap_ratio = user_overlap_count / len(user_tokens)

                        combined_overlap_count = sum(
                            1 for t in combined_tokens if t in active_tokens
                        )
                        combined_overlap_ratio = (
                            combined_overlap_count / len(combined_tokens)
                            if combined_tokens else 0.0
                        )

                        is_short_probe = len(user_tokens) <= 4
                        continuation_markers = {
                            "terus", "lalu", "kemudian", "jadi", "kalau",
                            "nah", "selanjutnya", "berikutnya", "kembali"
                        }
                        has_continuation_marker = any(
                            marker in user_tokens for marker in continuation_markers
                        )
                        has_strong_novelty = any(
                            trigger in user_tokens
                            for trigger in STRONG_NOVELTY_INDICATORS
                        )

                        if (user_overlap_ratio < CONFIG.topic_switch.MIN_ACTIVE_TOPIC_OVERLAP
                                and top_candidate.token_score < 0.1
                                and top_candidate.description_score < 0.1):
                            if has_strong_novelty:
                                classification = MemoryClassification.NEW_TOPIC
                            elif is_short_probe or has_continuation_marker:
                                classification = MemoryClassification.CONTINUE_TOPIC
                            else:
                                classification = MemoryClassification.NEW_TOPIC
                        else:
                            classification = MemoryClassification.CONTINUE_TOPIC
                    else:
                        classification = MemoryClassification.CONTINUE_TOPIC
                else:
                    classification = MemoryClassification.CONTINUE_TOPIC
            else:
                classification = MemoryClassification.REACTIVATE_TOPIC
        else:
            classification = MemoryClassification.NEW_TOPIC

        active_key_for_fragment = retrieval.selected_key_id
        if classification == MemoryClassification.NEW_TOPIC:
            active_key_for_fragment = None

        unresolved_continuation = False
        if (retrieval.continuation_intent
                and CONFIG.continuation.BLOCK_NEW_TOPIC_ON_CONTINUATION
                and classification == MemoryClassification.NEW_TOPIC):
            top_score = (
                retrieval.candidates[0].final_score if retrieval.candidates else 0.0
            )
            if top_score < CONFIG.continuation.MIN_CONTINUATION_CANDIDATE_SCORE:
                unresolved_continuation = True

        selected_candidate = next(
            (c for c in retrieval.candidates if c.key_id == active_key_for_fragment),
            None
        )
        (mutation_allowed, desc_allowed, mutation_reason,
         ev_strength, support_count, contradiction) = self._compute_mutation_permissions(
            envelope, retrieval, selected_candidate, classification
        )

        active_key_obj = None
        if active_key_for_fragment:
            all_keys = self.storage.get_all_topic_keys()
            active_key_obj = next(
                (k for k in all_keys if k.key_id == active_key_for_fragment), None
            )

        if (classification in (MemoryClassification.CONTINUE_TOPIC,
                               MemoryClassification.REACTIVATE_TOPIC)
                and active_key_obj is not None
                and selected_candidate is not None):
            _, history_ratio = self._calculate_active_state(envelope.session_id)
            turn_cont, cont_sources = self._compute_turn_continuity(
                envelope, active_key_obj, selected_candidate, history_ratio
            )
        else:
            turn_cont = 0.0
            if classification == MemoryClassification.NEW_TOPIC:
                cont_sources = ["new_topic"]
            else:
                cont_sources = []

        state = self.storage.get_session_state(envelope.session_id)
        active_fragment_id = state.active_fragment_id if state else None
        active_depth = state.active_depth if state else 0

        return TurnAnalysisResult(
            classification=classification,
            continuity_score=(
                retrieval.candidates[0].final_score if retrieval.candidates else 0.0
            ),
            active_topic_key_id=active_key_for_fragment,
            evidence_strength=ev_strength,
            support_count=support_count,
            contradiction_score=contradiction,
            mutation_allowed=mutation_allowed,
            description_update_allowed=desc_allowed,
            mutation_reason=mutation_reason,
            turn_continuity_score=turn_cont,
            continuity_evidence_sources=cont_sources,
            continuation_intent=retrieval.continuation_intent,
            unresolved_continuation=unresolved_continuation,
            active_fragment_id=active_fragment_id,
            active_depth=active_depth,
        )

    # ==========================================
    # PERSISTENCE
    # ==========================================

    def _persist_memory(
        self,
        envelope: InputEnvelope,
        response: str,
        analysis: TurnAnalysisResult,
        retrieval: RetrievalResult,
        old_active_id: Optional[str],
        composite_analysis: Optional[CompositeSourceAnalysis] = None,
    ) -> Tuple[Optional[str], Optional[FragmentPlacementResult], Optional[HelperPropagationResult]]:
        """
        Transaksi Penyimpanan Memori (Fail-Closed).

        URUTAN:
            exact text -> extract identity -> compress
            -> TX1: persist topic/payload/fragment/edges -> COMMIT
            -> TX2: helper propagation (delegate ke HelperPropagationManager)
        """
        raw_content = f"User: {envelope.raw_text}\nAI: {response}"

        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        new_tokens = [t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
        token_counts: Dict[str, int] = {}
        for t in new_tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        signal = retrieval.signal_profile
        quality_map = (signal.token_quality if signal else {}) or {}
        meaningful_tokens = [
            t for t in new_tokens
            if quality_map.get(t, TokenQuality.NORMAL) != TokenQuality.LOW_INFORMATION
        ]
        profile_tokens = meaningful_tokens if meaningful_tokens else []
        profile_counts: Dict[str, int] = {}
        for t in profile_tokens:
            profile_counts[t] = profile_counts.get(t, 0) + 1

        active_key_obj = None
        if analysis.active_topic_key_id:
            active_key_obj = self.storage.get_topic_key(analysis.active_topic_key_id)

        fragment_identity = self._extract_fragment_identity(
            envelope=envelope,
            assistant_response=response,
            retrieval=retrieval,
            topic_key=active_key_obj,
        )
        fragment_key = fragment_identity.fragment_key
        fragment_helper = fragment_identity.helper_description

        try:
            payload = self.compressor.compress(raw_content)
        except Exception as e:
            logger.error(f"Gagal kompresi: {e}")
            return None, None, None

        key_id = analysis.active_topic_key_id
        new_key = None
        is_new_topic = False

        if not key_id or analysis.classification == MemoryClassification.NEW_TOPIC:
            key_id = str(uuid.uuid4())
            is_new_topic = True
            title = " ".join(new_tokens[:3]).capitalize() if new_tokens else "Topik Baru"
            desc = " ".join(new_tokens[:15]).capitalize() if new_tokens else ""

            new_key = TopicKey(
                key_id=key_id,
                title=title,
                helper_description=desc,
                representative_tokens=new_tokens[:5],
                unique_tokens=list(set(new_tokens)),
                token_profile=token_counts,
                co_occurrence_profile={},
                created_at=envelope.timestamp,
                updated_at=envelope.timestamp,
                aggregated_helper_description=fragment_helper,
            )
            analysis.active_topic_key_id = key_id
        else:
            existing_keys = self.storage.get_all_topic_keys()
            existing_key = next((k for k in existing_keys if k.key_id == key_id), None)
            if existing_key:
                if analysis.mutation_allowed and profile_counts:
                    updated_profile = dict(existing_key.token_profile)
                    for t, count in profile_counts.items():
                        updated_profile[t] = updated_profile.get(t, 0) + count
                    existing_key.token_profile = updated_profile

                    updated_cooc = dict(existing_key.co_occurrence_profile)
                    for pair, count in self._build_window_pairs(profile_tokens).items():
                        updated_cooc[pair] = updated_cooc.get(pair, 0) + count
                    existing_key.co_occurrence_profile = updated_cooc

                    MAX_DESC_TOKENS = 50
                    current_desc_tokens = existing_key.helper_description.split()
                    if (analysis.description_update_allowed
                            and len(current_desc_tokens) < MAX_DESC_TOKENS):
                        current_desc_lower = existing_key.helper_description.lower()
                        new_rep_tokens = []
                        for t in profile_tokens:
                            if t not in current_desc_lower and len(new_rep_tokens) < 3:
                                new_rep_tokens.append(t)
                        if new_rep_tokens:
                            existing_key.helper_description += (
                                f" | {' '.join(new_rep_tokens)}"
                            )
                else:
                    logger.info("EVIDENCE GATE: profil topik tidak dimutasi.")

                existing_key.updated_at = envelope.timestamp
                existing_key.activation_count += 1
                new_key = existing_key

        frag_cooc = self._build_window_pairs(new_tokens)

        placement = self._determine_fragment_placement(
            topic_key_id=key_id,
            retrieval=retrieval,
            is_new_topic=is_new_topic,
            fragment_identity=fragment_identity,
        )

        is_composite = bool(composite_analysis and composite_analysis.is_composite)
        source_ids = (
            list(composite_analysis.source_fragment_ids)
            if is_composite and composite_analysis else []
        )
        if is_composite and composite_analysis:
            source_topic_ids = sorted({
                s.topic_key_id for s in composite_analysis.sources
            })
        else:
            source_topic_ids = []

        fragment = FragmentMetadata(
            fragment_id=str(uuid.uuid4()),
            topic_key_id=key_id,
            session_id=envelope.session_id,
            created_at=envelope.timestamp,
            temporal_index=envelope.timestamp,
            payload_reference=payload.payload_id,
            content_hash=payload.checksum,
            token_profile=token_counts,
            co_occurrence_profile=frag_cooc,
            relationship_strength=analysis.turn_continuity_score,
            fragment_key=fragment_key,
            fragment_helper_description=fragment_helper,
            aggregated_helper_description=fragment_helper,
            parent_fragment_id=placement.parent_fragment_id,
            depth=placement.depth,
            is_composite=is_composite,
            source_fragment_ids=source_ids,
            source_topic_key_ids=source_topic_ids,
        )

        # ==========================================
        # TX1: exact memory + topic + fragment + graph edges
        # ==========================================
        try:
            with self.storage.transaction() as tx:
                if new_key:
                    self.storage.save_topic_key(tx, new_key)
                self.storage.save_payload(tx, payload)
                self.storage.save_fragment(tx, fragment)

                if analysis.classification == MemoryClassification.CONTINUE_TOPIC:
                    frags = self.storage.get_fragment_metadata_by_key(key_id)
                    session_frags = [
                        f for f in frags
                        if f.session_id == envelope.session_id
                        and f.fragment_id != fragment.fragment_id
                    ]
                    if session_frags:
                        session_frags.sort(key=lambda x: x.created_at, reverse=True)
                        last_frag_id = session_frags[0].fragment_id
                        edge = MemoryGraphEdge(
                            edge_id=str(uuid.uuid4()),
                            source_node_id=last_frag_id,
                            target_node_id=fragment.fragment_id,
                            relationship_type=RelationshipType.CONTINUES,
                            strength=analysis.turn_continuity_score,
                            created_at=envelope.timestamp
                        )
                        self.storage.save_graph_edge(tx, edge)

                elif analysis.classification == MemoryClassification.REACTIVATE_TOPIC:
                    all_frags = self.storage.get_fragment_metadata_by_key(key_id)
                    global_frags = [
                        f for f in all_frags if f.fragment_id != fragment.fragment_id
                    ]
                    if global_frags:
                        global_frags.sort(key=lambda x: x.created_at, reverse=True)
                        last_global_frag_id = global_frags[0].fragment_id
                        edge = MemoryGraphEdge(
                            edge_id=str(uuid.uuid4()),
                            source_node_id=last_global_frag_id,
                            target_node_id=fragment.fragment_id,
                            relationship_type=RelationshipType.CROSS_SESSION_LINK,
                            strength=1.0,
                            evidence="Reactivated Topic",
                            created_at=envelope.timestamp
                        )
                        self.storage.save_graph_edge(tx, edge)

                if is_composite and composite_analysis is not None:
                    edge_type = RelationshipType(CONFIG.composite.COMPOSITE_EDGE_TYPE)
                    for src in composite_analysis.sources:
                        self.storage.save_graph_edge(tx, MemoryGraphEdge(
                            edge_id=str(uuid.uuid4()),
                            source_node_id=src.fragment_id,
                            target_node_id=fragment.fragment_id,
                            relationship_type=edge_type,
                            strength=src.contribution_score,
                            evidence="composite source",
                            created_at=envelope.timestamp,
                        ))

            logger.info(f"Memori tersimpan: Fragment {fragment.fragment_id}")
        except Exception as e:
            logger.critical(f"TRANSAKSI GAGAL: {e}")
            return None, placement, None

        # ==========================================
        # TX2: helper propagation (setelah commit)
        # [BATCH 4] Delegate ke HelperPropagationManager
        # ==========================================
        propagation: Optional[HelperPropagationResult] = None
        try:
            with self.storage.transaction() as tx2:
                propagation = self.helper_propagator.propagate(
                    session=tx2,
                    new_fragment=fragment,
                    topic_key_id=key_id,
                    analysis=analysis,
                )
        except Exception as e:
            logger.error(f"HELPER PROPAGATION GAGAL: {e}")
            propagation = None

        return fragment.fragment_id, placement, propagation
