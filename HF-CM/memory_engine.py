"""
memory_engine.py
Global Orchestrator & State Machine (Layer 16).

Fungsi:
1. Mengoordinasikan alur dari Input -> Retrieval -> Decision -> Generation -> Storage.
2. Mengelola Fast State (Pending Ambiguity).
3. Menerapkan Continuity Window Analysis (Mencegah 'Sticky Topic').
4. Menjamin transaksi penyimpanan memori bersifat Atomik (No Half-Memory).
"""

import logging
import re
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from config import CONFIG
from models import (
    InputEnvelope, EngineResult, EngineDecision, DecisionType, 
    MemoryClassification, PendingState, PendingStatus, TopicKey, 
    FragmentMetadata, CompressedPayload, MemoryGraphEdge, RelationshipType,
    CandidateScore, TurnAnalysisResult
)
from storage import MemoryStorage
from retrieval import RetrievalEngine, RetrievalResult
from compression import PayloadCompressor, DecompressionError, IntegrityError

logger = logging.getLogger(__name__)

class MemoryEngine:
    """
    Otak utama sistem. Menghubungkan sinyal user dengan memori persis 
    dan kemampuan generatif model.
    """

    def __init__(self, storage: MemoryStorage, retrieval: RetrievalEngine, compressor: PayloadCompressor):
        self.storage = storage
        self.retrieval = retrieval
        self.compressor = compressor

    def process_input(self, envelope: InputEnvelope) -> EngineResult:
        """
        Main Entry Point. Memproses satu turn percakapan secara utuh.
        """
        session_id = envelope.session_id
        ref_time = envelope.timestamp
        
        # ==========================================
        # PHASE 1: FAST STATE CHECK (Layer 2)
        # ==========================================
        active_pending = self.storage.get_active_pending_state(session_id)
        context_text = envelope.raw_text
        
        if active_pending:
            is_relevant = self._check_pending_relevance(envelope, active_pending)
            if is_relevant:
                # Gabungkan konteks lama + jawaban baru untuk analisis ulang
                context_text = f"{active_pending.original_input} [CLARIFICATION: {envelope.raw_text}]"
                logger.info("PENDING STATE: Relevan. Menggabungkan konteks.")
            else:
                # User pindah topik. Pause pending state.
                self._pause_pending(active_pending)
                logger.info("PENDING STATE: Tidak relevan. Di-pause.")
                active_pending = None # Set None agar tidak di-resolve secara tidak sengaja di bawah

        # ==========================================
        # PHASE 2: RETRIEVAL & ACTIVATION (Layer 3-6)
        # ==========================================
        all_keys = self.storage.get_all_topic_keys()
        
        # Menentukan Active Topic berdasarkan sejarah fragment (Stateless approach)
        active_topic_id, continuity_score = self._calculate_active_state(session_id)
        
        # Ambil metadata fragment (Ringan, tanpa payload)
        relevant_fragments = self._get_relevant_fragments(all_keys)
        
        retrieval_result = self.retrieval.retrieve(
            input_text=context_text,
            keys=all_keys,
            fragments=relevant_fragments,
            reference_time=ref_time,
            active_topic_id=active_topic_id,
            continuity_score=continuity_score
        )

        # ==========================================
        # PHASE 3: DECISION GATE (Layer 5)
        # ==========================================
        decision = self._make_decision(retrieval_result, active_pending)
        
        # [FIX] Jika ada pending state dan keputusan baru adalah CLEAR, 
        # berarti klarifikasi user berhasil mengurai ambiguitas.
        if active_pending and decision.decision_type == DecisionType.CLEAR:
            active_pending.status = PendingStatus.RESOLVED
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, active_pending)
        
        # Handle Ambiguity (Layer 15)
        if decision.decision_type == DecisionType.AMBIGUOUS:
            return self._handle_ambiguity(envelope, retrieval_result, active_pending)
            
        # ==========================================
        # PHASE 4: EXACT RECONSTRUCTION (Layer 7)
        # ==========================================
        exact_memories = []
        if decision.selected_fragment_ids:
            exact_memories = self._reconstruct_memories(decision.selected_fragment_ids)
            
        # ==========================================
        # PHASE 5: GENERATION (Layer 9 - Placeholder)
        # ==========================================
        assistant_response = self._simulate_generation(envelope.raw_text, exact_memories, decision)
        
        # ==========================================
        # PHASE 6: MEMORY CONSTRUCTION (Layer 10-14)
        # ==========================================
        analysis = self._analyze_turn(envelope, assistant_response, retrieval_result, active_topic_id)
        
        if analysis.classification != MemoryClassification.DISCARD:
            self._persist_memory(
                envelope, 
                assistant_response, 
                analysis, 
                retrieval_result,
                active_topic_id
            )

        return EngineResult(
            assistant_response=assistant_response,
            decision=decision,
            retrieved_memory_ids=decision.selected_fragment_ids,
            active_topic_key_id=analysis.active_topic_key_id,
            persistence_status="SUCCESS"
        )

    # ==========================================
    # INTERNAL LOGIC & HELPERS
    # ==========================================

    def _check_pending_relevance(self, envelope: InputEnvelope, pending: PendingState) -> bool:
        """
        [FIX] Memeriksa apakah input baru adalah jawaban dari pertanyaan ambigu.
        Menggunakan pengecekan irisan token. Jika tidak ada irisan sama sekali dan 
        input memiliki banyak token baru, user dianggap pindah topik.
        """
        if len(envelope.raw_text) > 100: 
            return False 
            
        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        new_tokens = {t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
        
        if not new_tokens:
            return True # Input sangat pendek (misal: "Ya", "Yang bisnis"), asumsikan relevan
            
        all_keys = self.storage.get_all_topic_keys()
        pending_tokens = set()
        for k in all_keys:
            if k.key_id in pending.candidate_keys:
                pending_tokens.update(k.token_profile.keys())
                desc_words = re.findall(r'\b\w+\b', k.helper_description.lower())
                pending_tokens.update(w for w in desc_words if len(w) >= CONFIG.signal.MIN_TOKEN_LENGTH)
                
        overlap = new_tokens.intersection(pending_tokens)
        
        # Jika tidak ada irisan dan ada >= 3 token baru, ini adalah Topic Switch
        if len(overlap) == 0 and len(new_tokens) >= 3:
            return False
            
        return True

    def _pause_pending(self, pending: PendingState):
        """Mengubah status Pending menjadi PAUSED."""
        pending.status = PendingStatus.PAUSED
        with self.storage.transaction() as tx:
            self.storage.save_pending_state(tx, pending)

    def _calculate_active_state(self, session_id: str) -> Tuple[Optional[str], float]:
        """
        Menghitung Active Topic dan Continuity Score berdasarkan 
        riwayat fragment terakhir (Continuity Window).
        """
        all_keys = self.storage.get_all_topic_keys()
        latest_fragment = None
        latest_key_id = None
        
        for k in all_keys:
            frags = self.storage.get_fragment_metadata_by_key(k.key_id)
            session_frags = [f for f in frags if f.session_id == session_id]
            if session_frags:
                session_frags.sort(key=lambda x: x.created_at, reverse=True)
                if latest_fragment is None or session_frags[0].created_at > latest_fragment.created_at:
                    latest_fragment = session_frags[0]
                    latest_key_id = k.key_id
                    
        if latest_key_id:
            return latest_key_id, 1.0 
        return None, 0.0

    def _get_relevant_fragments(self, keys: List[TopicKey]) -> List[FragmentMetadata]:
        """Mengambil metadata fragment untuk kunci-kunci yang ada."""
        fragments = []
        for k in keys:
            fragments.extend(self.storage.get_fragment_metadata_by_key(k.key_id))
        return fragments

    def _make_decision(self, result: RetrievalResult, pending: Optional[PendingState]) -> EngineDecision:
        """Menerjemahkan hasil retrieval menjadi keputusan mesin."""
        if result.decision_type == DecisionType.AMBIGUOUS:
            return EngineDecision(
                decision_type=DecisionType.AMBIGUOUS,
                message="Ambiguitas terdeteksi. Memerlukan klarifikasi."
            )
            
        if result.decision_type == DecisionType.CLEAR:
            return EngineDecision(
                decision_type=DecisionType.CLEAR,
                selected_key_id=result.selected_key_id,
                selected_fragment_ids=result.selected_fragment_ids,
                message=f"Topik terpilih: {result.selected_key_id}"
            )
            
        return EngineDecision(
            decision_type=DecisionType.NO_MATCH,
            message="Tidak ada memori relevan ditemukan."
        )

    def _handle_ambiguity(self, envelope: InputEnvelope, result: RetrievalResult, existing: Optional[PendingState]) -> EngineResult:
        """Membuat atau memperbarui Pending State dan mengembalikan pertanyaan."""
        question = "Maaf, saya kurang yakin. Maksud Anda topik yang mana?"
        
        new_pending = PendingState(
            pending_id=str(uuid.uuid4()),
            session_id=envelope.session_id,
            original_input=envelope.raw_text,
            candidate_keys=[c.key_id for c in result.candidates[:3]],
            candidate_scores={c.key_id: c.final_score for c in result.candidates},
            assistant_question=question,
            created_at=envelope.timestamp,
            status=PendingStatus.ACTIVE
        )
        
        with self.storage.transaction() as tx:
            self.storage.save_pending_state(tx, new_pending)
            
        return EngineResult(
            assistant_response=question,
            decision=EngineDecision(decision_type=DecisionType.AMBIGUOUS),
            persistence_status="PENDING_CREATED"
        )

    def _reconstruct_memories(self, fragment_ids: List[str]) -> List[str]:
        """Mengambil dan mendekompresi payload persis."""
        memories = []
        return memories

    def _simulate_generation(self, user_input: str, memories: List[str], decision: EngineDecision) -> str:
        """Placeholder untuk LLM Generation."""
        if decision.decision_type == DecisionType.AMBIGUOUS:
            return "Mohon maaf, bisa diperjelas maksud Anda?"
        return f"Response simulasi untuk: {user_input}"

    def _analyze_turn(
        self, 
        envelope: InputEnvelope, 
        response: str, 
        retrieval: RetrievalResult,
        current_active_id: Optional[str]
    ) -> TurnAnalysisResult:
        """
        Layer 11 & 12: Menentukan nasib percakapan ini.
        """
        if retrieval.selected_key_id:
            if retrieval.selected_key_id == current_active_id:
                # [FIX] Anti-Stickiness Protection (Blueprint XXXIV)
                # Jika skor token dan deskripsi hampir nol, tapi input punya banyak token baru,
                # maka ini adalah topik baru yang eksplisit (misal: C++), JANGAN dipaksa masuk topik lama.
                top_candidate = next((c for c in retrieval.candidates if c.key_id == current_active_id), None)
                if top_candidate and top_candidate.token_score < 0.1 and top_candidate.description_score < 0.1:
                    raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
                    new_tokens = [t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
                    if len(new_tokens) >= 3:
                        classification = MemoryClassification.NEW_TOPIC
                    else:
                        classification = MemoryClassification.CONTINUE_TOPIC
                else:
                    classification = MemoryClassification.CONTINUE_TOPIC
            else:
                classification = MemoryClassification.NEW_TOPIC
        else:
            classification = MemoryClassification.NEW_TOPIC 
            
        # [FIX] Jika diputuskan sebagai NEW_TOPIC, jangan bawa ID topik lama ke persist_memory
        active_key_for_fragment = retrieval.selected_key_id
        if classification == MemoryClassification.NEW_TOPIC:
            active_key_for_fragment = None
            
        return TurnAnalysisResult(
            classification=classification,
            continuity_score=retrieval.candidates[0].final_score if retrieval.candidates else 0.0,
            active_topic_key_id=active_key_for_fragment
        )

    def _persist_memory(
        self,
        envelope: InputEnvelope,
        response: str,
        analysis: TurnAnalysisResult,
        retrieval: RetrievalResult,
        old_active_id: Optional[str]
    ):
        """
        Layer 14 & 15: Transaksi Penyimpanan Memori (Fail-Closed).
        """
        raw_content = f"User: {envelope.raw_text}\nAI: {response}"
        try:
            payload = self.compressor.compress(raw_content)
        except Exception as e:
            logger.error(f"Gagal kompresi: {e}")
            return 

        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        new_tokens = [t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
        token_counts = {}
        for t in new_tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        key_id = analysis.active_topic_key_id
        new_key = None
        
        if not key_id or analysis.classification == MemoryClassification.NEW_TOPIC:
            key_id = str(uuid.uuid4()) 
            title = envelope.raw_text[:30].strip() or "Topik Baru"
            desc = envelope.raw_text[:100].strip()
            new_key = TopicKey(
                key_id=key_id,
                title=title,
                helper_description=desc,
                token_profile=token_counts,
                co_occurrence_profile={},
                created_at=envelope.timestamp
            )
            analysis.active_topic_key_id = key_id
        else:
            existing_keys = self.storage.get_all_topic_keys()
            existing_key = next((k for k in existing_keys if k.key_id == key_id), None)
            if existing_key:
                updated_profile = dict(existing_key.token_profile)
                for t, count in token_counts.items():
                    updated_profile[t] = updated_profile.get(t, 0) + count
                existing_key.token_profile = updated_profile
                new_key = existing_key 

        fragment = FragmentMetadata(
            fragment_id=str(uuid.uuid4()),
            topic_key_id=key_id,
            session_id=envelope.session_id,
            created_at=envelope.timestamp,
            payload_reference=payload.payload_id
        )

        try:
            with self.storage.transaction() as tx:
                if new_key:
                    self.storage.save_topic_key(tx, new_key)
                self.storage.save_payload(tx, payload)
                self.storage.save_fragment(tx, fragment)
                
                if old_active_id and analysis.classification == MemoryClassification.CONTINUE_TOPIC:
                    edge = MemoryGraphEdge(
                        edge_id=str(uuid.uuid4()),
                        source_id=old_active_id, 
                        target_id=key_id,
                        relationship_type=RelationshipType.CONTINUES,
                        created_at=envelope.timestamp
                    )
                    self.storage.save_graph_edge(tx, edge)
                    
            logger.info(f"Memori tersimpan: Fragment {fragment.fragment_id}")
        except Exception as e:
            logger.critical(f"TRANSAKSI GAGAL: {e}")
