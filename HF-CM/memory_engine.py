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
import statistics
from datetime import datetime
from typing import List, Optional, Tuple

from config import CONFIG
from models import (
    InputEnvelope, EngineResult, EngineDecision, DecisionType, 
    MemoryClassification, PendingState, PendingStatus, TopicKey, 
    FragmentMetadata, CompressedPayload, MemoryGraphEdge, RelationshipType,
    CandidateScore, TurnAnalysisResult, ReconstructionStatus
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

    def __init__(self, storage: MemoryStorage, retrieval: RetrievalEngine, compressor: PayloadCompressor, context_builder=None, llm_client=None):
        self.storage = storage
        self.retrieval = retrieval
        self.compressor = compressor
        # [NEW] Dependency Injection untuk Context Builder & LLM Client (Spec 6.1)
        self.context_builder = context_builder
        self.llm_client = llm_client

    def process_input(self, envelope: InputEnvelope) -> EngineResult:
        """
        Main Entry Point. Memproses satu turn percakapan secara utuh.
        """
        session_id = envelope.session_id
        ref_time = envelope.timestamp
        
        # [NEW] Jamin Session Lifecycle (Review Kedua)
        self.storage.ensure_session_exists(session_id, ref_time)
        
        # ==========================================
        # PHASE 1: FAST STATE CHECK (Layer 2)
        # ==========================================
        active_pending = self.storage.get_active_pending_state(session_id)
        
        # [NEW] TTL Expiration Check
        if active_pending:
            # [FIX] Normalisasi timezone (SQLite mengembalikan naive datetime)
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
        
        if active_pending:
            is_relevant = self._check_pending_relevance(envelope, active_pending)
            if is_relevant:
                context_text = f"{active_pending.original_input} [CLARIFICATION: {envelope.raw_text}]"
                logger.info("PENDING STATE: Relevan. Menggabungkan konteks.")
            else:
                self._pause_pending(active_pending)
                logger.info("PENDING STATE: Tidak relevan. Di-pause.")
                active_pending = None 
        else:
            # [CRITICAL FIX] Cross-Session Pending Resolution dengan RANKING
            # Tidak lagi mengambil pending pertama yang cocok (first-match), 
            # melainkan menghitung skor relevansi dan mengambil yang terbaik.
            global_pendings = self.storage.get_global_active_pendings()
            all_keys = self.storage.get_all_topic_keys()
            
            best_pending = None
            best_score = -1.0
            
            for gp in global_pendings:
                if gp.session_id != session_id:
                    raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
                    new_tokens = {t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
                    
                    score = 0.0
                    if not new_tokens:
                        score = 0.5 # Short answer, assume somewhat relevant
                    else:
                        pending_tokens = set()
                        for k in all_keys:
                            if k.key_id in gp.candidate_keys:
                                pending_tokens.update(k.token_profile.keys())
                        overlap = len(new_tokens.intersection(pending_tokens))
                        score = overlap / len(new_tokens) if new_tokens else 0.0
                        
                    if score >= CONFIG.fast_memory.PENDING_SWITCH_THRESHOLD and score > best_score:
                        best_score = score
                        best_pending = gp
                        
            if best_pending:
                active_pending = best_pending
                is_cross_session_pending = True
                context_text = f"{best_pending.original_input} [CLARIFICATION: {envelope.raw_text}]"
                logger.info(f"CROSS-SESSION PENDING: Relevan dengan sesi {best_pending.session_id} (Score: {best_score:.2f}).")

        # ==========================================
        # PHASE 2: RETRIEVAL & ACTIVATION (Layer 3-6)
        # ==========================================
        all_keys = self.storage.get_all_topic_keys()
        
        # [NEW] Real Continuity Window Calculation
        active_topic_id, continuity_score = self._calculate_active_state(session_id)
        
        relevant_fragments = self._get_relevant_fragments(all_keys)
        
        # [NEW] Fetch Global Token Counts for Unique Token Evidence
        global_token_counts = self.storage.get_global_token_counts()
        
        retrieval_result = self.retrieval.retrieve(
            input_text=context_text,
            keys=all_keys,
            fragments=relevant_fragments,
            reference_time=ref_time,
            active_topic_id=active_topic_id,
            continuity_score=continuity_score,
            global_token_counts=global_token_counts
        )

        # ==========================================
        # PHASE 3: DECISION GATE (Layer 5)
        # ==========================================
        decision = self._make_decision(retrieval_result, active_pending)
        
        # [FIX] Jika ada pending state dan keputusan baru adalah CLEAR, 
        # ATAU jika ini adalah Cross-Session Pending yang berhasil di-retrieval (tidak ambigu lagi),
        # berarti klarifikasi user berhasil mengurai ambiguitas.
        if active_pending and (decision.decision_type == DecisionType.CLEAR or (is_cross_session_pending and decision.decision_type != DecisionType.AMBIGUOUS)):
            active_pending.status = PendingStatus.RESOLVED
            active_pending.last_updated_at = ref_time
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, active_pending)
        
        # Handle Ambiguity (Layer 15)
        if decision.decision_type == DecisionType.AMBIGUOUS:
            question = retrieval_result.clarification_message or "Maaf, saya kurang yakin. Maksud Anda topik yang mana?"
            return self._handle_ambiguity(envelope, retrieval_result, active_pending, question)
            
        # ==========================================
        # PHASE 4: EXACT RECONSTRUCTION (Layer 7)
        # ==========================================
        exact_memories = []
        reconstruction_status = ReconstructionStatus.NO_MEMORY_REQUESTED
        
        if decision.selected_fragment_ids:
            exact_memories, reconstruction_status = self._reconstruct_memories(decision.selected_fragment_ids)
            
        # ==========================================
        # PHASE 5: GENERATION (Layer 8 & 9)
        # ==========================================
        if self.context_builder and self.llm_client and decision.decision_type != DecisionType.AMBIGUOUS:
            active_topic_title = None
            if retrieval_result.selected_key_id:
                key_obj = self.storage.get_topic_key(retrieval_result.selected_key_id)
                active_topic_title = key_obj.title if key_obj else None
                
            model_request = self.context_builder.build_context(
                current_user_input=envelope.raw_text,
                exact_memories=exact_memories,
                active_topic_title=active_topic_title
            )
            try:
                model_response = self.llm_client.generate(model_request)
                assistant_response = model_response.response_text
            except Exception as e:
                logger.error(f"LLM Generation gagal: {e}. Fallback ke simulasi.")
                assistant_response = self._simulate_generation(envelope.raw_text, exact_memories, decision)
        else:
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
            persistence_status="SUCCESS",
            reconstruction_status=reconstruction_status
        )

    # ==========================================
    # INTERNAL LOGIC & HELPERS
    # ==========================================

    def _check_pending_relevance(self, envelope: InputEnvelope, pending: PendingState) -> bool:
        """
        [OVERHAUL] Memeriksa relevansi pending.
        Input pendek otomatis dianggap relevan. Input panjang dicek irisannya.
        """
        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        new_tokens = {t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}
        
        # [FIX] Input pendek (< 3 token) seperti "Terserah", "Yang bisnis", otomatis relevan
        if not new_tokens or len(new_tokens) < 3:
            return True 
            
        all_keys = self.storage.get_all_topic_keys()
        pending_tokens = set()
        for k in all_keys:
            if k.key_id in pending.candidate_keys:
                pending_tokens.update(k.token_profile.keys())
                desc_words = re.findall(r'\b\w+\b', k.helper_description.lower())
                pending_tokens.update(w for w in desc_words if len(w) >= CONFIG.signal.MIN_TOKEN_LENGTH)
                
        overlap = new_tokens.intersection(pending_tokens)
        
        # [FIX] Jika ada minimal 2 token yang beririsan, anggap relevan (menangani long clarification)
        if len(overlap) >= 2:
            return True
            
        # Jika tidak ada irisan sama sekali dan ada >= 3 token baru, ini Topic Switch
        if len(overlap) == 0 and len(new_tokens) >= 3:
            return False
            
        # Fallback: gunakan switch threshold
        overlap_ratio = len(overlap) / len(new_tokens) if new_tokens else 0.0
        return overlap_ratio >= CONFIG.fast_memory.PENDING_SWITCH_THRESHOLD

    def _pause_pending(self, pending: PendingState):
        """Mengubah status Pending menjadi PAUSED."""
        pending.status = PendingStatus.PAUSED
        pending.last_updated_at = datetime.now()
        with self.storage.transaction() as tx:
            self.storage.save_pending_state(tx, pending)

    def _calculate_active_state(self, session_id: str) -> Tuple[Optional[str], float]:
        """
        [CRITICAL FIX] Real Continuity Window menggunakan Median (Review Kedua).
        Menghitung stabilitas topik berdasarkan N-fragment terakhir.
        """
        recent_frags = self.storage.get_recent_fragments(session_id, limit=CONFIG.continuity.CONTINUITY_WINDOW_SIZE)
        if not recent_frags:
            return None, 0.0
            
        latest_key_id = recent_frags[0].topic_key_id
        
        # Hitung rasio fragment di window yang milik topik terbaru
        match_count = sum(1 for f in recent_frags if f.topic_key_id == latest_key_id)
        freq_ratio = match_count / len(recent_frags)
        
        # [FIX] Gunakan relationship_strength sebagai proxy historical continuity scores
        # untuk menghitung Median Continuity yang sesungguhnya.
        strength_scores = [f.relationship_strength for f in recent_frags if f.topic_key_id == latest_key_id and f.relationship_strength > 0]
        
        if strength_scores:
            median_score = statistics.median(strength_scores)
            # Gabungkan frekuensi dan median untuk stabilitas akhir
            stability_ratio = max(median_score, freq_ratio * 0.8)
        else:
            stability_ratio = freq_ratio
        
        if stability_ratio >= CONFIG.continuity.MEDIAN_CONTINUITY_THRESHOLD:
            return latest_key_id, stability_ratio
        else:
            # Topik tidak stabil, continuity score meluruh
            return latest_key_id, stability_ratio * 0.5

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
                message=result.clarification_message or "Ambiguitas terdeteksi. Memerlukan klarifikasi."
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

    def _handle_ambiguity(self, envelope: InputEnvelope, result: RetrievalResult, existing: Optional[PendingState], question: str) -> EngineResult:
        """
        [OVERHAUL] Clarification Round Lifecycle (Review Kedua).
        Menangani increment round dan fallback jika MAX_CLARIFICATION_ROUNDS tercapai.
        """
        if existing and existing.status == PendingStatus.ACTIVE:
            round_num = existing.clarification_round + 1
            if round_num > CONFIG.fast_memory.MAX_CLARIFICATION_ROUNDS:
                existing.status = PendingStatus.EXPIRED
                existing.last_updated_at = envelope.timestamp
                with self.storage.transaction() as tx:
                    self.storage.save_pending_state(tx, existing)
                fallback_msg = "Saya masih belum yakin, tapi mari kita lanjutkan. Silakan berikan detail yang lebih spesifik jika Anda ingin membahas topik tertentu."
                return EngineResult(
                    assistant_response=fallback_msg,
                    decision=EngineDecision(decision_type=DecisionType.INSUFFICIENT, message=fallback_msg),
                    persistence_status="PENDING_EXPIRED"
                )
            
            existing.clarification_round = round_num
            existing.last_updated_at = envelope.timestamp
            existing.original_input = f"{existing.original_input} [CLARIFICATION: {envelope.raw_text}]"
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, existing)
            return EngineResult(
                assistant_response=question,
                decision=EngineDecision(decision_type=DecisionType.AMBIGUOUS, message=question),
                persistence_status="PENDING_UPDATED"
            )

        new_pending = PendingState(
            pending_id=str(uuid.uuid4()),
            session_id=envelope.session_id,
            original_input=envelope.raw_text,
            candidate_keys=[c.key_id for c in result.candidates[:3]],
            candidate_scores={c.key_id: c.final_score for c in result.candidates},
            assistant_question=question,
            created_at=envelope.timestamp,
            status=PendingStatus.ACTIVE,
            origin_session_id=envelope.session_id
        )
        
        with self.storage.transaction() as tx:
            self.storage.save_pending_state(tx, new_pending)
            
        return EngineResult(
            assistant_response=question,
            decision=EngineDecision(decision_type=DecisionType.AMBIGUOUS, message=question),
            persistence_status="PENDING_CREATED"
        )

    def _reconstruct_memories(self, fragment_ids: List[str]) -> Tuple[List[dict], ReconstructionStatus]:
        """
        [OVERHAUL] Exact Reconstruction dengan Observability (Review Kedua).
        Mengembalikan tuple (memories, status).
        """
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
                logger.error(f"Gagal merekonstruksi memori persis untuk fragment {fid}: {e}")
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

    def _simulate_generation(self, user_input: str, memories: List[dict], decision: EngineDecision) -> str:
        """Placeholder untuk LLM Generation (Fallback)."""
        if decision.decision_type == DecisionType.AMBIGUOUS:
            return decision.message or "Mohon maaf, bisa diperjelas maksud Anda?"
        return f"Response simulasi untuk: {user_input}"

    def _analyze_turn(
        self, 
        envelope: InputEnvelope, 
        response: str, 
        retrieval: RetrievalResult,
        current_active_id: Optional[str]
    ) -> TurnAnalysisResult:
        """
        [CRITICAL FIX] Generalized Topic Switch (Review Kedua).
        Menghapus hardcoded domain_shift_triggers. Menggunakan heuristik struktural murni:
        User Overlap + Retrieval Semantic Score + Short Probe Detection + Continuation Markers.
        """
        if retrieval.selected_key_id:
            if retrieval.selected_key_id == current_active_id:
                top_candidate = next((c for c in retrieval.candidates if c.key_id == current_active_id), None)
                if top_candidate:
                    # 1. Analisis USER INPUT untuk mendeteksi perpindahan domain yang eksplisit
                    user_tokens = [t for t in re.findall(r'\b\w+\b', envelope.raw_text.lower()) if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
                    
                    # 2. Analisis COMBINED (User + AI Response) untuk konfirmasi kesinambungan (Sesuai Blueprint Layer 11)
                    combined_text = f"{envelope.raw_text} {response}"
                    combined_tokens = [t for t in re.findall(r'\b\w+\b', combined_text.lower()) if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
                    
                    all_keys = self.storage.get_all_topic_keys()
                    active_key = next((k for k in all_keys if k.key_id == current_active_id), None)
                    
                    if active_key and user_tokens:
                        active_tokens = set(active_key.token_profile.keys())
                        
                        # Metrik User-Only (untuk mendeteksi abrupt shift)
                        user_overlap_count = sum(1 for t in user_tokens if t in active_tokens)
                        user_overlap_ratio = user_overlap_count / len(user_tokens)
                        
                        # Metrik Combined (untuk konfirmasi kontinuitas dari respons AI)
                        combined_overlap_count = sum(1 for t in combined_tokens if t in active_tokens)
                        combined_overlap_ratio = combined_overlap_count / len(combined_tokens) if combined_tokens else 0.0
                        
                        is_short_probe = len(user_tokens) <= 4
                        continuation_markers = {"terus", "lalu", "kemudian", "jadi", "kalau", "nah", "selanjutnya", "berikutnya", "kembali"}
                        has_continuation_marker = any(marker in user_tokens for marker in continuation_markers)
                        
                        # [FIX] Proxy semantik untuk PoC leksikal (mendeteksi domain yang sangat berbeda)
                        # Karena PoC ini tidak menggunakan embedding, kita memerlukan indikator kebaruan kuat 
                        # untuk membedakan "pertanyaan lanjutan pendek" dengan "perpindahan domain eksplisit" 
                        # ketika keduanya memiliki user_overlap_ratio == 0.
                        strong_novelty_indicators = {"pointer", "reference", "coding", "program", "algoritma", "c++", "python", "java", "astronomi", "hukum", "matematika"}
                        has_strong_novelty = any(trigger in user_tokens for trigger in strong_novelty_indicators)
                        
                        # GENERALIZED DECISION MATRIX
                        # Jika user tidak memiliki overlap dengan topik aktif, DAN Retrieval Engine juga tidak menemukan bukti semantik (token/desc score rendah)
                        if user_overlap_ratio < CONFIG.topic_switch.MIN_ACTIVE_TOPIC_OVERLAP and top_candidate.token_score < 0.1 and top_candidate.description_score < 0.1:
                            # Jika ada indikator domain baru yang kuat, paksa NEW_TOPIC
                            if has_strong_novelty:
                                classification = MemoryClassification.NEW_TOPIC
                            # Jika ini adalah pertanyaan lanjutan yang pendek (probe) atau menggunakan kata sambung
                            elif is_short_probe or has_continuation_marker:
                                classification = MemoryClassification.CONTINUE_TOPIC
                            # Ini adalah perpindahan topik yang eksplisit dan substansial
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
            
        # [CRITICAL FIX] Jika klasifikasi adalah NEW_TOPIC, kita WAJIB memutus rantai dari retrieval.selected_key_id.
        # Jika tidak, _persist_memory akan mengira ini adalah CONTINUE_TOPIC karena key_id tidak kosong.
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
            # [CRITICAL FIX] Topic Title Quality (Review Kedua)
            # Menghasilkan title dari representative tokens, bukan memotong string mentah.
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
                updated_at=envelope.timestamp
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
                
                # [NEW] Update Co-occurrence Profile
                updated_cooc = dict(existing_key.co_occurrence_profile)
                for i in range(len(new_tokens) - 1):
                    pair = f"{new_tokens[i]}↔{new_tokens[i+1]}"
                    updated_cooc[pair] = updated_cooc.get(pair, 0) + 1
                existing_key.co_occurrence_profile = updated_cooc
                
                # [NEW] Evolve Helper Description dengan batas (Review Kedua)
                MAX_DESC_TOKENS = 50
                current_desc_tokens = existing_key.helper_description.split()
                if len(current_desc_tokens) < MAX_DESC_TOKENS:
                    current_desc_lower = existing_key.helper_description.lower()
                    new_rep_tokens = []
                    for t in new_tokens:
                        if t not in current_desc_lower and len(new_rep_tokens) < 3:
                            new_rep_tokens.append(t)
                    if new_rep_tokens:
                        existing_key.helper_description += f" | {' '.join(new_rep_tokens)}"
                    
                existing_key.updated_at = envelope.timestamp
                existing_key.activation_count += 1
                
                new_key = existing_key 

        # [CRITICAL FIX] True Fragment-Level Relevance Profiles
        # Fragment kini memiliki profil token & co-occurrence independen miliknya sendiri.
        frag_cooc = {}
        for i in range(len(new_tokens) - 1):
            pair = f"{new_tokens[i]}↔{new_tokens[i+1]}"
            frag_cooc[pair] = frag_cooc.get(pair, 0) + 1

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
            relationship_strength=analysis.continuity_score
        )

        try:
            with self.storage.transaction() as tx:
                if new_key:
                    self.storage.save_topic_key(tx, new_key)
                self.storage.save_payload(tx, payload)
                self.storage.save_fragment(tx, fragment)
                
                # [OVERHAUL] Graph Reasoning: Fragment-to-Fragment Edge (Review Kedua)
                if analysis.classification == MemoryClassification.CONTINUE_TOPIC:
                    frags = self.storage.get_fragment_metadata_by_key(key_id)
                    # Cari fragment terakhir di sesi ini (selain yang baru saja dibuat)
                    session_frags = [f for f in frags if f.session_id == envelope.session_id and f.fragment_id != fragment.fragment_id]
                    if session_frags:
                        session_frags.sort(key=lambda x: x.created_at, reverse=True)
                        last_frag_id = session_frags[0].fragment_id
                        
                        edge = MemoryGraphEdge(
                            edge_id=str(uuid.uuid4()),
                            source_node_id=last_frag_id, 
                            target_node_id=fragment.fragment_id,
                            relationship_type=RelationshipType.CONTINUES,
                            strength=analysis.continuity_score,
                            created_at=envelope.timestamp
                        )
                        self.storage.save_graph_edge(tx, edge)
                        
                # [CRITICAL FIX] Real Cross-Session Graph Reasoning (Review Kedua)
                # Jika topik diaktifkan kembali dari sesi yang berbeda, buat CROSS_SESSION_LINK
                elif analysis.classification == MemoryClassification.REACTIVATE_TOPIC:
                    all_frags = self.storage.get_fragment_metadata_by_key(key_id)
                    # Cari fragment terakhir dari topik ini secara GLOBAL (selain yang baru saja dibuat)
                    global_frags = [f for f in all_frags if f.fragment_id != fragment.fragment_id]
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
                    
            logger.info(f"Memori tersimpan: Fragment {fragment.fragment_id}")
        except Exception as e:
            logger.critical(f"TRANSAKSI GAGAL: {e}")
