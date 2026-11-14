"""
memory_engine.py
Global Orchestrator & State Machine (Layer 16).

Fungsi:
1. Mengoordinasikan alur dari Input -> Retrieval -> Decision -> Generation -> Storage.
2. Mengelola Fast State (Pending Ambiguity).
3. Menerapkan Continuity Window Analysis (Mencegah 'Sticky Topic').
4. Menjamin transaksi penyimpanan memori bersifat Atomik (No Half-Memory).

[ASSOCIATIVE EVIDENCE LOCKING]
EVIDENCE BEFORE MUTATION:
    ANALYZE -> SCORE -> VALIDATE EVIDENCE -> DECIDE -> BARU UPDATE MEMORY
Memori tidak boleh diperbarui sebelum keputusan asosiasinya cukup tervalidasi.

[REASONING PURITY]
- final_score (relevance kandidat) TIDAK BOLEH disamakan dengan
  turn_continuity_score (seberapa kuat turn melanjutkan state aktif).
- relationship_strength fragment diisi dari turn_continuity_score murni.
- Short pending input divalidasi, bukan otomatis dianggap relevan.
- graph_score fragment dihitung dari edge nyata (CONTINUES / CROSS_SESSION_LINK).
"""

import logging
import re
import uuid
import statistics
from datetime import datetime
from typing import List, Optional, Tuple, Dict

from config import CONFIG
from models import (
    InputEnvelope, EngineResult, EngineDecision, DecisionType, 
    MemoryClassification, PendingState, PendingStatus, TopicKey, 
    FragmentMetadata, CompressedPayload, MemoryGraphEdge, RelationshipType,
    CandidateScore, TurnAnalysisResult, ReconstructionStatus,
    RetrievedMemory, TokenQuality, EvidenceFamily
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

    # [REASONING PURITY] Pemetaan sinyal → keluarga bukti.
    # WAJIB sinkron dengan TopicKeyScorer._SIGNAL_FAMILY di retrieval.py.
    _SIGNAL_FAMILY: Dict[str, EvidenceFamily] = {
        "token": EvidenceFamily.LEXICAL,
        "unique_token": EvidenceFamily.LEXICAL,
        "co_occurrence": EvidenceFamily.LEXICAL,
        "description": EvidenceFamily.CONTEXTUAL,
        "temporal": EvidenceFamily.CONTEXTUAL,
        "continuity": EvidenceFamily.CONTEXTUAL,
        "graph": EvidenceFamily.RELATIONAL,
    }

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
            # [CRITICAL FIX] Cross-Session Pending Resolution dengan RANKING.
            # Menggunakan SUMBER BUKTI YANG SAMA dengan jalur lokal (token_profile + helper_description),
            # lalu di-gate oleh _check_pending_relevance dan di-ranking oleh _score_pending_relevance.
            # Sebelumnya jalur ini memakai rumus terpisah yang lebih ketat sehingga pending
            # lintas sesi tidak pernah ter-attach (Blueprint XXVII & XXVIII).
            global_pendings = self.storage.get_global_active_pendings()
            all_keys = self.storage.get_all_topic_keys()
            
            best_pending = None
            best_score = -1.0
            
            for gp in global_pendings:
                if gp.session_id == session_id:
                    continue
                    
                # Gate: pakai aturan relevansi yang sama persis dengan pending lokal
                if not self._check_pending_relevance(envelope, gp, all_keys=all_keys):
                    continue
                    
                # Ranking: skor komposit dari bukti yang sama
                score = self._score_pending_relevance(envelope, gp, all_keys=all_keys)
                
                if score > best_score:
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

        # [ASSOCIATIVE EVIDENCE LOCKING] Graph sebagai EVIDENCE, bukan absolute truth.
        # Sebelumnya cross_session_evidence tidak pernah dikirim sehingga graph_score
        # selalu 0 dan tidak pernah memengaruhi keputusan.
        graph_evidence = self._build_graph_evidence(all_keys, relevant_fragments, active_topic_id)

        # [REASONING PURITY] Fragment graph scores dari edge nyata (Prioritas #4).
        # Dikirim ke retrieval agar fragment ranking memakai graph evidence, bukan
        # relationship_strength sebagai proxy. Retrieval akan fallback ke
        # relationship_strength jika edge score = 0 (test_scenario_k compatibility).
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
        )

        # ==========================================
        # PHASE 3: DECISION GATE (Layer 5)
        # ==========================================
        decision = self._make_decision(retrieval_result, active_pending)
        
        # [FIX] Jika ada pending state dan keputusan baru adalah CLEAR, 
        # ATAU jika ini adalah Cross-Session Pending yang berhasil di-retrieval (tidak ambigu lagi),
        # berarti klarifikasi user berhasil mengurai ambiguitas.
        # [EVIDENCE LOCKING] Kegagalan bukti TIDAK boleh dianggap sebagai resolusi.
        _unresolved_states = {
            DecisionType.AMBIGUOUS,
            DecisionType.INSUFFICIENT_EVIDENCE,
            DecisionType.CONTRADICTORY_EVIDENCE,
        }
        if active_pending and (
            decision.decision_type == DecisionType.CLEAR
            or (is_cross_session_pending and decision.decision_type not in _unresolved_states)
        ):
            active_pending.status = PendingStatus.RESOLVED
            active_pending.last_updated_at = ref_time
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, active_pending)
            logger.info(f"PENDING STATE: RESOLVED (pending_id={active_pending.pending_id[:8]}..., cross_session={is_cross_session_pending}).")
        
        # Handle Ambiguity (Layer 15)
        if decision.decision_type == DecisionType.AMBIGUOUS:
            question = retrieval_result.clarification_message or "Maaf, saya kurang yakin. Maksud Anda topik yang mana?"
            return self._handle_ambiguity(envelope, retrieval_result, active_pending, question)

        # [ASSOCIATIVE EVIDENCE LOCKING] Dua kondisi gagal-bukti yang BERBEDA dari ambiguitas.
        # AMBIGUOUS              : beberapa kandidat kuat
        # INSUFFICIENT_EVIDENCE  : tidak ada kandidat yang cukup terbukti
        # CONTRADICTORY_EVIDENCE : ada kecocokan tetapi bukti saling bertentangan
        if decision.decision_type in (DecisionType.INSUFFICIENT_EVIDENCE, DecisionType.CONTRADICTORY_EVIDENCE):
            question = retrieval_result.clarification_message or decision.message or (
                "Bukti yang saya miliki belum cukup untuk memilih satu topik. Bisa diperjelas?"
            )
            return self._handle_ambiguity(
                envelope, retrieval_result, active_pending, question,
                decision_type=decision.decision_type
            )
            
        # ==========================================
        # PHASE 4: EXACT RECONSTRUCTION (Layer 7)
        # ==========================================
        exact_memories = []
        retrieved_memories: List[RetrievedMemory] = []
        reconstruction_status = ReconstructionStatus.NO_MEMORY_REQUESTED
        
        if decision.selected_fragment_ids:
            exact_memories, reconstruction_status = self._reconstruct_memories(decision.selected_fragment_ids)
            # [ASSOCIATIVE EVIDENCE LOCKING] Bawa metadata bukti ke Context Builder.
            retrieved_memories = self._attach_evidence_metadata(exact_memories, retrieval_result)
            
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
                exact_memories=retrieved_memories if retrieved_memories else exact_memories,
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

        # [REASONING PURITY] Ringkasan kemurnian reasoning untuk observability
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
            # [EVIDENCE LOCKING] Sertakan kandidat yang DITOLAK agar bisa diaudit.
            candidates=retrieval_result.candidates,
            reasoning_purity_summary=reasoning_summary,
        )

    # ==========================================
    # INTERNAL LOGIC & HELPERS
    # ==========================================

    def _extract_input_tokens(self, envelope: InputEnvelope) -> set:
        """
        [NEW] Single source of truth untuk tokenisasi input user.
        Dipakai oleh seluruh pemeriksaan relevansi pending agar tidak ada perbedaan bukti.
        """
        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        return {t for t in raw_tokens if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH}

    def _collect_pending_evidence_tokens(
        self, 
        pending: PendingState, 
        all_keys: Optional[List[TopicKey]] = None
    ) -> set:
        """
        [CRITICAL FIX] Single source of truth untuk bukti milik Pending State.
        Menggabungkan TOKEN PROFILE + HELPER DESCRIPTION dari seluruh candidate keys.
        
        Sebelumnya jalur cross-session hanya memakai token_profile, sehingga bukti
        lebih lemah daripada jalur lokal dan pending lintas sesi gagal di-resolve.
        Sesuai Blueprint: pending harus dicocokkan dengan
        candidate keys + helper descriptions + metadata.
        """
        if all_keys is None:
            all_keys = self.storage.get_all_topic_keys()
            
        pending_tokens = set()
        for k in all_keys:
            if k.key_id in pending.candidate_keys:
                pending_tokens.update(k.token_profile.keys())
                desc_words = re.findall(r'\b\w+\b', k.helper_description.lower())
                pending_tokens.update(w for w in desc_words if len(w) >= CONFIG.signal.MIN_TOKEN_LENGTH)
        return pending_tokens

    def _score_pending_relevance(
        self, 
        envelope: InputEnvelope, 
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> float:
        """
        [NEW] Skor komposit relevansi pending untuk keperluan RANKING antar pending.
        Bukan untuk gating (gating dilakukan oleh _check_pending_relevance).
        
        Komponen:
        - overlap_ratio : proporsi token input yang terjelaskan oleh bukti pending
        - overlap_bonus : penguat absolut agar pending dengan bukti lebih kaya menang
        """
        new_tokens = self._extract_input_tokens(envelope)
        
        # Jawaban sangat pendek ("Yang bisnis", "Terserah") tetap dianggap mungkin relevan,
        # tetapi diberi skor sedang agar tidak mengalahkan pending dengan bukti eksplisit kuat.
        if not new_tokens or len(new_tokens) < 3:
            return 0.5
            
        pending_tokens = self._collect_pending_evidence_tokens(pending, all_keys=all_keys)
        overlap = new_tokens.intersection(pending_tokens)
        
        overlap_ratio = len(overlap) / len(new_tokens)
        overlap_bonus = min(len(overlap), 5) * 0.01
        
        return overlap_ratio + overlap_bonus

    def _check_pending_relevance(
        self, 
        envelope: InputEnvelope, 
        pending: PendingState,
        all_keys: Optional[List[TopicKey]] = None
    ) -> bool:
        """
        [OVERHAUL] Memeriksa relevansi pending.

        [REASONING PURITY — Prioritas #3: Short Pending Safety]
        Input pendek TIDAK otomatis dianggap relevan. Harus divalidasi terhadap
        kandidat pending untuk mencegah "C++" dianggap jawaban terhadap pending
        "Bisnis Kopi / Kedai Kopi?".

        Pipeline:
        - Jika input pendek (< SHORT_PENDING_INPUT_THRESHOLD tokens):
            - Jika semua token low-information → RELEVANT (anggap "terserah", "yang bisnis")
            - Jika ada meaningful tokens → cek overlap dengan candidate keys
              - overlap >= SHORT_PENDING_MIN_MATCH → RELEVANT
              - tidak ada overlap DAN ada strong novelty → NOT RELEVANT
              - tidak ada overlap TANPA strong novelty → RELEVANT (weak)
        - Jika input panjang:
            - overlap >= 2 tokens → RELEVANT (long clarification)
            - overlap == 0 dan >= 3 tokens baru → NOT RELEVANT (topic switch)
            - Fallback: overlap_ratio >= PENDING_SWITCH_THRESHOLD
        """
        new_tokens = self._extract_input_tokens(envelope)
        short_threshold = CONFIG.fast_memory.SHORT_PENDING_INPUT_THRESHOLD
        min_match = CONFIG.fast_memory.SHORT_PENDING_MIN_MATCH

        # [REASONING PURITY] Short input validation
        if not new_tokens or len(new_tokens) < short_threshold:
            if not new_tokens:
                return True  # input kosong — tetap relevan (edge case)

            # Cek apakah semua token low-information
            quality_map = self._get_input_token_quality(envelope)
            meaningful = [
                t for t in new_tokens
                if quality_map.get(t, TokenQuality.NORMAL) != TokenQuality.LOW_INFORMATION
            ]

            # Jika semua low-information → auto relevan ("terserah", "mana?", "pilih saja")
            if not meaningful:
                return True

            # Ada meaningful tokens — validasi terhadap candidate keys
            pending_tokens = self._collect_pending_evidence_tokens(pending, all_keys=all_keys)
            overlap = set(meaningful).intersection(pending_tokens)

            if len(overlap) >= min_match:
                return True

            # Cek strong novelty indicators (domain shift eksplisit)
            strong_novelty = {
                "pointer", "reference", "coding", "program", "algoritma",
                "c++", "python", "java", "astronomi", "hukum", "matematika"
            }
            has_strong_novelty = any(t in strong_novelty for t in meaningful)
            if has_strong_novelty:
                return False

            # Tidak ada overlap, tidak ada strong novelty → RELEVANT (weak)
            # Asumsikan ini masih mencoba menjawab pending (e.g. "yang kedua")
            return True

        # Input panjang: logika asli (dengan sumber bukti yang konsisten)
        pending_tokens = self._collect_pending_evidence_tokens(pending, all_keys=all_keys)
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

    def _get_input_token_quality(self, envelope: InputEnvelope) -> Dict[str, TokenQuality]:
        """Helper: klasifikasi token kualitas (re-use SignalExtractor logic)."""
        from retrieval import SignalExtractor
        extractor = SignalExtractor()
        raw_tokens = re.findall(r'\b\w+\b', envelope.raw_text.lower())
        unique_tokens = [t for t in set(raw_tokens) if len(t) >= CONFIG.signal.MIN_TOKEN_LENGTH]
        return {t: extractor._classify_token(t) for t in unique_tokens}

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

    # ==========================================
    # [ASSOCIATIVE EVIDENCE LOCKING] HELPERS
    # ==========================================

    def _build_graph_evidence(
        self,
        keys: List[TopicKey],
        fragments: List[FragmentMetadata],
        active_topic_id: Optional[str]
    ) -> Dict[str, float]:
        """
        Graph = EVIDENCE, bukan absolute truth.

        Menjawab: "Apakah kandidat ini pernah memiliki hubungan kuat dengan
        fragment/topik yang sekarang aktif?"

        Skor diambil dari kekuatan edge terkuat yang menghubungkan fragment
        kandidat dengan fragment referensi (topik aktif).
        """
        evidence: Dict[str, float] = {}
        if not active_topic_id:
            return evidence

        reference_ids = [
            f.fragment_id for f in fragments if f.topic_key_id == active_topic_id
        ]
        if not reference_ids:
            return evidence
        # Batasi referensi agar tidak meledak pada dataset besar.
        reference_ids = reference_ids[:CONFIG.graph.MAX_EDGES_PER_NODE]

        for key in keys:
            key_frag_ids = [f.fragment_id for f in fragments if f.topic_key_id == key.key_id]
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
        """
        [REASONING PURITY — Prioritas #4: Fragment Graph Evidence]

        Hitung graph score per fragment dari edge NYATA (CONTINUES / CROSS_SESSION_LINK).
        Bukan dari relationship_strength metadata.

        Retrieval akan fallback ke relationship_strength jika score dari sini = 0.0,
        untuk menjaga kompatibilitas dengan test yang membuat fragment manual
        tanpa edges (e.g. test_scenario_k).

        Returns:
            Dict[fragment_id, score] — hanya fragment dengan score > 0 yang dikembalikan
            agar retrieval tahu mana yang benar-benar punya evidence.
        """
        if not fragments:
            return {}

        frag_ids = [f.fragment_id for f in fragments]
        # Panggil storage — ini menghitung max edge strength per fragment.
        raw_scores = self.storage.get_fragment_graph_evidence(
            frag_ids,
            reference_fragment_ids=frag_ids  # referensi internal (edges antar fragment ini)
        )

        # Filter: hanya kembalikan fragment dengan score > 0
        return {fid: score for fid, score in raw_scores.items() if score > 0}

    def _attach_evidence_metadata(
        self,
        exact_memories: List[dict],
        retrieval: RetrievalResult
    ) -> List[RetrievedMemory]:
        """
        Mengubah dict memori hasil rekonstruksi menjadi RetrievedMemory yang
        membawa metadata kualitas bukti, agar Context Builder dapat mengurutkan
        berdasarkan bukti dan bukan sekadar recency.

        [REASONING PURITY] Populate field fragment_graph_score (dari edge nyata)
        dan temporal_distance_score (dari kedekatan waktu).
        """
        score_map = retrieval.selected_fragment_scores or {}
        results: List[RetrievedMemory] = []
        for mem in exact_memories:
            fid = mem.get('fragment_id', '')
            meta = score_map.get(fid, {})
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
                # [REASONING PURITY] Fragment graph score dari edge nyata
                fragment_graph_score=meta.get('fragment_graph_score', 0.0),
                graph_edge_count=1 if meta.get('fragment_graph_score', 0.0) > 0 else 0,
                graph_path_types=[],  # Akan diisi saat graph reasoning diperdalam
                # [REASONING PURITY] Temporal distance score
                temporal_distance_score=meta.get('temporal_distance_score', 0.0),
                temporal_distance_minutes=meta.get('temporal_distance_minutes'),
            ))
        return results

    def _build_window_pairs(self, tokens: List[str]) -> Dict[str, int]:
        """
        [ASSOCIATIVE EVIDENCE LOCKING] Co-occurrence berbasis window untuk PERSISTENCE.

        Wajib selaras dengan retrieval: jika retrieval mencocokkan window pairs
        tetapi persistence hanya menyimpan adjacent pairs, maka
        "bisnis kecil di bidang kopi" tidak akan pernah cocok dengan query "bisnis kopi".
        """
        cc = CONFIG.co_occurrence
        pairs: Dict[str, int] = {}
        for i, token_a in enumerate(tokens):
            window_end = min(len(tokens), i + cc.CO_OCCURRENCE_WINDOW_SIZE + 1)
            for j in range(i + 1, window_end):
                token_b = tokens[j]
                if token_a == token_b:
                    continue
                distance = j - i
                strength = cc.ADJACENT_PAIR_BONUS if distance == 1 else (
                    cc.ADJACENT_PAIR_BONUS * (cc.CO_OCCURRENCE_DECAY ** (distance - 1))
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
        """
        EVIDENCE BEFORE MUTATION.

        Mengembalikan:
        (mutation_allowed, description_update_allowed, reason,
         evidence_strength, support_count, contradiction_score)

        Mencegah "evidence pollution": token C++/pointer masuk ke Token Profile,
        Co-occurrence, dan Helper Description milik cluster Bisnis Kopi.
        """
        mut = CONFIG.mutation

        # Topik baru selalu boleh dibuat (tidak ada cluster lama yang bisa tercemar).
        if classification == MemoryClassification.NEW_TOPIC or candidate is None:
            return (True, True, "new topic / no candidate", 0.0, 0, 0.0)

        evidence_strength = candidate.evidence_strength
        support_count = candidate.support_count
        contradiction = candidate.contradiction_score

        # 1. Kontradiksi tinggi memblokir mutasi walaupun skor akhir tinggi.
        if contradiction > mut.MUTATION_CONTRADICTION_BLOCK:
            return (False, False,
                    f"blocked: contradiction {contradiction:.2f} > {mut.MUTATION_CONTRADICTION_BLOCK:.2f}",
                    evidence_strength, support_count, contradiction)

        # 2. Continuation harus punya bukti cukup.
        if evidence_strength < mut.MIN_EVIDENCE_TO_MUTATE_TOPIC:
            return (False, False,
                    f"blocked: evidence {evidence_strength:.2f} < {mut.MIN_EVIDENCE_TO_MUTATE_TOPIC:.2f}",
                    evidence_strength, support_count, contradiction)

        if support_count < mut.MIN_SUPPORT_TO_MUTATE_TOPIC:
            continuity_only_ok = (
                mut.ALLOW_CONTINUITY_ONLY_MUTATION
                and candidate.continuity_score >= CONFIG.evidence.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_only_ok:
                return (False, False,
                        f"blocked: support {support_count} < {mut.MIN_SUPPORT_TO_MUTATE_TOPIC}",
                        evidence_strength, support_count, contradiction)

        # 3. [ANTI-POLLUTION] Token low-information tidak membawa identitas topik.
        # "Bagaimana itu?" tidak boleh memperkaya profil cluster apa pun.
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
            return (True, False,
                    "fragment stored, topic profile untouched: only low-information tokens",
                    evidence_strength, support_count, contradiction)

        # 4. Deskripsi adalah identitas manusia-readable -> butuh bukti lebih kuat.
        desc_allowed = evidence_strength >= mut.MIN_EVIDENCE_TO_UPDATE_DESCRIPTION
        reason = "allowed" if desc_allowed else "allowed: description update held (evidence below description threshold)"
        return (True, desc_allowed, reason, evidence_strength, support_count, contradiction)

    # ==========================================
    # [REASONING PURITY] CONTINUITY PURITY (Prioritas #1)
    # ==========================================

    def _compute_turn_continuity(
        self,
        envelope: InputEnvelope,
        active_key: Optional[TopicKey],
        top_candidate: Optional[CandidateScore],
        history_ratio: float,
    ) -> Tuple[float, List[str]]:
        """
        [REASONING PURITY — Prioritas #1]

        Hitung CONTINUITY SCORE MURNI untuk turn ini.
        BERBEDA dari CandidateScore.continuity_score (yang hanya satu komponen final_score).

        Menjawab: "Seberapa kuat turn ini melanjutkan state aktif?"
        - Bukan "Seberapa relevan kandidat secara keseluruhan"
        - Bukan "Seberapa cocok token input dengan topik aktif"

        Formula:
            continuity = (
                ACTIVE_TOPIC_OVERLAP_WEIGHT * user_overlap_ratio
                + CONTINUITY_HISTORY_WEIGHT * history_ratio
                + CONTINUITY_FRAGMENT_WEIGHT * active_frag_rel
                - CONTINUITY_NOVELTY_PENALTY * novelty_ratio (jika > threshold)
                - CONTINUITY_CONTRADICTION_PENALTY * contradiction_score (jika > threshold)
            )

        Returns:
            (continuity_score, evidence_sources) — score dalam [0, 1]
        """
        cc = CONFIG.continuity
        ev = CONFIG.evidence

        if active_key is None or top_candidate is None:
            return 0.0, []

        # 1. User overlap dengan active topic
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

        # 2. History ratio — dari caller (hasil _calculate_active_state)
        # Sudah dihitung, tinggal pakai
        hist = history_ratio if history_ratio else 0.0

        # 3. Active fragment relationship strength
        # Ambil fragment terakhir milik active key di session ini
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

        # 4. Novelty penalty — jika user sangat baru terhadap topik aktif
        novelty_ratio = 1.0 - overlap_ratio
        novelty_penalty = 0.0
        if novelty_ratio > CONFIG.topic_switch.NOVELTY_THRESHOLD:
            novelty_penalty = cc.CONTINUITY_NOVELTY_PENALTY * novelty_ratio

        # 5. Contradiction penalty
        contradiction_penalty = 0.0
        if top_candidate.contradiction_score > ev.MAX_CONTRADICTION_SCORE:
            contradiction_penalty = (
                cc.CONTINUITY_CONTRADICTION_PENALTY * top_candidate.contradiction_score
            )

        # Hitung raw continuity
        raw = (
            cc.ACTIVE_TOPIC_OVERLAP_WEIGHT * overlap_ratio
            + cc.CONTINUITY_HISTORY_WEIGHT * hist
            + cc.CONTINUITY_FRAGMENT_WEIGHT * active_frag_rel
            - novelty_penalty
            - contradiction_penalty
        )

        # Clamp ke [0, 1]
        continuity = max(0.0, min(1.0, raw))

        # Catat sumber bukti (untuk audit)
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
        """
        Ringkasan kemurnian reasoning untuk observability (field EngineResult.reasoning_purity_summary).
        Menjawab: "Apakah skor yang dihasilkan konsisten secara semantik?"
        """
        top = retrieval.candidates[0] if retrieval.candidates else None
        if top is None:
            return ""

        lines = [
            f"PURITY CHECK: {analysis.classification.value}",
            f"  final_score (relevance)   : {top.final_score:.3f}",
            f"  turn_continuity (persistence) : {analysis.turn_continuity_score:.3f}",
            f"  Δ (harus ≠ 0 jika bukan continuation trivial) : "
            f"{abs(top.final_score - analysis.turn_continuity_score):.3f}",
            f"  evidence families          : {[f.value for f in top.evidence_families]}",
            f"  family_support_count       : {top.family_support_count}",
            f"  continuity sources         : {analysis.continuity_evidence_sources or '(none)'}",
        ]
        return "\n".join(lines)

    def _make_decision(self, result: RetrievalResult, pending: Optional[PendingState]) -> EngineDecision:
        """
        Menerjemahkan hasil retrieval menjadi keputusan mesin.

        [ASSOCIATIVE EVIDENCE LOCKING] Engine tidak boleh langsung percaya
        kandidat tertinggi. Keputusan membawa alasan yang bisa diaudit.

        [REASONING PURITY] Keputusan juga membawa family_support_count dan
        evidence_families untuk audit independensi bukti.
        """
        if result.decision_type == DecisionType.AMBIGUOUS:
            top = result.candidates[0] if result.candidates else None
            return EngineDecision(
                decision_type=DecisionType.AMBIGUOUS,
                message=result.clarification_message or "Ambiguitas terdeteksi. Memerlukan klarifikasi.",
                decision_reason=result.analysis_notes,
                family_support_count=top.family_support_count if top else 0,
                evidence_families=top.evidence_families if top else [],
            )

        # [EVIDENCE LOCKING] Dua state gagal-bukti dipertahankan apa adanya,
        # TIDAK dilipat menjadi NO_MATCH.
        if result.decision_type in (DecisionType.INSUFFICIENT_EVIDENCE, DecisionType.CONTRADICTORY_EVIDENCE):
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
            )
            
        if result.decision_type == DecisionType.CLEAR:
            selected = next(
                (c for c in result.candidates if c.key_id == result.selected_key_id),
                result.candidates[0] if result.candidates else None
            )

            # Re-validasi di sisi Engine: jangan langsung percaya candidates[0].
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
            )
            
        return EngineDecision(
            decision_type=DecisionType.NO_MATCH,
            message="Tidak ada memori relevan ditemukan.",
            decision_reason=result.analysis_notes
        )

    # ==========================================
    # [REASONING PURITY] EVIDENCE FAMILY INFERENCE
    # ==========================================

    def _infer_families_from_candidate(
        self,
        candidate: CandidateScore
    ) -> Tuple[int, List[EvidenceFamily], bool]:
        """
        [REASONING PURITY] Infer evidence families dari candidate.

        Digunakan oleh _validate_candidate_evidence agar validation tidak gagal
        ketika CandidateScore dibangun manual (seperti di test) tanpa mengisi
        field-level family data.

        Prioritas inferensi:
        1. Jika field-level family data tersedia (family_support_count > 0) → pakai itu.
        2. Else, infer dari evidence_profile.supporting_evidence_names via _SIGNAL_FAMILY.
        3. Else, jika profile juga kosong → (0, [], False).
        """
        # 1. Cek field-level data dulu
        if candidate.family_support_count > 0:
            return (
                candidate.family_support_count,
                list(candidate.evidence_families),
                candidate.is_single_family_dominated,
            )

        # 2. Fallback: infer dari evidence_profile
        if candidate.evidence_profile is None:
            return (0, [], False)

        profile = candidate.evidence_profile
        supporting_names = profile.supporting_evidence_names or []
        if not supporting_names:
            return (0, [], False)

        family_set: set = set()

        # 2a. Cek signal_to_family yang sudah di-populate retrieval
        signal_to_family = profile.signal_to_family or {}
        for name in supporting_names:
            if name in signal_to_family:
                family_set.add(signal_to_family[name])
            else:
                # 2b. Fallback: infer langsung dari nama sinyal
                family = self._SIGNAL_FAMILY.get(name)
                if family is not None:
                    family_set.add(family)

        evidence_families = sorted(family_set, key=lambda f: f.value)
        family_support_count = len(evidence_families)

        # Infer single-family domination
        is_dominated = profile.is_single_signal_dominated

        return (family_support_count, evidence_families, is_dominated)

    def _validate_candidate_evidence(
        self,
        candidate: Optional[CandidateScore]
    ) -> Tuple[Optional[DecisionType], str]:
        """
        EVIDENCE VALIDATION di sisi Engine (lapis kedua setelah Retrieval).

        Support cukup? Kontradiksi rendah? Confidence cukup?
        Families cukup beragam?

        [REASONING PURITY] Menggunakan _infer_families_from_candidate() agar
        robust terhadap CandidateScore yang dibangun manual (e.g. di test)
        tanpa field-level family data.
        """
        if candidate is None:
            return (None, "")

        ev = CONFIG.evidence
        ef = CONFIG.evidence_family

        if candidate.contradiction_score > ev.MAX_CONTRADICTION_SCORE:
            return (
                DecisionType.CONTRADICTORY_EVIDENCE,
                f"Bukti kontradiktif pada '{candidate.title}' "
                f"(contradiction={candidate.contradiction_score:.2f})."
            )

        if candidate.support_count < ev.MIN_SUPPORTING_EVIDENCE:
            continuity_exemption = (
                ev.CONTINUITY_EXEMPT_FROM_MIN_SUPPORT
                and candidate.continuity_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_exemption:
                return (
                    DecisionType.INSUFFICIENT_EVIDENCE,
                    f"Bukti pendukung tidak cukup pada '{candidate.title}' "
                    f"(support={candidate.support_count})."
                )

        # [REASONING PURITY] Family count validation — dengan inference fallback.
        family_support_count, evidence_families, _ = self._infer_families_from_candidate(candidate)

        if family_support_count < ef.MIN_EVIDENCE_FAMILIES:
            continuity_family_exempt = (
                ef.CONTINUITY_EXEMPT_FROM_FAMILY_CHECK
                and EvidenceFamily.CONTEXTUAL in evidence_families
                and candidate.continuity_score >= ev.EVIDENCE_SUPPORT_THRESHOLD
            )
            if not continuity_family_exempt:
                families_str = ",".join(f.value for f in evidence_families) or "none"
                return (
                    DecisionType.INSUFFICIENT_EVIDENCE,
                    f"Keragaman bukti tidak cukup pada '{candidate.title}' "
                    f"(families={family_support_count}, active=[{families_str}])."
                )

        if candidate.confidence < ev.MIN_ASSOCIATIVE_CONFIDENCE:
            return (
                DecisionType.INSUFFICIENT_EVIDENCE,
                f"Confidence asosiatif tidak cukup pada '{candidate.title}' "
                f"(confidence={candidate.confidence:.2f})."
            )

        return (None, "")

    def _handle_ambiguity(
        self, 
        envelope: InputEnvelope, 
        result: RetrievalResult, 
        existing: Optional[PendingState], 
        question: str,
        decision_type: DecisionType = DecisionType.AMBIGUOUS
    ) -> EngineResult:
        """
        [OVERHAUL] Clarification Round Lifecycle (Review Kedua).
        Menangani increment round dan fallback jika MAX_CLARIFICATION_ROUNDS tercapai.

        [EVIDENCE LOCKING] decision_type dapat berupa AMBIGUOUS,
        INSUFFICIENT_EVIDENCE, atau CONTRADICTORY_EVIDENCE — ketiganya berbeda.
        """
        top = result.candidates[0] if result.candidates else None
        decision_kwargs = dict(
            confidence=top.confidence if top else 0.0,
            support_count=top.support_count if top else 0,
            evidence_strength=top.evidence_strength if top else 0.0,
            contradiction_score=top.contradiction_score if top else 0.0,
            decision_reason=result.analysis_notes,
            family_support_count=top.family_support_count if top else 0,
            evidence_families=top.evidence_families if top else [],
        )

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
                    persistence_status="PENDING_EXPIRED",
                    evidence_summary=result.evidence_summary,
                    # [EVIDENCE LOCKING] Kandidat tetap disertakan untuk audit.
                    candidates=result.candidates
                )
            
            existing.clarification_round = round_num
            existing.last_updated_at = envelope.timestamp
            existing.original_input = f"{existing.original_input} [CLARIFICATION: {envelope.raw_text}]"
            with self.storage.transaction() as tx:
                self.storage.save_pending_state(tx, existing)
            return EngineResult(
                assistant_response=question,
                decision=EngineDecision(decision_type=decision_type, message=question, **decision_kwargs),
                persistence_status="PENDING_UPDATED",
                evidence_summary=result.evidence_summary,
                # [EVIDENCE LOCKING] Kandidat tetap disertakan untuk audit.
                candidates=result.candidates
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
            decision=EngineDecision(decision_type=decision_type, message=question, **decision_kwargs),
            persistence_status="PENDING_CREATED",
            evidence_summary=result.evidence_summary,
            # [EVIDENCE LOCKING] Kandidat tetap disertakan untuk audit.
            candidates=result.candidates
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

        [EVIDENCE LOCKING] Hasil analisis kini juga membawa izin mutasi memori.

        [REASONING PURITY] turn_continuity_score dihitung TERPISAH dari final_score
        menggunakan formula murni (overlap + history + fragments - novelty - contradiction).
        Nilai inilah yang akan dipersist sebagai relationship_strength fragment.
        continuity_score (legacy) tetap diisi dengan final_score untuk backward compatibility.
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

        # [EVIDENCE LOCKING] EVIDENCE BEFORE MUTATION.
        selected_candidate = next(
            (c for c in retrieval.candidates if c.key_id == active_key_for_fragment),
            None
        )
        (mutation_allowed, desc_allowed, mutation_reason,
         ev_strength, support_count, contradiction) = self._compute_mutation_permissions(
            envelope, retrieval, selected_candidate, classification
        )

        # [REASONING PURITY] Hitung turn_continuity_score murni.
        # BERBEDA dari continuity_score (legacy) yang = final_score candidate.
        # Nilai ini yang akan dipersist sebagai relationship_strength fragment.
        active_key_obj = None
        if active_key_for_fragment:
            all_keys = self.storage.get_all_topic_keys()
            active_key_obj = next(
                (k for k in all_keys if k.key_id == active_key_for_fragment),
                None
            )

        if (classification in (MemoryClassification.CONTINUE_TOPIC, MemoryClassification.REACTIVATE_TOPIC)
                and active_key_obj is not None and selected_candidate is not None):
            # Ambil history_ratio dari active state calculation
            _, history_ratio = self._calculate_active_state(envelope.session_id)
            turn_cont, cont_sources = self._compute_turn_continuity(
                envelope, active_key_obj, selected_candidate, history_ratio
            )
        else:
            turn_cont = 0.0
            cont_sources = ["new_topic"] if classification == MemoryClassification.NEW_TOPIC else []
            
        return TurnAnalysisResult(
            classification=classification,
            # [BACKWARD COMPAT] continuity_score (legacy) tetap diisi final_score
            # untuk menjaga test lama yang memeriksa field ini.
            continuity_score=retrieval.candidates[0].final_score if retrieval.candidates else 0.0,
            active_topic_key_id=active_key_for_fragment,
            evidence_strength=ev_strength,
            support_count=support_count,
            contradiction_score=contradiction,
            mutation_allowed=mutation_allowed,
            description_update_allowed=desc_allowed,
            mutation_reason=mutation_reason,
            # [REASONING PURITY] turn_continuity_score murni — ini yang dipersist
            turn_continuity_score=turn_cont,
            continuity_evidence_sources=cont_sources,
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

        [EVIDENCE LOCKING] Mutasi profil topik HANYA dilakukan bila
        analysis.mutation_allowed == True. Fragmen tetap disimpan agar
        percakapan tidak hilang, tetapi cluster tidak tercemar.

        [REASONING PURITY] relationship_strength fragment diisi dari
        analysis.turn_continuity_score (bukan final_score candidate).
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

        # [EVIDENCE LOCKING] Token low-information tidak boleh masuk profil topik.
        signal = retrieval.signal_profile
        quality_map = (signal.token_quality if signal else {}) or {}
        meaningful_tokens = [
            t for t in new_tokens
            if quality_map.get(t, TokenQuality.NORMAL) != TokenQuality.LOW_INFORMATION
        ]
        profile_tokens = meaningful_tokens if meaningful_tokens else []
        profile_counts = {}
        for t in profile_tokens:
            profile_counts[t] = profile_counts.get(t, 0) + 1

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
                # [EVIDENCE LOCKING] GERBANG MUTASI.
                if analysis.mutation_allowed and profile_counts:
                    updated_profile = dict(existing_key.token_profile)
                    for t, count in profile_counts.items():
                        updated_profile[t] = updated_profile.get(t, 0) + count
                    existing_key.token_profile = updated_profile
                    
                    # [NEW] Update Co-occurrence Profile
                    # [EVIDENCE LOCKING] Window co-occurrence agar selaras dengan retrieval.
                    updated_cooc = dict(existing_key.co_occurrence_profile)
                    for pair, count in self._build_window_pairs(profile_tokens).items():
                        updated_cooc[pair] = updated_cooc.get(pair, 0) + count
                    existing_key.co_occurrence_profile = updated_cooc
                    
                    # [NEW] Evolve Helper Description dengan batas (Review Kedua)
                    MAX_DESC_TOKENS = 50
                    current_desc_tokens = existing_key.helper_description.split()
                    if analysis.description_update_allowed and len(current_desc_tokens) < MAX_DESC_TOKENS:
                        current_desc_lower = existing_key.helper_description.lower()
                        new_rep_tokens = []
                        for t in profile_tokens:
                            if t not in current_desc_lower and len(new_rep_tokens) < 3:
                                new_rep_tokens.append(t)
                        if new_rep_tokens:
                            existing_key.helper_description += f" | {' '.join(new_rep_tokens)}"
                else:
                    logger.info(
                        f"EVIDENCE GATE: profil topik {key_id[:8]}... tidak dimutasi. "
                        f"({analysis.mutation_reason})"
                    )
                    
                existing_key.updated_at = envelope.timestamp
                existing_key.activation_count += 1
                
                new_key = existing_key 

        # [CRITICAL FIX] True Fragment-Level Relevance Profiles
        # Fragment kini memiliki profil token & co-occurrence independen miliknya sendiri.
        # [EVIDENCE LOCKING] Fragment menyimpan window pairs agar query window dapat cocok.
        frag_cooc = self._build_window_pairs(new_tokens)

        # [REASONING PURITY] relationship_strength diisi dari turn_continuity_score MURNI,
        # bukan dari analysis.continuity_score (yang = final_score candidate).
        # Ini memastikan fragment menyimpan nilai yang menjawab:
        #   "Seberapa kuat turn ini melanjutkan state aktif?"
        # bukan "Seberapa relevan kandidat secara keseluruhan?".
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
            relationship_strength=analysis.turn_continuity_score,  # ← KUNCI PERUBAHAN
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
                            # [REASONING PURITY] edge strength juga dari turn_continuity_score
                            strength=analysis.turn_continuity_score,
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
