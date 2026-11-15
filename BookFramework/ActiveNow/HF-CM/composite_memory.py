"""
composite_memory.py
Cross-Key Composite Fragment Detection (Mekanisme 4).

Tanggung jawab:
- Mendeteksi respons AI yang benar-benar menggunakan evidence dari
  beberapa TopicKey berbeda.
- Menghasilkan CompositeSourceAnalysis dengan daftar source fragments.
- Memvalidasi evidence family requirement (minimal 2 family berbeda).
- Memberikan cross-topic contribution bonus.

Prinsip kunci:
    Ownership tetap SATU primary TopicKey (single foreign key).
    Multi-source direpresentasikan lewat:
        - source_fragment_ids (daftar fragment sumber)
        - source_topic_key_ids (daftar TopicKey sumber)
        - N graph edges (REFERENCES) dari setiap source ke fragment composite.

    BUKAN dual primary TopicKey — itu akan merusak foreign-key ownership.

Kondisi composite (SEMUA harus terpenuhi):
    1. Contribution score >= MIN_SOURCE_CONTRIBUTION_SCORE
    2. Minimal MIN_SOURCE_EVIDENCE_FAMILIES family bukti (bukan token saja)
    3. Minimal MIN_DISTINCT_SOURCES source fragments
    4. Source fragments dari >= 2 TopicKey BERBEDA (cross-key)

Kenapa cross-key disyaratkan:
    Continuation dalam satu topik sudah direpresentasikan oleh CONTINUES.
    Composite khusus untuk hubungan LINTAS topik, agar semua fragment
    bertema "kopi" tidak otomatis saling menjadi source.

Anti-false-composite (Prioritas 26):
    Token overlap lemah + satu family bukti != composite.
    MIN_SOURCE_EVIDENCE_FAMILIES=2 memastikan source didukung oleh
    minimal 2 family (mis. token + helper, atau token + graph).

[REFACTOR BATCH 8]
Menambahkan DYNAMIC CROSS-MEMORY SOURCE DISCOVERY:
    SEBELUMNYA: hanya menganalisis retrieved_memories (kandidat yang
                kebetulan di-retrieve turn ini).
    SEKARANG:  juga mencari kandidat source dari GLOBAL memory lintas
                TopicKey. Memungkinkan deteksi composite meskipun source
                fragments tidak masuk retrieval top-K pada turn ini.

    Alur dua fase:
        Phase 1 - Discovery:
            retrieved_memories + discovered_global_fragments
            (deduplicate by fragment_id)
        Phase 2 - Validation:
            score setiap kandidat, apply composite thresholds

    [BATCH 8 FIX] Discovery hanya mencari TopicKey yang BELUM terwakili
    di retrieved_memories. Jika retrieval sudah menyentuh TopicKey X
    (mis. via src_weak), discovery TIDAK akan mencari fragment lain dari X.
    Ini karena:
        1. Retrieval sudah memberikan keputusan terbaik tentang X
        2. Menghindari "discovery" menemukan fragment strong yang retrieval
           sengaja tidak pilih (yang mengubah observable behavior test)
        3. Discovery = FILL GAPS, bukan re-fetch

    Backward compatibility:
        include_discovered=True (default) → perilaku baru
        include_discovered=False → perilaku lama (hanya retrieved_memories)
"""

import logging
import re
from typing import List, Optional, Set

from config import CONFIG
from models import (
    RetrievedMemory, CompositeSourceAnalysis, FragmentSourceContribution,
    EvidenceFamily,
)

logger = logging.getLogger(__name__)


class CompositeMemoryAnalyzer:
    """
    Analyzer untuk deteksi multi-source composite fragments.

    Dipanggil oleh memory_engine.py setelah AI response dihasilkan
    dan fragment-fragment terkait direkonstruksi.

    Dependency:
        - storage: untuk fetch FragmentMetadata dari fragment_id

    Kontrak:
        - detect_composite_sources() adalah PURE analysis function.
        - Tidak melakukan write ke storage.
        - Graph edges dibuat oleh caller (memory_engine._persist_memory).

    [BATCH 8] Menambahkan dynamic source discovery phase.
    """

    def __init__(self, storage):
        self.storage = storage

    # ==========================================
    # ENTRY POINT
    # ==========================================

    def detect_composite_sources(
        self,
        response: str,
        retrieved_memories: List[RetrievedMemory],
        primary_topic_key_id: Optional[str],
        include_discovered: bool = True,
    ) -> CompositeSourceAnalysis:
        """
        Analisis apakah response AI adalah composite dari beberapa sumber.

        Args:
            response: AI response text
            retrieved_memories: daftar memory yang direkonstruksi dari retrieval
            primary_topic_key_id: TopicKey utama (untuk cross-topic detection)
            include_discovered: [BATCH 8] jika True, augment dengan fragments
                                yang di-discover dari global memory lintas
                                TopicKey (yang BELUM terwakili di retrieval).
                                Default True.

        Returns:
            CompositeSourceAnalysis dengan:
                - is_composite: True jika semua kondisi terpenuhi
                - sources: daftar FragmentSourceContribution
                - source_fragment_ids: daftar fragment ID sumber
                - distinct_topic_count: jumlah TopicKey unik
                - reason: penjelasan keputusan
        """
        cs = CONFIG.composite
        analysis = CompositeSourceAnalysis(
            primary_topic_key_id=primary_topic_key_id
        )

        # Guard: response kosong
        if not response:
            analysis.reason = "empty response"
            return analysis

        # Tokenize response
        resp_tokens = self._tokenize(response)
        if not resp_tokens:
            analysis.reason = "no meaningful response tokens"
            return analysis

        # Build initial candidate list
        candidates: List[RetrievedMemory] = list(retrieved_memories or [])

        # [BATCH 8] Augment dengan discovered fragments dari global memory,
        # tetapi HANYA dari TopicKey yang BELUM terwakili di retrieved_memories.
        if include_discovered:
            # Hitung TopicKey mana yang sudah terwakili di retrieval
            already_represented: Set[str] = {
                mem.topic_key_id for mem in candidates
                if mem.topic_key_id is not None
            }
            discovered = self.discover_cross_topic_candidates(
                primary_topic_key_id=primary_topic_key_id,
                resp_tokens=resp_tokens,
                limit=cs.MAX_SOURCE_FRAGMENTS * 2,  # overshoot untuk merge
                already_represented_topic_keys=already_represented,
            )
            candidates = self._merge_candidate_lists(candidates, discovered)

        # Guard: tidak ada kandidat
        if not candidates:
            analysis.reason = "no candidate sources (retrieved or discovered)"
            return analysis

        # Analyze setiap kandidat sebagai potential source
        sources: List[FragmentSourceContribution] = []
        for mem in candidates:
            contribution = self._analyze_single_source(
                mem=mem,
                resp_tokens=resp_tokens,
                primary_topic_key_id=primary_topic_key_id,
            )
            if contribution is not None:
                sources.append(contribution)

        # Sort by contribution score descending
        sources.sort(key=lambda s: s.contribution_score, reverse=True)

        # Limit to MAX_SOURCE_FRAGMENTS
        sources = sources[:cs.MAX_SOURCE_FRAGMENTS]

        # Count distinct topics
        distinct_topics = {s.topic_key_id for s in sources}

        # Populate analysis
        analysis.sources = sources
        analysis.source_fragment_ids = [s.fragment_id for s in sources]
        analysis.distinct_topic_count = len(distinct_topics)

        # Final check: composite conditions
        if (len(sources) >= cs.MIN_DISTINCT_SOURCES
                and any(s.is_cross_topic for s in sources)
                and len(distinct_topics) >= 2):
            analysis.is_composite = True
            analysis.reason = (
                f"{len(sources)} sources across {len(distinct_topics)} topics"
            )
        else:
            analysis.is_composite = False
            analysis.source_fragment_ids = []
            analysis.reason = "not cross-key composite"

        return analysis

    # ==========================================
    # [BATCH 8] DYNAMIC SOURCE DISCOVERY
    # ==========================================

    def discover_cross_topic_candidates(
        self,
        primary_topic_key_id: Optional[str],
        resp_tokens: Set[str],
        limit: int = 20,
        already_represented_topic_keys: Optional[Set[str]] = None,
    ) -> List[RetrievedMemory]:
        """
        [BATCH 8] Fetch kandidat source dari GLOBAL memory lintas TopicKey.

        Alur:
            1. Fetch semua TopicKey aktif.
            2. Filter: bukan primary topic DAN belum terwakili di retrieval.
            3. Untuk setiap TopicKey:
                a. Quick pre-filter via TopicKey token_profile overlap.
                b. Fetch fragment-fragment.
                c. Filter: fragment yang punya token overlap dengan response.
                d. Konversi ke RetrievedMemory (synthetic).
            4. Gabungkan dan limit ke N kandidat.

        Args:
            primary_topic_key_id: TopicKey utama (akan di-skip)
            resp_tokens: set of response tokens (untuk overlap filter)
            limit: maximum number of candidates to return
            already_represented_topic_keys: [BATCH 8 FIX] TopicKey yang
                sudah terwakili di retrieved_memories. Discovery akan skip
                topik-topik ini (retrieval sudah memberikan keputusan
                terbaik tentang mereka).

        Returns: List[RetrievedMemory] — kandidat yang akan dianalisis bersama
                 retrieved_memories di detect_composite_sources.

        Catatan:
            - fragment_graph_score di-set 0.0 untuk synthetic memories (belum
              di-query via graph). Validasi graph evidence akan dilakukan di
              memory_engine saat composite edge dibuat.
            - Ini adalah heuristic "cheap filter" — tidak perlu sempurna,
              hanya untuk menyediakan kandidat awal yang akan divalidasi
              lebih lanjut di _analyze_single_source.
        """
        if primary_topic_key_id is None:
            # Tanpa primary topic, tidak ada "cross-topic"
            return []

        if not resp_tokens:
            return []

        represented = already_represented_topic_keys or set()

        # Fetch semua TopicKey aktif
        all_keys = self.storage.get_all_topic_keys()
        if not all_keys:
            return []

        candidates: List[RetrievedMemory] = []

        # Iterasi setiap TopicKey
        for key in all_keys:
            # Skip primary topic (continuation sudah ditangani CONTINUES)
            if key.key_id == primary_topic_key_id:
                continue

            # [BATCH 8 FIX] Skip topik yang sudah terwakili di retrieval
            if key.key_id in represented:
                continue

            # Fetch fragments milik TopicKey ini
            frags = self.storage.get_fragment_metadata_by_key(key.key_id)
            if not frags:
                continue

            # Quick pre-filter: cek apakah TopicKey ini punya overlap
            # dengan response tokens di tingkat TopicKey token profile.
            # Ini menghindari fetch fragments yang jelas tidak relevan.
            key_tokens = set(key.token_profile.keys())
            key_overlap = resp_tokens.intersection(key_tokens)
            if not key_overlap:
                # Tidak ada overlap di tingkat TopicKey — skip
                continue

            # Iterasi fragments, cari yang punya overlap
            for frag in frags:
                frag_tokens = set(frag.token_profile.keys()) if frag.token_profile else set()
                frag_overlap = resp_tokens.intersection(frag_tokens)
                if not frag_overlap:
                    # Tidak ada token overlap → skip
                    continue

                # Cek helper overlap juga (untuk evidence family CONTEXTUAL)
                helper_text = (
                    frag.aggregated_helper_description
                    or frag.fragment_helper_description
                    or ""
                )
                helper_tokens = self._tokenize(helper_text)
                helper_overlap = resp_tokens.intersection(helper_tokens) if helper_tokens else set()

                # Hanya kandidat dengan minimal evidence (token ATAU helper overlap)
                if not frag_overlap and not helper_overlap:
                    continue

                # Convert ke synthetic RetrievedMemory
                synthetic = self._build_synthetic_memory(frag)
                candidates.append(synthetic)

                # Early stop: cukup kandidat untuk dipertimbangkan
                if len(candidates) >= limit * 2:
                    break

            if len(candidates) >= limit * 2:
                break

        # Limit output
        return candidates[:limit]

    def _build_synthetic_memory(self, fragment) -> RetrievedMemory:
        """
        Convert FragmentMetadata ke synthetic RetrievedMemory.

        Dipakai oleh discover_cross_topic_candidates untuk membuat
        kandidat yang kompatibel dengan _analyze_single_source.

        Field yang diisi:
            - fragment_id, text="", timestamp, topic_key_id
            - fragment_key, fragment_helper_description
            - parent_fragment_id, depth
            - is_composite, source_fragment_ids

        Field yang di-set 0.0/None (belum di-query):
            - relevance_score, temporal_score, graph_score
            - fragment_graph_score, temporal_distance_score
            - evidence_strength, exact_temporal_match
        """
        # Synthetic text dari helper/key (untuk observability)
        synthetic_text = (
            fragment.fragment_helper_description
            or fragment.fragment_key
            or ""
        )

        return RetrievedMemory(
            fragment_id=fragment.fragment_id,
            text=synthetic_text,
            timestamp=(
                fragment.created_at.isoformat()
                if fragment.created_at else ""
            ),
            topic_key_id=fragment.topic_key_id,
            relevance_score=0.0,
            temporal_score=0.0,
            graph_score=0.0,
            evidence_strength=0.0,
            exact_temporal_match=False,
            fragment_graph_score=0.0,
            graph_edge_count=0,
            graph_path_types=[],
            temporal_distance_score=0.0,
            temporal_distance_minutes=None,
            topic_key_title="",
            fragment_key=fragment.fragment_key or "",
            fragment_helper_description=fragment.fragment_helper_description or "",
            parent_fragment_id=fragment.parent_fragment_id,
            depth=fragment.depth or 0,
            source_fragment_ids=list(fragment.source_fragment_ids or []),
            is_composite=bool(fragment.is_composite),
        )

    def _merge_candidate_lists(
        self,
        primary: List[RetrievedMemory],
        secondary: List[RetrievedMemory],
    ) -> List[RetrievedMemory]:
        """
        Gabungkan dua list kandidat, deduplicate by fragment_id.

        Prioritas: primary dipertahankan (mereka dari retrieval sebenarnya).
        Secondary ditambahkan jika fragment_id belum ada.

        Returns: list merged tanpa duplikat fragment_id.
        """
        seen_ids: Set[str] = set()
        merged: List[RetrievedMemory] = []

        # Tambahkan primary dulu (prioritas)
        for mem in primary:
            if mem.fragment_id not in seen_ids:
                seen_ids.add(mem.fragment_id)
                merged.append(mem)

        # Tambahkan secondary yang belum ada
        for mem in secondary:
            if mem.fragment_id not in seen_ids:
                seen_ids.add(mem.fragment_id)
                merged.append(mem)

        return merged

    # ==========================================
    # SINGLE SOURCE ANALYSIS
    # ==========================================

    def _analyze_single_source(
        self,
        mem: RetrievedMemory,
        resp_tokens: Set[str],
        primary_topic_key_id: Optional[str],
    ) -> Optional[FragmentSourceContribution]:
        """
        Analisis satu RetrievedMemory sebagai potential source.

        Returns FragmentSourceContribution jika lolos threshold, else None.

        Steps:
            1. Fetch fragment dari storage.
            2. Hitung token overlap.
            3. Hitung helper overlap.
            4. Check graph score.
            5. Validate evidence families (minimal 2).
            6. Hitung contribution score (dengan cross-topic bonus).
            7. Check contribution >= MIN_SOURCE_CONTRIBUTION_SCORE.
        """
        cs = CONFIG.composite

        # Fetch fragment detail
        frag = self.storage.get_fragment(mem.fragment_id)
        if frag is None:
            return None

        # Token overlap
        frag_tokens = set(frag.token_profile.keys())
        if not frag_tokens:
            return None
        overlap = resp_tokens.intersection(frag_tokens)
        token_ratio = len(overlap) / len(frag_tokens)
        if token_ratio < cs.MIN_SOURCE_TOKEN_OVERLAP:
            return None

        # Evidence families tracking
        evidence_names: List[str] = ["token"]
        families: Set[EvidenceFamily] = {EvidenceFamily.LEXICAL}

        # Helper overlap
        helper_text = (
            frag.aggregated_helper_description
            or frag.fragment_helper_description
            or ""
        )
        helper_tokens = self._tokenize(helper_text)
        helper_ratio = 0.0
        if helper_tokens:
            helper_ratio = len(resp_tokens.intersection(helper_tokens)) / len(helper_tokens)
            if helper_ratio >= cs.MIN_SOURCE_TOKEN_OVERLAP:
                evidence_names.append("helper")
                families.add(EvidenceFamily.CONTEXTUAL)

        # Graph score
        if mem.fragment_graph_score >= CONFIG.evidence.EVIDENCE_SUPPORT_THRESHOLD:
            evidence_names.append("graph")
            families.add(EvidenceFamily.RELATIONAL)

        # Validate evidence families (anti-false-composite)
        if len(families) < cs.MIN_SOURCE_EVIDENCE_FAMILIES:
            return None

        # Contribution score
        contribution = max(token_ratio, helper_ratio)
        is_cross = (
            primary_topic_key_id is not None
            and frag.topic_key_id != primary_topic_key_id
        )
        if is_cross:
            contribution = min(1.0, contribution + cs.CROSS_TOPIC_CONTRIBUTION_BONUS)

        # Final threshold
        if contribution < cs.MIN_SOURCE_CONTRIBUTION_SCORE:
            return None

        return FragmentSourceContribution(
            fragment_id=frag.fragment_id,
            topic_key_id=frag.topic_key_id,
            fragment_key=frag.fragment_key,
            contribution_score=contribution,
            supporting_evidence_names=evidence_names,
            is_cross_topic=is_cross,
        )

    # ==========================================
    # TOKENIZATION HELPER
    # ==========================================

    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text menjadi set of meaningful tokens.

        Filter:
            - Panjang >= MIN_TOKEN_LENGTH
            - Lowercase
            - Unique (set)
        """
        if not text:
            return set()
        min_len = CONFIG.signal.MIN_TOKEN_LENGTH
        raw = re.findall(r'\b\w+\b', text.lower())
        return {t for t in raw if len(t) >= min_len}
